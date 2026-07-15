"""
DMRL: Polarity-Coupled Optimal Transport with disagreement-aware fusion
for multimodal sentiment analysis.

The model is organized around a single core mechanism, Polarity-Coupled
Optimal Transport (PCOT):
1. Modality-specific stems + shared Transformer encoder with lightweight adapters
2. Sparse polar evidence tokens carrying an evidence vector, a polarity and a reliability
3. PCOT couples semantic and sentiment-polarity distances into one transport cost
4. The single transport plan is split into a consensus flow and a conflict flow
5. A closed-form cross-modal disagreement score rho gates the fusion
6. An ordinal distribution head produces the sentiment estimate
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer


TRANSFORMERS_MAP = {
    "bert": (BertTokenizer, BertModel),
    "roberta": (RobertaTokenizer, RobertaModel),
}

ORDINAL_ANCHORS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def build_sequence_mask(x, mask=None):
    """Build boolean mask [B, T], True means valid."""
    if mask is not None:
        seq_mask = mask.bool()
    elif x.dim() == 3:
        seq_mask = (x.abs().sum(dim=-1) > 0).bool()
    else:
        seq_mask = (x > 0).bool()

    if seq_mask.dim() == 2:
        empty_rows = ~seq_mask.any(dim=1)
        if empty_rows.any():
            seq_mask = seq_mask.clone()
            seq_mask[empty_rows, 0] = True
    return seq_mask


def masked_mean(x, mask):
    """Masked mean pooling for [B, T, D]."""
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)


def evidence_weighted_pool(ev):
    """Pool evidence slots by their reliability weights."""
    weights = ev["slot_weight"].unsqueeze(-1)
    return (weights * ev["evidence"]).sum(dim=1)


class BertTextEncoder(nn.Module):
    """BERT/RoBERTa text encoder returning token-level features [B, Lt, H]."""

    def __init__(self, use_finetune=False, transformers="bert", pretrained="bert-base-uncased"):
        super().__init__()
        tokenizer_class, model_class = TRANSFORMERS_MAP[transformers]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune
        self.transformers = transformers
        self.hidden_size = self.model.config.hidden_size

    def forward(self, text):
        input_ids = text[:, 0, :].long()
        input_mask = text[:, 1, :].long()
        segment_ids = text[:, 2, :].long()

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        if self.transformers == "bert":
            kwargs["token_type_ids"] = segment_ids

        if self.use_finetune:
            return self.model(**kwargs)[0]

        with torch.no_grad():
            return self.model(**kwargs)[0]


class MultiScaleTemporalStem(nn.Module):
    """Parallel multi-scale Conv1d stem with residual projection."""

    def __init__(self, d_in, d_model, kernels=(1, 3, 5), dropout=0.1):
        super().__init__()
        branch_dim = d_model
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_in, branch_dim, kernel_size=k, padding=k // 2, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernels
        ])
        self.out_proj = nn.Linear(branch_dim * len(kernels), d_model)
        self.res_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.res_proj(x)
        x_t = x.transpose(1, 2)
        feats = [branch(x_t).transpose(1, 2) for branch in self.branches]
        h = self.out_proj(torch.cat(feats, dim=-1))
        return self.norm(self.drop(h) + residual)


class ModalityAdapter(nn.Module):
    """Small modality-specific residual adapter after the shared encoder."""

    def __init__(self, d_model, bottleneck=None, dropout=0.1):
        super().__init__()
        bottleneck = bottleneck or max(d_model // 4, 16)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        return x + self.scale * self.net(x)


class MaskedTransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with mask support."""

    def __init__(self, d_model, n_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, mask=None):
        key_padding_mask = ~mask if mask is not None else None
        x = self.block(x, src_key_padding_mask=key_padding_mask)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class MaskedTransformerEncoder(nn.Module):
    """Stacked mask-aware Transformer encoder."""

    def __init__(self, d_model, n_layers=3, dropout=0.1, n_heads=4, ff_mult=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            MaskedTransformerBlock(d_model, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class SparsePolarEvidenceExtractor(nn.Module):
    """Slot attention extractor producing sparse polar evidence tokens.

    Each slot yields:
    - evidence vector z
    - polarity p in [-1, 1]   (drives polarity-coupled transport)
    - reliability r >= 0      (drives evidence pooling)
    """

    def __init__(self, d_model, k_slots=4, dropout=0.1):
        super().__init__()
        self.k_slots = k_slots

        self.slot_queries = nn.Parameter(torch.randn(1, k_slots, d_model) * 0.02)
        self.query_ctx_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.polarity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )
        self.drop = nn.Dropout(dropout)

    def _masked_softmax(self, scores, mask):
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attn = F.softmax(scores, dim=-1)
        attn = attn * mask.unsqueeze(1).to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return attn

    def forward(self, x, mask=None):
        B, T, D = x.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        global_ctx = masked_mean(x, mask)
        base_q = self.slot_queries.expand(B, -1, -1)
        ctx_q = self.query_ctx_proj(global_ctx).unsqueeze(1).expand(-1, self.k_slots, -1)
        gate = self.query_gate(torch.cat([base_q, ctx_q], dim=-1))
        q = self.query_norm(base_q + gate * ctx_q)

        k = self.key_norm(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)
        attn = self._masked_softmax(scores, mask)

        evidence = self.drop(torch.matmul(attn, self.value_proj(x)))
        polarity = self.polarity_head(evidence).squeeze(-1)
        reliability = self.reliability_head(evidence).squeeze(-1)
        slot_weight = F.softmax(reliability, dim=-1)

        return {
            "evidence": evidence,
            "polarity": polarity,
            "reliability": reliability,
            "slot_weight": slot_weight,
            "attn": attn,
        }


class PolarityCoupledOT(nn.Module):
    """Polarity-Coupled Optimal Transport (PCOT).

    The core mechanism of DMRL. It couples a semantic distance and a
    sentiment-polarity distance into a single entropic-OT transport cost, and
    then splits the resulting transport plan into:
    - a consensus flow T^+ over same-polarity aligned evidence, and
    - a conflict flow  T^- over opposite-polarity aligned evidence.
    The relative mass of the conflict flow yields a closed-form, sample-level
    cross-modal disagreement score rho in [0, 1].
    """

    def __init__(self, lambda_p=0.5, temperature=0.2, sinkhorn_iters=5, gate_temp=4.0):
        super().__init__()
        self.lambda_p = lambda_p
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.gate_temp = gate_temp

    def sinkhorn(self, log_alpha):
        for _ in range(self.sinkhorn_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        transport = log_alpha.exp()
        transport = transport / transport.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return transport

    def forward(self, ev_a, ev_b):
        za = F.normalize(ev_a["evidence"], dim=-1, eps=1e-6)
        zb = F.normalize(ev_b["evidence"], dim=-1, eps=1e-6)
        sim = torch.matmul(za, zb.transpose(1, 2))                    # [B, Ka, Kb]

        pa = ev_a["polarity"].unsqueeze(-1)                           # [B, Ka, 1]
        pb = ev_b["polarity"].unsqueeze(-2)                           # [B, 1, Kb]
        polarity_dist = (pa - pb).abs()

        # Polarity-coupled transport cost: semantic distance + polarity distance.
        cost = (1.0 - sim) + self.lambda_p * polarity_dist
        transport = self.sinkhorn(-cost / max(self.temperature, 1e-6))

        # Polarity-consistency gate: same sign -> consensus, opposite -> conflict.
        gate = torch.sigmoid(self.gate_temp * pa * pb)                # [B, Ka, Kb]
        t_plus = transport * gate
        t_minus = transport * (1.0 - gate)

        mass_plus = t_plus.sum(dim=(-1, -2), keepdim=True)            # [B, 1, 1]
        mass_minus = t_minus.sum(dim=(-1, -2), keepdim=True)

        fa = ev_a["evidence"]                                         # [B, Ka, d]
        fb = ev_b["evidence"]                                         # [B, Kb, d]
        pair_avg = 0.5 * (fa.unsqueeze(2) + fb.unsqueeze(1))          # [B, Ka, Kb, d]
        pair_diff = fa.unsqueeze(2) - fb.unsqueeze(1)                 # [B, Ka, Kb, d]

        # Consensus: transported average of aligned same-polarity evidence.
        consensus = (t_plus.unsqueeze(-1) * pair_avg).sum(dim=(1, 2))
        consensus = consensus / mass_plus.squeeze(-1).clamp_min(1e-6)
        # Conflict: transported signed difference of opposite-polarity evidence.
        conflict = (t_minus.unsqueeze(-1) * pair_diff).sum(dim=(1, 2))
        conflict = conflict / mass_minus.squeeze(-1).clamp_min(1e-6)

        # Closed-form disagreement: relative mass carried by the conflict flow.
        rho = (mass_minus / (mass_plus + mass_minus).clamp_min(1e-6)).view(-1, 1)

        return {
            "consensus": consensus,
            "conflict": conflict,
            "rho": rho,
            "transport": transport,
            "cost": cost,
        }


class RoleExpert(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ConsensusConflictExperts(nn.Module):
    """Two lightweight experts that refine the consensus and conflict streams."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.expert_consensus = RoleExpert(d_model, dropout=dropout)
        self.expert_conflict = RoleExpert(d_model, dropout=dropout)

    def forward(self, consensus, conflict):
        return self.expert_consensus(consensus), self.expert_conflict(conflict)


class OrdinalDistributionHead(nn.Module):
    """Ordinal distribution head over sentiment anchors.

    The final sentiment is the expectation over anchors. Predictive uncertainty
    is the variance of the ordinal distribution.
    """

    def __init__(self, d_model, anchors=ORDINAL_ANCHORS, dropout=0.1):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors).float())
        self.ordinal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(anchors)),
        )

    def forward(self, x):
        logits = self.ordinal_head(x)
        probs = F.softmax(logits, dim=-1)
        anchors = self.anchors.unsqueeze(0)
        sentiment = (probs * anchors).sum(dim=-1, keepdim=True)
        variance = (probs * (anchors - sentiment) ** 2).sum(dim=-1, keepdim=True)
        return {
            "ordinal_logits": logits,
            "ordinal_probs": probs,
            "output_logit": sentiment,
            "predictive_variance": variance,
        }


class DMRLLoss(nn.Module):
    """Compact loss with four terms: main + ordinal + polarity grounding + transport."""

    def __init__(
        self,
        anchors=ORDINAL_ANCHORS,
        w_main=1.0,
        w_pol=0.1,
        w_transport=0.02,
        w_ord=0.2,
        polarity_scale=4.0,
    ):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors).float())
        self.l1 = nn.L1Loss()
        self.w_main = w_main
        self.w_pol = w_pol
        self.w_transport = w_transport
        self.w_ord = w_ord
        self.polarity_scale = polarity_scale

    def _labels(self, labels):
        return labels.view(-1, 1).float()

    def soft_ordinal_target(self, labels):
        labels = self._labels(labels)
        anchors = self.anchors.view(1, -1).to(labels.device)
        dist = torch.exp(-((labels - anchors) ** 2) / 2.0)
        return dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def ordinal_loss(self, logits, labels):
        target = self.soft_ordinal_target(labels)
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def polarity_grounding_loss(self, out, labels):
        """Ground the aggregate evidence polarity to the sentiment sign.

        Only samples with a clear sentiment sign (|y| > 1e-3) are supervised, so
        near-neutral samples do not force a spurious polarity.
        """
        labels = self._labels(labels).to(out["output_logit"].device)
        mask = (labels.abs() > 1e-3).float()
        if mask.sum() < 1.0:
            return out["output_logit"].new_tensor(0.0)
        prob = torch.sigmoid(self.polarity_scale * out["aggregate_polarity"])
        target = (labels > 0).float()
        bce = F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        return (bce * mask).sum() / mask.sum().clamp_min(1.0)

    def transport_loss(self, out):
        terms = []
        for pair in ("ta", "tv"):
            t_key = f"transport_{pair}"
            c_key = f"cost_{pair}"
            if t_key in out and c_key in out:
                terms.append((out[t_key] * out[c_key]).sum(dim=(-1, -2)).mean())
        if not terms:
            return out["output_logit"].new_tensor(0.0)
        return sum(terms) / len(terms)

    def forward(self, out, labels):
        labels = self._labels(labels).to(out["output_logit"].device)
        losses = {}
        losses["main_loss"] = self.l1(out["output_logit"], labels)
        losses["ordinal_loss"] = self.ordinal_loss(out["ordinal_logits"], labels)
        losses["polarity_loss"] = self.polarity_grounding_loss(out, labels)
        losses["transport_loss"] = self.transport_loss(out)

        losses["total_loss"] = (
            self.w_main * losses["main_loss"]
            + self.w_ord * losses["ordinal_loss"]
            + self.w_pol * losses["polarity_loss"]
            + self.w_transport * losses["transport_loss"]
        )
        return losses


class DMRL(nn.Module):
    """DMRL with Polarity-Coupled Optimal Transport and disagreement-aware fusion."""

    def __init__(self, args):
        super().__init__()
        d_t, d_a, d_v = args.feature_dims
        d_model = args.d_model
        n_layers = args.n_layers
        dropout = args.dropout
        k_slots = getattr(args, "k_slots", 4)
        n_heads = getattr(args, "n_heads", 4)
        ff_mult = getattr(args, "ff_mult", 4)

        self.use_bert = getattr(args, "use_bert", True)
        if self.use_bert:
            self.bert = BertTextEncoder(
                use_finetune=getattr(args, "use_finetune", True),
                transformers=getattr(args, "transformers", "bert"),
                pretrained=getattr(args, "pretrained", "bert-base-uncased"),
            )
            d_t = self.bert.hidden_size

        # Modality-specific stems map all modalities into the same d_model space.
        self.stem_t = MultiScaleTemporalStem(d_t, d_model, dropout=dropout)
        self.stem_a = MultiScaleTemporalStem(d_a, d_model, dropout=dropout)
        self.stem_v = MultiScaleTemporalStem(d_v, d_model, dropout=dropout)

        # Shared encoder reduces parameters and encourages a common evidence space.
        self.mod_embed = nn.Parameter(torch.randn(3, 1, d_model) * 0.02)
        self.shared_encoder = MaskedTransformerEncoder(
            d_model,
            n_layers=n_layers,
            dropout=dropout,
            n_heads=n_heads,
            ff_mult=ff_mult,
        )
        self.adapter_t = ModalityAdapter(d_model, dropout=dropout)
        self.adapter_a = ModalityAdapter(d_model, dropout=dropout)
        self.adapter_v = ModalityAdapter(d_model, dropout=dropout)

        self.evidence_t = SparsePolarEvidenceExtractor(d_model, k_slots=k_slots, dropout=dropout)
        self.evidence_a = SparsePolarEvidenceExtractor(d_model, k_slots=k_slots, dropout=dropout)
        self.evidence_v = SparsePolarEvidenceExtractor(d_model, k_slots=k_slots, dropout=dropout)

        # Core mechanism: one polarity-coupled transport shared by T-A and T-V pairs.
        self.pcot = PolarityCoupledOT(
            lambda_p=getattr(args, "lambda_p", 0.5),
            temperature=getattr(args, "transport_temperature", 0.2),
            sinkhorn_iters=getattr(args, "sinkhorn_iters", 5),
            gate_temp=getattr(args, "gate_temp", 4.0),
        )
        self.consensus_proj = nn.Linear(d_model, d_model)
        self.conflict_proj = nn.Linear(d_model, d_model)
        self.experts = ConsensusConflictExperts(d_model, dropout=dropout)

        self.mod_proj_t = nn.Linear(d_model, d_model)
        self.mod_proj_a = nn.Linear(d_model, d_model)
        self.mod_proj_v = nn.Linear(d_model, d_model)
        self.fuse_unimodal = nn.Linear(d_model * 3, d_model)
        self.fuse_consensus = nn.Linear(d_model, d_model)
        self.fuse_conflict = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        self.ordinal_head = OrdinalDistributionHead(d_model, dropout=dropout)

        self.loss_module = DMRLLoss(
            w_main=getattr(args, "w_main", 1.0),
            w_pol=getattr(args, "w_pol", 0.1),
            w_transport=getattr(args, "w_transport", 0.02),
            w_ord=getattr(args, "w_ord", 0.2),
            polarity_scale=getattr(args, "gate_temp", 4.0),
        )

    def _text_mask_from_input(self, text, text_mask=None):
        if text_mask is not None:
            return build_sequence_mask(text[:, 1, :], text_mask)
        return build_sequence_mask(text[:, 1, :])

    def _encode_modalities(self, text, audio, video, text_mask, audio_mask, video_mask):
        t = self.bert(text) if self.use_bert else text

        ht = self.stem_t(t) + self.mod_embed[0]
        ha = self.stem_a(audio) + self.mod_embed[1]
        hv = self.stem_v(video) + self.mod_embed[2]

        ht = self.shared_encoder(ht, text_mask)
        ha = self.shared_encoder(ha, audio_mask)
        hv = self.shared_encoder(hv, video_mask)

        ht = self.adapter_t(ht)
        ha = self.adapter_a(ha)
        hv = self.adapter_v(hv)

        if text_mask is not None:
            ht = ht * text_mask.unsqueeze(-1).to(ht.dtype)
        if audio_mask is not None:
            ha = ha * audio_mask.unsqueeze(-1).to(ha.dtype)
        if video_mask is not None:
            hv = hv * video_mask.unsqueeze(-1).to(hv.dtype)

        return ht, ha, hv

    def _aggregate_polarity(self, *evs):
        """Reliability-weighted mean polarity aggregated over all modalities.

        Returns a scalar polarity per sample that summarizes the whole clip and
        is grounded to the sentiment sign by the polarity grounding loss.
        """
        per_mod = []
        for ev in evs:
            pol = (ev["slot_weight"] * ev["polarity"]).sum(dim=-1, keepdim=True)
            per_mod.append(pol)
        return torch.stack(per_mod, dim=0).mean(dim=0)

    def forward(self, text, audio, video, text_mask=None, audio_mask=None, video_mask=None, labels=None):
        text_mask = self._text_mask_from_input(text, text_mask)
        audio_mask = build_sequence_mask(audio, audio_mask)
        video_mask = build_sequence_mask(video, video_mask)

        ht, ha, hv = self._encode_modalities(text, audio, video, text_mask, audio_mask, video_mask)

        text_ev = self.evidence_t(ht, text_mask)
        audio_ev = self.evidence_a(ha, audio_mask)
        video_ev = self.evidence_v(hv, video_mask)

        # Polarity-coupled optimal transport for the two text-anchored pairs.
        pcot_ta = self.pcot(text_ev, audio_ev)
        pcot_tv = self.pcot(text_ev, video_ev)

        consensus = self.consensus_proj(pcot_ta["consensus"] + pcot_tv["consensus"])
        conflict = self.conflict_proj(pcot_ta["conflict"] + pcot_tv["conflict"])
        consensus, conflict = self.experts(consensus, conflict)

        # Closed-form disagreement gates consensus vs. conflict streams.
        rho = 0.5 * (pcot_ta["rho"] + pcot_tv["rho"])

        pooled_t = self.mod_proj_t(evidence_weighted_pool(text_ev))
        pooled_a = self.mod_proj_a(evidence_weighted_pool(audio_ev))
        pooled_v = self.mod_proj_v(evidence_weighted_pool(video_ev))
        unimodal = self.fuse_unimodal(torch.cat([pooled_t, pooled_a, pooled_v], dim=-1))

        fused = self.final_norm(
            (1.0 - rho) * self.fuse_consensus(consensus)
            + rho * self.fuse_conflict(conflict)
            + unimodal
        )
        main_out = self.ordinal_head(fused)

        aggregate_polarity = self._aggregate_polarity(text_ev, audio_ev, video_ev)

        output = {
            "output_logit": main_out["output_logit"],
            "ordinal_logits": main_out["ordinal_logits"],
            "ordinal_probs": main_out["ordinal_probs"],
            "predictive_variance": main_out["predictive_variance"],
            # Backward-compatible uncertainty interval proxy.
            "interval_width": main_out["predictive_variance"].clamp_min(1e-8).sqrt(),
            "fused": fused,
            "disagreement": rho,
            "consensus_repr": consensus,
            "conflict_repr": conflict,
            "aggregate_polarity": aggregate_polarity,
            "text_evidence": text_ev["evidence"],
            "audio_evidence": audio_ev["evidence"],
            "video_evidence": video_ev["evidence"],
            "text_polarity": text_ev["polarity"],
            "audio_polarity": audio_ev["polarity"],
            "video_polarity": video_ev["polarity"],
            "text_reliability": text_ev["reliability"],
            "audio_reliability": audio_ev["reliability"],
            "video_reliability": video_ev["reliability"],
            "text_slot_weight": text_ev["slot_weight"],
            "audio_slot_weight": audio_ev["slot_weight"],
            "video_slot_weight": video_ev["slot_weight"],
            "text_attn": text_ev["attn"],
            "audio_attn": audio_ev["attn"],
            "video_attn": video_ev["attn"],
            "transport_ta": pcot_ta["transport"],
            "transport_tv": pcot_tv["transport"],
            "cost_ta": pcot_ta["cost"],
            "cost_tv": pcot_tv["cost"],
            "rho_ta": pcot_ta["rho"],
            "rho_tv": pcot_tv["rho"],
        }

        if labels is not None:
            output["losses"] = self.loss_module(output, labels)

        return output