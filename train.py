"""Training entry point for the DMRL multimodal sentiment analysis model.

Example:
    python train.py --dataset mosi --data_path data/mosi/unaligned.pkl --seed 1111

The script trains the model defined in ``dmrl.py`` on the standard MMSA-style
processed data, validates each epoch, applies MAE-based early stopping and
reports the test metrics of the best checkpoint.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from config import build_args
from data_loader import build_dataloaders
from dmrl import DMRL
from metrics import compute_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_cli():
    parser = argparse.ArgumentParser(description="Train DMRL for multimodal sentiment analysis.")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "mosei", "sims"])
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed .pkl file.")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)

    # Optional overrides (None means use dataset/model defaults from config.py).
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--bert_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--early_stop", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--k_slots", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_finetune", action="store_true", help="Freeze the BERT encoder.")
    return parser.parse_args()


def build_optimizer(model, args):
    """Use a smaller learning rate for BERT than for the rest of the model."""
    bert_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            bert_params.append(param)
        else:
            other_params.append(param)

    groups = [{"params": other_params, "lr": args.learning_rate}]
    if bert_params:
        groups.append({"params": bert_params, "lr": args.bert_lr})
    return AdamW(groups, weight_decay=args.weight_decay)


def move_batch(batch, device):
    return (
        batch["text"].to(device),
        batch["audio"].to(device),
        batch["vision"].to(device),
        batch["label"].to(device),
    )


def run_epoch(model, loader, optimizer, device, grad_clip, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    preds, labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "eval"):
            text, audio, vision, label = move_batch(batch, device)

            output = model(text, audio, vision, labels=label)
            loss = output["losses"]["total_loss"]

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bsz = label.size(0)
            total_loss += loss.item() * bsz
            n += bsz
            preds.append(output["output_logit"].detach().cpu().view(-1))
            labels.append(label.detach().cpu().view(-1))

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return total_loss / max(n, 1), preds, labels


def format_metrics(metrics):
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main():
    cli = parse_cli()
    set_seed(cli.seed)

    overrides = {
        "batch_size": cli.batch_size,
        "learning_rate": cli.learning_rate,
        "bert_lr": cli.bert_lr,
        "weight_decay": cli.weight_decay,
        "num_epochs": cli.num_epochs,
        "early_stop": cli.early_stop,
        "d_model": cli.d_model,
        "n_layers": cli.n_layers,
        "k_slots": cli.k_slots,
        "dropout": cli.dropout,
    }
    if cli.no_finetune:
        overrides["use_finetune"] = False

    args = build_args(cli.dataset, overrides)
    device = torch.device(cli.device)
    print(f"[config] dataset={args.dataset} device={device} seed={cli.seed}")

    train_loader, valid_loader, test_loader = build_dataloaders(
        cli.data_path,
        batch_size=args.batch_size,
        use_bert=args.use_bert,
        num_workers=cli.num_workers,
    )

    model = DMRL(args).to(device)
    optimizer = build_optimizer(model, args)

    os.makedirs(cli.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cli.save_dir, f"dmrl_{args.dataset}_seed{cli.seed}.pth")

    best_mae = float("inf")
    best_metrics = None
    patience = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, device, cli.grad_clip, train=True
        )
        val_loss, val_preds, val_labels = run_epoch(
            model, valid_loader, optimizer, device, cli.grad_clip, train=False
        )
        val_metrics = compute_metrics(val_preds, val_labels, args.dataset)

        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} | {format_metrics(val_metrics)}"
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            patience = 0
            torch.save(model.state_dict(), ckpt_path)

            _, test_preds, test_labels = run_epoch(
                model, test_loader, optimizer, device, cli.grad_clip, train=False
            )
            best_metrics = compute_metrics(test_preds, test_labels, args.dataset)
            print(f"  -> new best val MAE={best_mae:.4f}; test: {format_metrics(best_metrics)}")
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"[early stop] no val improvement for {args.early_stop} epochs.")
                break

    print("\n========== Final Test (best val MAE) ==========")
    print(f"checkpoint: {ckpt_path}")
    if best_metrics is not None:
        print(format_metrics(best_metrics))


if __name__ == "__main__":
    main()
