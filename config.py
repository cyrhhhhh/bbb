"""Configuration for DMRL multimodal sentiment analysis training.

Defines per-dataset hyper-parameters and a helper that merges dataset defaults
with command-line overrides into a single namespace consumable by the model.
"""

from types import SimpleNamespace


# Per-dataset feature dimensions follow the standard MMSA-style processed data.
# feature_dims = (text_dim, audio_dim, video_dim).
# When use_bert is True, text_dim is replaced by the BERT hidden size at runtime.
DATASET_CONFIG = {
    "mosi": {
        "feature_dims": (768, 5, 20),
        "transformers": "bert",
        "pretrained": "bert-base-uncased",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "bert_lr": 5e-6,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "early_stop": 8,
    },
    "mosei": {
        "feature_dims": (768, 74, 35),
        "transformers": "bert",
        "pretrained": "bert-base-uncased",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "bert_lr": 5e-6,
        "weight_decay": 1e-4,
        "num_epochs": 30,
        "early_stop": 6,
    },
    "sims": {
        "feature_dims": (768, 33, 709),
        "transformers": "bert",
        "pretrained": "bert-base-chinese",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "bert_lr": 5e-6,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "early_stop": 8,
    },
}


# Model + loss defaults shared across datasets. These mirror the paper settings.
MODEL_DEFAULTS = {
    "d_model": 128,
    "n_layers": 3,
    "n_heads": 4,
    "ff_mult": 4,
    "dropout": 0.1,
    "k_slots": 4,
    "use_bert": True,
    "use_finetune": True,
    # Polarity-Coupled Optimal Transport (PCOT).
    "transport_temperature": 0.2,
    "sinkhorn_iters": 5,
    "lambda_p": 0.5,
    "gate_temp": 4.0,
    # Loss weights (w_main, w_ord, w_pol, w_transport).
    "w_main": 1.0,
    "w_ord": 0.2,
    "w_pol": 0.1,
    "w_transport": 0.02,
}


def build_args(dataset, overrides=None):
    """Merge dataset defaults, model defaults and CLI overrides into a namespace."""
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIG)}.")

    cfg = {}
    cfg.update(MODEL_DEFAULTS)
    cfg.update(DATASET_CONFIG[dataset])
    cfg["dataset"] = dataset

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value

    return SimpleNamespace(**cfg)
