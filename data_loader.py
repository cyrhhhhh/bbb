"""Data loading for DMRL multimodal sentiment analysis.

Reads the standard MMSA-style processed pickle file (``*.pkl``) used by
CMU-MOSI / CMU-MOSEI / CH-SIMS. The pickle stores a dict with ``train``,
``valid`` and ``test`` splits, each containing aligned text/audio/vision
features and regression labels.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _to_float_tensor(array):
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(array)


class MMDataset(Dataset):
    """Single split of an MMSA-style processed dataset."""

    def __init__(self, data, use_bert=True):
        self.use_bert = use_bert

        if use_bert and "text_bert" in data:
            # text_bert: [N, 3, L] -> (input_ids, attention_mask, token_type_ids)
            self.text = _to_float_tensor(data["text_bert"])
        else:
            self.text = _to_float_tensor(data["text"])

        self.audio = _to_float_tensor(data["audio"])
        self.vision = _to_float_tensor(data["vision"])

        labels = data.get("regression_labels", data.get("labels"))
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        return {
            "text": self.text[index],
            "audio": self.audio[index],
            "vision": self.vision[index],
            "label": self.labels[index],
        }


def load_pickle(data_path):
    with open(data_path, "rb") as f:
        return pickle.load(f)


def build_dataloaders(data_path, batch_size=32, use_bert=True, num_workers=0):
    """Return train/valid/test dataloaders from a single processed pickle file."""
    raw = load_pickle(data_path)

    split_aliases = {
        "train": ["train"],
        "valid": ["valid", "val", "dev"],
        "test": ["test"],
    }

    loaders = {}
    for split, aliases in split_aliases.items():
        key = next((a for a in aliases if a in raw), None)
        if key is None:
            raise KeyError(f"Split '{split}' not found in {data_path}. Keys: {list(raw)}")

        dataset = MMDataset(raw[key], use_bert=use_bert)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )

    return loaders["train"], loaders["valid"], loaders["test"]
