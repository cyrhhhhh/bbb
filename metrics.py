"""Evaluation metrics for multimodal sentiment analysis.

Implements the standard CMU-MOSI / CMU-MOSEI / CH-SIMS metrics:
MAE, Pearson correlation, 7-class accuracy, 5-class accuracy, binary
accuracy and F1 (both "negative/non-negative" and "negative/positive"
conventions).
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _flatten(preds, labels):
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    return preds, labels


def multiclass_acc(preds, labels):
    return accuracy_score(np.round(labels), np.round(preds))


def eval_mosi_mosei(preds, labels, exclude_zero=True):
    """Metrics for MOSI/MOSEI (labels in [-3, 3])."""
    preds, labels = _flatten(preds, labels)

    test_preds_a7 = np.clip(preds, -3.0, 3.0)
    test_truth_a7 = np.clip(labels, -3.0, 3.0)
    test_preds_a5 = np.clip(preds, -2.0, 2.0)
    test_truth_a5 = np.clip(labels, -2.0, 2.0)

    mae = np.mean(np.abs(preds - labels))
    corr = np.corrcoef(preds, labels)[0, 1] if np.std(preds) > 0 else 0.0
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    acc5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # Negative / non-negative convention (zeros count as non-negative).
    non_neg_preds = preds >= 0
    non_neg_truth = labels >= 0
    acc2_non_neg = accuracy_score(non_neg_truth, non_neg_preds)
    f1_non_neg = f1_score(non_neg_truth, non_neg_preds, average="weighted")

    # Negative / positive convention (drop samples whose label is exactly zero).
    nz = labels != 0
    pos_preds = preds[nz] > 0
    pos_truth = labels[nz] > 0
    acc2_pos = accuracy_score(pos_truth, pos_preds)
    f1_pos = f1_score(pos_truth, pos_preds, average="weighted")

    return {
        "mae": float(mae),
        "corr": float(corr),
        "acc7": float(acc7),
        "acc5": float(acc5),
        "acc2_non_neg": float(acc2_non_neg),
        "f1_non_neg": float(f1_non_neg),
        "acc2_pos": float(acc2_pos),
        "f1_pos": float(f1_pos),
    }


def eval_sims(preds, labels):
    """Metrics for CH-SIMS (labels in [-1, 1])."""
    preds, labels = _flatten(preds, labels)
    preds = np.clip(preds, -1.0, 1.0)
    labels = np.clip(labels, -1.0, 1.0)

    mae = np.mean(np.abs(preds - labels))
    corr = np.corrcoef(preds, labels)[0, 1] if np.std(preds) > 0 else 0.0

    def to_class(values, bounds):
        out = np.zeros_like(values, dtype=np.int64)
        for i, v in enumerate(values):
            for c, (lo, hi) in enumerate(bounds):
                if lo <= v < hi or (c == len(bounds) - 1 and v == hi):
                    out[i] = c
                    break
        return out

    bounds_2 = [(-1.01, 0.0), (0.0, 1.01)]
    bounds_3 = [(-1.01, -0.1), (-0.1, 0.1), (0.1, 1.01)]
    bounds_5 = [(-1.01, -0.7), (-0.7, -0.1), (-0.1, 0.1), (0.1, 0.7), (0.7, 1.01)]

    acc2 = accuracy_score(to_class(labels, bounds_2), to_class(preds, bounds_2))
    f1_2 = f1_score(to_class(labels, bounds_2), to_class(preds, bounds_2), average="weighted")
    acc3 = accuracy_score(to_class(labels, bounds_3), to_class(preds, bounds_3))
    acc5 = accuracy_score(to_class(labels, bounds_5), to_class(preds, bounds_5))

    return {
        "mae": float(mae),
        "corr": float(corr),
        "acc2": float(acc2),
        "f1": float(f1_2),
        "acc3": float(acc3),
        "acc5": float(acc5),
    }


def compute_metrics(preds, labels, dataset):
    if dataset.lower() == "sims":
        return eval_sims(preds, labels)
    return eval_mosi_mosei(preds, labels)
