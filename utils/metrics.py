import numpy as np
import torch


def compute_accuracy(outputs, labels):
    """Batch accuracy from model outputs."""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def compute_feature_norms(features):
    """Mean and std of L2 norms across feature vectors."""
    norms = np.linalg.norm(features, axis=1)
    return float(norms.mean()), float(norms.std())


# ── Scenario 3: Few-shot metrics ──────────────────────────────────────────────

def relative_performance_drop(acc_100, acc_5):
    """Δ = (Acc_100% - Acc_5%) / Acc_100%"""
    return (acc_100 - acc_5) / acc_100


def train_val_gap(train_acc, val_acc):
    """Positive gap = overfitting."""
    return train_acc - val_acc


# ── Scenario 4: Corruption metrics ────────────────────────────────────────────

def corruption_error(acc_corrupted):
    """CE = 1 - Acc_corrupted"""
    return 1.0 - acc_corrupted


def relative_robustness(acc_corrupted, acc_clean):
    """RR = Acc_corrupted / Acc_clean"""
    return acc_corrupted / acc_clean