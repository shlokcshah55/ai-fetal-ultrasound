from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)


def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float | list]:
    """Compute classification metrics for binary imbalanced classification.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold. If None, Youden's J optimal threshold is used.

    Returns:
        Dict with keys: auroc, auprc, macro_f1, sensitivity, specificity,
        threshold, confusion_matrix.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    auroc = float(roc_auc_score(y_true, y_pred_proba))
    auprc = float(average_precision_score(y_true, y_pred_proba))

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_pred_proba)

    y_pred = (y_pred_proba >= threshold).astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    sensitivity = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    specificity = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "auroc": auroc,
        "auprc": auprc,
        "macro_f1": macro_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": float(threshold),
        "confusion_matrix": cm,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Find the optimal classification threshold using Youden's J statistic.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_proba: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # sklearn may return inf as the first threshold (predict-all-negative point)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    youden_j = tpr - fpr
    optimal_idx = int(np.argmax(youden_j))
    return float(thresholds[optimal_idx])
