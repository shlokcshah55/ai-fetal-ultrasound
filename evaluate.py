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


def predictive_entropy(
    y_pred_proba: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Binary predictive entropy normalised to [0, 1].

    Entropy is highest when p(CHD)=0.5 and lowest near 0 or 1. This is a
    simple baseline uncertainty score for probabilistic binary classifiers.
    """
    p = np.asarray(y_pred_proba, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return entropy / np.log(2.0)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al. 2017), equal-width confidence bins.

    For binary classification we treat ``y_pred_proba`` as the model's
    confidence in the positive class. Within each bin the calibration gap is
    |mean predicted probability - empirical positive rate|, weighted by the
    fraction of samples in the bin. Temperature scaling should reduce this
    without affecting any ranking-based metric (AUROC/AUPRC).

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        ECE in [0, 1]; lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    if y_true.size == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y_true.size
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Left-open bins, with the first bin including 0.0.
        if lo == 0.0:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
        else:
            mask = (y_pred_proba > lo) & (y_pred_proba <= hi)
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        confidence = float(y_pred_proba[mask].mean())
        accuracy = float(y_true[mask].mean())  # empirical positive rate
        ece += (bin_count / n) * abs(accuracy - confidence)
    return float(ece)


def evaluate_uncertainty_as_ood(
    id_uncertainty: np.ndarray,
    ood_uncertainty: np.ndarray,
) -> dict[str, float]:
    """Evaluate whether uncertainty separates ID from OOD examples.

    Args:
        id_uncertainty: Uncertainty scores for in-distribution test examples.
        ood_uncertainty: Uncertainty scores for held-out/OOD examples.

    Returns:
        AUROC/AUPRC using uncertainty as the positive-class OOD score, plus
        mean uncertainty for each group.
    """
    id_uncertainty = np.asarray(id_uncertainty, dtype=float)
    ood_uncertainty = np.asarray(ood_uncertainty, dtype=float)

    y_true = np.concatenate([
        np.zeros_like(id_uncertainty, dtype=int),
        np.ones_like(ood_uncertainty, dtype=int),
    ])
    scores = np.concatenate([id_uncertainty, ood_uncertainty])

    return {
        "ood_auroc": float(roc_auc_score(y_true, scores)),
        "ood_auprc": float(average_precision_score(y_true, scores)),
        "id_uncertainty_mean": float(np.mean(id_uncertainty)),
        "heldout_uncertainty_mean": float(np.mean(ood_uncertainty)),
    }
