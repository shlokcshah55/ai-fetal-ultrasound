"""Post-hoc temperature scaling for calibration (Guo et al. 2017).

Temperature scaling fits a single positive scalar ``T`` on a held-out
validation set by minimising NLL, then divides logits by ``T`` before the
sigmoid. Because it is a monotonic rescaling of the score it leaves every
ranking-based metric (AUROC/AUPRC, OOD-AUROC, FPR@95) unchanged and only moves
calibration metrics (ECE/NLL). Fit on the *in-distribution* validation split
only -- never on the held-out OOD set.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Single-parameter temperature. ``T = exp(log_T)`` keeps ``T > 0``."""

    def __init__(self) -> None:
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """Fit a binary temperature by minimising BCE on validation logits.

    Args:
        logits: Raw binary logits (N,) for the positive class.
        labels: Ground-truth labels (N,) in {0, 1}.
        max_iter: LBFGS iterations.
        lr: LBFGS learning rate.

    Returns:
        The fitted temperature T (>= 1 typically means the model was
        over-confident). Falls back to 1.0 if the validation set has a single
        class (BCE then has no calibration signal).
    """
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if logits.size == 0 or np.unique(labels).size < 2:
        return 1.0

    scaler = TemperatureScaler()
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)
    nll = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(scaler(logits_t).squeeze(-1), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.detach().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated positive-class probabilities sigmoid(logits / T)."""
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits / float(temperature)))


def binary_logit_from_proba(proba: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Effective binary logit log(p / (1 - p)) from a positive-class probability.

    Use this to temperature-scale models that expose a probability rather than a
    raw logit (e.g. an evidential head, where p = alpha_1 / S). Equivalent for a
    binary Dirichlet to ``log(alpha_1) - log(alpha_0)`` since the strength S
    cancels -- so calibrating this logit leaves the evidential vacuity u = K / S
    untouched.
    """
    proba = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0 - eps)
    return np.log(proba) - np.log(1.0 - proba)
