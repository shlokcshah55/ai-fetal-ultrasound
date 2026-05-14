from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class VOSMLP(nn.Module):
    """Two-logit MLP classifier for VOS energy training."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class GaussianStats:
    means: dict[int, np.ndarray]
    covariance: np.ndarray
    covariance_factor: np.ndarray
    precision: np.ndarray


def energy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Energy score where larger values are more OOD-like."""
    return -torch.logsumexp(logits, dim=1)


def predict_vos(
    model: VOSMLP,
    x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-class probabilities, energy scores, and raw logits."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32)).to(device)
        logits = model(x_t)
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        energy = energy_from_logits(logits).cpu().numpy()
    return proba, energy, logits.cpu().numpy()


def fit_gaussian_stats(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    covariance: str = "ledoit_wolf",
    jitter: float = 1e-4,
) -> GaussianStats:
    """Fit class means and a shared covariance matrix over ID embeddings."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    classes = sorted(int(c) for c in np.unique(y_train))
    if classes != [0, 1]:
        raise ValueError(f"VOS expects binary classes [0, 1], got {classes}")

    means = {cls: x_train[y_train == cls].mean(axis=0) for cls in classes}

    if covariance == "ledoit_wolf":
        cov = LedoitWolf().fit(x_train).covariance_
    elif covariance == "diagonal":
        variances = np.var(x_train, axis=0, ddof=1)
        cov = np.diag(np.maximum(variances, jitter))
    elif covariance == "empirical":
        cov = np.cov(x_train, rowvar=False)
    else:
        raise ValueError(f"Unknown covariance estimator: {covariance}")

    cov = np.asarray(cov, dtype=np.float64)
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * jitter
    precision = np.linalg.pinv(cov)
    covariance_factor = _covariance_factor(cov)
    return GaussianStats(
        means=means,
        covariance=cov,
        covariance_factor=covariance_factor,
        precision=precision,
    )


def sample_virtual_outliers(
    stats: GaussianStats,
    *,
    n_per_class: int,
    tail_q_low: float = 0.95,
    tail_q_high: float = 0.999,
    rng: np.random.Generator | None = None,
    max_rounds: int = 20,
) -> np.ndarray:
    """Sample points from a Gaussian tail band around each class centroid."""
    if rng is None:
        rng = np.random.default_rng()
    if not 0.0 < tail_q_low < tail_q_high < 1.0:
        raise ValueError("Expected 0 < tail_q_low < tail_q_high < 1.")

    dim = stats.covariance.shape[0]
    low = _chi_square_quantile(tail_q_low, dim)
    high = _chi_square_quantile(tail_q_high, dim)
    samples: list[np.ndarray] = []

    for cls, mean in stats.means.items():
        class_samples: list[np.ndarray] = []
        candidates_seen: list[np.ndarray] = []
        distances_seen: list[np.ndarray] = []

        for _ in range(max_rounds):
            if sum(len(batch) for batch in class_samples) >= n_per_class:
                break
            n_candidates = max(2048, n_per_class * 32)
            candidates = _sample_gaussian(mean, stats.covariance_factor, n_candidates, rng)
            distances = _mahalanobis_squared(candidates, mean, stats.precision)
            mask = (distances >= low) & (distances <= high)

            candidates_seen.append(candidates)
            distances_seen.append(distances)
            if np.any(mask):
                class_samples.append(candidates[mask])

        if class_samples:
            accepted = np.concatenate(class_samples, axis=0)
        else:
            accepted = np.empty((0, dim), dtype=np.float64)

        if len(accepted) < n_per_class:
            all_candidates = np.concatenate(candidates_seen, axis=0)
            all_distances = np.concatenate(distances_seen, axis=0)
            order = np.argsort(np.abs(all_distances - low))
            fallback = all_candidates[order[: n_per_class - len(accepted)]]
            accepted = np.concatenate([accepted, fallback], axis=0)

        choice = rng.choice(len(accepted), size=n_per_class, replace=len(accepted) < n_per_class)
        samples.append(accepted[choice])

    return np.concatenate(samples, axis=0).astype(np.float32)


def train_vos_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    device: torch.device,
    checkpoint_path: str,
    *,
    input_dim: int = 768,
    show_progress: bool = True,
) -> tuple[VOSMLP, list[float], GaussianStats]:
    """Train a VOS MLP with virtual-outlier energy regularisation."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    stats = fit_gaussian_stats(
        x_train,
        y_train,
        covariance=str(config.get("covariance", "ledoit_wolf")),
        jitter=float(config.get("covariance_jitter", 1e-4)),
    )

    model = VOSMLP(
        input_dim=input_dim,
        dropout=float(config.get("dropout", 0.3)),
    ).to(device)

    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    vos_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", config.get("mlp_lr", 1e-3))),
        weight_decay=float(config.get("weight_decay", config.get("mlp_weight_decay", 1e-4))),
    )

    batch_size = int(config.get("batch_size", 64))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    x_val_t = torch.from_numpy(x_val).to(device)

    max_epochs = int(config.get("epochs", config.get("mlp_epochs", 100)))
    patience = int(config.get("early_stopping_patience", 10))
    lambda_vos = float(config.get("lambda_vos", 0.1))
    n_outliers_per_batch = int(config.get("virtual_outliers_per_batch", batch_size))
    tail_q_low = float(config.get("tail_q_low", 0.95))
    tail_q_high = float(config.get("tail_q_high", 0.999))

    best_val_auroc = -1.0
    patience_counter = 0
    val_auroc_history: list[float] = []

    epoch_iter = range(max_epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training VOS MLP", unit="epoch")

    for _ in epoch_iter:
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            virtual_np = sample_virtual_outliers(
                stats,
                n_per_class=max(1, n_outliers_per_batch // 2),
                tail_q_low=tail_q_low,
                tail_q_high=tail_q_high,
                rng=rng,
            )
            x_virtual = torch.from_numpy(virtual_np).to(device)

            optimizer.zero_grad()
            logits_real = model(x_batch)
            logits_virtual = model(x_virtual)

            cls_loss = ce_loss(logits_real, y_batch)
            energy_real = energy_from_logits(logits_real)
            energy_virtual = energy_from_logits(logits_virtual)
            id_logits = torch.cat([-energy_real, -energy_virtual], dim=0)
            id_targets = torch.cat(
                [
                    torch.ones_like(energy_real),
                    torch.zeros_like(energy_virtual),
                ],
                dim=0,
            )
            loss = cls_loss + lambda_vos * vos_loss(id_logits, id_targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_proba = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()

        try:
            val_auroc = float(roc_auc_score(y_val, val_proba))
        except ValueError:
            val_pred = (val_proba >= 0.5).astype(int)
            val_auroc = float(accuracy_score(y_val, val_pred))

        val_auroc_history.append(val_auroc)
        if show_progress and hasattr(epoch_iter, "set_postfix"):
            avg_loss = running_loss / max(n_batches, 1)
            epoch_iter.set_postfix(
                loss=f"{avg_loss:.4f}",
                val_auroc=f"{val_auroc:.4f}",
                best=f"{best_val_auroc:.4f}",
            )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, val_auroc_history, stats


def _mahalanobis_squared(
    x: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    delta = x - mean
    return np.einsum("ij,jk,ik->i", delta, precision, delta)


def _chi_square_quantile(q: float, dim: int) -> float:
    """Wilson-Hilferty approximation to avoid requiring scipy."""
    z = NormalDist().inv_cdf(q)
    return float(dim * (1.0 - 2.0 / (9.0 * dim) + z * np.sqrt(2.0 / (9.0 * dim))) ** 3)


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(np.sqrt(eigvals))


def _sample_gaussian(
    mean: np.ndarray,
    covariance_factor: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    z = rng.standard_normal(size=(n_samples, covariance_factor.shape[0]))
    return z @ covariance_factor.T + mean
