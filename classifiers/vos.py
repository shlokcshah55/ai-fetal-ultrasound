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
    """Two-logit MLP classifier for VOS energy training.

    Split into a ``trunk`` (768 -> ... -> feature_dim) and a final linear
    ``classifier`` (feature_dim -> 2). Faithful VOS synthesises virtual outliers
    in the *penultimate* feature space produced by ``trunk`` and routes them
    through ``classifier`` only, so the gradient from the energy regulariser
    reshapes that representation. BatchNorm1d mirrors the working ``train_mlp``
    recipe (vs. the under-confident LayerNorm variant).
    """

    def __init__(
        self,
        input_dim: int = 768,
        dropout: float = 0.3,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.trunk(x))

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, penultimate features)."""
        feats = self.trunk(x)
        return self.classifier(feats), feats

    def classify_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Logits for points already living in the penultimate space."""
        return self.classifier(feats)


class EnergyPhi(nn.Module):
    """Learnable logistic regressor on the scalar energy (VOS's ``phi``).

    Decouples the magnitude objective from softmax-CE: maps the energy score to
    an ID/OOD logit so the binary loss does not act directly on raw logsumexp.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(1, 1)

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        return self.net(energy.reshape(-1, 1)).reshape(-1)


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


def _penultimate_features(
    model: VOSMLP,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract penultimate (trunk) features for an array under model.eval()."""
    x = np.asarray(x, dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    was_training = model.training
    model.eval()
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            _, f = model.forward_features(x_batch.to(device))
            feats.append(f.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(feats, axis=0)


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
    """Train a VOS MLP with virtual-outlier energy regularisation.

    Outliers are synthesised in the model's *penultimate* feature space (refit
    each epoch from the current representation) and routed through the final
    linear layer only, so the energy regulariser reshapes the trunk -- the core
    mechanism of Du et al. (ICLR 2022). The regulariser is warmed up for
    ``vos_start_epoch`` epochs and its energies pass through a learnable ``phi``
    before the binary ID/OOD loss.
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    feature_dim = int(config.get("feature_dim", 64))
    model = VOSMLP(
        input_dim=input_dim,
        dropout=float(config.get("dropout", 0.3)),
        feature_dim=feature_dim,
    ).to(device)
    phi = EnergyPhi(hidden_dim=int(config.get("phi_hidden_dim", 16))).to(device)

    # Class weighting mirrors train_mlp's pos_weight = n_neg / n_pos applied to
    # the positive (CHD) class; weight[0] stays at 1.0 (no mean-1 rescaling that
    # otherwise shrinks the gradient and squashes logits toward the prior).
    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    pos_weight = float(class_counts[0]) / max(float(class_counts[1]), 1.0)
    class_weights = np.array([1.0, pos_weight], dtype=np.float32)
    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    vos_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(phi.parameters()),
        lr=float(config.get("lr", config.get("mlp_lr", 1e-3))),
        weight_decay=float(config.get("weight_decay", config.get("mlp_weight_decay", 1e-4))),
    )

    batch_size = int(config.get("batch_size", 64))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)

    max_epochs = int(config.get("epochs", config.get("mlp_epochs", 100)))
    patience = int(config.get("early_stopping_patience", 10))
    lambda_vos = float(config.get("lambda_vos", 0.1))
    n_outliers_per_batch = int(config.get("virtual_outliers_per_batch", batch_size))
    tail_q_low = float(config.get("tail_q_low", 0.95))
    tail_q_high = float(config.get("tail_q_high", 0.999))
    vos_start_epoch = int(config.get("vos_start_epoch", max(1, int(0.4 * max_epochs))))
    covariance = str(config.get("covariance", "ledoit_wolf"))
    jitter = float(config.get("covariance_jitter", 1e-4))

    best_val_auroc = -1.0
    patience_counter = 0
    val_auroc_history: list[float] = []
    stats: GaussianStats | None = None
    vos_was_active = False

    epoch_iter = range(max_epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training VOS MLP", unit="epoch")

    for epoch in epoch_iter:
        vos_active = epoch >= vos_start_epoch
        if vos_active and not vos_was_active:
            # Give the regularised phase its own early-stopping window so the
            # returned checkpoint always reflects VOS-shaped features rather than
            # an earlier CE-only epoch.
            best_val_auroc = -1.0
            patience_counter = 0
        vos_was_active = vos_active
        if vos_active:
            # Refit the class-conditional Gaussians from the *current* penultimate
            # representation so virtual outliers track the evolving feature space.
            feats_train = _penultimate_features(model, x_train, device)
            stats = fit_gaussian_stats(
                feats_train, y_train, covariance=covariance, jitter=jitter
            )

        model.train()
        running_cls = 0.0
        running_vos = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits_real, _ = model.forward_features(x_batch)
            cls_loss = ce_loss(logits_real, y_batch)
            loss = cls_loss

            if vos_active and stats is not None:
                virtual_np = sample_virtual_outliers(
                    stats,
                    n_per_class=max(1, n_outliers_per_batch // 2),
                    tail_q_low=tail_q_low,
                    tail_q_high=tail_q_high,
                    rng=rng,
                )
                feats_virtual = torch.from_numpy(virtual_np).to(device)
                logits_virtual = model.classify_features(feats_virtual)

                energy_real = energy_from_logits(logits_real)
                energy_virtual = energy_from_logits(logits_virtual)
                phi_logits = phi(torch.cat([energy_real, energy_virtual], dim=0))
                # Target 1 == OOD (virtual), 0 == ID (real).
                phi_targets = torch.cat(
                    [
                        torch.zeros_like(energy_real),
                        torch.ones_like(energy_virtual),
                    ],
                    dim=0,
                )
                vos_term = vos_loss(phi_logits, phi_targets)
                loss = cls_loss + lambda_vos * vos_term
                running_vos += float(vos_term.item())

            loss.backward()
            optimizer.step()

            running_cls += float(cls_loss.item())
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
            epoch_iter.set_postfix(
                cls=f"{running_cls / max(n_batches, 1):.4f}",
                vos=f"{running_vos / max(n_batches, 1):.4f}",
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
    # Return Gaussian stats fit on the *final* representation for downstream use.
    final_stats = fit_gaussian_stats(
        _penultimate_features(model, x_train, device),
        y_train,
        covariance=covariance,
        jitter=jitter,
    )
    return model, val_auroc_history, final_stats


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
