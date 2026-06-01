from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class EvidentialMLP(nn.Module):
    """MLP that outputs non-negative Dirichlet evidence per class."""

    def __init__(
        self,
        input_dim: int = 768,
        dropout: float = 0.3,
        evidence_activation: str = "softplus",
    ) -> None:
        super().__init__()
        if evidence_activation == "relu":
            final_activation: nn.Module = nn.ReLU()
        elif evidence_activation == "softplus":
            final_activation = nn.Softplus()
        else:
            raise ValueError("evidence_activation must be 'relu' or 'softplus'.")

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
            final_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return evidence with shape (B, 2)."""
        return self.net(x)


def evidence_to_dirichlet(
    evidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert non-negative evidence to alpha, total strength, and mean p."""
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    proba = alpha / strength
    return alpha, strength, proba


def dirichlet_uncertainty(
    evidence: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """Return EDL uncertainty mass K / S."""
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1)
    return float(num_classes) / strength


def edl_kl_divergence(alpha: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(1, ..., 1)] for a batch of Dirichlet parameters."""
    strength = alpha.sum(dim=1, keepdim=True)
    log_uniform_beta = torch.lgamma(
        torch.tensor(float(num_classes), dtype=alpha.dtype, device=alpha.device)
    )
    kl = (
        torch.lgamma(strength)
        - log_uniform_beta
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + (
            (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))
        ).sum(dim=1, keepdim=True)
    )
    return kl.squeeze(1)


def edl_loss(
    evidence: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    *,
    num_classes: int = 2,
    annealing_epochs: int = 10,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum-of-squares EDL loss with annealed KL regularisation."""
    if annealing_epochs < 1:
        raise ValueError("annealing_epochs must be >= 1.")

    labels = labels.long()
    alpha, strength, proba = evidence_to_dirichlet(evidence)
    targets = F.one_hot(labels, num_classes=num_classes).to(
        dtype=evidence.dtype,
        device=evidence.device,
    )

    err = (targets - proba).pow(2).sum(dim=1)
    var = (proba * (1.0 - proba) / (strength + 1.0)).sum(dim=1)

    alpha_tilde = targets + (1.0 - targets) * alpha
    loss_kl = edl_kl_divergence(alpha_tilde, num_classes=num_classes)
    annealing_coef = min(1.0, float(epoch) / float(annealing_epochs))
    loss = err + var + annealing_coef * loss_kl

    if class_weights is None:
        return loss.mean()

    sample_weights = class_weights.to(device=evidence.device, dtype=evidence.dtype)[
        labels
    ]
    return (loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)


def predict_edl(
    model: EvidentialMLP,
    x: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Run single-pass EDL inference on pooled subject embeddings."""
    x = np.asarray(x, dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    proba_batches: list[np.ndarray] = []
    uncertainty_batches: list[np.ndarray] = []
    evidence_batches: list[np.ndarray] = []
    alpha_batches: list[np.ndarray] = []
    strength_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            evidence = model(x_batch.to(device))
            alpha, strength, proba = evidence_to_dirichlet(evidence)
            uncertainty = dirichlet_uncertainty(evidence, num_classes=2)
            proba_batches.append(proba[:, 1].cpu().numpy())
            uncertainty_batches.append(uncertainty.cpu().numpy())
            evidence_batches.append(evidence.cpu().numpy())
            alpha_batches.append(alpha.cpu().numpy())
            strength_batches.append(strength.squeeze(1).cpu().numpy())

    return {
        "proba": np.concatenate(proba_batches, axis=0),
        "uncertainty": np.concatenate(uncertainty_batches, axis=0),
        "evidence": np.concatenate(evidence_batches, axis=0),
        "alpha": np.concatenate(alpha_batches, axis=0),
        "strength": np.concatenate(strength_batches, axis=0),
    }


def train_edl_mlp(
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
) -> tuple[EvidentialMLP, list[float]]:
    """Train an evidential MLP with early stopping on validation AUROC."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    model = EvidentialMLP(
        input_dim=input_dim,
        dropout=float(config.get("dropout", 0.3)),
        evidence_activation=str(config.get("evidence_activation", "softplus")),
    ).to(device)

    class_weights = None
    if bool(config.get("class_weighting", True)):
        class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
        weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", config.get("mlp_lr", 1e-3))),
        weight_decay=float(
            config.get("weight_decay", config.get("mlp_weight_decay", 1e-4))
        ),
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
    annealing_epochs = int(config.get("annealing_epochs", 10))

    best_val_auroc = -1.0
    patience_counter = 0
    val_auroc_history: list[float] = []

    epoch_iter = range(max_epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training evidential MLP", unit="epoch")

    for epoch_idx in epoch_iter:
        model.train()
        running_loss = 0.0
        n_batches = 0
        loss_epoch = epoch_idx + 1

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            evidence = model(x_batch)
            loss = edl_loss(
                evidence,
                y_batch,
                loss_epoch,
                num_classes=2,
                annealing_epochs=annealing_epochs,
                class_weights=class_weights,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_evidence = model(x_val_t)
            _, _, val_proba_t = evidence_to_dirichlet(val_evidence)
            val_proba = val_proba_t[:, 1].cpu().numpy()
            val_evidence_mean = val_evidence.mean(dim=0).cpu().numpy()
            val_proba_std = float(np.std(val_proba))

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
                val_p_std=f"{val_proba_std:.4f}",
                e0=f"{val_evidence_mean[0]:.3f}",
                e1=f"{val_evidence_mean[1]:.3f}",
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
    return model, val_auroc_history
