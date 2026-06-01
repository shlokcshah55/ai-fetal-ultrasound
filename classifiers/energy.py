from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class EnergyMLP(nn.Module):
    """Two-logit MLP classifier for deterministic energy OOD scoring."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        # BatchNorm1d (not LayerNorm): at inference it applies a fixed transform
        # from training-population running stats rather than renormalizing each
        # sample to unit scale. Per-sample logit magnitude survives, which is
        # exactly the signal the energy score E(x) = -T*logsumexp(logits/T) relies
        # on. Per-sample LayerNorm flattens that magnitude and kills the OOD signal.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def negative_energy_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return -E(x; f), so larger values are more in-distribution-like."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def predict_energy(
    model: EnergyMLP,
    x: np.ndarray,
    device: torch.device,
    *,
    temperature: float = 1.0,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-class probabilities, negative energy, and raw logits."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    x = np.asarray(x, dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    proba_batches: list[np.ndarray] = []
    energy_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            logits = model(x_batch.to(device))
            proba_batches.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            energy_batches.append(
                negative_energy_from_logits(logits, temperature).cpu().numpy()
            )
            logit_batches.append(logits.cpu().numpy())

    return (
        np.concatenate(proba_batches, axis=0),
        np.concatenate(energy_batches, axis=0),
        np.concatenate(logit_batches, axis=0),
    )


def train_energy_mlp(
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
) -> tuple[EnergyMLP, list[float]]:
    """Train a two-logit MLP with cross-entropy for post-hoc energy scoring."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    model = EnergyMLP(
        input_dim=input_dim,
        dropout=float(config.get("dropout", 0.3)),
    ).to(device)

    # Mirror train_mlp's pos_weight = n_neg / n_pos on the positive (CHD) class;
    # weight[0] stays 1.0. The previous mean-1 rescaling halved the loss scale and
    # helped pin softmax near the prior.
    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    pos_weight = float(class_counts[0]) / max(float(class_counts[1]), 1.0)
    class_weights = np.array([1.0, pos_weight], dtype=np.float32)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

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
        # drop_last=True so BatchNorm1d never sees a size-1 final batch.
        drop_last=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)

    max_epochs = int(config.get("epochs", config.get("mlp_epochs", 100)))
    patience = int(config.get("early_stopping_patience", 10))

    best_val_auroc = -1.0
    patience_counter = 0
    val_auroc_history: list[float] = []

    epoch_iter = range(max_epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training energy MLP", unit="epoch")

    for _ in epoch_iter:
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
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
    return model, val_auroc_history
