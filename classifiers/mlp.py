from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


class MLP(nn.Module):
    """Fully-connected MLP classifier for DINOv2 video embeddings.

    Architecture: 768 → 256 → 64 → 1 (raw logit)
    Each hidden layer is followed by BatchNorm, ReLU, and Dropout.
    """

    def __init__(self, input_dim: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, input_dim).

        Returns:
            Raw logits of shape (B,).
        """
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    device: torch.device,
    checkpoint_path: str,
    input_dim: int = 768,
    extra_params: list[nn.Parameter] | None = None,
    show_progress: bool = True,
) -> tuple[MLP, list[float]]:
    """Train the MLP with early stopping on validation AUROC.

    Args:
        X_train: Training features of shape (M_train, input_dim).
        y_train: Training labels of shape (M_train,).
        X_val: Validation features of shape (M_val, input_dim).
        y_val: Validation labels of shape (M_val,).
        config: Training config dict with keys:
            mlp_epochs, mlp_lr, mlp_weight_decay, early_stopping_patience.
        device: torch.device for training.
        checkpoint_path: Path to save the best model checkpoint.
        input_dim: Input feature dimension (default 768).
        extra_params: Additional nn.Parameters to include in the optimizer
            (e.g. attention head parameters for joint training).

    Returns:
        (best_model, val_auroc_history)
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Compute pos_weight from training labels
    y_train_arr = np.asarray(y_train)
    count_pos = int(y_train_arr.sum())
    count_neg = int(len(y_train_arr) - count_pos)
    pos_weight_val = count_neg / max(count_pos, 1)
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    model = MLP(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    params = list(model.parameters())
    if extra_params:
        params = params + list(extra_params)

    lr = float(config["mlp_lr"])
    weight_decay = float(config["mlp_weight_decay"])
    max_epochs = int(config["mlp_epochs"])
    patience = int(config["early_stopping_patience"])

    optimizer = torch.optim.Adam(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Build tensor datasets
    X_tr = torch.from_numpy(X_train.astype(np.float32))
    y_tr = torch.from_numpy(y_train_arr.astype(np.float32))
    X_v = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_v_np = np.asarray(y_val)

    train_ds = TensorDataset(X_tr, y_tr)
    # drop_last=True prevents BatchNorm1d from receiving a batch of size 1
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

    best_val_auroc = -1.0
    patience_counter = 0
    val_auroc_history: list[float] = []

    epoch_iter = range(max_epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training MLP", unit="epoch")

    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_proba = torch.sigmoid(val_logits).cpu().numpy()

        try:
            val_auroc = float(roc_auc_score(y_v_np, val_proba))
        except ValueError:
            # Only one class present in val set — fall back to accuracy
            val_pred = (val_proba >= 0.5).astype(int)
            val_auroc = float(accuracy_score(y_v_np, val_pred))

        val_auroc_history.append(val_auroc)

        if show_progress and hasattr(epoch_iter, "set_postfix"):
            avg_loss = running_loss / max(n_batches, 1)
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", val_auroc=f"{val_auroc:.4f}", best=f"{best_val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model, val_auroc_history
