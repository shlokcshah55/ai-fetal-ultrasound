from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mean_pool(features: np.ndarray) -> np.ndarray:
    """Mean pooling over frames.

    Args:
        features: Frame embeddings of shape (N, 768).

    Returns:
        Video embedding of shape (768,).
    """
    return features.mean(axis=0)


def max_pool(features: np.ndarray) -> np.ndarray:
    """Element-wise max pooling over frames.

    Args:
        features: Frame embeddings of shape (N, 768).

    Returns:
        Video embedding of shape (768,).
    """
    return features.max(axis=0)


class AttentionPooling(nn.Module):
    """Lightweight learned attention pooling over frames.

    A single linear layer maps each frame embedding to a scalar score,
    scores are softmax-normalised over frames, and the result is a
    weighted sum of frame embeddings.

    This is the only trainable component in the frozen DINOv2 baseline
    (aside from the downstream MLP head).
    """

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.attention_head = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted video embedding.

        Args:
            features: Frame embeddings of shape (N, embed_dim).

        Returns:
            Video embedding of shape (embed_dim,).
        """
        scores = self.attention_head(features)        # (N, 1)
        weights = torch.softmax(scores, dim=0)        # (N, 1)
        pooled = (weights * features).sum(dim=0)      # (embed_dim,)
        return pooled


def attention_pool(
    features: np.ndarray,
    attention_head: AttentionPooling,
) -> np.ndarray:
    """Apply AttentionPooling to numpy frame features.

    Args:
        features: Frame embeddings of shape (N, 768).
        attention_head: Trained AttentionPooling module.

    Returns:
        Video embedding of shape (768,).
    """
    t = torch.from_numpy(features).float()
    with torch.no_grad():
        pooled = attention_head(t)
    return pooled.numpy()
