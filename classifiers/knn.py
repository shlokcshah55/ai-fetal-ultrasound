from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


def get_knn() -> KNeighborsClassifier:
    """k-Nearest Neighbours classifier with cosine distance.

    Uses brute-force search (required for cosine metric) with
    distance-weighted voting.

    Returns:
        Configured KNeighborsClassifier instance.
    """
    return KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="cosine",
        algorithm="brute",  # required for cosine metric
        n_jobs=-1,
    )
