from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from aggregation.pooling import max_pool, mean_pool


def pool_subject_features_with_paths(
    feature_dict: dict[str, tuple[np.ndarray, int]],
    records: list[dict[str, Any]],
    *,
    video_pooling: str = "mean",
) -> tuple[list[int], np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    """Pool frame-level video features into one embedding per subject.

    The pooling is two-stage:
      1. frames -> video embedding using mean or max pooling
      2. videos -> subject embedding using mean pooling

    Args:
        feature_dict: Mapping of video_path to (frame_features, label).
        records: Dataset records containing video_path and subject_id metadata.
        video_pooling: Pooling strategy for frame-to-video aggregation.

    Returns:
        subject_ids, X_subject, y_subject, metadata_by_subject.
    """
    records_by_path = {record["video_path"]: record for record in records}
    subject_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
    subject_labels: dict[int, set[int]] = defaultdict(set)
    metadata_by_subject: dict[int, dict[str, Any]] = {}

    for video_path, (features, label) in feature_dict.items():
        if video_path not in records_by_path:
            raise KeyError(f"No dataset record found for cached feature path: {video_path}")

        record = records_by_path[video_path]
        subject_id = int(record["subject_id"])

        if video_pooling == "mean":
            video_vec = mean_pool(features)
        elif video_pooling == "max":
            video_vec = max_pool(features)
        else:
            raise ValueError(f"Unsupported video pooling: {video_pooling}")

        subject_vectors[subject_id].append(video_vec)
        subject_labels[subject_id].add(int(label))

        if subject_id not in metadata_by_subject:
            metadata_by_subject[subject_id] = dict(record)

    subject_ids = sorted(subject_vectors)
    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for subject_id in subject_ids:
        labels = subject_labels[subject_id]
        if len(labels) != 1:
            raise ValueError(
                f"Subject {subject_id} has inconsistent labels across videos: {sorted(labels)}"
            )
        x_list.append(np.stack(subject_vectors[subject_id], axis=0).mean(axis=0))
        y_list.append(next(iter(labels)))

    if not x_list:
        raise ValueError("No subject features were pooled; the feature dict is empty.")

    return (
        subject_ids,
        np.stack(x_list, axis=0).astype(np.float32),
        np.asarray(y_list, dtype=np.int64),
        metadata_by_subject,
    )
