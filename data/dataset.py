from __future__ import annotations

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit


_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VideoDataset(Dataset):
    """PyTorch Dataset for fetal echocardiography videos.

    Each item returns (frames_tensor, label, video_path) where
    frames_tensor has shape (N, 3, 224, 224).
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        n_frames: int = 16,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.records = records
        self.n_frames = n_frames
        self.transform = transform if transform is not None else _TRANSFORM

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[idx]
        video_path: str = record["video_path"]
        label: int = int(record["label"])

        frames = self._load_frames(video_path, record.get("frame_indices"))
        label_tensor = torch.tensor(label, dtype=torch.long)
        return frames, label_tensor, video_path

    def _load_frames(self, video_path: str, frame_indices: list[int] | None = None) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        try:
            if frame_indices is None:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    total_frames = 1
                frame_indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)

            frame_tensors: list[torch.Tensor] = []
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Substitute black frame on read failure
                    frame_tensor = torch.zeros(3, 224, 224)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frame_tensor = self.transform(pil_img)
                frame_tensors.append(frame_tensor)
        finally:
            cap.release()

        return torch.stack(frame_tensors, dim=0)  # (N, 3, 224, 224)


def get_dataloaders(
    csv_path: str,
    n_frames: int = 16,
    batch_size: int = 32,
    num_workers: int = 4,
    split: list[float] | None = None,
    seed: int = 42,
    sononet_dir: str | None = None,
    conf_threshold: float = 0.5,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[str]]]:
    """Build train/val/test DataLoaders with subject-level splits.

    Returns:
        (train_loader, val_loader, test_loader, split_info)
        split_info maps 'train_subjects', 'val_subjects', 'test_subjects'
        to lists of subject IDs.
    """
    if split is None:
        split = [0.8, 0.1, 0.1]

    df = pd.read_csv(csv_path)
    required_cols = {"video_path", "label", "subject_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if sononet_dir is not None:
        sononet_path = Path(sononet_dir)

        def _compute_frame_indices(row: dict) -> list[int] | None:
            pk_path = sononet_path / (Path(row["video_path"]).stem + ".pk")
            if not pk_path.exists():
                return None
            pk_df = pd.read_pickle(pk_path)
            mask = (pk_df["label"] == "4ch") & (pk_df["probability"] >= conf_threshold)
            qualifying = pk_df.loc[mask, "frame_number"].values
            if len(qualifying) == 0:
                return None
            return qualifying.tolist()

        n_before = len(df)
        df["frame_indices"] = df.apply(_compute_frame_indices, axis=1)
        df = df[df["frame_indices"].notna()].reset_index(drop=True)
        print(f"  SonoNet filter: kept {len(df)}/{n_before} videos with qualifying 4CH frames")

    groups = df["subject_id"].values

    # Pass 1: split off test set
    test_ratio = split[2]
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Pass 2: split train_val into train and val
    val_ratio_of_train_val = split[1] / (split[0] + split[1])
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_of_train_val, random_state=seed)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val["subject_id"].values))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Assert no subject leakage
    train_subjects = set(df_train["subject_id"].unique())
    val_subjects = set(df_val["subject_id"].unique())
    test_subjects = set(df_test["subject_id"].unique())
    assert train_subjects.isdisjoint(val_subjects), "Subject leakage: train ∩ val is non-empty"
    assert train_subjects.isdisjoint(test_subjects), "Subject leakage: train ∩ test is non-empty"
    assert val_subjects.isdisjoint(test_subjects), "Subject leakage: val ∩ test is non-empty"

    def _to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        cols = ["video_path", "label", "subject_id"]
        if "frame_indices" in dataframe.columns:
            cols.append("frame_indices")
        return dataframe[cols].to_dict("records")

    train_records = _to_records(df_train)
    val_records = _to_records(df_val)
    test_records = _to_records(df_test)

    train_dataset = VideoDataset(train_records, n_frames=n_frames)
    val_dataset = VideoDataset(val_records, n_frames=n_frames)
    test_dataset = VideoDataset(test_records, n_frames=n_frames)

    # Class-weighted sampler for training
    train_labels = [r["label"] for r in train_records]
    class_counts: dict[int, int] = {}
    for lbl in train_labels:
        class_counts[int(lbl)] = class_counts.get(int(lbl), 0) + 1
    sample_weights = [1.0 / class_counts[int(lbl)] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # mutually exclusive with sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    split_info: dict[str, list[str]] = {
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
    }

    return train_loader, val_loader, test_loader, split_info
