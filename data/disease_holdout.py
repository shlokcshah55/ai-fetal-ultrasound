from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.dataset import VideoDataset, collate_variable_frames


CONDITION_COLS = [
    "avsd",
    "hlhs",
    "tga",
    "tetralogy",
    "raa",
    "coa",
    "p_atresia",
    "a_stenosis",
    "p_stenosis",
]

CONDITION_ALIASES = {
    "avsd": "avsd",
    "atrioventricular_septal_defect": "avsd",
    "atrioventricular septal defect": "avsd",
    "hlhs": "hlhs",
    "hypoplastic_left_heart_syndrome": "hlhs",
    "hypoplastic left heart syndrome": "hlhs",
    "tga": "tga",
    "transposition_of_the_great_arteries": "tga",
    "transposition of the great arteries": "tga",
    "tetralogy": "tetralogy",
    "tetralogy_of_fallot": "tetralogy",
    "tetralogy of fallot": "tetralogy",
    "raa": "raa",
    "right_aortic_arch": "raa",
    "right aortic arch": "raa",
    "coa": "coa",
    "coarctation_of_the_aorta": "coa",
    "coarctation of the aorta": "coa",
    "p_atresia": "p_atresia",
    "pulmonary_atresia": "p_atresia",
    "pulmonary atresia": "p_atresia",
    "a_stenosis": "a_stenosis",
    "aortic_stenosis": "a_stenosis",
    "aortic stenosis": "a_stenosis",
    "p_stenosis": "p_stenosis",
    "pulmonary_stenosis": "p_stenosis",
    "pulmonary stenosis": "p_stenosis",
}


def normalise_condition_names(condition_names: list[str]) -> list[str]:
    """Map human-readable disease names to subject-label CSV column names."""
    normalised: list[str] = []
    for name in condition_names:
        key = name.strip().lower().replace("-", "_")
        key = " ".join(key.split())
        key = CONDITION_ALIASES.get(key, CONDITION_ALIASES.get(key.replace(" ", "_")))
        if key is None:
            raise ValueError(
                f"Unknown disease/condition name: {name!r}. "
                f"Valid columns are: {', '.join(CONDITION_COLS)}"
            )
        if key not in normalised:
            normalised.append(key)
    return normalised


def get_heldout_disease_dataloaders(
    csv_path: str,
    subject_labels_path: str,
    heldout_conditions: list[str],
    n_frames: int = 16,
    batch_size: int = 32,
    num_workers: int = 4,
    split: list[float] | None = None,
    seed: int = 42,
    n_folds: int = 1,
    fold: int = 0,
    sononet_dir: str | None = None,
    conf_threshold: float = 0.5,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Build dataloaders for a disease-held-out uncertainty experiment.

    Subjects with any held-out condition are excluded from train/val/ID-test
    and are returned only in the held-out test loader. All remaining normal
    and seen-disease subjects are split subject-wise into train/val/ID-test.
    """
    if split is None:
        split = [0.8, 0.1, 0.1]

    heldout_conditions = normalise_condition_names(heldout_conditions)
    seen_conditions = [c for c in CONDITION_COLS if c not in heldout_conditions]

    df = pd.read_csv(csv_path)
    required_cols = {"video_path", "label", "subject_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    subject_df = pd.read_csv(subject_labels_path)
    if "subject" not in subject_df.columns:
        raise ValueError("subject_labels_path must contain a 'subject' column.")
    missing_conditions = set(CONDITION_COLS) - set(subject_df.columns)
    if missing_conditions:
        raise ValueError(
            f"subject_labels_path missing condition columns: {sorted(missing_conditions)}"
        )

    subject_df = subject_df[["subject", *CONDITION_COLS]].copy()
    subject_df["subject_id"] = subject_df["subject"].astype(int)
    subject_df["has_heldout_disease"] = (
        subject_df[heldout_conditions].sum(axis=1) > 0
    )
    subject_df["has_seen_disease"] = subject_df[seen_conditions].sum(axis=1) > 0
    subject_df["disease_group"] = np.select(
        [
            subject_df["has_heldout_disease"],
            subject_df["has_seen_disease"],
        ],
        [
            "heldout_disease",
            "seen_disease",
        ],
        default="normal",
    )

    df["subject_id"] = df["subject_id"].astype(int)
    df = df.merge(
        subject_df[
            [
                "subject_id",
                "disease_group",
                "has_heldout_disease",
                "has_seen_disease",
                *CONDITION_COLS,
            ]
        ],
        on="subject_id",
        how="inner",
    )

    if sononet_dir is not None:
        df = _apply_sononet_filter(df, csv_path, sononet_dir, conf_threshold)

    id_df = df[df["disease_group"] != "heldout_disease"].reset_index(drop=True)
    heldout_df = df[df["disease_group"] == "heldout_disease"].reset_index(drop=True)
    if n_folds > 1:
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds}); got {fold}")
        # Keep val the same fraction of the total as the single-split config:
        # test is now 1/n_folds, so the val fraction of the train_val pool is
        # scaled up to preserve val's share of the whole ID set.
        val_frac = split[1] / (1.0 - 1.0 / n_folds)
        train_df, val_df, id_test_df = _subject_split_kfold(
            id_df, n_folds=n_folds, fold=fold, val_frac=val_frac, seed=seed
        )
    else:
        train_df, val_df, id_test_df = _subject_split(id_df, split, seed)

    train_loader = _build_loader(train_df, n_frames, batch_size, num_workers, is_train=True)
    val_loader = _build_loader(val_df, n_frames, batch_size, num_workers, is_train=False)
    id_test_loader = _build_loader(id_test_df, n_frames, batch_size, num_workers, is_train=False)
    heldout_loader = _build_loader(heldout_df, n_frames, batch_size, num_workers, is_train=False)

    split_info: dict[str, Any] = {
        "heldout_conditions": heldout_conditions,
        "seen_conditions": seen_conditions,
        "n_folds": int(n_folds),
        "fold": int(fold) if n_folds > 1 else 0,
        "train_subjects": _subject_ids(train_df),
        "val_subjects": _subject_ids(val_df),
        "id_test_subjects": _subject_ids(id_test_df),
        "heldout_subjects": _subject_ids(heldout_df),
        "counts": {
            "train_videos": int(len(train_df)),
            "val_videos": int(len(val_df)),
            "id_test_videos": int(len(id_test_df)),
            "heldout_videos": int(len(heldout_df)),
            "train_subjects": int(train_df["subject_id"].nunique()),
            "val_subjects": int(val_df["subject_id"].nunique()),
            "id_test_subjects": int(id_test_df["subject_id"].nunique()),
            "heldout_subjects": int(heldout_df["subject_id"].nunique()),
        },
    }

    return train_loader, val_loader, id_test_loader, heldout_loader, split_info


def _subject_ids(df: pd.DataFrame) -> list[int]:
    return sorted(int(subject_id) for subject_id in df["subject_id"].unique())


def _apply_sononet_filter(
    df: pd.DataFrame,
    csv_path: str,
    sononet_dir: str,
    conf_threshold: float,
) -> pd.DataFrame:
    sononet_path = Path(sononet_dir)
    cache_key = hashlib.md5(f"{sononet_dir}|{conf_threshold}".encode()).hexdigest()[:8]
    cache_file = Path(csv_path).parent / f"sononet_cache_{cache_key}.pkl"

    if cache_file.exists():
        print(f"  Loading SonoNet frame indices from cache ({cache_file})...")
        with open(cache_file, "rb") as f:
            indices_map: dict[str, list[int]] = pickle.load(f)
    else:
        print("  Scanning SonoNet .pk files (first run only, will be cached)...")
        indices_map = {}
        for video_path in tqdm(df["video_path"], desc="  Scanning .pk files"):
            pk_path = sononet_path / (Path(video_path).stem + ".pk")
            if not pk_path.exists():
                continue
            pk_df = pd.read_pickle(pk_path)
            mask = (pk_df["label"] == "4ch") & (pk_df["probability"] >= conf_threshold)
            qualifying = pk_df.loc[mask, "frame_number"].values
            if len(qualifying) > 0:
                indices_map[video_path] = qualifying.tolist()
        with open(cache_file, "wb") as f:
            pickle.dump(indices_map, f)
        print(f"  Cached to {cache_file}")

    n_before = len(df)
    df = df.copy()
    df["frame_indices"] = df["video_path"].map(indices_map)
    df = df[df["frame_indices"].notna()].reset_index(drop=True)
    print(f"  SonoNet filter: kept {len(df)}/{n_before} videos with qualifying 4CH frames")
    return df


def _subject_split(
    df: pd.DataFrame,
    split: list[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["subject_id"].values
    gss_test = GroupShuffleSplit(n_splits=1, test_size=split[2], random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    val_ratio_of_train_val = split[1] / (split[0] + split[1])
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio_of_train_val,
        random_state=seed,
    )
    train_idx, val_idx = next(
        gss_val.split(df_train_val, groups=df_train_val["subject_id"].values)
    )

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    train_subjects = set(df_train["subject_id"].unique())
    val_subjects = set(df_val["subject_id"].unique())
    test_subjects = set(df_test["subject_id"].unique())
    assert train_subjects.isdisjoint(val_subjects), "Subject leakage: train/val"
    assert train_subjects.isdisjoint(test_subjects), "Subject leakage: train/test"
    assert val_subjects.isdisjoint(test_subjects), "Subject leakage: val/test"

    return df_train, df_val, df_test


def _subject_split_kfold(
    df: pd.DataFrame,
    n_folds: int,
    fold: int,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Subject-level, label-stratified K-fold split for a single fold.

    Partitions the ID subjects into ``n_folds`` disjoint test folds with
    StratifiedGroupKFold (subjects never cross folds, and the positive rate is
    kept roughly even across folds). The requested ``fold`` becomes the test
    set; the remaining folds form the train_val pool, from which a
    subject-grouped validation set is carved. Folds are deterministic for a
    given ``seed``, so every method sees identical partitions.
    """
    groups = df["subject_id"].values
    labels = df["label"].values

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(df, y=labels, groups=groups))
    train_val_idx, test_idx = splits[fold]

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(
        gss_val.split(df_train_val, groups=df_train_val["subject_id"].values)
    )

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    train_subjects = set(df_train["subject_id"].unique())
    val_subjects = set(df_val["subject_id"].unique())
    test_subjects = set(df_test["subject_id"].unique())
    assert train_subjects.isdisjoint(val_subjects), "Subject leakage: train/val"
    assert train_subjects.isdisjoint(test_subjects), "Subject leakage: train/test"
    assert val_subjects.isdisjoint(test_subjects), "Subject leakage: val/test"

    return df_train, df_val, df_test


def _build_loader(
    df: pd.DataFrame,
    n_frames: int,
    batch_size: int,
    num_workers: int,
    *,
    is_train: bool,
) -> DataLoader:
    records = _to_records(df)
    dataset = VideoDataset(records, n_frames=n_frames)

    sampler = None
    if is_train:
        train_labels = [r["label"] for r in records]
        class_counts: dict[int, int] = {}
        for label in train_labels:
            class_counts[int(label)] = class_counts.get(int(label), 0) + 1
        sample_weights = [1.0 / class_counts[int(label)] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_variable_frames,
    )


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "video_path",
        "label",
        "subject_id",
        "disease_group",
        "has_heldout_disease",
        "has_seen_disease",
        *CONDITION_COLS,
    ]
    if "frame_indices" in df.columns:
        cols.append("frame_indices")
    return df[cols].to_dict("records")
