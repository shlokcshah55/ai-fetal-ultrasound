from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aggregation.subject_pooling import pool_subject_features_with_paths
from data.dataset import make_image_transform
from data.disease_holdout import CONDITION_COLS, get_heldout_disease_dataloaders
from features.extract import (
    checkpoint_id_from_path,
    extract_features,
    feature_cache_namespace,
    load_cached_features,
    load_frozen_backbone,
    resolve_fetal_clip_config,
)
from run_baseline import (
    _missing_cache_loader,
    _sequential_loader,
    _video_paths_from_loader,
    load_config,
    set_seeds,
)


DEFAULT_HELDOUT_DISEASES = ["tetralogy", "avsd", "a_stenosis"]
DISPLAY_NAMES = {
    "normal": "Healthy/normal",
    "seen_disease": "Seen disease",
    "tetralogy": "ToF",
    "avsd": "AVSD",
    "a_stenosis": "Aortic Stenosis",
}
COLORS = {
    "normal": "#9ca3af",
    "seen_disease": "#2563eb",
    "tetralogy": "#c2410c",
    "avsd": "#7c3aed",
    "a_stenosis": "#be123c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an optional 2D embedding scatter for held-out diseases. "
            "Use as an illustrative diagnostic only; omit the figure if the "
            "projection is ambiguous."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subject-labels", default="subject_level_labels.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--backbone", choices=["dinov2", "fetal_clip"], default="fetal_clip")
    parser.add_argument("--fetal-clip-checkpoint", default=None)
    parser.add_argument("--fetal-clip-config", default=None)
    parser.add_argument(
        "--feature-cache-namespace",
        default=None,
        help=(
            "Override feature-cache namespace. Useful when the current absolute "
            "checkpoint/config path hashes differ from the namespace used to "
            "create the cache."
        ),
    )
    parser.add_argument("--features-cache-dir", default=None)
    parser.add_argument("--sononet-dir", default=None)
    parser.add_argument("--sononet-conf-threshold", type=float, default=None)
    parser.add_argument("--no-sononet", action="store_true")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "id_test", "heldout"],
        default=["train", "val", "id_test", "heldout"],
    )
    parser.add_argument(
        "--extract-missing",
        action="store_true",
        help="Extract missing features instead of requiring all features to be cached.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "umap", "tsne", "pca"],
        default="auto",
        help="Projection method. auto uses UMAP if installed, otherwise t-SNE.",
    )
    parser.add_argument("--max-subjects", type=int, default=2500)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def records_from_loader(loader: torch.utils.data.DataLoader) -> list[dict[str, Any]]:
    records = getattr(loader.dataset, "records", None)
    if not isinstance(records, list):
        raise TypeError("Expected loader.dataset.records to be a list of dicts.")
    return records


def condition_label(record: dict[str, Any]) -> str:
    if record.get("disease_group") != "heldout_disease":
        return str(record.get("disease_group", "unknown"))
    for condition in DEFAULT_HELDOUT_DISEASES:
        if int(record.get(condition, 0)) == 1:
            return condition
    return "heldout_disease"


def resolve_backbone_meta(args: argparse.Namespace) -> dict[str, str]:
    if args.backbone == "dinov2":
        checkpoint_id = "facebookresearch_dinov2_vitb14"
    else:
        if args.fetal_clip_checkpoint is None:
            raise SystemExit("--fetal-clip-checkpoint is required for --backbone fetal_clip.")
        args.fetal_clip_config = resolve_fetal_clip_config(
            args.fetal_clip_checkpoint,
            args.fetal_clip_config,
        )
        config_id = (
            checkpoint_id_from_path(args.fetal_clip_config)
            if args.fetal_clip_config is not None
            else "no_config"
        )
        checkpoint_id = f"{checkpoint_id_from_path(args.fetal_clip_checkpoint)}_{config_id}"

    namespace = args.feature_cache_namespace or feature_cache_namespace(
        args.backbone,
        checkpoint_id,
    )
    return {
        "backbone": args.backbone,
        "checkpoint_id": checkpoint_id,
        "cache_namespace": namespace,
    }


def load_or_extract_features(
    args: argparse.Namespace,
    split_loaders: dict[str, torch.utils.data.DataLoader],
    cache_dir: str,
    device: torch.device,
    backbone_meta: dict[str, str],
) -> dict[str, dict[str, tuple[np.ndarray, int]]]:
    if not args.extract_missing:
        return {
            split: load_cached_features(
                _video_paths_from_loader(loader),
                cache_dir,
                split_name=split,
                cache_namespace=backbone_meta["cache_namespace"],
            )
            for split, loader in split_loaders.items()
        }

    backbone_model, backbone_meta_loaded = load_frozen_backbone(
        args.backbone,
        device,
        fetal_clip_checkpoint=args.fetal_clip_checkpoint,
        fetal_clip_config=args.fetal_clip_config,
    )
    cache_namespace = args.feature_cache_namespace or backbone_meta_loaded["cache_namespace"]
    for split, loader in split_loaders.items():
        extract_features(
            _missing_cache_loader(loader, cache_dir, split, cache_namespace),
            backbone_model,
            device,
            cache_dir,
            cache_namespace=cache_namespace,
        )
    return {
        split: load_cached_features(
            _video_paths_from_loader(loader),
            cache_dir,
            split_name=split,
            cache_namespace=cache_namespace,
        )
        for split, loader in split_loaders.items()
    }


def collect_subject_rows(
    features: dict[str, dict[str, tuple[np.ndarray, int]]],
    split_loaders: dict[str, torch.utils.data.DataLoader],
    pooling: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []

    for split_name, split_features in features.items():
        subject_ids, x_split, y_split, metadata = pool_subject_features_with_paths(
            split_features,
            records_from_loader(split_loaders[split_name]),
            video_pooling=pooling,
        )
        for idx, subject_id in enumerate(subject_ids):
            record = metadata[int(subject_id)]
            group = condition_label(record)
            rows.append(
                {
                    "subject_id": int(subject_id),
                    "split": split_name,
                    "label": int(y_split[idx]),
                    "disease_group": record.get("disease_group"),
                    "plot_group": group,
                    **{col: int(record.get(col, 0)) for col in CONDITION_COLS},
                }
            )
            arrays.append(x_split[idx])

    if not arrays:
        raise ValueError("No pooled subject embeddings were collected.")
    return pd.DataFrame(rows), np.vstack(arrays)


def sample_subjects(df: pd.DataFrame, x: np.ndarray, max_subjects: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if len(df) <= max_subjects:
        return df.reset_index(drop=True), x

    rng = np.random.default_rng(seed)
    keep = df["plot_group"].isin(DEFAULT_HELDOUT_DISEASES).to_numpy()
    remaining = np.where(~keep)[0]
    n_remaining = max(0, max_subjects - int(keep.sum()))
    if n_remaining < len(remaining):
        sampled = rng.choice(remaining, size=n_remaining, replace=False)
        keep[sampled] = True
    else:
        keep[remaining] = True
    return df.loc[keep].reset_index(drop=True), x[keep]


def standardize(x: np.ndarray) -> np.ndarray:
    mean_vec = x.mean(axis=0, keepdims=True)
    std_vec = x.std(axis=0, keepdims=True)
    std_vec[std_vec == 0.0] = 1.0
    return (x - mean_vec) / std_vec


def project_embeddings(args: argparse.Namespace, x: np.ndarray) -> tuple[np.ndarray, str]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    x_scaled = standardize(np.asarray(x, dtype=np.float32))
    method = args.method
    if method == "auto":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric="euclidean",
                random_state=args.random_state,
            )
            return reducer.fit_transform(x_scaled), "umap"
        except ModuleNotFoundError:
            method = "tsne"

    if method == "umap":
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=args.random_state,
        )
        return reducer.fit_transform(x_scaled), "umap"

    if method == "pca":
        return PCA(n_components=2, random_state=args.random_state).fit_transform(x_scaled), "pca"

    pca_dim = min(50, x_scaled.shape[0] - 1, x_scaled.shape[1])
    x_for_tsne = PCA(n_components=pca_dim, random_state=args.random_state).fit_transform(x_scaled)
    perplexity = min(float(args.perplexity), max(5.0, (len(x_for_tsne) - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    )
    return reducer.fit_transform(x_for_tsne), "tsne"


def overlap_diagnostics(df: pd.DataFrame, xy: np.ndarray) -> dict[str, float | int | str]:
    id_mask = df["plot_group"].isin(["normal", "seen_disease"]).to_numpy()
    heldout_mask = df["plot_group"].isin(DEFAULT_HELDOUT_DISEASES).to_numpy()
    seen_mask = df["plot_group"].eq("seen_disease").to_numpy()
    normal_mask = df["plot_group"].eq("normal").to_numpy()

    if not np.any(heldout_mask) or not np.any(seen_mask) or not np.any(normal_mask):
        return {"diagnostic_warning": "Insufficient groups for overlap diagnostics."}

    heldout_xy = xy[heldout_mask]
    id_xy = xy[id_mask]
    id_groups = df.loc[id_mask, "plot_group"].to_numpy()

    nearest_seen = 0
    for point in heldout_xy:
        distances = np.linalg.norm(id_xy - point[None, :], axis=1)
        nearest_seen += int(id_groups[int(np.argmin(distances))] == "seen_disease")

    seen_centroid = xy[seen_mask].mean(axis=0)
    normal_centroid = xy[normal_mask].mean(axis=0)
    heldout_to_seen = np.linalg.norm(heldout_xy - seen_centroid[None, :], axis=1)
    heldout_to_normal = np.linalg.norm(heldout_xy - normal_centroid[None, :], axis=1)

    return {
        "n_subjects_plotted": int(len(df)),
        "n_heldout_subjects": int(heldout_mask.sum()),
        "heldout_nearest_id_seen_fraction": float(nearest_seen / len(heldout_xy)),
        "heldout_mean_distance_to_seen_centroid": float(heldout_to_seen.mean()),
        "heldout_mean_distance_to_normal_centroid": float(heldout_to_normal.mean()),
        "heldout_seen_centroid_distance_ratio": float(
            heldout_to_seen.mean() / max(heldout_to_normal.mean(), 1e-12)
        ),
    }


def write_outputs(
    df: pd.DataFrame,
    xy: np.ndarray,
    diagnostics: dict[str, Any],
    meta: dict[str, Any],
    output_dir: Path,
    output_stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = df.copy()
    table["x"] = xy[:, 0]
    table["y"] = xy[:, 1]
    csv_path = output_dir / f"{output_stem}_points.csv"
    json_path = output_dir / f"{output_stem}_diagnostics.json"
    table.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump({"diagnostics": diagnostics, "meta": meta}, f, indent=2)
    return csv_path, json_path


def plot_scatter(
    df: pd.DataFrame,
    xy: np.ndarray,
    projection_method: str,
    diagnostics: dict[str, Any],
    output_dir: Path,
    output_stem: str,
    formats: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib is required to render the figure. Install project "
            "requirements, then rerun this script."
        ) from exc

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    for group in ["normal", "seen_disease", *DEFAULT_HELDOUT_DISEASES]:
        mask = df["plot_group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        zorder = 2 if group in ["normal", "seen_disease"] else 4
        size = 12 if group in ["normal", "seen_disease"] else 30
        alpha = 0.35 if group in ["normal", "seen_disease"] else 0.85
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size,
            c=COLORS.get(group, "#111827"),
            label=DISPLAY_NAMES.get(group, group),
            alpha=alpha,
            linewidths=0.0,
            zorder=zorder,
        )

    ratio = diagnostics.get("heldout_seen_centroid_distance_ratio")
    subtitle = ""
    if isinstance(ratio, (float, int)) and math.isfinite(float(ratio)):
        subtitle = f"  held-out/seen-vs-normal distance ratio={float(ratio):.2f}"
    ax.set_title(f"{projection_method.upper()} Projection of Pooled Subject Embeddings")
    ax.set_xlabel(f"{projection_method.upper()} 1")
    ax.set_ylabel(f"{projection_method.upper()} 2")
    ax.text(
        0.01,
        0.01,
        "Illustrative projection only; quantitative evidence is in OOD metrics." + subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#475569",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", markerscale=1.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    seed = int(config["training"]["seed"])
    set_seeds(seed)

    if not 0 <= args.fold < args.n_folds:
        raise SystemExit(f"--fold must be in [0, {args.n_folds}); got {args.fold}.")

    if args.features_cache_dir is not None:
        config["features"]["cache_dir"] = args.features_cache_dir
    if args.sononet_dir is not None:
        config.setdefault("sononet", {})
        config["sononet"]["dir"] = args.sononet_dir
    if args.sononet_conf_threshold is not None:
        config.setdefault("sononet", {})
        config["sononet"]["conf_threshold"] = args.sononet_conf_threshold

    sononet_cfg = config.get("sononet", {})
    sononet_dir = None if args.no_sononet else sononet_cfg.get("dir")
    sononet_conf = sononet_cfg.get("conf_threshold", 0.5)
    cache_dir = sononet_cfg.get("cache_dir") or config["features"]["cache_dir"]
    if args.no_sononet:
        cache_dir = config["features"]["cache_dir"]

    normalization = "clip" if args.backbone == "fetal_clip" else "imagenet"
    transform = make_image_transform(normalization)
    train_loader, val_loader, id_test_loader, heldout_loader, split_info = (
        get_heldout_disease_dataloaders(
            csv_path=config["data"]["csv_path"],
            subject_labels_path=args.subject_labels,
            heldout_conditions=DEFAULT_HELDOUT_DISEASES,
            n_frames=config["data"]["n_frames"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            split=config["data"]["split"],
            seed=seed,
            n_folds=args.n_folds,
            fold=args.fold,
            sononet_dir=sononet_dir,
            conf_threshold=sononet_conf,
            transform=transform,
        )
    )
    all_loaders = {
        "train": _sequential_loader(train_loader),
        "val": _sequential_loader(val_loader),
        "id_test": _sequential_loader(id_test_loader),
        "heldout": _sequential_loader(heldout_loader),
    }
    split_loaders = {name: all_loaders[name] for name in args.splits}

    device_str = args.device or config["features"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    backbone_meta = resolve_backbone_meta(args)
    print(f"Using cache namespace: {backbone_meta['cache_namespace']}")
    features = load_or_extract_features(args, split_loaders, cache_dir, device, backbone_meta)
    df, x = collect_subject_rows(features, split_loaders, args.pooling)
    df, x = sample_subjects(df, x, args.max_subjects, args.random_state)
    xy, projection_method = project_embeddings(args, x)
    diagnostics = overlap_diagnostics(df, xy)

    output_stem = args.output_stem
    if output_stem is None:
        output_stem = f"embedding_scatter_{args.backbone}_fold{args.fold}"
    output_dir = Path(args.output_dir)
    meta = {
        "backbone": args.backbone,
        "checkpoint_id": backbone_meta["checkpoint_id"],
        "cache_namespace": backbone_meta["cache_namespace"],
        "cache_dir": cache_dir,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "splits": args.splits,
        "projection_method": projection_method,
        "heldout_conditions": split_info["heldout_conditions"],
    }
    csv_path, json_path = write_outputs(df, xy, diagnostics, meta, output_dir, output_stem)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(json.dumps(diagnostics, indent=2))

    if args.no_plot:
        return

    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    plot_scatter(df, xy, projection_method, diagnostics, output_dir, output_stem, formats)


if __name__ == "__main__":
    main()
