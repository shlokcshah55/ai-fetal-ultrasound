from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, pairwise_distances, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aggregation.subject_pooling import pool_subject_features_with_paths
from classifiers.linear_probe import get_logistic_regression
from data.dataset import make_image_transform
from data.disease_holdout import CONDITION_COLS, get_heldout_disease_dataloaders
from evaluate import evaluate, evaluate_uncertainty_as_ood, find_optimal_threshold
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


DEFAULT_HELDOUT_DISEASES = [
    "tetralogy",
    "avsd",
    "a_stenosis",
]


def records_from_loader(loader: torch.utils.data.DataLoader) -> list[dict[str, Any]]:
    records = getattr(loader.dataset, "records", None)
    if not isinstance(records, list):
        raise TypeError("Expected loader.dataset.records to be a list of dicts.")
    return records


def safe_classification_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return find_optimal_threshold(y_true, y_proba)


def default_logreg_checkpoint(backbone: str, fold: int, n_folds: int, pooling: str) -> Path:
    if backbone == "fetal_clip":
        prefix = "heldout_disease_baseline_fetal_clip"
    else:
        prefix = "heldout_disease_logreg"
    if n_folds > 1:
        prefix = f"{prefix}_fold{fold}of{n_folds}"
    return Path("checkpoints") / f"{prefix}_{pooling}_LogisticRegression.joblib"


def positive_class_proba(probe: Any, x: np.ndarray) -> np.ndarray:
    classes = list(getattr(probe, "classes_", []))
    if 1 not in classes:
        raise ValueError("Probe checkpoint does not expose class 1 in classes_.")
    return probe.predict_proba(x)[:, classes.index(1)]


def centroid_distance(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_query: np.ndarray,
    *,
    label: int,
    metric: str,
) -> np.ndarray:
    mask = y_ref == label
    if not np.any(mask):
        return np.full(len(x_query), np.nan, dtype=float)
    centroid = x_ref[mask].mean(axis=0, keepdims=True)
    return pairwise_distances(x_query, centroid, metric=metric).reshape(-1)


def nearest_distance(
    x_ref: np.ndarray,
    x_query: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    return pairwise_distances(x_query, x_ref, metric=metric).min(axis=1)


def ood_metrics_for_score(
    name: str,
    id_score: np.ndarray,
    heldout_score: np.ndarray,
) -> dict[str, float]:
    id_score = np.asarray(id_score, dtype=float)
    heldout_score = np.asarray(heldout_score, dtype=float)
    keep_id = np.isfinite(id_score)
    keep_heldout = np.isfinite(heldout_score)
    if not np.any(keep_id) or not np.any(keep_heldout):
        return {
            f"{name}_ood_auroc": float("nan"),
            f"{name}_ood_auprc": float("nan"),
            f"{name}_id_mean": float("nan"),
            f"{name}_heldout_mean": float("nan"),
        }
    metrics = evaluate_uncertainty_as_ood(id_score[keep_id], heldout_score[keep_heldout])
    return {
        f"{name}_ood_auroc": metrics["ood_auroc"],
        f"{name}_ood_auprc": metrics["ood_auprc"],
        f"{name}_id_mean": metrics["id_uncertainty_mean"],
        f"{name}_heldout_mean": metrics["heldout_uncertainty_mean"],
    }


def build_prediction_rows(
    split_name: str,
    subject_ids: list[int],
    y_true: np.ndarray,
    y_proba: np.ndarray,
    scores: dict[str, np.ndarray],
    metadata: dict[int, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    y_pred = (y_proba >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for idx, subject_id in enumerate(subject_ids):
        record = metadata[int(subject_id)]
        row = {
            "split": split_name,
            "subject_id": int(subject_id),
            "disease_group": record["disease_group"],
            "label": int(y_true[idx]),
            "pred_proba_chd": float(y_proba[idx]),
            "pred_label": int(y_pred[idx]),
        }
        for name, values in scores.items():
            row[name] = float(values[idx])
        for col in CONDITION_COLS:
            row[col] = int(record[col])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe frozen-backbone representation quality under disease holdout."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subject-labels", default="subject_level_labels.csv")
    parser.add_argument("--heldout-diseases", nargs="+", default=DEFAULT_HELDOUT_DISEASES)
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--backbone", choices=["dinov2", "fetal_clip"], default="dinov2")
    parser.add_argument("--fetal-clip-checkpoint", default=None)
    parser.add_argument("--fetal-clip-config", default=None)
    parser.add_argument(
        "--backbone-normalization",
        choices=["auto", "imagenet", "clip"],
        default="auto",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--extract-splits",
        nargs="+",
        choices=["train", "val", "id_test", "heldout"],
        default=["train", "val", "id_test", "heldout"],
    )
    parser.add_argument("--features-cache-dir", default=None)
    parser.add_argument(
        "--feature-cache-namespace",
        default=None,
        help=(
            "Override the feature-cache namespace. Use this to reuse cached "
            "FETAL-CLIP embeddings when the checkpoint/config path-derived ID "
            "differs from the original cache namespace."
        ),
    )
    parser.add_argument("--sononet-dir", default=None)
    parser.add_argument("--sononet-conf-threshold", type=float, default=None)
    parser.add_argument("--no-sononet", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default="heldout_disease_representation_probe")
    parser.add_argument(
        "--probe-checkpoint",
        default=None,
        help=(
            "Optional LogisticRegression checkpoint to reuse for probe probability "
            "and margin scores. Defaults to the fold's baseline LR checkpoint for "
            "DINOv2."
        ),
    )
    parser.add_argument(
        "--fit-probe",
        action="store_true",
        help="Fit a fresh LogisticRegression probe instead of loading the baseline checkpoint.",
    )
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    seed = config["training"]["seed"]
    set_seeds(seed)

    if args.n_folds < 1:
        raise SystemExit("--n-folds must be >= 1.")
    if not 0 <= args.fold < args.n_folds:
        raise SystemExit(f"--fold must be in [0, {args.n_folds}); got {args.fold}.")
    if args.backbone == "fetal_clip" and args.fetal_clip_checkpoint is None:
        raise SystemExit("--fetal-clip-checkpoint is required for --backbone fetal_clip.")
    # Historical fetal-clip probe outputs include the backbone in the stem, but
    # the requested DINOv2 cell mirrors the original DINOv2 naming without
    # appending `_dinov2`.
    if args.backbone == "fetal_clip":
        args.output_prefix = f"{args.output_prefix}_{args.backbone}"
    if args.n_folds > 1:
        args.output_prefix = f"{args.output_prefix}_fold{args.fold}of{args.n_folds}"

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

    device_str = args.device or config["features"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    normalization = args.backbone_normalization
    if normalization == "auto":
        normalization = "clip" if args.backbone == "fetal_clip" else "imagenet"
    transform = make_image_transform(normalization)

    train_loader, val_loader, id_test_loader, heldout_loader, split_info = (
        get_heldout_disease_dataloaders(
            csv_path=config["data"]["csv_path"],
            subject_labels_path=args.subject_labels,
            heldout_conditions=args.heldout_diseases,
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

    split_loaders = {
        "train": _sequential_loader(train_loader),
        "val": _sequential_loader(val_loader),
        "id_test": _sequential_loader(id_test_loader),
        "heldout": _sequential_loader(heldout_loader),
    }

    checkpoint_id = "facebookresearch_dinov2_vitb14"
    if args.backbone == "fetal_clip":
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
    backbone_meta = {
        "backbone": args.backbone,
        "checkpoint_id": checkpoint_id,
        "cache_namespace": (
            feature_cache_namespace(args.backbone, checkpoint_id)
            if args.backbone == "fetal_clip"
            else None
        ),
    }
    if args.feature_cache_namespace is not None:
        backbone_meta["cache_namespace"] = args.feature_cache_namespace

    features: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    if args.skip_extract:
        for split_name, loader in split_loaders.items():
            features[split_name] = load_cached_features(
                _video_paths_from_loader(loader),
                cache_dir,
                split_name=split_name,
                cache_namespace=backbone_meta["cache_namespace"],
            )
    else:
        backbone_model, backbone_meta = load_frozen_backbone(
            args.backbone,
            device,
            fetal_clip_checkpoint=args.fetal_clip_checkpoint,
            fetal_clip_config=args.fetal_clip_config,
        )
        if args.backbone == "dinov2":
            backbone_meta["cache_namespace"] = None
        if args.feature_cache_namespace is not None:
            backbone_meta["cache_namespace"] = args.feature_cache_namespace
        for split_name, loader in split_loaders.items():
            if split_name not in args.extract_splits:
                features[split_name] = {}
                continue
            extract_loader = loader
            if args.extract_only:
                extract_loader = _missing_cache_loader(
                    loader,
                    cache_dir,
                    split_name,
                    backbone_meta["cache_namespace"],
                )
            features[split_name] = extract_features(
                extract_loader,
                backbone_model,
                device,
                cache_dir,
                cache_namespace=backbone_meta["cache_namespace"],
            )

    if args.extract_only:
        print("Extract-only mode: done after feature caching.")
        return

    train_subjects, x_train, y_train, _ = pool_subject_features_with_paths(
        features["train"],
        records_from_loader(split_loaders["train"]),
        video_pooling=args.pooling,
    )
    val_subjects, x_val, y_val, _ = pool_subject_features_with_paths(
        features["val"],
        records_from_loader(split_loaders["val"]),
        video_pooling=args.pooling,
    )
    id_subjects, x_id, y_id, id_metadata = pool_subject_features_with_paths(
        features["id_test"],
        records_from_loader(split_loaders["id_test"]),
        video_pooling=args.pooling,
    )
    heldout_subjects, x_heldout, y_heldout, heldout_metadata = (
        pool_subject_features_with_paths(
            features["heldout"],
            records_from_loader(split_loaders["heldout"]),
            video_pooling=args.pooling,
        )
    )
    embedding_dim = int(x_train.shape[1])

    probe_checkpoint_path: Path | None = None
    if args.probe_checkpoint is not None:
        probe_checkpoint_path = Path(args.probe_checkpoint)
    elif args.backbone == "dinov2" and not args.fit_probe:
        probe_checkpoint_path = default_logreg_checkpoint(
            args.backbone,
            args.fold,
            args.n_folds,
            args.pooling,
        )

    if probe_checkpoint_path is not None and not args.fit_probe:
        if not probe_checkpoint_path.exists():
            raise SystemExit(
                f"Probe checkpoint not found: {probe_checkpoint_path}. "
                "Pass --fit-probe to train a fresh probe instead."
            )
        probe = joblib.load(probe_checkpoint_path)
        print(f"Loaded LogisticRegression probe from {probe_checkpoint_path}")
    else:
        probe = get_logistic_regression(seed=seed)
        probe.fit(x_train, y_train)
        print("Fitted fresh LogisticRegression probe")

    val_proba = positive_class_proba(probe, x_val)
    threshold = safe_classification_threshold(y_val, val_proba)
    id_proba = positive_class_proba(probe, x_id)
    heldout_proba = positive_class_proba(probe, x_heldout)
    id_metrics = evaluate(y_id, id_proba, threshold=threshold)
    combined_y = np.concatenate([y_id, y_heldout])
    combined_proba = np.concatenate([id_proba, heldout_proba])
    combined_metrics = evaluate(combined_y, combined_proba, threshold=threshold)

    id_margin = probe.decision_function(x_id)
    heldout_margin = probe.decision_function(x_heldout)
    id_scores = {
        "nearest_train_distance": nearest_distance(
            x_train,
            x_id,
            metric=args.distance_metric,
        ),
        "healthy_centroid_distance": centroid_distance(
            x_train,
            y_train,
            x_id,
            label=0,
            metric=args.distance_metric,
        ),
        "seen_disease_centroid_distance": centroid_distance(
            x_train,
            y_train,
            x_id,
            label=1,
            metric=args.distance_metric,
        ),
        "linear_probe_margin_distance": -np.abs(id_margin),
    }
    heldout_scores = {
        "nearest_train_distance": nearest_distance(
            x_train,
            x_heldout,
            metric=args.distance_metric,
        ),
        "healthy_centroid_distance": centroid_distance(
            x_train,
            y_train,
            x_heldout,
            label=0,
            metric=args.distance_metric,
        ),
        "seen_disease_centroid_distance": centroid_distance(
            x_train,
            y_train,
            x_heldout,
            label=1,
            metric=args.distance_metric,
        ),
        "linear_probe_margin_distance": -np.abs(heldout_margin),
    }

    summary = {
        "heldout_conditions": ",".join(split_info["heldout_conditions"]),
        "backbone": args.backbone,
        "embedding_dim": embedding_dim,
        "checkpoint_id": backbone_meta["checkpoint_id"],
        "fold": int(split_info.get("fold", 0)),
        "n_folds": int(split_info.get("n_folds", 1)),
        "pooling": args.pooling,
        "classifier": "RepresentationProbe",
        "distance_metric": args.distance_metric,
        "threshold": float(threshold),
        "id_auroc": id_metrics["auroc"],
        "id_auprc": id_metrics["auprc"],
        "id_macro_f1": id_metrics["macro_f1"],
        "id_sensitivity": id_metrics["sensitivity"],
        "id_specificity": id_metrics["specificity"],
        "combined_auroc": combined_metrics["auroc"],
        "combined_auprc": combined_metrics["auprc"],
        "combined_macro_f1": combined_metrics["macro_f1"],
        "heldout_positive_rate_at_threshold": float(np.mean(heldout_proba >= threshold)),
        "probe_probability_ood_auroc": float(
            roc_auc_score(
                np.concatenate([np.zeros_like(id_proba), np.ones_like(heldout_proba)]),
                np.concatenate([id_proba, heldout_proba]),
            )
        ),
        "probe_probability_ood_auprc": float(
            average_precision_score(
                np.concatenate([np.zeros_like(id_proba), np.ones_like(heldout_proba)]),
                np.concatenate([id_proba, heldout_proba]),
            )
        ),
        **split_info["counts"],
        "train_subjects_pooled": len(train_subjects),
        "val_subjects_pooled": len(val_subjects),
        "id_test_subjects_pooled": len(id_subjects),
        "heldout_subjects_pooled": len(heldout_subjects),
    }
    for score_name in id_scores:
        summary.update(
            ood_metrics_for_score(
                score_name,
                id_scores[score_name],
                heldout_scores[score_name],
            )
        )

    metadata = {}
    metadata.update(id_metadata)
    metadata.update(heldout_metadata)
    prediction_rows = []
    prediction_rows.extend(
        build_prediction_rows(
            "id_test",
            id_subjects,
            y_id,
            id_proba,
            id_scores,
            metadata,
            threshold,
        )
    )
    prediction_rows.extend(
        build_prediction_rows(
            "heldout_disease",
            heldout_subjects,
            y_heldout,
            heldout_proba,
            heldout_scores,
            metadata,
            threshold,
        )
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / f"{args.output_prefix}_{args.pooling}_summary.csv"
    predictions_path = results_dir / f"{args.output_prefix}_{args.pooling}_predictions.csv"
    split_info_path = results_dir / f"{args.output_prefix}_{args.pooling}_split_info.json"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved summary to {summary_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved split info to {split_info_path}")
    print(
        "Key result: "
        f"ID AUPRC={summary['id_auprc']:.4f}, "
        f"nearest-train OOD AUROC={summary['nearest_train_distance_ood_auroc']:.4f}, "
        f"margin-distance OOD AUROC={summary['linear_probe_margin_distance_ood_auroc']:.4f}"
    )


if __name__ == "__main__":
    main()
