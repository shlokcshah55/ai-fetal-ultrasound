from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aggregation.subject_pooling import pool_subject_features_with_paths
from classifiers.energy import predict_energy, train_energy_mlp
from data.dataset import make_image_transform
from data.disease_holdout import CONDITION_COLS, get_heldout_disease_dataloaders
from evaluate import (
    evaluate,
    evaluate_uncertainty_as_ood,
    expected_calibration_error,
    find_optimal_threshold,
)
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


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def safe_rate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=bool)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def safe_classification_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return find_optimal_threshold(y_true, y_proba)


def fpr_at_tpr(
    id_score: np.ndarray,
    ood_score: np.ndarray,
    *,
    target_tpr: float = 0.95,
) -> float:
    id_score = np.asarray(id_score, dtype=float)
    ood_score = np.asarray(ood_score, dtype=float)
    keep_id = np.isfinite(id_score)
    keep_ood = np.isfinite(ood_score)
    if not np.any(keep_id) or not np.any(keep_ood):
        return float("nan")
    y_true = np.concatenate([
        np.zeros(int(keep_id.sum()), dtype=int),
        np.ones(int(keep_ood.sum()), dtype=int),
    ])
    scores = np.concatenate([id_score[keep_id], ood_score[keep_ood]])
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= target_tpr
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def condition_ood_metrics(
    condition: str,
    id_ood_score: np.ndarray,
    heldout_ood_score: np.ndarray,
    heldout_proba: np.ndarray,
    heldout_metadata: dict[int, dict[str, Any]],
    heldout_subjects: list[int],
    classification_threshold: float,
) -> dict[str, float | int]:
    mask = np.array(
        [bool(heldout_metadata[int(subject_id)][condition]) for subject_id in heldout_subjects],
        dtype=bool,
    )
    condition_scores = np.asarray(heldout_ood_score, dtype=float)[mask]
    condition_proba = np.asarray(heldout_proba, dtype=float)[mask]
    n_subjects = int(mask.sum())

    if n_subjects == 0:
        return {
            f"{condition}_heldout_subjects": 0,
            f"{condition}_mean_ood_score": float("nan"),
            f"{condition}_mean_pred_proba_chd": float("nan"),
            f"{condition}_called_chd_recall": float("nan"),
            f"{condition}_ood_auroc": float("nan"),
            f"{condition}_ood_auprc": float("nan"),
            f"{condition}_fpr_at_95_tpr": float("nan"),
            f"{condition}_low_n": 0,
        }

    y_true = np.concatenate([
        np.zeros_like(id_ood_score, dtype=int),
        np.ones_like(condition_scores, dtype=int),
    ])
    scores = np.concatenate([id_ood_score, condition_scores])
    return {
        f"{condition}_heldout_subjects": n_subjects,
        f"{condition}_mean_ood_score": safe_mean(condition_scores),
        f"{condition}_mean_pred_proba_chd": safe_mean(condition_proba),
        f"{condition}_called_chd_recall": float(np.mean(condition_proba >= classification_threshold)),
        f"{condition}_ood_auroc": float(roc_auc_score(y_true, scores)),
        f"{condition}_ood_auprc": float(average_precision_score(y_true, scores)),
        f"{condition}_fpr_at_95_tpr": fpr_at_tpr(id_ood_score, condition_scores),
        f"{condition}_low_n": int(n_subjects < 30),
    }


def check_fetal_clip_fold_counts(split_info: dict[str, Any]) -> None:
    if int(split_info.get("n_folds", 1)) != 3:
        return
    expected_id_subjects = {0: 1298, 1: 1299, 2: 1300}
    fold = int(split_info.get("fold", 0))
    counts = split_info.get("counts", {})
    expected_id = expected_id_subjects.get(fold)
    if expected_id is None:
        return
    actual_id = int(counts.get("id_test_subjects", -1))
    actual_heldout = int(counts.get("heldout_subjects", -1))
    if actual_id != expected_id or actual_heldout != 223:
        raise SystemExit(
            "Unexpected FETAL-CLIP split counts: "
            f"fold={fold}, id_test_subjects={actual_id} (expected {expected_id}), "
            f"heldout_subjects={actual_heldout} (expected 223)."
        )


def check_fetal_clip_pooled_counts(
    split_info: dict[str, Any],
    *,
    id_subjects_pooled: int,
    heldout_subjects_pooled: int,
) -> None:
    if int(split_info.get("n_folds", 1)) != 3:
        return
    expected_id_subjects = {0: 1298, 1: 1299, 2: 1300}
    fold = int(split_info.get("fold", 0))
    expected_id = expected_id_subjects.get(fold)
    if expected_id is None:
        return
    if id_subjects_pooled != expected_id or heldout_subjects_pooled != 223:
        raise SystemExit(
            "Unexpected FETAL-CLIP pooled subject counts: "
            f"fold={fold}, id_test_subjects_pooled={id_subjects_pooled} "
            f"(expected {expected_id}), heldout_subjects_pooled={heldout_subjects_pooled} "
            "(expected 223)."
        )


def build_prediction_rows(
    split_name: str,
    subject_ids: list[int],
    y_true: np.ndarray,
    y_proba: np.ndarray,
    neg_energy: np.ndarray,
    metadata: dict[int, dict[str, Any]],
    classification_threshold: float,
    energy_threshold: float,
) -> list[dict[str, Any]]:
    y_pred = (y_proba >= classification_threshold).astype(int)
    is_ood = neg_energy < energy_threshold
    ood_score = -neg_energy
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
            "neg_energy": float(neg_energy[idx]),
            "energy": float(-neg_energy[idx]),
            "ood_score": float(ood_score[idx]),
            "energy_threshold": float(energy_threshold),
            "is_ood": int(is_ood[idx]),
        }
        for col in CONDITION_COLS:
            row[col] = int(record[col])
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic energy-based OOD detection with selected fetal "
            "cardiac diseases held out."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subject-labels", default="subject_level_labels.csv")
    parser.add_argument(
        "--heldout-diseases",
        nargs="+",
        default=DEFAULT_HELDOUT_DISEASES,
        help=(
            "Disease names/columns to exclude from training. Defaults to "
            "Tetralogy of Fallot, AVSD, and Aortic Stenosis."
        ),
    )
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument("--backbone", choices=["dinov2", "fetal_clip"], default="dinov2")
    parser.add_argument("--fetal-clip-checkpoint", default=None)
    parser.add_argument("--fetal-clip-config", default=None)
    parser.add_argument(
        "--backbone-normalization",
        choices=["auto", "imagenet", "clip"],
        default="auto",
        help="Frame normalization. auto = ImageNet for DINOv2, CLIP stats for FETAL-CLIP.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--energy-percentile", type=float, default=5.0)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--extract-splits",
        nargs="+",
        choices=["train", "val", "id_test", "heldout"],
        default=["train", "val", "id_test", "heldout"],
    )
    parser.add_argument("--features-cache-dir", default=None)
    parser.add_argument("--sononet-dir", default=None)
    parser.add_argument("--sononet-conf-threshold", type=float, default=None)
    parser.add_argument("--no-sononet", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default="heldout_disease_energy")
    parser.add_argument("--n-folds", type=int, default=1, help="Number of CV folds (1 = single split).")
    parser.add_argument("--fold", type=int, default=0, help="Which fold is the test set (0-indexed).")
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
    if args.backbone == "fetal_clip":
        args.output_prefix = f"{args.output_prefix}_{args.backbone}"
    if args.n_folds > 1:
        # Tag every output (checkpoints + results) with the fold so runs across
        # folds don't overwrite each other.
        args.output_prefix = f"{args.output_prefix}_fold{args.fold}of{args.n_folds}"

    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1).")
    if not 0.0 < args.energy_percentile < 100.0:
        raise SystemExit("--energy-percentile must be in (0, 100).")

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

    device_str = args.device or config["features"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normalization = args.backbone_normalization
    if normalization == "auto":
        normalization = "clip" if args.backbone == "fetal_clip" else "imagenet"
    transform = make_image_transform(normalization)

    print("\n[1/5] Loading disease-held-out data...")
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
    print(f"  Held-out diseases: {', '.join(split_info['heldout_conditions'])}")
    print(f"  Counts: {split_info['counts']}")
    if args.backbone == "fetal_clip":
        check_fetal_clip_fold_counts(split_info)

    cache_dir = sononet_cfg.get("cache_dir") or config["features"]["cache_dir"]
    if args.no_sononet:
        cache_dir = config["features"]["cache_dir"]
    print(f"\nFeature cache dir: {cache_dir}")

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

    print(
        f"\n[2/5] Loading/extracting {args.backbone} features "
        f"(checkpoint_id={backbone_meta['checkpoint_id']})..."
    )
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
        for split_name, loader in split_loaders.items():
            if split_name not in args.extract_splits:
                print(f"  Skipping {split_name}; split not requested.")
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

    print(
        "  Videos - "
        + ", ".join(
            f"{name}: {len(split_features)}"
            for name, split_features in features.items()
        )
    )

    if args.extract_only:
        print("\nExtract-only mode: done after feature caching.")
        return

    print("\n[3/5] Pooling videos to subject embeddings...")
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
    print(
        "  Subjects - "
        f"train: {len(train_subjects)}, val: {len(val_subjects)}, "
        f"id_test: {len(id_subjects)}, heldout: {len(heldout_subjects)}"
    )
    if args.backbone == "fetal_clip":
        check_fetal_clip_pooled_counts(
            split_info,
            id_subjects_pooled=len(id_subjects),
            heldout_subjects_pooled=len(heldout_subjects),
        )
    embedding_dim = int(x_train.shape[1])

    print("\n[4/5] Training energy MLP...")
    Path("checkpoints").mkdir(exist_ok=True)
    energy_config = {
        **config.get("training", {}),
        **config.get("energy", {}),
        "dropout": args.dropout,
        "seed": seed,
    }
    checkpoint_path = (
        Path("checkpoints")
        / f"{args.output_prefix}_{args.pooling}_t{args.temperature:g}_energy_mlp.pt"
    )
    model, _ = train_energy_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        config=energy_config,
        device=device,
        checkpoint_path=str(checkpoint_path),
        input_dim=embedding_dim,
    )

    val_proba, val_neg_energy, _ = predict_energy(
        model,
        x_val,
        device,
        temperature=args.temperature,
    )
    classification_threshold = safe_classification_threshold(y_val, val_proba)
    energy_threshold = float(np.percentile(val_neg_energy, args.energy_percentile))

    print("\n[5/5] Evaluating classification and energy OOD scores...")
    id_proba, id_neg_energy, _ = predict_energy(
        model,
        x_id,
        device,
        temperature=args.temperature,
    )
    heldout_proba, heldout_neg_energy, _ = predict_energy(
        model,
        x_heldout,
        device,
        temperature=args.temperature,
    )

    combined_y = np.concatenate([y_id, y_heldout])
    combined_proba = np.concatenate([id_proba, heldout_proba])

    id_metrics = evaluate(y_id, id_proba, threshold=classification_threshold)
    combined_metrics = evaluate(
        combined_y,
        combined_proba,
        threshold=classification_threshold,
    )
    raw_ood_metrics = evaluate_uncertainty_as_ood(
        -id_neg_energy,
        -heldout_neg_energy,
    )
    ood_metrics = {
        "ood_auroc": raw_ood_metrics["ood_auroc"],
        "ood_auprc": raw_ood_metrics["ood_auprc"],
        "id_ood_score_mean": raw_ood_metrics["id_uncertainty_mean"],
        "heldout_ood_score_mean": raw_ood_metrics["heldout_uncertainty_mean"],
    }
    id_ood_score = -id_neg_energy
    heldout_ood_score = -heldout_neg_energy

    normal_neg_energy = id_neg_energy[y_id == 0]
    seen_disease_neg_energy = id_neg_energy[y_id == 1]
    heldout_pred = (heldout_proba >= classification_threshold).astype(int)
    id_ood_flags = id_neg_energy < energy_threshold
    heldout_ood_flags = heldout_neg_energy < energy_threshold
    id_ece = expected_calibration_error(y_id, id_proba)
    ood_fpr_95 = fpr_at_tpr(id_ood_score, heldout_ood_score)

    summary = {
        "heldout_conditions": ",".join(split_info["heldout_conditions"]),
        "backbone": args.backbone,
        "embedding_dim": embedding_dim,
        "checkpoint_id": backbone_meta["checkpoint_id"],
        "fold": int(split_info.get("fold", 0)),
        "n_folds": int(split_info.get("n_folds", 1)),
        "pooling": args.pooling,
        "classifier": "EnergyMLP",
        "temperature": float(args.temperature),
        "dropout": float(args.dropout),
        "threshold": float(classification_threshold),
        "energy_percentile": float(args.energy_percentile),
        "neg_energy_threshold": energy_threshold,
        "id_auroc": id_metrics["auroc"],
        "id_auprc": id_metrics["auprc"],
        "id_macro_f1": id_metrics["macro_f1"],
        "id_sensitivity": id_metrics["sensitivity"],
        "id_specificity": id_metrics["specificity"],
        "id_ece": float(id_ece),
        "combined_auroc": combined_metrics["auroc"],
        "combined_auprc": combined_metrics["auprc"],
        "combined_macro_f1": combined_metrics["macro_f1"],
        "heldout_positive_rate_at_threshold": float(np.mean(heldout_pred)),
        "normal_neg_energy_mean": safe_mean(normal_neg_energy),
        "seen_disease_neg_energy_mean": safe_mean(seen_disease_neg_energy),
        "heldout_neg_energy_mean": safe_mean(heldout_neg_energy),
        "val_neg_energy_mean": safe_mean(val_neg_energy),
        "normal_energy_mean": safe_mean(-normal_neg_energy),
        "seen_disease_energy_mean": safe_mean(-seen_disease_neg_energy),
        "heldout_energy_mean": safe_mean(-heldout_neg_energy),
        "val_energy_mean": safe_mean(-val_neg_energy),
        "id_ood_rate_at_tau": safe_rate(id_ood_flags),
        "heldout_ood_rate_at_tau": safe_rate(heldout_ood_flags),
        "normal_ood_rate_at_tau": safe_rate(id_ood_flags[y_id == 0]),
        "seen_disease_ood_rate_at_tau": safe_rate(id_ood_flags[y_id == 1]),
        "ood_fpr_at_95_tpr": float(ood_fpr_95),
        **ood_metrics,
        **split_info["counts"],
        "train_subjects_pooled": len(train_subjects),
        "val_subjects_pooled": len(val_subjects),
        "id_test_subjects_pooled": len(id_subjects),
        "heldout_subjects_pooled": len(heldout_subjects),
    }
    for condition in split_info["heldout_conditions"]:
        summary.update(
            condition_ood_metrics(
                condition,
                id_ood_score,
                heldout_ood_score,
                heldout_proba,
                heldout_metadata,
                heldout_subjects,
                classification_threshold,
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
            id_neg_energy,
            metadata,
            classification_threshold,
            energy_threshold,
        )
    )
    prediction_rows.extend(
        build_prediction_rows(
            "heldout_disease",
            heldout_subjects,
            y_heldout,
            heldout_proba,
            heldout_neg_energy,
            metadata,
            classification_threshold,
            energy_threshold,
        )
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / f"{args.output_prefix}_summary.csv"
    predictions_path = results_dir / f"{args.output_prefix}_predictions.csv"
    split_info_path = results_dir / f"{args.output_prefix}_split_info.json"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved split info to {split_info_path}")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(
        "Key result: "
        f"ID AUROC={summary['id_auroc']:.4f}, "
        f"OOD AUROC by energy={summary['ood_auroc']:.4f}, "
        f"heldout neg-energy mean={summary['heldout_neg_energy_mean']:.4f}, "
        f"heldout OOD@tau={summary['heldout_ood_rate_at_tau']:.4f}"
    )


if __name__ == "__main__":
    main()
