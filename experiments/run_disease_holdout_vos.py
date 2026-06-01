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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aggregation.subject_pooling import pool_subject_features_with_paths
from classifiers.vos import predict_vos, train_vos_mlp
from data.disease_holdout import CONDITION_COLS, get_heldout_disease_dataloaders
from evaluate import evaluate, evaluate_uncertainty_as_ood, find_optimal_threshold
from features.extract import extract_features, load_cached_features, load_dinov2
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


def safe_classification_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return find_optimal_threshold(y_true, y_proba)


def build_prediction_rows(
    split_name: str,
    subject_ids: list[int],
    y_true: np.ndarray,
    y_proba: np.ndarray,
    energy: np.ndarray,
    metadata: dict[int, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    y_pred = (y_proba >= threshold).astype(int)
    rows: list[dict[str, Any]] = []

    for subject_id, label, proba, pred, score in zip(
        subject_ids, y_true, y_proba, y_pred, energy
    ):
        record = metadata[int(subject_id)]
        row = {
            "split": split_name,
            "subject_id": int(subject_id),
            "disease_group": record["disease_group"],
            "label": int(label),
            "pred_proba_chd": float(proba),
            "pred_label": int(pred),
            "energy": float(score),
            "ood_score": float(score),
        }
        for col in CONDITION_COLS:
            row[col] = int(record[col])
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VOS with selected fetal cardiac diseases held out as OOD."
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
    parser.add_argument("--output-prefix", default="heldout_disease_vos")
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
    if args.n_folds > 1:
        # Tag every output (checkpoints + results) with the fold so runs across
        # folds don't overwrite each other.
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

    device_str = args.device or config["features"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        )
    )
    print(f"  Held-out diseases: {', '.join(split_info['heldout_conditions'])}")
    print(f"  Counts: {split_info['counts']}")

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

    print("\n[2/5] Loading/extracting DINOv2 features...")
    features: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    if args.skip_extract:
        for split_name, loader in split_loaders.items():
            features[split_name] = load_cached_features(
                _video_paths_from_loader(loader),
                cache_dir,
                split_name=split_name,
            )
    else:
        dinov2 = load_dinov2(device)
        for split_name, loader in split_loaders.items():
            if split_name not in args.extract_splits:
                print(f"  Skipping {split_name}; split not requested.")
                features[split_name] = {}
                continue
            extract_loader = loader
            if args.extract_only:
                extract_loader = _missing_cache_loader(loader, cache_dir, split_name)
            features[split_name] = extract_features(extract_loader, dinov2, device, cache_dir)

    print(
        "  Videos - "
        + ", ".join(f"{name}: {len(split_features)}" for name, split_features in features.items())
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

    print("\n[4/5] Training VOS MLP...")
    Path("checkpoints").mkdir(exist_ok=True)
    vos_config = {
        **config.get("training", {}),
        **config.get("vos", {}),
        "seed": seed,
    }
    checkpoint_path = Path("checkpoints") / f"{args.output_prefix}_{args.pooling}_vos_mlp.pt"
    model, _, gaussian_stats = train_vos_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        config=vos_config,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )
    stats_path = Path("checkpoints") / f"{args.output_prefix}_{args.pooling}_vos_gaussian_stats.npz"
    np.savez(
        stats_path,
        mean_0=gaussian_stats.means[0],
        mean_1=gaussian_stats.means[1],
        covariance=gaussian_stats.covariance,
    )

    val_proba, val_energy, _ = predict_vos(model, x_val, device)
    threshold = safe_classification_threshold(y_val, val_proba)

    print("\n[5/5] Evaluating classification and OOD energy...")
    id_proba, id_energy, _ = predict_vos(model, x_id, device)
    heldout_proba, heldout_energy, _ = predict_vos(model, x_heldout, device)

    combined_y = np.concatenate([y_id, y_heldout])
    combined_proba = np.concatenate([id_proba, heldout_proba])

    id_metrics = evaluate(y_id, id_proba, threshold=threshold)
    combined_metrics = evaluate(combined_y, combined_proba, threshold=threshold)
    ood_metrics = evaluate_uncertainty_as_ood(id_energy, heldout_energy)

    normal_energy = id_energy[y_id == 0]
    seen_disease_energy = id_energy[y_id == 1]
    heldout_pred = (heldout_proba >= threshold).astype(int)

    summary = {
        "heldout_conditions": ",".join(split_info["heldout_conditions"]),
        "fold": int(split_info.get("fold", 0)),
        "n_folds": int(split_info.get("n_folds", 1)),
        "pooling": args.pooling,
        "classifier": "VOSMLP",
        "threshold": float(threshold),
        "id_auroc": id_metrics["auroc"],
        "id_auprc": id_metrics["auprc"],
        "id_macro_f1": id_metrics["macro_f1"],
        "id_sensitivity": id_metrics["sensitivity"],
        "id_specificity": id_metrics["specificity"],
        "combined_auroc": combined_metrics["auroc"],
        "combined_auprc": combined_metrics["auprc"],
        "combined_macro_f1": combined_metrics["macro_f1"],
        "heldout_positive_rate_at_threshold": float(np.mean(heldout_pred)),
        "normal_energy_mean": safe_mean(normal_energy),
        "seen_disease_energy_mean": safe_mean(seen_disease_energy),
        "heldout_energy_mean": safe_mean(heldout_energy),
        "val_energy_mean": safe_mean(val_energy),
        **ood_metrics,
        **split_info["counts"],
        "train_subjects_pooled": len(train_subjects),
        "val_subjects_pooled": len(val_subjects),
        "id_test_subjects_pooled": len(id_subjects),
        "heldout_subjects_pooled": len(heldout_subjects),
    }

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
            id_energy,
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
            heldout_energy,
            metadata,
            threshold,
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
    print(f"Saved Gaussian stats to {stats_path}")
    print(
        "Key result: "
        f"ID AUROC={summary['id_auroc']:.4f}, "
        f"OOD AUROC by energy={summary['ood_auroc']:.4f}, "
        f"heldout energy mean={summary['heldout_energy_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
