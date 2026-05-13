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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aggregation.pooling import max_pool, mean_pool
from classifiers.knn import get_knn
from classifiers.linear_probe import get_linear_svc, get_logistic_regression
from classifiers.mlp import train_mlp
from data.disease_holdout import CONDITION_COLS, get_heldout_disease_dataloaders
from evaluate import (
    evaluate,
    evaluate_uncertainty_as_ood,
    find_optimal_threshold,
    predictive_entropy,
)
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


def pool_features_with_paths(
    feature_dict: dict[str, tuple[np.ndarray, int]],
    pooling: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    paths: list[str] = []
    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for video_path, (features, label) in feature_dict.items():
        if pooling == "mean":
            pooled = mean_pool(features)
        elif pooling == "max":
            pooled = max_pool(features)
        else:
            raise ValueError(f"Unsupported pooling for this baseline: {pooling}")

        paths.append(video_path)
        x_list.append(pooled)
        y_list.append(label)

    return paths, np.stack(x_list), np.array(y_list)


def fit_classifier(
    classifier_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    checkpoint_path: Path,
) -> tuple[Any, np.ndarray]:
    if classifier_name == "LogisticRegression":
        clf = get_logistic_regression(seed=seed)
        clf.fit(x_train, y_train)
        val_proba = clf.predict_proba(x_val)[:, 1]
        joblib.dump(clf, checkpoint_path)
        return clf, val_proba

    if classifier_name == "LinearSVC":
        clf = get_linear_svc(seed=seed)
        clf.fit(x_train, y_train)
        val_proba = clf.predict_proba(x_val)[:, 1]
        joblib.dump(clf, checkpoint_path)
        return clf, val_proba

    if classifier_name == "kNN":
        clf = get_knn()
        clf.fit(x_train, y_train)
        val_proba = clf.predict_proba(x_val)[:, 1]
        joblib.dump(clf, checkpoint_path)
        return clf, val_proba

    if classifier_name == "MLP":
        mlp_config = {
            "mlp_epochs": config["training"]["mlp_epochs"],
            "mlp_lr": config["training"]["mlp_lr"],
            "mlp_weight_decay": config["training"]["mlp_weight_decay"],
            "early_stopping_patience": config["training"]["early_stopping_patience"],
        }
        model, _ = train_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            config=mlp_config,
            device=device,
            checkpoint_path=str(checkpoint_path),
        )
        model.eval()
        with torch.no_grad():
            x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)
            val_proba = torch.sigmoid(model(x_val_t)).cpu().numpy()
        return model, val_proba

    raise ValueError(f"Unknown classifier: {classifier_name}")


def predict_classifier(
    model: Any,
    classifier_name: str,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if classifier_name == "MLP":
        model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).to(device)
            return torch.sigmoid(model(x_t)).cpu().numpy()
    return model.predict_proba(x)[:, 1]


def metadata_by_path(loader: torch.utils.data.DataLoader) -> dict[str, dict[str, Any]]:
    records = getattr(loader.dataset, "records", [])
    return {record["video_path"]: record for record in records}


def build_prediction_rows(
    split_name: str,
    paths: list[str],
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metadata: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    uncertainty = predictive_entropy(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    rows: list[dict[str, Any]] = []
    for path, label, proba, pred, unc in zip(paths, y_true, y_proba, y_pred, uncertainty):
        record = metadata[path]
        row = {
            "split": split_name,
            "video_path": path,
            "subject_id": record["subject_id"],
            "disease_group": record["disease_group"],
            "label": int(label),
            "pred_proba_chd": float(proba),
            "pred_label": int(pred),
            "uncertainty_entropy": float(unc),
        }
        for col in CONDITION_COLS:
            row[col] = int(record[col])
        rows.append(row)

    return rows


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a baseline CHD classifier with selected diseases held out from training."
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
    parser.add_argument(
        "--classifier",
        choices=["LogisticRegression", "LinearSVC", "MLP", "kNN"],
        default="LogisticRegression",
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
    parser.add_argument("--sononet-dir", default=None)
    parser.add_argument("--sononet-conf-threshold", type=float, default=None)
    parser.add_argument("--no-sononet", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default="heldout_disease_baseline")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    seed = config["training"]["seed"]
    set_seeds(seed)

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

    print("\n[3/5] Pooling features...")
    _, x_train, y_train = pool_features_with_paths(features["train"], args.pooling)
    _, x_val, y_val = pool_features_with_paths(features["val"], args.pooling)
    id_paths, x_id, y_id = pool_features_with_paths(features["id_test"], args.pooling)
    heldout_paths, x_heldout, y_heldout = pool_features_with_paths(features["heldout"], args.pooling)

    print("\n[4/5] Training baseline classifier...")
    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = (
        Path("checkpoints")
        / f"{args.output_prefix}_{args.pooling}_{args.classifier}.joblib"
    )
    if args.classifier == "MLP":
        checkpoint_path = checkpoint_path.with_suffix(".pt")

    model, val_proba = fit_classifier(
        args.classifier,
        x_train,
        y_train,
        x_val,
        y_val,
        config,
        device,
        seed,
        checkpoint_path,
    )
    threshold = find_optimal_threshold(y_val, val_proba)

    print("\n[5/5] Evaluating classification and uncertainty...")
    id_proba = predict_classifier(model, args.classifier, x_id, device)
    heldout_proba = predict_classifier(model, args.classifier, x_heldout, device)
    combined_y = np.concatenate([y_id, y_heldout])
    combined_proba = np.concatenate([id_proba, heldout_proba])

    id_metrics = evaluate(y_id, id_proba, threshold=threshold)
    combined_metrics = evaluate(combined_y, combined_proba, threshold=threshold)

    id_uncertainty = predictive_entropy(id_proba)
    heldout_uncertainty = predictive_entropy(heldout_proba)
    ood_metrics = evaluate_uncertainty_as_ood(id_uncertainty, heldout_uncertainty)

    normal_uncertainty = id_uncertainty[y_id == 0]
    seen_disease_uncertainty = id_uncertainty[y_id == 1]
    heldout_pred = (heldout_proba >= threshold).astype(int)

    summary = {
        "heldout_conditions": ",".join(split_info["heldout_conditions"]),
        "pooling": args.pooling,
        "classifier": args.classifier,
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
        "normal_uncertainty_mean": safe_mean(normal_uncertainty),
        "seen_disease_uncertainty_mean": safe_mean(seen_disease_uncertainty),
        "heldout_uncertainty_mean": safe_mean(heldout_uncertainty),
        **ood_metrics,
        **split_info["counts"],
    }

    metadata = {}
    for loader in [id_test_loader, heldout_loader]:
        metadata.update(metadata_by_path(loader))

    prediction_rows = []
    prediction_rows.extend(
        build_prediction_rows("id_test", id_paths, y_id, id_proba, metadata, threshold)
    )
    prediction_rows.extend(
        build_prediction_rows(
            "heldout_disease",
            heldout_paths,
            y_heldout,
            heldout_proba,
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
    print(
        "Key result: "
        f"ID AUROC={summary['id_auroc']:.4f}, "
        f"OOD AUROC by uncertainty={summary['ood_auroc']:.4f}, "
        f"heldout uncertainty mean={summary['heldout_uncertainty_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
