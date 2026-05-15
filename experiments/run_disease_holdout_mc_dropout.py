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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aggregation.subject_pooling import pool_subject_features_with_paths
from classifiers.mlp import MLP, train_mlp
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


def enable_mc_dropout(model: nn.Module) -> None:
    """Use eval mode for deterministic layers, but keep dropout stochastic."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def predict_mc_dropout(
    model: MLP,
    x: np.ndarray,
    device: torch.device,
    *,
    mc_passes: int = 50,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Run stochastic dropout inference on pooled subject embeddings."""
    if mc_passes < 1:
        raise ValueError("mc_passes must be >= 1.")

    x = np.asarray(x, dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    samples: list[np.ndarray] = []
    enable_mc_dropout(model)
    with torch.no_grad():
        for _ in range(mc_passes):
            pass_probs: list[np.ndarray] = []
            for (x_batch,) in loader:
                logits = model(x_batch.to(device))
                pass_probs.append(torch.sigmoid(logits).cpu().numpy())
            samples.append(np.concatenate(pass_probs, axis=0))

    proba_samples = np.stack(samples, axis=0)
    mean_proba = proba_samples.mean(axis=0)
    entropy = predictive_entropy(mean_proba)
    expected_entropy = predictive_entropy(proba_samples).mean(axis=0)
    mutual_info = entropy - expected_entropy
    proba_std = proba_samples.std(axis=0)
    return {
        "proba": mean_proba,
        "entropy": entropy,
        "expected_entropy": expected_entropy,
        "mutual_info": mutual_info,
        "proba_std": proba_std,
        "proba_samples": proba_samples,
    }


def build_prediction_rows(
    split_name: str,
    subject_ids: list[int],
    y_true: np.ndarray,
    mc_outputs: dict[str, np.ndarray],
    metadata: dict[int, dict[str, Any]],
    classification_threshold: float,
    uncertainty_threshold: float,
) -> list[dict[str, Any]]:
    y_proba = mc_outputs["proba"]
    entropy = mc_outputs["entropy"]
    y_pred = (y_proba >= classification_threshold).astype(int)
    is_uncertain = entropy >= uncertainty_threshold

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
            "uncertainty_entropy": float(entropy[idx]),
            "uncertainty_mutual_info": float(mc_outputs["mutual_info"][idx]),
            "uncertainty_expected_entropy": float(mc_outputs["expected_entropy"][idx]),
            "pred_proba_std": float(mc_outputs["proba_std"][idx]),
            "uncertain_at_tau": int(is_uncertain[idx]),
        }
        for col in CONDITION_COLS:
            row[col] = int(record[col])
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MC dropout with selected fetal cardiac diseases held out as OOD."
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
    parser.add_argument("--mc-passes", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--uncertainty-percentile", type=float, default=95.0)
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
    parser.add_argument("--output-prefix", default="heldout_disease_mc_dropout")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    seed = config["training"]["seed"]
    set_seeds(seed)

    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1).")
    if not 0.0 < args.uncertainty_percentile < 100.0:
        raise SystemExit("--uncertainty-percentile must be in (0, 100).")

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

    print("\n[4/5] Training MLP...")
    Path("checkpoints").mkdir(exist_ok=True)
    mlp_config = {
        **config.get("training", {}),
        "dropout": args.dropout,
    }
    checkpoint_path = (
        Path("checkpoints")
        / f"{args.output_prefix}_{args.pooling}_p{args.dropout:g}_mlp.pt"
    )
    model, _ = train_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        config=mlp_config,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )

    print("\n[5/5] Evaluating MC dropout uncertainty...")
    val_mc = predict_mc_dropout(
        model,
        x_val,
        device,
        mc_passes=args.mc_passes,
    )
    threshold = safe_classification_threshold(y_val, val_mc["proba"])
    uncertainty_threshold = float(
        np.percentile(val_mc["entropy"], args.uncertainty_percentile)
    )

    id_mc = predict_mc_dropout(
        model,
        x_id,
        device,
        mc_passes=args.mc_passes,
    )
    heldout_mc = predict_mc_dropout(
        model,
        x_heldout,
        device,
        mc_passes=args.mc_passes,
    )

    combined_y = np.concatenate([y_id, y_heldout])
    combined_proba = np.concatenate([id_mc["proba"], heldout_mc["proba"]])

    id_metrics = evaluate(y_id, id_mc["proba"], threshold=threshold)
    combined_metrics = evaluate(combined_y, combined_proba, threshold=threshold)
    entropy_ood_metrics = evaluate_uncertainty_as_ood(
        id_mc["entropy"],
        heldout_mc["entropy"],
    )
    mi_ood_metrics = {
        f"mutual_info_{key}": value
        for key, value in evaluate_uncertainty_as_ood(
            id_mc["mutual_info"],
            heldout_mc["mutual_info"],
        ).items()
    }
    std_ood_metrics = {
        f"proba_std_{key}": value
        for key, value in evaluate_uncertainty_as_ood(
            id_mc["proba_std"],
            heldout_mc["proba_std"],
        ).items()
    }

    normal_entropy = id_mc["entropy"][y_id == 0]
    seen_disease_entropy = id_mc["entropy"][y_id == 1]
    heldout_pred = (heldout_mc["proba"] >= threshold).astype(int)

    summary = {
        "heldout_conditions": ",".join(split_info["heldout_conditions"]),
        "pooling": args.pooling,
        "classifier": "MCDropoutMLP",
        "dropout": float(args.dropout),
        "mc_passes": int(args.mc_passes),
        "threshold": float(threshold),
        "uncertainty_percentile": float(args.uncertainty_percentile),
        "uncertainty_threshold_entropy": uncertainty_threshold,
        "id_auroc": id_metrics["auroc"],
        "id_auprc": id_metrics["auprc"],
        "id_macro_f1": id_metrics["macro_f1"],
        "id_sensitivity": id_metrics["sensitivity"],
        "id_specificity": id_metrics["specificity"],
        "combined_auroc": combined_metrics["auroc"],
        "combined_auprc": combined_metrics["auprc"],
        "combined_macro_f1": combined_metrics["macro_f1"],
        "heldout_positive_rate_at_threshold": float(np.mean(heldout_pred)),
        "normal_uncertainty_entropy_mean": safe_mean(normal_entropy),
        "seen_disease_uncertainty_entropy_mean": safe_mean(seen_disease_entropy),
        "heldout_uncertainty_entropy_mean": safe_mean(heldout_mc["entropy"]),
        "val_uncertainty_entropy_mean": safe_mean(val_mc["entropy"]),
        "id_uncertain_rate_at_tau": float(np.mean(id_mc["entropy"] >= uncertainty_threshold)),
        "heldout_uncertain_rate_at_tau": float(
            np.mean(heldout_mc["entropy"] >= uncertainty_threshold)
        ),
        **entropy_ood_metrics,
        **mi_ood_metrics,
        **std_ood_metrics,
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
            id_mc,
            metadata,
            threshold,
            uncertainty_threshold,
        )
    )
    prediction_rows.extend(
        build_prediction_rows(
            "heldout_disease",
            heldout_subjects,
            y_heldout,
            heldout_mc,
            metadata,
            threshold,
            uncertainty_threshold,
        )
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    result_stem = f"{args.output_prefix}_{args.pooling}_p{args.dropout:g}_T{args.mc_passes}"
    summary_path = results_dir / f"{result_stem}_summary.csv"
    predictions_path = results_dir / f"{result_stem}_predictions.csv"
    split_info_path = results_dir / f"{result_stem}_split_info.json"

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
        f"OOD AUROC by entropy={summary['ood_auroc']:.4f}, "
        f"heldout entropy mean={summary['heldout_uncertainty_entropy_mean']:.4f}, "
        f"heldout uncertain@tau={summary['heldout_uncertain_rate_at_tau']:.4f}"
    )


if __name__ == "__main__":
    main()
