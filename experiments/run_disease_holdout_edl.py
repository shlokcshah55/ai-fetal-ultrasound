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
from calibration import apply_temperature, binary_logit_from_proba, fit_temperature
from classifiers.edl import predict_edl, predict_edl_mc, train_edl_mlp
from data.disease_holdout import CONDITION_COLS, get_heldout_disease_dataloaders
from evaluate import (
    evaluate,
    evaluate_uncertainty_as_ood,
    expected_calibration_error,
    find_optimal_threshold,
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


def safe_rate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=bool)
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
    edl_outputs: dict[str, np.ndarray],
    metadata: dict[int, dict[str, Any]],
    classification_threshold: float,
    uncertainty_threshold: float,
) -> list[dict[str, Any]]:
    y_proba = edl_outputs["proba"]
    y_proba_cal = edl_outputs["proba_calibrated"]
    uncertainty = edl_outputs["uncertainty"]
    evidence = edl_outputs["evidence"]
    alpha = edl_outputs["alpha"]
    strength = edl_outputs["strength"]
    y_pred = (y_proba >= classification_threshold).astype(int)
    is_uncertain = uncertainty >= uncertainty_threshold

    rows: list[dict[str, Any]] = []
    for idx, subject_id in enumerate(subject_ids):
        record = metadata[int(subject_id)]
        row = {
            "split": split_name,
            "subject_id": int(subject_id),
            "disease_group": record["disease_group"],
            "label": int(y_true[idx]),
            "pred_proba_chd": float(y_proba[idx]),
            "pred_proba_chd_calibrated": float(y_proba_cal[idx]),
            "pred_label": int(y_pred[idx]),
            "uncertainty_dirichlet": float(uncertainty[idx]),
            "uncertain_at_tau": int(is_uncertain[idx]),
            "dirichlet_strength": float(strength[idx]),
            "evidence_healthy": float(evidence[idx, 0]),
            "evidence_chd": float(evidence[idx, 1]),
            "alpha_healthy": float(alpha[idx, 0]),
            "alpha_chd": float(alpha[idx, 1]),
        }
        for col in CONDITION_COLS:
            row[col] = int(record[col])
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run evidential deep learning with selected fetal cardiac diseases "
            "held out as OOD."
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
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--evidence-activation",
        choices=["relu", "softplus"],
        default=None,
    )
    parser.add_argument("--annealing-epochs", type=int, default=None)
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=None,
        help=(
            "Scale on the annealed KL-to-uniform regulariser. Effective coef = "
            "kl_weight * min(1, epoch/annealing_epochs). Use 0 for the no-KL "
            "ablation. Defaults to config.yaml edl.kl_weight."
        ),
    )
    parser.add_argument(
        "--no-class-weighting",
        action="store_true",
        help="Disable balanced class weighting in the EDL loss.",
    )
    parser.add_argument("--uncertainty-percentile", type=float, default=95.0)
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=1,
        help=(
            "MC-dropout passes for the sampling ablation. 1 = single deterministic "
            "EDL pass (default). >1 enables stochastic dropout sampling over the "
            "evidential model and reports entropy/MI/proba-std OOD signals."
        ),
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
    parser.add_argument("--output-prefix", default="heldout_disease_edl")
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

    edl_base_config = config.get("edl", {})
    dropout = (
        float(args.dropout)
        if args.dropout is not None
        else float(edl_base_config.get("dropout", 0.3))
    )
    evidence_activation = args.evidence_activation or str(
        edl_base_config.get("evidence_activation", "softplus")
    )
    annealing_epochs = (
        int(args.annealing_epochs)
        if args.annealing_epochs is not None
        else int(edl_base_config.get("annealing_epochs", 25))
    )
    kl_weight = (
        float(args.kl_weight)
        if args.kl_weight is not None
        else float(edl_base_config.get("kl_weight", 0.1))
    )
    class_weighting = bool(edl_base_config.get("class_weighting", True))
    if args.no_class_weighting:
        class_weighting = False

    if not 0.0 <= dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1).")
    if annealing_epochs < 1:
        raise SystemExit("--annealing-epochs must be >= 1.")
    if kl_weight < 0.0:
        raise SystemExit("--kl-weight must be >= 0.")
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
            features[split_name] = extract_features(
                extract_loader,
                dinov2,
                device,
                cache_dir,
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

    print("\n[4/5] Training evidential MLP...")
    Path("checkpoints").mkdir(exist_ok=True)
    edl_config = {
        **config.get("training", {}),
        **config.get("edl", {}),
        "dropout": dropout,
        "evidence_activation": evidence_activation,
        "annealing_epochs": annealing_epochs,
        "kl_weight": kl_weight,
        "class_weighting": class_weighting,
        "seed": seed,
    }
    # Tag the KL weight into the run name (kl0p10, kl0p00, ...) so the no-KL
    # ablation and the tuned run don't overwrite each other's artifacts.
    kl_tag = f"kl{kl_weight:.2f}".replace(".", "p")
    checkpoint_path = (
        Path("checkpoints")
        / (
            f"{args.output_prefix}_{args.pooling}_{evidence_activation}"
            f"_ann{annealing_epochs}_{kl_tag}_edl_mlp.pt"
        )
    )
    model, _ = train_edl_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        config=edl_config,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )

    print("\n[5/5] Evaluating EDL uncertainty...")
    if args.mc_passes < 1:
        raise SystemExit("--mc-passes must be >= 1.")
    if args.mc_passes > 1:
        print(f"  MC-dropout over EDL with {args.mc_passes} passes.")

        def _predict(x: np.ndarray) -> dict:
            return predict_edl_mc(model, x, device, mc_passes=args.mc_passes)
    else:
        def _predict(x: np.ndarray) -> dict:
            return predict_edl(model, x, device)

    val_edl = _predict(x_val)
    threshold = safe_classification_threshold(y_val, val_edl["proba"])
    uncertainty_threshold = float(
        np.percentile(val_edl["uncertainty"], args.uncertainty_percentile)
    )

    id_edl = _predict(x_id)
    heldout_edl = _predict(x_heldout)

    # Post-hoc temperature scaling (Guo et al. 2017). Fit a single scalar on the
    # in-distribution VAL split only (never on the held-out OOD set), acting on
    # the effective binary logit log(p/(1-p)) implied by the Dirichlet mean. This
    # calibrates the predictive probability (ECE) without touching the evidential
    # vacuity u = K/S that drives OOD detection. T is monotonic in the score, so
    # AUROC/AUPRC are unchanged; only ECE/threshold-dependent metrics move.
    temperature = fit_temperature(
        binary_logit_from_proba(val_edl["proba"]),
        y_val,
    )
    id_proba_cal = apply_temperature(
        binary_logit_from_proba(id_edl["proba"]), temperature
    )
    heldout_proba_cal = apply_temperature(
        binary_logit_from_proba(heldout_edl["proba"]), temperature
    )

    combined_y = np.concatenate([y_id, y_heldout])
    combined_proba = np.concatenate([id_edl["proba"], heldout_edl["proba"]])

    id_edl["proba_calibrated"] = id_proba_cal
    heldout_edl["proba_calibrated"] = heldout_proba_cal

    combined_proba_cal = np.concatenate([id_proba_cal, heldout_proba_cal])
    id_ece_uncalibrated = expected_calibration_error(y_id, id_edl["proba"])
    id_ece_calibrated = expected_calibration_error(y_id, id_proba_cal)
    combined_ece_uncalibrated = expected_calibration_error(combined_y, combined_proba)
    combined_ece_calibrated = expected_calibration_error(combined_y, combined_proba_cal)

    id_metrics = evaluate(y_id, id_edl["proba"], threshold=threshold)
    combined_metrics = evaluate(combined_y, combined_proba, threshold=threshold)
    uncertainty_ood_metrics = evaluate_uncertainty_as_ood(
        id_edl["uncertainty"],
        heldout_edl["uncertainty"],
    )

    # With MC sampling, also score the sample-derived signals as OOD detectors
    # (these, unlike the analytic vacuity, are what the sample count T affects).
    sampling_ood_metrics: dict[str, float] = {}
    if args.mc_passes > 1:
        for prefix, key in (("entropy", "entropy"), ("mutual_info", "mutual_info"), ("proba_std", "proba_std")):
            scored = evaluate_uncertainty_as_ood(id_edl[key], heldout_edl[key])
            sampling_ood_metrics.update({f"{prefix}_{k}": v for k, v in scored.items()})

    normal_uncertainty = id_edl["uncertainty"][y_id == 0]
    seen_disease_uncertainty = id_edl["uncertainty"][y_id == 1]
    heldout_pred = (heldout_edl["proba"] >= threshold).astype(int)
    id_uncertain_flags = id_edl["uncertainty"] >= uncertainty_threshold
    heldout_uncertain_flags = heldout_edl["uncertainty"] >= uncertainty_threshold

    summary = {
        "heldout_conditions": ",".join(split_info["heldout_conditions"]),
        "fold": int(split_info.get("fold", 0)),
        "n_folds": int(split_info.get("n_folds", 1)),
        "pooling": args.pooling,
        "classifier": "EvidentialMLP",
        "mc_passes": int(args.mc_passes),
        "dropout": float(dropout),
        "evidence_activation": evidence_activation,
        "annealing_epochs": int(annealing_epochs),
        "kl_weight": float(kl_weight),
        "class_weighting": bool(edl_config["class_weighting"]),
        "threshold": float(threshold),
        "temperature": float(temperature),
        "id_ece_uncalibrated": float(id_ece_uncalibrated),
        "id_ece_calibrated": float(id_ece_calibrated),
        "combined_ece_uncalibrated": float(combined_ece_uncalibrated),
        "combined_ece_calibrated": float(combined_ece_calibrated),
        "uncertainty_percentile": float(args.uncertainty_percentile),
        "uncertainty_threshold_dirichlet": uncertainty_threshold,
        "id_auroc": id_metrics["auroc"],
        "id_auprc": id_metrics["auprc"],
        "id_macro_f1": id_metrics["macro_f1"],
        "id_sensitivity": id_metrics["sensitivity"],
        "id_specificity": id_metrics["specificity"],
        "combined_auroc": combined_metrics["auroc"],
        "combined_auprc": combined_metrics["auprc"],
        "combined_macro_f1": combined_metrics["macro_f1"],
        "heldout_positive_rate_at_threshold": float(np.mean(heldout_pred)),
        "normal_uncertainty_dirichlet_mean": safe_mean(normal_uncertainty),
        "seen_disease_uncertainty_dirichlet_mean": safe_mean(
            seen_disease_uncertainty
        ),
        "heldout_uncertainty_dirichlet_mean": safe_mean(heldout_edl["uncertainty"]),
        "val_uncertainty_dirichlet_mean": safe_mean(val_edl["uncertainty"]),
        "normal_strength_mean": safe_mean(id_edl["strength"][y_id == 0]),
        "seen_disease_strength_mean": safe_mean(id_edl["strength"][y_id == 1]),
        "heldout_strength_mean": safe_mean(heldout_edl["strength"]),
        "val_strength_mean": safe_mean(val_edl["strength"]),
        "id_uncertain_rate_at_tau": safe_rate(id_uncertain_flags),
        "heldout_uncertain_rate_at_tau": safe_rate(heldout_uncertain_flags),
        "normal_uncertain_rate_at_tau": safe_rate(id_uncertain_flags[y_id == 0]),
        "seen_disease_uncertain_rate_at_tau": safe_rate(
            id_uncertain_flags[y_id == 1]
        ),
        **uncertainty_ood_metrics,
        **sampling_ood_metrics,
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
            id_edl,
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
            heldout_edl,
            metadata,
            threshold,
            uncertainty_threshold,
        )
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    mc_tag = f"_T{args.mc_passes}" if args.mc_passes > 1 else ""
    result_stem = (
        f"{args.output_prefix}_{args.pooling}_{evidence_activation}"
        f"_ann{annealing_epochs}_{kl_tag}{mc_tag}"
    )
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
        f"OOD AUROC by EDL uncertainty={summary['ood_auroc']:.4f}, "
        f"ID ECE {summary['id_ece_uncalibrated']:.4f}->"
        f"{summary['id_ece_calibrated']:.4f} (T={summary['temperature']:.3f}), "
        f"heldout uncertainty mean="
        f"{summary['heldout_uncertainty_dirichlet_mean']:.4f}, "
        f"heldout uncertain@tau={summary['heldout_uncertain_rate_at_tau']:.4f}"
    )


if __name__ == "__main__":
    main()
