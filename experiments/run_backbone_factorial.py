from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_common_args(cmd: list[str], args: argparse.Namespace, backbone: str, fold: int) -> None:
    cmd.extend([
        "--config",
        args.config,
        "--subject-labels",
        args.subject_labels,
        "--pooling",
        args.pooling,
        "--backbone",
        backbone,
        "--n-folds",
        str(args.n_folds),
        "--fold",
        str(fold),
    ])
    if args.skip_extract:
        cmd.append("--skip-extract")
    if args.no_sononet:
        cmd.append("--no-sononet")
    if args.features_cache_dir is not None:
        cmd.extend(["--features-cache-dir", args.features_cache_dir])
    if args.sononet_dir is not None:
        cmd.extend(["--sononet-dir", args.sononet_dir])
    if args.sononet_conf_threshold is not None:
        cmd.extend(["--sononet-conf-threshold", str(args.sononet_conf_threshold)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if backbone == "fetal_clip":
        if args.fetal_clip_checkpoint is None:
            raise SystemExit("--fetal-clip-checkpoint is required when fetal_clip is selected.")
        cmd.extend(["--fetal-clip-checkpoint", args.fetal_clip_checkpoint])
        if args.fetal_clip_config is not None:
            cmd.extend(["--fetal-clip-config", args.fetal_clip_config])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the backbone-factorial disease-holdout experiment matrix."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subject-labels", default="subject_level_labels.csv")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=["dinov2", "fetal_clip"],
        default=["dinov2", "fetal_clip"],
    )
    parser.add_argument("--fetal-clip-checkpoint", default=None)
    parser.add_argument("--fetal-clip-config", default=None)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--features-cache-dir", default=None)
    parser.add_argument("--sononet-dir", default=None)
    parser.add_argument("--sononet-conf-threshold", type=float, default=None)
    parser.add_argument("--no-sononet", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-representation-probe", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "mc_dropout", "edl", "vos"],
        default=["baseline", "mc_dropout", "edl", "vos"],
    )
    parser.add_argument("--mc-passes", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.n_folds < 1:
        raise SystemExit("--n-folds must be >= 1.")
    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    for fold in folds:
        if not 0 <= fold < args.n_folds:
            raise SystemExit(f"fold must be in [0, {args.n_folds}); got {fold}.")

    commands: list[list[str]] = []
    for backbone in args.backbones:
        for fold in folds:
            if not args.skip_representation_probe:
                cmd = [sys.executable, "experiments/run_representation_probe.py"]
                add_common_args(cmd, args, backbone, fold)
                commands.append(cmd)

            if "baseline" in args.methods:
                cmd = [
                    sys.executable,
                    "experiments/run_disease_holdout_baseline.py",
                    "--classifier",
                    "LogisticRegression",
                ]
                add_common_args(cmd, args, backbone, fold)
                commands.append(cmd)

            if "mc_dropout" in args.methods:
                cmd = [
                    sys.executable,
                    "experiments/run_disease_holdout_mc_dropout.py",
                    "--mc-passes",
                    str(args.mc_passes),
                ]
                add_common_args(cmd, args, backbone, fold)
                commands.append(cmd)

            if "edl" in args.methods:
                cmd = [
                    sys.executable,
                    "experiments/run_disease_holdout_edl.py",
                    "--mc-passes",
                    str(args.mc_passes),
                ]
                add_common_args(cmd, args, backbone, fold)
                commands.append(cmd)

            if "vos" in args.methods:
                cmd = [sys.executable, "experiments/run_disease_holdout_vos.py"]
                add_common_args(cmd, args, backbone, fold)
                commands.append(cmd)

    for cmd in commands:
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
