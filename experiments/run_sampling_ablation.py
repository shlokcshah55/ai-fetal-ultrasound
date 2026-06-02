"""Sampling-count ablation: run MC Dropout and EDL at several MC sample counts.

For each T in --mc-passes-list, runs both methods with T stochastic dropout
samples on every fold, then aggregates with aggregate_cv. MC Dropout uses its
existing ``--mc-passes``; EDL uses the MC-dropout-over-evidence path (``--mc-passes``
> 1, added in classifiers.edl.predict_edl_mc). T is encoded in each output stem
(``..._T{T}_...``), so aggregate_cv treats every (method, T) as its own group and
reports mean +/- std over folds.

The sample count is an inference-time knob, so each (method, fold) retrains at a
fixed seed (identical model across T) and is then evaluated at the requested T.

Example:
    # MC Dropout + EDL at T=10,100,1000 over 3 folds, reusing cached features:
    python experiments/run_sampling_ablation.py --n-folds 3 -- --skip-extract
Everything after `--` is forwarded verbatim to each experiment script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS: dict[str, str] = {
    "mc_dropout": "run_disease_holdout_mc_dropout.py",
    "edl": "run_disease_holdout_edl.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MC sample-count ablation for MC Dropout and EDL."
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(SCRIPTS), default=list(SCRIPTS)
    )
    parser.add_argument(
        "--mc-passes-list", type=int, nargs="+", default=[10, 100, 1000]
    )
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Subset of fold indices to run (default: all 0..n_folds-1).",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Args after `--` forwarded to each experiment script.",
    )
    args = parser.parse_args()

    if args.n_folds < 1:
        raise SystemExit("--n-folds must be >= 1.")
    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    for f in folds:
        if not 0 <= f < args.n_folds:
            raise SystemExit(f"fold {f} out of range [0, {args.n_folds}).")
    if any(t < 2 for t in args.mc_passes_list):
        raise SystemExit("--mc-passes-list values must be >= 2 (1 = no sampling).")

    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]

    exp_dir = Path(__file__).resolve().parent
    failures: list[str] = []
    for passes in args.mc_passes_list:
        for method in args.methods:
            for fold in folds:
                cmd = [
                    args.python,
                    str(exp_dir / SCRIPTS[method]),
                    "--n-folds", str(args.n_folds),
                    "--fold", str(fold),
                    "--mc-passes", str(passes),
                    *extra,
                ]
                print(f"\n=== {method} | T={passes} | fold {fold} of {args.n_folds} ===")
                print("  " + " ".join(cmd))
                if args.dry_run:
                    continue
                result = subprocess.run(cmd, cwd=str(REPO_ROOT))
                if result.returncode != 0:
                    failures.append(f"{method} T={passes} fold {fold} (exit {result.returncode})")
                    print(f"  !! FAILED: {failures[-1]}")

    if failures:
        print("\nFailures:\n  - " + "\n  - ".join(failures))

    if not args.no_aggregate and not args.dry_run:
        print("\n=== Aggregating (mean +/- std over folds, per method x T) ===")
        subprocess.run(
            [args.python, str(exp_dir / "aggregate_cv.py"), "--k", str(args.n_folds)],
            cwd=str(REPO_ROOT),
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
