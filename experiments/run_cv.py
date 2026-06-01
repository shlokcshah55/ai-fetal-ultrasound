"""Run the disease-holdout experiments across K folds, then aggregate.

Convenience orchestrator for local / sequential runs. On a cluster you may
prefer to submit each (method, fold) as its own job; this script simply loops
and shells out to the same experiment entry points, then calls aggregate_cv.

Examples:
    # All 6 methods, 3 folds, reusing the cached features:
    python experiments/run_cv.py --n-folds 3 -- --skip-extract

    # Just MLP + MC dropout, dry run to see the commands:
    python experiments/run_cv.py --methods mlp mc_dropout --dry-run

Everything after the literal `--` is forwarded verbatim to each experiment
script (e.g. --skip-extract, --device, --features-cache-dir).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# method -> (script filename, method-specific extra args). The two baseline
# variants share a script but use distinct output prefixes so LR and MLP land
# in separate result files.
METHODS: dict[str, tuple[str, list[str]]] = {
    "logreg": (
        "run_disease_holdout_baseline.py",
        ["--classifier", "LogisticRegression", "--output-prefix", "heldout_disease_logreg"],
    ),
    "mlp": (
        "run_disease_holdout_baseline.py",
        ["--classifier", "MLP", "--output-prefix", "heldout_disease_baseline"],
    ),
    "mc_dropout": ("run_disease_holdout_mc_dropout.py", []),
    "energy": ("run_disease_holdout_energy.py", []),
    "vos": ("run_disease_holdout_vos.py", []),
    "edl": ("run_disease_holdout_edl.py", []),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run disease-holdout experiments across folds and aggregate."
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
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

    if args.n_folds < 2:
        raise SystemExit("--n-folds must be >= 2 for cross-validation.")
    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    for f in folds:
        if not 0 <= f < args.n_folds:
            raise SystemExit(f"fold {f} out of range [0, {args.n_folds}).")

    # argparse.REMAINDER keeps a leading '--'; drop it.
    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]

    exp_dir = Path(__file__).resolve().parent
    failures: list[str] = []
    for method in args.methods:
        script, method_args = METHODS[method]
        for fold in folds:
            cmd = [
                args.python,
                str(exp_dir / script),
                "--n-folds",
                str(args.n_folds),
                "--fold",
                str(fold),
                *method_args,
                *extra,
            ]
            print(f"\n=== {method} | fold {fold} of {args.n_folds} ===")
            print("  " + " ".join(cmd))
            if args.dry_run:
                continue
            result = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if result.returncode != 0:
                failures.append(f"{method} fold {fold} (exit {result.returncode})")
                print(f"  !! FAILED: {failures[-1]}")

    if failures:
        print("\nFailures:\n  - " + "\n  - ".join(failures))

    if not args.no_aggregate and not args.dry_run:
        print("\n=== Aggregating ===")
        subprocess.run(
            [args.python, str(exp_dir / "aggregate_cv.py"), "--k", str(args.n_folds)],
            cwd=str(REPO_ROOT),
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
