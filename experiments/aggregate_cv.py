"""Aggregate per-fold disease-holdout summaries into mean +/- std.

Each experiment script writes one ``*_fold{f}of{K}_summary.csv`` per fold. This
scans the results directory, groups those files by method (stripping the fold
tag), computes the mean and sample std of every numeric metric across folds, and
prints LaTeX-ready rows for the in-distribution classification table in
``ReportsTable.md`` plus an OOD-detection companion table.

Usage:
    python experiments/aggregate_cv.py [--results-dir results] [--k 3]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# The fold tag can appear anywhere in the stem: baseline/energy/vos put it right
# before `_summary` (e.g. ..._fold0of3_summary.csv) while mc_dropout/edl append a
# hyperparam suffix after it (e.g. ..._fold0of3_mean_p0.3_T50_summary.csv).
FOLD_RE = re.compile(r"_fold(?P<fold>\d+)of(?P<k>\d+)")

# Columns that are bookkeeping, not metrics to average.
_EXCLUDE_FROM_STATS = {"fold", "n_folds"}

# Metrics shown in the in-distribution classification table (ReportsTable.md),
# in column order.
ID_METRICS = [
    ("id_auprc", "AUPRC"),
    ("id_auroc", "AUROC"),
    ("id_macro_f1", "F1"),
    ("id_sensitivity", "Sens."),
    ("id_specificity", "Spec."),
]
OOD_METRICS = [
    ("ood_auroc", "OOD AUROC"),
    ("ood_auprc", "OOD AUPRC"),
]

# Map the `classifier` column to the table row label and display order.
METHOD_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "MLP": "MLP",
    "MCDropoutMLP": "MLP + MC Dropout",
    "EnergyMLP": "MLP + Energy",
    "VOSMLP": "MLP + VoS",
    "EvidentialMLP": "MLP + EDL",
}
METHOD_ORDER = list(METHOD_LABELS)


def discover_fold_groups(
    results_dir: Path, k: int | None
) -> dict[tuple[str, int], list[Path]]:
    """Map (method-stem, K) -> list of per-fold summary CSV paths."""
    groups: dict[tuple[str, int], list[Path]] = {}
    for path in sorted(results_dir.glob("*_fold*of*_summary.csv")):
        m = FOLD_RE.search(path.name)
        if not m:
            continue
        kk = int(m.group("k"))
        if k is not None and kk != k:
            continue
        # Group key = filename with the fold tag removed, so all folds of one
        # method collapse to the original single-split stem.
        stem = FOLD_RE.sub("", path.name[: -len("_summary.csv")])
        groups.setdefault((stem, kk), []).append(path)
    return groups


def aggregate_group(paths: list[Path], n_folds: int) -> dict:
    """Mean/std of every numeric column across one method's fold files."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    numeric = df.select_dtypes(include="number")

    stats: dict = {}
    for col in numeric.columns:
        if col in _EXCLUDE_FROM_STATS:
            continue
        stats[f"{col}_mean"] = float(numeric[col].mean())
        stats[f"{col}_std"] = float(numeric[col].std(ddof=1))  # sample std

    # Carry identifying string columns (expected constant across folds).
    for col in df.columns:
        if col in numeric.columns:
            continue
        vals = [str(v) for v in df[col].dropna().unique()]
        stats[col] = vals[0] if len(vals) == 1 else "|".join(vals)

    stats["n_folds"] = n_folds
    stats["n_folds_found"] = len(paths)
    return stats


def fmt(stats: dict, key: str, decimals: int) -> str:
    """Format '<mean> $\\pm$ <std>' for a metric, or a blank cell if missing."""
    mean = stats.get(f"{key}_mean")
    std = stats.get(f"{key}_std")
    if mean is None:
        return "-- $\\pm$ --"
    if std is None or pd.isna(std):
        return f"{mean:.{decimals}f} $\\pm$ --"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def latex_rows(rows_by_method: dict[str, dict], metrics, decimals: int) -> list[str]:
    ordered = [m for m in METHOD_ORDER if m in rows_by_method]
    ordered += [m for m in rows_by_method if m not in METHOD_ORDER]
    lines = []
    for method in ordered:
        stats = rows_by_method[method]
        if all(f"{key}_mean" not in stats for key, _ in metrics):
            continue
        label = METHOD_LABELS.get(method, method)
        cells = " & ".join(fmt(stats, key, decimals) for key, _ in metrics)
        lines.append(f"{label:<20}& {cells} \\\\")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-fold CV summaries into mean +/- std tables."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--k", type=int, default=None, help="Only aggregate groups with this many folds."
    )
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    groups = discover_fold_groups(results_dir, args.k)
    if not groups:
        raise SystemExit(
            f"No '*_fold<f>of<K>_summary.csv' files found in {results_dir}/. "
            "Run the experiments with --n-folds > 1 first."
        )

    rows_by_method: dict[str, dict] = {}
    for (stem, kk), paths in sorted(groups.items()):
        stats = aggregate_group(paths, n_folds=kk)
        out_path = results_dir / f"{stem}_cv_summary.csv"
        pd.DataFrame([stats]).to_csv(out_path, index=False)
        n = stats["n_folds_found"]
        warn = "" if n == kk else f"  [WARNING: only {n}/{kk} folds present]"
        print(
            f"Wrote {out_path}  (classifier={stats.get('classifier', '?')}, "
            f"{n}/{kk} folds){warn}"
        )
        rows_by_method[str(stats.get("classifier", stem))] = stats

    print("\n% --- In-distribution classification (paste into ReportsTable.md) ---")
    for line in latex_rows(rows_by_method, ID_METRICS, args.decimals):
        print(line)

    print("\n% --- OOD detection (uncertainty as OOD score) ---")
    for line in latex_rows(rows_by_method, OOD_METRICS, args.decimals):
        print(line)


if __name__ == "__main__":
    main()
