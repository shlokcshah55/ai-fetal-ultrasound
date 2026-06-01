"""Per-disease group-wise analysis of disease-holdout predictions.

For each method's ``*_predictions.csv`` this breaks the results down by the 9
individual disease columns and reports, per disease:

  OOD-detectability
    - mean_ood_score        : mean uncertainty / OOD score for that disease
    - ood_auroc_vs_normal   : AUROC of (disease vs healthy/normal) by that score
    - detection_rate_at_tau : fraction flagged at the normal-{pct}th-percentile tau
  Classification
    - recall                : fraction of that disease's subjects predicted CHD
    - mean_proba_chd        : mean predicted P(CHD)

Metrics are computed *within each CV fold* and aggregated to mean +/- std, so it
works on the single-split predictions today (std is NaN with one fold) and on the
3-fold predictions once they land. The 3 held-out diseases (the OOD probe) and the
seen diseases are both broken out and flagged.

Usage:
    python experiments/groupwise_per_disease.py [--results-dir results] [--tau-percentile 95]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CONDITION_COLS = [
    "avsd", "hlhs", "tga", "tetralogy", "raa",
    "coa", "p_atresia", "a_stenosis", "p_stenosis",
]
FOLD_RE = re.compile(r"_fold(?P<fold>\d+)of(?P<k>\d+)")

# Uncertainty/OOD score column to use, in priority order. All are oriented so a
# HIGHER value means more uncertain / more OOD, matching evaluate_uncertainty_as_ood.
SCORE_CANDIDATES = ["ood_score", "uncertainty_dirichlet", "uncertainty_entropy"]

METRIC_KEYS = [
    "ood_auroc_vs_normal",
    "detection_rate_at_tau",
    "recall",
    "mean_ood_score",
    "mean_proba_chd",
]


def resolve_score_col(columns) -> str | None:
    for c in SCORE_CANDIDATES:
        if c in columns:
            return c
    return None


def discover_prediction_groups(results_dir: Path) -> dict[tuple[str, int], list[Path]]:
    """Map (method-stem, K) -> prediction CSVs (K=1 for single-split files)."""
    groups: dict[tuple[str, int], list[Path]] = {}
    for path in sorted(results_dir.glob("*_predictions.csv")):
        m = FOLD_RE.search(path.name)
        if m:
            k = int(m.group("k"))
            stem = FOLD_RE.sub("", path.name[: -len("_predictions.csv")])
        else:
            k = 1
            stem = path.name[: -len("_predictions.csv")]
        groups.setdefault((stem, k), []).append(path)
    return groups


def disease_kinds(pooled: pd.DataFrame) -> dict[str, str]:
    """Label each present disease 'held-out' (only in the OOD set) or 'seen'."""
    kinds: dict[str, str] = {}
    for d in CONDITION_COLS:
        if d not in pooled.columns:
            continue
        has_d = pooled[pooled[d] == 1]
        if len(has_d) == 0:
            continue
        in_id = int((has_d["disease_group"] != "heldout_disease").sum())
        kinds[d] = "held-out" if in_id == 0 else "seen"
    return kinds


def per_fold_disease_metrics(
    df: pd.DataFrame, score_col: str, tau_pct: float
) -> dict[str, dict]:
    normal_scores = df.loc[df["disease_group"] == "normal", score_col].to_numpy(float)
    tau = float(np.percentile(normal_scores, tau_pct)) if len(normal_scores) else np.nan

    out: dict[str, dict] = {}
    for d in CONDITION_COLS:
        if d not in df.columns:
            continue
        sub = df[df[d] == 1]
        n = len(sub)
        if n == 0:
            continue
        dscores = sub[score_col].to_numpy(float)

        if len(normal_scores):
            y = np.r_[np.zeros(len(normal_scores)), np.ones(n)]
            s = np.r_[normal_scores, dscores]
            auroc = float(roc_auc_score(y, s))
        else:
            auroc = np.nan

        out[d] = {
            "n": n,
            "ood_auroc_vs_normal": auroc,
            "detection_rate_at_tau": float((dscores >= tau).mean()) if not np.isnan(tau) else np.nan,
            "recall": float((sub["pred_label"] == 1).mean()) if "pred_label" in sub else np.nan,
            "mean_ood_score": float(dscores.mean()),
            "mean_proba_chd": float(sub["pred_proba_chd"].mean()) if "pred_proba_chd" in sub else np.nan,
        }
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return (np.nan, np.nan)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-disease group-wise analysis of disease-holdout predictions."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tau-percentile", type=float, default=95.0)
    parser.add_argument("--out", default=None, help="Output CSV (default results/groupwise_per_disease.csv).")
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    groups = discover_prediction_groups(results_dir)
    if not groups:
        raise SystemExit(f"No '*_predictions.csv' found in {results_dir}/.")

    rows: list[dict] = []
    for (stem, k), paths in sorted(groups.items()):
        frames = [pd.read_csv(p) for p in paths]
        score_col = resolve_score_col(frames[0].columns)
        if score_col is None:
            print(f"[skip] {stem}: no usable uncertainty/OOD score column.")
            continue
        pooled = pd.concat(frames, ignore_index=True)
        unit = "video" if pooled["subject_id"].duplicated().any() else "subject"
        kinds = disease_kinds(pooled)
        per_fold = [per_fold_disease_metrics(df, score_col, args.tau_percentile) for df in frames]
        # Most-prevalent first.
        diseases = sorted(kinds, key=lambda d: -sum(pf.get(d, {}).get("n", 0) for pf in per_fold))

        for d in diseases:
            n_vals = [pf[d]["n"] for pf in per_fold if d in pf]
            row = {
                "method": stem.replace("heldout_disease_", ""),
                "unit": unit,
                "score_col": score_col,
                "disease": d,
                "kind": kinds[d],
                "n_folds": len(paths),
                "n_per_fold": float(np.mean(n_vals)) if n_vals else 0.0,
            }
            for mk in METRIC_KEYS:
                m, s = mean_std([pf[d][mk] for pf in per_fold if d in pf])
                row[f"{mk}_mean"] = m
                row[f"{mk}_std"] = s
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else results_dir / "groupwise_per_disease.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out_df)} disease rows, {out_df['method'].nunique()} methods)\n")

    dd = args.decimals

    def cell(r: pd.Series, mk: str) -> str:
        m, s = r[f"{mk}_mean"], r[f"{mk}_std"]
        if pd.isna(m):
            return "--"
        return f"{m:.{dd}f}" + ("" if pd.isna(s) else f"±{s:.{dd}f}")

    for method in out_df["method"].unique():
        sub = out_df[out_df["method"] == method]
        print(f"### {method}  (score={sub['score_col'].iloc[0]}, unit={sub['unit'].iloc[0]})")
        print(f"  {'disease':<11}{'kind':<9}{'n/fold':>7}   "
              f"{'OOD-AUROC':>13}{'det@tau':>13}{'recall':>13}{'mean-score':>13}")
        for _, r in sub.iterrows():
            print(f"  {r['disease']:<11}{r['kind']:<9}{r['n_per_fold']:>7.0f}   "
                  f"{cell(r, 'ood_auroc_vs_normal'):>13}{cell(r, 'detection_rate_at_tau'):>13}"
                  f"{cell(r, 'recall'):>13}{cell(r, 'mean_ood_score'):>13}")
        print()


if __name__ == "__main__":
    main()
