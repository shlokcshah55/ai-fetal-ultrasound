from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


METHODS = ["LR", "MLP", "MC Dropout", "Energy", "VOS", "EDL"]
ENCODERS = ["DINOv2", "FETAL-CLIP"]


@dataclass(frozen=True)
class MetricSpec:
    method: str
    encoder: str
    source_pattern: str
    expected_classifier: str
    ood_metric: str = "ood_auroc"
    note: str = ""


SPECS = [
    MetricSpec("LR", "DINOv2", "heldout_disease_logreg_cv_summary.csv", "LogisticRegression"),
    MetricSpec("MLP", "DINOv2", "heldout_disease_baseline_cv_summary.csv", "MLP"),
    MetricSpec(
        "MC Dropout",
        "DINOv2",
        "heldout_disease_mc_dropout_mean_p0.3_T100_cv_summary.csv",
        "MCDropoutMLP",
    ),
    MetricSpec("Energy", "DINOv2", "heldout_disease_energy_cv_summary.csv", "EnergyMLP"),
    MetricSpec("VOS", "DINOv2", "heldout_disease_vos_cv_summary.csv", "VOSMLP"),
    MetricSpec(
        "EDL",
        "DINOv2",
        "heldout_disease_edl_mean_softplus_ann25_kl0p10_cv_summary.csv",
        "EvidentialMLP",
    ),
    MetricSpec(
        "LR",
        "FETAL-CLIP",
        "heldout_disease_representation_probe_fetal_clip_fold*of3_mean_summary.csv",
        "RepresentationProbe",
        "linear_probe_margin_distance_ood_auroc",
        "Fetal-clip LR baseline summaries were overwritten by the MLP run; using saved LogisticRegression probe margin OOD AUROC.",
    ),
    MetricSpec("MLP", "FETAL-CLIP", "heldout_disease_baseline_fetal_clip_cv_summary.csv", "MLP"),
    MetricSpec(
        "MC Dropout",
        "FETAL-CLIP",
        "heldout_disease_mc_dropout_fetal_clip_fold*of3_mean_p0.3_T100_summary.csv",
        "MCDropoutMLP",
    ),
    MetricSpec(
        "Energy",
        "FETAL-CLIP",
        "heldout_disease_energy_fetal_clip_fold*of3_summary.csv",
        "EnergyMLP",
        note="Uses the fetal-clip Energy checkpoint/config id saved in those summaries.",
    ),
    MetricSpec("VOS", "FETAL-CLIP", "heldout_disease_vos_fetal_clip_fold*of3_summary.csv", "VOSMLP"),
    MetricSpec(
        "EDL",
        "FETAL-CLIP",
        "heldout_disease_edl_fetal_clip_fold*of3_mean_softplus_ann25_kl0p10_T100_summary.csv",
        "EvidentialMLP",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure 4: OOD AUROC forest plot with fold-std whiskers."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--output-stem", default="ood_auroc_forest")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write the values/provenance CSV; do not import matplotlib or render.",
    )
    return parser.parse_args()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def read_ood_metric(results_dir: Path, spec: MetricSpec) -> dict[str, object]:
    sources = sorted(results_dir.glob(spec.source_pattern))
    if not sources:
        return missing_row(spec, f"NOT YET AVAILABLE: no files matched {spec.source_pattern}")

    source_rows = [(path, read_single_row(path)) for path in sources]
    source_rows = [
        (path, row)
        for path, row in source_rows
        if row.get("classifier") == spec.expected_classifier
    ]
    if not source_rows:
        return missing_row(
            spec,
            f"NOT YET AVAILABLE: no files matching {spec.source_pattern} had "
            f"classifier={spec.expected_classifier!r}.",
        )

    if len(source_rows) == 1 and source_rows[0][0].name.endswith("_cv_summary.csv"):
        row = source_rows[0][1]
        value_mean = float(row[f"{spec.ood_metric}_mean"])
        value_std = float(row[f"{spec.ood_metric}_std"])
        n_folds = int(float(row.get("n_folds_found") or row.get("n_folds") or 0))
    else:
        values = [float(row[spec.ood_metric]) for _, row in source_rows]
        value_mean = mean(values)
        value_std = stdev(values) if len(values) > 1 else 0.0
        n_folds = len(values)

    return {
        "method": spec.method,
        "encoder": spec.encoder,
        "ood_auroc_mean": value_mean,
        "ood_auroc_std": value_std,
        "n_folds": n_folds,
        "ood_metric": spec.ood_metric,
        "sources": ";".join(str(path) for path, _ in source_rows),
        "note": spec.note,
    }


def missing_row(spec: MetricSpec, note: str) -> dict[str, object]:
    return {
        "method": spec.method,
        "encoder": spec.encoder,
        "ood_auroc_mean": math.nan,
        "ood_auroc_std": math.nan,
        "n_folds": 0,
        "ood_metric": spec.ood_metric,
        "sources": "",
        "note": note,
    }


def write_values_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "encoder",
        "ood_auroc_mean",
        "ood_auroc_std",
        "n_folds",
        "ood_metric",
        "sources",
        "note",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "encoder": row["encoder"],
                    "ood_auroc_mean": fmt(float(row["ood_auroc_mean"])),
                    "ood_auroc_std": fmt(float(row["ood_auroc_std"])),
                    "n_folds": row["n_folds"],
                    "ood_metric": row["ood_metric"],
                    "sources": row["sources"],
                    "note": row["note"],
                }
            )


def plot(rows: list[dict[str, object]], output_dir: Path, output_stem: str, formats: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib is required to render the figure. Install project "
            "requirements, then rerun this script."
        ) from exc

    by_key = {(row["method"], row["encoder"]): row for row in rows}
    colors = {"DINOv2": "#2f6f9f", "FETAL-CLIP": "#c0523a"}
    offsets = {"DINOv2": 0.13, "FETAL-CLIP": -0.13}
    y_base = list(range(len(METHODS)))

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)

    for encoder in ENCODERS:
        xs: list[float] = []
        ys: list[float] = []
        xerr: list[float] = []
        for index, method in enumerate(METHODS):
            row = by_key[(method, encoder)]
            value = float(row["ood_auroc_mean"])
            if math.isnan(value):
                continue
            xs.append(value)
            ys.append(index + offsets[encoder])
            xerr.append(float(row["ood_auroc_std"]))

        ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            fmt="o",
            markersize=6,
            color=colors[encoder],
            ecolor=colors[encoder],
            elinewidth=1.4,
            capsize=3,
            label=encoder,
            zorder=3,
        )

    ax.axvline(0.5, color="#475569", linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.text(
        0.505,
        len(METHODS) - 0.35,
        "chance = 0.500",
        ha="left",
        va="center",
        fontsize=8,
        color="#475569",
    )

    ax.set_yticks(y_base)
    ax.set_yticklabels(METHODS)
    ax.invert_yaxis()
    ax.set_xlabel("OOD AUROC (held-out disease vs ID)")
    ax.set_title("OOD AUROC Forest Plot")
    ax.set_xlim(0.2, 0.9)
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt_name in formats:
        path = output_dir / f"{output_stem}.{fmt_name}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]

    rows = [read_ood_metric(results_dir, spec) for spec in SPECS]
    values_path = output_dir / f"{args.output_stem}_values.csv"
    write_values_csv(rows, values_path)
    print(f"Saved {values_path}")

    if args.no_plot:
        return

    plot(rows, output_dir, args.output_stem, formats)


if __name__ == "__main__":
    main()
