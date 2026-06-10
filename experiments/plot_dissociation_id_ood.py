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
    id_metric: str = "id_auprc"
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
        ood_metric="linear_probe_margin_distance_ood_auroc",
        note=(
            "Fetal-clip LR baseline summaries were overwritten by the MLP run; "
            "using the saved LogisticRegression representation-probe ID AUPRC "
            "and linear-probe margin OOD AUROC."
        ),
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
        description=(
            "Create Figure 3: two-panel ID-vs-OOD dissociation chart. "
            "Left panel is ID AUPRC; right panel is OOD AUROC."
        )
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--output-stem", default="dissociation_id_ood")
    parser.add_argument(
        "--prevalence-floor",
        type=float,
        default=None,
        help="Optional manual ID AUPRC prevalence floor. Defaults to ID-test positive prevalence.",
    )
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


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def read_metric_pair(results_dir: Path, spec: MetricSpec) -> dict[str, object]:
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
        id_mean = float(row[f"{spec.id_metric}_mean"])
        id_std = float(row[f"{spec.id_metric}_std"])
        ood_mean = float(row[f"{spec.ood_metric}_mean"])
        ood_std = float(row[f"{spec.ood_metric}_std"])
        n_folds = int(float(row.get("n_folds_found") or row.get("n_folds") or 0))
    else:
        id_values = [float(row[spec.id_metric]) for _, row in source_rows]
        ood_values = [float(row[spec.ood_metric]) for _, row in source_rows]
        id_mean = mean(id_values)
        id_std = stdev(id_values) if len(id_values) > 1 else 0.0
        ood_mean = mean(ood_values)
        ood_std = stdev(ood_values) if len(ood_values) > 1 else 0.0
        n_folds = len(source_rows)

    return {
        "method": spec.method,
        "encoder": spec.encoder,
        "id_auprc_mean": id_mean,
        "id_auprc_std": id_std,
        "ood_auroc_mean": ood_mean,
        "ood_auroc_std": ood_std,
        "n_folds": n_folds,
        "sources": ";".join(str(path) for path, _ in source_rows),
        "id_metric": spec.id_metric,
        "ood_metric": spec.ood_metric,
        "note": spec.note,
    }


def missing_row(spec: MetricSpec, note: str) -> dict[str, object]:
    return {
        "method": spec.method,
        "encoder": spec.encoder,
        "id_auprc_mean": math.nan,
        "id_auprc_std": math.nan,
        "ood_auroc_mean": math.nan,
        "ood_auroc_std": math.nan,
        "n_folds": 0,
        "sources": "",
        "id_metric": spec.id_metric,
        "ood_metric": spec.ood_metric,
        "note": note,
    }


def compute_prevalence_floor(results_dir: Path) -> tuple[float | None, str]:
    sources = sorted(results_dir.glob("heldout_disease_logreg_fold*of3_predictions.csv"))
    values: list[float] = []
    used_sources: list[str] = []
    for path in sources:
        labels: list[int] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == "id_test":
                    labels.append(int(float(row["label"])))
        if labels:
            values.append(sum(labels) / len(labels))
            used_sources.append(str(path))
    if not values:
        return None, ""
    return mean(values), ";".join(used_sources)


def write_values_csv(
    rows: list[dict[str, object]],
    prevalence: float | None,
    prevalence_sources: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "encoder",
        "id_auprc_mean",
        "id_auprc_std",
        "ood_auroc_mean",
        "ood_auroc_std",
        "n_folds",
        "id_metric",
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
                    "id_auprc_mean": fmt(float(row["id_auprc_mean"])),
                    "id_auprc_std": fmt(float(row["id_auprc_std"])),
                    "ood_auroc_mean": fmt(float(row["ood_auroc_mean"])),
                    "ood_auroc_std": fmt(float(row["ood_auroc_std"])),
                    "n_folds": row["n_folds"],
                    "id_metric": row["id_metric"],
                    "ood_metric": row["ood_metric"],
                    "sources": row["sources"],
                    "note": row["note"],
                }
            )
        writer.writerow(
            {
                "method": "prevalence_floor",
                "encoder": "ID-test positive prevalence",
                "id_auprc_mean": "" if prevalence is None else f"{prevalence:.6f}",
                "id_auprc_std": "",
                "ood_auroc_mean": "",
                "ood_auroc_std": "",
                "n_folds": "",
                "id_metric": "label prevalence",
                "ood_metric": "",
                "sources": prevalence_sources,
                "note": "Horizontal dashed line in the ID AUPRC panel.",
            }
        )


def plot(
    rows: list[dict[str, object]],
    prevalence: float | None,
    output_dir: Path,
    output_stem: str,
    formats: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib is required to render the figure. Install project "
            "requirements, then rerun this script."
        ) from exc

    by_key = {(row["method"], row["encoder"]): row for row in rows}
    colors = {"DINOv2": "#2f6f9f", "FETAL-CLIP": "#c0523a"}
    bar_width = 0.36
    offsets = {"DINOv2": -bar_width / 2, "FETAL-CLIP": bar_width / 2}
    x_positions = list(range(len(METHODS)))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], "id_auprc", "ID AUPRC", "In-distribution classification", prevalence),
        (axes[1], "ood_auroc", "OOD AUROC", "Held-out disease detection", 0.5),
    ]

    for ax, metric_prefix, ylabel, title, reference in panels:
        for encoder in ENCODERS:
            xs: list[float] = []
            heights: list[float] = []
            errors: list[float] = []
            for index, method in enumerate(METHODS):
                row = by_key[(method, encoder)]
                value = float(row[f"{metric_prefix}_mean"])
                if math.isnan(value):
                    continue
                xs.append(index + offsets[encoder])
                heights.append(value)
                errors.append(float(row[f"{metric_prefix}_std"]))

            ax.bar(
                xs,
                heights,
                width=bar_width,
                label=encoder,
                color=colors[encoder],
                edgecolor="#1f2933",
                linewidth=0.8,
                yerr=errors,
                capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
                zorder=3,
            )

        if reference is not None:
            ax.axhline(
                reference,
                color="#475569",
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                zorder=2,
            )
            label = "chance = 0.500" if metric_prefix == "ood_auroc" else f"prevalence = {reference:.3f}"
            ax.text(
                len(METHODS) - 0.45,
                reference + 0.008,
                label,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#475569",
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(METHODS, rotation=25, ha="right")
        ax.grid(axis="y", color="#d9e2ec", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0.0, 0.58)
    axes[1].set_ylim(0.25, 0.9)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Encoder Lift Improves ID Classification Without Matching OOD Gains", fontsize=13)

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

    rows = [read_metric_pair(results_dir, spec) for spec in SPECS]
    prevalence, prevalence_sources = compute_prevalence_floor(results_dir)
    if args.prevalence_floor is not None:
        prevalence = float(args.prevalence_floor)
        prevalence_sources = "manual --prevalence-floor"

    values_path = output_dir / f"{args.output_stem}_values.csv"
    write_values_csv(rows, prevalence, prevalence_sources, values_path)
    print(f"Saved {values_path}")

    if args.no_plot:
        return

    plot(rows, prevalence, output_dir, args.output_stem, formats)


if __name__ == "__main__":
    main()
