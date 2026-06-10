from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


METHODS = ["LR", "MLP", "MC Dropout", "Energy", "VOS", "EDL"]


@dataclass(frozen=True)
class MetricSpec:
    method: str
    encoder: str
    source_pattern: str | None
    expected_classifier: str | None = None
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
        "MLP",
        "FETAL-CLIP",
        "heldout_disease_baseline_fetal_clip_fold*of3_summary.csv",
        "MLP",
    ),
    MetricSpec(
        "LR",
        "FETAL-CLIP",
        "heldout_disease_representation_probe_fetal_clip_fold*of3_mean_summary.csv",
        "RepresentationProbe",
        "Fresh LogisticRegression representation probe; baseline fetal-clip stem now contains the MLP run.",
    ),
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
        "Uses fetal-clip checkpoint/config id saved in the Energy summaries.",
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
            "Create Figure 1: grouped encoder-lift ID AUPRC bar chart "
            "for DINOv2 vs FETAL-CLIP."
        )
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--output-stem", default="encoder_lift_auprc")
    parser.add_argument(
        "--prevalence-floor",
        type=float,
        default=None,
        help=(
            "Optional manual prevalence floor. By default this is computed "
            "from DINOv2 LR fold prediction CSVs over split=id_test."
        ),
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats supported by matplotlib.",
    )
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


def read_auprc_from_spec(results_dir: Path, spec: MetricSpec) -> dict[str, object]:
    if spec.source_pattern is None:
        return {
            "method": spec.method,
            "encoder": spec.encoder,
            "mean": math.nan,
            "std": math.nan,
            "n_folds": 0,
            "sources": "",
            "note": spec.note,
        }

    sources = sorted(results_dir.glob(spec.source_pattern))
    if not sources:
        return {
            "method": spec.method,
            "encoder": spec.encoder,
            "mean": math.nan,
            "std": math.nan,
            "n_folds": 0,
            "sources": "",
            "note": f"NOT YET AVAILABLE: no files matched {spec.source_pattern}",
        }

    source_rows = [(path, read_single_row(path)) for path in sources]
    if spec.expected_classifier is not None:
        source_rows = [
            (path, row)
            for path, row in source_rows
            if row.get("classifier") == spec.expected_classifier
        ]

    if not source_rows:
        return {
            "method": spec.method,
            "encoder": spec.encoder,
            "mean": math.nan,
            "std": math.nan,
            "n_folds": 0,
            "sources": "",
            "note": (
                f"NOT YET AVAILABLE: no files matching {spec.source_pattern} "
                f"had classifier={spec.expected_classifier!r}."
            ),
        }

    if len(source_rows) == 1 and source_rows[0][0].name.endswith("_cv_summary.csv"):
        row = source_rows[0][1]
        mean_value = float(row["id_auprc_mean"])
        std_value = float(row["id_auprc_std"])
        n_folds = int(float(row.get("n_folds_found") or row.get("n_folds") or 0))
    else:
        values = [float(row["id_auprc"]) for _, row in source_rows]
        mean_value = mean(values)
        std_value = pstdev(values) if len(values) > 1 else 0.0
        n_folds = len(values)

    return {
        "method": spec.method,
        "encoder": spec.encoder,
        "mean": mean_value,
        "std": std_value,
        "n_folds": n_folds,
        "sources": ";".join(str(path) for path, _ in source_rows),
        "note": spec.note,
    }


def compute_prevalence_floor(results_dir: Path) -> tuple[float | None, str]:
    sources = sorted(results_dir.glob("heldout_disease_logreg_fold*of3_predictions.csv"))
    fold_prevalences: list[float] = []
    used_sources: list[str] = []

    for path in sources:
        labels: list[int] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == "id_test":
                    labels.append(int(float(row["label"])))
        if labels:
            fold_prevalences.append(sum(labels) / len(labels))
            used_sources.append(str(path))

    if not fold_prevalences:
        return None, ""
    return mean(fold_prevalences), ";".join(used_sources)


def write_values_csv(rows: list[dict[str, object]], prevalence: float | None, prevalence_sources: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "encoder",
        "id_auprc_mean",
        "id_auprc_std",
        "n_folds",
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
                    "id_auprc_mean": format_metric(row["mean"]),
                    "id_auprc_std": format_metric(row["std"]),
                    "n_folds": row["n_folds"],
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
                "n_folds": "",
                "sources": prevalence_sources,
                "note": "Horizontal dashed line in the figure.",
            }
        )


def format_metric(value: object) -> str:
    if not isinstance(value, (float, int)) or math.isnan(float(value)):
        return ""
    return f"{float(value):.6f}"


def plot(rows: list[dict[str, object]], prevalence: float | None, output_dir: Path, output_stem: str, formats: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib is required to render the figure. Install project "
            "requirements, then rerun this script."
        ) from exc

    by_key = {(row["method"], row["encoder"]): row for row in rows}
    x_positions = list(range(len(METHODS)))
    bar_width = 0.36
    colors = {"DINOv2": "#2f6f9f", "FETAL-CLIP": "#c0523a"}
    offsets = {"DINOv2": -bar_width / 2, "FETAL-CLIP": bar_width / 2}

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)

    for encoder in ["DINOv2", "FETAL-CLIP"]:
        xs: list[float] = []
        heights: list[float] = []
        errors: list[float] = []
        for index, method in enumerate(METHODS):
            row = by_key[(method, encoder)]
            value = float(row["mean"])
            if math.isnan(value):
                continue
            xs.append(index + offsets[encoder])
            heights.append(value)
            errors.append(float(row["std"]))

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

    missing_row = by_key[("MLP", "FETAL-CLIP")]
    if int(missing_row["n_folds"]) == 0:
        x = METHODS.index("MLP") + offsets["FETAL-CLIP"]
        ax.text(
            x,
            0.035,
            "N/A",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#64748b",
            rotation=90,
        )

    if prevalence is not None:
        ax.axhline(
            prevalence,
            color="#475569",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            zorder=2,
        )
        ax.text(
            len(METHODS) - 0.45,
            prevalence + 0.008,
            f"prevalence floor = {prevalence:.3f}",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#475569",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("ID AUPRC")
    ax.set_xlabel("Classifier / UQ head")
    ax.set_title("Encoder Lift in In-Distribution CHD Classification")
    ax.set_ylim(0.0, 0.58)
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    rows = [read_auprc_from_spec(results_dir, spec) for spec in SPECS]
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
