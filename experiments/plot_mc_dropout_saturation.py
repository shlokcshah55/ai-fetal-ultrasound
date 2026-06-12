from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PASSES = [1, 10, 50, 100, 1000]
MC_PASSES = [10, 50, 100, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Figure 6: MC Dropout saturation plot over stochastic "
            "pass count T. T=1 is recomputed from deterministic MLP fold "
            "predictions after subject-level probability pooling."
        )
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--output-stem", default="mc_dropout_saturation")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--n-bins", type=int, default=15)
    return parser.parse_args()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def predictive_entropy(probability: float, eps: float = 1e-8) -> float:
    p = min(max(float(probability), eps), 1.0 - eps)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def expected_calibration_error(
    labels: list[int],
    probabilities: list[float],
    *,
    n_bins: int,
) -> float:
    if not labels:
        return math.nan

    ece = 0.0
    n = len(labels)
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        if idx == 0:
            keep = [
                (label, prob)
                for label, prob in zip(labels, probabilities)
                if lo <= prob <= hi
            ]
        else:
            keep = [
                (label, prob)
                for label, prob in zip(labels, probabilities)
                if lo < prob <= hi
            ]
        if not keep:
            continue

        empirical_rate = mean(label for label, _ in keep)
        confidence = mean(prob for _, prob in keep)
        ece += (len(keep) / n) * abs(empirical_rate - confidence)
    return float(ece)


def auroc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[idx:end])
        rank += end - idx
        idx = end

    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def deterministic_subject_level_fold(
    results_dir: Path,
    fold: int,
    *,
    n_folds: int,
    n_bins: int,
) -> tuple[float, float]:
    path = results_dir / f"heldout_disease_baseline_fold{fold}of{n_folds}_predictions.csv"
    grouped: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"split": "", "label": 0, "probabilities": []}
    )

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in {"id_test", "heldout_disease"}:
                continue
            key = (split, row["subject_id"])
            grouped[key]["split"] = split
            grouped[key]["label"] = int(float(row["label"]))
            grouped[key]["probabilities"].append(float(row["pred_proba_chd"]))

    id_labels: list[int] = []
    id_probabilities: list[float] = []
    id_uncertainty: list[float] = []
    heldout_uncertainty: list[float] = []

    for item in grouped.values():
        probabilities = item["probabilities"]
        assert isinstance(probabilities, list)
        probability = float(mean(probabilities))
        uncertainty = predictive_entropy(probability)
        if item["split"] == "id_test":
            id_labels.append(int(item["label"]))
            id_probabilities.append(probability)
            id_uncertainty.append(uncertainty)
        else:
            heldout_uncertainty.append(uncertainty)

    ood_labels = [0] * len(id_uncertainty) + [1] * len(heldout_uncertainty)
    ood_scores = id_uncertainty + heldout_uncertainty
    return (
        auroc(ood_labels, ood_scores),
        expected_calibration_error(id_labels, id_probabilities, n_bins=n_bins),
    )


def mc_dropout_fold(
    results_dir: Path,
    fold: int,
    passes: int,
    *,
    n_folds: int,
    n_bins: int,
) -> tuple[float, float]:
    summary_path = (
        results_dir
        / f"heldout_disease_mc_dropout_fold{fold}of{n_folds}_mean_p0.3_T{passes}_summary.csv"
    )
    predictions_path = (
        results_dir
        / f"heldout_disease_mc_dropout_fold{fold}of{n_folds}_mean_p0.3_T{passes}_predictions.csv"
    )

    summary = read_single_row(summary_path)
    ood_auroc = float(summary["ood_auroc"])

    labels: list[int] = []
    probabilities: list[float] = []
    with predictions_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "id_test":
                continue
            labels.append(int(float(row["label"])))
            probabilities.append(float(row["pred_proba_chd"]))

    return ood_auroc, expected_calibration_error(labels, probabilities, n_bins=n_bins)


def collect_values(results_dir: Path, *, n_folds: int, n_bins: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for passes in PASSES:
        fold_auroc: list[float] = []
        fold_ece: list[float] = []
        for fold in range(n_folds):
            if passes == 1:
                ood_value, ece_value = deterministic_subject_level_fold(
                    results_dir,
                    fold,
                    n_folds=n_folds,
                    n_bins=n_bins,
                )
            else:
                ood_value, ece_value = mc_dropout_fold(
                    results_dir,
                    fold,
                    passes,
                    n_folds=n_folds,
                    n_bins=n_bins,
                )
            fold_auroc.append(ood_value)
            fold_ece.append(ece_value)

        rows.append(
            {
                "passes": passes,
                "method_source": (
                    "Deterministic MLP, subject-pooled probabilities"
                    if passes == 1
                    else f"MC Dropout p=0.3, T={passes}"
                ),
                "ood_auroc_mean": mean(fold_auroc),
                "ood_auroc_std": sample_std(fold_auroc),
                "id_ece_mean": mean(fold_ece),
                "id_ece_std": sample_std(fold_ece),
                "n_folds": n_folds,
                "fold_ood_aurocs": ";".join(f"{value:.6f}" for value in fold_auroc),
                "fold_id_eces": ";".join(f"{value:.6f}" for value in fold_ece),
            }
        )
    return rows


def write_values_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "passes",
        "method_source",
        "ood_auroc_mean",
        "ood_auroc_std",
        "id_ece_mean",
        "id_ece_std",
        "n_folds",
        "fold_ood_aurocs",
        "fold_id_eces",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{row[key]:.6f}"
                        if isinstance(row[key], float)
                        else row[key]
                    )
                    for key in fieldnames
                }
            )


def line_points(
    rows: list[dict[str, object]],
    metric_key: str,
    transform_x,
    transform_y,
) -> str:
    return " ".join(
        f"{transform_x(float(row['passes'])):.2f},{transform_y(float(row[metric_key])):.2f}"
        for row in rows
    )


def band_points(
    rows: list[dict[str, object]],
    mean_key: str,
    std_key: str,
    transform_x,
    transform_y,
    y_min: float,
    y_max: float,
) -> str:
    upper = []
    lower = []
    for row in rows:
        x = transform_x(float(row["passes"]))
        mean_value = float(row[mean_key])
        std_value = float(row[std_key])
        upper.append((x, transform_y(min(y_max, mean_value + std_value))))
        lower.append((x, transform_y(max(y_min, mean_value - std_value))))
    points = upper + list(reversed(lower))
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    normalised_attrs = []
    for key, value in attrs.items():
        attr_name = key[:-1] if key.endswith("_") else key
        attr_name = attr_name.replace("_", "-")
        normalised_attrs.append(f'{attr_name}="{value}"')
    attr_text = " ".join(normalised_attrs)
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_text}>{html.escape(text)}</text>'


def write_svg(rows: list[dict[str, object]], output_path: Path) -> None:
    width = 980
    height = 640
    margin_left = 82
    margin_right = 84
    margin_top = 104
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    ood_min, ood_max = 0.30, 0.86
    ece_min, ece_max = 0.00, 0.44

    def tx(passes: float) -> float:
        return margin_left + (math.log10(passes) / 3.0) * plot_width

    def ty_left(value: float) -> float:
        return margin_top + (ood_max - value) / (ood_max - ood_min) * plot_height

    def ty_right(value: float) -> float:
        return margin_top + (ece_max - value) / (ece_max - ece_min) * plot_height

    ood_color = "#2563eb"
    ece_color = "#c2410c"
    grid_color = "#dbe3ef"
    text_color = "#1f2937"
    muted = "#64748b"

    x_ticks = PASSES
    ood_ticks = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    ece_ticks = [0.00, 0.10, 0.20, 0.30, 0.40]

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; }",
        ".title { font-size: 20px; font-weight: 700; fill: #111827; }",
        ".subtitle { font-size: 12px; fill: #475569; }",
        ".axis { font-size: 12px; fill: #334155; }",
        ".label { font-size: 13px; font-weight: 700; }",
        ".legend { font-size: 13px; fill: #1f2937; }",
        ".note { font-size: 11px; fill: #64748b; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            28,
            31,
            "Figure 6. MC Dropout saturation by inference pass count (DINOv2)",
            class_="title",
        ),
        svg_text(
            28,
            50,
            "Both means are flat from T=10; the material change is T=1 to T=10, sharpest for calibration.",
            class_="subtitle",
        ),
        svg_text(
            28,
            68,
            "Bands show fold standard deviation; the wide AUROC band reflects large fold variance.",
            class_="subtitle",
        ),
    ]

    # Saturation region after the first stochastic setting.
    elements.append(
        f'<rect x="{tx(10):.2f}" y="{margin_top:.2f}" '
        f'width="{tx(1000) - tx(10):.2f}" height="{plot_height:.2f}" '
        'fill="#f8fafc" opacity="0.80"/>'
    )
    elements.append(
        f'<line x1="{tx(10):.2f}" y1="{margin_top:.2f}" '
        f'x2="{tx(10):.2f}" y2="{height - margin_bottom:.2f}" '
        'stroke="#475569" stroke-width="1.6" stroke-dasharray="6 5"/>'
    )
    elements.append(
        svg_text(
            tx(63),
            margin_top + 22,
            "saturation regime: T>=10",
            class_="note",
            text_anchor="middle",
        )
    )

    for tick in ood_ticks:
        y = ty_left(tick)
        elements.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="{grid_color}" stroke-width="1"/>'
        )
        elements.append(
            svg_text(
                margin_left - 12,
                y + 4,
                f"{tick:.2f}",
                class_="axis",
                fill=ood_color,
                text_anchor="end",
            )
        )

    for tick in ece_ticks:
        y = ty_right(tick)
        elements.append(
            svg_text(
                width - margin_right + 12,
                y + 4,
                f"{tick:.2f}",
                class_="axis",
                fill=ece_color,
                text_anchor="start",
            )
        )

    for tick in x_ticks:
        x = tx(tick)
        elements.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom}" stroke="{grid_color}" '
            'stroke-width="1" stroke-dasharray="3 5"/>'
        )
        elements.append(
            f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom + 6}" stroke="#475569" stroke-width="1"/>'
        )
        elements.append(
            svg_text(
                x,
                height - margin_bottom + 25,
                str(tick),
                class_="axis",
                text_anchor="middle",
            )
        )

    # Axes.
    elements.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
            f'y2="{height - margin_bottom}" stroke="#334155" stroke-width="1.2"/>',
            f'<line x1="{width - margin_right}" y1="{margin_top}" x2="{width - margin_right}" '
            f'y2="{height - margin_bottom}" stroke="#334155" stroke-width="1.2"/>',
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
            f'x2="{width - margin_right}" y2="{height - margin_bottom}" '
            'stroke="#334155" stroke-width="1.2"/>',
        ]
    )

    # Bands and lines.
    elements.append(
        f'<polygon points="{band_points(rows, "ood_auroc_mean", "ood_auroc_std", tx, ty_left, ood_min, ood_max)}" '
        f'fill="{ood_color}" opacity="0.055"/>'
    )
    elements.append(
        f'<polygon points="{band_points(rows, "id_ece_mean", "id_ece_std", tx, ty_right, ece_min, ece_max)}" '
        f'fill="{ece_color}" opacity="0.065"/>'
    )
    elements.append(
        f'<polyline points="{line_points(rows, "ood_auroc_mean", tx, ty_left)}" '
        f'fill="none" stroke="{ood_color}" stroke-width="4.2" stroke-linejoin="round"/>'
    )
    elements.append(
        f'<polyline points="{line_points(rows, "id_ece_mean", tx, ty_right)}" '
        f'fill="none" stroke="{ece_color}" stroke-width="4.2" stroke-linejoin="round"/>'
    )

    for row in rows:
        x = tx(float(row["passes"]))
        y_ood = ty_left(float(row["ood_auroc_mean"]))
        y_ece = ty_right(float(row["id_ece_mean"]))
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y_ood:.2f}" r="5.5" fill="{ood_color}" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y_ece:.2f}" r="5.5" fill="{ece_color}" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )

    # Axis labels.
    elements.append(
        svg_text(
            margin_left + plot_width / 2,
            height - 20,
            "Inference pass count T (log scale)",
            class_="label",
            fill=text_color,
            text_anchor="middle",
        )
    )
    elements.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" '
        f'class="label" fill="{ood_color}" text-anchor="middle" '
        'transform="rotate(-90 24 '
        f'{margin_top + plot_height / 2:.2f})">OOD AUROC</text>'
    )
    elements.append(
        f'<text x="{width - 22}" y="{margin_top + plot_height / 2:.2f}" '
        f'class="label" fill="{ece_color}" text-anchor="middle" '
        f'transform="rotate(90 {width - 22} '
        f'{margin_top + plot_height / 2:.2f})">ID ECE</text>'
    )

    # Legend.
    legend_x = margin_left + 18
    legend_y = margin_top + plot_height - 54
    elements.extend(
        [
            f'<rect x="{legend_x - 10}" y="{legend_y - 24}" width="306" height="58" '
            'rx="6" fill="#ffffff" opacity="0.92" stroke="#e2e8f0"/>',
            f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 34}" '
            f'y2="{legend_y - 4}" stroke="{ood_color}" stroke-width="4.2"/>',
            f'<circle cx="{legend_x + 17}" cy="{legend_y - 4}" r="4.6" fill="{ood_color}" '
            'stroke="#ffffff" stroke-width="1.2"/>',
            svg_text(legend_x + 44, legend_y, "OOD AUROC +/- std", class_="legend"),
            f'<line x1="{legend_x}" y1="{legend_y + 20}" x2="{legend_x + 34}" '
            f'y2="{legend_y + 20}" stroke="{ece_color}" stroke-width="4.2"/>',
            f'<circle cx="{legend_x + 17}" cy="{legend_y + 20}" r="4.6" fill="{ece_color}" '
            'stroke="#ffffff" stroke-width="1.2"/>',
            svg_text(legend_x + 44, legend_y + 24, "ID ECE +/- std", class_="legend"),
        ]
    )

    elements.append(
        svg_text(
            width - margin_right,
            height - 42,
            "T=1: deterministic MLP recomputed at subject level; T>=10: saved MC Dropout folds.",
            class_="note",
            text_anchor="end",
            fill=muted,
        )
    )
    elements.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    rows = collect_values(results_dir, n_folds=args.n_folds, n_bins=args.n_bins)
    values_path = output_dir / f"{args.output_stem}_values.csv"
    svg_path = output_dir / f"{args.output_stem}.svg"

    write_values_csv(rows, values_path)
    write_svg(rows, svg_path)

    print(f"Saved {values_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
