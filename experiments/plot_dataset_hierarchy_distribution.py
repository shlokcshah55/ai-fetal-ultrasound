from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


DEFAULT_PALETTE = {
    "ink": "#1f2933",
    "muted": "#64748b",
    "grid": "#d9e2ec",
    "panel": "#f8fafc",
    "subject": "#0f766e",
    "subject_light": "#ccfbf1",
    "video": "#2563eb",
    "video_light": "#dbeafe",
    "frame": "#7c3aed",
    "frame_light": "#ede9fe",
    "accent": "#b45309",
    "accent_light": "#fef3c7",
    "danger": "#b91c1c",
    "danger_light": "#fee2e2",
    "hist": "#2563eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dataset hierarchy diagram plus a clips-per-subject "
            "distribution visualization as a standalone SVG."
        )
    )
    parser.add_argument("--csv-path", default="data/labels.csv")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--output-dir", default="results/dataset_diagram")
    parser.add_argument(
        "--output-name",
        default="dataset_hierarchy_distribution.svg",
        help="SVG filename to write inside --output-dir.",
    )
    parser.add_argument(
        "--summary-name",
        default="dataset_hierarchy_distribution_summary.json",
        help="JSON summary filename to write inside --output-dir.",
    )
    parser.add_argument(
        "--display-subjects",
        type=int,
        default=None,
        help=(
            "Override the subject count shown in the top-left box. By default, "
            "the count is computed from --csv-path."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=35,
        help="Number of histogram bins across the full clips-per-subject range.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="SVG canvas width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="SVG canvas height in pixels.",
    )
    return parser.parse_args()


def read_clips_per_subject(csv_path: Path, subject_col: str) -> list[int]:
    counts: Counter[str] = Counter()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        if subject_col not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} missing subject column {subject_col!r}; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            subject_id = row.get(subject_col, "").strip()
            if subject_id:
                counts[subject_id] += 1
    if not counts:
        raise ValueError(f"No subjects found in {csv_path}")
    return list(counts.values())


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_counts(counts: list[int]) -> dict[str, float | int]:
    values = sorted(float(value) for value in counts)
    n = len(values)
    total = int(sum(values))
    return {
        "subjects": n,
        "clips": total,
        "mean": sum(values) / n,
        "median": percentile(values, 0.5),
        "min": int(values[0]),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "max": int(values[-1]),
        "single_clip_subjects": sum(1 for value in values if value == 1),
        "max_clip_subjects": sum(1 for value in values if value == values[-1]),
    }


def make_histogram(counts: list[int], bins: int) -> tuple[list[int], list[float]]:
    min_value = min(counts)
    max_value = max(counts)
    if bins < 1:
        raise ValueError("--bins must be at least 1")
    if min_value == max_value:
        return [len(counts)], [min_value - 0.5, max_value + 0.5]

    left = 0.5
    right = max_value + 0.5
    width = (right - left) / bins
    hist = [0 for _ in range(bins)]
    for value in counts:
        idx = int((value - left) / width)
        idx = max(0, min(bins - 1, idx))
        hist[idx] += 1
    edges = [left + i * width for i in range(bins + 1)]
    return hist, edges


def fmt_int(value: int | float) -> str:
    return f"{int(round(value)):,}"


def fmt_float(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 24,
    weight: int = 400,
    fill: str = "#1f2933",
    anchor: str = "start",
    family: str = "Inter, Arial, sans-serif",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1.0,
    rx: float = 8.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.1f}" opacity="{opacity:.3f}"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    stroke_width: float = 2.0,
    marker_end: bool = False,
    dash: str | None = None,
) -> str:
    marker = ' marker-end="url(#arrow)"' if marker_end else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
        f'stroke-linecap="round"{marker}{dash_attr}/>'
    )


def polyline(points: Iterable[tuple[float, float]], *, stroke: str, stroke_width: float = 2.0) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.1f}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )


def draw_labeled_box(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    sublabel: str,
    *,
    fill: str,
    stroke: str,
    label_size: int = 26,
    sublabel_size: int = 16,
) -> list[str]:
    cy = y + h / 2
    return [
        rect(x, y, w, h, fill=fill, stroke=stroke, stroke_width=2.0, rx=8),
        svg_text(x + w / 2, cy - 3, label, size=label_size, weight=700, fill=stroke, anchor="middle"),
        svg_text(x + w / 2, cy + 25, sublabel, size=sublabel_size, fill=DEFAULT_PALETTE["muted"], anchor="middle"),
    ]


def draw_frame_stack(x: float, y: float, *, n_frames: int = 14) -> list[str]:
    parts: list[str] = []
    for idx in range(n_frames):
        offset = idx * 3.2
        shade = "#f5f3ff" if idx % 2 == 0 else "#ede9fe"
        parts.append(
            rect(
                x + offset,
                y + offset,
                74,
                44,
                fill=shade,
                stroke=DEFAULT_PALETTE["frame"],
                stroke_width=1.1,
                rx=4,
            )
        )
    parts.append(svg_text(x + 55, y + 95, "10-20 frames", size=15, fill=DEFAULT_PALETTE["frame"], anchor="middle", weight=700))
    return parts


def draw_hierarchy_panel(display_subjects: int, stats: dict[str, float | int]) -> list[str]:
    p = DEFAULT_PALETTE
    parts: list[str] = []
    parts.append(svg_text(50, 62, "Hierarchical Dataset Structure", size=32, weight=800, fill=p["ink"]))
    parts.append(svg_text(50, 92, "Subject-level splitting prevents leakage from correlated frames.", size=17, fill=p["muted"]))

    parts.extend(
        draw_labeled_box(
            115,
            128,
            450,
            76,
            f"{fmt_int(display_subjects)} Subjects",
            "independent grouping unit",
            fill=p["subject_light"],
            stroke=p["subject"],
        )
    )
    parts.extend(
        draw_labeled_box(
            215,
            262,
            250,
            64,
            "Subject example",
            f"median {fmt_int(stats['median'])} clips",
            fill="#ffffff",
            stroke=p["subject"],
            label_size=22,
        )
    )
    parts.append(line(340, 204, 340, 262, stroke=p["subject"], stroke_width=2.3, marker_end=True))
    parts.append(svg_text(356, 240, "videos per subject", size=15, fill=p["subject"], weight=700))

    video_xs = [78, 218, 358, 498]
    video_y = 390
    for idx, x in enumerate(video_xs, start=1):
        parts.append(polyline([(340, 326), (340, 362), (x + 47, 362), (x + 47, video_y)], stroke=p["grid"], stroke_width=2.0))
        parts.extend(
            draw_labeled_box(
                x,
                video_y,
                94,
                58,
                f"Video {idx}",
                "clip",
                fill=p["video_light"],
                stroke=p["video"],
                label_size=17,
                sublabel_size=13,
            )
        )

    parts.append(svg_text(84, 365, "3-4 example videos under one subject", size=15, fill=p["muted"]))

    stack_anchors = [88, 228, 368, 508]
    for x in stack_anchors:
        parts.append(line(x + 37, video_y + 58, x + 37, 503, stroke=p["grid"], stroke_width=2.0, marker_end=True))
        parts.extend(draw_frame_stack(x - 20, 518, n_frames=13))

    parts.append(svg_text(108, 694, "Frames per video", size=18, weight=800, fill=p["frame"]))
    parts.append(svg_text(108, 720, "Shown as stacks to emphasize temporal correlation.", size=15, fill=p["muted"]))

    parts.append(line(560, 646, 522, 690, stroke=p["accent"], stroke_width=3.0, marker_end=True))
    parts.append(
        rect(350, 690, 315, 105, fill=p["accent_light"], stroke=p["accent"], stroke_width=2.0, rx=8)
    )
    parts.append(svg_text(370, 725, "frames highly correlated", size=18, weight=800, fill=p["accent"]))
    parts.append(svg_text(370, 752, "-> subject-level split", size=18, weight=800, fill=p["accent"]))
    parts.append(svg_text(370, 779, "enforces independence", size=18, weight=800, fill=p["accent"]))

    return parts


def axis_x(value: float, x0: float, x1: float, min_x: float, max_x: float) -> float:
    return x0 + (value - min_x) / (max_x - min_x) * (x1 - x0)


def axis_y(value: float, y0: float, y1: float, max_y: float) -> float:
    return y0 - (value / max_y) * (y0 - y1)


def draw_distribution_panel(counts: list[int], stats: dict[str, float | int], bins: int) -> list[str]:
    p = DEFAULT_PALETTE
    parts: list[str] = []
    hist, edges = make_histogram(counts, bins)
    max_count = max(hist)
    y_grid_max = max(10, int(math.ceil(max_count / 100.0) * 100))
    x_min = 0.5
    x_max = max(counts) + 0.5

    x0, x1 = 850.0, 1510.0
    y0, y1 = 665.0, 170.0
    box_y = 752.0

    parts.append(svg_text(830, 62, "Clips per Subject Distribution", size=32, weight=800, fill=p["ink"]))
    parts.append(svg_text(830, 92, "Histogram with mean, median, range, and imbalance extremes.", size=17, fill=p["muted"]))

    # Shaded extremes.
    single_left = axis_x(0.5, x0, x1, x_min, x_max)
    single_right = axis_x(1.5, x0, x1, x_min, x_max)
    parts.append(rect(single_left, y1, single_right - single_left, y0 - y1, fill=p["danger_light"], stroke="none", rx=0, opacity=0.9))

    high_start_value = max(float(stats["q3"]), float(stats["max"]) - 25.0)
    outlier_left = axis_x(high_start_value, x0, x1, x_min, x_max)
    parts.append(rect(outlier_left, y1, x1 - outlier_left, y0 - y1, fill=p["accent_light"], stroke="none", rx=0, opacity=0.9))

    for tick in range(0, y_grid_max + 1, max(1, y_grid_max // 5)):
        y = axis_y(tick, y0, y1, y_grid_max)
        parts.append(line(x0, y, x1, y, stroke=p["grid"], stroke_width=1.0))
        parts.append(svg_text(x0 - 14, y + 5, fmt_int(tick), size=13, fill=p["muted"], anchor="end"))

    parts.append(line(x0, y0, x1, y0, stroke=p["ink"], stroke_width=2.0))
    parts.append(line(x0, y0, x0, y1, stroke=p["ink"], stroke_width=2.0))

    for idx, bin_count in enumerate(hist):
        left = axis_x(edges[idx], x0, x1, x_min, x_max)
        right = axis_x(edges[idx + 1], x0, x1, x_min, x_max)
        top = axis_y(bin_count, y0, y1, y_grid_max)
        parts.append(
            rect(
                left + 1.0,
                top,
                max(1.0, right - left - 2.0),
                y0 - top,
                fill=p["hist"],
                stroke="#ffffff",
                stroke_width=0.6,
                rx=1.5,
                opacity=0.86,
            )
        )

    tick_values = [1, 50, 100, 150, 200, 250, 300, int(stats["max"])]
    seen_ticks = set()
    for value in tick_values:
        if value in seen_ticks or value < 1 or value > stats["max"]:
            continue
        seen_ticks.add(value)
        x = axis_x(value, x0, x1, x_min, x_max)
        parts.append(line(x, y0, x, y0 + 8, stroke=p["ink"], stroke_width=1.5))
        parts.append(svg_text(x, y0 + 28, fmt_int(value), size=13, fill=p["muted"], anchor="middle"))

    mean_x = axis_x(float(stats["mean"]), x0, x1, x_min, x_max)
    median_x = axis_x(float(stats["median"]), x0, x1, x_min, x_max)
    max_x = axis_x(float(stats["max"]), x0, x1, x_min, x_max)
    min_x = axis_x(float(stats["min"]), x0, x1, x_min, x_max)

    parts.append(line(mean_x, y0, mean_x, y1, stroke=p["danger"], stroke_width=2.5, dash="7 6"))
    parts.append(line(median_x, y0, median_x, y1, stroke=p["subject"], stroke_width=2.5, dash="7 6"))
    parts.append(svg_text(mean_x + 8, y1 + 28, f"mean {fmt_float(float(stats['mean']))}", size=15, weight=800, fill=p["danger"]))
    parts.append(svg_text(median_x + 8, y1 + 53, f"median {fmt_float(float(stats['median']), 0)}", size=15, weight=800, fill=p["subject"]))

    parts.append(svg_text(x0 + 6, y1 - 24, f"single-clip subjects: {fmt_int(stats['single_clip_subjects'])}", size=14, weight=800, fill=p["danger"]))
    parts.append(svg_text(outlier_left + 8, y1 - 24, f"{fmt_int(stats['max'])}-clip outlier range", size=14, weight=800, fill=p["accent"]))
    parts.append(svg_text(max_x - 6, y1 + 28, f"max {fmt_int(stats['max'])}", size=14, weight=800, fill=p["accent"], anchor="end"))
    parts.append(svg_text(min_x + 6, y0 - 14, f"min {fmt_int(stats['min'])}", size=14, weight=800, fill=p["danger"]))

    parts.append(svg_text((x0 + x1) / 2, 725, "Clips per subject", size=18, weight=800, fill=p["ink"], anchor="middle"))
    parts.append(
        '<text x="780.0" y="417.5" font-family="Inter, Arial, sans-serif" '
        'font-size="18" font-weight="800" fill="#1f2933" text-anchor="middle" '
        'transform="rotate(-90 780.0 417.5)">Count</text>'
    )

    # Box plot.
    q1_x = axis_x(float(stats["q1"]), x0, x1, x_min, x_max)
    q3_x = axis_x(float(stats["q3"]), x0, x1, x_min, x_max)
    parts.append(line(min_x, box_y, max_x, box_y, stroke=p["muted"], stroke_width=2.0))
    parts.append(line(min_x, box_y - 16, min_x, box_y + 16, stroke=p["muted"], stroke_width=2.0))
    parts.append(line(max_x, box_y - 16, max_x, box_y + 16, stroke=p["muted"], stroke_width=2.0))
    parts.append(rect(q1_x, box_y - 24, q3_x - q1_x, 48, fill="#ffffff", stroke=p["video"], stroke_width=2.0, rx=5))
    parts.append(line(median_x, box_y - 26, median_x, box_y + 26, stroke=p["subject"], stroke_width=3.0))
    parts.append(svg_text(x0, box_y + 58, f"range {fmt_int(stats['min'])}-{fmt_int(stats['max'])}", size=15, fill=p["muted"]))
    parts.append(svg_text(q1_x, box_y - 34, f"Q1 {fmt_float(float(stats['q1']), 0)}", size=12, fill=p["muted"], anchor="middle"))
    parts.append(svg_text(q3_x, box_y - 34, f"Q3 {fmt_float(float(stats['q3']), 0)}", size=12, fill=p["muted"], anchor="middle"))

    stats_x, stats_y = 1070, 805
    parts.append(rect(stats_x, stats_y - 42, 380, 62, fill=p["panel"], stroke=p["grid"], stroke_width=1.5, rx=8))
    parts.append(svg_text(stats_x + 18, stats_y - 15, f"subjects {fmt_int(stats['subjects'])}", size=15, weight=800, fill=p["ink"]))
    parts.append(svg_text(stats_x + 158, stats_y - 15, f"clips {fmt_int(stats['clips'])}", size=15, weight=800, fill=p["ink"]))
    parts.append(svg_text(stats_x + 278, stats_y - 15, f"bins {bins}", size=15, weight=800, fill=p["ink"]))

    return parts


def build_svg(
    counts: list[int],
    stats: dict[str, float | int],
    *,
    display_subjects: int,
    bins: int,
    width: int,
    height: int,
) -> str:
    p = DEFAULT_PALETTE
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M2,2 L10,6 L2,10 Z" fill="{p["accent"]}"/>',
        "</marker>",
        "</defs>",
        rect(0, 0, width, height, fill="#ffffff", rx=0),
        rect(32, 30, 665, 820, fill=p["panel"], stroke=p["grid"], stroke_width=1.0, rx=12),
        rect(760, 30, 790, 820, fill=p["panel"], stroke=p["grid"], stroke_width=1.0, rx=12),
    ]
    parts.extend(draw_hierarchy_panel(display_subjects, stats))
    parts.extend(draw_distribution_panel(counts, stats, bins))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = read_clips_per_subject(csv_path, args.subject_col)
    stats = summarize_counts(counts)
    display_subjects = args.display_subjects if args.display_subjects is not None else int(stats["subjects"])

    svg = build_svg(
        counts,
        stats,
        display_subjects=display_subjects,
        bins=args.bins,
        width=args.width,
        height=args.height,
    )

    output_path = output_dir / args.output_name
    summary_path = output_dir / args.summary_name
    output_path.write_text(svg)
    summary = {
        "csv_path": str(csv_path),
        "subject_column": args.subject_col,
        "display_subjects": display_subjects,
        **stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")
    print(
        "Stats: "
        f"subjects={fmt_int(stats['subjects'])}, "
        f"clips={fmt_int(stats['clips'])}, "
        f"mean={fmt_float(float(stats['mean']))}, "
        f"median={fmt_float(float(stats['median']), 0)}, "
        f"range={fmt_int(stats['min'])}-{fmt_int(stats['max'])}"
    )


if __name__ == "__main__":
    main()
