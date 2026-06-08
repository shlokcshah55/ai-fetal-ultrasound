from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
PICKLE_SUFFIXES = (".pk", ".pkl", ".pickle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot frame-level and video-level histograms for SonoNet 4CH "
            "probabilities, matching the current pipeline's label/probability "
            "thresholding rule."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Override config.data.csv_path.",
    )
    parser.add_argument(
        "--sononet-dir",
        default=None,
        help="Override config.sononet.dir.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Primary threshold to annotate; defaults to config.sononet.conf_threshold.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Comma-separated thresholds for diagnostics. Defaults to "
            "0.3,0.4,0.5,0.6,0.7 plus the primary threshold."
        ),
    )
    parser.add_argument(
        "--view-label",
        default="4ch",
        help="SonoNet label to treat as four-chamber heart view.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sononet_analysis",
        help="Directory for CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of equal-width histogram bins over [0, 1].",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV/JSON outputs only.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return config


def resolve_from_config_dir(config_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return config_path.resolve().parent / path


def parse_thresholds(raw: str | None, primary_threshold: float) -> list[float]:
    thresholds = list(DEFAULT_THRESHOLDS)
    if raw is not None:
        thresholds = [float(value.strip()) for value in raw.split(",") if value.strip()]
    thresholds.append(float(primary_threshold))
    return sorted(set(round(threshold, 6) for threshold in thresholds))


def find_sononet_pickle(sononet_dir: Path, video_path: str) -> Path | None:
    stem = Path(video_path).stem
    for suffix in PICKLE_SUFFIXES:
        candidate = sononet_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def finite_probabilities(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return finite[(finite >= 0.0) & (finite <= 1.0)]


def scan_sononet_probabilities(
    labels_df: pd.DataFrame,
    sononet_dir: Path,
    view_label: str,
    thresholds: list[float],
    primary_threshold: float,
    bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_cols = {"video_path", "label", "subject_id"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels CSV missing columns: {sorted(missing)}")

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    frame_hist_counts = np.zeros(bins, dtype=np.int64)
    video_max_hist_counts = np.zeros(bins, dtype=np.int64)
    threshold_frame_counts = {threshold: 0 for threshold in thresholds}
    threshold_video_counts = {threshold: 0 for threshold in thresholds}
    threshold_video_frame_sums = {threshold: 0 for threshold in thresholds}

    rows: list[dict[str, Any]] = []
    n_missing_pickles = 0
    n_malformed_pickles = 0
    n_invalid_probability_rows = 0

    iterator = tqdm(
        labels_df.itertuples(index=False),
        total=len(labels_df),
        desc="Scanning SonoNet pickles",
    )
    for record in iterator:
        row = record._asdict()
        video_path = str(row["video_path"])
        pk_path = find_sononet_pickle(sononet_dir, video_path)

        base_row: dict[str, Any] = {
            "video_path": video_path,
            "subject_id": row["subject_id"],
            "label": row["label"],
            "sononet_path": str(pk_path) if pk_path is not None else "",
            "has_sononet_pickle": pk_path is not None,
            "n_sononet_rows": 0,
            "n_4ch_rows": 0,
            "n_valid_4ch_probabilities": 0,
            "max_4ch_probability": np.nan,
            "mean_4ch_probability": np.nan,
            "median_4ch_probability": np.nan,
            "kept_at_primary_threshold": False,
            "n_4ch_frames_at_primary_threshold": 0,
        }

        if pk_path is None:
            n_missing_pickles += 1
            rows.append(base_row)
            continue

        try:
            pk_df = pd.read_pickle(pk_path)
        except Exception as exc:  # pragma: no cover - depends on external files
            n_malformed_pickles += 1
            base_row["sononet_error"] = repr(exc)
            rows.append(base_row)
            continue

        if not isinstance(pk_df, pd.DataFrame):
            n_malformed_pickles += 1
            base_row["sononet_error"] = f"Expected DataFrame, got {type(pk_df)!r}"
            rows.append(base_row)
            continue

        required_pk_cols = {"label", "probability"}
        missing_pk_cols = required_pk_cols - set(pk_df.columns)
        if missing_pk_cols:
            n_malformed_pickles += 1
            base_row["sononet_error"] = (
                f"SonoNet pickle missing columns: {sorted(missing_pk_cols)}"
            )
            rows.append(base_row)
            continue

        view_df = pk_df[pk_df["label"] == view_label]
        probabilities = finite_probabilities(view_df["probability"])
        n_invalid_probability_rows += int(len(view_df) - len(probabilities))

        base_row["n_sononet_rows"] = int(len(pk_df))
        base_row["n_4ch_rows"] = int(len(view_df))
        base_row["n_valid_4ch_probabilities"] = int(len(probabilities))

        if len(probabilities) == 0:
            rows.append(base_row)
            continue

        frame_hist_counts += np.histogram(probabilities, bins=bin_edges)[0]

        max_probability = float(np.max(probabilities))
        video_max_hist_counts += np.histogram([max_probability], bins=bin_edges)[0]

        primary_count = int(np.sum(probabilities >= primary_threshold))
        base_row.update(
            {
                "max_4ch_probability": max_probability,
                "mean_4ch_probability": float(np.mean(probabilities)),
                "median_4ch_probability": float(np.median(probabilities)),
                "kept_at_primary_threshold": primary_count > 0,
                "n_4ch_frames_at_primary_threshold": primary_count,
            }
        )

        for threshold in thresholds:
            qualifying = int(np.sum(probabilities >= threshold))
            threshold_frame_counts[threshold] += qualifying
            threshold_video_frame_sums[threshold] += qualifying
            if qualifying > 0:
                threshold_video_counts[threshold] += 1
            base_row[f"n_4ch_frames_at_threshold_{threshold:g}"] = qualifying
            base_row[f"kept_at_threshold_{threshold:g}"] = qualifying > 0

        rows.append(base_row)

    summary_df = pd.DataFrame(rows)
    n_videos = int(len(labels_df))
    n_with_pickle = int(summary_df["has_sononet_pickle"].sum())
    n_with_valid_4ch = int((summary_df["n_valid_4ch_probabilities"] > 0).sum())
    n_total_4ch_probs = int(summary_df["n_valid_4ch_probabilities"].sum())

    threshold_rows = []
    for threshold in thresholds:
        n_kept = int(threshold_video_counts[threshold])
        n_frames = int(threshold_frame_counts[threshold])
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "videos_kept": n_kept,
                "videos_total": n_videos,
                "videos_with_sononet_pickle": n_with_pickle,
                "videos_with_valid_4ch_probability": n_with_valid_4ch,
                "video_keep_rate_all_videos": n_kept / n_videos if n_videos else np.nan,
                "video_keep_rate_with_pickle": (
                    n_kept / n_with_pickle if n_with_pickle else np.nan
                ),
                "video_keep_rate_with_valid_4ch_probability": (
                    n_kept / n_with_valid_4ch if n_with_valid_4ch else np.nan
                ),
                "qualifying_4ch_frames": n_frames,
                "total_valid_4ch_probability_rows": n_total_4ch_probs,
                "frame_keep_rate_among_valid_4ch_rows": (
                    n_frames / n_total_4ch_probs if n_total_4ch_probs else np.nan
                ),
                "mean_qualifying_frames_per_kept_video": (
                    threshold_video_frame_sums[threshold] / n_kept if n_kept else 0.0
                ),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)

    frame_hist_df = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": frame_hist_counts,
        }
    )
    video_max_hist_df = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": video_max_hist_counts,
        }
    )
    hist_df = frame_hist_df.copy()
    hist_df["histogram"] = "frame_4ch_probability"
    video_max_hist_df = video_max_hist_df.copy()
    video_max_hist_df["histogram"] = "video_max_4ch_probability"
    hist_df = pd.concat([hist_df, video_max_hist_df], ignore_index=True)

    metadata = {
        "view_label": view_label,
        "primary_threshold": float(primary_threshold),
        "thresholds": [float(threshold) for threshold in thresholds],
        "bins": int(bins),
        "videos_total": n_videos,
        "videos_with_sononet_pickle": n_with_pickle,
        "videos_missing_sononet_pickle": n_missing_pickles,
        "malformed_sononet_pickles": n_malformed_pickles,
        "videos_with_valid_4ch_probability": n_with_valid_4ch,
        "total_valid_4ch_probability_rows": n_total_4ch_probs,
        "invalid_4ch_probability_rows": n_invalid_probability_rows,
    }
    return summary_df, threshold_df, hist_df, metadata


def plot_histograms(
    hist_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    output_dir: Path,
    primary_threshold: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for PNG plots. Install requirements or rerun "
            "with --no-plots to write only CSV/JSON outputs."
        ) from exc

    plots = [
        (
            "frame_4ch_probability",
            "4CH frame probability",
            "4ch_frame_probability_histogram.png",
        ),
        (
            "video_max_4ch_probability",
            "Max 4CH probability per video",
            "4ch_video_max_probability_histogram.png",
        ),
    ]
    threshold_values = threshold_df["threshold"].to_numpy(dtype=float)

    for histogram_name, xlabel, filename in plots:
        plot_df = hist_df[hist_df["histogram"] == histogram_name].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        widths = plot_df["bin_right"].to_numpy() - plot_df["bin_left"].to_numpy()
        ax.bar(
            plot_df["bin_left"],
            plot_df["count"],
            width=widths,
            align="edge",
            color="#4477AA",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(
            primary_threshold,
            color="#CC3311",
            linewidth=2,
            label=f"Current threshold = {primary_threshold:g}",
        )
        for threshold in threshold_values:
            if threshold == primary_threshold:
                continue
            ax.axvline(threshold, color="#999999", linewidth=0.8, alpha=0.35)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(str(config_path))

    csv_path = (
        Path(args.csv_path)
        if args.csv_path is not None
        else resolve_from_config_dir(config_path, config["data"]["csv_path"])
    )
    sononet_cfg = config.get("sononet", {})
    sononet_dir_value = args.sononet_dir or sononet_cfg.get("dir")
    if not sononet_dir_value:
        raise SystemExit("SonoNet dir is not set. Use --sononet-dir or config.sononet.dir.")
    sononet_dir = Path(sononet_dir_value)
    if not sononet_dir.exists():
        raise SystemExit(f"SonoNet dir does not exist: {sononet_dir}")

    primary_threshold = float(
        args.threshold
        if args.threshold is not None
        else sononet_cfg.get("conf_threshold", 0.5)
    )
    thresholds = parse_thresholds(args.thresholds, primary_threshold)

    labels_df = pd.read_csv(csv_path)
    if args.max_videos is not None:
        labels_df = labels_df.head(args.max_videos).copy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, threshold_df, hist_df, metadata = scan_sononet_probabilities(
        labels_df=labels_df,
        sononet_dir=sononet_dir,
        view_label=args.view_label,
        thresholds=thresholds,
        primary_threshold=primary_threshold,
        bins=args.bins,
    )

    summary_path = output_dir / "4ch_probability_summary.csv"
    threshold_path = output_dir / "4ch_threshold_summary.csv"
    hist_path = output_dir / "4ch_probability_histograms.csv"
    metadata_path = output_dir / "4ch_probability_metadata.json"

    summary_df.to_csv(summary_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    hist_df.to_csv(hist_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if not args.no_plots:
        plot_histograms(hist_df, threshold_df, output_dir, primary_threshold)

    print(f"Wrote per-video summary: {summary_path}")
    print(f"Wrote threshold summary: {threshold_path}")
    print(f"Wrote histogram counts: {hist_path}")
    print(f"Wrote metadata: {metadata_path}")
    if not args.no_plots:
        print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
