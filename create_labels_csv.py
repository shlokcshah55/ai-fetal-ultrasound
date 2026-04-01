"""
Generate data/labels.csv from the lab machine video directories and subject_level_labels.csv.

Run this on the lab machine before training:
    python create_labels_csv.py

Writes: data/labels.csv  (columns: video_path, label, subject_id)

Binary label: 1 if any cardiac condition is present, 0 if all are zero.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

VIDEO_ROOT = Path("/vol/biomedic2/bkainz/ifind/fetalcardiac-srv")
VIDEO_DIRS = [
    VIDEO_ROOT / "fetal_cardiac_dataset",
    VIDEO_ROOT / "fetal_cardiac_dataset_from_xnat",
    VIDEO_ROOT / "fetal_cardiac_dataset_from_missing_extras",
]
LABELS_CSV = VIDEO_ROOT / "subject_level_labels.csv"
OUTPUT_CSV = Path(__file__).parent / "data" / "labels.csv"

CONDITION_COLS = ["avsd", "hlhs", "tga", "tetralogy", "raa", "coa", "p_atresia", "a_stenosis", "p_stenosis"]
SUBJECT_RE = re.compile(r"^(\d+)_")


def parse_subject_id(filename: str) -> int | None:
    m = SUBJECT_RE.match(filename)
    return int(m.group(1)) if m else None


def main() -> None:
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df["label"] = (labels_df[CONDITION_COLS].sum(axis=1) > 0).astype(int)
    subject_to_label = dict(zip(labels_df["subject"], labels_df["label"]))

    rows: list[dict] = []
    skipped_no_subject = 0
    skipped_not_in_labels = 0

    for video_dir in VIDEO_DIRS:
        if not video_dir.exists():
            print(f"  WARNING: directory not found, skipping: {video_dir}")
            continue
        for video_path in sorted(video_dir.rglob("*.mp4")):
            subject_id = parse_subject_id(video_path.name)
            if subject_id is None:
                skipped_no_subject += 1
                continue
            if subject_id not in subject_to_label:
                skipped_not_in_labels += 1
                continue
            rows.append({
                "video_path": str(video_path),
                "label": subject_to_label[subject_id],
                "subject_id": subject_id,
            })

    df = pd.DataFrame(rows, columns=["video_path", "label", "subject_id"])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    n_subjects = df["subject_id"].nunique()

    print(f"Written {len(df)} videos across {n_subjects} subjects → {OUTPUT_CSV}")
    print(f"  Positive (CHD): {n_pos}  |  Negative (normal): {n_neg}")
    if skipped_no_subject:
        print(f"  Skipped (could not parse subject ID): {skipped_no_subject}")
    if skipped_not_in_labels:
        print(f"  Skipped (subject not in labels CSV): {skipped_not_in_labels}")


if __name__ == "__main__":
    main()
