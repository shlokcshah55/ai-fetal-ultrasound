"""
Inspect a SonoNet frame-wise view classification pickle file.

Run on the lab machine:
    python inspect_sononet.py

Prints the schema and a sample of one pickle file so we can understand
the structure before integrating it into the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SONONET_DIR = Path("/vol/biomedic3/bkainz/sononet_logs")


def main() -> None:
    pickles = sorted(SONONET_DIR.rglob("*.pk"))
    if not pickles:
        pickles = sorted(SONONET_DIR.rglob("*.pkl"))
    if not pickles:
        pickles = sorted(SONONET_DIR.rglob("*.pickle"))

    if not pickles:
        print("No pickle files found in", SONONET_DIR)
        return

    print(f"Found {len(pickles)} pickle files in {SONONET_DIR}")
    print(f"First few filenames:")
    for p in pickles[:5]:
        print(f"  {p.name}")

    # Inspect the first file
    sample_path = pickles[0]
    print(f"\n--- Inspecting: {sample_path.name} ---")

    df = pd.read_pickle(sample_path)

    print(f"Type: {type(df)}")
    if isinstance(df, pd.DataFrame):
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nSample values per column:")
        for col in df.columns:
            print(f"  {col}: {df[col].unique()[:5]}")
    else:
        print(f"Value: {df}")


if __name__ == "__main__":
    main()
