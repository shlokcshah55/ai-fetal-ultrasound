from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from data.dataset import get_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute and cache SonoNet-derived frame indices (no DINO / training)."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent
    os.chdir(repo_root)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sononet_cfg = config.get("sononet", {})
    sononet_dir = sononet_cfg.get("dir")
    if not sononet_dir:
        raise SystemExit(
            "SonoNet filtering is not enabled. Set `sononet.dir` in config.yaml."
        )

    # This triggers:
    # - reading the labels CSV
    # - scanning SonoNet .pk files (first run) OR loading the cached indices map
    # - filtering out videos with no qualifying 4CH frames
    #
    # It does NOT decode videos or load frames; frames are only read when iterating the loaders.
    get_dataloaders(
        csv_path=config["data"]["csv_path"],
        n_frames=config["data"]["n_frames"],
        batch_size=1,
        num_workers=0,
        split=config["data"]["split"],
        seed=config["training"]["seed"],
        sononet_dir=sononet_dir,
        conf_threshold=sononet_cfg.get("conf_threshold", 0.5),
    )

    print("SonoNet cache step complete.")


if __name__ == "__main__":
    main()
