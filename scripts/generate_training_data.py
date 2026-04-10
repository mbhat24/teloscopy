#!/usr/bin/env python3
"""Generate a synthetic training dataset for Teloscopy models.

Usage
-----
    python scripts/generate_training_data.py [options]

Options
-------
--output-dir DIR    Output directory (default: ``training_data/``).
--n-train N         Training images (default: 200).
--n-val N           Validation images (default: 40).
--n-test N          Test images (default: 40).
--size WxH          Image size (default: 512x512).
--seed N            Random seed (default: 42).

Outputs
-------
* ``training_data/train/images/`` — compressed ``.npz`` image files.
* ``training_data/train/annotations/`` — JSON annotation files.
* ``training_data/val/`` and ``training_data/test/`` — same structure.
* ``training_data/manifest.json`` — dataset manifest with metadata.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from teloscopy.data.training import TrainingDatasetGenerator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("generate_training_data")


def parse_size(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid size: {s}")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training dataset for Teloscopy"
    )
    parser.add_argument("--output-dir", default="training_data")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--size", type=parse_size, default="512x512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="teloscopy_training_v1")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    logger.info(
        "Generating %d train / %d val / %d test images at %s",
        args.n_train, args.n_val, args.n_test, args.size,
    )

    gen = TrainingDatasetGenerator(
        output_dir=args.output_dir,
        name=args.name,
        version=args.version,
    )
    manifest = gen.generate(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        image_size=args.size,
        seed=args.seed,
    )

    logger.info(
        "Done! %d samples generated. Manifest: %s/manifest.json",
        manifest.total_samples,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
