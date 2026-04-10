#!/usr/bin/env python3
"""Run Teloscopy benchmarks and generate published results.

Usage
-----
    python scripts/run_benchmarks.py [--output-dir DIR] [--download]

Flags
-----
--output-dir DIR   Directory for results (default: ``benchmark_results/``).
--download         Attempt to download real datasets before benchmarking.

Outputs
-------
* ``results.json`` — machine-readable metric export.
* ``BENCHMARKS.md`` — human-readable Markdown report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from teloscopy.data.benchmarks import BenchmarkSuite  # noqa: E402
from teloscopy.data.datasets import DatasetManager  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_benchmarks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Teloscopy benchmarks")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for results (default: benchmark_results/)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download real datasets before benchmarking",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.download:
        logger.info("Downloading real datasets ...")
        dm = DatasetManager()
        for name in ["spotiflow_fish", "clinvar_variants", "nhanes_telomere", "bbbc039"]:
            try:
                dm.download(name)
            except Exception as exc:
                logger.warning("Could not download %s: %s", name, exc)

    logger.info("Running benchmark suite ...")
    suite = BenchmarkSuite(output_dir=str(out))
    report = suite.run_all()

    results_path = str(out / "results.json")
    report_path = str(out / "BENCHMARKS.md")

    suite.export_results(report, results_path)
    md = suite.generate_report(report)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    logger.info("Results saved to %s", results_path)
    logger.info("Report  saved to %s", report_path)
    logger.info("Overall score: %.4f", report.overall_score)


if __name__ == "__main__":
    main()
