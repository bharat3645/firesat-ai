#!/usr/bin/env python
"""Generate the synthetic multi-year monthly dataset for all study regions.

This is the offline stand-in for the live acquisition pipeline (Earth
Engine + ERA5 + FIRMS + Alaska Fire Service). Run it first; ``scripts/
train_demo.py`` and the FastAPI backend both expect its output under
``data/processed/``.

Usage:
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --start-year 2010 --end-year 2024 --seed 7
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesat.config import PROCESSED_DATA_DIR  # noqa: E402
from firesat.data.pipeline import run_synthetic_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Base RNG seed (region i gets seed+i). Default 5 is calibrated to "
        "produce a reasonable, non-degenerate number of fire events in both "
        "study regions over the default 2015-2024 window.",
    )
    parser.add_argument("--out-dir", type=str, default=str(PROCESSED_DATA_DIR))
    args = parser.parse_args()

    datasets = run_synthetic_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    for region_id, ds in datasets.items():
        n_fires = int(ds.ignition_indicator.sum())
        logger.info(
            "region=%s months=%d spatial_shape=%s fire_events=%d total_acres=%.0f",
            region_id,
            len(ds.times),
            ds.spatial.shape,
            n_fires,
            float(ds.ignition_severity.sum()),
        )
    logger.info("Done. Data written to %s", args.out_dir)


if __name__ == "__main__":
    main()
