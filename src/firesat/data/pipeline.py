"""Orchestrates the data pipeline: acquisition -> feature assembly -> disk.

Two modes:

- ``source="synthetic"`` (default, no credentials needed): uses
  ``firesat.data.synthetic`` to generate a physically-motivated stand-in
  dataset, so the whole repo runs end-to-end offline.
- ``source="live"``: acquires from Earth Engine / ERA5 / FIRMS / AK Fire
  Service via the other modules in this package. Requires the relevant
  credentials (GEE auth, CDS API key, FIRMS map key) -- see each module's
  docstring. Left as an explicit extension point (``_run_live_pipeline``)
  since wiring real per-pixel GEE exports is inherently an
  internet+credentials-dependent integration step outside a sandboxed MVP
  build, and the goal here is a pipeline whose *contract* (shapes, column
  names, channel order) is validated by the synthetic path today and does
  not need to change when the live acquisition calls are dropped in.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from firesat.config import PROCESSED_DATA_DIR, REGIONS, WEATHER_FEATURE_COLUMNS
from firesat.data.synthetic import SyntheticRegionDataset, generate_all_regions

logger = logging.getLogger(__name__)


def run_synthetic_pipeline(
    start_year: int = 2015,
    end_year: int = 2024,
    seed: int | None = None,
    out_dir: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, SyntheticRegionDataset]:
    """Generate the synthetic dataset for all configured regions and persist
    it to ``out_dir`` in the same layout ``load_processed_region`` expects.
    """
    datasets = generate_all_regions(REGIONS, start_year=start_year, end_year=end_year, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for region_id, ds in datasets.items():
        save_region_dataset(ds, out_dir)
    logger.info("Synthetic pipeline wrote %d regions to %s", len(datasets), out_dir)
    return datasets


def save_region_dataset(ds: SyntheticRegionDataset, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    region_dir = out_dir / ds.region.id
    region_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        region_dir / "spatial.npz",
        spatial=ds.spatial,
        labels=ds.labels,
        ignition_indicator=ds.ignition_indicator,
        ignition_severity=ds.ignition_severity,
    )
    ds.weather.to_parquet(region_dir / "weather.parquet", index=False)
    ds.firms_detections.to_parquet(region_dir / "firms.parquet", index=False)
    ds.perimeters.to_parquet(region_dir / "perimeters.parquet", index=False)
    with open(region_dir / "times.json", "w", encoding="utf-8") as fh:
        json.dump([list(t) for t in ds.times], fh)
    with open(region_dir / "region.json", "w", encoding="utf-8") as fh:
        json.dump(ds.region.__dict__, fh)


def load_processed_region(region_id: str, in_dir: str | Path = PROCESSED_DATA_DIR) -> SyntheticRegionDataset:
    from firesat.config import Region

    region_dir = Path(in_dir) / region_id
    if not region_dir.exists():
        raise FileNotFoundError(
            f"No processed data found for region '{region_id}' at {region_dir}. "
            "Run `python scripts/generate_demo_data.py` first."
        )
    npz = np.load(region_dir / "spatial.npz")
    weather = pd.read_parquet(region_dir / "weather.parquet")
    firms = pd.read_parquet(region_dir / "firms.parquet")
    perimeters = pd.read_parquet(region_dir / "perimeters.parquet")
    times = [tuple(t) for t in json.loads((region_dir / "times.json").read_text(encoding="utf-8"))]
    region_dict = json.loads((region_dir / "region.json").read_text(encoding="utf-8"))
    region = Region(**region_dict)

    assert set(WEATHER_FEATURE_COLUMNS) <= set(weather.columns)

    return SyntheticRegionDataset(
        region=region,
        times=times,
        spatial=npz["spatial"],
        weather=weather,
        ignition_indicator=npz["ignition_indicator"],
        ignition_severity=npz["ignition_severity"],
        labels=npz["labels"],
        firms_detections=firms,
        perimeters=perimeters,
    )


def load_all_processed_regions(in_dir: str | Path = PROCESSED_DATA_DIR) -> dict[str, SyntheticRegionDataset]:
    return {region_id: load_processed_region(region_id, in_dir) for region_id in REGIONS}


def _run_live_pipeline(*args, **kwargs):  # pragma: no cover - documented extension point
    raise NotImplementedError(
        "Live acquisition requires GEE/CDS/FIRMS credentials not available in "
        "this environment. Wire firesat.data.gee_client.EarthEngineClient, "
        "firesat.data.era5.ERA5Client, firesat.data.firms.FIRMSClient, and "
        "firesat.data.perimeters.PerimeterLoader here -- each already "
        "implements the real request/band-math logic; this function is the "
        "single integration point that assembles their outputs into the same "
        "SyntheticRegionDataset-shaped contract produced by "
        "run_synthetic_pipeline(), so nothing downstream (features, model, "
        "training, API) needs to change."
    )
