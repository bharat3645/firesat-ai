"""Central configuration: study regions, feature schema, model & path constants.

Scope note: per the proposal's "honest evaluation" framing, FireSat-AI is
deliberately scoped to two well-studied Alaska regions rather than the whole
state, so the pipeline, model, and evaluation stay tractable and reproducible
on a laptop/CI runner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Region:
    """A study region: bounding box in WGS84 lon/lat degrees."""

    id: str
    name: str
    description: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.min_lon + self.max_lon) / 2, (self.min_lat + self.max_lat) / 2)

    def polygon_geojson(self) -> dict:
        """Return the bbox as a GeoJSON Polygon feature geometry."""
        lo_x, lo_y, hi_x, hi_y = self.bbox
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [lo_x, lo_y],
                    [hi_x, lo_y],
                    [hi_x, hi_y],
                    [lo_x, hi_y],
                    [lo_x, lo_y],
                ]
            ],
        }


# The proposal explicitly scopes down to "one or two well-studied regions"
# rather than all of Alaska. These two are chosen because they are
# fire-active, well-instrumented, and frequently used in Alaska wildfire
# literature: the boreal Interior (Fairbanks / Yukon-Tanana Uplands) and the
# beetle-killed spruce forests of the Kenai Peninsula.
REGIONS: dict[str, Region] = {
    "interior-fairbanks": Region(
        id="interior-fairbanks",
        name="Interior Alaska (Fairbanks / Yukon-Tanana Uplands)",
        description=(
            "Boreal black-spruce dominated interior, one of the most "
            "fire-active landscapes in North America."
        ),
        min_lon=-148.5,
        min_lat=64.3,
        max_lon=-146.0,
        max_lat=65.3,
    ),
    "kenai-peninsula": Region(
        id="kenai-peninsula",
        name="Kenai Peninsula",
        description=(
            "Spruce-beetle-killed forest with elevated fuel load and "
            "wildland-urban interface fire risk."
        ),
        min_lon=-151.5,
        min_lat=60.0,
        max_lon=-149.0,
        max_lat=61.0,
    ),
}

DEFAULT_REGION_ORDER = ["interior-fairbanks", "kenai-peninsula"]

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# Spatial (per-pixel / per-grid-cell) channels produced by the feature
# pipeline for every monthly composite. Order matters: it defines channel
# indices consumed by the CNN encoder.
SPATIAL_FEATURE_CHANNELS: list[str] = [
    "ndvi",  # vegetation index (Sentinel-2 / Landsat / MODIS)
    "nbr",  # normalized burn ratio
    "lst_anomaly",  # thermal anomaly vs. climatological mean (MODIS/VIIRS)
    "sar_vv",  # Sentinel-1 SAR VV backscatter (moisture proxy)
    "sar_vh",  # Sentinel-1 SAR VH backscatter (vegetation structure)
    "fuel_moisture_proxy",  # derived from NDVI + humidity + SAR
]

# Scalar weather/reanalysis features appended to the temporal sequence
# (ERA5-derived, one value per region per month).
WEATHER_FEATURE_COLUMNS: list[str] = [
    "temp_2m_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
]

RISK_CLASSES: list[str] = ["No Risk", "Moderate", "High"]
HORIZONS_MONTHS: list[int] = [1, 3, 6]

GRID_SIZE = 16  # spatial patch is GRID_SIZE x GRID_SIZE cells per region
SEQUENCE_LENGTH = 24  # months of lookback fed to the temporal branch

N_SPATIAL_CHANNELS = len(SPATIAL_FEATURE_CHANNELS)
N_WEATHER_FEATURES = len(WEATHER_FEATURE_COLUMNS)
N_CLASSES = len(RISK_CLASSES)
N_HORIZONS = len(HORIZONS_MONTHS)

# ---------------------------------------------------------------------------
# External data source endpoints / identifiers (documented, not secrets)
# ---------------------------------------------------------------------------
GEE_S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
GEE_S1_COLLECTION = "COPERNICUS/S1_GRD"
GEE_LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
GEE_MODIS_NDVI_COLLECTION = "MODIS/061/MOD13Q1"
GEE_MODIS_LST_COLLECTION = "MODIS/061/MOD11A2"
GEE_VIIRS_FIRE_COLLECTION = "FIRMS"
ERA5_COLLECTION = "ECMWF/ERA5_LAND/MONTHLY_AGGR"

NASA_FIRMS_AREA_URL = "https://firms.modaps.eosdis.gov/api/area/csv"
ALASKA_FIRE_SERVICE_PERIMETERS_URL = (
    "https://fire.ak.blm.gov/content/maps/aicc/Data/Data%20(zipped%20Shapefiles)/"
)

RANDOM_SEED = int(os.environ.get("FIRESAT_SEED", "42"))
