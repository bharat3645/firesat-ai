"""ERA5-Land monthly reanalysis client (temperature, humidity, wind, precip).

Two acquisition paths are supported:

1. **Copernicus Climate Data Store** via the optional ``cdsapi`` package
   (requires a free CDS account + API key in ``~/.cdsapirc``).
2. **Google Earth Engine's ``ECMWF/ERA5_LAND/MONTHLY_AGGR`` mirror**, reusing
   the same ``EarthEngineClient`` credentials -- often the more convenient
   path since it needs no separate CDS account.

Both are optional/guarded so the rest of the pipeline works offline; see
``firesat.data.synthetic`` for the local-development stand-in used by tests,
demos, and CI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from firesat.config import ERA5_COLLECTION, Region, WEATHER_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import cdsapi  # type: ignore

    _CDS_AVAILABLE = True
except ImportError:  # pragma: no cover
    cdsapi = None  # type: ignore
    _CDS_AVAILABLE = False


class ERA5UnavailableError(RuntimeError):
    """Raised when a live ERA5 fetch is attempted without credentials/deps."""


@dataclass
class ERA5Client:
    """Fetches monthly-aggregated ERA5-Land weather variables for a region."""

    def fetch_via_cds(
        self, region: Region, year: int, months: list[int], out_dir: str
    ) -> str:
        """Download monthly ERA5-Land NetCDF via the Copernicus CDS API.

        Returns the path to the downloaded file. Requires ``cdsapi`` and a
        configured ``~/.cdsapirc``.
        """
        if not _CDS_AVAILABLE:
            raise ERA5UnavailableError(
                "cdsapi is not installed. Run `pip install cdsapi` and "
                "configure ~/.cdsapirc with your CDS API key, or use "
                "firesat.data.synthetic for offline development."
            )
        client = cdsapi.Client()
        out_path = f"{out_dir}/era5_{region.id}_{year}.nc"
        north, west, south, east = (
            region.max_lat,
            region.min_lon,
            region.min_lat,
            region.max_lon,
        )
        client.retrieve(
            "reanalysis-era5-land-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": [
                    "2m_temperature",
                    "2m_dewpoint_temperature",  # used to derive relative humidity
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "total_precipitation",
                ],
                "year": str(year),
                "month": [f"{m:02d}" for m in months],
                "time": "00:00",
                "area": [north, west, south, east],
                "format": "netcdf",
            },
            out_path,
        )
        return out_path

    def fetch_via_earth_engine(
        self, region: Region, year: int, months: list[int]
    ) -> pd.DataFrame:
        """Fetch region-mean ERA5-Land monthly aggregates through Earth Engine.

        Reuses ``ECMWF/ERA5_LAND/MONTHLY_AGGR``; requires an initialized
        ``firesat.data.gee_client.EarthEngineClient``.
        """
        from firesat.data.gee_client import EarthEngineClient, is_earth_engine_available

        if not is_earth_engine_available():
            raise ERA5UnavailableError(
                "earthengine-api is not installed; cannot fetch ERA5 via GEE. "
                "Use fetch_via_cds() or firesat.data.synthetic instead."
            )
        import ee  # type: ignore

        client = EarthEngineClient()
        client.initialize()
        aoi = ee.Geometry.Rectangle(list(region.bbox))
        rows = []
        for month in months:
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, "month")
            img = (
                ee.ImageCollection(ERA5_COLLECTION)
                .filterDate(start, end)
                .filterBounds(aoi)
                .first()
            )
            stats = img.select(
                [
                    "temperature_2m",
                    "dewpoint_temperature_2m",
                    "u_component_of_wind_10m",
                    "v_component_of_wind_10m",
                    "total_precipitation_sum",
                ]
            ).reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=9000, maxPixels=1e9)
            info = stats.getInfo()
            rows.append({"region_id": region.id, "year": year, "month": month, **info})
        return pd.DataFrame(rows)

    @staticmethod
    def to_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Derive the model-facing WEATHER_FEATURE_COLUMNS from raw ERA5 fields."""
        out = pd.DataFrame(index=df.index)
        if "temperature_2m" in df:
            out["temp_2m_c"] = df["temperature_2m"] - 273.15
        if {"temperature_2m", "dewpoint_temperature_2m"} <= set(df.columns):
            out["relative_humidity_pct"] = _relative_humidity_from_dewpoint(
                df["temperature_2m"] - 273.15, df["dewpoint_temperature_2m"] - 273.15
            )
        if {"u_component_of_wind_10m", "v_component_of_wind_10m"} <= set(df.columns):
            out["wind_speed_ms"] = (
                df["u_component_of_wind_10m"] ** 2 + df["v_component_of_wind_10m"] ** 2
            ) ** 0.5
        if "total_precipitation_sum" in df:
            out["precipitation_mm"] = df["total_precipitation_sum"] * 1000.0
        missing = set(WEATHER_FEATURE_COLUMNS) - set(out.columns)
        if missing:
            logger.warning("ERA5 derived frame missing columns: %s", missing)
        return out


def _relative_humidity_from_dewpoint(temp_c, dewpoint_c):
    """Magnus-formula approximation of relative humidity from T and Td (deg C)."""
    a, b = 17.625, 243.04
    numerator = np.exp((a * dewpoint_c) / (b + dewpoint_c))
    denominator = np.exp((a * temp_c) / (b + temp_c))
    return 100.0 * numerator / denominator


def is_cds_available() -> bool:
    return _CDS_AVAILABLE
