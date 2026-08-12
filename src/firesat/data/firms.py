"""NASA FIRMS active-fire detections client, used both as a supplementary
feature source and as ground truth for backtesting model predictions.

FIRMS exposes a free (rate-limited) CSV API keyed by a MAP_KEY that any user
can request at https://firms.modaps.eosdis.gov/api/area/. Network access is
optional here: :func:`fetch_area_csv` performs the real HTTP call when
``httpx`` is available and a key is provided; :func:`load_local_csv` and
``firesat.data.synthetic`` cover offline development and CI.
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

import pandas as pd

from firesat.config import NASA_FIRMS_AREA_URL, Region

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False

FIRMS_DATE_FMT = "%Y-%m-%d"
FIRMS_SOURCE = "VIIRS_SNPP_NRT"


class FIRMSUnavailableError(RuntimeError):
    """Raised when a live FIRMS fetch is attempted without deps/credentials."""


@dataclass
class FIRMSClient:
    """Fetches and parses NASA FIRMS active-fire-detection CSVs."""

    map_key: str | None = None

    def __post_init__(self) -> None:
        self.map_key = self.map_key or os.environ.get("FIRMS_MAP_KEY")

    def fetch_area_csv(
        self, region: Region, day_range: int = 10, source: str = FIRMS_SOURCE
    ) -> pd.DataFrame:
        """Fetch recent active-fire detections for a region's bounding box.

        ``day_range`` is capped at 10 by the FIRMS API for the free tier.
        """
        if not _HTTPX_AVAILABLE:
            raise FIRMSUnavailableError(
                "httpx is not installed; cannot fetch FIRMS data. "
                "Install httpx or use firesat.data.synthetic instead."
            )
        if not self.map_key:
            raise FIRMSUnavailableError(
                "No FIRMS_MAP_KEY set. Request a free key at "
                "https://firms.modaps.eosdis.gov/api/area/ and set the "
                "FIRMS_MAP_KEY environment variable."
            )
        min_lon, min_lat, max_lon, max_lat = region.bbox
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        url = f"{NASA_FIRMS_AREA_URL}/{self.map_key}/{source}/{bbox_str}/{day_range}"
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        return self.parse_csv(response.text)

    @staticmethod
    def parse_csv(csv_text: str) -> pd.DataFrame:
        df = pd.read_csv(io.StringIO(csv_text))
        if "acq_date" in df.columns:
            df["acq_date"] = pd.to_datetime(df["acq_date"])
        return df

    @staticmethod
    def load_local_csv(path: str) -> pd.DataFrame:
        """Load a previously downloaded/cached FIRMS CSV from disk."""
        return FIRMSClient.parse_csv(open(path, encoding="utf-8").read())

    @staticmethod
    def monthly_detection_counts(detections: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw point detections into a (year, month, count, mean_frp) table."""
        if detections.empty:
            return pd.DataFrame(columns=["year", "month", "detection_count", "mean_frp"])
        df = detections.copy()
        df["year"] = df["acq_date"].dt.year
        df["month"] = df["acq_date"].dt.month
        agg = df.groupby(["year", "month"]).agg(
            detection_count=("acq_date", "count"),
            mean_frp=("frp", "mean") if "frp" in df.columns else ("acq_date", "count"),
        )
        return agg.reset_index()


def is_firms_client_available() -> bool:
    return _HTTPX_AVAILABLE
