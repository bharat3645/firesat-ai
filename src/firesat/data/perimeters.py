"""Alaska Interagency Coordination Center (AICC) / Alaska Fire Service
historical fire perimeter loader, used as ground truth for backtesting.

AICC publishes historical fire perimeters as shapefiles/GeoJSON at
https://fire.ak.blm.gov. This module loads a local (downloaded) copy -- it
does not perform the download itself, since perimeter archives are large
and licensing/attribution terms are best handled by the user pulling them
directly from AICC. Geometry parsing uses ``shapely`` (a light dependency,
already required by the rest of the stack) rather than ``geopandas``/GDAL so
this module has no heavy native-binary dependency.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from shapely.geometry import shape

from firesat.config import Region

logger = logging.getLogger(__name__)


@dataclass
class FirePerimeterRecord:
    fire_name: str
    year: int
    ignition_date: str | None
    acres: float
    geometry: object  # shapely geometry


class PerimeterLoader:
    """Loads historical fire perimeters from a local GeoJSON export."""

    def load_geojson(self, path: str | Path) -> list[FirePerimeterRecord]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download historical perimeters from "
                "the Alaska Fire Service / AICC open data portal and place "
                "the GeoJSON export here, or use "
                "firesat.data.synthetic.generate_synthetic_perimeters for "
                "offline development."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            records.append(
                FirePerimeterRecord(
                    fire_name=props.get("FIRENAME", props.get("fire_name", "UNKNOWN")),
                    year=int(props.get("FIREYEAR", props.get("year", 0)) or 0),
                    ignition_date=props.get("IGNITIONDATE", props.get("ignition_date")),
                    acres=float(props.get("ACRES", props.get("acres", 0.0)) or 0.0),
                    geometry=shape(feature["geometry"]) if feature.get("geometry") else None,
                )
            )
        return records

    @staticmethod
    def filter_by_region(
        records: list[FirePerimeterRecord], region: Region
    ) -> list[FirePerimeterRecord]:
        from shapely.geometry import box

        aoi = box(*region.bbox)
        return [r for r in records if r.geometry is not None and r.geometry.intersects(aoi)]

    @staticmethod
    def to_monthly_burned_area(
        records: list[FirePerimeterRecord],
    ) -> pd.DataFrame:
        """Approximate monthly burned-acreage table from ignition dates.

        Real perimeter exports usually only carry a fire-season year, not a
        precise month; when ``ignition_date`` is missing this falls back to
        attributing the whole fire to July (the historical peak of the
        Alaska fire season), which is an explicit, documented approximation
        -- flagged again in docs/evaluation_report.md.
        """
        rows = []
        for r in records:
            month = 7
            if r.ignition_date:
                try:
                    month = pd.to_datetime(r.ignition_date).month
                except (ValueError, TypeError):
                    pass
            rows.append({"year": r.year, "month": month, "acres": r.acres, "fire_name": r.fire_name})
        if not rows:
            return pd.DataFrame(columns=["year", "month", "acres_burned", "fire_count"])
        df = pd.DataFrame(rows)
        agg = df.groupby(["year", "month"]).agg(
            acres_burned=("acres", "sum"), fire_count=("fire_name", "count")
        )
        return agg.reset_index()
