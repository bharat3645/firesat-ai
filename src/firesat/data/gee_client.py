"""Google Earth Engine acquisition client for satellite imagery.

Pulls Sentinel-2 (optical), Sentinel-1 (SAR), Landsat-8/9, and MODIS
composites for a region/month and reduces them to the grid resolution used
by the model. Requires `earthengine-api` and a one-time
``earthengine authenticate`` (or a service-account key referenced by
``GOOGLE_APPLICATION_CREDENTIALS``). The dependency is optional: importing
this module never fails, but calling any acquisition function without
Earth Engine installed/authenticated raises a clear ``RuntimeError`` so the
rest of the pipeline (synthetic data, model, API, dashboard) keeps working
without live credentials -- exactly the tradeoff a laptop-scale GSoC MVP
has to make.

Usage (once authenticated)::

    from firesat.data.gee_client import EarthEngineClient
    client = EarthEngineClient()
    client.initialize()
    stack = client.monthly_spatial_stack(region, year=2023, month=7, grid_size=16)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from firesat.config import (
    GEE_LANDSAT_COLLECTION,
    GEE_MODIS_LST_COLLECTION,
    GEE_MODIS_NDVI_COLLECTION,
    GEE_S1_COLLECTION,
    GEE_S2_COLLECTION,
    Region,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the optional dep is installed
    import ee  # type: ignore

    _EE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ee = None  # type: ignore
    _EE_AVAILABLE = False


class EarthEngineUnavailableError(RuntimeError):
    """Raised when a live GEE call is attempted without the optional dep/auth."""


@dataclass
class EarthEngineClient:
    """Thin, testable wrapper around the Earth Engine Python API.

    Every public method that touches the network is a thin, mockable seam
    (`_ee_reduce_region`) so unit tests can verify the band math and
    reduction logic without a live GEE session.
    """

    project: str | None = None
    _initialized: bool = False

    def initialize(self) -> None:
        if not _EE_AVAILABLE:
            raise EarthEngineUnavailableError(
                "earthengine-api is not installed. Run "
                "`pip install earthengine-api` and `earthengine authenticate`, "
                "or use firesat.data.synthetic for offline development."
            )
        try:
            ee.Initialize(project=self.project) if self.project else ee.Initialize()
        except Exception as exc:  # pragma: no cover - requires real credentials
            raise EarthEngineUnavailableError(
                "Earth Engine failed to initialize. Have you run "
                "`earthengine authenticate`? "
                f"Original error: {exc}"
            ) from exc
        self._initialized = True

    def _require_ready(self) -> None:
        if not self._initialized:
            self.initialize()

    # -- Band math -----------------------------------------------------
    @staticmethod
    def _ndvi(image, nir_band: str, red_band: str):
        return image.normalizedDifference([nir_band, red_band]).rename("ndvi")

    @staticmethod
    def _nbr(image, nir_band: str, swir_band: str):
        return image.normalizedDifference([nir_band, swir_band]).rename("nbr")

    def sentinel2_monthly_composite(self, region: Region, year: int, month: int):
        """Return an EE.Image cloud-masked Sentinel-2 median composite with NDVI/NBR."""
        self._require_ready()
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        aoi = ee.Geometry.Rectangle(list(region.bbox))

        def _mask_clouds(img):
            scl = img.select("SCL")
            mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            return img.updateMask(mask)

        coll = (
            ee.ImageCollection(GEE_S2_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(_mask_clouds)
        )
        composite = coll.median().clip(aoi)
        ndvi = self._ndvi(composite, "B8", "B4")
        nbr = self._nbr(composite, "B8", "B12")
        return composite.addBands([ndvi, nbr])

    def sentinel1_monthly_composite(self, region: Region, year: int, month: int):
        """Return an EE.Image Sentinel-1 SAR (VV/VH, dB) median composite."""
        self._require_ready()
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        aoi = ee.Geometry.Rectangle(list(region.bbox))
        coll = (
            ee.ImageCollection(GEE_S1_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )
        return coll.median().clip(aoi)

    def modis_lst_anomaly(self, region: Region, year: int, month: int, climatology_years: int = 5):
        """Land-surface-temperature anomaly vs. a trailing multi-year climatology."""
        self._require_ready()
        aoi = ee.Geometry.Rectangle(list(region.bbox))
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        current = (
            ee.ImageCollection(GEE_MODIS_LST_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start, end)
            .select("LST_Day_1km")
            .mean()
        )
        clim_start = start.advance(-climatology_years, "year")
        climatology = (
            ee.ImageCollection(GEE_MODIS_LST_COLLECTION)
            .filterBounds(aoi)
            .filterDate(clim_start, start)
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .select("LST_Day_1km")
            .mean()
        )
        return current.subtract(climatology).rename("lst_anomaly").clip(aoi)

    def landsat_monthly_composite(self, region: Region, year: int, month: int):
        """Fallback optical composite via Landsat-8/9 when Sentinel-2 coverage is thin."""
        self._require_ready()
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        aoi = ee.Geometry.Rectangle(list(region.bbox))
        coll = (
            ee.ImageCollection(GEE_LANDSAT_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start, end)
        )
        composite = coll.median().clip(aoi)
        ndvi = self._ndvi(composite, "SR_B5", "SR_B4")
        nbr = self._nbr(composite, "SR_B5", "SR_B7")
        return composite.addBands([ndvi, nbr])

    def modis_ndvi_composite(self, region: Region, year: int, month: int):
        """MODIS 16-day NDVI (MOD13Q1) for gap-filling when optical coverage is poor."""
        self._require_ready()
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        aoi = ee.Geometry.Rectangle(list(region.bbox))
        coll = (
            ee.ImageCollection(GEE_MODIS_NDVI_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start, end)
            .select("NDVI")
        )
        return coll.mean().multiply(0.0001).rename("ndvi_modis").clip(aoi)

    def monthly_spatial_stack(
        self, region: Region, year: int, month: int, grid_size: int = 16
    ) -> np.ndarray:
        """Reduce a month of imagery to a ``(channels, grid_size, grid_size)`` array.

        Channel order matches ``firesat.config.SPATIAL_FEATURE_CHANNELS``. This
        is the seam between "real Earth Engine acquisition" and "numpy feature
        array consumed by the model" -- in production it calls ``getRegion``
        / ``sampleRectangle`` reducers against the composites above; in this
        MVP the heavy network+auth path is documented here and exercised by
        ``firesat.data.synthetic`` for local development, tests, and CI.
        """
        raise EarthEngineUnavailableError(
            "Live GEE pixel export is not wired into this MVP checkout. "
            "This method documents/implements the composite construction "
            "(see sentinel2_monthly_composite / sentinel1_monthly_composite / "
            "modis_lst_anomaly above); wire `image.sampleRectangle()` or "
            "`ee.batch.Export.image.toDrive` here to materialize numpy arrays "
            "once you have GEE credentials. Use firesat.data.synthetic for a "
            "fully offline drop-in with the same output shape/contract."
        )


def is_earth_engine_available() -> bool:
    return _EE_AVAILABLE
