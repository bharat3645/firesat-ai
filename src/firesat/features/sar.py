"""SAR (Sentinel-1) backscatter-derived features.

C-band SAR is largely unaffected by cloud cover -- valuable in Alaska where
optical imagery is often unusable for weeks at a time -- and VV/VH
backscatter is sensitive to both surface/soil moisture and vegetation
structure/water content, both of which modulate fuel moisture.
"""
from __future__ import annotations

import numpy as np


def radar_vegetation_index(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """Dual-pol Radar Vegetation Index (RVI) computed from dB backscatter.

    Converts from dB to linear power first (RVI is only meaningful in linear
    units), then applies the standard dual-pol formulation
    ``RVI = 4*VH / (VV + VH)``. Higher RVI indicates more volume scattering
    (denser/wetter canopy); lower RVI indicates senesced/sparse vegetation.
    """
    vv_lin = _db_to_linear(vv_db)
    vh_lin = _db_to_linear(vh_db)
    return 4.0 * vh_lin / (vv_lin + vh_lin + 1e-12)


def cross_pol_ratio(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """VH/VV ratio in dB (simple difference), a coarse vegetation-density proxy."""
    return np.asarray(vh_db, dtype=np.float64) - np.asarray(vv_db, dtype=np.float64)


def soil_moisture_proxy(vv_db: np.ndarray, vv_dry_reference_db: float) -> np.ndarray:
    """Very coarse change-detection soil-moisture proxy.

    Backscatter increases with surface dielectric constant (i.e., moisture),
    holding roughness/vegetation roughly constant. This expresses the
    current VV backscatter as an offset (dB) above a dry-reference baseline
    (e.g., a rolling percentile of VV over the driest observed months for a
    given cell) -- a standard, if simplified, SAR soil-moisture heuristic.
    """
    vv_db = np.asarray(vv_db, dtype=np.float64)
    return vv_db - vv_dry_reference_db


def _db_to_linear(x_db: np.ndarray) -> np.ndarray:
    return 10.0 ** (np.asarray(x_db, dtype=np.float64) / 10.0)


def _linear_to_db(x_linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(np.asarray(x_linear, dtype=np.float64), 1e-12, None))
