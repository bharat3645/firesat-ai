"""Spectral vegetation/burn indices computed from raw optical bands.

These are the standard remote-sensing formulas used against real Sentinel-2 /
Landsat reflectance bands once acquired via ``firesat.data.gee_client``. Pure
numpy, no I/O, fully unit-testable.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index, range approx [-1, 1].

    Higher values indicate denser, healthier live vegetation; low/negative
    values indicate bare ground, water, snow, or senesced/burned vegetation.
    """
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    return (nir - red) / (nir + red + _EPS)


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio. Sharp drops between two dates indicate burned area."""
    nir = np.asarray(nir, dtype=np.float64)
    swir = np.asarray(swir, dtype=np.float64)
    return (nir - swir) / (nir + swir + _EPS)


def dnbr(pre_fire_nbr: np.ndarray, post_fire_nbr: np.ndarray) -> np.ndarray:
    """Delta NBR: pre-fire minus post-fire. Positive values indicate burn severity."""
    return np.asarray(pre_fire_nbr, dtype=np.float64) - np.asarray(post_fire_nbr, dtype=np.float64)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index -- optical proxy for canopy water content."""
    nir = np.asarray(nir, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)
    return (nir - swir1) / (nir + swir1 + _EPS)


def evi(
    nir: np.ndarray, red: np.ndarray, blue: np.ndarray, g: float = 2.5, c1: float = 6.0, c2: float = 7.5, canopy_bg: float = 1.0
) -> np.ndarray:
    """Enhanced Vegetation Index -- less saturating than NDVI in dense canopy."""
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    blue = np.asarray(blue, dtype=np.float64)
    return g * (nir - red) / (nir + c1 * red - c2 * blue + canopy_bg + _EPS)
