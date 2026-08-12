"""Assembles raw band/index arrays into the model's canonical feature tensor
and provides the normalization used consistently across training and
inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from firesat.config import SPATIAL_FEATURE_CHANNELS, WEATHER_FEATURE_COLUMNS
from firesat.features.indices import nbr as compute_nbr
from firesat.features.indices import ndvi as compute_ndvi
from firesat.features.sar import cross_pol_ratio
from firesat.features.weather import fuel_moisture_index


def spatial_stack_from_bands(
    nir: np.ndarray,
    red: np.ndarray,
    swir: np.ndarray,
    sar_vv: np.ndarray,
    sar_vh: np.ndarray,
    lst_anomaly: np.ndarray,
    humidity_pct: float,
) -> np.ndarray:
    """Build a ``(6, H, W)`` array matching ``SPATIAL_FEATURE_CHANNELS`` order
    from raw optical bands + SAR + thermal anomaly -- the seam a real
    Earth-Engine-backed pipeline would call after ``gee_client`` exports
    per-band arrays.
    """
    ndvi = compute_ndvi(nir, red)
    nbr = compute_nbr(nir, swir)
    fuel_moisture = fuel_moisture_index(
        temp_c=np.zeros_like(ndvi),  # temperature term folded in separately via weather features
        relative_humidity_pct=np.full_like(ndvi, humidity_pct),
        precipitation_mm=np.zeros_like(ndvi),
    )
    # Blend the optical fuel-moisture proxy with a SAR cross-pol signal so the
    # channel reflects both vegetation water content and canopy structure.
    xpol = cross_pol_ratio(sar_vv, sar_vh)
    fuel_moisture_proxy = np.clip(0.7 * fuel_moisture + 0.3 * _minmax(xpol), 0.0, 1.0)

    stack = np.stack(
        [
            ndvi,
            nbr,
            np.asarray(lst_anomaly, dtype=np.float64),
            np.asarray(sar_vv, dtype=np.float64),
            np.asarray(sar_vh, dtype=np.float64),
            fuel_moisture_proxy,
        ],
        axis=0,
    ).astype(np.float32)
    assert stack.shape[0] == len(SPATIAL_FEATURE_CHANNELS)
    return stack


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2, 98])
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


@dataclass
class NormalizationStats:
    """Per-channel mean/std for spatial features and per-column mean/std for
    weather features, fit on a training split and reused at inference time.
    """

    spatial_mean: np.ndarray  # (C,)
    spatial_std: np.ndarray  # (C,)
    weather_mean: np.ndarray  # (F,)
    weather_std: np.ndarray  # (F,)

    def to_dict(self) -> dict:
        return {
            "spatial_mean": self.spatial_mean.tolist(),
            "spatial_std": self.spatial_std.tolist(),
            "weather_mean": self.weather_mean.tolist(),
            "weather_std": self.weather_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationStats":
        return cls(
            spatial_mean=np.array(d["spatial_mean"], dtype=np.float32),
            spatial_std=np.array(d["spatial_std"], dtype=np.float32),
            weather_mean=np.array(d["weather_mean"], dtype=np.float32),
            weather_std=np.array(d["weather_std"], dtype=np.float32),
        )


def fit_normalization_stats(
    spatial: np.ndarray, weather: np.ndarray
) -> NormalizationStats:
    """Fit per-channel/per-column mean/std.

    ``spatial``: ``(T, C, H, W)``. ``weather``: ``(T, F)``.
    """
    spatial_mean = spatial.mean(axis=(0, 2, 3))
    spatial_std = spatial.std(axis=(0, 2, 3)) + 1e-6
    weather_mean = weather.mean(axis=0)
    weather_std = weather.std(axis=0) + 1e-6
    return NormalizationStats(
        spatial_mean=spatial_mean.astype(np.float32),
        spatial_std=spatial_std.astype(np.float32),
        weather_mean=weather_mean.astype(np.float32),
        weather_std=weather_std.astype(np.float32),
    )


def apply_normalization(
    spatial: np.ndarray, weather: np.ndarray, stats: NormalizationStats
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fitted stats to (possibly new) spatial/weather arrays."""
    norm_spatial = (spatial - stats.spatial_mean[None, :, None, None]) / stats.spatial_std[
        None, :, None, None
    ]
    norm_weather = (weather - stats.weather_mean[None, :]) / stats.weather_std[None, :]
    return norm_spatial.astype(np.float32), norm_weather.astype(np.float32)


assert len(WEATHER_FEATURE_COLUMNS) > 0  # sanity: config wiring
