"""Physically-motivated synthetic data generator.

This is **not** satellite data. It generates plausible seasonal cycles for
Alaska (temperature, humidity, NDVI green-up/senescence, snow-driven winter
NBR/NDVI collapse) plus a stochastic wildfire-ignition process driven by a
"dryness score" derived from those same synthetic drivers, so precursor
signal genuinely exists in the features -- the same way it would need to in
real Sentinel/MODIS/ERA5 data for the model to be learnable at all.

Its job is to let the *entire* pipeline (feature engineering -> CNN-LSTM
model -> training -> evaluation -> API -> dashboard) run end-to-end on a
laptop or in CI with zero external credentials. Swap it for
``firesat.data.gee_client`` + ``firesat.data.era5`` + ``firesat.data.firms``
+ ``firesat.data.perimeters`` once you have Earth Engine / CDS / FIRMS
credentials -- every downstream module (features, model, dataset, API)
consumes the same array/DataFrame contract either way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from firesat.config import (
    GRID_SIZE,
    HORIZONS_MONTHS,
    N_HORIZONS,
    RANDOM_SEED,
    Region,
    SPATIAL_FEATURE_CHANNELS,
)

try:
    from scipy.ndimage import gaussian_filter

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def _smooth_field(grid_size: int, rng: np.random.Generator, sigma: float = 2.5) -> np.ndarray:
    """A smooth, spatially-correlated random field in [0, 1] (proxy for terrain/aspect)."""
    noise = rng.normal(size=(grid_size, grid_size))
    if _HAS_SCIPY:
        smoothed = gaussian_filter(noise, sigma=sigma)
    else:  # pragma: no cover - fallback path when scipy is unavailable
        kernel = np.ones((3, 3)) / 9.0
        smoothed = noise
        for _ in range(4):
            padded = np.pad(smoothed, 1, mode="edge")
            smoothed = np.stack(
                [
                    sum(
                        kernel[di + 1, dj + 1] * padded[1 + di : 1 + di + grid_size, 1 + dj : 1 + dj + grid_size]
                        for di in (-1, 0, 1)
                        for dj in (-1, 0, 1)
                    )
                ]
            )[0]
    smoothed -= smoothed.min()
    smoothed /= (smoothed.max() - smoothed.min() + 1e-9)
    return smoothed


# Approximate monthly climate normals, index 0 = January. These are coarse
# literature-typical values for interior vs. maritime-influenced south-central
# Alaska; they only need to be *directionally* right to give the synthetic
# generator a realistic seasonal shape, not to be a climate product.
_CLIMATE_PROFILES: dict[str, dict[str, list[float]]] = {
    "interior-fairbanks": {
        "temp_2m_c": [-21, -16, -7, 3, 11, 17, 18, 15, 8, -3, -13, -19],
        "humidity_pct": [78, 75, 70, 60, 52, 50, 55, 62, 70, 78, 80, 80],
        "wind_ms": [2.6, 2.8, 3.0, 3.4, 3.6, 3.3, 3.0, 3.0, 3.1, 2.9, 2.7, 2.6],
        "precip_mm": [15, 12, 10, 10, 22, 38, 48, 55, 32, 22, 18, 16],
        "ndvi_peak": 0.78,
    },
    "kenai-peninsula": {
        "temp_2m_c": [-7, -5, -2, 3, 8, 12, 14, 13, 9, 3, -3, -6],
        "humidity_pct": [80, 78, 74, 68, 62, 60, 63, 68, 74, 80, 82, 82],
        "wind_ms": [3.4, 3.4, 3.6, 3.7, 3.5, 3.2, 3.0, 3.0, 3.4, 3.8, 3.7, 3.5],
        "precip_mm": [55, 45, 38, 32, 35, 42, 55, 65, 80, 90, 70, 60],
        "ndvi_peak": 0.68,
    },
}


@dataclass
class SyntheticRegionDataset:
    region: Region
    times: list[tuple[int, int]]  # (year, month) per timestep, ascending
    spatial: np.ndarray  # (T, C, H, W) float32, channel order == SPATIAL_FEATURE_CHANNELS
    weather: pd.DataFrame  # T rows, columns == WEATHER_FEATURE_COLUMNS (+ year, month)
    ignition_indicator: np.ndarray  # (T,) 1 if a synthetic fire ignited in-region that month
    ignition_severity: np.ndarray  # (T,) acres burned, 0 if no ignition
    labels: np.ndarray  # (T, N_HORIZONS) int class id in {0,1,2}, -1 where horizon runs past data end
    firms_detections: pd.DataFrame
    perimeters: pd.DataFrame


def _severity_to_class(total_severity: float, any_ignition: bool) -> int:
    if not any_ignition or total_severity <= 0:
        return 0  # No Risk (realized)
    if total_severity < 500:
        return 1  # Moderate
    return 2  # High


class SyntheticDataGenerator:
    """Generates a multi-year monthly synthetic dataset for one region."""

    def __init__(
        self,
        region: Region,
        start_year: int = 2015,
        end_year: int = 2024,
        grid_size: int = GRID_SIZE,
        seed: int | None = None,
    ) -> None:
        self.region = region
        self.start_year = start_year
        self.end_year = end_year
        self.grid_size = grid_size
        self.seed = seed if seed is not None else RANDOM_SEED
        self.rng = np.random.default_rng(self.seed)
        self.profile = _CLIMATE_PROFILES.get(region.id, _CLIMATE_PROFILES["interior-fairbanks"])

    def _months(self) -> list[tuple[int, int]]:
        return [
            (y, m)
            for y in range(self.start_year, self.end_year + 1)
            for m in range(1, 13)
        ]

    def generate(self) -> SyntheticRegionDataset:
        rng = self.rng
        times = self._months()
        n_t = len(times)
        gs = self.grid_size
        n_c = len(SPATIAL_FEATURE_CHANNELS)

        terrain_moisture = _smooth_field(gs, rng)  # static: low-lying/wet vs. ridge/dry
        terrain_fuel = _smooth_field(gs, rng)  # static: fuel load heterogeneity

        spatial = np.zeros((n_t, n_c, gs, gs), dtype=np.float32)
        weather_rows = []
        ignition_indicator = np.zeros(n_t, dtype=np.int64)
        ignition_severity = np.zeros(n_t, dtype=np.float64)
        firms_rows = []
        perimeter_rows = []

        cumulative_dryness = 0.0  # simple drought-accumulator across the season

        for t, (year, month) in enumerate(times):
            idx = month - 1
            temp = self.profile["temp_2m_c"][idx] + rng.normal(0, 1.5)
            humidity = np.clip(self.profile["humidity_pct"][idx] + rng.normal(0, 4), 15, 100)
            wind = max(0.1, self.profile["wind_ms"][idx] + rng.normal(0, 0.6))
            precip = max(0.0, self.profile["precip_mm"][idx] + rng.normal(0, 8))

            is_growing_season = 4 <= month <= 9
            if is_growing_season:
                dryness_delta = (
                    (65.0 - humidity) * 0.045
                    + (28.0 - precip) * 0.03
                    + max(0.0, temp - 12.0) * 0.05
                )
                cumulative_dryness = np.clip(cumulative_dryness * 0.88 + dryness_delta, 0.0, 20.0)
            else:
                cumulative_dryness *= 0.15  # snowpack resets drought signal

            # --- seasonal spatial fields -------------------------------
            season_frac = np.clip(np.sin((month - 3.5) / 12 * 2 * np.pi), -1, 1)
            ndvi_seasonal = max(0.03, self.profile["ndvi_peak"] * max(0.0, season_frac) ** 1.3)
            ndvi_field = np.clip(
                ndvi_seasonal * (0.6 + 0.5 * terrain_moisture) + rng.normal(0, 0.02, (gs, gs)),
                0.0,
                0.95,
            )
            nbr_field = np.clip(ndvi_field * 0.9 - 0.05 + 0.1 * terrain_fuel, -0.2, 0.9)

            dryness_field = np.clip(
                (cumulative_dryness / 4.0) * (1.0 - 0.6 * terrain_moisture) + rng.normal(0, 0.05, (gs, gs)),
                0.0,
                1.0,
            )
            lst_anomaly_field = (dryness_field - 0.3) * 6.0 + rng.normal(0, 0.5, (gs, gs))

            # SAR backscatter (dB): wetter/denser canopy -> higher (less negative) VV/VH
            sar_vv = -18 + 6 * (1 - dryness_field) + 3 * ndvi_field + rng.normal(0, 0.4, (gs, gs))
            sar_vh = sar_vv - (4 + 2 * ndvi_field) + rng.normal(0, 0.3, (gs, gs))

            fuel_moisture_proxy = np.clip(
                0.5 * (1 - dryness_field) + 0.3 * ndvi_field + 0.2 * (humidity / 100.0),
                0.0,
                1.0,
            )

            spatial[t, 0] = ndvi_field
            spatial[t, 1] = nbr_field
            spatial[t, 2] = lst_anomaly_field
            spatial[t, 3] = sar_vv
            spatial[t, 4] = sar_vh
            spatial[t, 5] = fuel_moisture_proxy

            weather_rows.append(
                {
                    "year": year,
                    "month": month,
                    "temp_2m_c": temp,
                    "relative_humidity_pct": humidity,
                    "wind_speed_ms": wind,
                    "precipitation_mm": precip,
                }
            )

            # --- stochastic ignition process -----------------------------
            region_dryness = float(np.clip(dryness_field.mean(), 0, 1))
            if is_growing_season:
                logit = -3.0 + 5.5 * region_dryness + 0.18 * (wind - 3.0) + 0.04 * max(0.0, temp - 15)
                ignition_prob = 1 / (1 + np.exp(-logit))
            else:
                ignition_prob = 0.001
            ignited = rng.random() < ignition_prob
            severity = 0.0
            if ignited:
                severity = float(rng.lognormal(mean=5.5 + 2.5 * region_dryness, sigma=1.1))
                severity = min(severity, 50000.0)
                ignition_indicator[t] = 1
                ignition_severity[t] = severity
                n_detections = int(np.clip(severity / 40, 3, 400))
                for _ in range(n_detections):
                    lon = rng.uniform(self.region.min_lon, self.region.max_lon)
                    lat = rng.uniform(self.region.min_lat, self.region.max_lat)
                    day = rng.integers(1, 28)
                    firms_rows.append(
                        {
                            "region_id": self.region.id,
                            "acq_date": pd.Timestamp(year=year, month=month, day=int(day)),
                            "latitude": lat,
                            "longitude": lon,
                            "frp": float(rng.gamma(2.0, 8.0)),
                            "confidence": "n" if rng.random() < 0.7 else "h",
                        }
                    )
                perimeter_rows.append(
                    {
                        "fire_name": f"SYN-{self.region.id[:3].upper()}-{year}{month:02d}",
                        "year": year,
                        "month": month,
                        "ignition_date": f"{year}-{month:02d}-15",
                        "acres": severity,
                    }
                )

        weather = pd.DataFrame(weather_rows)

        labels = np.full((n_t, N_HORIZONS), fill_value=-1, dtype=np.int64)
        for t in range(n_t):
            for h_idx, horizon in enumerate(HORIZONS_MONTHS):
                window_end = t + horizon
                if window_end > n_t:
                    continue  # not enough future data to label this horizon
                window = slice(t + 1, window_end + 1)
                any_ign = bool(ignition_indicator[window].any())
                total_sev = float(ignition_severity[window].sum())
                labels[t, h_idx] = _severity_to_class(total_sev, any_ign)

        firms_df = pd.DataFrame(firms_rows)
        perimeters_df = pd.DataFrame(perimeter_rows)

        return SyntheticRegionDataset(
            region=self.region,
            times=times,
            spatial=spatial,
            weather=weather,
            ignition_indicator=ignition_indicator,
            ignition_severity=ignition_severity,
            labels=labels,
            firms_detections=firms_df,
            perimeters=perimeters_df,
        )


def generate_all_regions(
    regions: dict[str, Region],
    start_year: int = 2015,
    end_year: int = 2024,
    grid_size: int = GRID_SIZE,
    seed: int | None = None,
) -> dict[str, SyntheticRegionDataset]:
    """Convenience wrapper: generate a synthetic dataset for every region."""
    out = {}
    for i, (region_id, region) in enumerate(regions.items()):
        region_seed = None if seed is None else seed + i
        gen = SyntheticDataGenerator(
            region=region,
            start_year=start_year,
            end_year=end_year,
            grid_size=grid_size,
            seed=region_seed,
        )
        out[region_id] = gen.generate()
    return out
