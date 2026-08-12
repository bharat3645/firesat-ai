"""Weather/reanalysis-derived fire-danger indicators.

These are simplified, transparently-documented heuristics rather than the
full Canadian Fire Weather Index (FWI) System or the US National Fire Danger
Rating System (NFDRS) -- both of which need daily (not monthly) inputs and
multi-day state carried across the season. Reproducing them faithfully is
out of scope for this MVP; see docs/evaluation_report.md for the explicit
limitation. What's implemented here is honest about being an approximation
while still capturing the right monotonic relationships (hot + dry + windy
=> higher danger), which is what the model needs as an input signal.
"""
from __future__ import annotations

import numpy as np


def fuel_moisture_index(
    temp_c: np.ndarray, relative_humidity_pct: np.ndarray, precipitation_mm: np.ndarray
) -> np.ndarray:
    """Coarse dead-fuel-moisture proxy in [0, 1] (higher = wetter fuel, lower risk).

    Combines relative humidity (dominant driver of fine dead-fuel moisture
    equilibrium), a temperature penalty (evaporative demand), and a recent
    precipitation bonus, then squashes to [0, 1].
    """
    temp_c = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(relative_humidity_pct, dtype=np.float64)
    precip = np.asarray(precipitation_mm, dtype=np.float64)

    humidity_term = rh / 100.0
    temp_penalty = np.clip((temp_c - 10.0) / 40.0, -0.3, 0.3)
    precip_bonus = np.clip(precip / 100.0, 0.0, 0.3)
    raw = humidity_term - temp_penalty + precip_bonus
    return np.clip(raw, 0.0, 1.0)


def fire_weather_danger_proxy(
    temp_c: np.ndarray,
    relative_humidity_pct: np.ndarray,
    wind_speed_ms: np.ndarray,
    precipitation_mm: np.ndarray,
) -> np.ndarray:
    """Simplified fire-weather danger score in [0, 1] (higher = more dangerous).

    Not a substitute for FWI/NFDRS -- see module docstring -- but a
    monotonic, unit-tested combination of the four ERA5 drivers the
    proposal lists (temperature, humidity, wind) plus precipitation.
    """
    fmi = fuel_moisture_index(temp_c, relative_humidity_pct, precipitation_mm)
    wind = np.asarray(wind_speed_ms, dtype=np.float64)
    wind_term = np.clip(wind / 10.0, 0.0, 1.0)
    danger = 0.7 * (1.0 - fmi) + 0.3 * wind_term
    return np.clip(danger, 0.0, 1.0)
