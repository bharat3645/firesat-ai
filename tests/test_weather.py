from __future__ import annotations

import numpy as np

from firesat.features.weather import fire_weather_danger_proxy, fuel_moisture_index


def test_fuel_moisture_index_bounds():
    temp = np.array([-10, 10, 30])
    rh = np.array([20, 50, 90])
    precip = np.array([0, 20, 100])
    result = fuel_moisture_index(temp, rh, precip)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_fuel_moisture_higher_with_more_humidity_and_rain():
    dry_hot = fuel_moisture_index(np.array([30.0]), np.array([20.0]), np.array([0.0]))
    wet_mild = fuel_moisture_index(np.array([15.0]), np.array([80.0]), np.array([50.0]))
    assert wet_mild[0] > dry_hot[0]


def test_fire_weather_danger_monotonic_in_wind():
    base = dict(temp_c=np.array([25.0]), relative_humidity_pct=np.array([25.0]), precipitation_mm=np.array([0.0]))
    calm = fire_weather_danger_proxy(wind_speed_ms=np.array([1.0]), **base)
    windy = fire_weather_danger_proxy(wind_speed_ms=np.array([12.0]), **base)
    assert windy[0] >= calm[0]


def test_fire_weather_danger_bounds():
    rng = np.random.default_rng(0)
    temp = rng.uniform(-30, 35, 100)
    rh = rng.uniform(10, 100, 100)
    wind = rng.uniform(0, 20, 100)
    precip = rng.uniform(0, 150, 100)
    danger = fire_weather_danger_proxy(temp, rh, wind, precip)
    assert np.all(danger >= 0.0) and np.all(danger <= 1.0)


def test_dry_hot_windy_dry_month_has_high_danger():
    danger_extreme = fire_weather_danger_proxy(
        temp_c=np.array([32.0]),
        relative_humidity_pct=np.array([15.0]),
        wind_speed_ms=np.array([15.0]),
        precipitation_mm=np.array([0.0]),
    )
    danger_mild = fire_weather_danger_proxy(
        temp_c=np.array([12.0]),
        relative_humidity_pct=np.array([85.0]),
        wind_speed_ms=np.array([1.0]),
        precipitation_mm=np.array([60.0]),
    )
    assert danger_extreme[0] > danger_mild[0]
