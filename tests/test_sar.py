from __future__ import annotations

import numpy as np

from firesat.features.sar import (
    _db_to_linear,
    _linear_to_db,
    cross_pol_ratio,
    radar_vegetation_index,
    soil_moisture_proxy,
)


def test_db_linear_roundtrip():
    db_values = np.array([-20.0, -10.0, -5.0, 0.0])
    linear = _db_to_linear(db_values)
    back = _linear_to_db(linear)
    assert np.allclose(back, db_values, atol=1e-6)


def test_radar_vegetation_index_bounds():
    rng = np.random.default_rng(0)
    vv_db = rng.uniform(-20, -5, size=50)
    vh_db = vv_db - rng.uniform(2, 8, size=50)  # VH typically lower than VV
    rvi = radar_vegetation_index(vv_db, vh_db)
    assert np.all(rvi >= 0.0)
    assert np.all(rvi <= 4.0)


def test_cross_pol_ratio_is_simple_difference():
    vv = np.array([-10.0, -8.0])
    vh = np.array([-14.0, -13.0])
    result = cross_pol_ratio(vv, vh)
    assert np.allclose(result, vh - vv)


def test_soil_moisture_proxy_higher_when_wetter():
    dry_reference = -16.0
    wet_month_vv = -10.0  # higher (less negative) backscatter => wetter
    dry_month_vv = -18.0
    wet_proxy = soil_moisture_proxy(np.array([wet_month_vv]), dry_reference)
    dry_proxy = soil_moisture_proxy(np.array([dry_month_vv]), dry_reference)
    assert wet_proxy[0] > dry_proxy[0]
