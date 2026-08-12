from __future__ import annotations

import numpy as np

from firesat.features.indices import dnbr, evi, nbr, ndmi, ndvi


def test_ndvi_range_and_known_values():
    nir = np.array([0.5, 0.2, 0.0])
    red = np.array([0.1, 0.2, 0.0])
    result = ndvi(nir, red)
    assert result.shape == (3,)
    # (0.5-0.1)/(0.5+0.1) = 0.6667
    assert np.isclose(result[0], 4 / 6, atol=1e-3)
    # equal reflectance -> ~0
    assert np.isclose(result[1], 0.0, atol=1e-6)
    assert np.all(result <= 1.0) and np.all(result >= -1.0)


def test_ndvi_dense_vegetation_higher_than_bare_ground():
    dense_veg = ndvi(np.array([0.6]), np.array([0.05]))
    bare_ground = ndvi(np.array([0.3]), np.array([0.28]))
    assert dense_veg[0] > bare_ground[0]


def test_nbr_drops_after_burn():
    pre_fire_nir, pre_fire_swir = 0.55, 0.15
    post_fire_nir, post_fire_swir = 0.2, 0.35
    pre_nbr = nbr(np.array([pre_fire_nir]), np.array([pre_fire_swir]))
    post_nbr = nbr(np.array([post_fire_nir]), np.array([post_fire_swir]))
    assert post_nbr[0] < pre_nbr[0]


def test_dnbr_positive_for_burned_pixel():
    pre = np.array([0.6])
    post = np.array([-0.1])
    result = dnbr(pre, post)
    assert result[0] > 0


def test_ndmi_and_evi_shapes_and_bounds():
    nir = np.random.default_rng(0).uniform(0.1, 0.6, size=10)
    swir1 = np.random.default_rng(1).uniform(0.05, 0.5, size=10)
    red = np.random.default_rng(2).uniform(0.05, 0.4, size=10)
    blue = np.random.default_rng(3).uniform(0.02, 0.2, size=10)

    moisture = ndmi(nir, swir1)
    assert moisture.shape == (10,)
    assert np.all(np.isfinite(moisture))

    vegetation_index = evi(nir, red, blue)
    assert vegetation_index.shape == (10,)
    assert np.all(np.isfinite(vegetation_index))
