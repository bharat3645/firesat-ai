from __future__ import annotations

import numpy as np

from firesat.config import HORIZONS_MONTHS, SPATIAL_FEATURE_CHANNELS, WEATHER_FEATURE_COLUMNS
from firesat.data.synthetic import SyntheticDataGenerator


def test_generate_shapes(tiny_dataset):
    n_t = len(tiny_dataset.times)
    assert n_t == 24  # 2 years
    assert tiny_dataset.spatial.shape == (n_t, len(SPATIAL_FEATURE_CHANNELS), 4, 4)
    assert set(WEATHER_FEATURE_COLUMNS) <= set(tiny_dataset.weather.columns)
    assert len(tiny_dataset.weather) == n_t
    assert tiny_dataset.labels.shape == (n_t, len(HORIZONS_MONTHS))


def test_labels_are_valid_class_ids_or_sentinel(tiny_dataset):
    labels = tiny_dataset.labels
    assert set(np.unique(labels)) <= {-1, 0, 1, 2}


def test_labels_missing_only_near_end_of_series(tiny_dataset):
    n_t = len(tiny_dataset.times)
    for h_idx, h in enumerate(HORIZONS_MONTHS):
        col = tiny_dataset.labels[:, h_idx]
        missing = np.where(col == -1)[0]
        if len(missing):
            assert missing.min() >= n_t - h


def test_deterministic_with_seed(tiny_region):
    gen1 = SyntheticDataGenerator(tiny_region, 2020, 2021, grid_size=4, seed=42)
    gen2 = SyntheticDataGenerator(tiny_region, 2020, 2021, grid_size=4, seed=42)
    d1, d2 = gen1.generate(), gen2.generate()
    assert np.allclose(d1.spatial, d2.spatial)
    assert np.array_equal(d1.labels, d2.labels)
    assert np.array_equal(d1.ignition_indicator, d2.ignition_indicator)


def test_different_seeds_produce_different_data(tiny_region):
    gen1 = SyntheticDataGenerator(tiny_region, 2020, 2021, grid_size=4, seed=1)
    gen2 = SyntheticDataGenerator(tiny_region, 2020, 2021, grid_size=4, seed=999)
    d1, d2 = gen1.generate(), gen2.generate()
    assert not np.allclose(d1.spatial, d2.spatial)


def test_spatial_values_finite_and_reasonable(tiny_dataset):
    assert np.all(np.isfinite(tiny_dataset.spatial))
    ndvi_channel = tiny_dataset.spatial[:, SPATIAL_FEATURE_CHANNELS.index("ndvi")]
    assert ndvi_channel.min() >= -0.5
    assert ndvi_channel.max() <= 1.0


def test_firms_detections_only_in_ignition_months(tiny_dataset):
    if tiny_dataset.firms_detections.empty:
        return
    ignition_months = {
        (y, m)
        for (y, m), flag in zip(tiny_dataset.times, tiny_dataset.ignition_indicator)
        if flag
    }
    for _, row in tiny_dataset.firms_detections.iterrows():
        ts = row["acq_date"]
        assert (ts.year, ts.month) in ignition_months
