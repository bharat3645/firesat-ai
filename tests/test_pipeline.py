from __future__ import annotations

import numpy as np

from firesat.data.pipeline import load_processed_region, save_region_dataset


def test_save_and_load_region_roundtrip(tiny_dataset, tmp_path):
    save_region_dataset(tiny_dataset, tmp_path)
    loaded = load_processed_region(tiny_dataset.region.id, tmp_path)

    assert loaded.region.id == tiny_dataset.region.id
    assert loaded.region.bbox == tiny_dataset.region.bbox
    assert loaded.times == tiny_dataset.times
    assert np.allclose(loaded.spatial, tiny_dataset.spatial)
    assert np.array_equal(loaded.labels, tiny_dataset.labels)
    assert np.array_equal(loaded.ignition_indicator, tiny_dataset.ignition_indicator)
    assert len(loaded.weather) == len(tiny_dataset.weather)
    assert set(tiny_dataset.weather.columns) <= set(loaded.weather.columns)


def test_load_missing_region_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_processed_region("does-not-exist", tmp_path)
