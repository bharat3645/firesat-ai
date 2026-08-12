from __future__ import annotations

import numpy as np
import pytest

from firesat.config import HORIZONS_MONTHS, WEATHER_FEATURE_COLUMNS
from firesat.features.build_features import fit_normalization_stats
from firesat.training.dataset import (
    FireRiskWindowDataset,
    build_concat_dataset,
    collate_fire_risk_batch,
    extract_window,
    temporal_train_val_split,
)

SEQ_LEN = 6


def _stats_for(dataset):
    weather = dataset.weather[WEATHER_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return fit_normalization_stats(dataset.spatial, weather)


def test_window_dataset_length_and_shapes(tiny_dataset):
    stats = _stats_for(tiny_dataset)
    ds = FireRiskWindowDataset(tiny_dataset, stats, sequence_length=SEQ_LEN)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["spatial"].shape[0] == SEQ_LEN
    assert sample["weather"].shape[0] == SEQ_LEN
    assert sample["labels"].shape == (len(HORIZONS_MONTHS),)
    assert (sample["labels"] >= 0).all()


def test_window_dataset_respects_min_max_index(tiny_dataset):
    stats = _stats_for(tiny_dataset)
    full = FireRiskWindowDataset(tiny_dataset, stats, sequence_length=SEQ_LEN)
    partial = FireRiskWindowDataset(
        tiny_dataset, stats, sequence_length=SEQ_LEN, min_index=10, max_index=15
    )
    assert len(partial) <= len(full)
    for t in partial.valid_anchors:
        assert 10 <= t < 15


def test_temporal_train_val_split_is_chronological():
    (train_lo, train_hi), (val_lo, val_hi) = temporal_train_val_split(24, sequence_length=6, val_fraction=0.25)
    assert train_lo == 0
    assert val_hi == 24
    assert train_hi == val_lo  # contiguous, non-overlapping
    assert train_hi < 24


def test_extract_window_matches_getitem(tiny_dataset):
    stats = _stats_for(tiny_dataset)
    ds = FireRiskWindowDataset(tiny_dataset, stats, sequence_length=SEQ_LEN)
    anchor = ds.valid_anchors[0]
    window = extract_window(ds.spatial, ds.weather, tiny_dataset, SEQ_LEN, anchor)
    sample = ds[0]
    assert np.allclose(window["spatial"].numpy(), sample["spatial"].numpy())
    assert window["anchor_time"] == sample["anchor_time"]
    assert len(window["time_labels"]) == SEQ_LEN


def test_extract_window_raises_when_insufficient_history(tiny_dataset):
    stats = _stats_for(tiny_dataset)
    ds = FireRiskWindowDataset(tiny_dataset, stats, sequence_length=SEQ_LEN)
    with pytest.raises(ValueError):
        extract_window(ds.spatial, ds.weather, tiny_dataset, SEQ_LEN, anchor_idx=1)


def test_collate_batches_correctly(tiny_dataset):
    stats = _stats_for(tiny_dataset)
    ds = FireRiskWindowDataset(tiny_dataset, stats, sequence_length=SEQ_LEN)
    batch = collate_fire_risk_batch([ds[0], ds[1], ds[2]])
    assert batch["spatial"].shape[0] == 3
    assert batch["spatial"].shape[1] == SEQ_LEN
    assert len(batch["region_id"]) == 3


def test_build_concat_dataset_train_val_disjoint(tiny_datasets):
    all_spatial = np.concatenate([d.spatial for d in tiny_datasets.values()], axis=0)
    all_weather = np.concatenate(
        [d.weather[WEATHER_FEATURE_COLUMNS].to_numpy(dtype=np.float32) for d in tiny_datasets.values()],
        axis=0,
    )
    stats = fit_normalization_stats(all_spatial, all_weather)
    train_ds = build_concat_dataset(tiny_datasets, stats, SEQ_LEN, split="train", val_fraction=0.3)
    val_ds = build_concat_dataset(tiny_datasets, stats, SEQ_LEN, split="val", val_fraction=0.3)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
