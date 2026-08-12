"""PyTorch ``Dataset`` that turns a region's monthly time series into
sliding-window ``(lookback -> multi-horizon labels)`` training samples.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from firesat.config import SEQUENCE_LENGTH, WEATHER_FEATURE_COLUMNS
from firesat.data.synthetic import SyntheticRegionDataset
from firesat.features.build_features import NormalizationStats, apply_normalization


class FireRiskWindowDataset(Dataset):
    """Sliding-window dataset for a single region.

    Sample ``i`` corresponds to an anchor month ``t``: the model sees the
    ``sequence_length`` months up to and including ``t`` and predicts risk
    class for each horizon in ``HORIZONS_MONTHS`` measured forward from
    ``t``. Only anchors with fully-observed labels for every horizon (i.e.
    not running past the end of the generated series) are included.
    """

    def __init__(
        self,
        data: SyntheticRegionDataset,
        stats: NormalizationStats,
        sequence_length: int = SEQUENCE_LENGTH,
        min_index: int | None = None,
        max_index: int | None = None,
    ) -> None:
        self.data = data
        self.sequence_length = sequence_length
        self.stats = stats

        weather_raw = data.weather[WEATHER_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        norm_spatial, norm_weather = apply_normalization(data.spatial, weather_raw, stats)
        self.spatial = norm_spatial
        self.weather = norm_weather

        n_t = data.spatial.shape[0]
        lo = sequence_length - 1 if min_index is None else max(min_index, sequence_length - 1)
        hi = n_t if max_index is None else min(max_index, n_t)
        self.valid_anchors = [
            t for t in range(lo, hi) if np.all(data.labels[t] >= 0)
        ]

    def __len__(self) -> int:
        return len(self.valid_anchors)

    def __getitem__(self, idx: int):
        t = self.valid_anchors[idx]
        window = extract_window(self.spatial, self.weather, self.data, self.sequence_length, t)
        window["labels"] = torch.from_numpy(self.data.labels[t]).long()
        return window


def extract_window(
    norm_spatial: np.ndarray,
    norm_weather: np.ndarray,
    data: SyntheticRegionDataset,
    sequence_length: int,
    anchor_idx: int,
) -> dict:
    """Slice out the ``sequence_length``-month window ending at ``anchor_idx``
    from already-normalized spatial/weather arrays. Shared by the training
    dataset (labels attached by the caller) and by inference (no labels
    needed, since the model is forecasting *past* the end of history).
    """
    start = anchor_idx - sequence_length + 1
    if start < 0:
        raise ValueError(
            f"anchor_idx={anchor_idx} needs {sequence_length} months of history "
            f"but only {anchor_idx + 1} are available."
        )
    spatial_window = norm_spatial[start : anchor_idx + 1]  # (T, C, H, W)
    weather_window = norm_weather[start : anchor_idx + 1]  # (T, F)
    time_labels = [f"{y:04d}-{m:02d}" for y, m in data.times[start : anchor_idx + 1]]
    year, month = data.times[anchor_idx]
    return {
        "spatial": torch.from_numpy(spatial_window),
        "weather": torch.from_numpy(weather_window),
        "region_id": data.region.id,
        "anchor_time": f"{year:04d}-{month:02d}",
        "time_labels": time_labels,
    }


def collate_fire_risk_batch(batch: list[dict]) -> dict:
    return {
        "spatial": torch.stack([b["spatial"] for b in batch]),
        "weather": torch.stack([b["weather"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "region_id": [b["region_id"] for b in batch],
        "anchor_time": [b["anchor_time"] for b in batch],
    }


def temporal_train_val_split(
    n_timesteps: int, sequence_length: int = SEQUENCE_LENGTH, val_fraction: float = 0.2
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Chronological split: validation is the trailing ``val_fraction`` of
    months, so no future information leaks into training (a real risk with a
    random shuffle split on a time series)."""
    split_at = int(n_timesteps * (1 - val_fraction))
    split_at = max(split_at, sequence_length)
    return (0, split_at), (split_at, n_timesteps)


def build_concat_dataset(
    datasets: dict[str, SyntheticRegionDataset],
    stats: NormalizationStats,
    sequence_length: int = SEQUENCE_LENGTH,
    split: str = "train",
    val_fraction: float = 0.2,
) -> ConcatDataset:
    """Build a ``ConcatDataset`` across all regions for the given split."""
    parts = []
    for region_dataset in datasets.values():
        n_t = region_dataset.spatial.shape[0]
        (train_lo, train_hi), (val_lo, val_hi) = temporal_train_val_split(
            n_t, sequence_length, val_fraction
        )
        lo, hi = (train_lo, train_hi) if split == "train" else (val_lo, val_hi)
        parts.append(
            FireRiskWindowDataset(
                region_dataset, stats, sequence_length, min_index=lo, max_index=hi
            )
        )
    return ConcatDataset(parts)
