"""Inference service: loads a trained checkpoint + processed regional data
and turns them into risk predictions, used by both the FastAPI backend and
any offline/batch scripts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from firesat.config import (
    HORIZONS_MONTHS,
    PROCESSED_DATA_DIR,
    REGIONS,
    RISK_CLASSES,
    SEQUENCE_LENGTH,
    WEATHER_FEATURE_COLUMNS,
)
from firesat.data.pipeline import load_all_processed_regions
from firesat.data.synthetic import SyntheticRegionDataset
from firesat.features.build_features import NormalizationStats, apply_normalization
from firesat.models.interpret import summarize_channel_attention, summarize_temporal_attention
from firesat.training.dataset import extract_window
from firesat.training.train import DEFAULT_CHECKPOINT_PATH, load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class RegionRiskPrediction:
    region_id: str
    as_of: str
    horizons: dict[str, dict]  # horizon_key -> {months, risk_class, probabilities}
    channel_attention: dict[str, float]
    temporal_attention: list[dict]


class InferenceService:
    """Holds the loaded model + processed datasets in memory and answers
    prediction requests without re-reading disk on every call."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        data_dir: str | Path = PROCESSED_DATA_DIR,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.data_dir = Path(data_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.datasets: dict[str, SyntheticRegionDataset] = {}
        self.model = None
        self.stats: NormalizationStats | None = None
        self.sequence_length = SEQUENCE_LENGTH
        self._norm_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.ready = False
        self.status_message = "not loaded"

    def load(self) -> None:
        try:
            self.datasets = load_all_processed_regions(self.data_dir)
        except FileNotFoundError as exc:
            self.status_message = (
                f"No processed data found ({exc}). Run "
                "`python scripts/generate_demo_data.py` first."
            )
            logger.warning(self.status_message)
            return

        if not self.checkpoint_path.exists():
            self.status_message = (
                f"No trained checkpoint at {self.checkpoint_path}. Run "
                "`python scripts/train_demo.py` first."
            )
            logger.warning(self.status_message)
            return

        self.model, metadata = load_checkpoint(self.checkpoint_path, device=self.device)
        self.stats = NormalizationStats.from_dict(metadata["normalization_stats"])
        self.sequence_length = metadata["config"].get("sequence_length", SEQUENCE_LENGTH)

        for region_id, ds in self.datasets.items():
            weather_raw = ds.weather[WEATHER_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            self._norm_cache[region_id] = apply_normalization(ds.spatial, weather_raw, self.stats)

        self.ready = True
        self.status_message = "ready"
        logger.info("InferenceService ready: regions=%s", list(self.datasets.keys()))

    def list_regions(self) -> list[dict]:
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "bbox": list(r.bbox),
                "centroid": list(r.centroid),
                "geometry": r.polygon_geojson(),
            }
            for r in REGIONS.values()
        ]

    def _require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(self.status_message)

    def predict(self, region_id: str, anchor_idx: int | None = None) -> RegionRiskPrediction:
        self._require_ready()
        if region_id not in self.datasets:
            raise KeyError(f"Unknown region_id '{region_id}'. Known: {list(self.datasets)}")

        ds = self.datasets[region_id]
        norm_spatial, norm_weather = self._norm_cache[region_id]
        n_t = ds.spatial.shape[0]
        idx = n_t - 1 if anchor_idx is None else anchor_idx

        window = extract_window(norm_spatial, norm_weather, ds, self.sequence_length, idx)
        spatial_t = window["spatial"].unsqueeze(0).to(self.device)
        weather_t = window["weather"].unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(spatial_t, weather_t)

        horizons_out = {}
        for h in HORIZONS_MONTHS:
            key = f"horizon_{h}m"
            probs = torch.softmax(out["logits"][key], dim=-1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            horizons_out[key] = {
                "months": h,
                "risk_class": RISK_CLASSES[pred_class],
                "risk_class_id": pred_class,
                "probabilities": {
                    RISK_CLASSES[i]: float(probs[i]) for i in range(len(RISK_CLASSES))
                },
            }

        channel_weights = out["channel_attention"][0].cpu().numpy()  # (T, C)
        temporal_weights = out["temporal_attention"][0].cpu().numpy()  # (T,)

        return RegionRiskPrediction(
            region_id=region_id,
            as_of=window["anchor_time"],
            horizons=horizons_out,
            channel_attention=summarize_channel_attention(channel_weights),
            temporal_attention=summarize_temporal_attention(
                temporal_weights, window["time_labels"]
            ),
        )

    def predict_history(self, region_id: str, n_points: int = 24) -> list[RegionRiskPrediction]:
        self._require_ready()
        ds = self.datasets[region_id]
        n_t = ds.spatial.shape[0]
        lo = self.sequence_length - 1
        anchors = list(range(max(lo, n_t - n_points), n_t))
        return [self.predict(region_id, anchor_idx=a) for a in anchors]

    def region_fire_history(self, region_id: str) -> dict:
        ds = self.datasets[region_id]
        ignitions = []
        for t, (year, month) in enumerate(ds.times):
            if ds.ignition_indicator[t]:
                ignitions.append(
                    {
                        "time": f"{year:04d}-{month:02d}",
                        "acres": float(ds.ignition_severity[t]),
                    }
                )
        return {
            "region_id": region_id,
            "ignitions": ignitions,
            "n_fire_events": len(ignitions),
            "total_acres_burned": float(ds.ignition_severity.sum()),
        }


_default_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Process-wide singleton, lazily loaded (used as a FastAPI dependency)."""
    global _default_service
    if _default_service is None:
        _default_service = InferenceService()
        _default_service.load()
    return _default_service
