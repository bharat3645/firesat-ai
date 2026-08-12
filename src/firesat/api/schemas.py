"""Pydantic response models for the FireSat-AI API."""
from __future__ import annotations

from pydantic import BaseModel


class RegionOut(BaseModel):
    id: str
    name: str
    description: str
    bbox: list[float]
    centroid: list[float]
    geometry: dict


class HorizonPrediction(BaseModel):
    months: int
    risk_class: str
    risk_class_id: int
    probabilities: dict[str, float]


class TemporalAttentionPoint(BaseModel):
    time: str
    weight: float


class RiskPredictionOut(BaseModel):
    region_id: str
    as_of: str
    horizons: dict[str, HorizonPrediction]
    channel_attention: dict[str, float]
    temporal_attention: list[TemporalAttentionPoint]


class FireIgnitionEvent(BaseModel):
    time: str
    acres: float


class FireHistoryOut(BaseModel):
    region_id: str
    ignitions: list[FireIgnitionEvent]
    n_fire_events: int
    total_acres_burned: float


class HealthOut(BaseModel):
    status: str
    ready: bool
    detail: str
    regions_loaded: list[str]
