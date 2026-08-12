from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from firesat.api.schemas import FireHistoryOut, RiskPredictionOut
from firesat.inference import InferenceService, get_inference_service

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/{region_id}", response_model=RiskPredictionOut)
def get_current_risk(
    region_id: str, service: InferenceService = Depends(get_inference_service)
) -> RiskPredictionOut:
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.status_message)
    try:
        prediction = service.predict(region_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RiskPredictionOut(
        region_id=prediction.region_id,
        as_of=prediction.as_of,
        horizons=prediction.horizons,
        channel_attention=prediction.channel_attention,
        temporal_attention=prediction.temporal_attention,
    )


@router.get("/{region_id}/history", response_model=list[RiskPredictionOut])
def get_risk_history(
    region_id: str,
    months: int = Query(24, ge=1, le=120),
    service: InferenceService = Depends(get_inference_service),
) -> list[RiskPredictionOut]:
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.status_message)
    try:
        predictions = service.predict_history(region_id, n_points=months)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [
        RiskPredictionOut(
            region_id=p.region_id,
            as_of=p.as_of,
            horizons=p.horizons,
            channel_attention=p.channel_attention,
            temporal_attention=p.temporal_attention,
        )
        for p in predictions
    ]


@router.get("/{region_id}/fire-history", response_model=FireHistoryOut)
def get_fire_history(
    region_id: str, service: InferenceService = Depends(get_inference_service)
) -> FireHistoryOut:
    if region_id not in service.datasets:
        raise HTTPException(status_code=404, detail=f"Unknown region_id '{region_id}'")
    return FireHistoryOut(**service.region_fire_history(region_id))
