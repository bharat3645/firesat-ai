from __future__ import annotations

from fastapi import APIRouter, Depends

from firesat.api.schemas import RegionOut
from firesat.inference import InferenceService, get_inference_service

router = APIRouter(prefix="/api/regions", tags=["regions"])


@router.get("", response_model=list[RegionOut])
def list_regions(service: InferenceService = Depends(get_inference_service)) -> list[dict]:
    return service.list_regions()
