"""FireSat-AI FastAPI application.

Serves the JSON risk/region/interpretability API under ``/api/*`` and, when
the ``dashboard/`` directory is present, the static Leaflet dashboard at
``/`` -- so ``uvicorn firesat.api.main:app`` alone is enough to get both the
API and the map UI running.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from firesat.api.routers import regions, risk
from firesat.api.schemas import HealthOut
from firesat.config import ROOT_DIR
from firesat.inference import get_inference_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    service = get_inference_service()
    if service.ready:
        logger.info("FireSat-AI API ready. Regions: %s", list(service.datasets.keys()))
    else:
        logger.warning(
            "FireSat-AI API starting in NOT READY state: %s", service.status_message
        )
    yield


app = FastAPI(
    title="FireSat-AI API",
    description=(
        "Hybrid CNN-LSTM + attention wildfire risk forecasting for Alaska. "
        "Scoped to two study regions (Interior/Fairbanks, Kenai Peninsula) "
        "as a reproducible applied climate-tech MVP -- see README for scope "
        "and honest evaluation notes."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(regions.router)
app.include_router(risk.router)


@app.get("/api/health", response_model=HealthOut)
def health(service=Depends(get_inference_service)) -> HealthOut:
    return HealthOut(
        status="ok",
        ready=service.ready,
        detail=service.status_message,
        regions_loaded=list(service.datasets.keys()),
    )


_dashboard_dir = ROOT_DIR / "dashboard"
if _dashboard_dir.exists():
    app.mount("/", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")
