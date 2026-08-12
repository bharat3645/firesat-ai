from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from firesat.config import REGIONS
from firesat.data.pipeline import save_region_dataset
from firesat.data.synthetic import SyntheticDataGenerator
from firesat.inference import InferenceService
from firesat.training.train import TrainConfig, train_firesat_model

SEQ_LEN = 6


@pytest.fixture(scope="module")
def api_service(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("api_data")
    data_dir = tmp_dir / "processed"

    datasets = {}
    for i, (region_id, region) in enumerate(REGIONS.items()):
        gen = SyntheticDataGenerator(region=region, start_year=2020, end_year=2021, grid_size=4, seed=10 + i)
        ds = gen.generate()
        datasets[region_id] = ds
        save_region_dataset(ds, data_dir)

    config = TrainConfig(epochs=1, batch_size=4, sequence_length=SEQ_LEN, val_fraction=0.3)
    checkpoint_path = tmp_dir / "model.pt"
    train_firesat_model(datasets, config=config, checkpoint_path=checkpoint_path)

    service = InferenceService(checkpoint_path=checkpoint_path, data_dir=data_dir)
    service.load()
    assert service.ready, service.status_message
    return service


@pytest.fixture
def client(api_service):
    from firesat.api.main import app
    from firesat.inference import get_inference_service

    app.dependency_overrides[get_inference_service] = lambda: api_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint_ready(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert set(body["regions_loaded"]) == set(REGIONS.keys())


def test_regions_endpoint_lists_both_study_regions(client):
    response = client.get("/api/regions")
    assert response.status_code == 200
    ids = {r["id"] for r in response.json()}
    assert ids == set(REGIONS.keys())


@pytest.mark.parametrize("region_id", list(REGIONS.keys()))
def test_current_risk_endpoint(client, region_id):
    response = client.get(f"/api/risk/{region_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["region_id"] == region_id
    assert set(body["horizons"].keys()) == {"horizon_1m", "horizon_3m", "horizon_6m"}
    for horizon in body["horizons"].values():
        probs = horizon["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-3
        assert horizon["risk_class"] in {"No Risk", "Moderate", "High"}
    assert len(body["channel_attention"]) == 6
    assert all(0.0 <= v <= 1.0 for v in body["channel_attention"].values())


def test_risk_endpoint_unknown_region_returns_404(client):
    response = client.get("/api/risk/not-a-real-region")
    assert response.status_code == 404


def test_risk_history_endpoint(client):
    region_id = next(iter(REGIONS.keys()))
    response = client.get(f"/api/risk/{region_id}/history", params={"months": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body) <= 5
    assert len(body) > 0
    times = [p["as_of"] for p in body]
    assert times == sorted(times)  # chronological


def test_fire_history_endpoint(client):
    region_id = next(iter(REGIONS.keys()))
    response = client.get(f"/api/risk/{region_id}/fire-history")
    assert response.status_code == 200
    body = response.json()
    assert body["region_id"] == region_id
    assert body["n_fire_events"] == len(body["ignitions"])
