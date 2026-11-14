"""Tests for the REST API (FastAPI TestClient, no real server needed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from demand_forecast.api import create_app
from demand_forecast.data import save_demand
from demand_forecast.models import MODEL_FACTORIES
from demand_forecast.train import train


@pytest.fixture(scope="module")
def client(small_demand, tmp_path_factory):
    out = tmp_path_factory.mktemp("api")
    save_demand(small_demand, out / "demand.csv")
    train(
        small_demand,
        out / "models",
        out / "reports",
        factories={"Moving average (28d)": MODEL_FACTORIES["Moving average (28d)"]},
    )
    app = create_app(model_path=out / "models" / "model.joblib", data_path=out / "demand.csv")
    with TestClient(app) as c:
        yield c


def body(**overrides):
    data = {
        "hospital_id": "H01",
        "medicine_id": "AMOXICILLIN",
        "current_stock": 200,
        "lead_time_days": 7,
    }
    return data | overrides


def test_health(client):
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model": "Moving average (28d)",
        "trained_until": "2025-02-28",
    }


def test_lists_hospitals_and_medicines(client):
    hospitals = client.get("/hospitals").json()
    medicines = client.get("/medicines").json()

    assert {"hospital_id": "H01", "name": "Amman Central Hospital", "city": "Amman"} in hospitals
    assert any(m["medicine_id"] == "AMOXICILLIN" for m in medicines)


def test_forecast_returns_14_days_and_a_recommendation(client):
    r = client.post("/forecast", json=body())

    assert r.status_code == 200
    data = r.json()
    assert len(data["forecast"]) == 14
    assert data["forecast"][0]["date"] == "2025-03-01"
    assert data["recommendation"]["status"] in {"Urgent order", "Reorder", "OK", "Overstock"}


def test_unknown_medicine_is_404(client):
    r = client.post("/forecast", json=body(medicine_id="ASPIRIN"))

    assert r.status_code == 404
    assert "ASPIRIN" in r.json()["detail"]


@pytest.mark.parametrize(
    "bad",
    [
        {"current_stock": -1},
        {"lead_time_days": 0},
        {"lead_time_days": 15},
        {"service_level": 1.5},
        {"hospital_id": ""},
    ],
)
def test_invalid_input_is_422(client, bad):
    r = client.post("/forecast", json=body(**bad))

    assert r.status_code == 422


def test_missing_model_file_fails_at_startup(tmp_path):
    app = create_app(model_path=tmp_path / "nope.joblib", data_path=tmp_path / "nope.csv")

    with pytest.raises(FileNotFoundError), TestClient(app):
        pass
