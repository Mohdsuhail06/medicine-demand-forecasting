"""REST API that serves forecasts and reorder recommendations.

Run with:  uvicorn demand_forecast.api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive documentation.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from demand_forecast.data import HOSPITALS, MEDICINES, load_demand
from demand_forecast.features import HORIZON
from demand_forecast.forecast import ModelBundle, recommend


class ForecastRequest(BaseModel):
    hospital_id: str = Field(min_length=1, max_length=20, examples=["H01"])
    medicine_id: str = Field(min_length=1, max_length=40, examples=["AMOXICILLIN"])
    current_stock: int = Field(ge=0, le=10_000_000, description="Units on the shelf now")
    lead_time_days: int = Field(default=7, ge=1, le=HORIZON, description="Supplier delivery time")
    service_level: float = Field(default=0.95, gt=0.5, lt=1.0, description="Target in-stock rate")


class ForecastDay(BaseModel):
    date: str
    units: float


class RecommendationOut(BaseModel):
    expected_demand: float
    safety_stock: float
    reorder_point: float
    order_quantity: int
    days_of_cover: float | None
    status: str


class ForecastResponse(BaseModel):
    hospital_id: str
    medicine_id: str
    model: str
    forecast: list[ForecastDay]
    recommendation: RecommendationOut


def create_app(
    model_path: Path | str | None = None, data_path: Path | str | None = None
) -> FastAPI:
    """Build the app. Paths default to the MODEL_PATH / DATA_PATH environment variables."""
    model_file = Path(model_path or os.environ.get("MODEL_PATH", "models/model.joblib"))
    data_file = Path(data_path or os.environ.get("DATA_PATH", "data/demand.csv"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for path in (model_file, data_file):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run 'demand-forecast train' first.")
        app.state.bundle = ModelBundle.load(model_file)
        app.state.history = load_demand(data_file)
        yield

    app = FastAPI(
        title="Medicine Demand Forecasting API",
        description="14-day demand forecasts and reorder recommendations for hospital pharmacies.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        bundle: ModelBundle = request.app.state.bundle
        return {
            "status": "ok",
            "model": bundle.name,
            "trained_until": bundle.trained_until.isoformat(),
        }

    @app.get("/hospitals")
    def hospitals() -> list[dict]:
        return [{"hospital_id": h.hospital_id, "name": h.name, "city": h.city} for h in HOSPITALS]

    @app.get("/medicines")
    def medicines() -> list[dict]:
        return [
            {"medicine_id": m.medicine_id, "name": m.name, "category": m.category}
            for m in MEDICINES
        ]

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(body: ForecastRequest, request: Request) -> dict:
        try:
            return recommend(
                request.app.state.bundle,
                request.app.state.history,
                body.hospital_id,
                body.medicine_id,
                current_stock=body.current_stock,
                lead_time_days=body.lead_time_days,
                service_level=body.service_level,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    return app


app = create_app()
