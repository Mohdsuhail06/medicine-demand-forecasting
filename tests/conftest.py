"""Shared fixtures: a small simulated dataset that keeps the tests fast."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from demand_forecast.data import HOSPITALS, MEDICINES, generate_demand


@pytest.fixture(scope="session")
def small_demand() -> pd.DataFrame:
    """Two hospitals x three medicines x ~14 months of daily demand."""
    return generate_demand(
        start=dt.date(2024, 1, 1),
        end=dt.date(2025, 2, 28),
        hospitals=HOSPITALS[:2],
        medicines=MEDICINES[:3],
        seed=7,
    )
