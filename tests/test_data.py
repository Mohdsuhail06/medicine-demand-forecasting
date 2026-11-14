"""Tests for the demand simulator: realistic shape and the patterns the model should learn."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from demand_forecast.data import (
    DEMAND_COLUMNS,
    HOSPITALS,
    MEDICINES,
    generate_demand,
    is_ramadan,
)

START, END = dt.date(2023, 1, 1), dt.date(2025, 12, 31)


def full():
    return generate_demand(START, END, seed=1)


def test_one_row_per_day_hospital_and_medicine(small_demand):
    days = (dt.date(2025, 2, 28) - dt.date(2024, 1, 1)).days + 1

    assert list(small_demand.columns) == DEMAND_COLUMNS
    assert len(small_demand) == days * 2 * 3
    assert not small_demand.duplicated(["date", "hospital_id", "medicine_id"]).any()


def test_demand_is_a_non_negative_whole_number(small_demand):
    assert (small_demand["units"] >= 0).all()
    assert pd.api.types.is_integer_dtype(small_demand["units"])


def test_same_seed_gives_identical_data():
    a = generate_demand(dt.date(2025, 1, 1), dt.date(2025, 1, 31), seed=3)
    b = generate_demand(dt.date(2025, 1, 1), dt.date(2025, 1, 31), seed=3)

    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = generate_demand(dt.date(2025, 1, 1), dt.date(2025, 1, 31), seed=3)
    b = generate_demand(dt.date(2025, 1, 1), dt.date(2025, 1, 31), seed=4)

    assert not a["units"].equals(b["units"])


def test_larger_hospitals_dispense_more():
    df = full()
    by_hospital = df.groupby("hospital_id")["units"].mean()
    largest = max(HOSPITALS, key=lambda h: h.size_factor).hospital_id
    smallest = min(HOSPITALS, key=lambda h: h.size_factor).hospital_id

    assert by_hospital[largest] > 2 * by_hospital[smallest]


def test_weekend_demand_is_lower_than_weekdays():
    # In Jordan the weekend is Friday and Saturday, when outpatient clinics are closed.
    df = full()
    weekday = pd.to_datetime(df["date"]).dt.day_of_week
    weekend = df[weekday.isin([4, 5])]["units"].mean()
    workday = df[~weekday.isin([4, 5])]["units"].mean()

    assert weekend < 0.85 * workday


def test_antibiotics_peak_in_winter():
    df = full()
    abx = df[df["category"] == "Antibiotic"]
    month = pd.to_datetime(abx["date"]).dt.month
    winter = abx[month.isin([12, 1, 2])]["units"].mean()
    summer = abx[month.isin([6, 7, 8])]["units"].mean()

    assert winter > 1.3 * summer


def test_rehydration_salts_peak_in_summer():
    df = full()
    ors = df[df["category"] == "Gastrointestinal"]
    ors = ors[ors["medicine_id"] == "ORS"]
    month = pd.to_datetime(ors["date"]).dt.month

    assert (
        ors[month.isin([6, 7, 8])]["units"].mean()
        > 1.3 * ors[month.isin([12, 1, 2])]["units"].mean()
    )


def test_chronic_medicines_grow_over_time():
    df = full()
    insulin = df[df["medicine_id"] == "INSULIN_GLARGINE"]
    year = pd.to_datetime(insulin["date"]).dt.year

    assert insulin[year == 2025]["units"].mean() > insulin[year == 2023]["units"].mean()


def test_every_medicine_has_a_known_category():
    categories = {m.category for m in MEDICINES}

    assert categories >= {"Antibiotic", "Diabetes", "Cardiovascular", "Respiratory"}


def test_ramadan_dates():
    assert is_ramadan(dt.date(2024, 3, 20))
    assert is_ramadan(dt.date(2025, 3, 1))
    assert not is_ramadan(dt.date(2025, 3, 31))
    assert not is_ramadan(dt.date(2024, 6, 1))
