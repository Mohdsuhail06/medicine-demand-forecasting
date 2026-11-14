"""Realistic simulated hospital medicine demand.

Real dispensing records are confidential, so this module simulates them. Every pattern is
written down explicitly below, which lets us check whether the model really learns them:

* hospital size          - bigger hospitals dispense more
* long-term trend        - chronic-disease medicines grow every year
* weekly cycle           - Friday/Saturday (the Jordanian weekend) are quieter
* yearly seasonality     - antibiotics peak in winter, rehydration salts in summer, ...
* Ramadan                - fewer daytime clinic visits, more antacid use
* outbreaks              - random flu waves push antibiotic/respiratory demand up
* noise                  - negative-binomial counts (real counts are noisier than Poisson)
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEMAND_COLUMNS = ["date", "hospital_id", "medicine_id", "category", "units"]

# Reference date for the trend, so the same day always gets the same trend level.
_TREND_ORIGIN = dt.date(2023, 1, 1)
# Negative-binomial dispersion: lower = noisier.
_DISPERSION = 25.0


@dataclass(frozen=True)
class Hospital:
    hospital_id: str
    name: str
    city: str
    size_factor: float


@dataclass(frozen=True)
class Medicine:
    medicine_id: str
    name: str
    category: str
    base_daily_units: float  # average units per day in a hospital of size 1.0
    annual_growth: float  # e.g. 0.05 = +5% per year
    peak_day_of_year: int  # day of the year with the highest seasonal demand
    seasonal_amplitude: float  # 0.4 = +40% at the peak, -40% at the trough
    unit_price_jod: float


# Fictional hospitals (no real hospital data is used).
HOSPITALS: tuple[Hospital, ...] = (
    Hospital("H01", "Amman Central Hospital", "Amman", 3.0),
    Hospital("H02", "Irbid North Hospital", "Irbid", 2.0),
    Hospital("H03", "Zarqa General Hospital", "Zarqa", 1.6),
    Hospital("H04", "Aqaba Coastal Hospital", "Aqaba", 0.9),
    Hospital("H05", "Karak South Hospital", "Karak", 0.7),
)

MEDICINES: tuple[Medicine, ...] = (
    Medicine("AMOXICILLIN", "Amoxicillin 500mg", "Antibiotic", 40, 0.02, 15, 0.45, 0.35),
    Medicine("AZITHROMYCIN", "Azithromycin 250mg", "Antibiotic", 18, 0.02, 20, 0.40, 0.90),
    Medicine("PARACETAMOL", "Paracetamol 500mg", "Analgesic", 120, 0.01, 20, 0.15, 0.05),
    Medicine("IBUPROFEN", "Ibuprofen 400mg", "Analgesic", 60, 0.01, 180, 0.05, 0.08),
    Medicine("SALBUTAMOL", "Salbutamol inhaler", "Respiratory", 15, 0.03, 25, 0.35, 3.20),
    Medicine("CETIRIZINE", "Cetirizine 10mg", "Allergy", 20, 0.02, 105, 0.40, 0.15),
    Medicine("INSULIN_GLARGINE", "Insulin glargine pen", "Diabetes", 12, 0.08, 1, 0.03, 9.50),
    Medicine("METFORMIN", "Metformin 850mg", "Diabetes", 55, 0.06, 1, 0.03, 0.06),
    Medicine("ATORVASTATIN", "Atorvastatin 20mg", "Cardiovascular", 35, 0.05, 1, 0.03, 0.25),
    Medicine("AMLODIPINE", "Amlodipine 5mg", "Cardiovascular", 30, 0.04, 1, 0.03, 0.12),
    Medicine("OMEPRAZOLE", "Omeprazole 20mg", "Gastrointestinal", 45, 0.02, 1, 0.05, 0.10),
    Medicine("ORS", "Oral rehydration salts", "Gastrointestinal", 14, 0.01, 200, 0.50, 0.30),
)

# Monday=0 ... Sunday=6. Friday and Saturday are the weekend in Jordan.
WEEKDAY_FACTOR = (1.05, 1.0, 1.0, 1.0, 0.60, 0.75, 1.10)

# First and last day of Ramadan (approximate, based on the published calendars).
RAMADAN_PERIODS = (
    (dt.date(2023, 3, 23), dt.date(2023, 4, 20)),
    (dt.date(2024, 3, 11), dt.date(2024, 4, 9)),
    (dt.date(2025, 3, 1), dt.date(2025, 3, 29)),
    (dt.date(2026, 2, 18), dt.date(2026, 3, 19)),
    (dt.date(2027, 2, 8), dt.date(2027, 3, 9)),
    (dt.date(2028, 1, 28), dt.date(2028, 2, 25)),
    (dt.date(2029, 1, 16), dt.date(2029, 2, 13)),
    (dt.date(2030, 1, 6), dt.date(2030, 2, 4)),
)
OUTBREAK_CATEGORIES = {"Antibiotic", "Respiratory"}


def is_ramadan(day: dt.date) -> bool:
    """True if `day` falls inside a known Ramadan period."""
    return any(start <= day <= end for start, end in RAMADAN_PERIODS)


def _seasonal(day_of_year: np.ndarray, med: Medicine) -> np.ndarray:
    angle = 2 * np.pi * (day_of_year - med.peak_day_of_year) / 365.25
    return 1 + med.seasonal_amplitude * np.cos(angle)


def _outbreaks(n_days: int, doy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative bumps from random winter flu waves (about 2 per winter)."""
    factor = np.ones(n_days)
    winter = (doy <= 60) | (doy >= 330)
    starts = np.flatnonzero(winter & (rng.random(n_days) < 2 / 120))
    for start in starts:
        length = int(rng.integers(7, 15))
        strength = rng.uniform(0.3, 0.8)
        bump = strength * np.sin(np.linspace(0, np.pi, length))
        end = min(start + length, n_days)
        factor[start:end] += bump[: end - start]
    return factor


def generate_demand(
    start: dt.date,
    end: dt.date,
    hospitals: Sequence[Hospital] = HOSPITALS,
    medicines: Sequence[Medicine] = MEDICINES,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate daily units dispensed for every hospital x medicine between start and end.

    The result is deterministic for a given seed.
    """
    if start > end:
        raise ValueError("start must be on or before end")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.day_of_year.to_numpy()
    years = ((dates - pd.Timestamp(_TREND_ORIGIN)).days / 365.25).to_numpy()
    weekday = np.array(WEEKDAY_FACTOR)[dates.day_of_week.to_numpy()]
    ramadan = np.array([is_ramadan(d.date()) for d in dates])
    day_list = [d.date() for d in dates]

    frames = []
    for hospital in hospitals:
        flu = _outbreaks(n, doy, rng)
        for med in medicines:
            mean = (
                med.base_daily_units
                * hospital.size_factor
                * (1 + med.annual_growth) ** years
                * weekday
                * _seasonal(doy, med)
            )
            if med.category == "Gastrointestinal" and med.medicine_id == "OMEPRAZOLE":
                mean = mean * np.where(ramadan, 1.25, 1.0)
            else:
                mean = mean * np.where(ramadan, 0.9, 1.0)
            if med.category in OUTBREAK_CATEGORIES:
                mean = mean * flu
            # Negative binomial = Poisson with a gamma-distributed rate.
            rate = rng.gamma(shape=_DISPERSION, scale=mean / _DISPERSION)
            units = rng.poisson(rate)
            frames.append(
                pd.DataFrame(
                    {
                        "date": day_list,
                        "hospital_id": hospital.hospital_id,
                        "medicine_id": med.medicine_id,
                        "category": med.category,
                        "units": units.astype("int64"),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)[DEMAND_COLUMNS]


def save_demand(df: pd.DataFrame, path: Path | str) -> None:
    """Write the demand table as CSV (dates in ISO format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_demand(path: Path | str) -> pd.DataFrame:
    """Read a demand CSV written by `save_demand`."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["units"] = df["units"].astype("int64")
    return df[DEMAND_COLUMNS]
