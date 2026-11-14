# Medicine Demand Forecasting 💊📈

[![CI](https://github.com/Mohdsuhail06/medicine-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohdsuhail06/medicine-demand-forecasting/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

Forecasts how many units of each medicine a hospital will dispense over the **next 14 days**,
and turns that forecast into a **reorder recommendation** (order now? how much?).
It is the machine-learning engine behind my
[MedPredict dashboard](https://github.com/Mohdsuhail06/medpredict) idea, built to reduce
both **shortages** and **expired stock** in hospital pharmacies.

> **Result:** the best model (Gradient Boosting with Poisson loss) cuts forecast error by
> **26%** compared with the naive baseline (WAPE **25.8% → 19.2%**) on a 3-month holdout
> it never saw during model selection. 👉 [Full results](reports/RESULTS.md)

![Model comparison](reports/model_comparison.png)

## How it works

```mermaid
flowchart LR
    D[Daily demand<br/>5 hospitals × 12 medicines<br/>2023–2025] --> F[Feature engineering<br/>lags ≥ 14 days, rolling means,<br/>calendar, Ramadan]
    F --> B[Rolling-origin backtest<br/>5 models]
    B -->|validation: pick best| M[Final model]
    B -->|holdout: confirm| M
    M --> A[FastAPI<br/>POST /forecast]
    A --> R[14-day forecast +<br/>reorder recommendation]
```

### 1. Data

Real hospital dispensing records are confidential, so the project uses a **documented
simulator** ([`data.py`](src/demand_forecast/data.py)) that produces 65,760 rows of daily
demand with realistic patterns:

| Pattern | Example |
|---|---|
| Hospital size | the large Amman hospital dispenses ~4× more than the small Karak one |
| Weekly cycle | Friday/Saturday (the Jordanian weekend) are 25–40% quieter |
| Seasonality | antibiotics and inhalers peak in winter, rehydration salts in summer, antihistamines in spring |
| Trend | diabetes and heart medicines grow 4–8% per year |
| Ramadan | fewer daytime visits; more antacid (omeprazole) use |
| Outbreaks | random winter flu waves push antibiotic demand up |
| Noise | negative-binomial counts, noisier than a simple Poisson process |

Because the true patterns are known, we can check that the model actually learns them.

### 2. Features without data leakage

To forecast day *t* we may only use what was known **14 days earlier**, so every lag is
≥ 14 days and every rolling window ends at *t − 14*
([`features.py`](src/demand_forecast/features.py)). A dedicated test proves it: changing the
last 14 days of history does not change a single feature.

### 3. Honest evaluation

A random train/test split would let the model see the future. Instead, **rolling-origin
backtesting** stands at a forecast date, trains only on the past, forecasts the next 14
days, then moves forward 14 days and repeats ([`evaluate.py`](src/demand_forecast/evaluate.py)).

| Period | Dates | Used for |
|---|---|---|
| Training | 2023-03 → each forecast origin | fitting |
| Validation | 2025-04-01 → 2025-09-30 (13 origins) | **choosing** the model |
| Holdout | 2025-10-01 → 2025-12-31 (6 origins) | **final check**, untouched until the end |

### 4. Models compared

| Model | Holdout WAPE | Holdout MAE | Bias |
|---|---|---|---|
| **Gradient boosting** (Poisson loss) | **19.2%** | **12.4** | −0.8% |
| Random forest | 19.4% | 12.6 | −1.1% |
| Ridge regression (log target) | 22.6% | 14.6 | −3.8% |
| Moving average, 28 days (baseline) | 25.1% | 16.3 | −4.4% |
| Seasonal naive (baseline) | 25.8% | 16.7 | −2.0% |

*WAPE = total absolute error ÷ total demand. Validation and holdout scores agree closely
(19.3% vs 19.2%), so the model is not overfitting. Part of the remaining error is the
random day-to-day noise built into the data, which no model can predict.*

![Forecast examples](reports/forecast_example.png)

The most important features are recent demand levels (28- and 7-day means) and the
weekend flag, which matches how the data was generated:

![Feature importance](reports/feature_importance.png)

### 5. From forecast to decision

[`inventory.py`](src/demand_forecast/inventory.py) applies the classic reorder-point method:

```
safety stock  = z(service level) × daily forecast error × √(lead time)
reorder point = forecast demand during lead time + safety stock
order now     = max(0, reorder point − current stock)
```

The daily forecast error is measured per hospital and medicine on the holdout period, so
less predictable medicines automatically get a bigger buffer. Each recommendation gets a
status: **Urgent order**, **Reorder**, **OK** or **Overstock** (more than 60 days of cover,
which risks expiry).

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Mohdsuhail06/medicine-demand-forecasting.git
cd medicine-demand-forecasting
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"

demand-forecast generate        # 1. simulate 3 years of data  -> data/demand.csv
demand-forecast train           # 2. compare models (~1 min)   -> models/ + reports/
demand-forecast forecast --hospital H01 --medicine AMOXICILLIN --stock 300 --lead-time 7
```

### REST API

```bash
demand-forecast serve           # then open http://127.0.0.1:8000/docs
```

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"hospital_id": "H01", "medicine_id": "AMOXICILLIN", "current_stock": 300, "lead_time_days": 7}'
```

```json
{
  "hospital_id": "H01",
  "medicine_id": "AMOXICILLIN",
  "model": "Gradient boosting",
  "forecast": [{"date": "2026-01-01", "units": 191.9}, {"date": "2026-01-02", "units": 118.3}, "..."],
  "recommendation": {
    "expected_demand": 1188.2,
    "safety_stock": 133.6,
    "reorder_point": 1321.8,
    "order_quantity": 1022,
    "days_of_cover": 1.8,
    "status": "Urgent order"
  }
}
```

Other endpoints: `GET /health`, `GET /hospitals`, `GET /medicines`. Inputs are validated
(for example `lead_time_days` must be 1–14), and unknown hospitals or medicines return `404`.

## Testing

66 tests: data patterns, leakage checks, metrics, backtesting, the reorder logic, the API
and the CLI. Warnings are treated as errors.

```bash
pytest --cov=demand_forecast
ruff check . && ruff format --check .
```

GitHub Actions runs everything on Python 3.11, 3.12 and 3.13 for every push.

## Project structure

```
medicine-demand-forecasting/
├── src/demand_forecast/
│   ├── data.py        # demand simulator (hospitals, medicines, patterns)
│   ├── features.py    # leakage-free features for 14-day-ahead forecasts
│   ├── models.py      # 2 baselines + Ridge, Random Forest, Gradient Boosting
│   ├── evaluate.py    # metrics and rolling-origin backtesting
│   ├── train.py       # model selection, final training, report
│   ├── forecast.py    # model bundle + inference
│   ├── inventory.py   # safety stock and reorder recommendation
│   ├── api.py         # FastAPI service
│   └── cli.py         # `demand-forecast` command
├── tests/             # pytest suite
└── reports/           # results and charts from the latest training run
```

## Limitations

- The data is **simulated**. On real hospital data the patterns would be less clean and
  extra signals (prices, stock-outs, public holidays, disease surveillance) would help.
- The model is trained once and stored with `joblib`. Only load model files you created
  yourself, because `joblib` files can run code when loaded.
- The API has no authentication; it is meant to run locally or behind a gateway.

## License

[MIT](LICENSE) © 2026 Mohammed Suhail
