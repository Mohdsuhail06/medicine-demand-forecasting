"""Command-line interface.

Examples:
    demand-forecast generate
    demand-forecast train
    demand-forecast forecast --hospital H01 --medicine AMOXICILLIN --stock 400 --lead-time 7
    demand-forecast serve
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from demand_forecast.data import generate_demand, load_demand, save_demand
from demand_forecast.models import MODEL_FACTORIES

DEFAULT_DATA = Path("data/demand.csv")
DEFAULT_MODEL = Path("models/model.joblib")


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' is not a YYYY-MM-DD date") from exc


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="demand-forecast", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="simulate hospital demand data")
    g.add_argument("--start", type=_date, default=dt.date(2023, 1, 1))
    g.add_argument("--end", type=_date, default=dt.date(2025, 12, 31))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", type=Path, default=DEFAULT_DATA)

    t = sub.add_parser("train", help="compare models, save the best one and a report")
    t.add_argument("--data", type=Path, default=DEFAULT_DATA)
    t.add_argument("--models-dir", type=Path, default=Path("models"))
    t.add_argument("--reports-dir", type=Path, default=Path("reports"))
    t.add_argument("--models", nargs="+", help=f"subset of: {', '.join(MODEL_FACTORIES)}")

    f = sub.add_parser("forecast", help="forecast one hospital/medicine and suggest an order")
    f.add_argument("--data", type=Path, default=DEFAULT_DATA)
    f.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    f.add_argument("--hospital", required=True)
    f.add_argument("--medicine", required=True)
    f.add_argument("--stock", type=int, required=True, help="units currently in stock")
    f.add_argument("--lead-time", type=int, default=7, help="supplier lead time in days (1-14)")
    f.add_argument("--json", action="store_true", help="print raw JSON")

    s = sub.add_parser("serve", help="start the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return p


def _cmd_generate(args) -> int:
    df = generate_demand(args.start, args.end, seed=args.seed)
    save_demand(df, args.out)
    print(f"Wrote {len(df):,} rows to {args.out}")
    return 0


def _cmd_train(args) -> int:
    from demand_forecast.train import train  # heavy imports only when needed

    if not args.data.exists():
        print(
            f"error: {args.data} not found. Run 'demand-forecast generate' first.", file=sys.stderr
        )
        return 2
    factories = MODEL_FACTORIES
    if args.models:
        unknown = [m for m in args.models if m not in MODEL_FACTORIES]
        if unknown:
            print(f"error: Unknown model(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        factories = {m: MODEL_FACTORIES[m] for m in args.models}
    try:
        result = train(load_demand(args.data), args.models_dir, args.reports_dir, factories)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Holdout scores:")
    print(result.holdout_scores.round(2).to_string(index=False))
    print(f"\nBest model: {result.best_model} -> saved to {result.bundle_path}")
    return 0


def _cmd_forecast(args) -> int:
    from demand_forecast.forecast import ModelBundle, recommend

    for path, step in ((args.model, "train"), (args.data, "generate")):
        if not path.exists():
            print(f"error: {path} not found. Run 'demand-forecast {step}' first.", file=sys.stderr)
            return 2
    bundle = ModelBundle.load(args.model)
    try:
        result = recommend(
            bundle, load_demand(args.data), args.hospital, args.medicine, args.stock, args.lead_time
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    rec = result["recommendation"]
    print(f"{args.hospital} / {args.medicine}  (model: {result['model']})")
    for day in result["forecast"]:
        print(f"  {day['date']}  {day['units']:8.1f} units")
    print(f"\nStatus: {rec['status']}  |  order now: {rec['order_quantity']} units")
    print(
        f"Expected demand in {args.lead_time} days: {rec['expected_demand']}  "
        f"+ safety stock {rec['safety_stock']}"
    )
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("demand_forecast.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands = {
        "generate": _cmd_generate,
        "train": _cmd_train,
        "forecast": _cmd_forecast,
        "serve": _cmd_serve,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
