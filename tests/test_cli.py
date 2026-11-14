"""Tests for the `demand-forecast` command line."""

from __future__ import annotations

import json

from demand_forecast import cli


def test_generate_writes_a_csv(tmp_path, capsys):
    out = tmp_path / "demand.csv"

    code = cli.main(["generate", "--start", "2025-01-01", "--end", "2025-01-31", "--out", str(out)])

    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "date,hospital_id,medicine_id,category,units"
    assert len(lines) == 1 + 31 * 5 * 12
    assert "rows" in capsys.readouterr().out


def test_train_then_forecast(tmp_path, capsys):
    data = tmp_path / "demand.csv"
    cli.main(["generate", "--start", "2024-01-01", "--end", "2025-02-28", "--out", str(data)])

    code = cli.main(
        [
            "train",
            "--data",
            str(data),
            "--models-dir",
            str(tmp_path / "m"),
            "--reports-dir",
            str(tmp_path / "r"),
            "--models",
            "Seasonal naive",
            "Moving average (28d)",
        ]
    )
    assert code == 0
    capsys.readouterr()

    code = cli.main(
        [
            "forecast",
            "--data",
            str(data),
            "--model",
            str(tmp_path / "m" / "model.joblib"),
            "--hospital",
            "H02",
            "--medicine",
            "METFORMIN",
            "--stock",
            "300",
            "--lead-time",
            "5",
            "--json",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["hospital_id"] == "H02"
    assert len(result["forecast"]) == 14


def test_unknown_model_name_is_an_error(tmp_path, capsys):
    data = tmp_path / "demand.csv"
    cli.main(["generate", "--start", "2025-01-01", "--end", "2025-01-31", "--out", str(data)])

    code = cli.main(["train", "--data", str(data), "--models", "Magic model"])

    assert code == 2
    assert "Unknown model" in capsys.readouterr().err


def test_forecast_for_unknown_medicine_is_an_error(tmp_path, capsys):
    data = tmp_path / "demand.csv"
    cli.main(["generate", "--start", "2024-01-01", "--end", "2025-02-28", "--out", str(data)])
    cli.main(
        [
            "train",
            "--data",
            str(data),
            "--models-dir",
            str(tmp_path / "m"),
            "--reports-dir",
            str(tmp_path / "r"),
            "--models",
            "Seasonal naive",
        ]
    )

    code = cli.main(
        [
            "forecast",
            "--data",
            str(data),
            "--model",
            str(tmp_path / "m" / "model.joblib"),
            "--hospital",
            "H01",
            "--medicine",
            "ASPIRIN",
            "--stock",
            "1",
        ]
    )

    assert code == 2
    assert "ASPIRIN" in capsys.readouterr().err
