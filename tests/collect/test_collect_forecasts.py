from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from collect_fixtures import snapshot, ts

from football.collect import forecasts
from football.collect.archive import iter_snapshots
from football.collect.base import CollectorResult, Context
from football.collect.http import Fetcher
from football.schemas import WEATHER_FORECASTS, validate

NOW = ts("2026-10-31T13:00:00+00:00")

STADIUMS = """stadium_id,name,lat,lon,roof_type,is_international
BUF00,Highmark Stadium,42.7738,-78.7870,outdoors,false
DET00,Ford Field,42.3400,-83.0456,dome,false
DAL00,AT&T Stadium,32.7473,-97.0945,retractable,false
MAD01,Bernabeu,40.4531,-3.6883,retractable,true
NOWHERE,Mystery,,,outdoors,false
"""


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [
                "2026_08_BAL_BUF",
                "2026_08_MIN_DET",
                "2026_08_ARI_DAL",
                "2026_09_WAS_MIA",
                "2026_09_X_Y",
                "2026_10_BAL_BUF",
                "2026_07_NYJ_BUF",
            ],
            "gameday": [
                "2026-11-01",
                "2026-11-01",
                "2026-11-01",
                "2026-11-06",
                "2026-11-05",
                "2026-11-15",
                "2026-10-25",
            ],
            "gametime": ["13:00", "13:00", "16:25", "09:30", "20:15", "13:00", "13:00"],
            "stadium_id": ["BUF00", "DET00", "DAL00", "MAD01", "NOWHERE", "BUF00", "BUF00"],
            "roof": ["outdoors", "dome", None, "outdoors", "outdoors", "outdoors", "outdoors"],
        }
    )


def _hourly(start: str, end: str) -> dict[str, Any]:
    return {
        "properties": {
            "updateTime": "2026-10-31T11:06:34+00:00",
            "generatedAt": "2026-10-31T12:00:00+00:00",
            "periods": [
                {
                    "startTime": start,
                    "endTime": end,
                    "temperature": 41,
                    "temperatureUnit": "F",
                    "windSpeed": "10 to 15 mph",
                    "probabilityOfPrecipitation": {"value": 40},
                }
            ],
        }
    }


def test_kickoff_utc_handles_daylight_time() -> None:
    assert forecasts.kickoff_utc("2026-10-25", "13:00") == ts("2026-10-25T17:00:00+00:00")
    assert forecasts.kickoff_utc("2026-11-01", "13:00") == ts("2026-11-01T18:00:00+00:00")


def test_targets_skip_domes_and_unknown_window(tmp_path: Path) -> None:
    csv = tmp_path / "stadiums.csv"
    csv.write_text(STADIUMS)
    games = forecasts.upcoming_games(_schedule(), NOW)
    assert games["game_id"].to_list() == [
        "2026_08_BAL_BUF",
        "2026_08_MIN_DET",
        "2026_08_ARI_DAL",
        "2026_09_X_Y",
        "2026_09_WAS_MIA",
    ]
    targets = forecasts.forecast_targets(games, forecasts.load_stadiums(csv))
    assert targets["game_id"].to_list() == [
        "2026_08_BAL_BUF",
        "2026_08_ARI_DAL",
        "2026_09_X_Y",
        "2026_09_WAS_MIA",
    ]


def _transport(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.startswith("/points/42.7738"):
        hourly = "https://api.weather.gov/gridpoints/BUF/39,42/forecast/hourly"
        return httpx.Response(
            200,
            json={
                "properties": {"gridId": "BUF", "gridX": 39, "gridY": 42, "forecastHourly": hourly}
            },
        )
    if path.startswith("/points/32.7473"):
        return httpx.Response(500, json={"title": "Unexpected Problem"})
    if path == "/gridpoints/BUF/39,42/forecast/hourly":
        return httpx.Response(
            200, json=_hourly("2026-11-01T13:00:00-05:00", "2026-11-01T14:00:00-05:00")
        )
    return httpx.Response(404)


def test_collect_writes_one_snapshot_per_stadium(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "stadiums.csv"
    csv.write_text(STADIUMS)
    monkeypatch.setattr(forecasts, "stadiums_csv", lambda: csv)
    monkeypatch.setattr(forecasts, "load_schedule", lambda season: _schedule())
    fetcher = Fetcher(transport=httpx.MockTransport(_transport), retries=0, min_interval_s=0)
    result = CollectorResult(collector=forecasts.NAME)
    forecasts.collect(Context(fetcher=fetcher, run_id="r1", started_at=NOW), result)
    assert result.ok == ["BUF00"]
    assert set(result.failed) == {"2026_09_X_Y", "DAL00"}
    assert any("2026_09_WAS_MIA" in note for note in result.notes)
    (stored,) = list(iter_snapshots(forecasts.NAME))
    assert stored.meta["grid"] == "BUF/39,42"
    assert [game["game_id"] for game in stored.meta["games"]] == ["2026_08_BAL_BUF"]
    rows = forecasts.forecast_rows([stored])
    frame = validate(pl.DataFrame(rows, schema=WEATHER_FORECASTS.columns), WEATHER_FORECASTS)
    row = frame.row(0, named=True)
    assert row["valid_time"] == ts("2026-11-01T18:00:00+00:00")
    assert row["run_time"] == ts("2026-10-31T11:06:34+00:00")
    assert (row["temp_f"], row["wind_mph"], row["precip_prob"]) == (41.0, 15.0, 0.4)
    assert row["available_at"] == stored.fetched_at


def test_collect_skips_without_stadium_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forecasts, "stadiums_csv", lambda: tmp_path / "missing.csv")
    result = CollectorResult(collector=forecasts.NAME)
    fetcher = Fetcher(transport=httpx.MockTransport(_transport), retries=0)
    forecasts.collect(Context(fetcher=fetcher, run_id="r1", started_at=NOW), result)
    assert result.status == "skipped"
    assert "not found" in result.notes[0]


def test_forecast_rows_never_use_post_kickoff_fetches() -> None:
    meta = {
        "grid": "BUF/39,42",
        "games": [{"game_id": "g", "kickoff_utc": "2026-11-01T18:00:00+00:00"}],
    }
    body = {"hourly": _hourly("2026-11-01T18:00:00+00:00", "2026-11-01T19:00:00+00:00")}
    late = snapshot(forecasts.NAME, "BUF00", body, "2026-11-01T18:00:00+00:00", meta)
    early = snapshot(forecasts.NAME, "BUF00", body, "2026-10-31T09:00:00+00:00", meta)
    rows = forecasts.forecast_rows([late, early])
    assert len(rows) == 1
    assert rows[0]["run_time"] == early.fetched_at
