from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from football import storage
from football.ingest.weather_live import (
    keep_first_seen,
    live_targets,
    nws_rows,
    open_meteo_rows,
    wind_speed,
)
from football.ingest.weather_mos import KNOTS_TO_MPH
from football.ingest.weather_observed import (
    iem_station,
    nflverse_observed,
    parse_asos,
    summarize_asos,
)
from football.schemas import UTC_TS, WEATHER_FORECASTS, WEATHER_OBSERVED, validate

KICKOFF = datetime(2022, 12, 18, 18, tzinfo=UTC)


def targets() -> pl.DataFrame:
    return pl.DataFrame(
        {"game_id": ["g1", "g2"], "kickoff_utc": [KICKOFF, KICKOFF], "station": ["KGRB", "KGRB"]},
        schema={"game_id": pl.String(), "kickoff_utc": UTC_TS, "station": pl.String()},
    )


def test_nflverse_observed_uses_reported_values() -> None:
    schedules = pl.DataFrame(
        {"game_id": ["g1", "g2", "g3"], "temp": [20, None, 50], "wind": [12, 5, 3]},
        schema={"game_id": pl.String(), "temp": pl.Int32(), "wind": pl.Int32()},
    )
    rows = nflverse_observed(schedules, targets())
    assert rows["game_id"].to_list() == ["g1"]
    assert rows["temp_f"].to_list() == [20.0]
    assert rows["available_at"].to_list() == [KICKOFF + timedelta(hours=4)]


def test_summarize_asos_nearest_and_precip() -> None:
    payload = (
        b"station,valid,tmpf,sknt,p01i\n"
        b"GRB,2022-12-18 16:53,21.0,8.00,0.00\n"
        b"GRB,2022-12-18 17:53,23.0,10.00,0.01\n"
        b"GRB,2022-12-18 18:53,24.0,11.00,0.02\n"
        b"GRB,2022-12-18 19:53,25.0,,0.03\n"
        b"GRB,2022-12-18 22:53,25.0,9.00,0.50\n"
    )
    obs = parse_asos(payload)
    rows = validate(summarize_asos(obs, targets().head(1), True), WEATHER_OBSERVED)
    row = rows.row(0, named=True)
    assert row["temp_f"] == 23.0
    assert row["wind_mph"] == pytest.approx(10 * KNOTS_TO_MPH)
    assert row["precip_in"] == pytest.approx(0.05)
    assert row["source"] == "iem_asos"
    foreign = summarize_asos(obs, targets().head(1), False)
    assert foreign["precip_in"].null_count() == 1
    assert iem_station("KGRB") == "GRB"
    assert iem_station("EGLL") == "EGLL"


def test_wind_speed_parses_ranges() -> None:
    assert wind_speed("5 to 10 mph") == 7.5
    assert wind_speed("12 mph") == 12.0
    assert wind_speed(None) is None


def hourly_payload() -> dict[str, object]:
    periods = [
        {
            "startTime": (KICKOFF + timedelta(hours=h)).astimezone().isoformat(),
            "endTime": (KICKOFF + timedelta(hours=h + 1)).astimezone().isoformat(),
            "temperature": 30 + h,
            "temperatureUnit": "F",
            "windSpeed": f"{5 + h} mph",
            "probabilityOfPrecipitation": {"value": 20},
        }
        for h in range(-2, 3)
    ]
    return {"properties": {"updateTime": "2022-12-17T12:00:00+00:00", "periods": periods}}


def test_nws_rows_pick_hour_containing_kickoff() -> None:
    games = pl.DataFrame(
        {"game_id": ["g1"], "kickoff_utc": [KICKOFF + timedelta(minutes=25)]},
        schema={"game_id": pl.String(), "kickoff_utc": UTC_TS},
    )
    fetched = datetime(2022, 12, 17, 13, tzinfo=UTC)
    rows = validate(nws_rows(hourly_payload(), games, fetched), WEATHER_FORECASTS)
    row = rows.row(0, named=True)
    assert row["valid_time"] == KICKOFF
    assert row["temp_f"] == 30.0
    assert row["wind_mph"] == 5.0
    assert row["precip_prob"] == 0.2
    assert row["run_time"] == datetime(2022, 12, 17, 12, tzinfo=UTC)
    assert row["available_at"] == fetched


def test_open_meteo_rows() -> None:
    payload = {
        "hourly": {
            "time": ["2022-12-18T17:00", "2022-12-18T18:00", "2022-12-18T19:00"],
            "temperature_2m": [40.0, 41.5, 42.0],
            "wind_speed_10m": [5.0, 6.0, 7.0],
            "wind_gusts_10m": [9.0, 11.0, 12.0],
            "precipitation_probability": [0, 35, 50],
        }
    }
    games = pl.DataFrame(
        {"game_id": ["g1"], "kickoff_utc": [KICKOFF + timedelta(minutes=30)]},
        schema={"game_id": pl.String(), "kickoff_utc": UTC_TS},
    )
    fetched = datetime(2022, 12, 16, tzinfo=UTC)
    rows = validate(open_meteo_rows(payload, games, fetched), WEATHER_FORECASTS)
    assert rows.select("temp_f", "wind_gust_mph", "precip_prob").row(0) == (41.5, 11.0, 0.35)
    assert rows["source"].to_list() == ["open_meteo"]


def test_live_targets_and_first_seen() -> None:
    now = KICKOFF - timedelta(days=2)
    venues = pl.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "stadium_id": ["S1", "S2", "S3", "S4"],
            "kickoff_utc": [
                KICKOFF,
                KICKOFF,
                KICKOFF + timedelta(days=9),
                now - timedelta(hours=1),
            ],
            "lat": [1.0] * 4,
            "lon": [1.0] * 4,
            "is_international": [False] * 4,
            "roof_type": ["outdoors", "dome", "outdoors", "retractable"],
        },
        schema_overrides={"kickoff_utc": UTC_TS},
    )
    assert live_targets(venues, now)["game_id"].to_list() == ["a"]
    games = pl.DataFrame(
        {"game_id": ["g1"], "kickoff_utc": [KICKOFF]},
        schema={"game_id": pl.String(), "kickoff_utc": UTC_TS},
    )
    first = validate(nws_rows(hourly_payload(), games, now), WEATHER_FORECASTS)
    storage.upsert(first, "clean", WEATHER_FORECASTS.name, WEATHER_FORECASTS.keys)
    again = validate(nws_rows(hourly_payload(), games, now + timedelta(hours=1)), WEATHER_FORECASTS)
    assert keep_first_seen(again).is_empty()
