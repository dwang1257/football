from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from football.ingest.weather import coverage_table, forecast_errors, summarize_coverage
from football.schemas import UTC_TS

KICKOFF = datetime(2015, 10, 4, 17, tzinfo=UTC)


def venues(n: int, station: str = "KGRB", season: int = 2015) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"{season}_{station}_{i}" for i in range(n)],
            "season": [season] * n,
            "kickoff_utc": [KICKOFF.replace(year=season) + timedelta(days=i) for i in range(n)],
            "roof_type": ["outdoors"] * n,
            "weather_station": [station] * n,
        },
        schema_overrides={"kickoff_utc": UTC_TS, "season": pl.Int32()},
    )


def forecasts(games: pl.DataFrame, lead: timedelta) -> pl.DataFrame:
    return games.select(
        "game_id",
        pl.lit("mos_gfs").alias("source"),
        (pl.col("kickoff_utc") - timedelta(hours=12)).alias("run_time"),
        pl.lit(50.0).alias("temp_f"),
        pl.lit(10.0).alias("wind_mph"),
        (pl.col("kickoff_utc") - lead).alias("available_at"),
    )


def test_coverage_counts_only_pre_kickoff_forecasts() -> None:
    games = pl.concat([venues(19), venues(1, "EGLL")])
    early = forecasts(games.head(19), timedelta(hours=6))
    table = coverage_table(games, early)
    passed, detail = summarize_coverage(table)
    assert passed
    assert "19/20" in detail
    assert "1 at non-US venues" in detail
    late = forecasts(games.head(19), -timedelta(hours=1))
    passed, detail = summarize_coverage(coverage_table(games, late))
    assert not passed
    assert detail.startswith("below 90%: [2015]")


def test_coverage_ignores_2000_shortfall() -> None:
    games = venues(10, season=2000)
    passed, _ = summarize_coverage(coverage_table(games, forecasts(games.head(5), timedelta(1))))
    assert passed


def test_forecast_errors_use_latest_pre_kickoff_run() -> None:
    games = venues(2)
    early = forecasts(games, timedelta(hours=10))
    late = forecasts(games, timedelta(hours=2)).with_columns(
        pl.col("run_time") + timedelta(hours=8), pl.lit(60.0).alias("temp_f")
    )
    after = forecasts(games, -timedelta(hours=1)).with_columns(
        pl.col("run_time") + timedelta(hours=20), pl.lit(99.0).alias("temp_f")
    )
    observed = games.select(
        "game_id",
        pl.lit("nflverse").alias("source"),
        pl.lit(58.0).alias("temp_f"),
        pl.lit(13.0).alias("wind_mph"),
    )
    errors = forecast_errors(games, pl.concat([early, late, after]), observed)
    row = errors.row(0, named=True)
    assert row["games"] == 2
    assert row["temp_mae"] == pytest.approx(2.0)
    assert row["wind_bias"] == pytest.approx(-3.0)
