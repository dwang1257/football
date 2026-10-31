from datetime import UTC, datetime

import polars as pl

from football.ingest.lines_common import prepare_schedule

FETCHED_AT = datetime(2026, 10, 31, 18, 0, tzinfo=UTC)

GAMES: list[dict[str, object]] = [
    {
        "game_id": "2012_01_DAL_NYG",
        "season": 2012,
        "week": 1,
        "gameday": "2012-09-05",
        "weekday": "Wednesday",
        "gametime": "20:30",
        "away_team": "DAL",
        "home_team": "NYG",
        "espn": "320905019",
        "spread_line": 3.5,
        "total_line": 46.5,
        "home_moneyline": -175,
        "away_moneyline": 155,
        "home_score": 17,
    },
    {
        "game_id": "2016_02_SEA_LA",
        "season": 2016,
        "week": 2,
        "gameday": "2016-09-18",
        "weekday": "Sunday",
        "gametime": "16:05",
        "away_team": "SEA",
        "home_team": "LA",
        "espn": "400874572",
        "spread_line": -6.0,
        "total_line": 38.0,
        "home_moneyline": 225,
        "away_moneyline": -265,
        "home_score": 9,
    },
    {
        "game_id": "2005_01_ARI_NYG",
        "season": 2005,
        "week": 1,
        "gameday": "2005-09-11",
        "weekday": "Sunday",
        "gametime": "13:00",
        "away_team": "ARI",
        "home_team": "NYG",
        "espn": "250911019",
        "spread_line": 3.0,
        "total_line": 42.0,
        "home_moneyline": None,
        "away_moneyline": None,
        "home_score": 42,
    },
    {
        "game_id": "2026_05_MIN_CLE",
        "season": 2026,
        "week": 5,
        "gameday": "2026-10-04",
        "weekday": "Sunday",
        "gametime": "09:30",
        "away_team": "MIN",
        "home_team": "CLE",
        "espn": "401873000",
        "spread_line": -3.5,
        "total_line": 44.5,
        "home_moneyline": 150,
        "away_moneyline": -180,
        "home_score": 20,
    },
    {
        "game_id": "2026_09_LV_NYJ",
        "season": 2026,
        "week": 9,
        "gameday": "2026-11-08",
        "weekday": "Sunday",
        "gametime": "13:00",
        "away_team": "LV",
        "home_team": "NYJ",
        "espn": "401873100",
        "spread_line": 1.5,
        "total_line": 45.5,
        "home_moneyline": -123,
        "away_moneyline": 101,
        "home_score": None,
    },
]


def schedule_frame(games: list[dict[str, object]] | None = None) -> pl.DataFrame:
    raw = pl.DataFrame(
        games or GAMES,
        schema={
            "game_id": pl.String(),
            "season": pl.Int32(),
            "week": pl.Int32(),
            "gameday": pl.String(),
            "weekday": pl.String(),
            "gametime": pl.String(),
            "away_team": pl.String(),
            "home_team": pl.String(),
            "espn": pl.String(),
            "spread_line": pl.Float64(),
            "total_line": pl.Float64(),
            "home_moneyline": pl.Int32(),
            "away_moneyline": pl.Int32(),
            "home_score": pl.Int32(),
        },
    ).with_columns(
        pl.lit("REG").alias("game_type"),
        pl.lit(None, dtype=pl.Int32).alias("home_spread_odds"),
        pl.lit(None, dtype=pl.Int32).alias("away_spread_odds"),
        pl.lit(-110, dtype=pl.Int32).alias("over_odds"),
        pl.lit(-110, dtype=pl.Int32).alias("under_odds"),
    )
    return prepare_schedule(raw)
