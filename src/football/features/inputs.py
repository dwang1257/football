from dataclasses import dataclass, fields
from datetime import datetime

import polars as pl

from football import storage
from football.features.common import franchise_expr, utc

OPTIONAL_TABLES: tuple[str, ...] = (
    "qb_games",
    "injuries",
    "depth_charts",
    "snap_counts",
    "lines",
    "benchmarks",
    "team_ratings",
    "weather_forecasts",
    "weather_observed",
    "game_venues",
    "team_homes",
    "win_totals",
)


@dataclass(frozen=True)
class Tables:
    games: pl.DataFrame
    team_games: pl.DataFrame
    qb_games: pl.DataFrame | None = None
    injuries: pl.DataFrame | None = None
    depth_charts: pl.DataFrame | None = None
    snap_counts: pl.DataFrame | None = None
    lines: pl.DataFrame | None = None
    benchmarks: pl.DataFrame | None = None
    team_ratings: pl.DataFrame | None = None
    weather_forecasts: pl.DataFrame | None = None
    weather_observed: pl.DataFrame | None = None
    game_venues: pl.DataFrame | None = None
    team_homes: pl.DataFrame | None = None
    win_totals: pl.DataFrame | None = None

    def items(self) -> list[tuple[str, pl.DataFrame | None]]:
        return [(f.name, getattr(self, f.name)) for f in fields(self)]


class MissingTableError(RuntimeError):
    pass


def _read_optional(name: str) -> pl.DataFrame | None:
    if not storage.exists("clean", name):
        return None
    df = storage.read("clean", name)
    return df if df.height else None


def load_tables() -> Tables:
    for name in ("games", "team_games"):
        if not storage.exists("clean", name):
            raise MissingTableError(f"required clean table {name} is missing; run ingest first")
    optional = {name: _read_optional(name) for name in OPTIONAL_TABLES}
    return Tables(
        games=storage.read("clean", "games"),
        team_games=storage.read("clean", "team_games"),
        **optional,
    )


VENUE_COLUMNS: dict[str, pl.DataType] = {
    "venue_lat": pl.Float64(),
    "venue_lon": pl.Float64(),
    "venue_tz": pl.String(),
    "elevation_m": pl.Float64(),
    "is_international": pl.Boolean(),
}


def _clean_text(expr: pl.Expr) -> pl.Expr:
    return expr.str.to_lowercase().str.strip_chars()


def _static_roof(roof: pl.Expr) -> pl.Expr:
    text = _clean_text(roof)
    return (
        pl.when(text.str.contains("retract") | text.is_in(["open", "closed"]))
        .then(pl.lit("retractable"))
        .when(text.str.contains("dome") | text.str.contains("indoor"))
        .then(pl.lit("dome"))
        .when(text.is_not_null())
        .then(pl.lit("outdoors"))
        .otherwise(None)
    )


def _venues(games: pl.DataFrame, venues: pl.DataFrame | None) -> pl.DataFrame:
    base = games.select(
        "game_id",
        _static_roof(pl.col("roof")).alias("roof_type"),
        _clean_text(pl.col("surface")).alias("surface"),
    )
    if venues is None:
        return base.with_columns(
            pl.lit(None, dtype).alias(name) for name, dtype in VENUE_COLUMNS.items()
        )
    static = venues.unique("game_id", keep="last", maintain_order=True).select(
        "game_id",
        _static_roof(pl.col("roof_type")).alias("venue_roof"),
        _clean_text(pl.col("surface")).alias("venue_surface"),
        pl.col("lat").cast(pl.Float64).alias("venue_lat"),
        pl.col("lon").cast(pl.Float64).alias("venue_lon"),
        pl.col("tz").alias("venue_tz"),
        pl.col("elevation_m").cast(pl.Float64),
        pl.col("is_international").cast(pl.Boolean),
    )
    return (
        base.join(static, on="game_id", how="left")
        .with_columns(
            pl.coalesce("venue_roof", "roof_type").alias("roof_type"),
            pl.coalesce("venue_surface", "surface").alias("surface"),
        )
        .drop("venue_roof", "venue_surface")
    )


SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "kickoff_utc",
    "home_team",
    "away_team",
    "neutral_site",
    "stadium_id",
    "div_game",
    "home_rest",
    "away_rest",
    "home_coach",
    "away_coach",
)


def make_targets(tables: Tables, as_of: datetime | None) -> pl.DataFrame:
    games = tables.games
    cutoff = (
        pl.col("kickoff_utc")
        if as_of is None
        else pl.min_horizontal(pl.col("kickoff_utc"), pl.lit(utc(as_of)))
    )
    schedule = games.select(
        *SCHEDULE_COLUMNS,
        cutoff.alias("cutoff"),
        franchise_expr("home_team").alias("home_fr"),
        franchise_expr("away_team").alias("away_fr"),
    )
    return schedule.join(_venues(games, tables.game_venues), on="game_id", how="left").sort(
        "kickoff_utc", "game_id"
    )


def labels(games: pl.DataFrame) -> pl.DataFrame:
    margin = (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64)
    return games.select(
        "game_id",
        pl.when(margin > 0)
        .then(1.0)
        .when(margin < 0)
        .then(0.0)
        .when(margin == 0)
        .then(0.5)
        .otherwise(None)
        .alias("home_win"),
        margin.alias("home_margin"),
        pl.col("home_score").cast(pl.Float64).alias("home_points"),
        pl.col("away_score").cast(pl.Float64).alias("away_points"),
    )
