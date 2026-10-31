from collections.abc import Iterable
from datetime import timedelta

import polars as pl
from rich.console import Console

from football import storage
from football.schemas import UTC_TS, TableSpec, validate
from football.teams import team_code

EASTERN = "America/New_York"
LOCAL_ALIASES = {"PHO": "ARI", "SL": "LA"}

console = Console()


def log(message: str) -> None:
    console.print(f"  {message}")


def eastern_to_utc(local: pl.Expr) -> pl.Expr:
    return (
        local.dt.replace_time_zone(EASTERN, ambiguous="latest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .cast(UTC_TS)
    )


def eastern_wall_clock(date: pl.Expr, hour: int, minute: int = 0) -> pl.Expr:
    return eastern_to_utc(date.cast(pl.Datetime("us")) + pl.duration(hours=hour, minutes=minute))


def to_eastern_date(ts: pl.Expr) -> pl.Expr:
    return ts.dt.convert_time_zone(EASTERN).dt.date()


def hours(value: float) -> timedelta:
    return timedelta(hours=value)


def _normalize_code(team: str, season: int) -> str | None:
    code = team.strip().upper()
    if not code:
        return None
    try:
        return team_code(LOCAL_ALIASES.get(code, code), season)
    except ValueError:
        return None


def normalize_team(df: pl.DataFrame, column: str, season_column: str = "season") -> pl.DataFrame:
    pairs = df.select(column, season_column).unique().drop_nulls()
    mapping = pairs.with_columns(
        pl.struct(column, season_column)
        .map_elements(
            lambda row: _normalize_code(row[column], row[season_column]), return_dtype=pl.String
        )
        .alias("_team_code")
    )
    return (
        df.join(mapping, on=[column, season_column], how="left", nulls_equal=True)
        .with_columns(pl.col("_team_code").alias(column))
        .drop("_team_code")
    )


def replace_seasons(
    df: pl.DataFrame, spec: TableSpec, seasons: Iterable[int], season_column: str = "season"
) -> pl.DataFrame:
    fresh = validate(df, spec)
    replaced = set(seasons) | set(fresh[season_column].unique().to_list())
    combined = fresh
    if storage.exists("clean", spec.name):
        old = storage.read("clean", spec.name).filter(~pl.col(season_column).is_in(replaced))
        combined = validate(pl.concat([old, fresh], how="diagonal_relaxed"), spec)
    combined = combined.sort(list(spec.keys))
    storage.write(combined, "clean", spec.name)
    log(f"{spec.name}: wrote {fresh.height} rows for seasons {sorted(replaced)}")
    return combined


def write_table(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    table = validate(df, spec).sort(list(spec.keys))
    storage.write(table, "clean", spec.name)
    log(f"{spec.name}: wrote {table.height} rows")
    return table
