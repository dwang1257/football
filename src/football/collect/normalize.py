from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl
from rich.console import Console

from football import storage
from football.collect import action_network, espn, forecasts, team_reports
from football.collect.archive import iter_snapshots, snapshot_days
from football.collect.games import GameResolver
from football.collect.runner import COLLECTORS
from football.ingest.base import IngestFn, IngestOptions
from football.schemas import (
    LINES,
    PRACTICE_REPORTS,
    TEAM_RATINGS,
    WEATHER_FORECASTS,
    TableSpec,
    validate,
)
from football.validate.base import CheckFn, CheckResult

MAX_STALENESS = timedelta(days=2)


def _frame(rows: Sequence[dict[str, Any]], spec: TableSpec) -> pl.DataFrame:
    frame = pl.DataFrame(list(rows), schema=spec.columns)
    return (
        frame.sort("available_at", maintain_order=True)
        .unique(subset=list(spec.keys), keep="first", maintain_order=True)
        .sort(list(spec.keys))
    )


def _report(name: str, frame: pl.DataFrame) -> None:
    Console().print(f"  {name}: {frame.height} rows")


def _season_filter(frame: pl.DataFrame, options: IngestOptions) -> pl.DataFrame:
    if "season" in frame.columns:
        return frame.filter(pl.col("season").is_in(options.seasons))
    seasons = pl.col("game_id").str.slice(0, 4).cast(pl.Int32, strict=False)
    return frame.filter(seasons.is_in(options.seasons))


def _upsert(frame: pl.DataFrame, spec: TableSpec) -> None:
    validated = validate(frame, spec)
    if validated.height:
        storage.upsert(validated, "clean", spec.name, spec.keys)


def ingest_practice(options: IngestOptions) -> None:
    frame = team_reports.practice_reports(iter_snapshots(team_reports.NAME))
    frame = _season_filter(frame, options)
    validated = validate(frame, PRACTICE_REPORTS)
    if storage.exists("clean", PRACTICE_REPORTS.name):
        old = storage.read("clean", PRACTICE_REPORTS.name)
        validated = pl.concat(
            [old.filter(~pl.col("season").is_in(options.seasons)), validated],
            how="diagonal_relaxed",
        )
    storage.write(validated.sort(list(PRACTICE_REPORTS.keys)), "clean", PRACTICE_REPORTS.name)
    _report(PRACTICE_REPORTS.name, frame)


def ingest_lines(options: IngestOptions) -> None:
    resolver = GameResolver.load()
    rows = espn.odds_lines(iter_snapshots("espn_odds", prefix="odds_"), resolver)
    rows += action_network.history_lines(iter_snapshots("action_network"), resolver)
    frame = _season_filter(_frame(rows, LINES), options)
    _upsert(frame, LINES)
    _report(LINES.name, frame)


def ingest_fpi(options: IngestOptions) -> None:
    rows = espn.fpi_ratings(iter_snapshots("espn_fpi"))
    frame = _season_filter(_frame(rows, TEAM_RATINGS), options)
    _upsert(frame, TEAM_RATINGS)
    _report(TEAM_RATINGS.name, frame)


def ingest_forecasts(options: IngestOptions) -> None:
    rows = forecasts.forecast_rows(iter_snapshots(forecasts.NAME))
    frame = _season_filter(_frame(rows, WEATHER_FORECASTS), options)
    _upsert(frame, WEATHER_FORECASTS)
    _report(WEATHER_FORECASTS.name, frame)


def _gaps(days: list[date]) -> int:
    if not days:
        return 0
    span = (days[-1] - days[0]).days + 1
    return span - len(set(days))


def check_freshness(now: datetime | None = None) -> CheckResult:
    today = (now or datetime.now(UTC)).date()
    parts: list[str] = []
    stale: list[str] = []
    for collector in COLLECTORS:
        days = snapshot_days(collector)
        if not days:
            stale.append(collector)
            parts.append(f"{collector}: none")
            continue
        age = today - days[-1]
        if age > MAX_STALENESS:
            stale.append(collector)
        parts.append(
            f"{collector}: latest {days[-1].isoformat()}, {len(days)} days, {_gaps(days)} missing"
        )
    passed = not stale
    detail = "; ".join(parts) + ("" if passed else f"; stale: {', '.join(stale)}")
    return CheckResult(name="archive_freshness", passed=passed, detail=detail)


SOURCES: dict[str, IngestFn] = {
    "archive_practice": ingest_practice,
    "archive_lines": ingest_lines,
    "archive_fpi": ingest_fpi,
    "archive_forecasts": ingest_forecasts,
}
CHECKS: dict[str, CheckFn] = {"archive_freshness": check_freshness}
