import polars as pl

from football import storage
from football.ingest.base import IngestFn, IngestOptions
from football.ingest.venues import WEATHER_ROOFS, build_game_venues
from football.ingest.weather_base import load_schedules, utc_now
from football.ingest.weather_live import ingest_weather_live
from football.ingest.weather_mos import ingest_weather_mos
from football.ingest.weather_observed import ingest_weather_observed
from football.schemas import WEATHER_FORECASTS, WEATHER_OBSERVED
from football.validate.base import CheckFn, CheckResult

MIN_COVERAGE = 0.90
FIRST_CHECKED_SEASON = 2001


def ingest_mos(options: IngestOptions) -> None:
    ingest_weather_mos(build_game_venues(options.refresh), options.seasons, options.refresh)


def ingest_observed(options: IngestOptions) -> None:
    schedules = load_schedules(options.refresh)
    ingest_weather_observed(schedules, build_game_venues(), options.seasons, options.refresh)


def ingest_live(options: IngestOptions) -> None:
    ingest_weather_live(build_game_venues(options.refresh))


def coverage_table(venues: pl.DataFrame, forecasts: pl.DataFrame) -> pl.DataFrame:
    played = venues.filter(
        pl.col("roof_type").is_in(WEATHER_ROOFS),
        pl.col("season") >= 2000,
        pl.col("kickoff_utc") < utc_now(),
    )
    covered = (
        forecasts.join(played.select("game_id", "kickoff_utc"), on="game_id")
        .filter(pl.col("available_at") < pl.col("kickoff_utc"))
        .select("game_id")
        .unique()
        .with_columns(pl.lit(True).alias("covered"))
    )
    return played.join(covered, on="game_id", how="left").with_columns(
        pl.col("covered").fill_null(False),
        pl.col("weather_station").str.starts_with("K").alias("mos_station"),
    )


def summarize_coverage(table: pl.DataFrame) -> tuple[bool, str]:
    by_season = (
        table.group_by("season")
        .agg(
            pl.len().alias("games"),
            pl.col("covered").mean().alias("share"),
            pl.col("covered").filter(pl.col("mos_station")).mean().alias("us_share"),
        )
        .sort("season")
    )
    failing = by_season.filter(
        pl.col("season") >= FIRST_CHECKED_SEASON, pl.col("share") < MIN_COVERAGE
    )
    seasons = ", ".join(
        f"{row['season']} {row['share']:.1%} (US {row['us_share'] or 0:.1%})"
        for row in by_season.iter_rows(named=True)
    )
    gaps = table.filter(~pl.col("covered"))
    foreign = gaps.filter(~pl.col("mos_station"))
    domestic = gaps.filter(pl.col("mos_station"))
    worst = domestic.group_by("weather_station").len().sort("len", descending=True).head(5).rows()
    detail = (
        f"{table['covered'].sum()}/{table.height} outdoor or retractable games covered; "
        f"by season: {seasons}; gaps: {foreign.height} at non-US venues without MOS, "
        f"{domestic.height} at US stations missing archived MOS (top {worst})"
    )
    if failing.height:
        detail = f"below {MIN_COVERAGE:.0%}: {failing['season'].to_list()}; {detail}"
    return failing.height == 0, detail


def check_forecast_coverage() -> CheckResult:
    if not storage.exists("clean", WEATHER_FORECASTS.name):
        return CheckResult("forecast_coverage", False, "weather_forecasts table missing")
    table = coverage_table(build_game_venues(), storage.read("clean", WEATHER_FORECASTS.name))
    passed, detail = summarize_coverage(table)
    return CheckResult("forecast_coverage", passed, detail)


def forecast_errors(
    venues: pl.DataFrame, forecasts: pl.DataFrame, observed: pl.DataFrame
) -> pl.DataFrame:
    latest = (
        forecasts.join(venues.select("game_id", "kickoff_utc"), on="game_id")
        .filter(pl.col("available_at") < pl.col("kickoff_utc"))
        .sort("available_at", "run_time")
        .group_by("game_id", "source")
        .last()
    )
    joined = latest.join(
        observed.select(
            "game_id",
            pl.col("source").alias("observed_source"),
            pl.col("temp_f").alias("obs_temp_f"),
            pl.col("wind_mph").alias("obs_wind_mph"),
        ),
        on="game_id",
    )
    return (
        joined.group_by("source", "observed_source")
        .agg(
            pl.len().alias("games"),
            (pl.col("temp_f") - pl.col("obs_temp_f")).abs().mean().alias("temp_mae"),
            (pl.col("wind_mph") - pl.col("obs_wind_mph")).abs().mean().alias("wind_mae"),
            (pl.col("wind_mph") - pl.col("obs_wind_mph")).mean().alias("wind_bias"),
        )
        .sort("source", "observed_source")
    )


def check_forecast_error() -> CheckResult:
    missing = [
        spec.name
        for spec in (WEATHER_FORECASTS, WEATHER_OBSERVED)
        if not storage.exists("clean", spec.name)
    ]
    if missing:
        return CheckResult("forecast_error", True, f"report only; tables missing {missing}")
    errors = forecast_errors(
        build_game_venues(),
        storage.read("clean", WEATHER_FORECASTS.name),
        storage.read("clean", WEATHER_OBSERVED.name),
    )
    detail = "; ".join(
        f"{row['source']} vs {row['observed_source']}: n={row['games']} "
        f"temp MAE {row['temp_mae']:.1f}F, wind MAE {row['wind_mae']:.1f} mph "
        f"(bias {row['wind_bias']:+.1f})"
        for row in errors.iter_rows(named=True)
    )
    return CheckResult("forecast_error", True, f"report only; {detail or 'no overlap'}")


SOURCES: dict[str, IngestFn] = {
    "weather_mos": ingest_mos,
    "weather_observed": ingest_observed,
    "weather_live": ingest_live,
}
CHECKS: dict[str, CheckFn] = {
    "forecast_coverage": check_forecast_coverage,
    "forecast_error": check_forecast_error,
}
