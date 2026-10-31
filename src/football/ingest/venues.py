from datetime import timedelta

import polars as pl
from timezonefinder import TimezoneFinder

from football import storage
from football.ingest.base import IngestFn, IngestOptions
from football.ingest.weather_base import console, kickoff_expr, load_schedules, static_path
from football.schemas import STADIUMS, TEAM_HOMES, UTC_TS, TableSpec, validate
from football.validate.base import CheckFn, CheckResult

ROOF_TYPES = ("dome", "outdoors", "retractable")
WEATHER_ROOFS = ("outdoors", "retractable")
VENUE_LEAD = timedelta(days=7)

GAME_VENUES = TableSpec(
    "game_venues",
    ("game_id",),
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "game_type": pl.String(),
        "kickoff_utc": UTC_TS,
        "home_team": pl.String(),
        "away_team": pl.String(),
        "neutral_site": pl.Boolean(),
        "stadium_id": pl.String(),
        "nflverse_stadium_id": pl.String(),
        "corrected": pl.Boolean(),
        "roof_type": pl.String(),
        "surface": pl.String(),
        "lat": pl.Float64(),
        "lon": pl.Float64(),
        "tz": pl.String(),
        "elevation_m": pl.Float64(),
        "is_international": pl.Boolean(),
        "weather_station": pl.String(),
        "available_at": UTC_TS,
    },
)

_STADIUM_CSV_SCHEMA = {
    **STADIUMS.columns,
    "coord_source": pl.String(),
}


def load_stadiums() -> pl.DataFrame:
    frame = pl.read_csv(static_path("stadiums.csv"), schema=_STADIUM_CSV_SCHEMA)
    return fill_timezones(frame)


def load_overrides() -> pl.DataFrame:
    return pl.read_csv(
        static_path("game_venue_overrides.csv"),
        schema={"game_id": pl.String(), "stadium_id": pl.String(), "note": pl.String()},
    )


def fill_timezones(stadiums: pl.DataFrame) -> pl.DataFrame:
    if not stadiums["tz"].null_count():
        return stadiums
    finder = TimezoneFinder()
    zones = [
        tz or finder.timezone_at(lat=lat, lng=lon)
        for tz, lat, lon in stadiums.select("tz", "lat", "lon").iter_rows()
    ]
    return stadiums.with_columns(pl.Series("tz", zones, dtype=pl.String()))


def clean_surface(expr: pl.Expr) -> pl.Expr:
    text = expr.str.strip_chars().str.to_lowercase()
    return pl.when(text == "").then(None).otherwise(text)


def game_venues_frame(
    schedules: pl.DataFrame, stadiums: pl.DataFrame, overrides: pl.DataFrame
) -> pl.DataFrame:
    games = (
        schedules.select(
            "game_id",
            pl.col("season").cast(pl.Int32),
            "game_type",
            kickoff_expr(),
            "home_team",
            "away_team",
            (pl.col("location") == "Neutral").alias("neutral_site"),
            pl.col("stadium_id").alias("nflverse_stadium_id"),
            clean_surface(pl.col("surface")).alias("game_surface"),
        )
        .join(overrides.select("game_id", "stadium_id"), on="game_id", how="left")
        .with_columns(
            pl.col("stadium_id").is_not_null().alias("corrected"),
            pl.coalesce("stadium_id", "nflverse_stadium_id").alias("stadium_id"),
        )
    )
    season_surface = (
        games.filter(~pl.col("neutral_site"), ~pl.col("corrected"))
        .drop_nulls("game_surface")
        .group_by("stadium_id", "season")
        .agg(pl.col("game_surface").mode().sort().first().alias("season_surface"))
    )
    site = stadiums.select(
        "stadium_id",
        "roof_type",
        pl.col("surface").alias("stadium_surface"),
        "lat",
        "lon",
        "tz",
        "elevation_m",
        "is_international",
        "weather_station",
    )
    unreliable = pl.col("corrected") | pl.col("neutral_site") | pl.col("is_international")
    surface = (
        pl.when(unreliable)
        .then(pl.coalesce("season_surface", "stadium_surface"))
        .otherwise(pl.coalesce("game_surface", "season_surface", "stadium_surface"))
    )
    frame = (
        games.join(season_surface, on=["stadium_id", "season"], how="left")
        .join(site, on="stadium_id", how="left")
        .with_columns(
            surface.alias("surface"),
            (pl.col("kickoff_utc") - VENUE_LEAD).alias("available_at"),
        )
    )
    return validate(frame.select(list(GAME_VENUES.columns)), GAME_VENUES).sort("game_id")


def team_homes_frame(venues: pl.DataFrame, stadiums: pl.DataFrame) -> pl.DataFrame:
    counts = (
        venues.filter(~pl.col("neutral_site"), ~pl.col("corrected"))
        .group_by("season", pl.col("home_team").alias("team"), "stadium_id")
        .agg(pl.len().alias("games"), pl.col("kickoff_utc").max().alias("last_game"))
        .sort(
            ["season", "team", "games", "last_game", "stadium_id"],
            descending=[False, False, True, True, False],
        )
        .unique(["season", "team"], keep="first", maintain_order=True)
    )
    frame = counts.join(
        stadiums.select("stadium_id", "lat", "lon", "tz", "roof_type"), on="stadium_id", how="left"
    )
    return validate(frame.select(list(TEAM_HOMES.columns)), TEAM_HOMES).sort("season", "team")


def build_game_venues(refresh: bool = False) -> pl.DataFrame:
    return game_venues_frame(load_schedules(refresh), load_stadiums(), load_overrides())


def ingest_stadiums(options: IngestOptions) -> None:
    stadiums = validate(load_stadiums(), STADIUMS)
    used = set(load_schedules(options.refresh)["stadium_id"].drop_nulls().unique())
    used |= set(load_overrides()["stadium_id"])
    missing = sorted(used - set(stadiums["stadium_id"]))
    if missing:
        console.print(f"[red]stadiums.csv lacks stadium_ids {missing}[/red]")
    storage.write(stadiums, "clean", STADIUMS.name)
    console.print(f"stadiums: {stadiums.height} rows, {len(used)} used by schedules or overrides")


def ingest_game_venues(options: IngestOptions) -> None:
    venues = build_game_venues(options.refresh).filter(pl.col("season").is_in(options.seasons))
    storage.upsert(venues, "clean", GAME_VENUES.name, GAME_VENUES.keys)
    console.print(
        f"game_venues: {venues.height} rows, {venues['corrected'].sum()} corrected, "
        f"{venues['stadium_id'].n_unique()} stadiums"
    )


def ingest_team_homes(options: IngestOptions) -> None:
    homes = team_homes_frame(build_game_venues(options.refresh), load_stadiums()).filter(
        pl.col("season").is_in(options.seasons)
    )
    storage.upsert(homes, "clean", TEAM_HOMES.name, TEAM_HOMES.keys)
    console.print(f"team_homes: {homes.height} rows")


def suspicious_neutral_games(venues: pl.DataFrame, stadiums: pl.DataFrame) -> pl.DataFrame:
    homes = team_homes_frame(venues, stadiums).select(
        "season", pl.col("team").alias("home_team"), pl.col("stadium_id").alias("home_stadium")
    )
    return (
        venues.filter(pl.col("neutral_site"), pl.col("game_type") == "REG", ~pl.col("corrected"))
        .join(homes, on=["season", "home_team"], how="inner")
        .filter(pl.col("stadium_id") == pl.col("home_stadium"))
    )


def check_stadium_coverage() -> CheckResult:
    stadiums = load_stadiums()
    venues = build_game_venues()
    problems: list[str] = []
    unknown = venues.filter(pl.col("lat").is_null())["stadium_id"].unique().sort().to_list()
    if unknown:
        problems.append(f"stadium_ids missing from stadiums.csv: {unknown}")
    incomplete = stadiums.filter(
        pl.any_horizontal(pl.col(c).is_null() for c in ("lat", "lon", "tz", "roof_type"))
        | ~pl.col("roof_type").is_in(ROOF_TYPES)
    )["stadium_id"].to_list()
    if incomplete:
        problems.append(f"stadiums lacking lat, lon, tz or a valid roof_type: {incomplete}")
    suspicious = suspicious_neutral_games(venues, stadiums)["game_id"].to_list()
    if suspicious:
        problems.append(f"neutral games listed at the home team's own stadium: {suspicious}")
    schedules = load_schedules()
    retractable_hint = (
        schedules.filter(pl.col("roof").is_in(["open", "closed"]))["stadium_id"].unique().to_list()
    )
    disagree = (
        stadiums.filter(pl.col("stadium_id").is_in(retractable_hint))
        .filter(pl.col("roof_type") != "retractable")["stadium_id"]
        .to_list()
    )
    detail = (
        f"{stadiums.height} stadiums, {venues.height} games, "
        f"{venues['stadium_id'].n_unique()} venues used, {venues['corrected'].sum()} corrected; "
        f"curated roof differs from nflverse open/closed at {disagree or 'none'}"
    )
    return CheckResult("stadium_coverage", not problems, "; ".join([*problems, detail]))


SOURCES: dict[str, IngestFn] = {
    "stadiums": ingest_stadiums,
    "team_homes": ingest_team_homes,
    "game_venues": ingest_game_venues,
}
CHECKS: dict[str, CheckFn] = {"stadium_coverage": check_stadium_coverage}
