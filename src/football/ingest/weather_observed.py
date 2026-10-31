import io
from datetime import datetime, timedelta

import httpx
import polars as pl

from football import storage
from football.ingest.weather_base import (
    FetchError,
    Http,
    console,
    iem_http,
    parse_utc,
    raw_dir,
    utc_now,
)
from football.ingest.weather_mos import KNOTS_TO_MPH
from football.schemas import UTC_TS, WEATHER_OBSERVED, validate

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
REPORT_DELAY = timedelta(hours=4)
MAX_OBS_GAP = timedelta(minutes=90)
PRECIP_SPAN = timedelta(hours=3, minutes=30)
PAD = timedelta(hours=6)


def observed_targets(venues: pl.DataFrame, seasons: list[int], now: datetime) -> pl.DataFrame:
    return venues.filter(
        pl.col("season").is_in(seasons),
        pl.col("roof_type").is_in(["outdoors", "retractable"]),
        pl.col("kickoff_utc") + REPORT_DELAY <= now,
    ).select("game_id", "season", "kickoff_utc", pl.col("weather_station").alias("station"))


def nflverse_observed(schedules: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    frame = (
        schedules.select("game_id", "temp", "wind")
        .join(targets.select("game_id", "kickoff_utc"), on="game_id", how="inner")
        .filter(pl.col("temp").is_not_null(), pl.col("wind").is_not_null())
    )
    return validate(
        frame.select(
            "game_id",
            pl.lit("nflverse").alias("source"),
            pl.col("temp").cast(pl.Float64).alias("temp_f"),
            pl.col("wind").cast(pl.Float64).alias("wind_mph"),
            pl.lit(None, pl.Float64()).alias("precip_in"),
            (pl.col("kickoff_utc") + REPORT_DELAY).alias("available_at"),
        ),
        WEATHER_OBSERVED,
    )


def iem_station(station: str) -> str:
    return station[1:] if len(station) == 4 and station.startswith("K") else station


def parse_asos(payload: bytes) -> pl.DataFrame:
    schema = {"valid": UTC_TS, "tmpf": pl.Float64(), "sknt": pl.Float64(), "p01i": pl.Float64()}
    if not payload.startswith(b"station,"):
        return pl.DataFrame(schema=schema)
    raw = pl.read_csv(io.BytesIO(payload), infer_schema_length=0)
    return raw.select(
        parse_utc("valid", "%Y-%m-%d %H:%M"),
        *(pl.col(c).cast(pl.Float64, strict=False) for c in ("tmpf", "sknt", "p01i")),
    ).drop_nulls("valid")


def summarize_asos(obs: pl.DataFrame, games: pl.DataFrame, us_station: bool) -> pl.DataFrame:
    kickoff = pl.col("kickoff_utc")
    pairs = games.select("game_id", "kickoff_utc").join(obs, how="cross")
    gap = (pl.col("valid") - kickoff).abs()
    nearest = (
        pairs.filter(gap <= MAX_OBS_GAP, pl.col("tmpf").is_not_null(), pl.col("sknt").is_not_null())
        .sort("game_id", gap)
        .unique("game_id", keep="first", maintain_order=True)
        .select("game_id", "kickoff_utc", "tmpf", "sknt")
    )
    precip = (
        pairs.filter(pl.col("valid") > kickoff, pl.col("valid") <= kickoff + PRECIP_SPAN)
        .group_by("game_id")
        .agg(pl.col("p01i").sum().alias("precip_in"), pl.col("p01i").count().alias("n"))
        .select("game_id", pl.when(pl.col("n") > 0).then("precip_in").alias("precip_in"))
    )
    frame = nearest.join(precip, on="game_id", how="left")
    return frame.select(
        "game_id",
        pl.lit("iem_asos").alias("source"),
        pl.col("tmpf").alias("temp_f"),
        (pl.col("sknt") * KNOTS_TO_MPH).alias("wind_mph"),
        (pl.col("precip_in") if us_station else pl.lit(None, pl.Float64())).alias("precip_in"),
        (kickoff + REPORT_DELAY).alias("available_at"),
    )


def fetch_asos(http: Http, station: str, games: pl.DataFrame, refresh: bool) -> bytes | None:
    start = games["kickoff_utc"].min()
    end = games["kickoff_utc"].max()
    assert isinstance(start, datetime) and isinstance(end, datetime)
    start, end = start - PAD, end + PAD
    path = raw_dir("weather_asos") / f"{station}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.csv"
    if path.exists() and not refresh:
        return path.read_bytes()
    params = httpx.QueryParams(
        {
            "station": iem_station(station),
            "data": ["tmpf", "sknt", "p01i"],
            "sts": f"{start:%Y-%m-%dT%H:%MZ}",
            "ets": f"{end:%Y-%m-%dT%H:%MZ}",
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "no",
            "missing": "empty",
            "trace": "0.0001",
            "report_type": "3",
        }
    )
    try:
        response = http.get(ASOS_URL, params=params)
    except FetchError as exc:
        console.print(f"[yellow]asos {path.name}: {exc}[/yellow]")
        return None
    if response.status_code != 200:
        console.print(f"[yellow]asos {path.name}: HTTP {response.status_code}[/yellow]")
        return None
    tmp = path.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return response.content


def asos_observed(targets: pl.DataFrame, refresh: bool) -> pl.DataFrame:
    http = iem_http()
    frames: list[pl.DataFrame] = []
    try:
        groups = targets.group_by("station", "season", maintain_order=True)
        for (station, _season), group in groups:
            payload = fetch_asos(http, str(station), group, refresh)
            if payload is not None:
                us = str(station).startswith("K")
                frames.append(summarize_asos(parse_asos(payload), group, us))
    finally:
        http.close()
    if not frames:
        return pl.DataFrame(schema=WEATHER_OBSERVED.columns)
    return validate(pl.concat(frames), WEATHER_OBSERVED)


def ingest_weather_observed(
    schedules: pl.DataFrame, venues: pl.DataFrame, seasons: list[int], refresh: bool
) -> None:
    targets = observed_targets(venues, seasons, utc_now())
    reported = nflverse_observed(schedules, targets)
    missing = targets.join(reported.select("game_id"), on="game_id", how="anti")
    fallback = asos_observed(missing, refresh)
    rows = pl.concat([reported, fallback])
    storage.upsert(rows, "clean", WEATHER_OBSERVED.name, WEATHER_OBSERVED.keys)
    console.print(
        f"weather_observed: {targets.height} games, {reported.height} from nflverse, "
        f"{fallback.height} of {missing.height} remaining from IEM ASOS"
    )
