import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

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
from football.schemas import UTC_TS, WEATHER_FORECASTS, validate

MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
KNOTS_TO_MPH = 1.150779
LOOKBACK = timedelta(hours=96)
MAX_VALID_GAP = timedelta(hours=3)
GAME_LENGTH = timedelta(hours=3)


@dataclass(frozen=True)
class MosModel:
    source: str
    model: str
    first_season: int
    lag: timedelta


MODELS: tuple[MosModel, ...] = (
    MosModel("mos_gfs", "GFS", 2000, timedelta(hours=5)),
    MosModel("mos_nbs", "NBS", 2021, timedelta(hours=3)),
    MosModel("mos_nam", "NAM", 2002, timedelta(hours=4)),
)


@dataclass(frozen=True)
class Window:
    station: str
    label: str
    start: datetime
    end: datetime
    settled: bool

    def cache_name(self) -> str:
        return f"{self.station}_{self.label}_{self.start:%Y%m%d%H}_{self.end:%Y%m%d%H}.csv"


def mos_games(venues: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    return venues.filter(
        pl.col("season").is_in(seasons),
        pl.col("roof_type").is_in(["outdoors", "retractable"]),
        pl.col("weather_station").str.starts_with("K"),
    ).select("game_id", "season", "kickoff_utc", pl.col("weather_station").alias("station"))


def plan_windows(games: pl.DataFrame, now: datetime) -> list[tuple[Window, pl.DataFrame]]:
    season_done = games.group_by("season").agg(
        (pl.col("kickoff_utc").max() <= now).alias("season_done")
    )
    started = games.filter(pl.col("kickoff_utc") - LOOKBACK < now)
    keyed = started.join(season_done, on="season").with_columns(
        pl.when(pl.col("season_done"))
        .then(pl.col("season").cast(pl.String))
        .otherwise(pl.col("kickoff_utc").dt.strftime("%Y-%m"))
        .alias("label")
    )
    hour = now.replace(minute=0, second=0, microsecond=0)
    windows: list[tuple[Window, pl.DataFrame]] = []
    for (station, label), group in keyed.group_by("station", "label", maintain_order=True):
        first = group["kickoff_utc"].min()
        last = group["kickoff_utc"].max()
        assert isinstance(first, datetime) and isinstance(last, datetime)
        start = (first - LOOKBACK).replace(minute=0, second=0, microsecond=0)
        settled = last <= now
        end = last if settled else hour
        end = end.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        windows.append((Window(str(station), str(label), start, end, settled), group))
    return sorted(windows, key=lambda item: (item[0].start, item[0].station))


def parse_mos(payload: bytes) -> pl.DataFrame:
    schema = {
        "runtime": UTC_TS,
        "ftime": UTC_TS,
        "tmp": pl.Float64(),
        "wsp": pl.Float64(),
        "gst": pl.Float64(),
        "p06": pl.Float64(),
        "p12": pl.Float64(),
    }
    if not payload.strip() or not payload.startswith(b"runtime"):
        return pl.DataFrame(schema=schema)
    raw = pl.read_csv(io.BytesIO(payload), infer_schema_length=0)
    columns = [
        parse_utc(name, "%Y-%m-%d %H:%M:%S")
        if dtype == UTC_TS
        else (
            pl.col(name).str.strip_chars().cast(pl.Float64, strict=False)
            if name in raw.columns
            else pl.lit(None, pl.Float64())
        ).alias(name)
        for name, dtype in schema.items()
    ]
    return raw.select(columns).drop_nulls(["runtime", "ftime"])


def select_runs(mos: pl.DataFrame, games: pl.DataFrame, model: MosModel) -> pl.DataFrame:
    kickoff = pl.col("kickoff_utc")
    pairs = games.select("game_id", "station", "kickoff_utc").join(mos, how="cross")
    pairs = pairs.filter(pl.col("runtime") >= kickoff - LOOKBACK, pl.col("runtime") < kickoff)
    gap = (pl.col("ftime") - kickoff).abs()
    nearest = (
        pairs.filter(gap <= MAX_VALID_GAP, pl.col("tmp").is_not_null())
        .sort("game_id", "runtime", gap, "ftime")
        .unique(["game_id", "runtime"], keep="first", maintain_order=True)
    )
    p06 = (
        pairs.filter(
            pl.col("ftime") > kickoff, pl.col("ftime") <= kickoff + GAME_LENGTH + timedelta(hours=6)
        )
        .group_by("game_id", "runtime")
        .agg(pl.col("p06").max())
    )
    p12 = (
        pairs.filter(
            pl.col("ftime") > kickoff,
            pl.col("ftime") <= kickoff + GAME_LENGTH + timedelta(hours=12),
        )
        .group_by("game_id", "runtime")
        .agg(pl.col("p12").max())
    )
    return (
        nearest.drop("p06", "p12")
        .filter(pl.col("runtime") + model.lag < kickoff)
        .join(p06, on=["game_id", "runtime"], how="left")
        .join(p12, on=["game_id", "runtime"], how="left")
        .select(
            "game_id",
            pl.lit(model.source).alias("source"),
            "station",
            pl.col("runtime").alias("run_time"),
            pl.col("ftime").alias("valid_time"),
            pl.col("tmp").alias("temp_f"),
            (pl.col("wsp") * KNOTS_TO_MPH).alias("wind_mph"),
            (pl.col("gst") * KNOTS_TO_MPH).alias("wind_gust_mph"),
            (pl.coalesce("p06", "p12") / 100.0).alias("precip_prob"),
            (pl.col("runtime") + model.lag).alias("available_at"),
        )
        .sort("game_id", "run_time")
    )


def cached_window(http: Http, model: MosModel, window: Window, refresh: bool) -> bytes | None:
    folder = raw_dir("weather_mos") / model.model
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / window.cache_name()
    if path.exists() and not refresh:
        return path.read_bytes()
    params = {
        "station": window.station,
        "model": model.model,
        "sts": f"{window.start:%Y-%m-%dT%H:%MZ}",
        "ets": f"{window.end:%Y-%m-%dT%H:%MZ}",
        "format": "csv",
    }
    try:
        response = http.get(MOS_URL, params=params)
    except FetchError as exc:
        console.print(f"[yellow]mos {model.model} {window.cache_name()}: {exc}[/yellow]")
        return None
    if response.status_code != 200:
        console.print(
            f"[yellow]mos {model.model} {window.cache_name()}: HTTP {response.status_code}"
        )
        return None
    _drop_superseded(folder, window)
    _atomic_write(path, response.content)
    return response.content


def _drop_superseded(folder: Path, window: Window) -> None:
    prefix = f"{window.station}_{window.label}_{window.start:%Y%m%d%H}_"
    for stale in folder.glob(f"{prefix}*.csv"):
        stale.unlink()


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def ingest_model(http: Http, model: MosModel, games: pl.DataFrame, refresh: bool) -> pl.DataFrame:
    eligible = games.filter(pl.col("season") >= model.first_season)
    windows = plan_windows(eligible, utc_now())
    frames: list[pl.DataFrame] = []
    for index, (window, group) in enumerate(windows, start=1):
        payload = cached_window(http, model, window, refresh and window.settled)
        if payload is not None:
            frames.append(select_runs(parse_mos(payload), group, model))
        if index % 50 == 0 or index == len(windows):
            console.print(f"weather_mos {model.model}: {index}/{len(windows)} windows")
    if not frames:
        return pl.DataFrame(schema=WEATHER_FORECASTS.columns)
    return validate(pl.concat(frames), WEATHER_FORECASTS)


def ingest_weather_mos(venues: pl.DataFrame, seasons: list[int], refresh: bool) -> None:
    games = mos_games(venues, seasons)
    http = iem_http()
    try:
        for model in MODELS:
            rows = ingest_model(http, model, games, refresh)
            storage.upsert(rows, "clean", WEATHER_FORECASTS.name, WEATHER_FORECASTS.keys)
            console.print(
                f"weather_mos {model.source}: {rows.height} runs for {rows['game_id'].n_unique()} "
                f"of {games.height} games"
            )
    finally:
        http.close()
