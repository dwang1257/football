import json
import re
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from football import storage
from football.ingest.weather_base import FetchError, Http, Throttle, console, raw_dir, utc_now
from football.schemas import UTC_TS, WEATHER_FORECASTS, validate
from football.storage import write_raw

NWS_URL = "https://api.weather.gov"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HORIZON = timedelta(days=8)
NWS_HEADERS = {"Accept": "application/geo+json"}
OPEN_METEO_HOURLY = "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability"
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def live_targets(venues: pl.DataFrame, now: datetime) -> pl.DataFrame:
    return venues.filter(
        pl.col("roof_type").is_in(["outdoors", "retractable"]),
        pl.col("kickoff_utc") > now,
        pl.col("kickoff_utc") <= now + HORIZON,
    ).select("game_id", "stadium_id", "kickoff_utc", "lat", "lon", "is_international")


def wind_speed(text: str | None) -> float | None:
    numbers = [float(n) for n in _NUMBER.findall(text or "")]
    return sum(numbers) / len(numbers) if numbers else None


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=WEATHER_FORECASTS.columns)


def nws_rows(hourly: dict[str, Any], games: pl.DataFrame, fetched_at: datetime) -> pl.DataFrame:
    props = hourly["properties"]
    run_time = datetime.fromisoformat(props["updateTime"])
    periods = [
        {
            "valid_time": datetime.fromisoformat(period["startTime"]),
            "valid_end": datetime.fromisoformat(period["endTime"]),
            "temp_f": float(period["temperature"]),
            "wind_mph": wind_speed(period.get("windSpeed")),
            "precip_prob": (period.get("probabilityOfPrecipitation") or {}).get("value"),
        }
        for period in props["periods"]
        if period.get("temperatureUnit", "F") == "F"
    ]
    if not periods:
        return _empty()
    frame = pl.DataFrame(
        periods,
        schema={
            "valid_time": UTC_TS,
            "valid_end": UTC_TS,
            "temp_f": pl.Float64(),
            "wind_mph": pl.Float64(),
            "precip_prob": pl.Float64(),
        },
    )
    matched = (
        games.select("game_id", "kickoff_utc")
        .join(frame, how="cross")
        .filter(
            pl.col("valid_time") <= pl.col("kickoff_utc"),
            pl.col("kickoff_utc") < pl.col("valid_end"),
        )
    )
    return matched.select(
        "game_id",
        pl.lit("nws").alias("source"),
        pl.lit(None, pl.String()).alias("station"),
        pl.lit(run_time).cast(UTC_TS).alias("run_time"),
        "valid_time",
        "temp_f",
        "wind_mph",
        pl.lit(None, pl.Float64()).alias("wind_gust_mph"),
        (pl.col("precip_prob") / 100.0).alias("precip_prob"),
        pl.lit(fetched_at).cast(UTC_TS).alias("available_at"),
    )


def open_meteo_rows(
    payload: dict[str, Any], games: pl.DataFrame, fetched_at: datetime
) -> pl.DataFrame:
    hourly = payload["hourly"]
    frame = pl.DataFrame(
        {
            "valid_time": hourly["time"],
            "temp_f": hourly["temperature_2m"],
            "wind_mph": hourly["wind_speed_10m"],
            "wind_gust_mph": hourly["wind_gusts_10m"],
            "precip_prob": hourly["precipitation_probability"],
        },
        schema_overrides={
            c: pl.Float64() for c in ("temp_f", "wind_mph", "wind_gust_mph", "precip_prob")
        },
    ).with_columns(
        pl.col("valid_time").str.to_datetime("%Y-%m-%dT%H:%M", time_unit="us", time_zone="UTC")
    )
    matched = (
        games.select("game_id", "kickoff_utc")
        .join(frame, how="cross")
        .filter(
            pl.col("valid_time") <= pl.col("kickoff_utc"),
            pl.col("kickoff_utc") < pl.col("valid_time") + timedelta(hours=1),
        )
    )
    return matched.select(
        "game_id",
        pl.lit("open_meteo").alias("source"),
        pl.lit(None, pl.String()).alias("station"),
        pl.lit(fetched_at).cast(UTC_TS).alias("run_time"),
        "valid_time",
        "temp_f",
        "wind_mph",
        "wind_gust_mph",
        (pl.col("precip_prob") / 100.0).alias("precip_prob"),
        pl.lit(fetched_at).cast(UTC_TS).alias("available_at"),
    )


def _forecast_hourly_url(http: Http, stadium_id: str, lat: float, lon: float) -> str | None:
    path = raw_dir("weather_live") / f"nws_points_{stadium_id}.json"
    if path.exists():
        return json.loads(path.read_text())["properties"]["forecastHourly"]
    response = http.get(f"{NWS_URL}/points/{lat:.4f},{lon:.4f}", headers=NWS_HEADERS)
    if response.status_code != 200:
        console.print(f"[yellow]nws points {stadium_id}: HTTP {response.status_code}[/yellow]")
        return None
    path.write_bytes(response.content)
    return response.json()["properties"]["forecastHourly"]


def nws_forecast(http: Http, stadium_id: str, games: pl.DataFrame) -> pl.DataFrame:
    lat, lon = float(games["lat"][0]), float(games["lon"][0])
    url = _forecast_hourly_url(http, stadium_id, lat, lon)
    if url is None:
        return _empty()
    response = http.get(url, params={"units": "us"}, headers=NWS_HEADERS)
    fetched_at = utc_now()
    if response.status_code != 200:
        console.print(f"[yellow]nws hourly {stadium_id}: HTTP {response.status_code}[/yellow]")
        return _empty()
    write_raw("weather_live", f"nws_{stadium_id}.json", response.content)
    return nws_rows(response.json(), games, fetched_at)


def open_meteo_forecast(http: Http, stadium_id: str, games: pl.DataFrame) -> pl.DataFrame:
    params: dict[str, str | int | float] = {
        "latitude": float(games["lat"][0]),
        "longitude": float(games["lon"][0]),
        "hourly": OPEN_METEO_HOURLY,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "GMT",
        "forecast_days": 10,
    }
    response = http.get(OPEN_METEO_URL, params=params)
    fetched_at = utc_now()
    if response.status_code != 200:
        console.print(
            f"[yellow]open-meteo {stadium_id}: HTTP {response.status_code} "
            f"{response.text[:120]}[/yellow]"
        )
        return _empty()
    write_raw("weather_live", f"open_meteo_{stadium_id}.json", response.content)
    return open_meteo_rows(response.json(), games, fetched_at)


def keep_first_seen(rows: pl.DataFrame) -> pl.DataFrame:
    if not storage.exists("clean", WEATHER_FORECASTS.name):
        return rows
    existing = storage.read("clean", WEATHER_FORECASTS.name).select(WEATHER_FORECASTS.keys)
    return rows.join(existing, on=list(WEATHER_FORECASTS.keys), how="anti")


def ingest_weather_live(venues: pl.DataFrame) -> None:
    targets = live_targets(venues, utc_now())
    nws = Http(Throttle("nws", 1.0), attempts=3, backoff_s=5.0)
    open_meteo = Http(Throttle("open_meteo", 1.0), attempts=2, backoff_s=5.0)
    frames: list[pl.DataFrame] = [_empty()]
    try:
        for (stadium_id, international), games in targets.group_by(
            "stadium_id", "is_international", maintain_order=True
        ):
            fetch = open_meteo_forecast if international else nws_forecast
            client = open_meteo if international else nws
            try:
                frames.append(fetch(client, str(stadium_id), games))
            except (FetchError, KeyError, ValueError) as exc:
                console.print(f"[yellow]weather_live {stadium_id}: {exc}[/yellow]")
    finally:
        nws.close()
        open_meteo.close()
    rows = validate(pl.concat(frames), WEATHER_FORECASTS)
    fresh = keep_first_seen(rows)
    storage.upsert(fresh, "clean", WEATHER_FORECASTS.name, WEATHER_FORECASTS.keys)
    console.print(
        f"weather_live: {targets.height} upcoming outdoor or retractable games, "
        f"{rows['game_id'].n_unique()} with a forecast, {fresh.height} new rows"
    )
