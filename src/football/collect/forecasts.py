import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import nflreadpy
import polars as pl

import football
from football.collect.archive import Snapshot
from football.collect.base import CollectorResult, Context
from football.collect.http import FetchError
from football.schemas import UTC_TS
from football.seasons import current_season

NAME = "forecasts"
NWS = "https://api.weather.gov"
HORIZON = timedelta(days=8)
INDOOR_ROOFS = frozenset({"dome", "indoor", "fixed", "fixed dome", "indoors"})
EASTERN = ZoneInfo("America/New_York")
NWS_HEADERS = {"Accept": "application/geo+json"}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def stadiums_csv() -> Path:
    return Path(football.__file__).parent / "static" / "stadiums.csv"


def load_stadiums(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=0)
    missing = {"stadium_id", "lat", "lon"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    absent = pl.lit(None, pl.String())
    roof = pl.col("roof_type") if "roof_type" in frame.columns else absent
    abroad = pl.col("is_international") if "is_international" in frame.columns else absent
    return frame.select(
        pl.col("stadium_id").str.strip_chars(),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
        roof.str.strip_chars().str.to_lowercase().alias("stadium_roof"),
        abroad.str.strip_chars().str.to_lowercase().is_in(["true", "1", "yes"]).alias("abroad"),
    )


def load_schedule(season: int) -> pl.DataFrame:
    return nflreadpy.load_schedules([season])


def kickoff_utc(gameday: str, gametime: str) -> datetime:
    local = datetime.fromisoformat(f"{gameday}T{gametime}").replace(tzinfo=EASTERN)
    return local.astimezone(UTC)


def upcoming_games(schedule: pl.DataFrame, now: datetime) -> pl.DataFrame:
    rows = [
        {
            "game_id": row["game_id"],
            "stadium_id": row["stadium_id"],
            "roof": row["roof"],
            "kickoff_utc": kickoff_utc(str(row["gameday"]), str(row["gametime"])),
        }
        for row in schedule.select("game_id", "gameday", "gametime", "stadium_id", "roof")
        .drop_nulls(["game_id", "gameday", "gametime", "stadium_id"])
        .iter_rows(named=True)
    ]
    schema = {
        "game_id": pl.String(),
        "stadium_id": pl.String(),
        "roof": pl.String(),
        "kickoff_utc": UTC_TS,
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row")
    return frame.filter(
        (pl.col("kickoff_utc") > now) & (pl.col("kickoff_utc") <= now + HORIZON)
    ).sort("kickoff_utc")


def forecast_targets(games: pl.DataFrame, stadiums: pl.DataFrame) -> pl.DataFrame:
    joined = games.join(stadiums, on="stadium_id", how="left")
    roof = pl.coalesce(pl.col("stadium_roof"), pl.col("roof").str.to_lowercase())
    return joined.filter(roof.is_null() | ~roof.is_in(list(INDOOR_ROOFS)))


def collect(ctx: Context, result: CollectorResult) -> None:
    path = stadiums_csv()
    if not path.exists():
        result.skipped = True
        result.notes.append(f"{path} not found; forecasts skipped until the venues table exists")
        return
    stadiums = load_stadiums(path)
    games = upcoming_games(load_schedule(current_season(ctx.started_at)), ctx.started_at)
    targets = forecast_targets(games, stadiums)
    abroad = targets.filter(pl.col("abroad").fill_null(False))
    if abroad.height:
        result.notes.append(f"outside NWS coverage: {', '.join(abroad['game_id'].to_list())}")
    targets = targets.filter(~pl.col("abroad").fill_null(False))
    unlocated = targets.filter(pl.col("lat").is_null() | pl.col("lon").is_null())
    for game in unlocated["game_id"].to_list():
        result.failed[game] = "stadium has no coordinates in stadiums.csv"
    located = targets.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
    for (stadium_id,), group in located.group_by(["stadium_id"], maintain_order=True):
        _collect_stadium(ctx, result, str(stadium_id), group)
    result.notes.append(
        f"{games.height} games in next {HORIZON.days} days, {targets.height} not indoor, "
        f"{located['stadium_id'].n_unique()} stadiums"
    )


def _collect_stadium(
    ctx: Context, result: CollectorResult, stadium_id: str, group: pl.DataFrame
) -> None:
    lat = float(group["lat"][0])
    lon = float(group["lon"][0])
    meta: dict[str, Any] = {
        "stadium_id": stadium_id,
        "lat": lat,
        "lon": lon,
        "games": [
            {"game_id": row["game_id"], "kickoff_utc": row["kickoff_utc"].isoformat()}
            for row in group.iter_rows(named=True)
        ],
    }
    try:
        points = ctx.fetcher.get(f"{NWS}/points/{lat:.4f},{lon:.4f}", headers=NWS_HEADERS)
        if not points.ok:
            result.failed[stadium_id] = f"points HTTP {points.status}"
            return
        properties = points.json()["properties"]
        meta["grid"] = f"{properties['gridId']}/{properties['gridX']},{properties['gridY']}"
        hourly = ctx.fetcher.get(str(properties["forecastHourly"]), headers=NWS_HEADERS)
    except (FetchError, KeyError, ValueError) as exc:
        result.failed[stadium_id] = f"{type(exc).__name__}: {exc}"
        return
    if not hourly.ok:
        ctx.save(result, stadium_id, hourly, hourly.text[:2000], meta)
        result.failed[stadium_id] = f"hourly HTTP {hourly.status}"
        return
    ctx.save(result, stadium_id, hourly, {"points": points.json(), "hourly": hourly.json()}, meta)
    result.ok.append(stadium_id)


def _max_number(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, int | float):
        return float(text)
    values = [float(item) for item in _NUMBER.findall(str(text))]
    return max(values) if values else None


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _temperature_f(period: dict[str, Any]) -> float | None:
    value = period.get("temperature")
    if isinstance(value, dict):
        value = value.get("value")
    number = _max_number(value)
    if number is None:
        return None
    return number * 9 / 5 + 32 if period.get("temperatureUnit") == "C" else number


def forecast_rows(snapshots: Iterable[Snapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        body = snapshot.body
        if not snapshot.ok or not isinstance(body, dict) or "hourly" not in body:
            continue
        properties = body["hourly"].get("properties") or {}
        run = _parse(properties.get("updateTime")) or _parse(properties.get("generatedAt"))
        run_time = min(run or snapshot.fetched_at, snapshot.fetched_at)
        periods = [
            (_parse(period.get("startTime")), _parse(period.get("endTime")), period)
            for period in properties.get("periods") or []
        ]
        for game in snapshot.meta.get("games", []):
            kickoff = _parse(game.get("kickoff_utc"))
            if kickoff is None or snapshot.fetched_at >= kickoff:
                continue
            match = next(
                (
                    (start, period)
                    for start, end, period in periods
                    if start is not None and end is not None and start <= kickoff < end
                ),
                None,
            )
            if match is None:
                continue
            start, period = match
            precip = (period.get("probabilityOfPrecipitation") or {}).get("value")
            rows.append(
                {
                    "game_id": str(game["game_id"]),
                    "source": "nws_archive",
                    "station": str(snapshot.meta.get("grid") or snapshot.meta.get("stadium_id")),
                    "run_time": run_time,
                    "valid_time": start,
                    "temp_f": _temperature_f(period),
                    "wind_mph": _max_number(period.get("windSpeed")),
                    "wind_gust_mph": _max_number(period.get("windGust")),
                    "precip_prob": float(precip) / 100 if precip is not None else None,
                    "available_at": snapshot.fetched_at,
                }
            )
    return rows
