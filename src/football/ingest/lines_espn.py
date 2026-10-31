import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from football.ingest.base import IngestOptions
from football.ingest.lines_common import (
    LIVE_MAX_AGE,
    Cached,
    RawCache,
    Schedule,
    close_available,
    console,
    load_schedule,
    parse_american,
    parse_number,
    publish,
    reject_frame,
    safe_franchise,
    tuesday_noon_before,
)

SOURCE = "lines_espn"
FIRST_SEASON = 2012
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
ODDS_URL = CORE + "/events/{event}/competitions/{event}/odds?limit=100"
MIN_INTERVAL = 0.26
LOOKAHEAD = timedelta(days=14)
FINAL_AFTER = timedelta(hours=8)
PICK_PROVIDERS = frozenset({"1001", "1002", "1003", "1004"})
OPENING_PROVIDER = "23"
MIN_TOTAL = 25.0
MAX_TOTAL = 75.0
MAX_BOOK_DEVIATION = 3.0

ESPN_TEAM_IDS: dict[str, str] = {
    "1": "ATL",
    "2": "BUF",
    "3": "CHI",
    "4": "CIN",
    "5": "CLE",
    "6": "DAL",
    "7": "DEN",
    "8": "DET",
    "9": "GB",
    "10": "TEN",
    "11": "IND",
    "12": "KC",
    "13": "LV",
    "14": "LA",
    "15": "MIA",
    "16": "MIN",
    "17": "NE",
    "18": "NO",
    "19": "NYG",
    "20": "NYJ",
    "21": "PHI",
    "22": "ARI",
    "23": "PIT",
    "24": "LAC",
    "25": "SF",
    "26": "SEA",
    "27": "TB",
    "28": "WAS",
    "29": "CAR",
    "30": "JAX",
    "33": "BAL",
    "34": "HOU",
}


def team_from_ref(ref: object) -> str | None:
    if not isinstance(ref, dict):
        return None
    url = ref.get("$ref")
    if not isinstance(url, str) or "/teams/" not in url:
        return None
    team_id = url.split("/teams/")[1].split("?")[0].strip("/")
    return ESPN_TEAM_IDS.get(team_id)


def needs_fetch(entry: Cached | None, kickoff: datetime, now: datetime, refresh: bool) -> bool:
    if refresh or entry is None:
        return True
    if entry.fetched_at >= kickoff + FINAL_AFTER:
        return False
    return now - entry.fetched_at > LIVE_MAX_AGE


def event_targets(schedule: Schedule, seasons: list[int]) -> pl.DataFrame:
    now = datetime.now(UTC)
    return schedule.games.filter(
        pl.col("season").is_in(seasons),
        pl.col("season") >= FIRST_SEASON,
        pl.col("espn").is_not_null(),
        pl.col("kickoff_utc") <= now + LOOKAHEAD,
    ).sort("kickoff_utc")


def fetch_events(targets: pl.DataFrame, refresh: bool) -> RawCache:
    cache = RawCache(SOURCE)
    now = datetime.now(UTC)
    todo = [
        (row["season"], row["espn"])
        for row in targets.select("season", "espn", "kickoff_utc").iter_rows(named=True)
        if needs_fetch(
            cache.latest(f"{row['season']}_{row['espn']}.json"), row["kickoff_utc"], now, refresh
        )
    ]
    console.print(f"  {SOURCE}: fetching {len(todo)} of {targets.height} events")
    for done, (season, event) in enumerate(todo, start=1):
        name = f"{season}_{event}.json"
        try:
            cache.fetch(name, ODDS_URL.format(event=event), refresh=True, min_interval=MIN_INTERVAL)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            cache.store(name, json.dumps({"count": 0, "items": [], "status": 404}).encode())
        if done % 250 == 0:
            console.print(f"  {SOURCE}: {done}/{len(todo)}")
    return cache


def side(item: dict[str, Any], team: str, phase: str) -> dict[str, Any]:
    odds = item.get(f"{team}TeamOdds")
    if not isinstance(odds, dict):
        return {}
    block = odds.get(phase)
    return block if isinstance(block, dict) else {}


def american_of(block: dict[str, Any], key: str) -> object:
    value = block.get(key)
    return value.get("american") if isinstance(value, dict) else None


def phase_row(item: dict[str, Any], phase: str) -> dict[str, float | None] | None:
    home = side(item, "home", phase)
    away = side(item, "away", phase)
    totals = item.get(phase)
    total_block = totals.get("total") if isinstance(totals, dict) else None
    home_line = parse_number(american_of(home, "pointSpread"))
    away_line = parse_number(american_of(away, "pointSpread"))
    spread = -home_line if home_line is not None else away_line
    row = {
        "home_spread": spread,
        "total": parse_number(total_block.get("american"))
        if isinstance(total_block, dict)
        else None,
        "home_ml": parse_american(american_of(home, "moneyLine")),
        "away_ml": parse_american(american_of(away, "moneyLine")),
    }
    return row if any(v is not None for v in row.values()) else None


def top_row(item: dict[str, Any]) -> dict[str, float | None] | None:
    spread = parse_number(item.get("spread"))
    home = item.get("homeTeamOdds") if isinstance(item.get("homeTeamOdds"), dict) else {}
    away = item.get("awayTeamOdds") if isinstance(item.get("awayTeamOdds"), dict) else {}
    assert isinstance(home, dict) and isinstance(away, dict)
    row = {
        "home_spread": -spread if spread is not None else None,
        "total": parse_number(item.get("overUnder")),
        "home_ml": parse_american(home.get("moneyLine")),
        "away_ml": parse_american(away.get("moneyLine")),
    }
    return row if any(v is not None for v in row.values()) else None


def initial_row(item: dict[str, Any]) -> dict[str, float | None] | None:
    spread = parse_number(item.get("initialSpread"))
    total = parse_number(item.get("initialOverUnder"))
    if spread is None and total is None:
        return None
    return {
        "home_spread": -spread if spread is not None else None,
        "total": total,
        "home_ml": None,
        "away_ml": None,
    }


def book_name(provider: dict[str, Any]) -> str:
    name = str(provider.get("name") or provider.get("id") or "unknown").lower()
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").replace("__", "_")


def parse_event(payload: dict[str, Any], started: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    items = payload.get("items")
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider") if isinstance(item.get("provider"), dict) else {}
        assert isinstance(provider, dict)
        provider_id = str(provider.get("id", ""))
        name = str(provider.get("name", ""))
        if provider_id in PICK_PROVIDERS or "live" in name.lower():
            continue
        book = book_name(provider)
        espn_home = team_from_ref((item.get("homeTeamOdds") or {}).get("team"))
        base = {"book": book, "provider_id": provider_id, "espn_home": espn_home}
        if provider_id == OPENING_PROVIDER:
            top = top_row(item)
            if top:
                rows.append({**base, **top, "kind": "open", "phase": "opening_provider"})
            continue
        initial = initial_row(item)
        if initial:
            rows.append({**base, **initial, "kind": "open", "phase": "initial"})
        opened = phase_row(item, "open")
        if opened:
            rows.append({**base, **opened, "kind": "open", "phase": "open"})
        closed = phase_row(item, "close") if started else None
        if closed:
            rows.append({**base, **closed, "kind": "close", "phase": "close"})
            continue
        top = top_row(item)
        if top:
            rows.append({**base, **top, "kind": "close" if started else "snapshot", "phase": "top"})
    return rows


ROW_SCHEMA = {
    "game_id": pl.String(),
    "season": pl.Int32(),
    "book": pl.String(),
    "provider_id": pl.String(),
    "espn_home": pl.String(),
    "home_spread": pl.Float64(),
    "total": pl.Float64(),
    "home_ml": pl.Float64(),
    "away_ml": pl.Float64(),
    "kind": pl.String(),
    "phase": pl.String(),
    "fetched_at": pl.Datetime("us", "UTC"),
}


def parse_cached(cache: RawCache, targets: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in targets.select("game_id", "season", "espn", "kickoff_utc").iter_rows(named=True):
        for entry in cache.versions(f"{game['season']}_{game['espn']}.json"):
            try:
                payload = json.loads(entry.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            started = entry.fetched_at >= game["kickoff_utc"]
            for row in parse_event(payload, started):
                rows.append(
                    {
                        **row,
                        "game_id": game["game_id"],
                        "season": game["season"],
                        "fetched_at": entry.fetched_at,
                    }
                )
    return pl.DataFrame(rows, schema=ROW_SCHEMA)


def orient_rows(rows: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    joined = rows.join(
        games.select("game_id", "kickoff_utc", "home_team", "home_fr"), on="game_id", how="left"
    ).with_columns(
        pl.col("espn_home")
        .map_elements(safe_franchise, return_dtype=pl.String)
        .alias("espn_home_fr")
    )
    swapped = pl.col("espn_home_fr").is_not_null() & (pl.col("espn_home_fr") != pl.col("home_fr"))
    return joined.with_columns(
        pl.when(swapped)
        .then(-pl.col("home_spread"))
        .otherwise(pl.col("home_spread"))
        .alias("home_spread"),
        pl.when(swapped).then(pl.col("away_ml")).otherwise(pl.col("home_ml")).alias("home_ml"),
        pl.when(swapped).then(pl.col("home_ml")).otherwise(pl.col("away_ml")).alias("away_ml"),
    )


def implausible() -> pl.Expr:
    bad_total = pl.col("total").is_not_null() & ~pl.col("total").is_between(MIN_TOTAL, MAX_TOTAL)
    ml_conflict = (
        pl.col("home_spread").is_not_null()
        & pl.col("home_ml").is_not_null()
        & pl.col("away_ml").is_not_null()
        & (
            ((pl.col("home_spread") >= 3) & (pl.col("home_ml") > pl.col("away_ml")))
            | ((pl.col("home_spread") <= -3) & (pl.col("home_ml") < pl.col("away_ml")))
        )
    )
    deviation = (
        pl.col("home_spread") - pl.col("home_spread").median().over("game_id", "kind")
    ).abs()
    outlier = (pl.len().over("game_id", "kind") >= 3) & (deviation > MAX_BOOK_DEVIATION)
    return (
        pl.when(bad_total)
        .then(pl.lit("implausible_total"))
        .when(ml_conflict)
        .then(pl.lit("spread_moneyline_conflict"))
        .when(outlier.fill_null(False))
        .then(pl.lit("outlier_vs_books"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def build_lines(rows: pl.DataFrame, games: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    oriented = orient_rows(rows, games).with_columns(
        pl.when(pl.col("kind") == "close")
        .then(close_available(pl.col("kickoff_utc")))
        .when(pl.col("kind") == "open")
        .then(tuesday_noon_before(pl.col("kickoff_utc")))
        .otherwise(pl.col("fetched_at"))
        .alias("captured_at")
    )
    flagged = oriented.with_columns(implausible().alias("reason"))
    rejected = flagged.filter(pl.col("reason").is_not_null())
    kept = (
        flagged.filter(pl.col("reason").is_null())
        .with_columns(pl.col("captured_at").alias("available_at"))
        .sort("fetched_at")
        .unique(["game_id", "book", "kind", "captured_at"], keep="last", maintain_order=True)
    )
    return kept.drop("reason", "fetched_at", "espn_home", "espn_home_fr", "home_fr"), rejected


def ingest(options: IngestOptions) -> None:
    schedule = load_schedule(options.refresh)
    targets = event_targets(schedule, options.seasons)
    cache = fetch_events(targets, options.refresh)
    rows = parse_cached(cache, targets)
    lines, rejected = build_lines(rows, schedule.games)
    empty = targets.join(rows.select("game_id").unique(), on="game_id", how="anti").filter(
        pl.col("kickoff_utc") <= datetime.now(UTC)
    )
    rejects = [
        reject_frame(
            SOURCE,
            reason,
            group["season"].to_list(),
            (
                group["game_id"] + ":" + group["book"] + ":" + group["kind"] + ":" + group["phase"]
            ).to_list(),
        )
        for (reason,), group in rejected.group_by("reason")
    ]
    rejects.append(
        reject_frame(
            SOURCE, "no_market_odds", empty["season"].to_list(), empty["game_id"].to_list()
        )
    )
    seasons = [s for s in options.seasons if s >= FIRST_SEASON]
    publish(SOURCE, lines, rejects, seasons)
