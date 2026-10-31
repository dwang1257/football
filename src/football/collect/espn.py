import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from football.collect.archive import Snapshot
from football.collect.base import CollectorResult, Context
from football.collect.games import GameResolver, nflverse_week
from football.collect.http import Fetched, FetchError
from football.teams import team_code

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

ESPN_TEAM_IDS: dict[str, int] = {
    "ARI": 22,
    "ATL": 1,
    "BAL": 33,
    "BUF": 2,
    "CAR": 29,
    "CHI": 3,
    "CIN": 4,
    "CLE": 5,
    "DAL": 6,
    "DEN": 7,
    "DET": 8,
    "GB": 9,
    "HOU": 34,
    "IND": 11,
    "JAX": 30,
    "KC": 12,
    "LA": 14,
    "LAC": 24,
    "LV": 13,
    "MIA": 15,
    "MIN": 16,
    "NE": 17,
    "NO": 18,
    "NYG": 19,
    "NYJ": 20,
    "PHI": 21,
    "PIT": 23,
    "SEA": 26,
    "SF": 25,
    "TB": 27,
    "TEN": 10,
    "WAS": 28,
}
TEAM_BY_ESPN_ID = {espn_id: team for team, espn_id in ESPN_TEAM_IDS.items()}

_TEAM_REF = re.compile(r"/teams/(\d+)")


@dataclass(frozen=True)
class EspnWeek:
    season: int
    season_type: int
    week: int
    nflverse_week: int | None


@dataclass(frozen=True)
class EspnEvent:
    event_id: str
    kickoff: datetime
    state: str
    home: str
    away: str


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def https(url: str) -> str:
    return re.sub(r"^http://", "https://", url)


def scoreboard(ctx: Context) -> Fetched:
    cached = ctx.cache.get("espn_scoreboard")
    if isinstance(cached, Fetched):
        return cached
    fetched = ctx.fetcher.get(SCOREBOARD_URL)
    if not fetched.ok:
        raise FetchError(f"{SCOREBOARD_URL}: HTTP {fetched.status}")
    ctx.cache["espn_scoreboard"] = fetched
    return fetched


def scoreboard_week(body: dict[str, Any]) -> EspnWeek:
    season = int(body["season"]["year"])
    season_type = int(body["season"]["type"])
    week = int(body["week"]["number"])
    return EspnWeek(season, season_type, week, nflverse_week(season_type, week))


def scoreboard_events(body: dict[str, Any]) -> list[EspnEvent]:
    events: list[EspnEvent] = []
    for event in body.get("events", []):
        competition = event["competitions"][0]
        sides = {
            item["homeAway"]: item["team"]["abbreviation"] for item in competition["competitors"]
        }
        events.append(
            EspnEvent(
                event_id=str(event["id"]),
                kickoff=parse_time(event["date"]),
                state=str(competition["status"]["type"]["state"]),
                home=sides["home"],
                away=sides["away"],
            )
        )
    return events


def week_meta(week: EspnWeek) -> dict[str, Any]:
    return {
        "season": week.season,
        "season_type": week.season_type,
        "espn_week": week.week,
        "week": week.nflverse_week,
    }


def collect_odds(ctx: Context, result: CollectorResult) -> None:
    fetched = scoreboard(ctx)
    body = ctx.save_json(result, "scoreboard", fetched)
    week = scoreboard_week(body)
    upcoming = [event for event in scoreboard_events(body) if event.state == "pre"]
    for event in upcoming:
        url = f"{CORE}/events/{event.event_id}/competitions/{event.event_id}/odds"
        meta = {
            **week_meta(week),
            "event_id": event.event_id,
            "home": event.home,
            "away": event.away,
            "kickoff": event.kickoff.isoformat(),
        }
        try:
            odds = ctx.fetcher.get(url, params={"limit": 100})
        except FetchError as exc:
            result.failed[event.event_id] = str(exc)
            continue
        if not odds.ok:
            ctx.save(result, f"odds_{event.event_id}", odds, odds.text[:2000], meta)
            result.failed[event.event_id] = f"HTTP {odds.status}"
            continue
        payload = ctx.save_json(result, f"odds_{event.event_id}", odds, meta)
        result.ok.append(event.event_id)
        result.notes.append(f"{event.away}@{event.home}: {len(payload.get('items', []))} books")
    if not upcoming:
        result.notes.append(f"no upcoming events in ESPN week {week.week}")


def collect_fpi(ctx: Context, result: CollectorResult) -> None:
    week = scoreboard_week(scoreboard(ctx).json())
    url = f"{CORE}/seasons/{week.season}/powerindex"
    fetched = ctx.fetcher.get(url, params={"limit": 100})
    if not fetched.ok:
        ctx.save(result, "powerindex", fetched, fetched.text[:2000], week_meta(week))
        result.failed["powerindex"] = f"HTTP {fetched.status}"
        return
    body = ctx.save_json(result, "powerindex", fetched, week_meta(week))
    count = len(body.get("items", []))
    result.ok.append("powerindex")
    result.notes.append(f"{count} teams, season {week.season} week {week.week}")


def _resolve_refs(ctx: Context, refs: list[str]) -> tuple[list[Any], list[str]]:
    def fetch(url: str) -> Any:
        fetched = ctx.fetcher.get(https(url))
        if not fetched.ok:
            raise FetchError(f"{url}: HTTP {fetched.status}")
        return fetched.json()

    items: list[Any] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, url) for url in refs]
        for url, future in zip(refs, futures, strict=True):
            try:
                items.append(future.result())
            except (FetchError, ValueError) as exc:
                errors.append(f"{url}: {exc}")
    return items, errors


def collect_injuries(ctx: Context, result: CollectorResult) -> None:
    total = 0
    for team, espn_id in sorted(ESPN_TEAM_IDS.items()):
        url = f"{CORE}/teams/{espn_id}/injuries"
        try:
            listing = ctx.fetcher.get(url, params={"limit": 500})
        except FetchError as exc:
            result.failed[team] = str(exc)
            continue
        if not listing.ok:
            ctx.save(result, team, listing, listing.text[:2000], {"team": team})
            result.failed[team] = f"HTTP {listing.status}"
            continue
        body = listing.json()
        refs = [str(item["$ref"]) for item in body.get("items", []) if "$ref" in item]
        items, errors = _resolve_refs(ctx, refs)
        meta = {"team": team, "espn_team_id": espn_id, "listed": len(refs), "resolved": len(items)}
        ctx.save(result, team, listing, {"list": body, "items": items, "errors": errors}, meta)
        total += len(items)
        if errors and not items:
            result.failed[team] = f"{len(errors)} entries failed"
            continue
        result.ok.append(team)
        if errors:
            result.notes.append(f"{team}: {len(errors)} of {len(refs)} entries failed")
    result.notes.append(f"{total} injury entries")


def _team_from_ref(ref: dict[str, Any] | None) -> str | None:
    if not ref:
        return None
    match = _TEAM_REF.search(str(ref.get("$ref", "")))
    return TEAM_BY_ESPN_ID.get(int(match.group(1))) if match else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def odds_lines(snapshots: Iterable[Snapshot], resolver: GameResolver) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if not snapshot.ok or not isinstance(snapshot.body, dict):
            continue
        meta = snapshot.meta
        kickoff = parse_time(str(meta["kickoff"]))
        if snapshot.fetched_at >= kickoff:
            continue
        gid = resolver.resolve(
            int(meta["season"]), meta.get("week"), str(meta["home"]), str(meta["away"]), kickoff
        )
        if gid is None:
            continue
        for item in snapshot.body.get("items", []):
            provider = item.get("provider") or {}
            book = re.sub(r"[^a-z0-9]+", "_", str(provider.get("name", "unknown")).lower())
            spread = _number(item.get("spread"))
            rows.append(
                {
                    "game_id": gid,
                    "source": "espn",
                    "book": book.strip("_") or "unknown",
                    "kind": "snapshot",
                    "captured_at": snapshot.fetched_at,
                    "home_spread": 0.0 - spread if spread is not None else None,
                    "total": _number(item.get("overUnder")),
                    "home_ml": _number((item.get("homeTeamOdds") or {}).get("moneyLine")),
                    "away_ml": _number((item.get("awayTeamOdds") or {}).get("moneyLine")),
                    "available_at": snapshot.fetched_at,
                }
            )
    return rows


def _predictives(item: dict[str, Any]) -> dict[str, float | None]:
    return {
        str(entry.get("name")): _number(entry.get("value"))
        for entry in item.get("predictives", [])
        if isinstance(entry, dict)
    }


def fpi_ratings(snapshots: Iterable[Snapshot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if not snapshot.ok or not isinstance(snapshot.body, dict):
            continue
        week = snapshot.meta.get("week")
        if week is None:
            continue
        for item in snapshot.body.get("items", []):
            franchise_code = _team_from_ref(item.get("team"))
            season = int(item.get("season") or snapshot.meta["season"])
            if franchise_code is None or season != int(snapshot.meta["season"]):
                continue
            updated = item.get("lastUpdated")
            captured = parse_time(str(updated)) if updated else snapshot.fetched_at
            values = _predictives(item)
            rows.append(
                {
                    "source": "espn_fpi_archive",
                    "season": season,
                    "week": int(week),
                    "team": team_code(franchise_code, season),
                    "rating": values.get("fpi"),
                    "offense": values.get("epaoffense"),
                    "defense": values.get("epadefense"),
                    "special_teams": values.get("epaspecialteams"),
                    "captured_at": min(captured, snapshot.fetched_at),
                    "available_at": snapshot.fetched_at,
                }
            )
    return rows
