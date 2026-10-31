from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from football.collect.archive import Snapshot
from football.collect.base import CollectorResult, Context
from football.collect.espn import parse_time
from football.collect.games import GameResolver, nflverse_week
from football.collect.http import FetchError

API = "https://api.actionnetwork.com/web"
SCOREBOARD_URL = f"{API}/v1/scoreboard/nfl"
REGULAR_SEASON_WEEKS = 18

BOOKS: dict[int, str] = {
    15: "consensus",
    30: "open",
    49: "caesars_nv",
    68: "draftkings",
    69: "fanduel",
    71: "betrivers",
    75: "betmgm",
    79: "bet365",
    123: "caesars",
}


@dataclass(frozen=True)
class AnGame:
    event_id: int
    season: int
    week: int | None
    season_type: str
    start: datetime
    status: str
    home: str
    away: str

    def meta(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "season": self.season,
            "week": self.week,
            "season_type": self.season_type,
            "start_time": self.start.isoformat(),
            "status": self.status,
            "home": self.home,
            "away": self.away,
        }


def book_name(book_id: int | str) -> str:
    return BOOKS.get(int(book_id), f"an_{book_id}")


def parse_games(body: dict[str, Any]) -> list[AnGame]:
    games: list[AnGame] = []
    for game in body.get("games", []):
        abbrs = {int(team["id"]): str(team["abbr"]) for team in game.get("teams", [])}
        home = abbrs.get(int(game["home_team_id"]))
        away = abbrs.get(int(game["away_team_id"]))
        if home is None or away is None:
            continue
        week = game.get("week")
        games.append(
            AnGame(
                event_id=int(game["id"]),
                season=int(game["season"]),
                week=int(week) if week is not None else None,
                season_type=str(game.get("type", "")),
                start=parse_time(str(game["start_time"])),
                status=str(game.get("status", "")),
                home=home,
                away=away,
            )
        )
    return games


def _save_scoreboard(
    ctx: Context, result: CollectorResult, params: dict[str, str | int]
) -> list[AnGame]:
    fetched = ctx.fetcher.get(SCOREBOARD_URL, params=params)
    label = f"w{params['week']}" if "week" in params else "current"
    if not fetched.ok:
        ctx.save(result, f"scoreboard_{label}", fetched, fetched.text[:2000], dict(params))
        raise FetchError(f"scoreboard {label}: HTTP {fetched.status}")
    body = fetched.json()
    games = parse_games(body)
    if games:
        label = f"{games[0].season}_w{games[0].week:02d}" if games[0].week else label
    ctx.save(result, f"scoreboard_{label}", fetched, body, dict(params))
    return games


def collect(ctx: Context, result: CollectorResult) -> None:
    games = _save_scoreboard(ctx, result, {"period": "game"})
    result.ok.append("scoreboard_current")
    if games and games[0].season_type == "reg" and games[0].week:
        week = games[0].week + 1
        if week <= REGULAR_SEASON_WEEKS:
            params: dict[str, str | int] = {
                "period": "game",
                "season": games[0].season,
                "week": week,
            }
            try:
                games += _save_scoreboard(ctx, result, params)
                result.ok.append(f"scoreboard_w{week}")
            except FetchError as exc:
                result.failed[f"scoreboard_w{week}"] = str(exc)
    upcoming = [game for game in games if game.start > ctx.started_at and game.status != "complete"]
    for game in upcoming:
        url = f"{API}/v2/markets/event/{game.event_id}/history"
        name = f"history_{game.event_id}"
        try:
            fetched = ctx.fetcher.get(url)
        except FetchError as exc:
            result.failed[name] = str(exc)
            continue
        if not fetched.ok:
            ctx.save(result, name, fetched, fetched.text[:2000], game.meta())
            result.failed[name] = f"HTTP {fetched.status}"
            continue
        ctx.save_json(result, name, fetched, game.meta())
        result.ok.append(name)
    result.notes.append(f"{len(games)} games listed, {len(upcoming)} upcoming histories")


def _ceil_second(moment: datetime) -> datetime:
    if moment.microsecond:
        return moment.replace(microsecond=0) + timedelta(seconds=1)
    return moment


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row(gid: str, book: str, kind: str, moment: datetime, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": gid,
        "source": "action_network",
        "book": book,
        "kind": kind,
        "captured_at": moment,
        "home_spread": state.get("home_spread"),
        "total": state.get("total"),
        "home_ml": state.get("home_ml"),
        "away_ml": state.get("away_ml"),
        "available_at": moment,
    }


def _scoreboard_rows(snapshot: Snapshot, resolver: GameResolver) -> Iterable[dict[str, Any]]:
    body = snapshot.body if isinstance(snapshot.body, dict) else {}
    raw = {int(game["id"]): game for game in body.get("games", [])}
    for game in parse_games(body):
        if snapshot.fetched_at >= game.start:
            continue
        gid = _resolve(resolver, game)
        if gid is None:
            continue
        for odds in raw[game.event_id].get("odds") or []:
            if odds.get("type", "game") != "game" or odds.get("book_id") is None:
                continue
            spread_home = _float(odds.get("spread_home"))
            state = {
                "home_spread": 0.0 - spread_home if spread_home is not None else None,
                "total": _float(odds.get("total")),
                "home_ml": _float(odds.get("ml_home")),
                "away_ml": _float(odds.get("ml_away")),
            }
            if all(value is None for value in state.values()):
                continue
            yield _row(gid, book_name(odds["book_id"]), "snapshot", snapshot.fetched_at, state)


def _resolve(resolver: GameResolver, game: AnGame) -> str | None:
    week = nflverse_week(game.season_type, game.week)
    return resolver.resolve(game.season, week, game.home, game.away, game.start)


def _outcome_field(market: str, side: str) -> tuple[str, float] | None:
    if market == "spread" and side == "home":
        return "home_spread", -1.0
    if market == "total" and side == "over":
        return "total", 1.0
    if market == "moneyline" and side == "home":
        return "home_ml", 1.0
    if market == "moneyline" and side == "away":
        return "away_ml", 1.0
    return None


def _ticks(body: dict[str, Any], start: datetime) -> Iterable[tuple[str, str, datetime, float]]:
    for book_id, book_body in body.items():
        event = book_body.get("event") if isinstance(book_body, dict) else None
        if not isinstance(event, dict):
            continue
        for market, outcomes in event.items():
            for outcome in outcomes or []:
                if outcome.get("period", "event") != "event" or outcome.get("is_live"):
                    continue
                target = _outcome_field(str(market), str(outcome.get("side", "")))
                if target is None:
                    continue
                name, sign = target
                key = "odds" if name.endswith("_ml") else "value"
                for tick in outcome.get("history") or []:
                    raw = _float(tick.get(key))
                    stamp = tick.get("updated_at")
                    if raw is None or not stamp:
                        continue
                    moment = parse_time(str(stamp))
                    if _ceil_second(moment) < start:
                        yield book_name(book_id), name, moment, sign * raw + 0.0


def history_lines(snapshots: Iterable[Snapshot], resolver: GameResolver) -> list[dict[str, Any]]:
    ticks: dict[tuple[str, str], set[tuple[datetime, str, float]]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if not snapshot.ok or not isinstance(snapshot.body, dict):
            continue
        if snapshot.name.startswith("scoreboard"):
            rows.extend(_scoreboard_rows(snapshot, resolver))
            continue
        meta = snapshot.meta
        start = parse_time(str(meta["start_time"]))
        game = AnGame(
            event_id=int(meta["event_id"]),
            season=int(meta["season"]),
            week=meta.get("week"),
            season_type=str(meta.get("season_type", "")),
            start=start,
            status=str(meta.get("status", "")),
            home=str(meta["home"]),
            away=str(meta["away"]),
        )
        gid = _resolve(resolver, game)
        if gid is None:
            continue
        for book, name, moment, value in _ticks(snapshot.body, start):
            ticks[(gid, book)].add((moment, name, value))
    for (gid, book), items in ticks.items():
        rows.extend(_timeline_rows(gid, book, sorted(items)))
    return rows


def _timeline_rows(
    gid: str, book: str, items: list[tuple[datetime, str, float]]
) -> list[dict[str, Any]]:
    by_second: dict[datetime, list[tuple[str, float]]] = defaultdict(list)
    for moment, name, value in items:
        by_second[_ceil_second(moment)].append((name, value))
    state: dict[str, Any] = {}
    first: dict[str, tuple[datetime, float]] = {}
    rows: list[dict[str, Any]] = []
    for moment in sorted(by_second):
        for name, value in by_second[moment]:
            state[name] = value
            first.setdefault(name, (moment, value))
        rows.append(_row(gid, book, "snapshot", moment, state))
    if first:
        opened = max(moment for moment, _ in first.values())
        rows.append(_row(gid, book, "open", opened, {name: v for name, (_, v) in first.items()}))
    return rows
