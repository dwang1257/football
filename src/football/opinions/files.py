import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import ValidationError

from football.config import get_settings
from football.opinions.blend import BlendParams
from football.opinions.schema import GameOpinion, OpinionFile, TeamOpinion
from football.teams import team_code


class OpinionError(ValueError):
    pass


class _OpinionLoader(yaml.SafeLoader):
    pass


_OpinionLoader.yaml_implicit_resolvers = {
    first: [item for item in resolvers if item[0] != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_OpinionLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


@dataclass(frozen=True)
class GameEntry:
    game_id: str
    pick: str
    conf: int | None
    pick_prob: float
    user_prob: float
    why: str | None


@dataclass(frozen=True)
class OpinionSet:
    path: Path
    sha256: str
    entered_at: datetime
    games: dict[str, GameEntry] = field(default_factory=dict[str, GameEntry])
    nudges: dict[str, float] = field(default_factory=dict[str, float])
    team_whys: dict[str, str] = field(default_factory=dict[str, str])
    late: tuple[str, ...] = ()

    def warnings(self) -> list[str]:
        stamp = self.entered_at.isoformat(timespec="seconds")
        return [
            f"{self.path.name} was last modified at {stamp}, after kickoff of {game_id}; "
            "its opinions for that game are ignored"
            for game_id in self.late
        ]


def opinions_dir() -> Path:
    return get_settings().repo_dir / "opinions"


def opinion_path(season: int, week: int) -> Path:
    return opinions_dir() / f"{season}-w{week:02d}.yaml"


def _entry_label(raw: dict[str, Any], loc: tuple[int | str, ...]) -> str:
    if len(loc) < 2 or loc[0] not in ("games", "teams") or not isinstance(loc[1], int):
        return ""
    entries = raw.get(loc[0])
    entry = entries[loc[1]] if isinstance(entries, list) and loc[1] < len(entries) else None
    label = entry.get("game" if loc[0] == "games" else "team") if isinstance(entry, dict) else None
    return f" ({label})" if label else ""


def _format_validation(path: Path, raw: dict[str, Any], error: ValidationError) -> str:
    lines = [f"{path}: invalid opinion file"]
    for issue in error.errors():
        where = ".".join(str(part) for part in issue["loc"]) or "file"
        lines.append(f"  {where}{_entry_label(raw, issue['loc'])}: {issue['msg']}")
    return "\n".join(lines)


def parse_opinion_text(text: str, path: Path) -> OpinionFile:
    try:
        raw: Any = yaml.load(text, Loader=_OpinionLoader)
    except yaml.YAMLError as error:
        raise OpinionError(f"{path}: not valid YAML: {error}") from error
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise OpinionError(f"{path}: expected a mapping with games and teams")
    try:
        return OpinionFile.model_validate(raw)
    except ValidationError as error:
        raise OpinionError(_format_validation(path, raw, error)) from error


def _team(text: str, season: int, where: str) -> str:
    try:
        return team_code(text, season)
    except ValueError as error:
        raise OpinionError(f"{where}: unknown team {text!r}") from error


def _match_game(entry: GameOpinion, season: int, week: int, games: pl.DataFrame, where: str) -> str:
    ref = entry.game.strip()
    ids = set(games["game_id"].to_list())
    if ref in ids:
        return ref
    if "@" not in ref:
        raise OpinionError(
            f"{where}: unknown game {ref!r}; use AWAY@HOME or a game_id from week {week}"
        )
    away_text, home_text = ref.split("@", 1)
    away = _team(away_text, season, where)
    home = _team(home_text, season, where)
    found = games.filter((pl.col("away_team") == away) & (pl.col("home_team") == home))
    if found.height == 1:
        return str(found["game_id"][0])
    swapped = games.filter((pl.col("away_team") == home) & (pl.col("home_team") == away))
    hint = f"; did you mean {home}@{away}?" if swapped.height else ""
    raise OpinionError(f"{where}: no game {away}@{home} in {season} week {week}{hint}")


def _resolve_games(
    parsed: OpinionFile, season: int, week: int, games: pl.DataFrame, params: BlendParams
) -> dict[str, GameEntry]:
    teams_by_game = {
        str(row["game_id"]): (str(row["away_team"]), str(row["home_team"]))
        for row in games.iter_rows(named=True)
    }
    resolved: dict[str, GameEntry] = {}
    seen: set[str] = set()
    for index, entry in enumerate(parsed.games):
        where = f"games.{index} ({entry.game})"
        game_id = _match_game(entry, season, week, games, where)
        if game_id in seen:
            raise OpinionError(f"{where}: game {game_id} is listed more than once")
        seen.add(game_id)
        if entry.pick is None:
            continue
        away, home = teams_by_game[game_id]
        pick = _team(entry.pick, season, where)
        if pick not in (away, home):
            raise OpinionError(f"{where}: pick {pick} is not playing in {away}@{home}")
        pick_prob = params.pick_prob(entry.conf, entry.prob)
        resolved[game_id] = GameEntry(
            game_id=game_id,
            pick=pick,
            conf=entry.conf,
            pick_prob=pick_prob,
            user_prob=pick_prob if pick == home else 1.0 - pick_prob,
            why=entry.why,
        )
    return resolved


def _resolve_teams(
    entries: list[TeamOpinion], season: int, week: int, playing: set[str]
) -> tuple[dict[str, float], dict[str, str]]:
    points: dict[str, float] = {}
    whys: dict[str, str] = {}
    for index, entry in enumerate(entries):
        where = f"teams.{index} ({entry.team})"
        team = _team(entry.team, season, where)
        if team in points:
            raise OpinionError(f"{where}: team {team} is listed more than once")
        if team not in playing:
            raise OpinionError(f"{where}: {team} does not play in {season} week {week}")
        points[team] = entry.points
        if entry.why:
            whys[team] = entry.why
    return points, whys


def _check_header(parsed: OpinionFile, season: int, week: int, path: Path) -> None:
    if parsed.season is not None and parsed.season != season:
        raise OpinionError(f"{path}: file says season {parsed.season}, expected {season}")
    if parsed.week is not None and parsed.week != week:
        raise OpinionError(f"{path}: file says week {parsed.week}, expected {week}")


def load_opinions(
    season: int, week: int, games: pl.DataFrame, params: BlendParams
) -> OpinionSet | None:
    path = opinion_path(season, week)
    if not path.exists():
        return None
    entered_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    payload = path.read_bytes()
    parsed = parse_opinion_text(payload.decode("utf-8"), path)
    _check_header(parsed, season, week, path)
    week_games = games.filter((pl.col("season") == season) & (pl.col("week") == week))
    entries = _resolve_games(parsed, season, week, week_games, params)
    playing = set(week_games["away_team"].to_list()) | set(week_games["home_team"].to_list())
    points, whys = _resolve_teams(parsed.teams, season, week, playing)
    late: list[str] = []
    counted: dict[str, GameEntry] = {}
    nudges: dict[str, float] = {}
    for row in week_games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        game_id = str(row["game_id"])
        nudge = points.get(row["home_team"], 0.0) - points.get(row["away_team"], 0.0)
        has_opinion = game_id in entries or nudge != 0.0
        if has_opinion and entered_at >= row["kickoff_utc"]:
            late.append(game_id)
            continue
        if game_id in entries:
            counted[game_id] = entries[game_id]
        if nudge != 0.0:
            nudges[game_id] = nudge
    return OpinionSet(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        entered_at=entered_at,
        games=counted,
        nudges=nudges,
        team_whys=whys,
        late=tuple(late),
    )
