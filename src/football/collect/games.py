from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

from football import storage
from football.teams import game_id, team_code

MAX_KICKOFF_GAP = timedelta(days=3)
ESPN_POSTSEASON_WEEKS = {1: 19, 2: 20, 3: 21, 5: 22}


def nflverse_week(season_type: int | str | None, week: int | None) -> int | None:
    if week is None:
        return None
    kind = str(season_type).lower()
    if kind in {"2", "reg", "regular"}:
        return week
    if kind in {"3", "post", "postseason"}:
        return ESPN_POSTSEASON_WEEKS.get(week)
    return None


@dataclass
class GameResolver:
    by_matchup: dict[tuple[int, str, str], list[tuple[datetime | None, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @classmethod
    def load(cls) -> "GameResolver":
        resolver = cls()
        if storage.exists("clean", "games"):
            resolver.add_games(storage.read("clean", "games"))
        return resolver

    def add_games(self, games: pl.DataFrame) -> None:
        columns = ["game_id", "season", "home_team", "away_team", "kickoff_utc"]
        for row in games.select(columns).iter_rows(named=True):
            key = (int(row["season"]), str(row["home_team"]), str(row["away_team"]))
            self.by_matchup[key].append((row["kickoff_utc"], str(row["game_id"])))

    def resolve(
        self,
        season: int,
        week: int | None,
        home: str,
        away: str,
        kickoff: datetime | None,
    ) -> str | None:
        try:
            home_code = team_code(home, season)
            away_code = team_code(away, season)
        except ValueError:
            return None
        candidates = self.by_matchup.get((season, home_code, away_code), [])
        if kickoff is not None:
            timed = [(abs(start - kickoff), gid) for start, gid in candidates if start is not None]
            close = sorted(item for item in timed if item[0] <= MAX_KICKOFF_GAP)
            if close:
                return close[0][1]
        if week is None:
            return None
        computed = game_id(season, week, away_code, home_code)
        if candidates and computed not in {gid for _, gid in candidates}:
            return None
        return computed
