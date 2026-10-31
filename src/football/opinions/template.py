from pathlib import Path
from typing import Any

import polars as pl
import yaml

from football.opinions.files import OpinionError, opinion_path
from football.predict.store import latest_model_probs, week_games


def template_document(
    season: int, week: int, games: pl.DataFrame, model_probs: dict[str, float] | None
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        entry: dict[str, Any] = {
            "game": f"{row['away_team']}@{row['home_team']}",
            "kickoff": row["kickoff_utc"].isoformat(timespec="minutes"),
        }
        if model_probs is not None and row["game_id"] in model_probs:
            entry["model_prob"] = round(model_probs[row["game_id"]], 3)
        entry.update({"pick": None, "conf": None, "why": None})
        entries.append(entry)
    return {"season": season, "week": week, "games": entries, "teams": []}


def write_template(season: int, week: int, blind: bool, force: bool) -> Path:
    path = opinion_path(season, week)
    if path.exists() and not force:
        raise OpinionError(f"{path} already exists; pass --force to overwrite it")
    games = week_games(season, week)
    probs = None if blind else latest_model_probs(games["game_id"].to_list())
    document = template_document(season, week, games, probs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    return path
