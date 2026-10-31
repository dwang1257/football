from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from football import storage
from football.config import get_settings
from football.schemas import UTC_TS


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "repo"
    path.mkdir()
    monkeypatch.setenv("FOOTBALL_REPO_DIR", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def games() -> pl.DataFrame:
    rows = [
        ("2026_09_KC_BUF", "KC", "BUF", datetime(2026, 11, 8, 18, 0, tzinfo=UTC)),
        ("2026_09_NYG_ATL", "NYG", "ATL", datetime(2026, 11, 8, 21, 25, tzinfo=UTC)),
        ("2026_09_LA_SF", "LA", "SF", datetime(2026, 11, 6, 1, 15, tzinfo=UTC)),
    ]
    return pl.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [2026] * len(rows),
            "week": [9] * len(rows),
            "away_team": [r[1] for r in rows],
            "home_team": [r[2] for r in rows],
            "kickoff_utc": [r[3] for r in rows],
            "home_score": [None] * len(rows),
            "away_score": [None] * len(rows),
        },
        schema_overrides={
            "season": pl.Int32,
            "week": pl.Int32,
            "kickoff_utc": UTC_TS,
            "home_score": pl.Int32,
            "away_score": pl.Int32,
        },
    )


@pytest.fixture
def stored_games(games: pl.DataFrame) -> pl.DataFrame:
    storage.write(games, "clean", "games")
    return games
