from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from football import storage
from football.config import get_settings
from football.predict import run
from football.schemas import UTC_TS

BASE = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
HOME_PROBS = {
    "2026_08_DAL_PHI": 0.6,
    "2026_08_MIA_NE": 0.45,
    "2026_09_LA_SF": 0.62,
    "2026_09_KC_BUF": 0.30,
    "2026_09_NYG_ATL": 0.55,
}


def schedule(base: datetime) -> pl.DataFrame:
    rows = [
        ("2026_08_DAL_PHI", 8, "DAL", "PHI", base - timedelta(days=3), 24, 20),
        ("2026_08_MIA_NE", 8, "MIA", "NE", base - timedelta(days=3, hours=-3), 17, 10),
        (
            "2026_09_LA_SF",
            9,
            "LA",
            "SF",
            base + timedelta(days=1, hours=13, minutes=15),
            None,
            None,
        ),
        ("2026_09_KC_BUF", 9, "KC", "BUF", base + timedelta(days=4, hours=6), None, None),
        (
            "2026_09_NYG_ATL",
            9,
            "NYG",
            "ATL",
            base + timedelta(days=4, hours=9, minutes=25),
            None,
            None,
        ),
    ]
    names = ("game_id", "week", "away_team", "home_team", "kickoff_utc", "home_score", "away_score")
    return pl.DataFrame(
        [dict(zip(names, row, strict=True)) for row in rows],
        schema={
            "game_id": pl.String,
            "week": pl.Int32,
            "away_team": pl.String,
            "home_team": pl.String,
            "kickoff_utc": UTC_TS,
            "home_score": pl.Int32,
            "away_score": pl.Int32,
        },
    ).with_columns(pl.lit(2026, dtype=pl.Int32).alias("season"), pl.lit("REG").alias("game_type"))


class FakeModel:
    def __init__(self, track: str = "independent") -> None:
        self.name = "fake"
        self.track: Any = track
        self.train: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame) -> None:
        self.train = train

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        assert "home_win" not in games.columns
        return games.select("game_id", home_win_prob=pl.col("x"), home_margin=pl.lit(1.5))


class Fakes:
    def __init__(self) -> None:
        self.models: list[FakeModel] = []
        self.feature_calls: list[tuple[datetime | None, Any]] = []

    def build_features(self, as_of: datetime | None = None, seasons: Any = None) -> pl.DataFrame:
        self.feature_calls.append((as_of, seasons))
        games = storage.read("clean", "games")
        diff = pl.col("home_score") - pl.col("away_score")
        assert as_of is not None
        return games.select(
            "game_id",
            "season",
            "week",
            "game_type",
            "kickoff_utc",
            pl.min_horizontal("kickoff_utc", pl.lit(as_of).cast(UTC_TS)).alias("cutoff"),
            "home_team",
            "away_team",
            pl.when(diff > 0).then(1.0).when(diff < 0).then(0.0).alias("home_win"),
            diff.cast(pl.Float64).alias("home_margin"),
            pl.col("game_id").replace_strict(HOME_PROBS, return_dtype=pl.Float64).alias("x"),
        )

    def build_model(self, name: str, **params: Any) -> FakeModel:
        model = FakeModel(**params)
        self.models.append(model)
        return model


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "repo"
    path.mkdir()
    monkeypatch.setenv("FOOTBALL_REPO_DIR", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def fakes(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Fakes:
    storage.write(schedule(BASE), "clean", "games")
    fake = Fakes()
    monkeypatch.setattr(run, "build_features", fake.build_features)
    monkeypatch.setattr(run.registry, "build", fake.build_model)
    return fake
