from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from football import storage
from football.models import registry
from football.models.base import Model, Track
from football.schemas import FEATURES, UTC_TS, validate

TEAMS: tuple[str, ...] = tuple(f"T{index}" for index in range(8))
SEASONS: tuple[int, ...] = tuple(range(2018, 2026))
REGULAR_WEEKS = 6
PLAYOFF_WEEK = REGULAR_WEEKS + 1
SIGMA = 13.0
CANCELLED_GAME = "2020_99_T0_T1"


def synthetic_features(seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0.0, 4.0, len(TEAMS))
    rows: list[dict[str, Any]] = []

    def add(season: int, week: int, game_type: str, slot: int, home: int, away: int) -> None:
        kickoff = datetime(season, 9, 8, 17, tzinfo=UTC) + timedelta(
            days=7 * (week - 1), hours=slot
        )
        diff = float(strength[home] - strength[away] + 1.5)
        margin = float(round(diff + rng.normal(0.0, SIGMA)))
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_{TEAMS[away]}_{TEAMS[home]}",
                "season": season,
                "week": week,
                "game_type": game_type,
                "kickoff_utc": kickoff,
                "cutoff": kickoff - timedelta(hours=1),
                "home_team": TEAMS[home],
                "away_team": TEAMS[away],
                "home_win": 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5,
                "home_margin": margin,
                "strength_diff": diff,
            }
        )

    for season in SEASONS:
        strength = 0.7 * strength + rng.normal(0.0, 2.0, len(TEAMS))
        for week in range(1, REGULAR_WEEKS + 1):
            order = rng.permutation(len(TEAMS))
            for slot in range(len(TEAMS) // 2):
                add(season, week, "REG", 3 * slot, int(order[2 * slot]), int(order[2 * slot + 1]))
        ranked = np.argsort(-strength)
        add(season, PLAYOFF_WEEK, "WC", 0, int(ranked[0]), int(ranked[3]))
        add(season, PLAYOFF_WEEK, "WC", 3, int(ranked[1]), int(ranked[2]))
    frame = pl.DataFrame(rows).with_columns(
        pl.col("season", "week").cast(pl.Int32),
        pl.col("kickoff_utc", "cutoff").cast(UTC_TS),
    )
    cancelled = (
        frame.filter(pl.col("season") == 2020)
        .head(1)
        .with_columns(
            game_id=pl.lit(CANCELLED_GAME),
            home_win=pl.lit(None, pl.Float64),
            home_margin=pl.lit(None, pl.Float64),
        )
    )
    return validate(pl.concat([frame, cancelled]), FEATURES)


class HomeRateModel:
    name = "home_rate"
    track: Track = "independent"

    def __init__(self) -> None:
        self.rate = 0.5

    def fit(self, train: pl.DataFrame) -> None:
        self.rate = float(train["home_win"].to_numpy().mean())

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        if "home_win" in games.columns or "home_margin" in games.columns:
            raise AssertionError("labels leaked into predict")
        return games.select(
            "game_id", home_win_prob=pl.lit(self.rate), home_margin=pl.lit(None, pl.Float64)
        )


class StrengthModel:
    name = "strength"
    track: Track = "independent"

    def __init__(self) -> None:
        self.slope = 1.0

    def fit(self, train: pl.DataFrame) -> None:
        x = train["strength_diff"].to_numpy()
        y = train["home_margin"].to_numpy()
        self.slope = float(np.dot(x, y) / np.dot(x, x))

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        if "home_win" in games.columns or "home_margin" in games.columns:
            raise AssertionError("labels leaked into predict")
        margin = self.slope * games["strength_diff"].to_numpy()
        return pl.DataFrame(
            {
                "game_id": games["game_id"],
                "home_win_prob": norm.cdf(margin / SIGMA),
                "home_margin": margin,
            }
        )


FAKES: dict[str, Callable[[], Model]] = {
    "home_rate": HomeRateModel,
    "strength": StrengthModel,
    "elo": StrengthModel,
}


@pytest.fixture
def factories() -> dict[str, Callable[[], Model]]:
    return FAKES


@pytest.fixture(scope="session")
def features() -> pl.DataFrame:
    return synthetic_features()


@pytest.fixture
def stored_features(features: pl.DataFrame) -> pl.DataFrame:
    storage.write(features, "features", FEATURES.name)
    return features


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def build(name: str, **params: Any) -> Model:
        calls.append((name, params))
        return FAKES[name]()

    monkeypatch.setattr(registry, "build", build)
    return calls
