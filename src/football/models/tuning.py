import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from football.models.common import PROB_EPS, seasons_of
from football.models.registry import build


@dataclass(frozen=True)
class Trial:
    params: dict[str, Any]
    season_log_loss: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.season_log_loss))

    @property
    def stderr(self) -> float:
        n = len(self.season_log_loss)
        return float(np.std(self.season_log_loss, ddof=1) / np.sqrt(n)) if n > 1 else 0.0


@dataclass(frozen=True)
class TuneResult:
    best: Trial
    trials: tuple[Trial, ...]


def grid(space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*space.values())]


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs, PROB_EPS, 1 - PROB_EPS)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def inner_seasons(train: pl.DataFrame, count: int = 3, min_train: int = 3) -> list[int]:
    seasons = seasons_of(train.filter(pl.col("home_win").is_not_null()))
    return seasons[min_train:][-count:]


def walk_forward(
    name: str, params: Mapping[str, Any], data: pl.DataFrame, test_seasons: Sequence[int]
) -> tuple[float, ...]:
    scores: list[float] = []
    for season in test_seasons:
        train = data.filter(pl.col("season") < season)
        test = data.filter((pl.col("season") == season) & pl.col("home_win").is_not_null())
        model = build(name, **params)
        model.fit(train)
        pred = model.predict(test).join(test.select("game_id", "home_win"), on="game_id")
        scores.append(log_loss(pred["home_win_prob"].to_numpy(), pred["home_win"].to_numpy()))
    return tuple(scores)


def tune(
    name: str,
    space: Mapping[str, Sequence[Any]],
    train: pl.DataFrame,
    fixed: Mapping[str, Any] | None = None,
    seasons: Sequence[int] | None = None,
) -> TuneResult:
    test_seasons = list(seasons) if seasons is not None else inner_seasons(train)
    if not test_seasons:
        raise ValueError("tuning needs at least one inner validation season")
    trials = tuple(
        Trial(params, walk_forward(name, {**(fixed or {}), **params}, train, test_seasons))
        for params in grid(space)
    )
    best = min(trials, key=lambda t: t.mean)
    threshold = best.mean + best.stderr
    chosen = next(t for t in trials if t.mean <= threshold)
    return TuneResult(chosen, trials)
