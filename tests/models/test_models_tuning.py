import numpy as np
import polars as pl
import pytest

from football.models.tuning import grid, inner_seasons, log_loss, tune


def test_grid_orders_combinations() -> None:
    assert grid({"c": [0.1, 1.0], "use_matchup": [False, True]}) == [
        {"c": 0.1, "use_matchup": False},
        {"c": 0.1, "use_matchup": True},
        {"c": 1.0, "use_matchup": False},
        {"c": 1.0, "use_matchup": True},
    ]


def test_log_loss_scores_ties_as_half() -> None:
    assert log_loss(np.array([0.5]), np.array([0.5])) == pytest.approx(np.log(2))


def test_inner_seasons_skip_short_history(train: pl.DataFrame) -> None:
    assert inner_seasons(train, count=2, min_train=3) == [2018, 2019]


def test_tune_prefers_simplest_within_one_stderr(train: pl.DataFrame) -> None:
    result = tune("logreg", {"c": [0.001, 0.1, 10.0]}, train, seasons=[2018, 2019])
    assert len(result.trials) == 3
    best = min(result.trials, key=lambda t: t.mean)
    assert result.best.mean <= best.mean + best.stderr
    assert result.best.params["c"] <= best.params["c"]
