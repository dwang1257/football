from collections.abc import Callable

import polars as pl
import pytest

from football.backtest.metrics import compare
from football.backtest.runner import ModelOutputError, run_backtest
from football.models.base import Model, Track

Factories = dict[str, Callable[[], Model]]


def test_backtest_predicts_every_labeled_game_once(
    features: pl.DataFrame, factories: Factories
) -> None:
    result = run_backtest(factories["strength"], features, [2020, 2021, 2022])
    expected = features.filter(
        pl.col("season").is_in([2020, 2021, 2022]) & pl.col("home_win").is_not_null()
    )
    assert result.predictions.height == expected.height
    assert set(result.predictions["game_id"]) == set(expected["game_id"])
    assert result.predictions.columns == [
        "game_id",
        "season",
        "week",
        "game_type",
        "home_win",
        "home_margin",
        "home_win_prob",
        "pred_margin",
    ]
    assert result.predictions["home_win_prob"].null_count() == 0
    assert result.model_name == "strength"
    assert result.overall()["n"] == expected.height
    assert {"overall", "weeks 1-4", "weeks 5+", "playoffs", "2020", "2022"} <= set(
        result.metrics_dict()
    )


def test_informative_model_beats_home_rate(features: pl.DataFrame, factories: Factories) -> None:
    seasons = list(range(2019, 2026))
    strong = run_backtest(factories["strength"], features, seasons)
    naive = run_backtest(factories["home_rate"], features, seasons, retrain="season")
    assert strong.overall()["log_loss"] < naive.overall()["log_loss"]
    comparison = compare(strong.predictions, naive.predictions, draws=2000, seed=3)
    assert comparison.bootstrap.log_loss.hi < 0
    assert comparison.adoptable


def test_season_mode_fits_once_per_season(features: pl.DataFrame) -> None:
    fits: list[int] = []

    class Counting:
        name = "counting"
        track: Track = "independent"

        def fit(self, train: pl.DataFrame) -> None:
            fits.append(train.height)

        def predict(self, games: pl.DataFrame) -> pl.DataFrame:
            return games.select("game_id", home_win_prob=pl.lit(0.5))

    result = run_backtest(Counting, features, [2021, 2022], retrain="season")
    assert len(fits) == 2
    assert result.predictions["pred_margin"].null_count() == result.predictions.height


def test_bad_model_output_is_rejected(features: pl.DataFrame) -> None:
    class Broken:
        name = "broken"
        track: Track = "independent"

        def fit(self, train: pl.DataFrame) -> None:
            return None

        def predict(self, games: pl.DataFrame) -> pl.DataFrame:
            return games.head(1).select("game_id", home_win_prob=pl.lit(1.5))

    with pytest.raises(ModelOutputError):
        run_backtest(Broken, features, [2021])
