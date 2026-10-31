import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from football.features.columns import BENCHMARK_COLUMNS
from football.models.common import MARKET_COLUMNS
from football.models.registry import MODEL_NAMES, build

FAST: dict[str, dict[str, object]] = {
    "matchup": {"n_sims": 2000},
    "ensemble": {"components": "elo,logreg,margin,lgbm"},
}
INDEPENDENT = [name for name in MODEL_NAMES if name != "market"]
MARKET_TRACK = ["market", "logreg", "margin", "lgbm", "ensemble"]


def _fit_predict(
    name: str,
    train: pl.DataFrame,
    games: pl.DataFrame,
    track: str = "independent",
    **params: object,
) -> pl.DataFrame:
    model = build(name, track=track, **{**FAST.get(name, {}), **params})
    model.fit(train)
    return model.predict(games)


def _scramble(df: pl.DataFrame, columns: tuple[str, ...], seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    present = [c for c in columns if c in df.columns]
    return df.with_columns(
        pl.Series(c, rng.uniform(0.01, 0.99, df.height) * 20 - 5, dtype=pl.Float64) for c in present
    )


def _log_loss(pred: pl.DataFrame, games: pl.DataFrame) -> float:
    joined = pred.join(games.select("game_id", "home_win"), on="game_id")
    decided = joined.filter(pl.col("home_win") != 0.5)
    p = decided["home_win_prob"].to_numpy()
    y = decided["home_win"].to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@pytest.mark.parametrize(
    ("name", "track"),
    [*((n, "independent") for n in INDEPENDENT), *((n, "market") for n in MARKET_TRACK)],
)
def test_fit_predict_contract(
    name: str, track: str, train: pl.DataFrame, test_games: pl.DataFrame
) -> None:
    first = _fit_predict(name, train, test_games, track)
    second = _fit_predict(name, train, test_games, track)
    assert first.columns == ["game_id", "home_win_prob", "home_margin"]
    assert first.schema["home_win_prob"] == pl.Float64
    assert first.schema["home_margin"] == pl.Float64
    assert first["game_id"].to_list() == test_games["game_id"].to_list()
    probs = first["home_win_prob"].to_numpy()
    assert np.all((probs > 0) & (probs < 1))
    assert_frame_equal(first, second)


@pytest.mark.parametrize("name", ["elo", "matchup", "logreg", "margin", "lgbm", "ensemble"])
def test_models_learn_signal(name: str, train: pl.DataFrame, test_games: pl.DataFrame) -> None:
    home = _log_loss(_fit_predict("home", train, test_games), test_games)
    assert _log_loss(_fit_predict(name, train, test_games), test_games) < home - 0.01


@pytest.mark.parametrize("name", INDEPENDENT)
def test_independent_track_ignores_market_columns(
    name: str, train: pl.DataFrame, test_games: pl.DataFrame
) -> None:
    base = _fit_predict(name, train, test_games)
    scrambled = _fit_predict(
        name, _scramble(train, MARKET_COLUMNS, 1), _scramble(test_games, MARKET_COLUMNS, 2)
    )
    removed = _fit_predict(name, train.drop(MARKET_COLUMNS), test_games.drop(MARKET_COLUMNS))
    assert_frame_equal(base, scrambled)
    assert_frame_equal(base, removed)


@pytest.mark.parametrize(
    ("name", "track"),
    [*((n, "independent") for n in INDEPENDENT), *((n, "market") for n in MARKET_TRACK)],
)
def test_benchmark_columns_never_read(
    name: str, track: str, train: pl.DataFrame, test_games: pl.DataFrame
) -> None:
    base = _fit_predict(name, train, test_games, track)
    scrambled = _fit_predict(
        name,
        _scramble(train, BENCHMARK_COLUMNS, 3),
        _scramble(test_games, BENCHMARK_COLUMNS, 4),
        track,
    )
    assert_frame_equal(base, scrambled)


@pytest.mark.parametrize("name", INDEPENDENT)
def test_tolerates_missing_optional_columns(
    name: str, train: pl.DataFrame, test_games: pl.DataFrame
) -> None:
    optional = ["qb_value_diff", "home_qb_value", "away_qb_value", "fcst_wind_mph", "rest_diff"]
    optional += [c for c in train.columns if "rush_yards_per_play" in c]
    pred = _fit_predict(name, train.drop(optional), test_games.drop(optional))
    probs = pred["home_win_prob"].to_numpy()
    assert np.all((probs > 0) & (probs < 1))


@pytest.mark.parametrize("name", INDEPENDENT)
def test_unplayed_and_null_feature_rows(
    name: str, train: pl.DataFrame, test_games: pl.DataFrame
) -> None:
    unplayed = test_games.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("home_win"),
        pl.lit(None, dtype=pl.Float64).alias("home_margin"),
        pl.lit(None, dtype=pl.Float64).alias("elo_diff"),
        pl.lit(None, dtype=pl.Float64).alias("home_off_points_per_game"),
    )
    pred = _fit_predict(name, pl.concat([train, unplayed]), unplayed)
    assert pred["home_win_prob"].is_finite().all()
    assert_frame_equal(pred, _fit_predict(name, train, unplayed))


def test_market_model_requires_market_track() -> None:
    with pytest.raises(ValueError, match="track=market"):
        build("market", track="independent")
    with pytest.raises(ValueError, match="requires track=market"):
        build("logreg", features="elo,market")
