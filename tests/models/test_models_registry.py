import numpy as np
import polars as pl
import pytest

from football.models.baselines import HomeModel, RecordModel
from football.models.ensemble import EnsembleModel
from football.models.linear import LogRegModel, cross_fit_matchup_mu
from football.models.matchup import DEFAULT_ALPHA, PointsModel
from football.models.registry import MODEL_NAMES, build, parse_param


def test_model_names() -> None:
    assert MODEL_NAMES == (
        "home",
        "record",
        "elo",
        "market",
        "matchup",
        "logreg",
        "margin",
        "lgbm",
        "ensemble",
    )


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("true", True),
        ("False", False),
        ("3", 3),
        ("0.25", 0.25),
        ("1e-2", 0.01),
        ("platt", "platt"),
    ],
)
def test_parse_param(text: str, value: object) -> None:
    parsed = parse_param(text)
    assert parsed == value
    assert type(parsed) is type(value)


def test_build_from_cli_strings() -> None:
    model = build("logreg", c="0.5", use_matchup="true", track="independent")
    assert isinstance(model, LogRegModel)
    assert model.c == 0.5
    assert model.use_matchup is True
    ensemble = build("ensemble", components="elo,margin", calibration="isotonic")
    assert isinstance(ensemble, EnsembleModel)
    assert [m.name for m in ensemble.models] == ["elo", "margin"]


def test_build_rejects_unknown_names_and_params() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build("oracle")
    with pytest.raises(ValueError, match="no parameters"):
        build("elo", depth="3")
    with pytest.raises(ValueError, match="track"):
        build("elo", track="vegas")


def test_home_model_drops_ties(train: pl.DataFrame) -> None:
    ties = train.with_columns(pl.lit(0.5).alias("home_win"))
    model = HomeModel()
    model.fit(pl.concat([train.head(10).with_columns(pl.lit(1.0).alias("home_win")), ties]))
    assert model.rate == pytest.approx(11 / 12)


def test_record_model_prefers_better_record(train: pl.DataFrame) -> None:
    model = RecordModel()
    model.fit(train)
    assert model.home_pick_rate > 0.5 > model.away_pick_rate


def test_matchup_mu_is_cross_fitted_by_season(train: pl.DataFrame) -> None:
    crossed = cross_fit_matchup_mu(train, "league_relative")
    assert crossed is not None
    held = crossed.filter(pl.col("season") == 2017)
    others = PointsModel("league_relative", DEFAULT_ALPHA).fit(
        train.filter(pl.col("season") != 2017)
    )
    expected = others.mu(others.inputs(held))
    np.testing.assert_allclose(held["matchup_mu"].to_numpy(), expected)


def test_logreg_uses_matchup_mu(train: pl.DataFrame, test_games: pl.DataFrame) -> None:
    model = LogRegModel(use_matchup=True, features="qb")
    model.fit(train)
    assert model.coefficients["matchup_mu"] > 0
    single = LogRegModel(use_matchup=True)
    single.fit(train.filter(pl.col("season") == 2019))
    assert "matchup_mu" in single.dropped
    probs = single.predict(test_games)["home_win_prob"].to_numpy()
    assert np.all((probs > 0) & (probs < 1))


@pytest.mark.parametrize("calibration", ["none", "platt", "isotonic"])
def test_ensemble_calibrations(
    calibration: str, train: pl.DataFrame, test_games: pl.DataFrame
) -> None:
    model = build("ensemble", components="elo,margin", calibration=calibration)
    model.fit(train)
    probs = model.predict(test_games)["home_win_prob"].to_numpy()
    assert np.all((probs > 0) & (probs < 1))
