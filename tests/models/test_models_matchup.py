import numpy as np
import polars as pl
import pytest

from football.models.matchup import MatchupModel, combined_sd, expected_stat
from football.models.registry import build


def test_plan_example_league_relative() -> None:
    assert expected_stat(250.0, 200.0, 225.0, "league_relative") == pytest.approx(222.22, abs=0.01)


def test_plan_example_average() -> None:
    assert expected_stat(250.0, 200.0, 225.0, "average") == pytest.approx(225.0)


def test_signed_stats_combine_additively() -> None:
    assert expected_stat(0.1, -0.05, 0.0, "league_relative", multiplicative=False) == (
        pytest.approx(0.05)
    )


def test_expected_stat_matches_on_expressions() -> None:
    frame = pl.DataFrame({"off": [250.0], "allowed": [200.0], "lg": [225.0]})
    expr = expected_stat(pl.col("off"), pl.col("allowed"), pl.col("lg"), "league_relative")
    assert frame.select(expr).item() == pytest.approx(250.0 * 200.0 / 225.0)


def test_combined_sd_forms() -> None:
    one = np.array([1.0])
    average = combined_sd(250 * one, 200 * one, 225 * one, 30 * one, 40 * one, "average")
    assert average[0] == pytest.approx(np.sqrt((30**2 + 40**2) / 2))
    at_league = combined_sd(225 * one, 225 * one, 225 * one, 30 * one, 40 * one, "league_relative")
    assert at_league[0] == pytest.approx(average[0])
    relative = combined_sd(250 * one, 200 * one, 225 * one, 30 * one, 40 * one, "league_relative")
    expected = np.sqrt(((30 * 200 / 225) ** 2 + (40 * 250 / 225) ** 2) / 2)
    assert relative[0] == pytest.approx(expected)


@pytest.mark.parametrize("combine", ["league_relative", "average"])
def test_simulate_agrees_with_analytic(
    combine: str, train: pl.DataFrame, test_games: pl.DataFrame
) -> None:
    simulate = build("matchup", combine=combine, mode="simulate")
    analytic = build("matchup", combine=combine, mode="analytic")
    simulate.fit(train)
    analytic.fit(train)
    sim = simulate.predict(test_games)
    ana = analytic.predict(test_games)
    gap = np.abs(sim["home_win_prob"].to_numpy() - ana["home_win_prob"].to_numpy())
    assert gap.max() < 0.02
    assert sim["home_margin"].to_list() == ana["home_margin"].to_list()


def test_simulation_is_independent_of_batch(train: pl.DataFrame, test_games: pl.DataFrame) -> None:
    model = MatchupModel(n_sims=2000)
    model.fit(train)
    full = model.predict(test_games)
    tail = model.predict(test_games.tail(5))
    assert full.tail(5)["home_win_prob"].to_list() == tail["home_win_prob"].to_list()


def test_points_model_learns_home_field_and_sigma(train: pl.DataFrame) -> None:
    model = MatchupModel(mode="analytic")
    model.fit(train)
    assert 11.0 < model.sigma < 16.0
    index = model.points.context.index("home_field")
    home_points = model.points.coef[len(model.points.stats) + index]
    assert 0.5 < home_points / model.points.context_scale[index] < 4.0


def test_rejects_unknown_choices() -> None:
    with pytest.raises(ValueError, match="combine"):
        MatchupModel(combine="product")
    with pytest.raises(ValueError, match="mode"):
        MatchupModel(mode="exact")
