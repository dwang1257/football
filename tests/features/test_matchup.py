from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from football.features.common import decay_factor, weighted_moments
from football.features.matchup import MatchupParams, matchup_features
from football.schemas import TEAM_GAMES

START = datetime(2020, 9, 13, 17, tzinfo=UTC)


def test_weighted_moments_match_numpy() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=12)
    w = rng.random(12)
    mean, var = weighted_moments(
        np.array([w.sum()]),
        np.array([(w * w).sum()]),
        np.array([(w * x).sum()]),
        np.array([(w * x * x).sum()]),
    )
    m = np.average(x, weights=w)
    pop = np.average((x - m) ** 2, weights=w)
    unbiased = pop / (1 - (w * w).sum() / w.sum() ** 2)
    assert mean[0] == pytest.approx(m)
    assert var[0] == pytest.approx(unbiased)


def test_single_observation_has_no_variance() -> None:
    _, var = weighted_moments(np.array([1.0]), np.array([1.0]), np.array([3.0]), np.array([9.0]))
    assert np.isnan(var[0])


def row(
    season: int, week: int, team: str, opp: str, points: int, takeaways: int
) -> dict[str, object]:
    kickoff = START + timedelta(days=365 * (season - 2020) + 7 * (week - 1))
    home, away = sorted((team, opp))
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "week": week,
        "kickoff_utc": kickoff,
        "team": team,
        "opponent": opp,
        "is_home": team == home,
        "points_for": points,
        "points_against": 0,
        "plays": 60,
        "pass_plays": 30,
        "rush_plays": 30,
        "pass_yards": 200.0,
        "rush_yards": 100.0,
        "total_yards": 300.0,
        "pass_epa": 0.0,
        "rush_epa": 0.0,
        "total_epa": 6.0,
        "success_plays": 27,
        "wp_filtered_plays": 0,
        "wp_filtered_epa": 0.0,
        "takeaways": takeaways,
        "giveaways": 1,
        "fumbles_lost": 0,
        "interceptions_thrown": 1,
        "drives": 11,
        "red_zone_trips": 0,
        "red_zone_tds": 0,
        "avg_start_yardline": 25.0,
        "available_at": kickoff + timedelta(hours=6),
    }


def fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    points = [10, 20, 40]
    rows = []
    for week, p in enumerate(points, start=1):
        rows.append(row(2020, week, "KC", "BUF", p, week))
        rows.append(row(2020, week, "BUF", "KC", 0, 0))
    for week in (1, 2, 3, 4):
        rows.append(row(2021, week, "KC", "BUF", 50, 0))
        rows.append(row(2021, week, "BUF", "KC", 0, 0))
    team_games = pl.DataFrame(rows, schema=TEAM_GAMES.columns)
    kickoffs = team_games.select("season", "week", "kickoff_utc").unique().sort("kickoff_utc")
    targets = kickoffs.select(
        pl.format("{}_{}", "season", "week").alias("game_id"),
        "season",
        pl.col("kickoff_utc").alias("cutoff"),
        pl.lit("KC").alias("home_fr"),
        pl.lit("BUF").alias("away_fr"),
    )
    return team_games, targets


def test_decayed_mean_and_sd_within_season() -> None:
    team_games, targets = fixture()
    params = MatchupParams(prior_games=1e-9)
    out = matchup_features(targets, team_games, params)
    week4 = out.filter(pl.col("game_id") == "2021_1").row(0, named=True)
    d = decay_factor(6.0)
    w = np.array([d * d, d, 1.0])
    x = np.array([10.0, 20.0, 40.0])
    m = (w * x).sum() / w.sum()
    pop = (w * (x - m) ** 2).sum() / w.sum()
    var = pop * w.sum() ** 2 / (w.sum() ** 2 - (w * w).sum())
    week3 = out.filter(pl.col("game_id") == "2020_3").row(0, named=True)
    assert week3["home_off_points_per_game"] == pytest.approx((d * 10 + 20) / (d + 1))
    assert week3["home_games_played"] == 2
    assert week3["home_def_turnovers_per_game"] == pytest.approx((d * 1 + 2) / (d + 1))
    assert week3["away_off_turnovers_per_game"] == pytest.approx(1.0)
    assert week4["home_games_played"] == 0
    lg = week4["lg_points_per_game"]
    assert week4["home_off_points_per_game"] == pytest.approx(lg + (2 / 3) * (m - lg))
    assert week4["home_off_points_per_game_sd"] == pytest.approx(np.sqrt(var))


def test_prior_weight_decays_over_games_played() -> None:
    team_games, targets = fixture()
    out = matchup_features(targets, team_games, MatchupParams())
    fourth = out.filter(pl.col("game_id") == "2021_4").row(0, named=True)
    d = decay_factor(6.0)
    last = (d * d * 10 + d * 20 + 40) / (d * d + d + 1)
    lg = fourth["lg_points_per_game"]
    prior = lg + (2 / 3) * (last - lg)
    assert fourth["home_games_played"] == 3
    assert fourth["home_off_points_per_game"] == pytest.approx(0.5 * prior + 0.5 * 50.0)


def test_first_ever_game_has_no_ratings() -> None:
    team_games, targets = fixture()
    out = matchup_features(targets, team_games).filter(pl.col("game_id") == "2020_1")
    assert out["home_off_points_per_game"].to_list() == [None]
    assert out["home_games_played"].to_list() == [0.0]


def test_opponent_adjustment_runs_and_shifts_values() -> None:
    team_games, targets = fixture()
    plain = matchup_features(targets, team_games, MatchupParams())
    adjusted = matchup_features(targets, team_games, MatchupParams(opponent_passes=2))
    assert plain.columns == adjusted.columns
    late = pl.col("game_id") == "2021_4"
    assert plain.filter(late)["home_off_points_per_game"].item() != pytest.approx(
        adjusted.filter(late)["home_off_points_per_game"].item()
    )
