from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from football.features.columns import BENCHMARK_COLUMNS
from football.schemas import FEATURES, MATCHUP_STATS

SEASONS = tuple(range(2015, 2021))
N_TEAMS = 16
N_WEEKS = 16
HOME_FIELD = 2.0
SIGMA = 13.0

STAT_SHAPE: dict[str, tuple[float, float, float]] = {
    "plays_per_game": (63.0, 0.4, 4.0),
    "pass_rate": (0.58, 0.004, 0.06),
    "pass_yards_per_play": (6.3, 0.12, 1.4),
    "rush_yards_per_play": (4.3, 0.05, 1.0),
    "epa_per_play": (0.0, 0.02, 0.18),
    "success_rate": (0.45, 0.006, 0.06),
    "turnovers_per_game": (1.4, -0.06, 1.1),
    "points_per_game": (22.0, 1.0, 9.0),
    "red_zone_td_rate": (0.56, 0.01, 0.2),
    "start_yardline": (28.0, 0.3, 4.0),
}


def _team_values(
    rng: np.random.Generator, strength: np.ndarray, week: int
) -> dict[str, np.ndarray]:
    noise = 1.0 / np.sqrt(week)
    values: dict[str, np.ndarray] = {}
    for stat in MATCHUP_STATS:
        lg, slope, sd = STAT_SHAPE[stat]
        off = lg + slope * strength[:, 0] + rng.normal(0, abs(slope) * 2 * noise, N_TEAMS)
        allowed = lg - slope * strength[:, 1] + rng.normal(0, abs(slope) * 2 * noise, N_TEAMS)
        values[f"off_{stat}"] = off
        values[f"def_{stat}"] = allowed
        values[f"off_{stat}_sd"] = sd * rng.uniform(0.8, 1.2, N_TEAMS)
        values[f"def_{stat}_sd"] = sd * rng.uniform(0.8, 1.2, N_TEAMS)
    return values


def _season_rows(
    rng: np.random.Generator, season: int, strength: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    wins = np.zeros(N_TEAMS)
    played = np.zeros(N_TEAMS)
    start = datetime(season, 9, 10, 17, tzinfo=UTC)
    qb = rng.normal(0, 1.5, N_TEAMS)
    win_total = 8.5 + 0.4 * strength.sum(axis=1) + rng.normal(0, 0.7, N_TEAMS)
    for week in range(1, N_WEEKS + 1):
        team_values = _team_values(rng, strength, week)
        order = rng.permutation(N_TEAMS)
        kickoff = start + timedelta(days=7 * (week - 1))
        for g in range(N_TEAMS // 2):
            home, away = int(order[2 * g]), int(order[2 * g + 1])
            rows.append(
                _game(
                    rng,
                    season,
                    week,
                    kickoff,
                    home,
                    away,
                    strength,
                    team_values,
                    wins,
                    played,
                    qb,
                    win_total,
                )
            )
            result = rows[-1]["home_margin"]
            assert isinstance(result, float)
            wins[home] += float(result > 0) + 0.5 * float(result == 0)
            wins[away] += float(result < 0) + 0.5 * float(result == 0)
            played[[home, away]] += 1
    return rows


def _game(
    rng: np.random.Generator,
    season: int,
    week: int,
    kickoff: datetime,
    home: int,
    away: int,
    strength: np.ndarray,
    values: dict[str, np.ndarray],
    wins: np.ndarray,
    played: np.ndarray,
    qb: np.ndarray,
    win_total: np.ndarray,
) -> dict[str, object]:
    neutral = bool(rng.random() < 0.03)
    rest_diff = float(rng.choice([-3, 0, 0, 0, 3, 7]))
    qb_diff = float(qb[home] - qb[away] + rng.normal(0, 0.5))
    edge = float(strength[home].sum() - strength[away].sum())
    mu = edge + HOME_FIELD * (not neutral) + 0.15 * rest_diff + 0.5 * qb_diff
    margin = float(np.round(mu + rng.normal(0, SIGMA)))
    elo_diff = 25.0 * edge + 55.0 * (not neutral) + rng.normal(0, 40)
    market_prob = float(norm.cdf((mu + rng.normal(0, 1.0)) / SIGMA))
    row: dict[str, object] = {
        "game_id": f"{season}_{week:02d}_T{away:02d}_T{home:02d}",
        "season": season,
        "week": week,
        "game_type": "REG",
        "kickoff_utc": kickoff,
        "cutoff": kickoff - timedelta(hours=1),
        "home_team": f"T{home:02d}",
        "away_team": f"T{away:02d}",
        "home_win": 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5,
        "home_margin": margin,
        "home_elo": 1500 + elo_diff / 2,
        "away_elo": 1500 - elo_diff / 2,
        "elo_diff": elo_diff,
        "elo_home_prob": 1 / (1 + 10 ** (-elo_diff / 400)),
        "rest_diff": rest_diff,
        "neutral_site": neutral,
        "div_game": bool(rng.random() < 0.3),
        "travel_km_home": 0.0 if not neutral else float(rng.uniform(0, 3000)),
        "travel_km_away": float(rng.uniform(0, 4000)),
        "home_qb_value": float(qb[home]),
        "away_qb_value": float(qb[away]),
        "qb_value_diff": qb_diff,
        "fcst_wind_mph": float(rng.gamma(2.0, 4.0)),
        "home_season_win_pct": wins[home] / played[home] if played[home] else None,
        "away_season_win_pct": wins[away] / played[away] if played[away] else None,
        "home_win_total": float(win_total[home]),
        "away_win_total": float(win_total[away]),
        "win_total_diff": float(win_total[home] - win_total[away]),
        "market_home_prob": market_prob,
        "market_home_spread": float(np.round(-mu * 2) / 2),
        "market_total": 44.0,
        "market_spread_move": 0.0,
        **{name: float(rng.uniform(0.2, 0.8)) for name in BENCHMARK_COLUMNS},
    }
    for stat in MATCHUP_STATS:
        early = week == 1 and rng.random() < 0.5
        for side, team in (("home", home), ("away", away)):
            for unit in ("off", "def"):
                for suffix in ("", "_sd"):
                    value = values[f"{unit}_{stat}{suffix}"][team]
                    row[f"{side}_{unit}_{stat}{suffix}"] = None if early else float(value)
        row[f"lg_{stat}"] = STAT_SHAPE[stat][0]
        row[f"lg_{stat}_sd"] = STAT_SHAPE[stat][2]
    row["home_games_played"] = float(played[home])
    row["away_games_played"] = float(played[away])
    return row


def make_features(seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 2.5, (N_TEAMS, 2))
    rows: list[dict[str, object]] = []
    for season in SEASONS:
        strength = 0.6 * strength + rng.normal(0, 2.0, (N_TEAMS, 2))
        rows.extend(_season_rows(rng, season, strength))
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return frame.with_columns(pl.col(name).cast(dtype) for name, dtype in FEATURES.columns.items())


@pytest.fixture(scope="session")
def features() -> pl.DataFrame:
    return make_features()


@pytest.fixture(scope="session")
def train(features: pl.DataFrame) -> pl.DataFrame:
    return features.filter(pl.col("season") < 2020)


@pytest.fixture(scope="session")
def test_games(features: pl.DataFrame) -> pl.DataFrame:
    return features.filter(pl.col("season") == 2020)
