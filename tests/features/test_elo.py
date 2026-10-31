import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from football.features.elo import EloParams, expected, mov_multiplier, regress, run_elo, update
from football.schemas import GAMES

START = datetime(2020, 9, 13, 17, tzinfo=UTC)


def games(rows: list[tuple[int, int, str, str, int | None, int | None, bool]]) -> pl.DataFrame:
    records = []
    for index, (season, week, home, away, hs, as_, neutral) in enumerate(rows):
        kickoff = START + timedelta(days=365 * (season - 2020) + 7 * (week - 1), minutes=index)
        records.append(
            {
                "game_id": f"{season}_{week:02d}_{away}_{home}",
                "season": season,
                "week": week,
                "game_type": "REG",
                "kickoff_utc": kickoff,
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "neutral_site": neutral,
                "home_rest": 7,
                "away_rest": 7,
                "available_at": kickoff + timedelta(hours=4),
            }
        )
    schema = {k: v for k, v in GAMES.columns.items() if k in records[0]}
    return pl.DataFrame(records, schema=schema)


def test_expected_probability() -> None:
    assert expected(0.0, 400.0) == pytest.approx(0.5)
    assert expected(400.0, 400.0) == pytest.approx(10 / 11)


def test_mov_multiplier_matches_538() -> None:
    assert mov_multiplier(7, 0.0) == pytest.approx(math.log(8))
    assert mov_multiplier(14, 100.0) == pytest.approx(math.log(15) * 2.2 / 2.3)
    assert mov_multiplier(0, 0.0) == pytest.approx(math.log(2))


def test_update_is_zero_sum() -> None:
    params = EloParams()
    home, away = update(1505.0, 1505.0, 48.0, 10.0, params)
    assert home + away == pytest.approx(3010.0)
    shift = 20 * math.log(11) * 2.2 / (48 * 0.001 + 2.2) * (1 - expected(48.0, 400.0))
    assert home - 1505.0 == pytest.approx(shift)


def test_regress_one_third_toward_mean() -> None:
    assert regress(1605.0, 1, EloParams()) == pytest.approx(1505 + 100 * 2 / 3)


def test_run_elo_uses_only_prior_results_and_regresses() -> None:
    df = games(
        [
            (2020, 1, "KC", "BUF", 27, 20, False),
            (2020, 2, "BUF", "KC", None, None, True),
            (2021, 1, "KC", "BUF", 10, 3, False),
        ]
    )
    out = run_elo(df).sort("game_id")
    first = out.filter(pl.col("game_id") == "2020_01_BUF_KC").row(0, named=True)
    assert first["home_elo"] == pytest.approx(1505.0)
    assert first["elo_diff"] == pytest.approx(48.0)
    params = EloParams()
    kc, buf = update(1505.0, 1505.0, 48.0, 7.0, params)
    second = out.filter(pl.col("game_id") == "2020_02_KC_BUF").row(0, named=True)
    assert second["home_elo"] == pytest.approx(buf)
    assert second["away_elo"] == pytest.approx(kc)
    assert second["elo_diff"] == pytest.approx(buf - kc)
    third = out.filter(pl.col("game_id") == "2021_01_BUF_KC").row(0, named=True)
    assert third["home_elo"] == pytest.approx(regress(kc, 1, params))
    assert third["elo_home_prob"] == pytest.approx(
        expected(regress(kc, 1, params) - regress(buf, 1, params) + 48.0, 400.0)
    )


def test_relocated_team_keeps_rating() -> None:
    df = games([(2020, 1, "SD", "KC", 30, 0, False), (2020, 2, "LAC", "KC", None, None, False)])
    out = run_elo(df).filter(pl.col("game_id") == "2020_02_KC_LAC").row(0, named=True)
    assert out["home_elo"] > 1505.0
