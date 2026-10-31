from datetime import UTC, datetime

import polars as pl
import pytest

from football import storage
from football.ingest.nflverse_common import replace_seasons
from football.ingest.nflverse_games import clean_schedules
from football.ingest.nflverse_pbp import build_qb_games, build_team_games
from football.schemas import GAMES, QB_GAMES, TEAM_GAMES, validate


def schedule_rows() -> pl.DataFrame:
    base = {
        "game_type": "REG",
        "weekday": "Sunday",
        "location": "Home",
        "overtime": 0,
        "old_game_id": "x",
        "espn": "1",
        "pfr": "p",
        "away_rest": 7,
        "home_rest": 7,
        "away_moneyline": 120,
        "home_moneyline": -140,
        "spread_line": 2.5,
        "away_spread_odds": -110,
        "home_spread_odds": -110,
        "total_line": 44.5,
        "under_odds": -110,
        "over_odds": -110,
        "div_game": 0,
        "roof": "outdoors",
        "surface": "grass",
        "temp": 60,
        "wind": 5,
        "away_qb_id": "QB_A",
        "home_qb_id": "QB_H",
        "away_qb_name": "A",
        "home_qb_name": "H",
        "away_coach": "c1",
        "home_coach": "c2",
        "referee": "r",
        "stadium_id": "S1",
        "stadium": "Stadium",
    }
    rows = [
        {
            **base,
            "game_id": "2024_01_AAA_HHH",
            "season": 2024,
            "week": 1,
            "gameday": "2024-09-08",
            "gametime": "13:00",
            "away_team": "AAA",
            "home_team": "HHH",
            "away_score": 10,
            "home_score": 17,
            "div_game": 1,
        },
        {
            **base,
            "game_id": "2024_10_AAA_HHH",
            "season": 2024,
            "week": 10,
            "gameday": "2024-11-10",
            "gametime": "09:30",
            "away_team": "AAA",
            "home_team": "HHH",
            "away_score": None,
            "home_score": None,
            "location": "Neutral",
            "roof": "",
        },
        {
            **base,
            "game_id": "1999_01_AAA_HHH",
            "season": 1999,
            "week": 1,
            "gameday": "1999-09-12",
            "gametime": None,
            "away_team": "AAA",
            "home_team": "HHH",
            "away_score": 3,
            "home_score": 0,
        },
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("season", "week", "away_score", "home_score").cast(pl.Int32)
    )


def test_clean_schedules_converts_eastern_kickoffs() -> None:
    games = clean_schedules(schedule_rows())
    rows = {row["game_id"]: row for row in games.iter_rows(named=True)}
    week1 = rows["2024_01_AAA_HHH"]
    assert week1["kickoff_utc"] == datetime(2024, 9, 8, 17, 0, tzinfo=UTC)
    assert week1["available_at"] == datetime(2024, 9, 8, 21, 0, tzinfo=UTC)
    assert week1["div_game"] is True
    assert week1["neutral_site"] is False
    london = rows["2024_10_AAA_HHH"]
    assert london["kickoff_utc"] == datetime(2024, 11, 10, 14, 30, tzinfo=UTC)
    assert london["neutral_site"] is True
    assert london["roof"] is None
    assert london["home_score"] is None
    old = rows["1999_01_AAA_HHH"]
    assert old["kickoff_utc"] == datetime(1999, 9, 12, 17, 0, tzinfo=UTC)


def play(**values: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "game_id": "2024_01_AAA_HHH",
        "home_team": "HHH",
        "away_team": "AAA",
        "posteam": "HHH",
        "pass": 0.0,
        "rush": 0.0,
        "sack": 0.0,
        "qb_scramble": 0.0,
        "qb_dropback": 0.0,
        "qb_kneel": 0.0,
        "two_point_attempt": 0.0,
        "epa": 0.0,
        "qb_epa": None,
        "cpoe": None,
        "success": 0.0,
        "wp": 0.5,
        "yards_gained": 0.0,
        "yardline_100": 75.0,
        "interception": 0.0,
        "fumble_lost": 0.0,
        "fumbled_1_team": None,
        "fixed_drive": 1.0,
        "fixed_drive_result": "Punt",
        "passer_id": None,
        "passer": None,
        "total_home_score": 0.0,
        "total_away_score": 0.0,
    }
    return {**defaults, **values}


def synthetic_pbp() -> pl.DataFrame:
    td = "Touchdown"
    rows = [
        play(play_id=1, rush=1.0, yards_gained=5.0, epa=0.2, success=1.0, yardline_100=75.0),
        play(
            play_id=2,
            **{"pass": 1.0},
            qb_dropback=1.0,
            yards_gained=60.0,
            epa=3.0,
            qb_epa=3.0,
            cpoe=10.0,
            success=1.0,
            passer_id="QB_H",
            passer="H.Qb",
            yardline_100=70.0,
            fixed_drive_result="Punt",
        ),
        play(
            play_id=3,
            **{"pass": 1.0},
            qb_dropback=1.0,
            sack=1.0,
            yards_gained=-7.0,
            epa=-1.5,
            qb_epa=-1.5,
            passer_id="QB_H",
            passer="H.Qb",
            yardline_100=10.0,
            wp=0.95,
        ),
        play(
            play_id=10,
            fixed_drive=2.0,
            fixed_drive_result=td,
            **{"pass": 1.0},
            qb_dropback=1.0,
            qb_scramble=1.0,
            yards_gained=12.0,
            epa=1.0,
            success=1.0,
            passer_id="QB_B",
            passer="B.Qb",
            yardline_100=18.0,
            total_home_score=6.0,
        ),
        play(
            play_id=11,
            fixed_drive=2.0,
            fixed_drive_result=td,
            two_point_attempt=1.0,
            **{"pass": 1.0},
            qb_dropback=1.0,
            yards_gained=2.0,
            epa=1.0,
            passer_id="QB_B",
            passer="B.Qb",
            yardline_100=2.0,
            total_home_score=8.0,
        ),
        play(
            play_id=12,
            fixed_drive=3.0,
            posteam="AAA",
            **{"pass": 1.0},
            qb_dropback=1.0,
            yards_gained=0.0,
            epa=-4.0,
            qb_epa=-4.0,
            interception=1.0,
            passer_id="QB_A",
            passer="A.Qb",
            yardline_100=40.0,
            fixed_drive_result="Turnover",
            total_home_score=8.0,
            total_away_score=0.0,
        ),
        play(
            play_id=13,
            fixed_drive=4.0,
            posteam="AAA",
            epa=None,
            fumble_lost=1.0,
            fumbled_1_team="HHH",
            total_home_score=8.0,
            total_away_score=3.0,
        ),
        play(play_id=14, fixed_drive=5.0, qb_kneel=1.0, yards_gained=-1.0, epa=-0.1),
    ]
    return pl.DataFrame(rows).with_columns(pl.col("play_id").cast(pl.Float64))


def synthetic_games() -> pl.DataFrame:
    return clean_schedules(
        schedule_rows().filter(pl.col("game_id") == "2024_01_AAA_HHH")
    ).with_columns(
        pl.lit(8, dtype=pl.Int32).alias("home_score"), pl.lit(3, dtype=pl.Int32).alias("away_score")
    )


def test_build_team_games_definitions() -> None:
    table = build_team_games(synthetic_pbp(), synthetic_games())
    rows = {row["team"]: row for row in table.iter_rows(named=True)}
    home, away = rows["HHH"], rows["AAA"]
    assert home["plays"] == 4
    assert home["pass_plays"] == 3
    assert home["rush_plays"] == 1
    assert home["pass_yards"] == pytest.approx(65.0)
    assert home["rush_yards"] == pytest.approx(5.0)
    assert home["scramble_yards"] == pytest.approx(12.0)
    assert home["sack_yards"] == pytest.approx(-7.0)
    assert home["kneel_yards"] == pytest.approx(-1.0)
    assert home["total_epa"] == pytest.approx(2.7)
    assert home["success_plays"] == 3
    assert home["wp_filtered_plays"] == 3
    assert home["drives"] == 2
    assert home["red_zone_trips"] == 2
    assert home["red_zone_tds"] == 1
    assert home["avg_start_yardline"] == pytest.approx((25 + 82) / 2)
    assert home["fumbles_lost"] == 1
    assert home["interceptions_thrown"] == 0
    assert home["giveaways"] == 1
    assert home["takeaways"] == 1
    assert away["giveaways"] == 1
    assert away["takeaways"] == 1
    assert home["points_for"] == 8 and home["points_against"] == 3
    assert home["pbp_points"] == 8 and away["pbp_points"] == 3
    assert home["available_at"] == datetime(2024, 9, 9, 5, 0, tzinfo=UTC)
    validate(table, TEAM_GAMES)


def test_build_qb_games_marks_starter() -> None:
    table = build_qb_games(synthetic_pbp(), synthetic_games())
    rows = {row["player_id"]: row for row in table.iter_rows(named=True)}
    assert rows["QB_H"]["is_starter"] is True
    assert rows["QB_H"]["dropbacks"] == 2
    assert rows["QB_H"]["qb_epa"] == pytest.approx(1.5)
    assert rows["QB_H"]["cpoe"] == pytest.approx(10.0)
    assert rows["QB_B"]["is_starter"] is False
    assert rows["QB_B"]["dropbacks"] == 1
    assert rows["QB_A"]["team"] == "AAA"
    assert rows["QB_A"]["is_starter"] is True
    validate(table, QB_GAMES)


def test_replace_seasons_is_idempotent_and_drops_stale_rows() -> None:
    games = clean_schedules(schedule_rows())
    first = replace_seasons(games, GAMES, [1999, 2024])
    second = replace_seasons(games, GAMES, [1999, 2024])
    assert first.equals(second)
    replace_seasons(games.filter(pl.col("game_id") == "2024_01_AAA_HHH"), GAMES, [2024])
    stored = storage.read("clean", GAMES.name)
    assert sorted(stored["game_id"].to_list()) == ["1999_01_AAA_HHH", "2024_01_AAA_HHH"]
