from datetime import UTC, date, datetime

import polars as pl

from football.ingest.nflverse_checks import flag_drops
from football.ingest.nflverse_common import normalize_team
from football.ingest.nflverse_misc import clean_draft_picks, clean_trades
from football.ingest.nflverse_players import (
    clean_daily_depth,
    clean_injuries,
    final_report_time,
    game_week_capture,
)
from football.schemas import DEPTH_CHARTS, INJURIES, UTC_TS, validate


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def kickoffs(*values: datetime) -> pl.DataFrame:
    return pl.DataFrame({"kickoff_utc": list(values)}, schema={"kickoff_utc": UTC_TS})


def test_final_report_time_by_game_day() -> None:
    frame = kickoffs(
        utc(2024, 10, 13, 17, 0),
        utc(2024, 10, 15, 0, 15),
        utc(2024, 10, 11, 0, 15),
        utc(2024, 12, 21, 21, 30),
    ).select(final_report_time(pl.col("kickoff_utc")).alias("report"))
    assert frame["report"].to_list() == [
        utc(2024, 10, 11, 22, 0),
        utc(2024, 10, 12, 22, 0),
        utc(2024, 10, 9, 22, 0),
        utc(2024, 12, 19, 23, 0),
    ]


def test_game_week_capture_is_wednesday_noon_or_a_day_before_kickoff() -> None:
    frame = kickoffs(
        utc(2024, 10, 13, 17, 0),
        utc(2024, 10, 11, 0, 15),
        utc(2024, 12, 25, 18, 0),
    ).select(game_week_capture(pl.col("kickoff_utc")).alias("captured"))
    assert frame["captured"].to_list() == [
        utc(2024, 10, 9, 16, 0),
        utc(2024, 10, 9, 16, 0),
        utc(2024, 12, 24, 18, 0),
    ]


def team_games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2025_01_BUF_KC", "2025_02_KC_BUF"],
            "season": [2025, 2025],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "kickoff_utc": [utc(2025, 9, 7, 17, 0), utc(2025, 9, 15, 0, 15)],
            "home_team": ["KC", "BUF"],
            "away_team": ["BUF", "KC"],
            "home_qb_id": ["Q1", "Q2"],
            "away_qb_id": ["Q2", "Q1"],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32, "kickoff_utc": UTC_TS},
    )


def test_clean_injuries_maps_games_and_respects_late_modifications() -> None:
    raw = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "game_type": ["REG", "REG", "REG"],
            "team": ["KC", "KC", "KC"],
            "gsis_id": ["P1", "P1", "P1"],
            "full_name": ["A", "A", "A"],
            "position": ["QB", "QB", "QB"],
            "report_status": ["Out", "Questionable", None],
            "practice_status": ["DNP", "Limited", None],
            "report_primary_injury": [None, None, None],
            "report_secondary_injury": [None, None, None],
            "practice_primary_injury": [None, None, None],
            "date_modified": [None, utc(2025, 9, 14, 3, 0), None],
        },
        schema_overrides={"date_modified": UTC_TS},
    )
    table = validate(clean_injuries(raw, team_games()), INJURIES)
    assert table["week"].to_list() == [1, 2]
    assert table["available_at"].to_list() == [utc(2025, 9, 5, 22, 0), utc(2025, 9, 14, 3, 0)]
    assert (table["available_at"] < table["kickoff_utc"]).all()


def test_daily_depth_assigns_next_game_and_keeps_only_changes() -> None:
    def snapshot(stamp: str, qb: str) -> dict[str, object]:
        return {
            "dt": stamp,
            "team": "KC",
            "player_name": qb,
            "espn_id": "9",
            "gsis_id": qb,
            "pos_grp": "Offense",
            "pos_abb": "QB",
            "pos_rank": 1,
        }

    raw = pl.DataFrame(
        [
            snapshot("2025-09-03T06:00:00Z", "Q1"),
            snapshot("2025-09-04T06:00:00Z", "Q1"),
            snapshot("2025-09-05T06:00:00Z", "Q3"),
            snapshot("2025-09-08T06:00:00Z", "Q3"),
            snapshot("2025-09-20T06:00:00Z", "Q3"),
        ]
    )
    table = validate(clean_daily_depth(raw, team_games(), 2025), DEPTH_CHARTS)
    assert table.sort("captured_at").select("week", "player_id").rows() == [
        (1, "Q1"),
        (1, "Q3"),
        (2, "Q3"),
    ]
    assert (table["available_at"] == table["captured_at"]).all()


def test_normalize_team_handles_legacy_codes() -> None:
    frame = pl.DataFrame(
        {"team": ["SL", "HST", "OAK", "Packers", "PHO"], "season": [2002, 2002, 2021, 2010, 1992]}
    )
    assert normalize_team(frame, "team")["team"].to_list() == ["STL", "HOU", "LV", "GB", "ARI"]


def test_trades_and_draft_availability() -> None:
    trades = clean_trades(
        pl.DataFrame(
            {
                "trade_id": [5, 5],
                "season": [2024, 2024],
                "trade_date": [date(2024, 10, 29), date(2024, 10, 29)],
                "gave": ["KC", "TEN"],
                "received": ["TEN", "KC"],
                "pick_season": [2025, None],
                "pick_round": [5, None],
                "pick_number": [None, None],
                "conditional": [0, None],
                "pfr_id": [None, "HopkDe00"],
                "pfr_name": [None, "DeAndre Hopkins"],
            }
        )
    )
    assert trades["asset_index"].to_list() == [1, 2]
    assert trades["available_at"].to_list() == [utc(2024, 10, 30, 4, 0)] * 2
    picks = clean_draft_picks(
        pl.DataFrame(
            {
                "season": [2024],
                "round": [1],
                "pick": [1],
                "team": ["CHI"],
                "gsis_id": ["G"],
                "pfr_player_id": ["P"],
                "pfr_player_name": ["C. Williams"],
                "position": ["QB"],
                "college": ["USC"],
                "age": [22],
                "w_av": [10],
            }
        )
    )
    assert "w_av" not in picks.columns
    assert picks["available_at"].item() == utc(2024, 5, 15, 4, 0)


def test_flag_drops_ignores_current_season_and_growth() -> None:
    counts = pl.DataFrame(
        {
            "season": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "rows": [100, 110, 40, 120, 130, 140, 20],
        }
    )
    assert flag_drops(counts, current=2024) == [2020]
