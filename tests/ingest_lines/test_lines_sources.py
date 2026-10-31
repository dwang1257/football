from collections.abc import Callable
from datetime import UTC, datetime

import polars as pl
import pytest
from lines_fixtures import FETCHED_AT

from football import storage
from football.ingest import lines
from football.ingest.base import IngestOptions
from football.ingest.lines_common import REJECTS, Schedule, match_by_ids
from football.schemas import LINES, UTC_TS

Serve = Callable[[dict[str, bytes]], list[str]]

NFELO_HEADER = (
    ",game_id,season,week,home_team,away_team,home_spread_open,home_spread_open_price,"
    "away_spread_open_price,home_spread_open_source,home_spread_open_timestamp,home_spread_last,"
    "home_spread_last_price,away_spread_last_price,home_spread_last_source,"
    "home_spread_last_timestamp,home_spread_tickets_pct,home_spread_money_pct,"
    "home_spread_pcts_source,home_spread_pct_timestamp,home_ml_open,away_ml_open,ml_open_source,"
    "ml_open_timestamp,home_ml_last,away_ml_last,ml_last_source,ml_last_timestamp,total_line_open,"
    "under_price_open,over_price_open,total_line_open_source,total_line_open_timestamp,"
    "total_line_last,under_price_last,over_price_last,total_line_last_source,"
    "total_line_last_timestamp"
)
NFELO_ROWS = [
    "0,2005_01_ARI_NYG,2005,1,NYG,ARI,-3.0,,,legacy,,-3.0,,,legacy,,,,,,,,,,,,,,,,,,,42.0,,,legacy,",
    "1,2016_02_SEA_LAR,2016,2,LAR,SEA,5.0,,,legacy,,6.0,,,legacy,,,,,,,,,,,,,,,,,,,38.0,,,legacy,",
    "2,2026_05_CLE_MIN,2026,5,MIN,CLE,-2.5,-110,-110,draftkings,2026-09-28 04:00:00.123-07:00,"
    "-3.5,-110,-110,action-hq-consensus,2026-10-04 06:20:00.0-07:00,0.6,0.7,"
    "action-hq-consensus,2026-10-04 06:20:00.0-07:00,-140,120,draftkings,"
    "2026-09-28 04:00:00.123-07:00,-180,150,action-hq-consensus,2026-10-04 06:20:00.0-07:00,"
    "44.0,-110,-110,draftkings,2026-09-28 04:00:00.123-07:00,44.5,-110,-110,"
    "action-hq-consensus,2026-10-04 06:20:00.0-07:00",
    "3,2026_09_OAK_NYJ,2026,9,NYJ,OAK,-1.5,-110,-110,draftkings,2026-11-03 04:00:00.0-07:00,"
    "-2.0,-110,-110,action-hq-consensus,2026-11-07 09:00:00.0-07:00,,,,,"
    ",,,,,,,,,,,,,,,,,",
    "4,2026_09_XXX_NYJ,2026,9,NYJ,XXX,-1.5,,,legacy,,-2.0,,,legacy,,,,,,,,,,,,,,,,,,,,,,,",
]


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def lines_for(source: str) -> pl.DataFrame:
    return storage.read("clean", LINES.name).filter(pl.col("source") == source)


def row(df: pl.DataFrame, game_id: str, kind: str) -> dict[str, object]:
    rows = df.filter(pl.col("game_id") == game_id, pl.col("kind") == kind).to_dicts()
    assert len(rows) == 1, rows
    return rows[0]


def rejects_for(source: str) -> pl.DataFrame:
    return storage.read("clean", REJECTS.name).filter(pl.col("source") == source)


@pytest.mark.usefixtures("use_schedule")
def test_nflverse_close_and_snapshot_keep_nflverse_sign() -> None:
    lines.ingest_nflverse(IngestOptions(seasons=[2012, 2016, 2026]))
    table = lines_for("lines_nflverse")
    close = row(table, "2012_01_DAL_NYG", "close")
    assert close["home_spread"] == 3.5
    assert close["home_ml"] == -175.0
    assert close["available_at"] == utc(2012, 9, 6, 0, 29)
    assert row(table, "2016_02_SEA_LA", "close")["home_spread"] == -6.0
    snapshot = row(table, "2026_09_LV_NYJ", "snapshot")
    assert snapshot["captured_at"] == FETCHED_AT
    assert snapshot["available_at"] == FETCHED_AT
    assert snapshot["home_spread"] == 1.5
    lines.ingest_nflverse(IngestOptions(seasons=[2012, 2016, 2026]))
    assert lines_for("lines_nflverse").height == table.height


def nfelo_matched(schedule: Schedule) -> pl.DataFrame:
    raw = lines.read_csv("\n".join([NFELO_HEADER, *NFELO_ROWS]).encode())
    keyed = lines.split_game_id(raw.drop("", "season", "week"))
    return match_by_ids(keyed, schedule.games).matched


def test_nfelo_signs_swaps_timestamps_and_legacy_opens(schedule: Schedule) -> None:
    fetched = pl.lit(FETCHED_AT).cast(UTC_TS)
    table, legacy = lines.nfelo_lines(nfelo_matched(schedule), fetched)
    assert legacy["row_id"].to_list() == ["2005_01_ARI_NYG"]
    assert row(table, "2005_01_ARI_NYG", "close")["home_spread"] == 3.0
    assert table.filter(pl.col("game_id") == "2005_01_ARI_NYG", pl.col("kind") == "open").is_empty()
    assert row(table, "2016_02_SEA_LA", "close")["home_spread"] == -6.0
    legacy_open = row(table, "2016_02_SEA_LA", "open")
    assert legacy_open["home_spread"] == -5.0
    assert legacy_open["available_at"] == utc(2016, 9, 13, 16, 0)
    neutral = row(table, "2026_05_MIN_CLE", "close")
    assert neutral["home_spread"] == -3.5
    assert neutral["home_ml"] == 150.0
    assert neutral["away_ml"] == -180.0
    assert neutral["home_tickets_pct"] == pytest.approx(0.4)
    assert neutral["captured_at"] == utc(2026, 10, 4, 13, 20)
    assert neutral["available_at"] == utc(2026, 10, 4, 13, 29)
    neutral_open = row(table, "2026_05_MIN_CLE", "open")
    assert neutral_open["home_spread"] == -2.5
    assert neutral_open["available_at"] == datetime(2026, 9, 28, 11, 0, 0, 123000, tzinfo=UTC)
    upcoming = row(table, "2026_09_LV_NYJ", "snapshot")
    assert upcoming["home_spread"] == 2.0
    assert upcoming["captured_at"] == utc(2026, 11, 7, 16, 0)


@pytest.mark.usefixtures("use_schedule")
def test_nfelo_ingest_counts_unmatched_and_legacy(serve: Serve) -> None:
    serve({lines.NFELO_URL: "\n".join([NFELO_HEADER, *NFELO_ROWS]).encode()})
    lines.ingest_nfelo(IngestOptions(seasons=[2005, 2016, 2026]))
    rejects = rejects_for("lines_nfelo")
    assert sorted(rejects["reason"].to_list()) == ["open_equals_close_before_2007", "unmatched"]
    assert set(lines_for("lines_nfelo")["game_id"]) == {
        "2005_01_ARI_NYG",
        "2016_02_SEA_LA",
        "2026_05_MIN_CLE",
        "2026_09_LV_NYJ",
    }


CLOSING = b"""game_id,alt_game_id,type,side,line,odds,outcome
2012090500,2012_01_DAL_NYG,MONEYLINE,NYG,,-175,0
2012090500,2012_01_DAL_NYG,MONEYLINE,DAL,,155,1
2012090500,2012_01_DAL_NYG,SPREAD,DAL,3.5,,1
2012090500,2012_01_DAL_NYG,SPREAD,NYG,-3.5,,0
2012090500,2012_01_DAL_NYG,TOTAL,Under,46.5,,1
2012090500,2012_01_DAL_NYG,TOTAL,Over,46.5,,0
2016091800,2016_02_LA_SEA,SPREAD,LA,6,,0
2016091800,2016_02_LA_SEA,SPREAD,SEA,-6,,1
"""
SC_LINES = b"""season,week,away_team,home_team,game_id,side,line
2016,2,SEA,LA,2016091800,SEA,-6.5
2016,2,SEA,LA,2016091800,LA,6.5
"""


@pytest.mark.usefixtures("use_schedule")
def test_nfldata_closing_and_supercontest_signs(serve: Serve) -> None:
    serve(
        {
            lines.NFLDATA_URL.format(name="closing_lines.csv"): CLOSING,
            lines.NFLDATA_URL.format(name="sc_lines.csv"): SC_LINES,
        }
    )
    lines.ingest_nfldata(IngestOptions(seasons=[2012, 2016]))
    table = lines_for("lines_nfldata")
    close = row(table, "2012_01_DAL_NYG", "close")
    assert (close["home_spread"], close["total"], close["home_ml"], close["away_ml"]) == (
        3.5,
        46.5,
        -175.0,
        155.0,
    )
    assert row(table, "2016_02_SEA_LA", "close")["home_spread"] == -6.0
    contest = row(table, "2016_02_SEA_LA", "wednesday")
    assert contest["home_spread"] == -6.5
    assert contest["book"] == "supercontest"
    assert contest["available_at"] == utc(2016, 9, 15, 1, 0)


HF_2026 = b"""game_id,date_et,start_time_utc,season,season_type,stage,round,week,home_team,\
away_team,home_team_id,away_team_id,home_score,away_score,overtime,bookmakers,open_home_win_prob,\
close_home_win_prob,close_fair_home_odds,close_fair_away_odds,open_spread_home,close_spread_home,\
close_home_cover_prob,open_total,close_total,close_over_prob,moneyline_winner,favorite,\
favorite_won,ats_result,ats_margin,ats_result_open,total_result,total_margin,total_result_open
abc,2026-10-04,2026-10-04T13:30:00Z,2026,regular,Regular season,Week 5,5,Minnesota Vikings,\
Cleveland Browns,x,y,20,24,False,21,0.58,0.6,1.6,2.5,-3.0,-3.5,0.5,44.0,44.5,0.5,away,home,\
False,away,0,away,under,0,under
"""


@pytest.mark.usefixtures("use_schedule")
def test_hf_spread_and_probability_orientation(serve: Serve) -> None:
    serve({"https://huggingface.co/datasets/minmaxxxer/nfl-odds-2026": HF_2026})
    lines.ingest_hf(IngestOptions(seasons=[2026]))
    table = lines_for("lines_hf")
    close = row(table, "2026_05_MIN_CLE", "close")
    assert close["home_spread"] == -3.5
    assert close["home_prob_novig"] == pytest.approx(0.4)
    assert close["book"] == "median"
    opened = row(table, "2026_05_MIN_CLE", "open")
    assert opened["home_spread"] == -3.0
    assert opened["home_prob_novig"] == pytest.approx(0.42)
    assert opened["available_at"] == utc(2026, 9, 29, 16, 0)
    assert close["available_at"] == utc(2026, 10, 4, 13, 29)
