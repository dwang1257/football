from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from lines_fixtures import schedule_frame

from football import storage
from football.ingest import lines_common
from football.ingest.lines_common import (
    RawCache,
    close_available,
    finalize,
    match_by_date,
    match_by_ids,
    no_vig_prob,
    parse_american,
    parse_number,
    split_open_conflicts,
    tuesday_noon_before,
    wednesday_evening_before,
    write_lines,
)
from football.schemas import LINES, UTC_TS


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def apply(expr_fn: Callable[[pl.Expr], pl.Expr], stamps: list[datetime]) -> list[datetime]:
    frame = pl.DataFrame({"k": stamps}, schema={"k": UTC_TS})
    return frame.select(expr_fn(pl.col("k")).alias("out"))["out"].to_list()


def test_kickoff_converts_eastern_to_utc_and_defaults_missing_times() -> None:
    games = schedule_frame(
        [
            {
                "game_id": "2012_01_DAL_NYG",
                "season": 2012,
                "week": 1,
                "gameday": "2012-09-05",
                "weekday": "Wednesday",
                "gametime": "20:30",
                "away_team": "DAL",
                "home_team": "NYG",
            },
            {
                "game_id": "1999_01_KC_CHI",
                "season": 1999,
                "week": 1,
                "gameday": "1999-09-12",
                "weekday": "Sunday",
                "gametime": None,
                "away_team": "KC",
                "home_team": "CHI",
            },
            {
                "game_id": "1999_01_MIA_DEN",
                "season": 1999,
                "week": 1,
                "gameday": "1999-09-13",
                "weekday": "Monday",
                "gametime": None,
                "away_team": "MIA",
                "home_team": "DEN",
            },
        ]
    )
    assert games["kickoff_utc"].to_list() == [
        utc(2012, 9, 6, 0, 30),
        utc(1999, 9, 12, 17, 0),
        utc(1999, 9, 14, 1, 0),
    ]
    assert games["home_fr"].to_list() == ["NYG", "CHI", "DEN"]


def test_tuesday_noon_estimate_is_strictly_before_kickoff() -> None:
    wednesday_night = utc(2012, 9, 6, 0, 30)
    tuesday_night = utc(2010, 12, 29, 1, 0)
    tuesday_morning = utc(2010, 12, 28, 15, 0)
    winter_sunday = utc(2020, 12, 6, 18, 0)
    assert apply(
        tuesday_noon_before, [wednesday_night, tuesday_night, tuesday_morning, winter_sunday]
    ) == [
        utc(2012, 9, 4, 16, 0),
        utc(2010, 12, 28, 17, 0),
        utc(2010, 12, 21, 17, 0),
        utc(2020, 12, 1, 17, 0),
    ]


def test_wednesday_evening_estimate() -> None:
    thursday = utc(2016, 9, 9, 0, 30)
    wednesday = utc(2012, 9, 6, 0, 30)
    sunday = utc(2016, 9, 18, 20, 5)
    assert apply(wednesday_evening_before, [thursday, wednesday, sunday]) == [
        utc(2016, 9, 8, 1, 0),
        utc(2012, 8, 30, 1, 0),
        utc(2016, 9, 15, 1, 0),
    ]


def test_close_available_is_one_minute_before_kickoff() -> None:
    assert apply(close_available, [utc(2020, 1, 1, 18)]) == [utc(2020, 1, 1, 17, 59)]


@pytest.mark.parametrize(
    ("text", "number", "american"),
    [
        ("pk", 0.0, None),
        ("PK", 0.0, None),
        ("+3", 3.0, None),
        ("-2.5", -2.5, None),
        ("-110", -110.0, -110.0),
        ("+136", 136.0, 136.0),
        ("EVEN", 0.0, 100.0),
        ("NL", None, None),
        ("", None, None),
        (None, None, None),
        (7, 7.0, None),
    ],
)
def test_parse_number_and_american(
    text: object, number: float | None, american: float | None
) -> None:
    assert parse_number(text) == number
    assert parse_american(text) == american


def test_no_vig_prob() -> None:
    assert no_vig_prob(-110, -110) == pytest.approx(0.5)
    assert no_vig_prob(-200, 170) == pytest.approx((2 / 3) / (2 / 3 + 100 / 270))
    assert no_vig_prob(None, 100) is None


def test_match_by_ids_maps_relocations_and_swaps() -> None:
    games = schedule_frame()
    rows = pl.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "season": [2026, 2016, 2012, 2012],
            "week": [9, 2, 1, 1],
            "away": ["OAK", "LAR", "DAL", "XXX"],
            "home": ["NYJ", "SEA", "NYG", "NYG"],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    result = match_by_ids(rows, games)
    got = {r["row_id"]: (r["game_id"], r["swapped"]) for r in result.matched.iter_rows(named=True)}
    assert got == {
        "a": ("2026_09_LV_NYJ", False),
        "b": ("2016_02_SEA_LA", True),
        "c": ("2012_01_DAL_NYG", False),
    }
    assert result.unmatched["row_id"].to_list() == ["d"]
    assert result.matched["kickoff_utc"].null_count() == 0


def test_match_by_date_handles_neutral_ambiguous_city_and_tolerance() -> None:
    games = schedule_frame()
    rows = pl.DataFrame(
        {
            "row_id": ["neutral", "city", "far", "home_first"],
            "season": [2026, 2016, 2012, 2012],
            "game_date": [
                datetime(2026, 10, 4).date(),
                datetime(2016, 9, 18).date(),
                datetime(2012, 9, 9).date(),
                datetime(2012, 9, 6).date(),
            ],
            "team_a": ["Minnesota Vikings", "Seattle", "Dallas", "NYGiants"],
            "team_b": ["Cleveland Browns", "LosAngeles", "NYGiants", "Dallas"],
        },
        schema_overrides={"season": pl.Int32},
    )
    result = match_by_date(rows, games)
    got = {r["row_id"]: (r["game_id"], r["swapped"]) for r in result.matched.iter_rows(named=True)}
    assert got == {
        "neutral": ("2026_05_MIN_CLE", True),
        "city": ("2016_02_SEA_LA", True),
        "home_first": ("2012_01_DAL_NYG", False),
    }
    assert result.unmatched["row_id"].to_list() == ["far"]


def test_raw_cache_reuses_until_refresh_or_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_get(url: str, **_: object) -> bytes:
        calls.append(url)
        return f"payload{len(calls)}".encode()

    monkeypatch.setattr(lines_common, "http_get", fake_get)
    cache = RawCache("unit_test")
    first = cache.fetch("a.csv", "https://example.test/a.csv")
    again = RawCache("unit_test").fetch("a.csv", "https://example.test/a.csv")
    assert again.path == first.path
    assert again.read_bytes() == b"payload1"
    assert len(calls) == 1
    RawCache("unit_test").fetch("a.csv", "https://example.test/a.csv", refresh=True)
    assert len(calls) == 2
    stale = RawCache("unit_test")
    old = stale.versions("a.csv")[0]
    stale.index["a.csv"] = [lines_common.Cached(old.path, old.fetched_at - timedelta(days=1))]
    stale.fetch("a.csv", "https://example.test/a.csv", max_age=timedelta(hours=6))
    assert len(calls) == 3


def line_rows(source: str, rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    stamp = utc(2020, 1, 1)
    return finalize(
        pl.DataFrame(
            {
                "game_id": [r[0] for r in rows],
                "book": ["consensus"] * len(rows),
                "kind": [r[1] for r in rows],
                "captured_at": [stamp] * len(rows),
                "available_at": [stamp] * len(rows),
                "home_spread": [r[2] for r in rows],
                "total": [None] * len(rows),
                "home_ml": [None] * len(rows),
                "away_ml": [None] * len(rows),
            },
            schema_overrides={"total": pl.Float64, "home_ml": pl.Float64, "away_ml": pl.Float64},
        ),
        source,
    )


def test_write_lines_replaces_source_seasons_but_keeps_snapshots_and_others() -> None:
    write_lines(
        line_rows(
            "lines_x",
            [
                ("2019_01_A_B", "close", 3.0),
                ("2020_01_A_B", "close", 1.0),
                ("2020_02_A_B", "snapshot", 2.0),
            ],
        ),
        "lines_x",
        [2019, 2020],
    )
    write_lines(line_rows("lines_y", [("2020_01_A_B", "close", 1.5)]), "lines_y", [2020])
    write_lines(line_rows("lines_x", [("2020_03_A_B", "close", 7.0)]), "lines_x", [2020])
    table = storage.read("clean", LINES.name)
    got = sorted((r["source"], r["game_id"], r["kind"]) for r in table.iter_rows(named=True))
    assert got == [
        ("lines_x", "2019_01_A_B", "close"),
        ("lines_x", "2020_02_A_B", "snapshot"),
        ("lines_x", "2020_03_A_B", "close"),
        ("lines_y", "2020_01_A_B", "close"),
    ]


def test_split_open_conflicts_flags_opposite_favorites() -> None:
    lines = line_rows(
        "lines_x",
        [
            ("g1", "close", 10.0),
            ("g1", "open", -10.0),
            ("g2", "close", 2.0),
            ("g2", "open", -2.5),
            ("g3", "open", -7.0),
        ],
    )
    kept, conflicts = split_open_conflicts(lines)
    assert conflicts["game_id"].to_list() == ["g1"]
    assert kept.height == 4
