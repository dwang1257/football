import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from football import storage
from football.ingest import lines_espn, lines_sbro
from football.ingest.base import IngestOptions
from football.ingest.lines_common import REJECTS, Cached, Schedule
from football.ingest.lines_sbro import SideLine, split_column
from football.schemas import LINES

Serve = Callable[[dict[str, bytes]], list[str]]


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (45.5, 4.0, SideLine(-4.0, 45.5, True)),
        (3.5, 46.5, SideLine(3.5, 46.5, True)),
        (0.0, 41.0, SideLine(0.0, 41.0, True)),
        (47.0, 24.0, SideLine(-24.0, 47.0, True)),
        (52.5, 26.5, SideLine(-26.5, 52.5, True)),
        (None, None, SideLine(None, None, True)),
        (44.0, None, SideLine(None, 44.0, True)),
        (None, 7.0, SideLine(-7.0, None, True)),
        (0.0, 0.0, SideLine(0.0, None, True)),
        (4.0, 11.0, SideLine(None, None, False)),
        (13.5, 0.0, SideLine(None, None, False)),
    ],
)
def test_split_column_finds_favorite_and_total(
    a: float | None, b: float | None, expected: SideLine
) -> None:
    assert split_column(a, b) == expected


def sbro_page(rows: list[list[str]]) -> bytes:
    header = "<tr>" + "".join(f"<td>{h}</td>" for h in lines_sbro.HEADER) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<html><body><table>{header}{body}</table></body></html>".encode()


SBRO_2012 = sbro_page(
    [
        ["905", "101", "V", "Dallas", "7", "7", "0", "10", "24", "45.5", "46.5", "155", "21"],
        ["905", "102", "H", "NYGiants", "3", "7", "0", "7", "17", "4", "3.5", "-175", "3"],
        ["101", "201", "V", "Miami", "0", "0", "0", "0", "0", "10", "48", "450", "21"],
        ["101", "202", "H", "NewEngland", "7", "14", "0", "7", "28", "49.5", "11", "-610", "6"],
    ]
)
SBRO_2016 = sbro_page(
    [
        ["918", "301", "V", "Seattle", "3", "0", "0", "0", "3", "6", "6.5", "-265", "3"],
        ["918", "302", "H", "LosAngeles", "3", "3", "3", "0", "9", "38.5", "38", "225", "19"],
    ]
)


def test_parse_season_pairs_rows_and_dates() -> None:
    parsed, problems = lines_sbro.parse_season(SBRO_2012, 2012)
    assert problems == []
    assert parsed["game_date"].to_list() == [date(2012, 9, 5), date(2013, 1, 1)]
    first = parsed.row(0, named=True)
    assert (first["open_spread_a"], first["open_total"]) == (-4.0, 45.5)
    assert (first["close_spread_a"], first["close_total"]) == (-3.5, 46.5)
    assert (first["ml_a"], first["ml_b"]) == (155.0, -175.0)


@pytest.mark.usefixtures("use_schedule")
def test_sbro_ingest_orients_to_nflverse_home(serve: Serve) -> None:
    serve(
        {
            lines_sbro.URL.format(season=2012, next=13): SBRO_2012,
            lines_sbro.URL.format(season=2016, next=17): SBRO_2016,
        }
    )
    lines_sbro.ingest(IngestOptions(seasons=[2005, 2012, 2016]))
    table = storage.read("clean", LINES.name).filter(pl.col("source") == "lines_sbro")
    got = {
        (r["game_id"], r["kind"]): (r["home_spread"], r["total"], r["home_ml"], r["away_ml"])
        for r in table.iter_rows(named=True)
    }
    assert got == {
        ("2012_01_DAL_NYG", "close"): (3.5, 46.5, -175.0, 155.0),
        ("2012_01_DAL_NYG", "open"): (4.0, 45.5, None, None),
        ("2016_02_SEA_LA", "close"): (-6.5, 38.0, 225.0, -265.0),
        ("2016_02_SEA_LA", "open"): (-6.0, 38.5, None, None),
    }
    rejects = storage.read("clean", REJECTS.name).filter(pl.col("source") == "lines_sbro")
    assert rejects["reason"].to_list() == ["unmatched"]
    assert rejects["ref"].to_list() == ["2012:101:Miami@NewEngland"]


def odds_item(provider: tuple[str, str], **fields: object) -> dict[str, object]:
    return {"provider": {"id": provider[0], "name": provider[1]}, **fields}


def team_odds(team_id: str, ml: int | None = None, **phases: object) -> dict[str, object]:
    ref = f"http://x/seasons/2026/teams/{team_id}?lang=en"
    return {"team": {"$ref": ref}, "moneyLine": ml, **phases}


def phase(spread: str, ml: str) -> dict[str, object]:
    return {"pointSpread": {"american": spread}, "moneyLine": {"american": ml}}


EVENT = {
    "count": 6,
    "items": [
        odds_item(("1004", "consensus"), spread=6.5, homeTeamOdds=team_odds("5")),
        odds_item(
            ("45", "Caesars Sportsbook (New Jersey)"),
            spread=-3.5,
            overUnder=44.5,
            homeTeamOdds=team_odds("5", -180),
            awayTeamOdds=team_odds("16", 150),
        ),
        odds_item(("1001", "accuscore"), spread=10.0, homeTeamOdds=team_odds("5", 300)),
        odds_item(("59", "ESPN Bet - Live Odds"), spread=-10.5, homeTeamOdds=team_odds("5", -900)),
        odds_item(
            ("100", "DraftKings"),
            spread=-3.0,
            overUnder=44.0,
            homeTeamOdds=team_odds(
                "5", -170, open=phase("+2.5", "+114"), close=phase("-3.5", "-185")
            ),
            awayTeamOdds=team_odds(
                "16", 145, open=phase("-2.5", "-135"), close=phase("+3.5", "+155")
            ),
            open={"total": {"american": "43.5"}},
            close={"total": {"american": "44.5"}},
        ),
        odds_item(
            ("23", "Opening"),
            spread=-2.0,
            overUnder=43.0,
            homeTeamOdds=team_odds("5", -130),
            awayTeamOdds=team_odds("16", 110),
        ),
        odds_item(
            ("31", "Caesars"),
            spread=-3.5,
            initialSpread=-1.5,
            overUnder=21.5,
            homeTeamOdds=team_odds("5", -175),
            awayTeamOdds=team_odds("16", 150),
        ),
    ],
}
NEUTRAL_EVENT = {
    "items": [
        odds_item(
            ("40", "DraftKings"),
            spread=-3.5,
            overUnder=44.5,
            homeTeamOdds=team_odds("16", -180),
            awayTeamOdds=team_odds("5", 150),
        )
    ]
}


def test_parse_event_kinds_and_skips_predictors_and_live() -> None:
    rows = lines_espn.parse_event(EVENT, started=True)
    got = {(r["book"], r["kind"], r["phase"]): (r["home_spread"], r["total"]) for r in rows}
    assert got == {
        ("caesars_sportsbook_new_jersey", "close", "top"): (3.5, 44.5),
        ("draftkings", "open", "open"): (-2.5, 43.5),
        ("draftkings", "close", "close"): (3.5, 44.5),
        ("opening", "open", "opening_provider"): (2.0, 43.0),
        ("caesars", "open", "initial"): (1.5, None),
        ("caesars", "close", "top"): (3.5, 21.5),
    }
    upcoming = lines_espn.parse_event(EVENT, started=False)
    assert {(r["book"], r["kind"]) for r in upcoming if r["kind"] != "open"} == {
        ("caesars_sportsbook_new_jersey", "snapshot"),
        ("draftkings", "snapshot"),
        ("caesars", "snapshot"),
    }


@pytest.mark.usefixtures("use_schedule")
def test_espn_ingest_orients_filters_and_counts(serve: Serve, schedule: Schedule) -> None:
    neutral = schedule.games.filter(pl.col("game_id") == "2026_05_MIN_CLE").row(0, named=True)
    sea = schedule.games.filter(pl.col("game_id") == "2016_02_SEA_LA").row(0, named=True)
    requested = serve(
        {
            lines_espn.ODDS_URL.format(event=neutral["espn"]): json.dumps(NEUTRAL_EVENT).encode(),
            lines_espn.ODDS_URL.format(event=sea["espn"]): json.dumps(EVENT).encode(),
            "https://sports.core.api.espn.com": b'{"count": 0, "items": []}',
        }
    )
    lines_espn.ingest(IngestOptions(seasons=[2005, 2012, 2016, 2026]))
    assert not any("250911019" in url for url in requested)
    table = storage.read("clean", LINES.name).filter(pl.col("source") == "lines_espn")
    neutral_close = table.filter(pl.col("game_id") == "2026_05_MIN_CLE").row(0, named=True)
    assert neutral_close["home_spread"] == -3.5
    assert (neutral_close["home_ml"], neutral_close["away_ml"]) == (150.0, -180.0)
    assert neutral_close["available_at"] == utc(2026, 10, 4, 13, 29)
    sea_rows = table.filter(pl.col("game_id") == "2016_02_SEA_LA")
    assert ("caesars", "close") not in set(zip(sea_rows["book"], sea_rows["kind"], strict=True))
    rejects = storage.read("clean", REJECTS.name).filter(pl.col("source") == "lines_espn")
    reasons = dict(zip(rejects["ref"], rejects["reason"], strict=True))
    assert reasons["2016_02_SEA_LA:caesars:close:top"] == "implausible_total"
    assert reasons["2012_01_DAL_NYG"] == "no_market_odds"
    before = len(requested)
    lines_espn.ingest(IngestOptions(seasons=[2016]))
    assert len(requested) == before


def test_needs_fetch_policy() -> None:
    kickoff = utc(2026, 10, 4, 13, 30)
    now = utc(2026, 10, 31)
    final = Cached(Path("final"), kickoff + timedelta(days=1))
    early = Cached(Path("early"), kickoff - timedelta(days=1))
    fresh = Cached(Path("fresh"), now - timedelta(hours=1))
    assert lines_espn.needs_fetch(None, kickoff, now, refresh=False)
    assert not lines_espn.needs_fetch(final, kickoff, now, refresh=False)
    assert lines_espn.needs_fetch(final, kickoff, now, refresh=True)
    assert lines_espn.needs_fetch(early, kickoff, now, refresh=False)
    assert not lines_espn.needs_fetch(fresh, utc(2026, 11, 8), now, refresh=False)
