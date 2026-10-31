import json

import polars as pl
import pytest

from football.ingest import lines_espn, lines_sbro, lines_wintotals
from football.ingest.lines_common import RawCache


@pytest.mark.network
def test_live_nfldata_win_totals_parse() -> None:
    entry = RawCache("network_test").fetch("win_totals.csv", lines_wintotals.NFLDATA_URL)
    parsed = lines_wintotals.parse_nfldata(entry.read_bytes())
    assert parsed.filter(pl.col("season") == 2010).height == 32


@pytest.mark.network
def test_live_espn_odds_have_home_spread() -> None:
    entry = RawCache("network_test").fetch(
        "odds.json", lines_espn.ODDS_URL.format(event="401671789")
    )
    rows = lines_espn.parse_event(json.loads(entry.read_bytes()), started=True)
    assert any(r["kind"] == "close" and r["home_spread"] == 2.5 for r in rows)


@pytest.mark.network
def test_live_sbro_season_parses() -> None:
    entry = RawCache("network_test").fetch(
        "sbro.html", lines_sbro.URL.format(season=2015, next=16), browser=True
    )
    parsed, problems = lines_sbro.parse_season(entry.read_bytes(), 2015)
    assert parsed.height >= 260
    assert len(problems) <= 2
