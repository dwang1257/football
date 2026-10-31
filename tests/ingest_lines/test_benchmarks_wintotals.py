import json
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from football import storage
from football.ingest import benchmarks, lines_wintotals
from football.ingest.base import IngestOptions
from football.schemas import BENCHMARKS, TEAM_RATINGS, WIN_TOTALS

Serve = Callable[[dict[str, bytes]], list[str]]


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


NFELO_GAMES = b""",game_id,nfelo_home_probability_close,nfelo_home_line_close
0,2012_01_DAL_NYG,0.62,-3.0
1,2026_05_CLE_MIN,0.58,-2.0
"""
ELO = b"""date,season,neutral,playoff,team1,team2,elo1_pre,elo2_pre,elo_prob1,elo_prob2
2012-09-05,2012,0,,dal,nyg,1500,1550,0.4,0.6
2016-09-18,2016,0,,lar,sea,1450,1600,0.3,0.7
1999-09-12,1999,0,,kc,chi,1500,1500,0.5,0.5
"""


@pytest.mark.usefixtures("use_schedule")
def test_nfelo_and_elo538_benchmarks_orient_to_home(serve: Serve) -> None:
    serve({benchmarks.NFELO_URL: NFELO_GAMES, benchmarks.ELO_URL: ELO})
    options = IngestOptions(seasons=[1999, 2012, 2016, 2026])
    benchmarks.ingest_nfelo(options)
    benchmarks.ingest_elo538(options)
    table = storage.read("clean", BENCHMARKS.name)
    got = {
        (r["source"], r["game_id"]): (r["home_win_prob"], r["home_margin"], r["available_at"])
        for r in table.iter_rows(named=True)
    }
    assert got[("nfelo", "2012_01_DAL_NYG")] == (0.62, 3.0, utc(2012, 9, 6, 0, 29))
    neutral = got[("nfelo", "2026_05_MIN_CLE")]
    assert neutral[0] == pytest.approx(0.42)
    assert neutral[1] == -2.0
    nyg = got[("elo538", "2012_01_DAL_NYG")]
    assert nyg[0] == pytest.approx(0.6)
    assert nyg[1] == pytest.approx(16 * math.log10(0.6 / 0.4))
    rams = got[("elo538", "2016_02_SEA_LA")]
    assert rams[0] == pytest.approx(0.3)
    assert rams[1] == pytest.approx(16 * math.log10(0.3 / 0.7))
    assert len(got) == 4


def fpi_item(team_id: str, fpi: float, updated: str, run_key: int) -> dict[str, object]:
    return {
        "team": {"$ref": f"http://x/seasons/2016/teams/{team_id}?lang=en"},
        "lastUpdated": updated,
        "runDateTimeKey": run_key,
        "predictives": [
            {"name": "fpi", "value": fpi},
            {"name": "epaoffense", "value": fpi - 1},
            {"name": "epadefense", "value": 1.0},
            {"name": "epaspecialteams", "value": 0.0},
            {"name": "fpirank", "value": 3.0},
        ],
    }


def test_fpi_stamp_uses_later_of_last_updated_and_eastern_run_key() -> None:
    stale = fpi_item("14", 1.0, "2015-10-23T02:03Z", 20151117030000)
    assert benchmarks.stamp(stale) == utc(2015, 11, 17, 8, 0)
    current = fpi_item("14", 1.0, "2026-10-31T06:00Z", 20261031020009)
    assert benchmarks.stamp(current) == utc(2026, 10, 31, 6, 0, 9)
    assert benchmarks.stamp({"lastUpdated": "bad"}) is None


@pytest.mark.usefixtures("use_schedule")
def test_espn_fpi_ingest_maps_team_ids_and_stamps(serve: Serve) -> None:
    page = {
        "count": 2,
        "pageCount": 1,
        "items": [
            fpi_item("14", -2.5, "2016-09-20T10:00Z", 20160920060000),
            fpi_item("26", 6.0, "2016-09-20T10:00Z", 20160920060000),
        ],
    }
    url = benchmarks.FPI_URL.format(season=2016, type=2, week=2, page=1)
    requested = serve({url: json.dumps(page).encode()})
    benchmarks.ingest_fpi(IngestOptions(seasons=[2014, 2016]))
    assert requested == [url]
    table = storage.read("clean", TEAM_RATINGS.name)
    got = {r["team"]: r for r in table.iter_rows(named=True)}
    assert set(got) == {"LA", "SEA"}
    assert got["LA"]["rating"] == -2.5
    assert got["LA"]["offense"] == -3.5
    assert got["LA"]["week"] == 2
    assert got["LA"]["available_at"] == utc(2016, 9, 20, 10, 0)
    assert got["LA"]["source"] == "espn_fpi"


def test_fpi_weeks_maps_postseason_and_starts_in_2015_week_10() -> None:
    games = pl.DataFrame(
        {
            "season": [2015, 2015, 2015, 2015, 2021, 2021],
            "week": [9, 10, 17, 21, 18, 19],
            "game_type": ["REG", "REG", "REG", "SB", "REG", "WC"],
            "kickoff_utc": [
                utc(2015, 11, 8),
                utc(2015, 11, 15),
                utc(2016, 1, 3),
                utc(2016, 2, 7),
                utc(2022, 1, 9),
                utc(2022, 1, 15),
            ],
        }
    )
    weeks = [(s, t, w, nw) for s, t, w, nw, _ in benchmarks.fpi_weeks(games, [2015, 2021])]
    assert weeks == [
        (2015, 2, 10, 10),
        (2015, 2, 17, 17),
        (2015, 3, 5, 21),
        (2021, 2, 18, 18),
        (2021, 3, 1, 19),
    ]


SOH = b"""<html><table class='soh1'><thead><tr><th>Team</th><th>Win Total</th></tr></thead>
<tbody>
<tr><td><a>Las Vegas Raiders   </a></td><td>6.5</td><td> -130 </td><td> +110 </td>
<td>Week 18</td><td>8</td><td>Over</td></tr>
<tr><td><a>Washington Commanders</a></td><td>7</td><td>-110</td><td>-110</td>
<td>Week 9</td><td>12</td><td>Over</td></tr>
</tbody></table></html>"""
NFLDATA_TOTALS = b"""season,team,line,over_odds,under_odds
2016,OAK,7.5,-120,100
2016,LA,6.5,-110,-110
"""


@pytest.mark.usefixtures("use_schedule")
def test_win_totals_sources_and_availability(serve: Serve) -> None:
    serve(
        {
            lines_wintotals.NFLDATA_URL: NFLDATA_TOTALS,
            lines_wintotals.SOH_URL.format(season=2026): SOH,
        }
    )
    lines_wintotals.ingest(IngestOptions(seasons=[2016, 2026]))
    table = storage.read("clean", WIN_TOTALS.name)
    got = {(r["season"], r["team"], r["source"]): r for r in table.iter_rows(named=True)}
    assert set(got) == {
        (2016, "OAK", "nfldata"),
        (2016, "LA", "nfldata"),
        (2026, "LV", "sportsoddshistory"),
        (2026, "WAS", "sportsoddshistory"),
    }
    lv = got[(2026, "LV", "sportsoddshistory")]
    assert lv["win_total"] == 6.5
    over, under = 130 / 230, 100 / 210
    assert lv["over_prob"] == pytest.approx(over / (over + under))
    assert lv["available_at"] == utc(2026, 10, 4, 13, 30) - timedelta(days=7)
    assert got[(2016, "LA", "nfldata")]["over_prob"] == pytest.approx(0.5)
    assert got[(2016, "OAK", "nfldata")]["available_at"] == utc(2016, 9, 18, 20, 5) - timedelta(
        days=7
    )
