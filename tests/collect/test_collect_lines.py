import polars as pl
from collect_fixtures import an_game, an_history, an_outcome, snapshot, ts

from football.collect.action_network import history_lines
from football.collect.archive import Snapshot
from football.collect.espn import fpi_ratings, odds_lines, scoreboard_events, scoreboard_week
from football.collect.games import GameResolver, nflverse_week
from football.schemas import LINES, TEAM_RATINGS, UTC_TS, validate

KICKOFF = "2026-11-01T18:00:00+00:00"
META = {
    "event_id": 290949,
    "season": 2026,
    "week": 8,
    "season_type": "reg",
    "start_time": KICKOFF,
    "status": "scheduled",
    "home": "BUF",
    "away": "BAL",
}


def _history() -> dict[str, object]:
    return an_history(
        15,
        {
            "spread": [
                an_outcome(
                    "home",
                    [
                        ("2026-10-20T17:00:00.100Z", -2.5),
                        ("2026-10-25T12:00:00.000Z", -3.0),
                        ("2026-11-01T18:30:00.000Z", 7.0),
                    ],
                ),
                an_outcome("away", [("2026-10-20T17:00:00.100Z", 2.5)]),
            ],
            "total": [
                an_outcome("over", [("2026-10-21T09:00:00Z", 52.5)]),
                an_outcome("under", [("2026-10-21T09:00:00Z", 52.5)]),
            ],
            "moneyline": [
                an_outcome("home", [("2026-10-20T17:00:00.200Z", -140)], key="odds"),
                an_outcome("away", [("2026-10-20T17:00:00.900Z", 120)], key="odds"),
            ],
        },
    )


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return validate(pl.DataFrame(rows, schema=LINES.columns), LINES).sort("captured_at")


def test_history_ticks_build_point_in_time_states() -> None:
    first = snapshot(
        "action_network", "history_290949", _history(), "2026-10-26T13:00:00+00:00", META
    )
    again = snapshot(
        "action_network", "history_290949", _history(), "2026-10-27T13:00:00+00:00", META
    )
    frame = _frame(history_lines([first, again], GameResolver()))
    assert set(frame["game_id"]) == {"2026_08_BAL_BUF"}
    assert set(frame["book"]) == {"consensus"}
    ticks = frame.filter(pl.col("kind") == "snapshot")
    assert ticks["captured_at"].to_list() == [
        ts("2026-10-20T17:00:01+00:00"),
        ts("2026-10-21T09:00:00+00:00"),
        ts("2026-10-25T12:00:00+00:00"),
    ]
    assert ticks["home_spread"].to_list() == [2.5, 2.5, 3.0]
    assert ticks["total"].to_list() == [None, 52.5, 52.5]
    assert ticks["home_ml"].to_list() == [-140.0] * 3
    assert ticks["away_ml"].to_list() == [120.0] * 3
    assert (frame["captured_at"] < ts(KICKOFF)).all()
    assert (frame["available_at"] == frame["captured_at"]).all()
    opened = frame.filter(pl.col("kind") == "open").row(0, named=True)
    assert opened["captured_at"] == ts("2026-10-21T09:00:00+00:00")
    assert (opened["home_spread"], opened["total"], opened["home_ml"]) == (2.5, 52.5, -140.0)


def test_scoreboard_snapshot_and_sign_convention() -> None:
    body = {
        "games": [
            an_game(
                290949,
                "2026-11-01T18:00:00.000Z",
                (125, "BUF"),
                (130, "BAL"),
                spread_home=-3.0,
                total=54.5,
                ml_home=-170,
                ml_away=142,
            ),
            an_game(
                290950, "2026-10-30T00:15:00.000Z", (147, "GB"), (150, "CAR"), spread_home=-3.5
            ),
            an_game(
                290951, "2026-11-01T18:00:00.000Z", (134, "IND"), (148, "JAC"), spread_home=0.0
            ),
        ]
    }
    board = snapshot("action_network", "scoreboard_2026_w08", body, "2026-10-31T13:00:00+00:00")
    frame = _frame(history_lines([board], GameResolver()))
    assert frame["game_id"].to_list() == ["2026_08_BAL_BUF", "2026_08_JAX_IND"]
    row = frame.row(0, named=True)
    assert (row["home_spread"], row["total"], row["home_ml"], row["away_ml"]) == (
        3.0,
        54.5,
        -170.0,
        142.0,
    )
    assert row["captured_at"] == board.fetched_at
    assert str(frame["home_spread"][1]) == "0.0"


def _espn_odds(fetched_at: str) -> Snapshot:
    meta = {
        "season": 2026,
        "season_type": 2,
        "espn_week": 8,
        "week": 8,
        "event_id": "401873026",
        "home": "BUF",
        "away": "BAL",
        "kickoff": KICKOFF,
    }
    body = {
        "items": [
            {
                "provider": {"name": "DraftKings"},
                "spread": -3.0,
                "overUnder": 54.5,
                "homeTeamOdds": {"moneyLine": -170},
                "awayTeamOdds": {"moneyLine": 142},
            }
        ]
    }
    return snapshot("espn_odds", "odds_401873026", body, fetched_at, meta)


def test_espn_odds_lines() -> None:
    before = _espn_odds("2026-10-31T13:00:00+00:00")
    after = _espn_odds("2026-11-01T18:05:00+00:00")
    frame = _frame(odds_lines([before, after], GameResolver()))
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert (row["source"], row["book"], row["kind"]) == ("espn", "draftkings", "snapshot")
    assert (row["home_spread"], row["total"], row["home_ml"]) == (3.0, 54.5, -170.0)


def test_resolver_prefers_games_table() -> None:
    resolver = GameResolver()
    resolver.add_games(
        pl.DataFrame(
            {
                "game_id": ["2026_21_BUF_KC"],
                "season": [2026],
                "home_team": ["KC"],
                "away_team": ["BUF"],
                "kickoff_utc": [ts("2027-01-24T23:30:00+00:00")],
            },
            schema_overrides={"kickoff_utc": UTC_TS},
        )
    )
    kickoff = ts("2027-01-24T23:30:00+00:00")
    assert resolver.resolve(2026, None, "KC", "BUF", kickoff) == "2026_21_BUF_KC"
    assert resolver.resolve(2026, None, "KC", "DEN", kickoff) is None
    assert resolver.resolve(2026, 3, "LAR", "WSH", None) == "2026_03_WAS_LA"
    assert resolver.resolve(2026, 3, "XXX", "WSH", None) is None
    assert nflverse_week(3, 5) == 22
    assert nflverse_week("post", 4) is None
    assert nflverse_week("reg", 7) == 7
    assert nflverse_week(1, 2) is None


def test_espn_scoreboard_parsing() -> None:
    body = {
        "season": {"year": 2026, "type": 2},
        "week": {"number": 8},
        "events": [
            {
                "id": "401873026",
                "date": "2026-11-01T18:00Z",
                "competitions": [
                    {
                        "status": {"type": {"state": "pre"}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "BUF"}},
                            {"homeAway": "away", "team": {"abbreviation": "BAL"}},
                        ],
                    }
                ],
            }
        ],
    }
    week = scoreboard_week(body)
    assert (week.season, week.week, week.nflverse_week) == (2026, 8, 8)
    (event,) = scoreboard_events(body)
    assert (event.home, event.away, event.state) == ("BUF", "BAL", "pre")
    assert event.kickoff == ts(KICKOFF)


def test_fpi_ratings_keep_version_time_and_fetch_time() -> None:
    body = {
        "items": [
            {
                "team": {
                    "$ref": "http://sports.core.api.espn.com/v2/.../seasons/2026/teams/14?lang=en"
                },
                "season": 2026,
                "lastUpdated": "2026-10-31T06:00Z",
                "predictives": [
                    {"name": "fpi", "value": 2.5},
                    {"name": "epaoffense", "value": 1.0},
                    {"name": "epadefense", "value": 1.2},
                    {"name": "epaspecialteams", "value": 0.3},
                ],
            },
            {"team": {"$ref": "http://x/teams/99"}, "season": 2026, "predictives": []},
        ]
    }
    meta = {"season": 2026, "season_type": 2, "espn_week": 8, "week": 8}
    rows = fpi_ratings(
        [
            snapshot("espn_fpi", "powerindex", body, "2026-10-31T13:00:00+00:00", meta),
            snapshot("espn_fpi", "powerindex", body, "2026-10-31T19:00:00+00:00", meta),
        ]
    )
    frame = pl.DataFrame(rows, schema=TEAM_RATINGS.columns).sort("available_at")
    frame = frame.unique(subset=list(TEAM_RATINGS.keys), keep="first", maintain_order=True)
    validate(frame, TEAM_RATINGS)
    row = frame.row(0, named=True)
    assert frame.height == 1
    assert (row["team"], row["rating"], row["offense"], row["defense"]) == ("LA", 2.5, 1.0, 1.2)
    assert row["captured_at"] == ts("2026-10-31T06:00:00+00:00")
    assert row["available_at"] == ts("2026-10-31T13:00:00+00:00")
