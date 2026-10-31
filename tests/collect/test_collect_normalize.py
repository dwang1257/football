from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml
from collect_fixtures import an_game, club_table, report_page, snapshot, store

from football import storage
from football.collect import normalize
from football.ingest.base import IngestOptions
from football.ingest.commands import all_sources
from football.schemas import LINES, PRACTICE_REPORTS, TEAM_RATINGS, validate
from football.validate.commands import all_checks

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/collect.yml"


def _fill_archive() -> None:
    page = report_page(
        "REG-8",
        [
            club_table(
                "Buffalo Bills",
                ["Wed", "Thu", "Fri"],
                [("Josh Allen", "QB", "Hand", ["LP", "FP", "FP"], "")],
            )
        ],
    )
    board = {
        "games": [
            an_game(
                290949,
                "2026-11-01T18:00:00.000Z",
                (125, "BUF"),
                (130, "BAL"),
                spread_home=-3.0,
                total=54.5,
            )
        ]
    }
    fpi = {
        "items": [
            {
                "team": {"$ref": "http://x/teams/2"},
                "season": 2026,
                "lastUpdated": "2026-10-31T06:00Z",
                "predictives": [{"name": "fpi", "value": 4.3}],
            }
        ]
    }
    store(
        snapshot("team_injury_reports", "BUF", page, "2026-10-31T13:00:00+00:00", {"team": "BUF"}),
        snapshot("action_network", "scoreboard_2026_w08", board, "2026-10-31T13:00:00+00:00"),
        snapshot(
            "espn_fpi", "powerindex", fpi, "2026-10-31T13:00:00+00:00", {"season": 2026, "week": 8}
        ),
    )


def test_sources_and_checks_are_registered() -> None:
    assert {"archive_practice", "archive_lines", "archive_fpi", "archive_forecasts"} <= set(
        all_sources()
    )
    assert "archive_freshness" in all_checks()


def test_normalizers_write_clean_tables_and_keep_other_sources() -> None:
    other = pl.DataFrame(
        {
            "game_id": ["2026_08_BAL_BUF"],
            "source": ["lines_nflverse"],
            "book": ["consensus"],
            "kind": ["close"],
            "captured_at": [datetime(2026, 11, 1, 18, tzinfo=UTC)],
            "home_spread": [3.0],
            "total": [54.5],
            "home_ml": [None],
            "away_ml": [None],
            "available_at": [datetime(2026, 11, 1, 18, tzinfo=UTC)],
        },
        schema=LINES.columns,
    )
    storage.write(other, "clean", LINES.name)
    _fill_archive()
    options = IngestOptions(seasons=[2026])
    for source in normalize.SOURCES.values():
        source(options)
        source(options)
    practice = validate(storage.read("clean", PRACTICE_REPORTS.name), PRACTICE_REPORTS)
    assert practice.height == 3
    lines = validate(storage.read("clean", LINES.name), LINES)
    assert sorted(lines["source"].to_list()) == ["action_network", "lines_nflverse"]
    assert lines.filter(pl.col("source") == "action_network")["home_spread"].to_list() == [3.0]
    ratings = validate(storage.read("clean", TEAM_RATINGS.name), TEAM_RATINGS)
    assert ratings["team"].to_list() == ["BUF"]
    assert not storage.exists("clean", "weather_forecasts")


def test_practice_rebuild_keeps_other_seasons() -> None:
    old = pl.DataFrame(
        {
            "season": [2025],
            "week": [3],
            "team": ["BUF"],
            "player_name": ["Old"],
            "position": ["QB"],
            "injury": [None],
            "report_day": ["Wed"],
            "participation": ["FP"],
            "game_status": [None],
            "captured_at": [datetime(2025, 9, 17, tzinfo=UTC)],
            "available_at": [datetime(2025, 9, 17, tzinfo=UTC)],
        },
        schema=PRACTICE_REPORTS.columns,
    )
    storage.write(old, "clean", PRACTICE_REPORTS.name)
    _fill_archive()
    normalize.ingest_practice(IngestOptions(seasons=[2026]))
    table = storage.read("clean", PRACTICE_REPORTS.name)
    assert table.group_by("season").len().sort("season")["len"].to_list() == [1, 3]


def test_freshness_check() -> None:
    assert not normalize.check_freshness().passed
    for collector in normalize.COLLECTORS:
        for day in ("2026-10-27", "2026-10-29", "2026-10-30"):
            store(snapshot(collector, "x", {}, f"{day}T13:00:00+00:00"))
    fresh = normalize.check_freshness(datetime(2026, 10, 31, 12, tzinfo=UTC))
    assert fresh.passed, fresh.detail
    assert "latest 2026-10-30, 3 days, 1 missing" in fresh.detail
    stale = normalize.check_freshness(datetime(2026, 11, 3, tzinfo=UTC))
    assert not stale.passed
    assert "stale" in stale.detail


def test_workflow_is_valid() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    triggers = workflow[True]
    assert "workflow_dispatch" in triggers
    crons = [entry["cron"] for entry in triggers["schedule"]]
    assert "17 13 * * *" in crons
    assert all(len(cron.split()) == 5 for cron in crons)
    assert workflow["permissions"] == {"contents": "write"}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    steps = workflow["jobs"]["collect"]["steps"]
    uses = [step["uses"] for step in steps if "uses" in step]
    assert all("@v" in item and item.count(".") == 2 for item in uses)
    script = "\n".join(step.get("run", "") for step in steps)
    for fragment in (
        "uv sync --frozen",
        "uv run football collect",
        "--orphan",
        "[ ! -e ",
        "git -C",
    ):
        assert fragment in script
