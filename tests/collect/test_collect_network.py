import pytest

from football.collect.espn import SCOREBOARD_URL, scoreboard_week
from football.collect.http import Fetcher
from football.collect.team_reports import parse_report, report_url

pytestmark = pytest.mark.network


def test_club_injury_report_parses() -> None:
    with Fetcher() as fetcher:
        fetched = fetcher.get(report_url("CHI"))
    assert fetched.ok
    page = parse_report(fetched.text)
    assert page.recognized
    assert page.week_label is not None


def test_espn_scoreboard_week() -> None:
    with Fetcher() as fetcher:
        week = scoreboard_week(fetcher.get(SCOREBOARD_URL).json())
    assert week.season >= 2026
    assert week.week >= 1
