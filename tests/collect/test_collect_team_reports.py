from collect_fixtures import club_table, report_page, snapshot

from football.collect.team_reports import (
    CLUB_DOMAINS,
    NAME,
    parse_report,
    practice_reports,
    report_fragment,
)
from football.schemas import PRACTICE_REPORTS, validate
from football.teams import FRANCHISES

DAYS = ["Wed", "Thu", "Fri"]


def _titans(status: str = "QUESTIONABLE", fri: str = "FP") -> str:
    return club_table(
        "Tennessee Titans",
        DAYS,
        [
            ("Calvin Ridley", "WR", "Hip", ["LP", "LP", fri], status),
            ("Tony Pollard", "RB", "NIR-Rest", ["(-)", "DNP", "(-)"], "(-)"),
            ("Late Add", "CB", "Knee", ["(-)", "(-)", "(-)"], "OUT"),
        ],
    )


def _bengals() -> str:
    return club_table("Cincinnati Bengals", DAYS, [("Joe Burrow", "QB", "Toe", ["FP"] * 3, "")])


def test_every_franchise_has_a_domain() -> None:
    assert set(CLUB_DOMAINS) == set(FRANCHISES)


def test_parse_report() -> None:
    page = parse_report(report_page("REG-8", [_titans(), _bengals()]))
    assert page.week_label == "REG-8"
    assert [club.team for club in page.clubs] == ["TEN", "CIN"]
    titans = page.clubs[0]
    assert titans.days == ("Wed", "Thu", "Fri")
    ridley, pollard, late = titans.rows
    assert ridley["practice"] == {"Wed": "LP", "Thu": "LP", "Fri": "FP"}
    assert ridley["game_status"] == "Questionable"
    assert pollard["practice"] == {"Wed": None, "Thu": "DNP", "Fri": None}
    assert pollard["game_status"] is None
    assert late["game_status"] == "Out"


def test_unrecognized_page() -> None:
    page = parse_report("<html><body>maintenance</body></html>")
    assert not page.recognized
    assert page.clubs == ()


def test_fragment_keeps_report_and_drops_chrome() -> None:
    html = report_page("REG-8", [_titans()])
    fragment = report_fragment(html)
    assert "menu" not in fragment
    assert parse_report(fragment) == parse_report(html)
    assert report_fragment("<p>other</p>") == "<p>other</p>"


def test_practice_reports_change_log_and_own_page_preference() -> None:
    wed = snapshot(
        NAME,
        "TEN",
        report_page("REG-8", [club_table("Tennessee Titans", DAYS, [])]),
        "2026-10-28T23:00:00+00:00",
        {"team": "TEN"},
    )
    own = snapshot(
        NAME, "TEN", report_page("REG-8", [_titans()]), "2026-10-30T22:00:00+00:00", {"team": "TEN"}
    )
    opponent_stale = snapshot(
        NAME,
        "CIN",
        report_page("REG-8", [_bengals(), _titans(status="", fri="LP")]),
        "2026-10-30T22:00:05+00:00",
        {"team": "CIN"},
        run_id=own.run_id,
    )
    repeat = snapshot(
        NAME, "TEN", report_page("REG-8", [_titans()]), "2026-10-31T13:00:00+00:00", {"team": "TEN"}
    )
    changed = snapshot(
        NAME,
        "TEN",
        report_page("REG-8", [_titans(status="OUT")]),
        "2026-10-31T19:00:00+00:00",
        {"team": "TEN"},
    )
    frame = validate(
        practice_reports([wed, own, opponent_stale, repeat, changed]), PRACTICE_REPORTS
    )
    ridley = frame.filter(frame["player_name"] == "Calvin Ridley").sort("captured_at", "report_day")
    assert ridley["report_day"].to_list() == ["Fri", "Thu", "Wed", "Fri", "Thu", "Wed"]
    assert ridley["game_status"].to_list() == ["Questionable"] * 3 + ["Out"] * 3
    assert ridley["participation"].to_list()[:3] == ["FP", "LP", "LP"]
    assert (frame["available_at"] == frame["captured_at"]).all()
    assert frame["available_at"].min() == own.fetched_at
    late = frame.filter(frame["player_name"] == "Late Add")
    assert late["report_day"].to_list() == ["Fri"]
    assert late["participation"].to_list() == [None]
    burrow = frame.filter(frame["team"] == "CIN")
    assert burrow.height == 3
    assert burrow["available_at"].min() == opponent_stale.fetched_at


def test_practice_reports_skips_preseason_and_errors() -> None:
    pre = snapshot(NAME, "TEN", report_page("PRE-2", [_titans()]), "2026-08-20T20:00:00+00:00")
    failed = snapshot(NAME, "TEN", "oops", "2026-10-30T20:00:00+00:00", status=503)
    assert practice_reports([pre, failed]).is_empty()
