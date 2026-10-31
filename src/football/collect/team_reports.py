import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl
from bs4 import BeautifulSoup, Tag

from football.collect.archive import Snapshot
from football.collect.base import CollectorResult, Context
from football.collect.http import FetchError
from football.schemas import PRACTICE_REPORTS, UTC_TS
from football.seasons import current_season
from football.teams import franchise, team_code

NAME = "team_injury_reports"

CLUB_DOMAINS: dict[str, str] = {
    "ARI": "www.azcardinals.com",
    "ATL": "www.atlantafalcons.com",
    "BAL": "www.baltimoreravens.com",
    "BUF": "www.buffalobills.com",
    "CAR": "www.panthers.com",
    "CHI": "www.chicagobears.com",
    "CIN": "www.bengals.com",
    "CLE": "www.clevelandbrowns.com",
    "DAL": "www.dallascowboys.com",
    "DEN": "www.denverbroncos.com",
    "DET": "www.detroitlions.com",
    "GB": "www.packers.com",
    "HOU": "www.houstontexans.com",
    "IND": "www.colts.com",
    "JAX": "www.jaguars.com",
    "KC": "www.chiefs.com",
    "LA": "www.therams.com",
    "LAC": "www.chargers.com",
    "LV": "www.raiders.com",
    "MIA": "www.miamidolphins.com",
    "MIN": "www.vikings.com",
    "NE": "www.patriots.com",
    "NO": "www.neworleanssaints.com",
    "NYG": "www.giants.com",
    "NYJ": "www.newyorkjets.com",
    "PHI": "www.philadelphiaeagles.com",
    "PIT": "www.steelers.com",
    "SEA": "www.seahawks.com",
    "SF": "www.49ers.com",
    "TB": "www.buccaneers.com",
    "TEN": "www.tennesseetitans.com",
    "WAS": "www.commanders.com",
}

_WEEK_LABEL = re.compile(r"/week/((?:PRE|REG|POST)-\d+)")
_PARTICIPATION = {"DNP", "LP", "FP"}
_EMPTY = {"", "-", "--", "(-)", "N/A", "NA", "UNSPECIFIED", "NONE"}
_FIXED_COLUMNS = {"player", "position", "injury", "game status"}


@dataclass(frozen=True)
class ClubReport:
    club: str
    team: str | None
    days: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReportPage:
    week_label: str | None
    clubs: tuple[ClubReport, ...]
    recognized: bool


def report_url(team: str) -> str:
    return f"https://{CLUB_DOMAINS[team]}/team/injury-report/"


def _text(node: Tag | None) -> str:
    return node.get_text(" ", strip=True) if node is not None else ""


def _clean_participation(value: str) -> str | None:
    token = value.strip().upper()
    if token in _EMPTY:
        return None
    return token if token in _PARTICIPATION else value.strip()


def _clean_status(value: str) -> str | None:
    token = value.strip()
    if token.upper() in _EMPTY:
        return None
    return token.title()


def _team_for(club: str) -> str | None:
    try:
        return franchise(club)
    except ValueError:
        return None


def _parse_table(club: str, table: Tag) -> ClubReport:
    headers = [_text(cell) for cell in table.select("thead th")]
    lowered = [header.lower() for header in headers]
    days = tuple(header for header in headers if header.lower() not in _FIXED_COLUMNS)
    rows: list[dict[str, Any]] = []
    for tr in table.select("tbody tr"):
        cells = [_text(cell) for cell in tr.find_all(["td", "th"])]
        if len(cells) != len(headers):
            continue
        record = dict(zip(lowered, cells, strict=True))
        player = record.get("player", "").strip()
        if not player:
            continue
        rows.append(
            {
                "player_name": player,
                "position": record.get("position") or None,
                "injury": record.get("injury") or None,
                "practice": {day: _clean_participation(record[day.lower()]) for day in days},
                "game_status": _clean_status(record.get("game status", "")),
            }
        )
    return ClubReport(club=club, team=_team_for(club), days=days, rows=tuple(rows))


def parse_report(html: str) -> ReportPage:
    soup = BeautifulSoup(html, "lxml")
    selected = soup.select_one(".d3-o-season-selector option[selected]")
    label = None
    if selected is not None:
        match = _WEEK_LABEL.search(str(selected.get("value", "")))
        label = match.group(1) if match else None
    clubs: list[ClubReport] = []
    for container in soup.select(".nfl-o-injury-report__container"):
        table = container.find("table")
        club = _text(container.select_one(".nfl-o-injury-report__club-name"))
        if isinstance(table, Tag) and club:
            clubs.append(_parse_table(club, table))
    recognized = selected is not None or bool(clubs)
    return ReportPage(week_label=label, clubs=tuple(clubs), recognized=recognized)


def report_fragment(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    parts = [
        str(node) for node in soup.select(".d3-o-season-selector, .nfl-o-injury-report__container")
    ]
    return "\n".join(parts) if parts else html


def parsed_rows(page: ReportPage) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for club in page.clubs:
        for row in club.rows:
            out.append({"club": club.club, "team": club.team, **row})
    return out


def collect(ctx: Context, result: CollectorResult) -> None:
    for team in sorted(CLUB_DOMAINS):
        url = report_url(team)
        try:
            fetched = ctx.fetcher.get(url)
        except FetchError as exc:
            result.failed[team] = str(exc)
            continue
        if not fetched.ok:
            ctx.save(result, team, fetched, fetched.text[:2000], {"team": team})
            result.failed[team] = f"HTTP {fetched.status}"
            continue
        page = parse_report(fetched.text)
        rows = parsed_rows(page)
        meta = {
            "team": team,
            "week_label": page.week_label,
            "clubs": [club.club for club in page.clubs],
            "parsed": rows,
        }
        ctx.save(result, team, fetched, report_fragment(fetched.text), meta)
        if not page.recognized:
            result.failed[team] = "injury report layout not recognized"
            continue
        result.ok.append(team)
    result.notes.append(f"{len(result.ok)} club pages parsed")


def week_from_label(label: str | None) -> int | None:
    if not label or not label.startswith("REG-"):
        return None
    return int(label.removeprefix("REG-"))


def _snapshot_records(snapshot: Snapshot) -> Iterable[dict[str, Any]]:
    if not snapshot.ok or not isinstance(snapshot.body, str):
        return
    page = parse_report(snapshot.body)
    week = week_from_label(page.week_label)
    if week is None:
        return
    season = current_season(snapshot.fetched_at)
    owner = snapshot.meta.get("team")
    for club in page.clubs:
        if club.team is None:
            continue
        team = team_code(club.team, season)
        for row in club.rows:
            filled = [(day, value) for day, value in row["practice"].items() if value]
            if not filled and row["game_status"] and club.days:
                filled = [(club.days[-1], None)]
            for day, participation in filled:
                yield {
                    "run_id": snapshot.run_id,
                    "own_page": owner == club.team,
                    "season": season,
                    "week": week,
                    "team": team,
                    "player_name": row["player_name"],
                    "position": row["position"],
                    "injury": row["injury"],
                    "report_day": day,
                    "participation": participation,
                    "game_status": row["game_status"],
                    "captured_at": snapshot.fetched_at,
                }


_PRACTICE_SCHEMA: dict[str, Any] = {
    "run_id": pl.String(),
    "own_page": pl.Boolean(),
    "season": pl.Int32(),
    "week": pl.Int32(),
    "team": pl.String(),
    "player_name": pl.String(),
    "position": pl.String(),
    "injury": pl.String(),
    "report_day": pl.String(),
    "participation": pl.String(),
    "game_status": pl.String(),
    "captured_at": UTC_TS,
}

_VALUE_COLUMNS = ("position", "injury", "participation", "game_status")
_ROW_KEY = ("season", "week", "team", "player_name", "report_day")


def practice_reports(snapshots: Iterable[Snapshot]) -> pl.DataFrame:
    records = [record for snapshot in snapshots for record in _snapshot_records(snapshot)]
    frame = pl.DataFrame(records, schema=_PRACTICE_SCHEMA)
    preferred = (
        frame.with_columns(
            pl.col("own_page").any().over("run_id", "team").alias("has_own"),
            pl.col("captured_at").min().over("run_id", "team").alias("first_seen"),
        )
        .filter(
            pl.when(pl.col("has_own"))
            .then(pl.col("own_page"))
            .otherwise(pl.col("captured_at") == pl.col("first_seen"))
        )
        .sort("captured_at")
        .unique(subset=["run_id", *_ROW_KEY], keep="first", maintain_order=True)
    )
    signature = pl.concat_str(
        [pl.col(name).fill_null("\u0000") for name in _VALUE_COLUMNS], separator="|"
    )
    changes = (
        preferred.sort([*_ROW_KEY, "captured_at"])
        .with_columns(signature.alias("signature"))
        .filter(pl.col("signature") != pl.col("signature").shift(1).over(_ROW_KEY).fill_null(""))
        .with_columns(pl.col("captured_at").alias("available_at"))
    )
    return changes.select(list(PRACTICE_REPORTS.columns))
