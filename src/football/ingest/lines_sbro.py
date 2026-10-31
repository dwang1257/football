from dataclasses import dataclass
from datetime import date

import polars as pl
from bs4 import BeautifulSoup

from football.ingest.base import IngestOptions
from football.ingest.lines_common import (
    RawCache,
    close_available,
    load_schedule,
    match_by_date,
    orient,
    parse_american,
    parse_number,
    publish,
    reject_frame,
    tuesday_noon_before,
)

SOURCE = "lines_sbro"
SEASONS = range(2007, 2022)
URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{season}-{next:02d}/"
MAX_SPREAD = 27.0
MIN_TOTAL = 28.0
HEADER = (
    "date",
    "rot",
    "vh",
    "team",
    "1st",
    "2nd",
    "3rd",
    "4th",
    "final",
    "open",
    "close",
    "ml",
    "2h",
)


@dataclass(frozen=True)
class SideLine:
    spread_a: float | None
    total: float | None
    ok: bool


def split_column(a: float | None, b: float | None) -> SideLine:
    if a is None and b is None:
        return SideLine(None, None, True)
    if a is None or b is None:
        value = a if a is not None else b
        assert value is not None
        if value >= MIN_TOTAL:
            return SideLine(None, value, True)
        if abs(value) <= MAX_SPREAD:
            return SideLine(abs(value) if b is None else -abs(value), None, True)
        return SideLine(None, None, False)
    if a == 0 and b == 0:
        return SideLine(0.0, None, True)
    low, high = sorted((abs(a), abs(b)))
    if high >= MIN_TOTAL and low <= MAX_SPREAD and low < high:
        return SideLine(low if abs(a) == low else -low, high, True)
    return SideLine(None, None, False)


def game_date(code: str, season: int) -> date | None:
    digits = code.strip()
    if not digits.isdigit() or len(digits) < 3:
        return None
    month, day = int(digits[:-2]), int(digits[-2:])
    year = season if month >= 7 else season + 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def table_rows(html: bytes) -> list[list[str]]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[list[str]] = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) >= len(HEADER) and cells[0].lower() != "date":
            rows.append(cells[: len(HEADER)])
    return rows


def parse_season(html: bytes, season: int) -> tuple[pl.DataFrame, list[str]]:
    rows = table_rows(html)
    games: list[dict[str, object]] = []
    problems: list[str] = []
    index = 0
    while index < len(rows):
        first = rows[index]
        if index + 1 >= len(rows):
            problems.append(f"{season}:{index}:unpaired:{first[3]}")
            break
        second = rows[index + 1]
        pair_ok = first[0] == second[0] and (first[2], second[2]) in {
            ("V", "H"),
            ("N", "N"),
            ("H", "V"),
        }
        if not pair_ok:
            problems.append(f"{season}:{index}:unpaired:{first[3]}")
            index += 1
            continue
        index += 2
        day = game_date(first[0], season)
        opened = split_column(parse_number(first[9]), parse_number(second[9]))
        closed = split_column(parse_number(first[10]), parse_number(second[10]))
        ref = f"{season}:{first[0]}:{first[3]}@{second[3]}"
        if day is None:
            problems.append(f"{ref}:bad_date")
            continue
        if not (opened.ok and closed.ok):
            problems.append(f"{ref}:ambiguous_lines")
        games.append(
            {
                "row_id": ref,
                "season": season,
                "game_date": day,
                "team_a": first[3],
                "team_b": second[3],
                "open_spread_a": opened.spread_a,
                "open_total": opened.total,
                "close_spread_a": closed.spread_a,
                "close_total": closed.total,
                "ml_a": parse_american(first[11]),
                "ml_b": parse_american(second[11]),
            }
        )
    schema = {
        "row_id": pl.String(),
        "season": pl.Int32(),
        "game_date": pl.Date(),
        "team_a": pl.String(),
        "team_b": pl.String(),
        "open_spread_a": pl.Float64(),
        "open_total": pl.Float64(),
        "close_spread_a": pl.Float64(),
        "close_total": pl.Float64(),
        "ml_a": pl.Float64(),
        "ml_b": pl.Float64(),
    }
    return pl.DataFrame(games, schema=schema), problems


def to_lines(matched: pl.DataFrame) -> pl.DataFrame:
    close = matched.with_columns(
        *orient("swapped", "close_spread_a", "ml_a", "ml_b"),
        pl.col("close_total").alias("total"),
        pl.lit("close").alias("kind"),
        close_available(pl.col("kickoff_utc")).alias("captured_at"),
    )
    opened = matched.with_columns(
        pl.when(pl.col("swapped"))
        .then(-pl.col("open_spread_a"))
        .otherwise(pl.col("open_spread_a"))
        .alias("home_spread"),
        pl.col("open_total").alias("total"),
        pl.lit(None, dtype=pl.Float64).alias("home_ml"),
        pl.lit(None, dtype=pl.Float64).alias("away_ml"),
        pl.lit("open").alias("kind"),
        tuesday_noon_before(pl.col("kickoff_utc")).alias("captured_at"),
    )
    columns = ["game_id", "kind", "captured_at", "home_spread", "total", "home_ml", "away_ml"]
    return pl.concat([close.select(columns), opened.select(columns)]).with_columns(
        pl.lit("consensus").alias("book"), pl.col("captured_at").alias("available_at")
    )


def ingest(options: IngestOptions) -> None:
    seasons = [s for s in options.seasons if s in SEASONS]
    if not seasons:
        return
    schedule = load_schedule(options.refresh)
    cache = RawCache(SOURCE)
    frames: list[pl.DataFrame] = []
    problems: list[str] = []
    for season in seasons:
        entry = cache.fetch(
            f"nfl-odds-{season}.html",
            URL.format(season=season, next=(season + 1) % 100),
            refresh=options.refresh,
            browser=True,
            min_interval=2.0,
        )
        parsed, issues = parse_season(entry.read_bytes(), season)
        frames.append(parsed)
        problems.extend(issues)
    parsed_all = pl.concat(frames)
    result = match_by_date(parsed_all, schedule.games)
    rejects = [
        reject_frame(
            SOURCE,
            "unmatched",
            result.unmatched["season"].to_list(),
            result.unmatched["row_id"].to_list(),
        ),
        reject_frame(SOURCE, "parse", [int(p.split(":")[0]) for p in problems], problems),
    ]
    publish(SOURCE, to_lines(result.matched), rejects, seasons)
