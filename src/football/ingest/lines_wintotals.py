import io

import polars as pl
from bs4 import BeautifulSoup

from football import storage
from football.ingest.base import IngestOptions
from football.ingest.lines_common import (
    LIVE_MAX_AGE,
    RawCache,
    console,
    load_schedule,
    no_vig_expr,
    parse_american,
    parse_number,
    safe_team_code,
)
from football.schemas import WIN_TOTALS, validate

SOURCE = "win_totals"
NFLDATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/win_totals.csv"
SOH_URL = "https://www.covers.com/sportsoddshistory/nfl-win/?y={season}&sa=nfl&t=win&o=r"
SOH_FIRST = 2020
LEAD_DAYS = 7


def parse_soh(html: bytes, season: int) -> pl.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="soh1")
    rows: list[dict[str, object]] = []
    if table is not None:
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 4:
                continue
            rows.append(
                {
                    "season": season,
                    "team_name": cells[0],
                    "win_total": parse_number(cells[1]),
                    "over_odds": parse_american(cells[2]),
                    "under_odds": parse_american(cells[3]),
                }
            )
    schema = {
        "season": pl.Int32(),
        "team_name": pl.String(),
        "win_total": pl.Float64(),
        "over_odds": pl.Float64(),
        "under_odds": pl.Float64(),
    }
    return pl.DataFrame(rows, schema=schema).with_columns(
        pl.lit("sportsoddshistory").alias("source")
    )


def parse_nfldata(payload: bytes) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(payload)).select(
        pl.col("season").cast(pl.Int32),
        pl.col("team").alias("team_name"),
        pl.col("line").cast(pl.Float64).alias("win_total"),
        pl.col("over_odds").cast(pl.Float64),
        pl.col("under_odds").cast(pl.Float64),
        pl.lit("nfldata").alias("source"),
    )


def finalize(raw: pl.DataFrame, season_start: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    coded = raw.with_columns(
        pl.struct("team_name", "season")
        .map_elements(
            lambda row: safe_team_code(row["team_name"], row["season"]), return_dtype=pl.String
        )
        .alias("team")
    ).join(season_start, on="season", how="left")
    bad = coded.filter(
        pl.col("team").is_null() | pl.col("first_kickoff").is_null() | pl.col("win_total").is_null()
    )
    good = coded.filter(
        ~pl.col("team").is_null()
        & ~pl.col("first_kickoff").is_null()
        & ~pl.col("win_total").is_null()
    )
    table = good.select(
        "season",
        "team",
        "source",
        "win_total",
        no_vig_expr("over_odds", "under_odds").alias("over_prob"),
        (pl.col("first_kickoff") - pl.duration(days=LEAD_DAYS)).alias("available_at"),
        "over_odds",
        "under_odds",
    )
    return validate(table, WIN_TOTALS), bad


def ingest(options: IngestOptions) -> None:
    schedule = load_schedule(options.refresh)
    season_start = schedule.games.group_by("season").agg(
        pl.col("kickoff_utc").min().alias("first_kickoff")
    )
    cache = RawCache(SOURCE)
    frames: list[pl.DataFrame] = []
    nfldata = parse_nfldata(
        cache.fetch("nfldata_win_totals.csv", NFLDATA_URL, refresh=options.refresh).read_bytes()
    )
    frames.append(nfldata.filter(pl.col("season").is_in(options.seasons)))
    for season in [s for s in options.seasons if s >= SOH_FIRST]:
        live = season == schedule.games["season"].max()
        entry = cache.fetch(
            f"soh_{season}.html",
            SOH_URL.format(season=season),
            refresh=options.refresh,
            max_age=LIVE_MAX_AGE if live else None,
            browser=True,
            min_interval=2.0,
        )
        frames.append(parse_soh(entry.read_bytes(), season))
    table, bad = finalize(pl.concat(frames), season_start)
    if not table.is_empty():
        storage.upsert(table, "clean", WIN_TOTALS.name, WIN_TOTALS.keys)
    counts = table.group_by("source").len().sort("source")
    detail = ", ".join(f"{s}={n}" for s, n in counts.iter_rows())
    console.print(f"  {SOURCE}: {table.height} rows ({detail}); unparsed {bad.height}")
    if bad.height:
        console.print(bad.select("season", "source", "team_name", "win_total").head(10))
