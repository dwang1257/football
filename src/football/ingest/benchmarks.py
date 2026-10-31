import io
import json
import math
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from football import storage
from football.ingest.base import IngestFn, IngestOptions
from football.ingest.lines_common import (
    ET,
    LIVE_MAX_AGE,
    REJECTS,
    RawCache,
    close_available,
    console,
    load_schedule,
    match_by_date,
    match_by_ids,
    reject_frame,
    safe_team_code,
    write_rejects,
)
from football.ingest.lines_espn import ESPN_TEAM_IDS
from football.schemas import BENCHMARKS, TEAM_RATINGS, UTC_TS, WIN_TOTALS, validate
from football.validate.base import CheckFn, CheckResult

NFELO_URL = "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv"
ELO_URL = "https://datahub.io/fivethirtyeight/nfl-elo/_r/-/data/nfl_elo.csv"
FPI_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}"
    "/types/{type}/weeks/{week}/powerindex?page={page}"
)
FPI_FIRST = (2015, 10)
FPI_WINDOW_BEFORE = timedelta(days=10)
FPI_WINDOW_AFTER = timedelta(days=7)
POSTSEASON_WEEKS = {1: 1, 2: 2, 3: 3, 5: 4}
COVERAGE_THRESHOLD = 0.95
ELO_SPREAD_PER_LOGIT10 = 16.0


def benchmark_frame(
    matched: pl.DataFrame, prob: str, margin: pl.Expr | None, source: str
) -> pl.DataFrame:
    flip = pl.col("swapped")
    p = pl.col(prob).cast(pl.Float64)
    home_margin = pl.lit(None, dtype=pl.Float64) if margin is None else margin
    return validate(
        matched.select(
            "game_id",
            pl.lit(source).alias("source"),
            pl.when(flip).then(1 - p).otherwise(p).alias("home_win_prob"),
            pl.when(flip).then(-home_margin).otherwise(home_margin).alias("home_margin"),
            close_available(pl.col("kickoff_utc")).alias("available_at"),
        ).filter(pl.col("home_win_prob").is_not_null()),
        BENCHMARKS,
    )


def write_benchmarks(
    df: pl.DataFrame, source: str, unmatched: pl.DataFrame, seasons: list[int]
) -> None:
    write_rejects(
        source,
        seasons,
        [
            reject_frame(
                source, "unmatched", unmatched["season"].to_list(), unmatched["row_id"].to_list()
            )
        ],
    )
    if not df.is_empty():
        storage.upsert(df, "clean", BENCHMARKS.name, BENCHMARKS.keys)
    console.print(f"  {source}: {df.height} rows; unmatched {unmatched.height}")


def ingest_nfelo(options: IngestOptions) -> None:
    schedule = load_schedule(options.refresh)
    entry = RawCache("bench_nfelo").fetch(
        "nfelo_games.csv", NFELO_URL, refresh=options.refresh, max_age=LIVE_MAX_AGE
    )
    raw = pl.read_csv(io.BytesIO(entry.read_bytes()), infer_schema_length=None)
    parts = pl.col("game_id").str.split("_")
    keyed = raw.select(
        pl.col("game_id").alias("row_id"),
        parts.list.get(0).cast(pl.Int32).alias("season"),
        parts.list.get(1).cast(pl.Int32).alias("week"),
        parts.list.get(2).alias("away"),
        parts.list.get(3).alias("home"),
        "nfelo_home_probability_close",
        (-pl.col("nfelo_home_line_close").cast(pl.Float64)).alias("margin"),
    ).filter(pl.col("season").is_in(options.seasons))
    result = match_by_ids(keyed, schedule.games)
    frame = benchmark_frame(
        result.matched, "nfelo_home_probability_close", pl.col("margin"), "nfelo"
    )
    write_benchmarks(frame, "nfelo", result.unmatched, options.seasons)


def ingest_elo538(options: IngestOptions) -> None:
    schedule = load_schedule(options.refresh)
    entry = RawCache("bench_elo538").fetch("nfl_elo.csv", ELO_URL, refresh=options.refresh)
    raw = pl.read_csv(io.BytesIO(entry.read_bytes()), infer_schema_length=None)
    keyed = raw.filter(pl.col("season").is_in(options.seasons)).select(
        pl.concat_str(["date", "team2", "team1"], separator="_").alias("row_id"),
        pl.col("season").cast(pl.Int32),
        pl.col("date").str.to_date("%Y-%m-%d").alias("game_date"),
        pl.col("team1").alias("team_a"),
        pl.col("team2").alias("team_b"),
        pl.col("elo_prob1").cast(pl.Float64),
    )
    result = match_by_date(keyed, schedule.games)
    logit10 = (pl.col("elo_prob1") / (1 - pl.col("elo_prob1"))).log(10)
    frame = benchmark_frame(result.matched, "elo_prob1", ELO_SPREAD_PER_LOGIT10 * logit10, "elo538")
    write_benchmarks(frame, "elo538", result.unmatched, options.seasons)


def fpi_weeks(games: pl.DataFrame, seasons: list[int]) -> list[tuple[int, int, int, int, datetime]]:
    starts = (
        games.group_by("season", "week")
        .agg(pl.col("kickoff_utc").min().alias("start"))
        .sort("season", "week")
    )
    reg_weeks = (
        games.filter(pl.col("game_type") == "REG")
        .group_by("season")
        .agg(pl.col("week").max().alias("reg"))
    )
    lookup = {(s, w): t for s, w, t in starts.iter_rows()}
    regs = dict(reg_weeks.iter_rows())
    weeks: list[tuple[int, int, int, int, datetime]] = []
    for season in seasons:
        if season not in regs:
            continue
        for week in range(1, regs[season] + 1):
            if (season, week) >= FPI_FIRST and (season, week) in lookup:
                weeks.append((season, 2, week, week, lookup[(season, week)]))
        for espn_week, offset in POSTSEASON_WEEKS.items():
            week = regs[season] + offset
            if (season, week) in lookup and season >= FPI_FIRST[0]:
                weeks.append((season, 3, espn_week, week, lookup[(season, week)]))
    return weeks


def fpi_max_age(start: datetime, now: datetime) -> timedelta | None:
    return LIVE_MAX_AGE if start - FPI_WINDOW_BEFORE <= now <= start + FPI_WINDOW_AFTER else None


def fetch_fpi(weeks: list[tuple[int, int, int, int, datetime]], refresh: bool) -> RawCache:
    cache = RawCache("bench_espn_fpi")
    now = datetime.now(UTC)
    for season, kind, espn_week, _, start in weeks:
        if start - FPI_WINDOW_BEFORE > now:
            continue
        max_age = fpi_max_age(start, now)
        page = 1
        while True:
            name = f"{season}_{kind}_{espn_week}_p{page}.json"
            url = FPI_URL.format(season=season, type=kind, week=espn_week, page=page)
            entry = cache.fetch(name, url, refresh=refresh, max_age=max_age, min_interval=0.3)
            payload = json.loads(entry.read_bytes())
            if page >= int(payload.get("pageCount") or 0):
                break
            page += 1
    return cache


def predictive(item: dict[str, Any], name: str) -> float | None:
    for entry in item.get("predictives") or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            value = entry.get("value")
            return float(value) if isinstance(value, int | float) and math.isfinite(value) else None
    return None


def stamp(item: dict[str, Any]) -> datetime | None:
    stamps: list[datetime] = []
    updated = item.get("lastUpdated")
    if isinstance(updated, str):
        with suppress(ValueError):
            stamps.append(datetime.fromisoformat(updated.replace("Z", "+00:00")).astimezone(UTC))
    run_key = str(item.get("runDateTimeKey") or "")
    if len(run_key) == 14:
        with suppress(ValueError):
            local = datetime.strptime(run_key, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo(ET))
            stamps.append(local.astimezone(UTC))
    return max(stamps) if stamps else None


def parse_fpi(
    cache: RawCache, weeks: list[tuple[int, int, int, int, datetime]]
) -> tuple[pl.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for season, kind, espn_week, week, _ in weeks:
        for page in range(1, 4):
            for entry in cache.versions(f"{season}_{kind}_{espn_week}_p{page}.json"):
                payload = json.loads(entry.read_bytes())
                for item in payload.get("items") or []:
                    ref = (item.get("team") or {}).get("$ref", "")
                    team_id = ref.split("/teams/")[1].split("?")[0] if "/teams/" in ref else ""
                    team = safe_team_code(ESPN_TEAM_IDS.get(team_id), season)
                    captured = stamp(item)
                    if team is None or captured is None:
                        skipped += 1
                        continue
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": team,
                            "rating": predictive(item, "fpi"),
                            "offense": predictive(item, "epaoffense"),
                            "defense": predictive(item, "epadefense"),
                            "special_teams": predictive(item, "epaspecialteams"),
                            "captured_at": captured,
                            "fpi_rank": predictive(item, "fpirank"),
                            "projected_wins": predictive(item, "projectedw"),
                            "season_type": kind,
                        }
                    )
    schema = {
        "season": pl.Int32(),
        "week": pl.Int32(),
        "team": pl.String(),
        "rating": pl.Float64(),
        "offense": pl.Float64(),
        "defense": pl.Float64(),
        "special_teams": pl.Float64(),
        "captured_at": UTC_TS,
        "fpi_rank": pl.Float64(),
        "projected_wins": pl.Float64(),
        "season_type": pl.Int32(),
    }
    return pl.DataFrame(rows, schema=schema), skipped


def ingest_fpi(options: IngestOptions) -> None:
    schedule = load_schedule(options.refresh)
    weeks = fpi_weeks(schedule.games, options.seasons)
    cache = fetch_fpi(weeks, options.refresh)
    parsed, skipped = parse_fpi(cache, weeks)
    table = validate(
        parsed.unique(["season", "week", "team", "captured_at"], keep="last", maintain_order=True)
        .filter(pl.col("rating").is_not_null())
        .with_columns(
            pl.lit("espn_fpi").alias("source"), pl.col("captured_at").alias("available_at")
        ),
        TEAM_RATINGS,
    )
    if not table.is_empty():
        storage.upsert(table, "clean", TEAM_RATINGS.name, TEAM_RATINGS.keys)
    console.print(f"  espn_fpi: {table.height} rows from {len(weeks)} weeks; skipped {skipped}")


def coverage_line(bench: pl.DataFrame, games: pl.DataFrame, source: str) -> tuple[str, bool]:
    rows = bench.filter(pl.col("source") == source).join(
        games.drop("week"), on="game_id", how="inner"
    )
    if rows.is_empty():
        return f"{source}: no rows", False
    first, last = rows["season"].min(), rows["season"].max()
    played = games.filter(
        pl.col("season").is_between(first, last), pl.col("home_score").is_not_null()
    )
    covered = played.join(rows.select("game_id"), on="game_id", how="semi").height
    share = covered / played.height if played.height else 0.0
    early = float(rows.select((pl.col("available_at") < pl.col("kickoff_utc")).mean()).item())
    ok = share >= COVERAGE_THRESHOLD and early == 1.0
    return (
        f"{source}: seasons {first}-{last} rows={rows.height} played-game coverage={share:.1%} "
        f"available-before-kickoff={early:.1%}",
        ok,
    )


def benchmark_coverage() -> CheckResult:
    games = load_schedule(max_age=None).games.select(
        "game_id", "season", "week", "kickoff_utc", "home_score"
    )
    lines: list[str] = []
    ok = True
    if storage.exists("clean", BENCHMARKS.name):
        bench = storage.read("clean", BENCHMARKS.name)
        for source in ("nfelo", "elo538"):
            text, good = coverage_line(bench, games, source)
            lines.append(text)
            ok &= good
    else:
        lines.append("no benchmarks table")
        ok = False
    if storage.exists("clean", TEAM_RATINGS.name):
        fpi = storage.read("clean", TEAM_RATINGS.name).filter(pl.col("source") == "espn_fpi")
        snaps = fpi.group_by("season", "week", "captured_at").agg(pl.len().alias("teams"))
        complete = (
            float(snaps.select((pl.col("teams") == 32).mean()).item()) if snaps.height else 0.0
        )
        week_start = games.group_by("season", "week").agg(
            pl.col("kickoff_utc").min().alias("start")
        )
        late = snaps.join(week_start, on=["season", "week"], how="left").filter(
            pl.col("captured_at") >= pl.col("start")
        )
        lines.append(
            f"espn_fpi: rows={fpi.height} snapshots={snaps.height} "
            f"seasons={fpi['season'].n_unique()} complete(32 teams)={complete:.1%} "
            f"stamped at/after their week's first kickoff={late.height}"
        )
        ok &= complete >= COVERAGE_THRESHOLD
    else:
        lines.append("no team_ratings table")
        ok = False
    if storage.exists("clean", WIN_TOTALS.name):
        totals = storage.read("clean", WIN_TOTALS.name).group_by("source", "season").len()
        for source, group in totals.group_by("source"):
            full = int((group["len"] == 32).sum())
            lines.append(f"win_totals {source[0]}: seasons={group.height} with 32 teams={full}")
            ok &= full == group.height
    else:
        lines.append("no win_totals table")
        ok = False
    if storage.exists("clean", REJECTS.name):
        rejects = storage.read("clean", REJECTS.name).filter(
            pl.col("source").is_in(["nfelo", "elo538"]), pl.col("reason") == "unmatched"
        )
        for (source,), group in rejects.group_by("source"):
            lines.append(f"{source} unmatched={group.height} e.g. {group['ref'].head(3).to_list()}")
    return CheckResult("benchmark_coverage", ok, "\n".join(lines))


SOURCES: dict[str, IngestFn] = {
    "nfelo": ingest_nfelo,
    "espn_fpi": ingest_fpi,
    "elo538": ingest_elo538,
}
CHECKS: dict[str, CheckFn] = {"benchmark_coverage": benchmark_coverage}
