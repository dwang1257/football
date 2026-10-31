from datetime import UTC, datetime, timedelta

import polars as pl

from football import storage
from football.ingest.nflverse_common import normalize_team
from football.ingest.nflverse_fetch import fetch_optional
from football.ingest.nflverse_specs import (
    CONTRACTS,
    DRAFT_PICKS,
    QBR,
    ROSTERS_WEEKLY,
    SNAP_COUNTS,
    TRADES,
)
from football.schemas import DEPTH_CHARTS, GAMES, INJURIES, QB_GAMES, TEAM_GAMES
from football.seasons import current_season
from football.validate.base import CheckResult

PBP_GRACE = timedelta(days=2)
KNOWN_PBP_MISSING_GAMES = ("1999_01_BAL_STL", "2000_03_SD_KC", "2000_06_BUF_MIA")
KNOWN_PBP_SCORE_ERRORS = (
    "2001_01_PIT_JAX",
    "2001_02_TEN_JAX",
    "2001_03_CLE_JAX",
    "2001_06_BUF_JAX",
    "2001_09_CIN_JAX",
    "2001_11_BAL_JAX",
    "2001_12_GB_JAX",
    "2001_16_KC_JAX",
    "2002_01_IND_JAX",
    "2002_04_NYJ_JAX",
    "2002_05_PHI_JAX",
    "2002_08_HOU_JAX",
    "2002_10_WAS_JAX",
    "2002_13_PIT_JAX",
    "2002_14_CLE_JAX",
    "2002_16_TEN_JAX",
    "2011_13_DET_NO",
)
YARDS_TOLERANCE = 1.0
YARDS_MIN_RATE = 0.99
TURNOVER_MIN_RATE = 0.995
DROP_RATIO = 0.6
COVERAGE_TABLES = (
    GAMES,
    TEAM_GAMES,
    QB_GAMES,
    INJURIES,
    DEPTH_CHARTS,
    SNAP_COUNTS,
    ROSTERS_WEEKLY,
    QBR,
    CONTRACTS,
    DRAFT_PICKS,
    TRADES,
)


def _missing(name: str, *tables: str) -> CheckResult | None:
    absent = [table for table in tables if not storage.exists("clean", table)]
    if absent:
        return CheckResult(name, False, f"missing clean tables {absent}")
    return None


def scores_match_pbp(now: datetime | None = None) -> CheckResult:
    name = "scores_match_pbp"
    if failure := _missing(name, GAMES.name, TEAM_GAMES.name):
        return failure
    cutoff = (now or datetime.now(UTC)) - PBP_GRACE
    games = storage.read("clean", GAMES.name)
    team_games = storage.read("clean", TEAM_GAMES.name)
    finished = games.filter(
        pl.col("home_score").is_not_null()
        & (pl.col("kickoff_utc") < cutoff)
        & pl.col("season").is_in(team_games["season"].unique().implode())
    )
    covered = team_games.select("game_id").unique()
    missing = finished.join(covered, on="game_id", how="anti")["game_id"]
    compared = team_games.filter(pl.col("game_id").is_in(finished["game_id"].implode()))
    wrong = compared.filter(
        pl.col("pbp_points").is_null() | (pl.col("pbp_points") != pl.col("points_for"))
    )["game_id"].unique()
    unexplained_missing = sorted(set(missing.to_list()) - set(KNOWN_PBP_MISSING_GAMES))
    unexplained_wrong = sorted(set(wrong.to_list()) - set(KNOWN_PBP_SCORE_ERRORS))
    matched = compared.height - 2 * wrong.len()
    detail = (
        f"{matched}/{compared.height} team-game scores match in {covered.height} games; "
        f"{wrong.len()} games with wrong pbp score columns "
        f"({wrong.len() - len(unexplained_wrong)} known nflfastR errors), "
        f"{missing.len()} finished games absent from pbp "
        f"({missing.len() - len(unexplained_missing)} known gaps)"
    )
    if unexplained_wrong:
        detail += f"; unexplained mismatches {unexplained_wrong[:5]}"
    if unexplained_missing:
        detail += f"; unexplained missing {unexplained_missing[:5]}"
    return CheckResult(name, not unexplained_wrong and not unexplained_missing, detail)


def team_stats_frame(seasons: list[int]) -> pl.DataFrame:
    frames = [
        frame
        for season in seasons
        if (frame := fetch_optional(f"stats_team/stats_team_week_{season}.parquet")) is not None
    ]
    if not frames:
        return pl.DataFrame()
    columns = [
        "game_id",
        "season",
        "team",
        "passing_yards",
        "sack_yards_lost",
        "rushing_yards",
        "passing_interceptions",
        "fumbles_lost_total",
    ]
    stats = pl.concat([frame.select(columns) for frame in frames], how="vertical_relaxed")
    return normalize_team(stats.with_columns(pl.col("season").cast(pl.Int32)), "team")


def team_stats_agreement(team_games: pl.DataFrame, stats: pl.DataFrame) -> dict[str, float]:
    joined = team_games.join(stats.drop("season"), on=["game_id", "team"], how="inner")
    official_pass = pl.col("passing_yards") + pl.col("sack_yards_lost")
    our_pass = pl.col("pass_yards") - pl.col("scramble_yards")
    our_rush = pl.col("rush_yards") + pl.col("scramble_yards") + pl.col("kneel_yards")
    rates = joined.select(
        ((our_pass - official_pass).abs() <= YARDS_TOLERANCE).mean().alias("pass_yards"),
        ((our_rush - pl.col("rushing_yards")).abs() <= YARDS_TOLERANCE).mean().alias("rush_yards"),
        (pl.col("interceptions_thrown") == pl.col("passing_interceptions"))
        .mean()
        .alias("interceptions"),
        (pl.col("fumbles_lost") == pl.col("fumbles_lost_total")).mean().alias("fumbles_lost"),
        ((pl.col("pass_yards") - official_pass).abs() <= YARDS_TOLERANCE)
        .mean()
        .alias("raw_pass_yards"),
        ((pl.col("rush_yards") - pl.col("rushing_yards")).abs() <= YARDS_TOLERANCE)
        .mean()
        .alias("raw_rush_yards"),
    ).row(0, named=True)
    rates["matched"] = joined.height / team_games.height if team_games.height else 0.0
    return {key: float(value or 0.0) for key, value in rates.items()}


def team_stats_match() -> CheckResult:
    name = "team_stats_match"
    if failure := _missing(name, TEAM_GAMES.name):
        return failure
    team_games = storage.read("clean", TEAM_GAMES.name)
    stats = team_stats_frame(sorted(team_games["season"].unique().to_list()))
    if stats.is_empty():
        return CheckResult(name, False, "no nflverse team stats available")
    rates = team_stats_agreement(team_games, stats)
    passed = (
        rates["pass_yards"] >= YARDS_MIN_RATE
        and rates["rush_yards"] >= YARDS_MIN_RATE
        and rates["interceptions"] >= TURNOVER_MIN_RATE
        and rates["fumbles_lost"] >= TURNOVER_MIN_RATE
    )
    detail = (
        f"matched {rates['matched']:.2%} of team-games; "
        f"within 1 yd: pass {rates['pass_yards']:.2%}, "
        f"rush {rates['rush_yards']:.2%} after moving scrambles to rushing and adding kneels "
        f"(unadjusted {rates['raw_pass_yards']:.2%} / {rates['raw_rush_yards']:.2%}); exact: "
        f"interceptions {rates['interceptions']:.2%}, fumbles lost {rates['fumbles_lost']:.2%}"
    )
    return CheckResult(name, passed, detail)


def flag_drops(counts: pl.DataFrame, current: int) -> list[int]:
    complete = counts.filter(pl.col("season") < current).sort("season")
    seasons = complete["season"].to_list()
    values = complete["rows"].to_list()
    flagged: list[int] = []
    for index, season in enumerate(seasons):
        neighbors = values[max(0, index - 1) : index] + values[index + 1 : index + 2]
        if neighbors and values[index] < DROP_RATIO * min(neighbors):
            flagged.append(season)
    return flagged


def coverage_by_season(now: datetime | None = None) -> CheckResult:
    current = current_season(now)
    parts: list[str] = []
    problems: list[str] = []
    for spec in COVERAGE_TABLES:
        if not storage.exists("clean", spec.name):
            problems.append(f"{spec.name} missing")
            continue
        counts = storage.read("clean", spec.name).group_by("season").agg(pl.len().alias("rows"))
        seasons: list[int] = counts["season"].to_list()
        first, last = min(seasons), max(seasons)
        gaps = sorted(set(range(first, last + 1)) - set(seasons))
        drops = flag_drops(counts, current)
        parts.append(f"{spec.name} {first}-{last} ({int(counts['rows'].sum())} rows)")
        if gaps:
            problems.append(f"{spec.name} gap seasons {gaps}")
        if drops:
            problems.append(f"{spec.name} drops in {drops}")
    detail = "; ".join(parts)
    if problems:
        detail += " | problems: " + "; ".join(problems)
    return CheckResult("coverage_by_season", not problems, detail)
