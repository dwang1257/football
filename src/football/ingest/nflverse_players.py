from datetime import timedelta

import polars as pl

from football.ingest.base import IngestOptions
from football.ingest.nflverse_common import (
    eastern_wall_clock,
    hours,
    log,
    normalize_team,
    replace_seasons,
    to_eastern_date,
)
from football.ingest.nflverse_fetch import fetch_frame, fetch_optional
from football.ingest.nflverse_games import load_games, team_schedule
from football.ingest.nflverse_specs import ROSTERS_WEEKLY, SNAP_COUNTS
from football.schemas import DEPTH_CHARTS, INJURIES, UTC_TS

FIRST_INJURY_SEASON = 2009
FIRST_DEPTH_SEASON = 2001
FIRST_DAILY_DEPTH_SEASON = 2025
FIRST_SNAP_SEASON = 2012
FIRST_ROSTER_SEASON = 2002
SNAP_DELAY = hours(12)
INACTIVES_LEAD = hours(1)
INJURY_REPORT_HOUR_ET = 18
DEPTH_HOUR_ET = 12
WEDNESDAY = 3
THURSDAY = 4


def column(df: pl.DataFrame, name: str, dtype: pl.DataType) -> pl.Expr:
    if name in df.columns:
        return pl.col(name).cast(dtype)
    return pl.lit(None, dtype=dtype).alias(name)


def prefixed_id(columns: list[tuple[str, str]]) -> tuple[pl.Expr, pl.Expr]:
    ids = [pl.col(name).cast(pl.String).str.strip_chars().replace("", None) for name, _ in columns]
    value = pl.coalesce(
        ids[0],
        *(
            pl.concat_str(pl.lit(f"{kind}:"), expr)
            for expr, (_, kind) in zip(ids[1:], columns[1:], strict=True)
        ),
    )
    kind = pl.coalesce(
        *(
            pl.when(expr.is_not_null()).then(pl.lit(kind))
            for expr, (_, kind) in zip(ids, columns, strict=True)
        )
    )
    return value.alias("player_id"), kind.alias("player_id_type")


def season_frames(dataset: str, season: int, refresh: bool) -> pl.DataFrame | None:
    frame = fetch_optional(f"{dataset}_{season}.parquet", refresh)
    if frame is None or frame.is_empty():
        log(f"{dataset}: season {season} not published or empty, skipped")
        return None
    return frame


def week_schedule(games: pl.DataFrame) -> pl.DataFrame:
    return team_schedule(games).select(
        "season",
        "week",
        "team",
        "game_id",
        "kickoff_utc",
        pl.col("game_type").alias("sched_game_type"),
    )


def week_first_kickoff(games: pl.DataFrame) -> pl.DataFrame:
    return games.group_by("season", "week").agg(pl.col("kickoff_utc").min().alias("week_kickoff"))


def week_last_kickoff(games: pl.DataFrame) -> pl.DataFrame:
    return games.group_by("season", "week").agg(
        pl.col("kickoff_utc").max().alias("week_last_kickoff")
    )


def report_lead_days(kickoff: pl.Expr) -> pl.Expr:
    weekday = kickoff.dt.convert_time_zone("America/New_York").dt.weekday()
    return pl.when(weekday.is_in([1, 6, 7])).then(2).otherwise(1)


def final_report_time(kickoff: pl.Expr) -> pl.Expr:
    report_date = to_eastern_date(kickoff) - pl.duration(days=report_lead_days(kickoff))
    return eastern_wall_clock(report_date, INJURY_REPORT_HOUR_ET)


def clean_injuries(raw: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    frame = raw.with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        column(raw, "date_modified", UTC_TS),
        pl.col("gsis_id").str.strip_chars().replace("", None).alias("player_id"),
    )
    frame = normalize_team(frame, "team").join(
        week_schedule(games), on=["season", "week", "team"], how="left"
    )
    unmapped = frame.filter(pl.col("kickoff_utc").is_null()).height
    missing_id = frame.filter(pl.col("player_id").is_null()).height
    if unmapped or missing_id:
        log(f"injuries: dropped {unmapped} rows without a team game, {missing_id} without gsis id")
    report_time = final_report_time(pl.col("kickoff_utc"))
    return (
        frame.filter(pl.col("kickoff_utc").is_not_null() & pl.col("player_id").is_not_null())
        .select(
            "season",
            "week",
            pl.coalesce("game_type", "sched_game_type").alias("game_type"),
            "team",
            "player_id",
            "full_name",
            "position",
            "report_status",
            "practice_status",
            pl.max_horizontal(report_time, pl.col("date_modified")).alias("available_at"),
            "game_id",
            "kickoff_utc",
            report_time.alias("final_report_at"),
            "date_modified",
            "report_primary_injury",
            "report_secondary_injury",
            "practice_primary_injury",
        )
        .unique(["season", "week", "team", "player_id"], keep="last", maintain_order=True)
    )


def ingest_injuries(options: IngestOptions) -> None:
    games = load_games(options)
    for season in options.seasons:
        if season < FIRST_INJURY_SEASON:
            continue
        raw = season_frames("injuries/injuries", season, options.refresh)
        if raw is None:
            continue
        replace_seasons(clean_injuries(raw, games), INJURIES, [season])


def game_week_capture(kickoff: pl.Expr) -> pl.Expr:
    local_date = to_eastern_date(kickoff)
    weekday = kickoff.dt.convert_time_zone("America/New_York").dt.weekday()
    wednesday = local_date - pl.duration(days=(weekday - WEDNESDAY + 7) % 7)
    return pl.min_horizontal(
        eastern_wall_clock(wednesday, DEPTH_HOUR_ET), kickoff - timedelta(hours=24)
    )


def clean_weekly_depth(raw: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    player_id, player_id_type = prefixed_id([("gsis_id", "gsis"), ("elias_id", "elias")])
    frame = raw.filter(pl.col("week").is_not_null()).with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("club_code").alias("team"),
        player_id,
        player_id_type,
        pl.coalesce(
            pl.col("depth_position").str.strip_chars().replace("", None), pl.col("position")
        ).alias("slot"),
        pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth_rank"),
    )
    frame = (
        normalize_team(frame, "team")
        .join(week_schedule(games), on=["season", "week", "team"], how="left")
        .join(week_first_kickoff(games), on=["season", "week"], how="left")
    )
    unmapped = frame.filter(pl.col("week_kickoff").is_null()).height
    if unmapped:
        log(f"depth_charts: dropped {unmapped} weekly rows in weeks without games")
    anchor = pl.coalesce("kickoff_utc", "week_kickoff")
    return frame.filter(
        pl.col("week_kickoff").is_not_null()
        & pl.col("player_id").is_not_null()
        & pl.col("depth_rank").is_not_null()
    ).select(
        "season",
        "week",
        "team",
        "player_id",
        "full_name",
        pl.col("slot").alias("position"),
        "depth_rank",
        game_week_capture(anchor).alias("captured_at"),
        game_week_capture(anchor).alias("available_at"),
        "player_id_type",
        pl.col("position").alias("player_position"),
        "formation",
        "game_id",
        pl.lit("weekly").alias("chart_kind"),
    )


def clean_daily_depth(raw: pl.DataFrame, games: pl.DataFrame, season: int) -> pl.DataFrame:
    player_id, player_id_type = prefixed_id([("gsis_id", "gsis"), ("espn_id", "espn")])
    frame = raw.with_columns(
        pl.col("dt")
        .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC")
        .cast(UTC_TS)
        .alias("captured_at"),
        player_id,
        player_id_type,
        pl.lit(season, dtype=pl.Int32).alias("season"),
    )
    frame = normalize_team(frame, "team")
    upcoming = (
        week_schedule(games)
        .filter(pl.col("season") == season)
        .select("team", "week", "game_id", "kickoff_utc")
        .sort("kickoff_utc")
    )
    frame = frame.sort("captured_at").join_asof(
        upcoming,
        left_on="captured_at",
        right_on="kickoff_utc",
        by="team",
        strategy="forward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    unmapped = frame.filter(pl.col("week").is_null()).height
    if unmapped:
        log(f"depth_charts: dropped {unmapped} daily rows captured after the team's last game")
    frame = frame.filter(pl.col("week").is_not_null() & pl.col("player_id").is_not_null()).select(
        "season",
        "week",
        "team",
        "player_id",
        pl.col("player_name").alias("full_name"),
        pl.col("pos_abb").alias("position"),
        pl.col("pos_rank").cast(pl.Int32).alias("depth_rank"),
        "captured_at",
        pl.col("captured_at").alias("available_at"),
        "player_id_type",
        pl.lit(None, dtype=pl.String).alias("player_position"),
        pl.col("pos_grp").alias("formation"),
        "game_id",
        pl.lit("daily").alias("chart_kind"),
    )
    return changed_snapshots(frame)


def changed_snapshots(frame: pl.DataFrame) -> pl.DataFrame:
    signature = (
        frame.group_by("season", "week", "team", "captured_at")
        .agg(
            pl.concat_str(
                "position", "depth_rank", "player_id", "formation", separator="|", ignore_nulls=True
            )
            .sort()
            .str.join(";")
            .hash()
            .alias("signature")
        )
        .sort("captured_at")
        .with_columns(pl.col("signature").shift(1).over("season", "week", "team").alias("previous"))
        .filter(pl.col("previous").is_null() | (pl.col("signature") != pl.col("previous")))
    )
    return frame.join(
        signature.select("season", "week", "team", "captured_at"),
        on=["season", "week", "team", "captured_at"],
        how="semi",
    )


def ingest_depth_charts(options: IngestOptions) -> None:
    games = load_games(options)
    for season in options.seasons:
        if season < FIRST_DEPTH_SEASON:
            continue
        raw = season_frames("depth_charts/depth_charts", season, options.refresh)
        if raw is None:
            continue
        if "dt" in raw.columns:
            frame = clean_daily_depth(raw, games, season)
        else:
            frame = clean_weekly_depth(raw, games)
        keys = list(DEPTH_CHARTS.keys)
        replace_seasons(
            frame.unique(keys, keep="first", maintain_order=True), DEPTH_CHARTS, [season]
        )


def players_crosswalk(refresh: bool) -> pl.DataFrame:
    players = fetch_frame("players/players.parquet", refresh)
    return (
        players.select("pfr_id", "gsis_id")
        .drop_nulls()
        .unique("pfr_id", keep="first", maintain_order=True)
    )


def clean_snap_counts(
    raw: pl.DataFrame, games: pl.DataFrame, crosswalk: pl.DataFrame
) -> pl.DataFrame:
    frame = raw.join(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left")
    player_id, player_id_type = prefixed_id([("gsis_id", "gsis"), ("pfr_player_id", "pfr")])
    frame = frame.with_columns(player_id, player_id_type, pl.col("season").cast(pl.Int32))
    frame = normalize_team(frame, "team").join(
        games.select("game_id", "kickoff_utc"), on="game_id", how="left"
    )
    unmapped = frame.filter(pl.col("kickoff_utc").is_null() | pl.col("player_id").is_null()).height
    if unmapped:
        log(f"snap_counts: dropped {unmapped} rows without a known game or player id")
    return (
        frame.filter(pl.col("kickoff_utc").is_not_null() & pl.col("player_id").is_not_null())
        .select(
            "game_id",
            "season",
            pl.col("week").cast(pl.Int32),
            "game_type",
            "team",
            "player_id",
            "player_id_type",
            "pfr_player_id",
            pl.col("player").alias("full_name"),
            "position",
            *(
                pl.col(name).cast(pl.Float64)
                for name in (
                    "offense_snaps",
                    "offense_pct",
                    "defense_snaps",
                    "defense_pct",
                    "st_snaps",
                    "st_pct",
                )
            ),
            (pl.col("kickoff_utc") + SNAP_DELAY).alias("available_at"),
        )
        .unique(["game_id", "team", "player_id"], keep="first", maintain_order=True)
    )


def ingest_snap_counts(options: IngestOptions) -> None:
    games = load_games(options)
    crosswalk = players_crosswalk(options.refresh)
    for season in options.seasons:
        if season < FIRST_SNAP_SEASON:
            continue
        raw = season_frames("snap_counts/snap_counts", season, options.refresh)
        if raw is None:
            continue
        replace_seasons(clean_snap_counts(raw, games, crosswalk), SNAP_COUNTS, [season])


def clean_rosters(raw: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    player_id, player_id_type = prefixed_id(
        [
            ("gsis_id", "gsis"),
            ("pfr_id", "pfr"),
            ("espn_id", "espn"),
            ("sportradar_id", "sportradar"),
        ]
    )
    frame = raw.with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), player_id, player_id_type
    )
    frame = (
        normalize_team(frame, "team")
        .join(week_schedule(games), on=["season", "week", "team"], how="left")
        .join(week_last_kickoff(games), on=["season", "week"], how="left")
    )
    known = pl.col("week_last_kickoff").is_not_null() & pl.col("player_id").is_not_null()
    unmapped = frame.filter(~known).height
    if unmapped:
        log(f"rosters_weekly: dropped {unmapped} rows outside game weeks or without a player id")
    available = pl.coalesce(pl.col("kickoff_utc") - INACTIVES_LEAD, pl.col("week_last_kickoff"))
    return (
        frame.filter(known)
        .select(
            "season",
            "week",
            pl.coalesce("game_type", "sched_game_type").alias("game_type"),
            "team",
            "game_id",
            "player_id",
            "player_id_type",
            "full_name",
            "position",
            column(raw, "depth_chart_position", pl.String()),
            "status",
            column(raw, "status_description_abbr", pl.String()),
            column(raw, "years_exp", pl.Int32()),
            column(raw, "rookie_year", pl.Int32()),
            column(raw, "draft_number", pl.Int32()),
            column(raw, "birth_date", pl.Date()),
            available.alias("available_at"),
        )
        .unique(["season", "week", "team", "player_id"], keep="last", maintain_order=True)
    )


def ingest_rosters_weekly(options: IngestOptions) -> None:
    games = load_games(options)
    for season in options.seasons:
        if season < FIRST_ROSTER_SEASON:
            continue
        raw = season_frames("weekly_rosters/roster_weekly", season, options.refresh)
        if raw is None:
            continue
        replace_seasons(clean_rosters(raw, games), ROSTERS_WEEKLY, [season])
