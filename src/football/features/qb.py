from collections import Counter, deque
from dataclasses import dataclass

import polars as pl

from football.features.asof import all_before, latest_before
from football.features.columns import INFO_COLUMNS, QB_FEATURES
from football.features.common import (
    decay_factor,
    empty_like,
    event_time,
    franchise_expr,
    last_per_time,
    long_sides,
    widen,
)
from football.schemas import UTC_TS

OUT_STATUSES = ("out", "doubtful")


@dataclass(frozen=True)
class QbParams:
    half_life_games: float = 16.0
    prior_epa_per_dropback: float = -0.05
    prior_dropbacks: float = 150.0
    usual_window: int = 8


def qb_value_history(qb_games: pl.DataFrame, params: QbParams) -> pl.DataFrame:
    rows = (
        qb_games.filter(pl.col("player_id").is_not_null())
        .with_columns(event_time("available_at", "kickoff_utc"))
        .sort("available_at", "kickoff_utc", "game_id", "player_id")
    )
    d = decay_factor(params.half_life_games)
    sums: dict[str, tuple[float, float]] = {}
    epa_out: list[float] = []
    db_out: list[float] = []
    for player, dropbacks, epa in zip(
        rows["player_id"].to_list(),
        rows["dropbacks"].fill_null(0).to_list(),
        rows["qb_epa"].fill_null(0.0).to_list(),
        strict=True,
    ):
        s_epa, s_db = sums.get(player, (0.0, 0.0))
        s_epa, s_db = s_epa * d + float(epa), s_db * d + float(dropbacks)
        sums[player] = (s_epa, s_db)
        epa_out.append(s_epa)
        db_out.append(s_db)
    history = rows.select(
        pl.col("player_id").alias("qb_id"),
        pl.col("available_at").cast(UTC_TS),
        pl.Series("s_epa", epa_out, dtype=pl.Float64),
        pl.Series("s_db", db_out, dtype=pl.Float64),
    )
    return last_per_time(history, ["qb_id"])


def usual_starter_history(qb_games: pl.DataFrame, params: QbParams) -> pl.DataFrame:
    starts = (
        qb_games.filter(pl.col("is_starter") & pl.col("player_id").is_not_null())
        .with_columns(
            event_time("available_at", "kickoff_utc"), franchise_expr("team").alias("team")
        )
        .sort("available_at", "kickoff_utc", "game_id", "team", "player_id")
        .unique(["game_id", "team"], keep="first", maintain_order=True)
    )
    recent: dict[str, deque[str]] = {}
    usual: list[str] = []
    for team, player in zip(starts["team"].to_list(), starts["player_id"].to_list(), strict=True):
        window = recent.setdefault(team, deque(maxlen=params.usual_window))
        window.append(player)
        counts = Counter(window)
        top = max(counts.values())
        usual.append(next(p for p in reversed(window) if counts[p] == top))
    history = starts.select(
        "team", pl.col("available_at").cast(UTC_TS), pl.Series("usual_qb", usual, dtype=pl.String)
    )
    return last_per_time(history, ["team"])


def latest_report(queries: pl.DataFrame, injuries: pl.DataFrame) -> pl.DataFrame:
    rows = injuries.with_columns(franchise_expr("team").alias("team"))
    reports = rows.group_by("team", "season", pl.col("week").alias("report_week")).agg(
        pl.col("available_at").min()
    )
    matched = latest_before(queries, reports, by=["team", "season"]).drop("available_at")
    entries = rows.select(
        "team",
        "season",
        pl.col("week").alias("report_week"),
        "player_id",
        "position",
        pl.col("report_status").str.to_lowercase().alias("status"),
        "available_at",
    )
    return all_before(
        matched.drop_nulls("report_week"), entries, on=["team", "season", "report_week"]
    )


def out_players(queries: pl.DataFrame, injuries: pl.DataFrame | None) -> pl.DataFrame:
    if injuries is None:
        return pl.DataFrame(schema={"qid": pl.Int64(), "player_id": pl.String()})
    report = latest_report(queries, injuries)
    return report.filter(pl.col("status").is_in(OUT_STATUSES)).select("qid", "player_id")


def expected_starters(
    queries: pl.DataFrame,
    depth_charts: pl.DataFrame | None,
    injuries: pl.DataFrame | None,
) -> pl.DataFrame:
    schema = {"qid": pl.Int64(), "chart_qb": pl.String()}
    if depth_charts is None:
        return pl.DataFrame(schema=schema)
    qbs = depth_charts.filter(
        (pl.col("position").str.to_uppercase() == "QB") & pl.col("player_id").is_not_null()
    ).with_columns(franchise_expr("team").alias("team"))
    if qbs.is_empty():
        return pl.DataFrame(schema=schema)
    charts = qbs.group_by("team", "season", "captured_at").agg(pl.col("available_at").min())
    matched = latest_before(
        queries.select("qid", "team", "season", "cutoff"), charts, by=["team", "season"]
    )
    entries = qbs.select("team", "season", "captured_at", "player_id", "depth_rank", "available_at")
    candidates = all_before(
        matched.drop_nulls("captured_at").drop("available_at"),
        entries,
        on=["team", "season", "captured_at"],
    )
    healthy = candidates.join(out_players(queries, injuries), on=["qid", "player_id"], how="anti")
    return (
        healthy.sort("qid", "depth_rank", "player_id")
        .group_by("qid", maintain_order=True)
        .agg(pl.col("player_id").first().alias("chart_qb"))
    )


def qb_features(
    targets: pl.DataFrame,
    qb_games: pl.DataFrame | None,
    depth_charts: pl.DataFrame | None,
    injuries: pl.DataFrame | None,
    params: QbParams | None = None,
) -> pl.DataFrame:
    params = params or QbParams()
    if qb_games is None:
        return empty_like(targets, QB_FEATURES, pl.Float64()).join(
            empty_like(targets, INFO_COLUMNS, pl.String()), on="game_id"
        )
    long = long_sides(targets).with_columns(pl.int_range(pl.len()).alias("qid"))
    queries = long.select("qid", "team", "season", "cutoff")
    usual = latest_before(queries, usual_starter_history(qb_games, params), by=["team"]).select(
        "qid", "usual_qb"
    )
    chart = expected_starters(queries, depth_charts, injuries)
    picked = (
        long.join(usual, on="qid", how="left")
        .join(chart, on="qid", how="left")
        .with_columns(pl.coalesce("chart_qb", "usual_qb").alias("qb_id"))
    )
    valued = latest_before(
        picked.drop_nulls("qb_id"), qb_value_history(qb_games, params), by=["qb_id"]
    )
    prior = params.prior_epa_per_dropback * params.prior_dropbacks
    valued = valued.select(
        "qid",
        (
            (pl.col("s_epa").fill_null(0.0) + prior)
            / (pl.col("s_db").fill_null(0.0) + params.prior_dropbacks)
        ).alias("qb_value"),
    )
    long = picked.join(valued, on="qid", how="left").with_columns(
        pl.when(pl.col("usual_qb").is_null() | pl.col("qb_id").is_null())
        .then(None)
        .otherwise((pl.col("qb_id") != pl.col("usual_qb")).cast(pl.Float64))
        .alias("qb_change"),
        pl.col("qb_id").alias("exp_qb_id"),
    )
    wide = widen(long, ["qb_value", "qb_change", "exp_qb_id"])
    return wide.with_columns(
        (pl.col("home_qb_value") - pl.col("away_qb_value")).alias("qb_value_diff")
    ).select("game_id", *QB_FEATURES, *INFO_COLUMNS)
