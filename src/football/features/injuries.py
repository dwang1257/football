import polars as pl

from football.features.asof import latest_before
from football.features.columns import INJURY_FEATURES
from football.features.common import (
    decay_factor,
    empty_like,
    event_time,
    last_per_time,
    long_sides,
    widen,
)
from football.features.qb import OUT_STATUSES, latest_report
from football.schemas import UTC_TS

GROUPS: tuple[str, ...] = ("ol", "db", "other")
OL_POSITIONS = ("T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG")
DB_POSITIONS = ("CB", "S", "FS", "SS", "DB", "SAF", "NB")
SNAP_HALF_LIFE_GAMES = 4.0
UNKNOWN_SNAP_SHARE = 0.5


def snap_share_history(snap_counts: pl.DataFrame) -> pl.DataFrame:
    rows = snap_counts.filter(pl.col("player_id").is_not_null())
    share = pl.max_horizontal(
        pl.col("offense_pct").cast(pl.Float64).fill_null(0.0),
        pl.col("defense_pct").cast(pl.Float64).fill_null(0.0),
    )
    fraction = pl.when(share > 1.0).then(share / 100.0).otherwise(share).clip(0.0, 1.0)
    rows = rows.with_columns(fraction.alias("share"), event_time("available_at")).sort(
        "available_at", "game_id", "player_id"
    )
    d = decay_factor(SNAP_HALF_LIFE_GAMES)
    sums: dict[str, tuple[float, float]] = {}
    out: list[float] = []
    for player, value in zip(rows["player_id"].to_list(), rows["share"].to_list(), strict=True):
        s, w = sums.get(player, (0.0, 0.0))
        s, w = s * d + float(value), w * d + 1.0
        sums[player] = (s, w)
        out.append(s / w)
    history = rows.select(
        "player_id",
        pl.col("available_at").cast(UTC_TS),
        pl.Series("snap_share", out, dtype=pl.Float64),
    )
    return last_per_time(history, ["player_id"])


def _group(position: pl.Expr) -> pl.Expr:
    pos = position.str.to_uppercase()
    return (
        pl.when(pos.is_in(OL_POSITIONS))
        .then(pl.lit("ol"))
        .when(pos.is_in(DB_POSITIONS))
        .then(pl.lit("db"))
        .otherwise(pl.lit("other"))
    )


def injury_features(
    targets: pl.DataFrame,
    injuries: pl.DataFrame | None,
    snap_counts: pl.DataFrame | None,
) -> pl.DataFrame:
    if injuries is None:
        return empty_like(targets, INJURY_FEATURES, pl.Float64())
    long = long_sides(targets).with_columns(pl.int_range(pl.len()).alias("qid"))
    report = latest_report(long.select("qid", "team", "season", "cutoff"), injuries)
    reported = report.select("qid").unique()
    hurt = report.filter(
        pl.col("status").is_in(OUT_STATUSES)
        & (pl.col("position").fill_null("").str.to_uppercase() != "QB")
    ).drop("available_at")
    if snap_counts is not None:
        hurt = latest_before(hurt, snap_share_history(snap_counts), by=["player_id"])
    else:
        hurt = hurt.with_columns(pl.lit(None, pl.Float64).alias("snap_share"))
    weights = hurt.group_by("qid").agg(
        pl.col("snap_share")
        .fill_null(UNKNOWN_SNAP_SHARE)
        .filter(_group(pl.col("position").fill_null("")) == group)
        .sum()
        .alias(f"inj_{group}")
        for group in GROUPS
    )
    counted = reported.join(weights, on="qid", how="left").with_columns(
        pl.col(f"inj_{group}").fill_null(0.0) for group in GROUPS
    )
    long = long.join(counted, on="qid", how="left")
    return widen(long, [f"inj_{group}" for group in GROUPS]).select("game_id", *INJURY_FEATURES)
