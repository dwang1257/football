from collections.abc import Iterable, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy.special import ndtr

from football.features.asof import latest_before
from football.features.columns import SIDES
from football.schemas import UTC_TS
from football.teams import FRANCHISES

KEY = "_k"

_FRANCHISE_MAP: dict[str, str] = {
    code: franchise for franchise, codes in FRANCHISES.items() for code in codes
}


def franchise_expr(col: str) -> pl.Expr:
    return pl.col(col).replace(_FRANCHISE_MAP)


def event_time(*cols: str) -> pl.Expr:
    return pl.max_horizontal([pl.col(col) for col in cols]).alias("available_at")


def last_per_time(history: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    return history.unique(subset=[*by, "available_at"], keep="last", maintain_order=True)


def latest_global(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    joined = latest_before(
        left.with_columns(pl.lit(1, pl.Int8).alias(KEY)),
        right.with_columns(pl.lit(1, pl.Int8).alias(KEY)),
        by=[KEY],
    )
    return joined.drop(KEY)


def long_sides(targets: pl.DataFrame, extra: Sequence[str] = ()) -> pl.DataFrame:
    frames = [
        targets.select(
            "game_id",
            "season",
            "cutoff",
            *extra,
            pl.lit(side).alias("side"),
            pl.col(f"{side}_fr").alias("team"),
            pl.col(f"{other}_fr").alias("opp"),
        )
        for side, other in (("home", "away"), ("away", "home"))
    ]
    return pl.concat(frames)


def widen(long: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    parts = [
        long.filter(pl.col("side") == side).select(
            "game_id", *(pl.col(col).alias(f"{side}_{col}") for col in cols)
        )
        for side in SIDES
    ]
    return parts[0].join(parts[1], on="game_id", how="full", coalesce=True)


def empty_like(targets: pl.DataFrame, columns: Iterable[str], dtype: pl.DataType) -> pl.DataFrame:
    return targets.select("game_id", *(pl.lit(None, dtype).alias(col) for col in columns))


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den).otherwise(None)


def normal_cdf(x: pl.Expr) -> pl.Expr:
    return x.map_batches(
        lambda s: pl.Series(ndtr(s.cast(pl.Float64).to_numpy()), dtype=pl.Float64).fill_nan(None),
        return_dtype=pl.Float64,
    )


def haversine_km(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    p1, p2 = lat1.radians(), lat2.radians()
    dlat = p2 - p1
    dlon = (lon2 - lon1).radians()
    a = (dlat / 2).sin().pow(2) + p1.cos() * p2.cos() * (dlon / 2).sin().pow(2)
    return 2 * 6371.0088 * a.sqrt().arcsin()


def utc_offset_hours(tz: pl.Series, at: pl.Series) -> pl.Series:
    cache: dict[str, ZoneInfo] = {}
    values: list[float | None] = []
    for name, moment in zip(tz.to_list(), at.to_list(), strict=True):
        if name is None or moment is None:
            values.append(None)
            continue
        zone = cache.setdefault(name, ZoneInfo(name))
        offset = moment.astimezone(zone).utcoffset()
        values.append(None if offset is None else offset.total_seconds() / 3600)
    return pl.Series(values, dtype=pl.Float64)


def decay_factor(half_life: float) -> float:
    return 0.5 ** (1.0 / half_life) if half_life > 0 else 0.0


def weighted_moments(
    w: np.ndarray, w2: np.ndarray, s1: np.ndarray, s2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(w > 0, s1 / w, np.nan)
        pop = np.maximum(s2 / w - mean * mean, 0.0)
        denom = w * w - w2
        var = np.where(denom > 1e-9 * w * w, pop * w * w / denom, np.nan)
    return mean, var


def utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("as_of must be timezone aware")
    return pl.Series([moment], dtype=UTC_TS).item()


def source_rank(col: str, priority: Sequence[str]) -> pl.Expr:
    name = pl.col(col).str.to_lowercase()
    rank: pl.Expr = pl.lit(len(priority), pl.Int32)
    for index, keyword in reversed(list(enumerate(priority))):
        rank = (
            pl.when(name.str.contains(keyword, literal=True))
            .then(pl.lit(index, pl.Int32))
            .otherwise(rank)
        )
    return rank
