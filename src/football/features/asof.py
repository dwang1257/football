from collections.abc import Sequence
from datetime import datetime

import polars as pl


def before(df: pl.DataFrame, cutoff: datetime, time_col: str = "available_at") -> pl.DataFrame:
    return df.filter(pl.col(time_col) < cutoff)


def latest_before(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    by: Sequence[str],
    cutoff_col: str = "cutoff",
    time_col: str = "available_at",
) -> pl.DataFrame:
    return left.sort(cutoff_col).join_asof(
        right.sort(time_col),
        left_on=cutoff_col,
        right_on=time_col,
        by=list(by),
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )


def all_before(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: Sequence[str],
    cutoff_col: str = "cutoff",
    time_col: str = "available_at",
) -> pl.DataFrame:
    return left.join(right, on=list(on), how="inner").filter(pl.col(time_col) < pl.col(cutoff_col))
