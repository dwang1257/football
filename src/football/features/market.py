import polars as pl

from football.features.asof import all_before, latest_before
from football.features.columns import BENCHMARK_COLUMNS, LINE_FEATURES
from football.features.common import empty_like, franchise_expr, normal_cdf, source_rank

SPREAD_SD = 13.5
LINE_SOURCE_PRIORITY: tuple[str, ...] = (
    "consensus",
    "nflverse",
    "huggingface",
    "hf",
    "espn",
    "action",
    "sbro",
    "nfelo",
    "nfldata",
)
KIND_PRIORITY: tuple[str, ...] = ("close",)
NFELO_SOURCES: tuple[str, ...] = ("nfelo",)
ELO538_SOURCES: tuple[str, ...] = ("538", "fivethirtyeight")
FPI_SOURCES: tuple[str, ...] = ("fpi",)


def _consensus(rows: pl.DataFrame, latest: bool) -> pl.DataFrame:
    order = pl.col("captured_at")
    per_book = (
        rows.sort("game_id", "book", "captured_at", descending=[False, False, latest])
        .group_by("game_id", "book", maintain_order=True)
        .first()
    )
    return per_book.group_by("game_id").agg(
        pl.col("home_spread").median(), pl.col("total").median(), order.max()
    )


def _best_group(rows: pl.DataFrame) -> pl.DataFrame:
    score = pl.col("kind_rank") * 1000 + pl.col("source_rank")
    return rows.filter(score == score.min().over("game_id"))


def market_features(targets: pl.DataFrame, lines: pl.DataFrame | None) -> pl.DataFrame:
    if lines is None:
        return empty_like(targets, LINE_FEATURES, pl.Float64())
    ranked = lines.filter(pl.col("home_spread").is_not_null()).with_columns(
        source_rank("kind", KIND_PRIORITY).alias("kind_rank"),
        source_rank("source", LINE_SOURCE_PRIORITY).alias("source_rank"),
        pl.col("book").fill_null(""),
    )
    rows = all_before(targets.select("game_id", "cutoff"), ranked, on=["game_id"])
    current = _consensus(_best_group(rows), latest=True)
    opens = rows.filter(pl.col("kind") == "open").with_columns(
        pl.lit(0, pl.Int32).alias("kind_rank")
    )
    opening = _consensus(_best_group(opens), latest=False).select(
        "game_id", pl.col("home_spread").alias("open_spread")
    )
    spread = pl.col("market_home_spread")
    return (
        targets.select("game_id")
        .join(current, on="game_id", how="left")
        .join(opening, on="game_id", how="left")
        .with_columns(
            pl.col("home_spread").alias("market_home_spread"),
            pl.col("total").alias("market_total"),
        )
        .with_columns(
            normal_cdf(spread / SPREAD_SD).alias("market_home_prob"),
            (spread - pl.col("open_spread")).alias("market_spread_move"),
        )
        .select("game_id", *LINE_FEATURES)
    )


def _latest_by_source(
    targets: pl.DataFrame,
    rows: pl.DataFrame | None,
    keywords: tuple[str, ...],
    value: str,
    by: str,
) -> pl.DataFrame:
    if rows is None:
        return pl.DataFrame(schema={"game_id": pl.String(), value: pl.Float64()})
    picked = rows.filter(source_rank("source", keywords) < len(keywords))
    picked = picked.sort("available_at").unique(
        [by, "available_at"], keep="last", maintain_order=True
    )
    return latest_before(
        targets, picked.select(by, "available_at", pl.col(value).cast(pl.Float64)), by=[by]
    )


def benchmark_columns(
    targets: pl.DataFrame,
    benchmarks: pl.DataFrame | None,
    team_ratings: pl.DataFrame | None,
) -> pl.DataFrame:
    games = targets.select("game_id", "cutoff")
    out = games.select("game_id")
    for name, keywords in (
        ("bench_nfelo_prob", NFELO_SOURCES),
        ("bench_elo538_prob", ELO538_SOURCES),
    ):
        found = _latest_by_source(games, benchmarks, keywords, "home_win_prob", "game_id")
        out = out.join(
            found.select("game_id", pl.col("home_win_prob").alias(name)), on="game_id", how="left"
        )
    if team_ratings is None:
        return out.with_columns(pl.lit(None, pl.Float64).alias("bench_fpi_diff")).select(
            "game_id", *BENCHMARK_COLUMNS
        )
    ratings = team_ratings.with_columns(franchise_expr("team").alias("team"))
    sides = []
    for side in ("home", "away"):
        query = targets.select("game_id", "cutoff", pl.col(f"{side}_fr").alias("team"))
        found = _latest_by_source(query, ratings, FPI_SOURCES, "rating", "team")
        sides.append(found.select("game_id", pl.col("rating").alias(f"{side}_fpi")))
    return (
        out.join(sides[0], on="game_id", how="left")
        .join(sides[1], on="game_id", how="left")
        .with_columns((pl.col("home_fpi") - pl.col("away_fpi")).alias("bench_fpi_diff"))
        .select("game_id", *BENCHMARK_COLUMNS)
    )
