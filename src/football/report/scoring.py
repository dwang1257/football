import math
from dataclasses import dataclass

import polars as pl

from football.predict.store import BLENDED_STORED, PredictError, clean_games, read_table
from football.schemas import PREDICTIONS_MODEL

PROB_EPS = 1e-15
CONF_LABELS: tuple[str, ...] = ("1", "2", "3", "4", "5", "prob")


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    log_loss: float
    brier: float


@dataclass(frozen=True)
class Paired:
    n: int
    mean: float
    se: float


@dataclass(frozen=True)
class Scorecard:
    label: str
    games: int
    model: Metrics
    blended: Metrics
    opinion_games: int
    model_on_opinions: Metrics
    blended_on_opinions: Metrics
    owner: Metrics
    flips_won: int
    flips_lost: int
    paired: Paired


def _home_pick(prob: pl.Expr) -> pl.Expr:
    return prob >= 0.5


def scored_games(season: int | None = None) -> pl.DataFrame:
    model = read_table(PREDICTIONS_MODEL.name)
    if model is None:
        raise PredictError("no stored predictions; run football predict first")
    blended = read_table(BLENDED_STORED.name)
    if blended is None:
        blended = pl.DataFrame(schema=BLENDED_STORED.columns)
    games = clean_games().filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if season is not None:
        games = games.filter(pl.col("season") == season)
    chosen = (
        model.join(games.select("game_id"), on="game_id", how="semi")
        .sort("is_final", "created_at", descending=True)
        .unique("game_id", keep="first")
        .select("run_id", "game_id", "model", "is_final", "home_win_prob")
    )
    chosen = chosen.join(
        blended.drop("model_prob", "created_at", "is_final"), on=["run_id", "game_id"], how="left"
    )
    diff = pl.col("home_score") - pl.col("away_score")
    return (
        chosen.join(
            games.select(
                "game_id",
                "season",
                "week",
                "kickoff_utc",
                "away_team",
                "home_team",
                "home_score",
                "away_score",
            ),
            on="game_id",
        )
        .with_columns(
            model_prob=pl.col("home_win_prob"),
            blended_prob=pl.coalesce("blended_prob", "home_win_prob"),
            home_win=pl.when(diff > 0).then(1.0).when(diff < 0).then(0.0).otherwise(0.5),
            team_nudge_points=pl.col("team_nudge_points").fill_null(0.0),
            flipped=pl.col("flipped").fill_null(False),
            capped=pl.col("capped").fill_null(False),
        )
        .with_columns(
            user_picked_home=pl.col("user_pick") == pl.col("home_team"),
            has_opinion=pl.col("user_prob").is_not_null() | (pl.col("team_nudge_points") != 0.0),
        )
        .sort("kickoff_utc", "game_id")
    )


def metrics(df: pl.DataFrame, prob_col: str) -> Metrics:
    if df.is_empty():
        return Metrics(0, math.nan, math.nan, math.nan)
    y = pl.col("home_win")
    p = pl.col(prob_col).clip(PROB_EPS, 1.0 - PROB_EPS)
    correct = (_home_pick(pl.col(prob_col)) == (y == 1.0)).cast(pl.Float64)
    row = df.select(
        accuracy=correct.filter(y != 0.5).mean(),
        log_loss=-(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean(),
        brier=((pl.col(prob_col) - y) ** 2).mean(),
    ).row(0, named=True)
    accuracy = row["accuracy"]
    return Metrics(
        n=df.height,
        accuracy=math.nan if accuracy is None else float(accuracy),
        log_loss=float(row["log_loss"]),
        brier=float(row["brier"]),
    )


def _log_losses(prob_col: str) -> pl.Expr:
    y = pl.col("home_win")
    p = pl.col(prob_col).clip(PROB_EPS, 1.0 - PROB_EPS)
    return -(y * p.log() + (1.0 - y) * (1.0 - p).log())


def paired_log_loss(df: pl.DataFrame) -> Paired:
    if df.is_empty():
        return Paired(0, math.nan, math.nan)
    diff = _log_losses("blended_prob") - _log_losses("model_prob")
    row = df.select(mean=diff.mean(), std=diff.std(ddof=1)).row(0, named=True)
    se = math.nan if row["std"] is None else row["std"] / math.sqrt(df.height)
    return Paired(df.height, float(row["mean"]), se)


def blended_correct() -> pl.Expr:
    return _home_pick(pl.col("blended_prob")) == (pl.col("home_win") == 1.0)


def flip_ledger(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("flipped")).with_columns(
        outcome=pl.when(pl.col("home_win") == 0.5)
        .then(pl.lit("tie"))
        .when(blended_correct())
        .then(pl.lit("won"))
        .otherwise(pl.lit("lost"))
    )


def calibration(df: pl.DataFrame) -> pl.DataFrame:
    opinions = df.filter(pl.col("user_pick").is_not_null())
    hit = (
        pl.when(pl.col("user_picked_home"))
        .then(pl.col("home_win"))
        .otherwise(1.0 - pl.col("home_win"))
    )
    level = pl.col("user_conf").cast(pl.String).fill_null("prob")
    summary = opinions.group_by(level.alias("level")).agg(
        n=pl.len().cast(pl.Int64),
        assumed=pl.col("user_pick_prob").mean(),
        actual=hit.mean(),
    )
    grid = pl.DataFrame({"level": list(CONF_LABELS)})
    return (
        grid.join(summary, on="level", how="left")
        .with_columns(pl.col("n").fill_null(0))
        .filter(pl.col("n") > 0)
    )


def scorecard(df: pl.DataFrame, label: str) -> Scorecard:
    opinions = df.filter(pl.col("has_opinion"))
    picks = df.filter(pl.col("user_prob").is_not_null())
    flips = flip_ledger(df)
    return Scorecard(
        label=label,
        games=df.height,
        model=metrics(df, "model_prob"),
        blended=metrics(df, "blended_prob"),
        opinion_games=opinions.height,
        model_on_opinions=metrics(opinions, "model_prob"),
        blended_on_opinions=metrics(opinions, "blended_prob"),
        owner=metrics(picks, "user_prob"),
        flips_won=flips.filter(pl.col("outcome") == "won").height,
        flips_lost=flips.filter(pl.col("outcome") == "lost").height,
        paired=paired_log_loss(opinions),
    )
