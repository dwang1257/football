import math
from dataclasses import dataclass

import polars as pl

from football.features.asof import latest_before
from football.features.columns import ELO_FEATURES
from football.features.common import event_time, franchise_expr, last_per_time, long_sides, widen
from football.schemas import UTC_TS

BYE_REST_DAYS = 13


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_field: float = 48.0
    mean: float = 1505.0
    initial: float = 1505.0
    season_regression: float = 1.0 / 3.0
    bye_bonus: float = 0.0
    scale: float = 400.0
    mov: bool = True


def expected(diff: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def mov_multiplier(margin: float, winner_diff: float) -> float:
    return math.log(max(abs(margin), 1.0) + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2)


def regress(rating: float, seasons: int, params: EloParams) -> float:
    return params.mean + (rating - params.mean) * (1.0 - params.season_regression) ** seasons


def game_diff(
    home: float, away: float, neutral: bool, home_bye: bool, away_bye: bool, params: EloParams
) -> float:
    field = 0.0 if neutral else params.home_field
    bye = params.bye_bonus * (int(home_bye) - int(away_bye))
    return home - away + field + bye


def update(
    home: float, away: float, diff: float, margin: float, params: EloParams
) -> tuple[float, float]:
    result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    winner_diff = diff if margin > 0 else -diff if margin < 0 else 0.0
    multiplier = mov_multiplier(margin, winner_diff) if params.mov else 1.0
    shift = params.k * multiplier * (result - expected(diff, params.scale))
    return home + shift, away - shift


def results(games: pl.DataFrame) -> pl.DataFrame:
    return (
        games.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
        .with_columns(
            event_time("available_at", "kickoff_utc"),
            franchise_expr("home_team").alias("home_fr"),
            franchise_expr("away_team").alias("away_fr"),
            (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64).alias("margin"),
            pl.col("neutral_site").fill_null(False),
            (pl.col("home_rest").fill_null(7) >= BYE_REST_DAYS).alias("home_bye"),
            (pl.col("away_rest").fill_null(7) >= BYE_REST_DAYS).alias("away_bye"),
        )
        .sort("available_at", "kickoff_utc", "game_id")
    )


def elo_history(games: pl.DataFrame, params: EloParams) -> pl.DataFrame:
    state: dict[str, tuple[float, int]] = {}
    teams: list[str] = []
    times: list[object] = []
    seasons: list[int] = []
    ratings: list[float] = []

    def pregame(team: str, season: int) -> float:
        if team not in state:
            return params.initial
        rating, last = state[team]
        return regress(rating, max(season - last, 0), params)

    for row in results(games).iter_rows(named=True):
        season = int(row["season"])
        home = pregame(row["home_fr"], season)
        away = pregame(row["away_fr"], season)
        diff = game_diff(home, away, row["neutral_site"], row["home_bye"], row["away_bye"], params)
        new_home, new_away = update(home, away, diff, row["margin"], params)
        for team, rating in ((row["home_fr"], new_home), (row["away_fr"], new_away)):
            state[team] = (rating, season)
            teams.append(team)
            times.append(row["available_at"])
            seasons.append(season)
            ratings.append(rating)
    history = pl.DataFrame(
        {"team": teams, "available_at": times, "elo_season": seasons, "elo": ratings},
        schema={
            "team": pl.String(),
            "available_at": UTC_TS,
            "elo_season": pl.Int32(),
            "elo": pl.Float64(),
        },
    )
    return last_per_time(history, ["team"])


def elo_features(targets: pl.DataFrame, history: pl.DataFrame, params: EloParams) -> pl.DataFrame:
    long = latest_before(long_sides(targets), history, by=["team"])
    gap = (pl.col("season") - pl.col("elo_season")).clip(lower_bound=0)
    shrink = pl.lit(1.0 - params.season_regression).pow(gap)
    long = long.with_columns(
        pl.when(pl.col("elo").is_null())
        .then(pl.lit(params.initial))
        .otherwise(params.mean + (pl.col("elo") - params.mean) * shrink)
        .alias("elo")
    )
    wide = widen(long, ["elo"])
    context = targets.select(
        "game_id",
        pl.col("neutral_site").fill_null(False).alias("_neutral"),
        (pl.col("home_rest").fill_null(7) >= BYE_REST_DAYS).cast(pl.Float64).alias("_hb"),
        (pl.col("away_rest").fill_null(7) >= BYE_REST_DAYS).cast(pl.Float64).alias("_ab"),
    )
    field = pl.when(pl.col("_neutral")).then(0.0).otherwise(params.home_field)
    diff = (
        pl.col("home_elo")
        - pl.col("away_elo")
        + field
        + params.bye_bonus * (pl.col("_hb") - pl.col("_ab"))
    )
    return (
        wide.join(context, on="game_id", how="left")
        .with_columns(diff.alias("elo_diff"))
        .with_columns(
            (1.0 / (1.0 + pl.lit(10.0).pow(-pl.col("elo_diff") / params.scale))).alias(
                "elo_home_prob"
            )
        )
        .select("game_id", *ELO_FEATURES)
    )


def run_elo(
    games: pl.DataFrame, params: EloParams | None = None, targets: pl.DataFrame | None = None
) -> pl.DataFrame:
    params = params or EloParams()
    if targets is None:
        targets = games.select(
            "game_id",
            "season",
            "neutral_site",
            "home_rest",
            "away_rest",
            pl.col("kickoff_utc").alias("cutoff"),
            franchise_expr("home_team").alias("home_fr"),
            franchise_expr("away_team").alias("away_fr"),
        )
    return elo_features(targets, elo_history(games, params), params)
