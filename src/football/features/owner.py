from collections.abc import Sequence

import polars as pl

from football.features.asof import all_before, latest_before
from football.features.columns import OWNER_FEATURES, WIN_TOTAL_FEATURES
from football.features.common import event_time, franchise_expr, last_per_time, long_sides, widen
from football.features.elo import results
from football.features.weather import COLD_F, WINDY_MPH

PYTHAG_EXPONENT = 2.37
INDOOR_PAST_ROOFS = ("dome", "closed")


def cumulative(events: pl.DataFrame, keys: Sequence[str], values: Sequence[str]) -> pl.DataFrame:
    ordered = events.drop_nulls(list(keys)).sort("available_at", "game_id")
    summed = ordered.select(
        *keys,
        "available_at",
        *(pl.col(v).cum_sum().over(list(keys)).alias(v) for v in values),
    )
    return last_per_time(summed, keys)


def team_results(games: pl.DataFrame) -> pl.DataFrame:
    played = results(games)
    win = (
        pl.when(pl.col("margin") > 0).then(1.0).when(pl.col("margin") < 0).then(0.0).otherwise(0.5)
    )
    common = (
        "game_id",
        "season",
        "stadium_id",
        "available_at",
        "roof",
        "temp_observed",
        "wind_observed",
    )
    frames = []
    for side, other, sign in (("home", "away", 1.0), ("away", "home", -1.0)):
        frames.append(
            played.select(
                *common,
                pl.col(f"{side}_fr").alias("team"),
                pl.col(f"{other}_fr").alias("opp"),
                pl.col(f"{side}_coach").alias("coach"),
                (win if sign > 0 else 1.0 - win).alias("win"),
                pl.col(f"{side}_score").cast(pl.Float64).alias("pf"),
                pl.col(f"{other}_score").cast(pl.Float64).alias("pa"),
                pl.lit(1.0).alias("games"),
            )
        )
    return pl.concat(frames)


def _laplace(wins: str, games: str) -> pl.Expr:
    return (pl.col(wins).fill_null(0.0) + 1.0) / (pl.col(games).fill_null(0.0) + 2.0)


def _record(
    queries: pl.DataFrame, events: pl.DataFrame, keys: Sequence[str], prefix: str
) -> pl.DataFrame:
    hist = cumulative(events, keys, ["win", "games"])
    found = latest_before(
        queries.select("qid", *keys, "cutoff").drop_nulls(list(keys)), hist, by=keys
    )
    return found.select(
        "qid",
        _laplace("win", "games").alias(f"{prefix}_win_pct"),
        pl.col("games").alias(f"{prefix}_games"),
    )


def _weather_events(tr: pl.DataFrame, observed: pl.DataFrame | None) -> pl.DataFrame:
    outdoor = ~pl.col("roof").fill_null("").str.to_lowercase().is_in(INDOOR_PAST_ROOFS)

    def flags(temp: str, wind: str, suffix: str = "") -> list[pl.Expr]:
        return [
            (outdoor & (pl.col(temp) < COLD_F))
            .fill_null(False)
            .cast(pl.Float64)
            .alias(f"cold{suffix}"),
            (outdoor & (pl.col(wind) > WINDY_MPH))
            .fill_null(False)
            .cast(pl.Float64)
            .alias(f"windy{suffix}"),
        ]

    base = tr.with_columns(flags("temp_observed", "wind_observed"))
    events = [base]
    if observed is not None:
        wx = (
            observed.sort("available_at", "source")
            .unique("game_id", keep="first", maintain_order=True)
            .select(
                "game_id",
                pl.col("temp_f").cast(pl.Float64).alias("wx_temp"),
                pl.col("wind_mph").cast(pl.Float64).alias("wx_wind"),
                pl.col("available_at").alias("wx_at"),
            )
        )
        revised = (
            base.join(wx, on="game_id", how="inner")
            .with_columns(flags("wx_temp", "wx_wind", "_wx"))
            .with_columns(
                (pl.col("cold_wx") - pl.col("cold")).alias("cold"),
                (pl.col("windy_wx") - pl.col("windy")).alias("windy"),
                event_time("available_at", "wx_at"),
            )
        )
        events.append(revised.select(base.columns))
    return pl.concat(events).with_columns(
        (pl.col("cold") * pl.col("win")).alias("cold_wins"),
        (pl.col("windy") * pl.col("win")).alias("windy_wins"),
    )


def _fumble_events(team_games: pl.DataFrame) -> pl.DataFrame:
    rows = team_games.select(
        "game_id",
        "season",
        "kickoff_utc",
        "available_at",
        "fumbles_lost",
        franchise_expr("team").alias("team"),
        franchise_expr("opponent").alias("opp"),
    )
    opp = rows.select(
        "game_id",
        pl.col("team").alias("opp"),
        pl.col("fumbles_lost").alias("opp_fumbles_lost"),
        pl.col("available_at").alias("opp_at"),
    )
    return rows.join(opp, on=["game_id", "opp"], how="inner").select(
        "game_id",
        "team",
        "season",
        event_time("available_at", "opp_at", "kickoff_utc"),
        (pl.col("opp_fumbles_lost") - pl.col("fumbles_lost"))
        .cast(pl.Float64)
        .alias("fumble_margin"),
        pl.lit(1.0).alias("fumble_games"),
    )


def _season_record(queries: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    hist = cumulative(tr, ["team", "season"], ["win", "games", "pf", "pa"])
    found = latest_before(
        queries.select("qid", "team", "season", "cutoff"), hist, by=["team", "season"]
    )
    pf = pl.col("pf").pow(PYTHAG_EXPONENT)
    pa = pl.col("pa").pow(PYTHAG_EXPONENT)
    return found.select(
        "qid",
        (pl.col("win") / pl.col("games")).alias("season_win_pct"),
        pl.when(pf + pa > 0).then(pf / (pf + pa)).otherwise(0.5).alias("pythag"),
        ((pl.col("pf") - pl.col("pa")) / pl.col("games")).alias("pd_per_game"),
    )


def _fumble_luck(queries: pl.DataFrame, team_games: pl.DataFrame) -> pl.DataFrame:
    hist = cumulative(
        _fumble_events(team_games), ["team", "season"], ["fumble_margin", "fumble_games"]
    )
    found = latest_before(
        queries.select("qid", "team", "season", "cutoff"), hist, by=["team", "season"]
    )
    return found.select(
        "qid", (pl.col("fumble_margin") / pl.col("fumble_games")).alias("fumble_luck")
    )


def _win_totals(queries: pl.DataFrame, win_totals: pl.DataFrame | None) -> pl.DataFrame:
    if win_totals is None:
        return pl.DataFrame(schema={"qid": pl.Int64(), "win_total": pl.Float64()})
    rows = win_totals.select(
        franchise_expr("team").alias("team"),
        pl.col("season").cast(pl.Int32),
        "source",
        pl.col("win_total").cast(pl.Float64),
        "available_at",
    ).drop_nulls("win_total")
    found = all_before(
        queries.select("qid", "team", "season", "cutoff"), rows, on=["team", "season"]
    )
    return (
        found.sort("qid", "available_at", "source")
        .group_by("qid", maintain_order=True)
        .agg(pl.col("win_total").last())
    )


def owner_features(
    targets: pl.DataFrame,
    games: pl.DataFrame,
    team_games: pl.DataFrame,
    weather_observed: pl.DataFrame | None,
    win_totals: pl.DataFrame | None,
) -> pl.DataFrame:
    long = (
        long_sides(targets, extra=["stadium_id", "home_coach", "away_coach"])
        .with_columns(
            pl.when(pl.col("side") == "home")
            .then(pl.col("home_coach"))
            .otherwise(pl.col("away_coach"))
            .alias("coach"),
        )
        .with_columns(pl.int_range(pl.len()).alias("qid"))
    )
    tr = team_results(games)
    wx = _weather_events(tr, weather_observed)
    parts = [
        _season_record(long, tr),
        _record(long, tr, ["team", "stadium_id"], "stadium"),
        _record(long, tr, ["coach", "opp"], "coach_vs_opp"),
        _fumble_luck(long, team_games),
        _win_totals(long, win_totals),
    ]
    for flag in ("cold", "windy"):
        hist = cumulative(wx, ["team"], [flag, f"{flag}_wins"])
        found = latest_before(long.select("qid", "team", "cutoff"), hist, by=["team"])
        parts.append(found.select("qid", _laplace(f"{flag}_wins", flag).alias(f"{flag}_win_pct")))
    for part in parts:
        long = long.join(part, on="qid", how="left")
    cols = [
        "season_win_pct",
        "pythag",
        "pd_per_game",
        "stadium_win_pct",
        "stadium_games",
        "coach_vs_opp_win_pct",
        "coach_vs_opp_games",
        "cold_win_pct",
        "windy_win_pct",
        "fumble_luck",
        "win_total",
    ]
    wide = widen(long, cols)
    return wide.with_columns(
        pl.col("home_stadium_games").fill_null(0.0),
        pl.col("home_coach_vs_opp_games").fill_null(0.0),
        pl.col("away_coach_vs_opp_games").fill_null(0.0),
        pl.col("home_stadium_win_pct").fill_null(0.5),
        pl.col("home_coach_vs_opp_win_pct").fill_null(0.5),
        pl.col("away_coach_vs_opp_win_pct").fill_null(0.5),
        (pl.col("home_season_win_pct") - pl.col("away_season_win_pct")).alias(
            "season_win_pct_diff"
        ),
        (pl.col("home_pythag") - pl.col("away_pythag")).alias("pythag_diff"),
        (pl.col("home_pd_per_game") - pl.col("away_pd_per_game")).alias("pd_per_game_diff"),
        (pl.col("home_win_total") - pl.col("away_win_total")).alias("win_total_diff"),
    ).select(
        "game_id", *(pl.col(c).cast(pl.Float64) for c in (*OWNER_FEATURES, *WIN_TOTAL_FEATURES))
    )
