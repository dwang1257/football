from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta

import polars as pl
import pytest

from football.features.build import compute_features
from football.features.inputs import Tables, make_targets
from football.features.market import market_features
from football.features.owner import owner_features
from football.features.qb import qb_features
from football.features.situational import situational_features
from football.features.weather import weather_features

MakeTables = Callable[..., Tables]
Required = Callable[[pl.DataFrame | None], pl.DataFrame]


def target(game_id: str) -> pl.Expr:
    return pl.col("game_id") == game_id


def pick(df: pl.DataFrame, game_id: str) -> dict[str, object]:
    return df.filter(target(game_id)).row(0, named=True)


def test_expected_starter_skips_qb_ruled_out(make_tables: MakeTables, required: Required) -> None:
    tables = make_tables(4)
    targets = make_targets(tables, None)
    game = targets.filter(pl.col("season") == 2021).row(3, named=True)
    team = game["home_team"]
    injuries = required(tables.injuries).with_columns(
        pl.when((pl.col("team") == team) & (pl.col("player_id") == f"{team}_QB1"))
        .then(pl.lit("Questionable"))
        .otherwise(pl.col("report_status"))
        .alias("report_status")
    )
    healthy = qb_features(targets, tables.qb_games, tables.depth_charts, injuries)
    assert pick(healthy, game["game_id"])["home_exp_qb_id"] == f"{team}_QB1"
    ruled_out = injuries.with_columns(
        pl.when(
            (pl.col("team") == team)
            & (pl.col("player_id") == f"{team}_QB1")
            & (pl.col("season") == game["season"])
            & (pl.col("week") == game["week"])
        )
        .then(pl.lit("Doubtful"))
        .otherwise(pl.col("report_status"))
        .alias("report_status")
    )
    out = qb_features(targets, tables.qb_games, tables.depth_charts, ruled_out)
    assert pick(out, game["game_id"])["home_exp_qb_id"] == f"{team}_QB2"


def test_market_prefers_close_and_home_favorite_is_positive(
    make_tables: MakeTables, required: Required
) -> None:
    tables = make_tables(2)
    targets = make_targets(tables, None)
    game_id = targets["game_id"][10]
    lines = (
        required(tables.lines)
        .filter(target(game_id))
        .with_columns(
            pl.when(pl.col("kind") == "close")
            .then(pl.when(pl.col("source") == "lines_espn").then(7.0).otherwise(1.0))
            .when(pl.col("kind") == "open")
            .then(3.0)
            .otherwise(-5.0)
            .alias("home_spread")
        )
    )
    row = pick(market_features(targets, lines), game_id)
    assert row["market_home_spread"] == 7.0
    assert row["market_spread_move"] == 4.0
    assert row["market_home_prob"] == pytest.approx(0.69795, abs=1e-4)
    early = make_targets(
        tables, targets.filter(target(game_id))["kickoff_utc"][0] - timedelta(hours=1)
    )
    assert pick(market_features(early, lines), game_id)["market_home_spread"] == -5.0


def test_weather_is_neutral_in_domes_and_flags_cold(
    make_tables: MakeTables, required: Required
) -> None:
    tables = make_tables(3)
    targets = make_targets(tables, None)
    forecasts = required(tables.weather_forecasts).with_columns(
        pl.lit(20.0).alias("temp_f"), pl.lit(18.0).alias("wind_mph")
    )
    out = weather_features(targets, forecasts, tables.team_homes).join(
        targets.select("game_id", "roof_type", "home_fr", "away_fr"), on="game_id"
    )
    dome = out.filter(pl.col("roof_type") == "dome").row(0, named=True)
    assert (dome["fcst_temp_f"], dome["fcst_wind_mph"], dome["cold_game"]) == (70.0, 0.0, 0.0)
    outdoor = out.filter((pl.col("roof_type") == "outdoors") & (pl.col("away_fr") == "LAC")).row(
        0, named=True
    )
    assert (outdoor["cold_game"], outdoor["wind_15plus"], outdoor["away_dome_team_in_cold"]) == (
        1.0,
        1.0,
        1.0,
    )


def test_situational_travel_and_body_clock(make_tables: MakeTables, required: Required) -> None:
    tables = make_tables(5)
    targets = make_targets(tables, None)
    out = situational_features(targets, tables.team_homes).join(targets, on="game_id")
    home_games = out.filter(pl.col("neutral_site") == 0.0)
    assert home_games["travel_km_home"].max() == pytest.approx(0.0)
    sf_at_buf = out.filter((pl.col("away_team") == "SF") & (pl.col("home_team") == "BUF"))
    if sf_at_buf.height:
        row = sf_at_buf.row(0, named=True)
        assert row["travel_km_away"] == pytest.approx(3600, rel=0.05)
        assert row["tz_shift_away"] == 3.0
        early = row["kickoff_utc"].hour == 17
        assert row["west_coast_early"] == (1.0 if early else 0.0)
    london = out.filter(pl.col("is_international") == 1.0)
    assert london.height
    assert london.select((pl.col("travel_km_away") > 5000).all()).item()


def test_owner_season_record_and_pythag(make_tables: MakeTables, required: Required) -> None:
    tables = make_tables(6)
    targets = make_targets(tables, None)
    out = owner_features(targets, tables.games, tables.team_games, None, None).join(
        targets, on="game_id"
    )
    games = tables.games.filter(pl.col("home_score").is_not_null())
    late = out.filter((pl.col("season") == 2021) & (pl.col("week") == 4)).row(0, named=True)
    team = late["home_team"]
    prior = games.filter((pl.col("season") == 2021) & (pl.col("week") < 4))
    home = prior.filter(pl.col("home_team") == team)
    away = prior.filter(pl.col("away_team") == team)
    scores = pl.concat(
        [
            home.select(pl.col("home_score").alias("pf"), pl.col("away_score").alias("pa")),
            away.select(pl.col("away_score").alias("pf"), pl.col("home_score").alias("pa")),
        ]
    ).cast(pl.Float64)
    pf = float(scores["pf"].sum())
    pa = float(scores["pa"].sum())
    wins = scores.select(
        (
            (pl.col("pf") > pl.col("pa")).cast(pl.Float64) + 0.5 * (pl.col("pf") == pl.col("pa"))
        ).sum()
    ).item()
    assert late["home_season_win_pct"] == pytest.approx(wins / 3)
    assert late["home_pd_per_game"] == pytest.approx((pf - pa) / 3)
    assert late["home_pythag"] == pytest.approx(pf**2.37 / (pf**2.37 + pa**2.37))


def test_missing_optional_tables_give_null_columns(
    make_tables: MakeTables, required: Required
) -> None:
    tables = make_tables(7)
    bare = replace(
        tables,
        **{name: None for name, _ in tables.items() if name not in ("games", "team_games")},
    )
    out = compute_features(bare)
    assert out.height == tables.games.height
    for col in ("market_home_prob", "home_qb_value", "home_inj_ol", "travel_km_away"):
        assert out[col].null_count() == out.height
    outdoor = out.join(tables.games, on="game_id").filter(pl.col("roof") == "outdoors")
    assert outdoor["fcst_temp_f"].null_count() == outdoor.height
    assert out["elo_home_prob"].null_count() == 0
