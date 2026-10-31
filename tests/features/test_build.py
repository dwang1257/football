from collections.abc import Callable
from datetime import timedelta

import polars as pl
from typer.testing import CliRunner

from football import storage
from football.cli import app
from football.features.build import FEATURES_FULL, build_features
from football.features.columns import (
    FEATURE_COLUMNS,
    INDEPENDENT_FEATURES,
    LABELS,
    MARKET_FEATURES,
    SCORE_LABELS,
    WIN_TOTAL_FEATURES,
)
from football.features.inputs import MissingTableError, Tables
from football.schemas import validate

MakeTables = Callable[..., Tables]


def write_tables(make_tables: MakeTables, seed: int = 0) -> None:
    for name, df in make_tables(seed).items():
        if df is not None:
            storage.write(df, "clean", name)


def test_labels_are_never_features() -> None:
    labels = {*LABELS, *SCORE_LABELS}
    assert not labels & set(FEATURE_COLUMNS)
    assert not set(WIN_TOTAL_FEATURES) & set(INDEPENDENT_FEATURES)
    assert set(WIN_TOTAL_FEATURES) <= set(MARKET_FEATURES)


def test_missing_required_table_raises() -> None:
    try:
        build_features()
    except MissingTableError as exc:
        assert "games" in str(exc)
    else:
        raise AssertionError("expected MissingTableError")


def test_build_features_labels_and_schema(make_tables: MakeTables) -> None:
    write_tables(make_tables)
    df = validate(build_features(), FEATURES_FULL)
    games = storage.read("clean", "games")
    assert df.height == games.height
    joined = df.join(games, on="game_id")
    played = joined.filter(pl.col("home_score").is_not_null())
    expected = (
        pl.when(pl.col("home_score") > pl.col("away_score"))
        .then(1.0)
        .when(pl.col("home_score") < pl.col("away_score"))
        .then(0.0)
        .otherwise(0.5)
    )
    assert played.select((pl.col("home_win") == expected).all()).item()
    assert played.select(
        (pl.col("home_points") == pl.col("home_score")).all()
        & (pl.col("away_points") == pl.col("away_score")).all()
    ).item()
    assert joined.filter(pl.col("home_score").is_null())["home_win"].null_count() == 3
    assert (df["cutoff"] == df["kickoff_utc"]).all()


def test_seasons_and_as_of(make_tables: MakeTables) -> None:
    write_tables(make_tables)
    full = build_features()
    only = build_features(seasons=[2021])
    assert set(only["season"].to_list()) == {2021}
    assert only.drop("cutoff").equals(full.filter(pl.col("season") == 2021).drop("cutoff"))
    moment = full["kickoff_utc"][20] - timedelta(hours=1)
    early = build_features(as_of=moment)
    assert (early["cutoff"] <= moment).all()
    assert early.filter(pl.col("kickoff_utc") <= moment).equals(
        full.filter(pl.col("kickoff_utc") <= moment)
    )


def test_cli_writes_features_table(make_tables: MakeTables) -> None:
    write_tables(make_tables, 1)
    result = CliRunner().invoke(app, ["build-features", "--seasons", "2020-2021"])
    assert result.exit_code == 0, result.output
    assert "36 games built" in result.output
    stored = storage.read("features", "games")
    validate(stored, FEATURES_FULL)
    result = CliRunner().invoke(app, ["build-features", "--as-of", "2021-10-01T00:00:00"])
    assert result.exit_code != 0
