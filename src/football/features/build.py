from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from football import storage
from football.features.columns import (
    BENCHMARK_COLUMNS,
    FEATURE_COLUMNS,
    INFO_COLUMNS,
    SCORE_LABELS,
)
from football.features.elo import EloParams, elo_features, elo_history
from football.features.injuries import injury_features
from football.features.inputs import Tables, labels, load_tables, make_targets
from football.features.market import benchmark_columns, market_features
from football.features.matchup import MatchupParams, matchup_features
from football.features.owner import owner_features
from football.features.qb import QbParams, qb_features
from football.features.situational import situational_features
from football.features.weather import weather_features
from football.schemas import FEATURES, TableSpec, validate

FEATURES_FULL = TableSpec(
    FEATURES.name,
    FEATURES.keys,
    {
        **FEATURES.columns,
        **{name: pl.Float64() for name in (*SCORE_LABELS, *FEATURE_COLUMNS, *BENCHMARK_COLUMNS)},
        **{name: pl.String() for name in INFO_COLUMNS},
    },
)


@dataclass(frozen=True)
class FeatureParams:
    elo: EloParams = field(default_factory=EloParams)
    matchup: MatchupParams = field(default_factory=MatchupParams)
    qb: QbParams = field(default_factory=QbParams)


def compute_features(
    tables: Tables,
    as_of: datetime | None = None,
    seasons: Sequence[int] | None = None,
    params: FeatureParams | None = None,
) -> pl.DataFrame:
    params = params or FeatureParams()
    targets = make_targets(tables, as_of)
    parts = [
        elo_features(targets, elo_history(tables.games, params.elo), params.elo),
        matchup_features(targets, tables.team_games, params.matchup),
        situational_features(targets, tables.team_homes),
        qb_features(targets, tables.qb_games, tables.depth_charts, tables.injuries, params.qb),
        injury_features(targets, tables.injuries, tables.snap_counts),
        weather_features(targets, tables.weather_forecasts, tables.team_homes),
        owner_features(
            targets, tables.games, tables.team_games, tables.weather_observed, tables.win_totals
        ),
        market_features(targets, tables.lines),
        benchmark_columns(targets, tables.benchmarks, tables.team_ratings),
    ]
    out = targets.select(
        "game_id",
        "season",
        "week",
        "game_type",
        "kickoff_utc",
        "cutoff",
        "home_team",
        "away_team",
    ).join(labels(tables.games), on="game_id", how="left")
    for part in parts:
        out = out.join(part, on="game_id", how="left")
    if seasons is not None:
        out = out.filter(pl.col("season").is_in(list(seasons)))
    out = out.with_columns(
        pl.col(name).cast(dtype) for name, dtype in FEATURES_FULL.columns.items()
    ).with_columns(pl.col(name).fill_nan(None) for name in (*FEATURE_COLUMNS, *BENCHMARK_COLUMNS))
    return validate(out.sort("kickoff_utc", "game_id"), FEATURES_FULL)


def build_features(
    as_of: datetime | None = None, seasons: Sequence[int] | None = None
) -> pl.DataFrame:
    return compute_features(load_tables(), as_of, seasons)


def write_features(df: pl.DataFrame) -> pl.DataFrame:
    return storage.upsert(df, "features", FEATURES.name, list(FEATURES.keys))
