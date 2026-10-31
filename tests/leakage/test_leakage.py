import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from synthetic import make_tables, required

from football.features.build import FeatureParams, compute_features
from football.features.columns import BENCHMARK_COLUMNS, FEATURE_COLUMNS, INFO_COLUMNS
from football.features.inputs import Tables
from football.features.matchup import MatchupParams

OUTPUT_COLUMNS = ("cutoff", *FEATURE_COLUMNS, *BENCHMARK_COLUMNS, *INFO_COLUMNS)
FEATURES_DIR = Path(__file__).resolve().parents[2] / "src" / "football" / "features"
GAME_RESULT_COLUMNS = (
    "home_score",
    "away_score",
    "home_qb_id",
    "away_qb_id",
    "temp_observed",
    "wind_observed",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
)


def _map_tables(tables: Tables, fn) -> Tables:
    changed = {
        name: fn(name, df)
        for name, df in tables.items()
        if df is not None and "available_at" in df.columns
    }
    return replace(tables, **changed)


def truncate(tables: Tables, moment: datetime) -> Tables:
    return _map_tables(tables, lambda _, df: df.filter(pl.col("available_at") < moment))


def features_for(df: pl.DataFrame, game_ids: list[str]) -> pl.DataFrame:
    return (
        df.filter(pl.col("game_id").is_in(game_ids))
        .select("game_id", *OUTPUT_COLUMNS)
        .sort("game_id")
    )


def assert_same(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert left.height == right.height
    assert_frame_equal(left, right, check_exact=False, rel_tol=1e-9, abs_tol=1e-12)


def kickoff_range(tables: Tables) -> tuple[datetime, datetime]:
    kickoffs = tables.games["kickoff_utc"].sort().to_list()
    return kickoffs[0] - timedelta(days=2), kickoffs[-1] + timedelta(days=3)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    fraction=st.floats(0.0, 1.0),
    use_as_of=st.booleans(),
    passes=st.sampled_from([0, 2]),
)
def test_truncation_leaves_earlier_features_unchanged(
    seed: int, fraction: float, use_as_of: bool, passes: int
) -> None:
    tables = make_tables(seed)
    start, end = kickoff_range(tables)
    moment = start + (end - start) * fraction
    as_of = moment if use_as_of else None
    params = FeatureParams(matchup=MatchupParams(opponent_passes=passes))
    full = compute_features(tables, as_of=as_of, params=params)
    cut = compute_features(truncate(tables, moment), as_of=as_of, params=params)
    eligible = cut.filter(pl.col("cutoff") <= moment)["game_id"].to_list()
    assert eligible or fraction < 0.2
    assert_same(features_for(full, eligible), features_for(cut, eligible))


def _scramble(df: pl.DataFrame, mask: pl.Expr, columns: list[str]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for col in columns:
        dtype = df.schema[col]
        if dtype.is_numeric():
            exprs.append(pl.when(mask).then(pl.col(col) * 3 + 7).otherwise(pl.col(col)).alias(col))
        elif dtype == pl.String:
            exprs.append(
                pl.when(mask).then(pl.lit("zz_") + pl.col(col)).otherwise(pl.col(col)).alias(col)
            )
    return df.with_columns(exprs)


VALUE_COLUMNS: dict[str, list[str]] = {
    "games": list(GAME_RESULT_COLUMNS),
    "team_games": [
        "points_for",
        "points_against",
        "plays",
        "pass_yards",
        "takeaways",
        "wp_filtered_epa",
        "fumbles_lost",
    ],
    "qb_games": ["player_id", "dropbacks", "qb_epa"],
    "injuries": ["report_status", "player_id"],
    "depth_charts": ["player_id", "depth_rank"],
    "snap_counts": ["offense_pct", "defense_pct"],
    "lines": ["home_spread", "total"],
    "benchmarks": ["home_win_prob"],
    "team_ratings": ["rating"],
    "weather_forecasts": ["temp_f", "wind_mph", "precip_prob"],
    "weather_observed": ["temp_f", "wind_mph"],
    "win_totals": ["win_total"],
}


@dataclass(frozen=True)
class Game:
    game_id: str
    kickoff: datetime

    @property
    def own(self) -> pl.Expr:
        return pl.col("game_id").eq(self.game_id)


def _pick_game(tables: Tables, index: int) -> Game:
    played = tables.games.filter(pl.col("week") > 1).sort("kickoff_utc", "game_id")
    game_id, kickoff = played.select("game_id", "kickoff_utc").row(index % played.height)
    return Game(game_id, kickoff)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", ["change", "delete"])
def test_perturbing_rows_after_kickoff_leaves_game_unchanged(seed: int, mode: str) -> None:
    tables = make_tables(seed)
    game = _pick_game(tables, seed * 7 + 5)
    later = pl.col("available_at").ge(game.kickoff)
    own = game.own

    def perturb(name: str, df: pl.DataFrame) -> pl.DataFrame:
        if name == "games":
            keep = (own | ~later) if mode == "delete" else pl.lit(True)
            return _scramble(df, later, VALUE_COLUMNS[name]).filter(keep)
        if mode == "delete":
            return df.filter(~later)
        return _scramble(df, later, VALUE_COLUMNS[name])

    base = compute_features(tables)
    moved = compute_features(_map_tables(tables, perturb))
    ids = [game.game_id]
    assert_same(features_for(base, ids), features_for(moved, ids))


@pytest.mark.parametrize("seed", [0, 5, 9])
@pytest.mark.parametrize("use_as_of", [False, True])
def test_rows_available_exactly_at_cutoff_are_excluded(seed: int, use_as_of: bool) -> None:
    tables = make_tables(seed)
    game = _pick_game(tables, seed + 11)
    cutoff = game.kickoff - timedelta(hours=30) if use_as_of else game.kickoff
    as_of = cutoff if use_as_of else None
    window = pl.col("available_at").lt(cutoff) & pl.col("available_at").ge(
        cutoff - timedelta(days=8)
    )

    def at_boundary(_: str, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(window)
            .then(pl.lit(cutoff))
            .otherwise(pl.col("available_at"))
            .alias("available_at")
        )

    def dropped(_: str, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(~window)

    moved = _map_tables(tables, at_boundary)
    changed = sum(
        (
            df is not None
            and "available_at" in df.columns
            and df.filter(pl.col("available_at") == cutoff).height > 0
        )
        for _, df in moved.items()
    )
    assert changed >= 6
    ids = [game.game_id]
    with_boundary = compute_features(moved, as_of=as_of)
    without = compute_features(_map_tables(tables, dropped), as_of=as_of)
    assert_same(features_for(with_boundary, ids), features_for(without, ids))


def test_no_time_filtering_outside_asof() -> None:
    patterns = [
        re.compile(r"join_asof"),
        re.compile(r"(available_at|cutoff)[\"']\)?\s*(<=|>=|<|>|==|!=)"),
        re.compile(r"(<=|>=|<|>|==|!=)\s*pl\.col\([\"'](available_at|cutoff)"),
        re.compile(r"\.is_between\("),
    ]
    offenders = []
    files = sorted(p for p in FEATURES_DIR.glob("*.py") if p.name != "asof.py")
    assert files
    for path in files:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("seed", [0, 3, 8])
def test_forbidden_columns_of_target_game_do_not_matter(seed: int) -> None:
    tables = make_tables(seed)
    game = _pick_game(tables, seed + 4)
    own = game.own
    early = pl.lit(game.kickoff - timedelta(days=3))
    games = _scramble(tables.games, own, list(GAME_RESULT_COLUMNS)).with_columns(
        pl.when(own & (pl.col("roof") == "closed"))
        .then(pl.lit("open"))
        .otherwise(pl.col("roof"))
        .alias("roof"),
        pl.when(own).then(early).otherwise(pl.col("available_at")).alias("available_at"),
    )
    team_games = _scramble(tables.team_games, own, VALUE_COLUMNS["team_games"]).with_columns(
        pl.when(own).then(early).otherwise(pl.col("available_at")).alias("available_at")
    )
    qb_games = _scramble(required(tables.qb_games), own, VALUE_COLUMNS["qb_games"]).with_columns(
        pl.when(own).then(early).otherwise(pl.col("available_at")).alias("available_at")
    )
    moved = replace(tables, games=games, team_games=team_games, qb_games=qb_games)
    ids = [game.game_id]
    assert_same(
        features_for(compute_features(tables), ids), features_for(compute_features(moved), ids)
    )
