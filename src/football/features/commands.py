import time
from datetime import datetime
from typing import Annotated

import polars as pl
import typer
from rich.console import Console

from football import storage
from football.features.build import build_features, write_features
from football.features.columns import (
    ELO_FEATURES,
    INJURY_FEATURES,
    MARKET_FEATURES,
    MATCHUP_COLUMNS,
    OWNER_FEATURES,
    QB_FEATURES,
    SITUATIONAL_FEATURES,
    WEATHER_FEATURES,
)
from football.seasons import parse_seasons

GROUPS: dict[str, tuple[str, ...]] = {
    "elo": ELO_FEATURES,
    "matchup": MATCHUP_COLUMNS,
    "situational": SITUATIONAL_FEATURES,
    "qb": QB_FEATURES,
    "injuries": INJURY_FEATURES,
    "weather": WEATHER_FEATURES,
    "owner": OWNER_FEATURES,
    "market": MARKET_FEATURES,
}


def null_rates(df: pl.DataFrame) -> dict[str, float]:
    if df.is_empty():
        return dict.fromkeys(GROUPS, 0.0)
    return {
        group: df.select(pl.mean_horizontal(pl.col(c).is_null().mean() for c in cols)).item()
        for group, cols in GROUPS.items()
    }


def _parse_as_of(text: str | None) -> datetime | None:
    if text is None:
        return None
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise typer.BadParameter("--as-of needs a timezone, for example 2026-10-31T12:00:00+00:00")
    return moment


def register(app: typer.Typer) -> None:
    @app.command("build-features", help="Build point-in-time features for every game.")
    def build_features_command(
        seasons: Annotated[
            str | None, typer.Option(help="Seasons to output, for example 1999-2026")
        ] = None,
        as_of: Annotated[
            str | None,
            typer.Option(
                help="ISO timestamp with timezone; writes the separate games_as_of table",
            ),
        ] = None,
    ) -> None:
        console = Console()
        started = time.perf_counter()
        moment = _parse_as_of(as_of)
        df = build_features(moment, parse_seasons(seasons) if seasons else None)
        if moment is None:
            stored = write_features(df)
        else:
            storage.write(df, "features", "games_as_of")
            stored = df
        elapsed = time.perf_counter() - started
        console.print(f"features: {df.height} games built, {stored.height} stored, {elapsed:.1f}s")
        for group, rate in null_rates(df).items():
            console.print(f"  {group}: {rate:.1%} null")
