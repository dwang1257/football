import importlib
from typing import Annotated

import typer
from rich.console import Console

from football.ingest.base import IngestFn, IngestOptions
from football.seasons import current_season, parse_seasons

SOURCE_MODULES = (
    "football.ingest.nflverse",
    "football.ingest.venues",
    "football.ingest.lines",
    "football.ingest.benchmarks",
    "football.ingest.weather",
    "football.collect.normalize",
)


def all_sources() -> dict[str, IngestFn]:
    sources: dict[str, IngestFn] = {}
    for module_name in SOURCE_MODULES:
        sources.update(importlib.import_module(module_name).SOURCES)
    return sources


def register(app: typer.Typer) -> None:
    @app.command(help="Download and normalize data sources into clean tables.")
    def ingest(
        sources: Annotated[str, typer.Option(help="Comma separated source names, or all")] = "all",
        seasons: Annotated[str, typer.Option(help="Seasons such as 1999-2026")] = "",
        refresh: Annotated[bool, typer.Option(help="Re-download cached raw data")] = False,
    ) -> None:
        console = Console()
        available = all_sources()
        selected = list(available) if sources == "all" else [s.strip() for s in sources.split(",")]
        unknown = [name for name in selected if name not in available]
        if unknown:
            raise typer.BadParameter(f"unknown sources {unknown}; available {list(available)}")
        options = IngestOptions(
            seasons=parse_seasons(seasons) if seasons else list(range(1999, current_season() + 1)),
            refresh=refresh,
        )
        for name in selected:
            console.print(f"[bold]ingest[/bold] {name}")
            available[name](options)
