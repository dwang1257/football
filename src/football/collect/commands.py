from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football.collect.archive import import_archive
from football.collect.base import CollectorResult
from football.collect.runner import COLLECTORS, all_failed, run_collectors


def _summary(results: list[CollectorResult]) -> Table:
    table = Table("collector", "status", "files", "ok", "failed", "detail")
    for result in results:
        failures = [f"{key}: {value}" for key, value in sorted(result.failed.items())]
        detail = "; ".join([*result.notes[-2:], *failures[:6]])
        if len(failures) > 6:
            detail += f"; +{len(failures) - 6} more failures"
        table.add_row(
            result.collector,
            result.status,
            str(len(result.files)),
            str(len(result.ok)),
            str(len(result.failed)),
            detail,
        )
    return table


def _selected(only: str) -> list[str]:
    if not only:
        return list(COLLECTORS)
    names = [name.strip() for name in only.split(",") if name.strip()]
    unknown = [name for name in names if name not in COLLECTORS]
    if unknown:
        raise typer.BadParameter(f"unknown collectors {unknown}; available {list(COLLECTORS)}")
    return names


def register(app: typer.Typer) -> None:
    @app.command(help="Snapshot present-only data sources into the archive layer.")
    def collect(
        only: Annotated[
            str, typer.Option(help=f"Comma separated collectors from {', '.join(COLLECTORS)}")
        ] = "",
    ) -> None:
        console = Console()
        results = run_collectors(_selected(only))
        console.print(_summary(results))
        if all_failed(results):
            raise typer.Exit(code=1)

    @app.command(
        "archive-import",
        help="Copy a checkout of the data-archive branch into data/archive without overwriting.",
    )
    def archive_import(
        source: Annotated[
            Path, typer.Argument(help="Directory holding <collector>/<YYYY-MM-DD>/ snapshots")
        ],
    ) -> None:
        if not source.is_dir():
            raise typer.BadParameter(f"{source} is not a directory")
        result = import_archive(source)
        Console().print(
            f"copied {result.copied}, already present {result.skipped}, ignored {result.ignored}"
        )
