import importlib
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from football.ingest.commands import SOURCE_MODULES
from football.validate.base import CheckFn


def all_checks() -> dict[str, CheckFn]:
    checks: dict[str, CheckFn] = {}
    for module_name in SOURCE_MODULES:
        checks.update(importlib.import_module(module_name).CHECKS)
    return checks


def register(app: typer.Typer) -> None:
    @app.command(help="Run data accuracy and coverage checks.")
    def validate(
        checks: Annotated[str, typer.Option(help="Comma separated check names, or all")] = "all",
    ) -> None:
        available = all_checks()
        selected = list(available) if checks == "all" else [c.strip() for c in checks.split(",")]
        table = Table("check", "status", "detail")
        failed = 0
        for name in selected:
            result = available[name]()
            failed += not result.passed
            table.add_row(result.name, "pass" if result.passed else "FAIL", result.detail)
        Console().print(table)
        if failed:
            raise typer.Exit(code=1)
