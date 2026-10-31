from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.console import Console

from football.opinions.files import OpinionError
from football.opinions.template import write_template
from football.predict.store import PredictError, next_week
from football.seasons import current_season


def register(app: typer.Typer) -> None:
    @app.command(help="Write the weekly opinion template opinions/<season>-w<WW>.yaml.")
    def opine(
        season: Annotated[int | None, typer.Option(help="Season, default current")] = None,
        week: Annotated[int | None, typer.Option(help="Week, default next week")] = None,
        blind: Annotated[bool, typer.Option(help="Hide model numbers to avoid anchoring")] = False,
        force: Annotated[bool, typer.Option(help="Overwrite an existing opinion file")] = False,
    ) -> None:
        console = Console()
        now = datetime.now(UTC)
        chosen_season = season or current_season(now)
        try:
            chosen_week = week or next_week(chosen_season, now)
            path = write_template(chosen_season, chosen_week, blind, force)
        except (OpinionError, PredictError) as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(1) from error
        mode = "blind" if blind else "with model probabilities"
        console.print(f"wrote {path} ({mode}); fill in pick and conf (1-5) or prob per game")
