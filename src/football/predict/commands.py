from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.console import Console

from football.backtest.commands import ParamOpt, TrackOption, parse_params
from football.models import registry
from football.opinions.files import OpinionError
from football.predict.display import weekly_table
from football.predict.run import (
    DEFAULT_LOCK_HOURS,
    PredictOutcome,
    PredictRequest,
    run_predict,
)
from football.predict.store import PredictError, next_week
from football.seasons import current_season


def _report(console: Console, outcome: PredictOutcome, title: str) -> None:
    if outcome.opinions is None:
        console.print("[dim]no opinion file for this week; blended equals model[/dim]")
    else:
        stamp = outcome.opinions.entered_at.isoformat(timespec="seconds")
        console.print(f"opinions from {outcome.opinions.path} (entered_at {stamp})")
        for warning in outcome.opinions.warnings():
            console.print(f"[yellow]warning:[/yellow] {warning}")
    console.print(weekly_table(outcome, title))
    console.print(f"run {outcome.run_id}")
    if outcome.locked:
        console.print(f"[cyan]locked final:[/cyan] {', '.join(outcome.locked)}")
    for path in outcome.exports:
        console.print(f"exported {path}")


def register(app: typer.Typer) -> None:
    @app.command(help="Predict a week's games, then blend in the owner's opinions.")
    def predict(
        season: Annotated[int | None, typer.Option(help="Season, default current")] = None,
        week: Annotated[int | None, typer.Option(help="Week, default next week")] = None,
        model: Annotated[str, typer.Option(help="Registered model name")] = "ensemble",
        param: ParamOpt = None,
        track: Annotated[
            TrackOption, typer.Option(help="Model track, passed to the model as track=...")
        ] = TrackOption.independent,
        final: Annotated[
            bool, typer.Option(help="Lock games kicking off within --lock-hours; never rewritten")
        ] = False,
        lock_hours: Annotated[
            float, typer.Option(help="With --final, lock games kicking off within this many hours")
        ] = DEFAULT_LOCK_HOURS,
        train_from: Annotated[
            int | None, typer.Option(help="First season of training data, default all")
        ] = None,
    ) -> None:
        console = Console()
        if model not in registry.MODEL_NAMES:
            raise typer.BadParameter(
                f"choose from {list(registry.MODEL_NAMES)}", param_hint="--model"
            )
        now = datetime.now(UTC)
        chosen_season = season or current_season(now)
        try:
            request = PredictRequest(
                season=chosen_season,
                week=week or next_week(chosen_season, now),
                model=model,
                params=parse_params(param, track),
                final=final,
                lock_hours=lock_hours,
                train_from=train_from,
            )
            outcome = run_predict(request, now)
        except OpinionError as error:
            console.print(f"[red]{error}[/red]")
            console.print(
                "model predictions were stored before reading opinions; fix the file and rerun"
            )
            raise typer.Exit(1) from error
        except PredictError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(1) from error
        _report(console, outcome, f"{request.season} week {request.week}: {model}")
        if final and not outcome.locked:
            console.print(f"[yellow]no unlocked games kick off within {lock_hours} hours[/yellow]")
        for game_id in outcome.refused:
            console.print(f"[red]refused:[/red] {game_id} already has a locked final prediction")
        if outcome.refused:
            raise typer.Exit(1)
