from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.markdown import Markdown

from football.config import get_settings
from football.opinions.blend import BlendParams
from football.opinions.learning import current_params, fit_weights, load_weights, save_weights
from football.predict.store import PredictError
from football.report.markdown import render_report, scorecard_section
from football.report.scoring import Scorecard, scorecard, scored_games
from football.seasons import current_season


def reports_dir() -> Path:
    path = get_settings().repo_dir / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def week_cards(season_games: pl.DataFrame, week: int) -> list[Scorecard]:
    to_date = season_games.filter(pl.col("week") <= week)
    return [
        scorecard(to_date.filter(pl.col("week") == week), f"Week {week}"),
        scorecard(to_date, "Season to date"),
    ]


def _latest_week(season_games: pl.DataFrame, season: int) -> int:
    if season_games.is_empty():
        raise PredictError(f"no scored games with predictions in {season}")
    return int(season_games.select(pl.col("week").max()).item())


def _load(season: int, week: int | None) -> tuple[pl.DataFrame, int]:
    games = scored_games(season)
    chosen = week or _latest_week(games, season)
    if games.filter(pl.col("week") == chosen).is_empty():
        raise PredictError(f"no scored games with predictions in {season} week {chosen}")
    return games, chosen


def register(app: typer.Typer) -> None:
    @app.command(help="Score predictions against results and refit the opinion weights.")
    def score(
        season: Annotated[int | None, typer.Option(help="Season, default current")] = None,
        week: Annotated[int | None, typer.Option(help="Week, default latest scored")] = None,
    ) -> None:
        console = Console()
        now = datetime.now(UTC)
        chosen_season = season or current_season(now)
        try:
            games, chosen_week = _load(chosen_season, week)
            history = scored_games()
        except PredictError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(1) from error
        cards = week_cards(games, chosen_week)
        console.print(Markdown("\n".join(scorecard_section(cards))))
        fitted = fit_weights(history, BlendParams.from_settings(), now)
        path = save_weights(fitted)
        console.print(
            f"opinion weights w {fitted.weight:.3f}, w_t {fitted.team_weight:.3f} "
            f"from {fitted.n_games} opinion games; saved {path}"
        )

    @app.command(help="Print the weekly report and write reports/<season>-w<WW>.md.")
    def report(
        season: Annotated[int | None, typer.Option(help="Season, default current")] = None,
        week: Annotated[int | None, typer.Option(help="Week, default latest scored")] = None,
    ) -> None:
        console = Console()
        now = datetime.now(UTC)
        chosen_season = season or current_season(now)
        try:
            games, chosen_week = _load(chosen_season, week)
        except PredictError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(1) from error
        to_date = games.filter(pl.col("week") <= chosen_week)
        text = render_report(
            chosen_season,
            chosen_week,
            to_date,
            week_cards(games, chosen_week),
            current_params(),
            load_weights(),
            now,
        )
        path = reports_dir() / f"{chosen_season}-w{chosen_week:02d}.md"
        path.write_text(text)
        console.print(Markdown(text))
        console.print(f"wrote {path}")
