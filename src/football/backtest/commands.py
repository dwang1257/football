import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from football import runlog
from football.backtest.metrics import Comparison, Interval, compare, summary
from football.backtest.runner import BacktestResult, run_logged
from football.seasons import parse_seasons

BACKTEST_SEASONS = "2012-2022"
HOLDOUT_SEASON_TEXT = "2023-2025"
HOLDOUT_SEASONS: tuple[int, ...] = tuple(parse_seasons(HOLDOUT_SEASON_TEXT))
REFERENCE_MODEL = "elo"


class RetrainOption(StrEnum):
    weekly = "weekly"
    season = "season"


class TrackOption(StrEnum):
    independent = "independent"
    market = "market"


class KindOption(StrEnum):
    backtest = "backtest"
    holdout = "holdout"


ModelOpt = Annotated[str, typer.Option("--model", help="Registered model name, such as elo")]
ParamOpt = Annotated[
    list[str] | None,
    typer.Option("--param", help="Model parameter as key=value, repeatable; values parse as JSON"),
]
TrackOpt = Annotated[
    TrackOption | None, typer.Option(help="Model track, passed to the model as track=...")
]
RetrainOpt = Annotated[
    RetrainOption, typer.Option(help="Refit every week, or once per season on prior seasons")
]


def parse_params(items: list[str] | None, track: TrackOption | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"expected key=value, got {item!r}", param_hint="--param")
        try:
            params[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            params[key.strip()] = raw
    if track is not None:
        params["track"] = track.value
    return params


def _pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{100 * value:.1f}%"


def _num(value: float, digits: int = 4) -> str:
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def _interval(value: Interval) -> str:
    return f"{value.mean:+.4f}\n[{value.lo:+.4f}, {value.hi:+.4f}]"


def metrics_table(title: str, metrics: pl.DataFrame) -> Table:
    table = Table(title=title)
    for name in ("Split", "Games", "Ties", "Accuracy", "Log loss", "Brier", "ECE"):
        table.add_column(name, justify="left" if name == "Split" else "right")
    in_seasons = False
    for row in metrics.iter_rows(named=True):
        if row["split"].isdigit() and not in_seasons and table.row_count:
            table.add_section()
        in_seasons = row["split"].isdigit()
        table.add_row(
            row["split"],
            str(row["n"]),
            str(row["ties"]),
            _pct(row["accuracy"]),
            _num(row["log_loss"]),
            _num(row["brier"]),
            _num(row["ece"]),
        )
    return table


def print_result(
    console: Console, label: str, result: BacktestResult, record: runlog.RunRecord
) -> None:
    seasons = f"{min(record.seasons)}-{max(record.seasons)}"
    title = f"{label}: {record.model} ({result.track}, {record.retrain}) {seasons}"
    console.print(metrics_table(title, result.metrics))
    dirty = " (dirty tree)" if record.git_dirty else ""
    console.print(f"logged run [bold]{record.run_id}[/bold] at git {record.git_sha}{dirty}")


def warn_holdout(console: Console, seasons: list[int]) -> None:
    touched = sorted(set(seasons) & set(HOLDOUT_SEASONS))
    if touched:
        console.print(
            f"[bold yellow]warning:[/bold yellow] seasons {touched} belong to the one-time "
            "holdout; do not let these results drive design choices"
        )


def logged_or_exit(
    console: Console,
    kind: runlog.RunKind,
    model: str,
    params: dict[str, Any],
    seasons: list[int],
    retrain: RetrainOption,
    min_train_season: int | None,
) -> tuple[BacktestResult, runlog.RunRecord]:
    try:
        return run_logged(kind, model, params, seasons, retrain.value, min_train_season)
    except FileNotFoundError as error:
        console.print(f"[bold red]error:[/bold red] {error}")
        raise typer.Exit(1) from error


@dataclass(frozen=True)
class Entry:
    run: dict[str, Any]
    predictions: pl.DataFrame


def load_entries(kind: str, seasons: list[int], console: Console) -> list[Entry]:
    entries: list[Entry] = []
    for run in runlog.read_runs().filter(pl.col("kind") == kind).iter_rows(named=True):
        path = runlog.predictions_path(run["run_id"])
        if not path.exists():
            console.print(f"[yellow]skipping {run['run_id']}: predictions file missing[/yellow]")
            continue
        predictions = runlog.read_predictions(run["run_id"]).filter(
            pl.col("season").is_in(seasons) & pl.col("home_win").is_not_null()
        )
        if predictions.height:
            entries.append(Entry(run, predictions))
    return entries


def pick_reference(entries: list[Entry], reference: str | None) -> Entry:
    if reference is not None:
        matches = [
            entry
            for entry in entries
            if entry.run["run_id"] == reference or short_id(entry.run["run_id"]) == reference
        ]
        if len(matches) == 1:
            return matches[0]
        problem = "is ambiguous" if matches else "not found for these seasons and kind"
        raise typer.BadParameter(f"run {reference} {problem}")
    pool = [entry for entry in entries if entry.run["model"] == REFERENCE_MODEL] or entries
    return min(
        pool,
        key=lambda entry: (-entry.predictions.height, summary(entry.predictions)["log_loss"]),
    )


@dataclass(frozen=True)
class Standing:
    entry: Entry
    stats: dict[str, Any]
    comparison: Comparison | None


def standings(entries: list[Entry], reference: Entry) -> tuple[list[Standing], list[str]]:
    rows: list[Standing] = []
    skipped: list[str] = []
    for entry in entries:
        if entry.run["run_id"] == reference.run["run_id"]:
            rows.append(Standing(entry, summary(entry.predictions), None))
            continue
        try:
            comparison = compare(entry.predictions, reference.predictions)
        except ValueError:
            skipped.append(entry.run["run_id"])
            continue
        rows.append(Standing(entry, comparison.candidate, comparison))
    return sorted(rows, key=lambda row: row.stats["log_loss"]), skipped


def short_id(run_id: str) -> str:
    return run_id.rsplit("-", 1)[-1]


def _setup(run: dict[str, Any]) -> str:
    params = {key: value for key, value in json.loads(run["params"]).items() if key != "track"}
    parts = [run["track"]]
    if run["retrain"] != "weekly":
        parts.append(run["retrain"])
    if params:
        parts.append(json.dumps(params))
    return " ".join(parts)


def runs_table(rows: list[Standing], title: str) -> Table:
    counts = {row.stats["n"] for row in rows}
    show_games = len(counts) > 1
    if not show_games:
        title = f"{title}, {counts.pop()} games each"
    table = Table(title=title)
    headers = ["#", "Run", "Model", "Setup"]
    headers += ["Games"] if show_games else []
    headers += ["Acc", "Log loss", "Brier", "ECE"]
    for header in headers:
        table.add_column(header, no_wrap=header != "Setup")
    for index, row in enumerate(rows, start=1):
        run = row.entry.run
        games = [str(row.stats["n"])] if show_games else []
        table.add_row(
            str(index),
            short_id(run["run_id"]),
            run["model"],
            _setup(run),
            *games,
            _pct(row.stats["accuracy"]),
            _num(row.stats["log_loss"]),
            _num(row.stats["brier"]),
            _num(row.stats["ece"]),
            style="bold" if row.comparison is None else None,
        )
    return table


def comparison_table(rows: list[Standing], title: str) -> Table:
    table = Table(title=title, show_lines=True)
    headers = (
        "#",
        "Model",
        "dLogLoss (95% CI)",
        "Flips W/L",
        "Seasons",
        "Adopt",
    )
    for header in headers:
        table.add_column(header, no_wrap=True)
    for index, row in enumerate(rows, start=1):
        comparison = row.comparison
        if comparison is None:
            cells = ["reference", "", "", ""]
        else:
            test = comparison.mcnemar
            cells = [
                _interval(comparison.bootstrap.log_loss),
                f"{test.candidate_only}/{test.reference_only}\np={test.p_value:.3f}",
                f"{comparison.seasons.won}/{comparison.seasons.total}",
                "yes" if comparison.adoptable else "no",
            ]
        table.add_row(str(index), row.entry.run["model"], *cells)
    return table


def register(app: typer.Typer) -> None:
    @app.command(help="Walk-forward backtest of a registered model, recorded in the run log.")
    def backtest(
        model: ModelOpt,
        param: ParamOpt = None,
        track: TrackOpt = None,
        seasons: Annotated[str, typer.Option(help="Test seasons such as 2012-2022")] = (
            BACKTEST_SEASONS
        ),
        retrain: RetrainOpt = RetrainOption.weekly,
        min_train_season: Annotated[
            int | None, typer.Option(help="Ignore training games before this season")
        ] = None,
    ) -> None:
        console = Console()
        selected = parse_seasons(seasons)
        warn_holdout(console, selected)
        params = parse_params(param, track)
        result, record = logged_or_exit(
            console, "backtest", model, params, selected, retrain, min_train_season
        )
        print_result(console, "backtest", result, record)

    @app.command(help="Compare logged runs on identical games against a reference run.")
    def leaderboard(
        seasons: Annotated[
            str | None,
            typer.Option(help="Seasons to compare on; default 2012-2022, or 2023-2025 for holdout"),
        ] = None,
        reference: Annotated[
            str | None, typer.Option(help="Reference run id; default is the best elo run")
        ] = None,
        kind: Annotated[KindOption, typer.Option(help="Which runs to compare")] = (
            KindOption.backtest
        ),
    ) -> None:
        console = Console()
        if seasons is None:
            seasons = HOLDOUT_SEASON_TEXT if kind == KindOption.holdout else BACKTEST_SEASONS
        entries = load_entries(kind.value, parse_seasons(seasons), console)
        if not entries:
            console.print(f"no {kind.value} runs with predictions for seasons {seasons}")
            raise typer.Exit(1)
        chosen = pick_reference(entries, reference)
        title = f"{kind.value} leaderboard {seasons}, reference {short_id(chosen.run['run_id'])}"
        rows, skipped = standings(entries, chosen)
        console.print(runs_table(rows, title))
        console.print(comparison_table(rows, "Candidate minus reference on common games"))
        if skipped:
            console.print(f"[yellow]no common games with the reference: {skipped}[/yellow]")
        console.print(
            "Differences are candidate minus reference, negative is better, with 95% intervals "
            "from a paired bootstrap over weeks. Flips W/L counts games only the candidate or "
            "only the reference picked right, with the exact McNemar p-value. Adopt: log loss "
            "interval below zero or at least 70% of seasons won."
        )

    @app.command(help="Score the frozen 2023-2025 holdout once for a model and its parameters.")
    def holdout(
        model: ModelOpt,
        param: ParamOpt = None,
        track: TrackOpt = None,
        retrain: RetrainOpt = RetrainOption.weekly,
        force: Annotated[bool, typer.Option(help="Run again even if already scored")] = False,
    ) -> None:
        console = Console()
        params = parse_params(param, track)
        previous = runlog.find_runs("holdout", model, params)
        if previous.height and not force:
            console.print(
                f"[bold red]refusing:[/bold red] holdout already scored for {model} "
                f"{runlog.canonical_json(params)} in runs {previous['run_id'].to_list()}; "
                "pass --force to run it again"
            )
            raise typer.Exit(1)
        result, record = logged_or_exit(
            console, "holdout", model, params, list(HOLDOUT_SEASONS), retrain, None
        )
        print_result(console, "holdout", result, record)
