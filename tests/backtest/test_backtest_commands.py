from typing import Any

import polars as pl
import pytest
import typer
from typer.testing import CliRunner

from football import runlog
from football.backtest.commands import parse_params, register, short_id

Calls = list[tuple[str, dict[str, Any]]]


@pytest.fixture
def app(stored_features: pl.DataFrame, fake_registry: Calls) -> typer.Typer:
    application = typer.Typer()
    register(application)
    return application


def invoke(app: typer.Typer, *args: str) -> Any:
    return CliRunner().invoke(app, list(args), env={"COLUMNS": "200"})


def test_parse_params() -> None:
    params = parse_params(["k=20", "hfa=1.5", "mode=fast", "flag=true"], None)
    assert params == {"k": 20, "hfa": 1.5, "mode": "fast", "flag": True}
    with pytest.raises(typer.BadParameter):
        parse_params(["oops"], None)


def test_backtest_command_logs_run(app: typer.Typer, fake_registry: Calls) -> None:
    result = invoke(
        app,
        "backtest",
        "--model",
        "elo",
        "--param",
        "k=20",
        "--seasons",
        "2020-2022",
        "--track",
        "independent",
    )
    assert result.exit_code == 0, result.output
    assert "weeks 1-4" in result.output
    assert "playoffs" in result.output
    assert "warning" not in result.output
    assert fake_registry[0] == ("elo", {"k": 20, "track": "independent"})
    runs = runlog.read_runs()
    assert runs["kind"].to_list() == ["backtest"]
    assert runs["seasons"].to_list() == [[2020, 2021, 2022]]
    assert runs["features_hash"][0] is not None
    assert runlog.read_predictions(runs["run_id"][0]).height > 0


def test_backtest_warns_about_holdout_seasons(app: typer.Typer) -> None:
    result = invoke(app, "backtest", "--model", "elo", "--seasons", "2022-2023")
    assert result.exit_code == 0, result.output
    assert "warning" in result.output


def test_leaderboard_uses_best_elo_reference(app: typer.Typer) -> None:
    for model, retrain in (("elo", "weekly"), ("home_rate", "season"), ("strength", "weekly")):
        result = invoke(
            app, "backtest", "--model", model, "--seasons", "2019-2022", "--retrain", retrain
        )
        assert result.exit_code == 0, result.output
    elo_run = runlog.read_runs().filter(pl.col("model") == "elo")["run_id"][0]
    result = invoke(app, "leaderboard", "--seasons", "2019-2022")
    assert result.exit_code == 0, result.output
    assert f"reference {short_id(elo_run)}" in result.output
    assert "home_rate" in result.output
    assert "reference" in result.output
    for chosen in (elo_run, short_id(elo_run)):
        explicit = invoke(app, "leaderboard", "--seasons", "2019-2022", "--reference", chosen)
        assert explicit.exit_code == 0, explicit.output
        assert f"reference {short_id(elo_run)}" in explicit.output
    missing = invoke(app, "leaderboard", "--seasons", "2019-2022", "--reference", "missing")
    assert missing.exit_code != 0


def test_leaderboard_without_runs(app: typer.Typer) -> None:
    result = invoke(app, "leaderboard")
    assert result.exit_code == 1


def test_holdout_runs_once_unless_forced(app: typer.Typer) -> None:
    first = invoke(app, "holdout", "--model", "elo", "--param", "k=20")
    assert first.exit_code == 0, first.output
    holdouts = runlog.read_runs().filter(pl.col("kind") == "holdout")
    assert holdouts["seasons"].to_list() == [[2023, 2024, 2025]]
    predictions = runlog.read_predictions(holdouts["run_id"][0])
    assert set(predictions["season"]) == {2023, 2024, 2025}

    second = invoke(app, "holdout", "--model", "elo", "--param", "k=20")
    assert second.exit_code == 1
    assert "refusing" in second.output

    other = invoke(app, "holdout", "--model", "elo", "--param", "k=25")
    assert other.exit_code == 0, other.output

    forced = invoke(app, "holdout", "--model", "elo", "--param", "k=20", "--force")
    assert forced.exit_code == 0, forced.output
    assert runlog.read_runs().filter(pl.col("kind") == "holdout").height == 3


def test_backtest_without_features_fails_cleanly(fake_registry: Calls) -> None:
    application = typer.Typer()
    register(application)
    result = invoke(application, "backtest", "--model", "elo")
    assert result.exit_code == 1
    assert "build-features" in result.output


def test_holdout_leaderboard_defaults_to_holdout_seasons(app: typer.Typer) -> None:
    for model in ("elo", "home_rate"):
        assert invoke(app, "holdout", "--model", model).exit_code == 0
    result = invoke(app, "leaderboard", "--kind", "holdout")
    assert result.exit_code == 0, result.output
    assert "holdout leaderboard 2023-2025" in result.output
