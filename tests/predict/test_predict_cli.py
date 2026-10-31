from datetime import UTC, datetime
from pathlib import Path

import yaml
from typer.testing import CliRunner

from conftest import Fakes, schedule
from football import storage
from football.cli import app
from football.opinions.files import opinion_path

runner = CliRunner()


def test_opine_predict_and_lock_from_the_cli(fakes: Fakes, repo: Path) -> None:
    storage.write(schedule(datetime.now(UTC)), "clean", "games")
    blind = runner.invoke(app, ["opine", "--season", "2026", "--week", "9", "--blind"])
    assert blind.exit_code == 0, blind.output
    refused = runner.invoke(app, ["opine", "--season", "2026", "--week", "9"])
    assert refused.exit_code == 1
    assert "already exists" in refused.output

    path = opinion_path(2026, 9)
    document = yaml.safe_load(path.read_text())
    document["games"][2].update(pick="NYG", conf=5, why="ATL secondary is banged up")
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    result = runner.invoke(
        app,
        ["predict", "--season", "2026", "--week", "9", "--model", "elo"],
        env={"COLUMNS": "120"},
    )
    assert result.exit_code == 0, result.output
    assert "NYG@ATL" in result.output
    assert "FLIPPED" in result.output
    assert "ATL secondary is banged up" in result.output

    filled = runner.invoke(app, ["opine", "--season", "2026", "--week", "9", "--force"])
    assert filled.exit_code == 0, filled.output
    assert "model_prob: 0.55" in path.read_text()

    locked = runner.invoke(
        app,
        [
            "predict",
            "--season",
            "2026",
            "--week",
            "9",
            "--model",
            "elo",
            "--final",
            "--lock-hours",
            "200",
        ],
    )
    assert locked.exit_code == 0, locked.output
    relock = runner.invoke(
        app,
        [
            "predict",
            "--season",
            "2026",
            "--week",
            "9",
            "--model",
            "elo",
            "--final",
            "--lock-hours",
            "200",
        ],
    )
    assert relock.exit_code == 1
    assert "refused" in relock.output


def test_unknown_model_is_rejected(fakes: Fakes) -> None:
    result = runner.invoke(app, ["predict", "--season", "2026", "--week", "9", "--model", "nope"])
    assert result.exit_code == 2
