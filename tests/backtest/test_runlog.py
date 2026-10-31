import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from football import runlog
from football.config import get_settings


def predictions(prob: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": [2020, 2020],
            "home_win": [1.0, 0.0],
            "home_win_prob": [prob, prob],
        }
    )


def log(model: str, prob: float, kind: runlog.RunKind = "backtest") -> runlog.RunRecord:
    return runlog.log_run(
        kind=kind,
        model=model,
        params={"k": 20, "track": "independent"},
        track="independent",
        seasons=[2021, 2020, 2020],
        retrain="weekly",
        min_train_season=None,
        features_hash="abc",
        metrics={"overall": {"log_loss": 0.65, "ece": float("nan")}},
        predictions=predictions(prob),
    )


def test_run_log_round_trip(isolated_data_dir: Path) -> None:
    first = log("elo", 0.6)
    second = log("matchup", 0.7, kind="holdout")
    assert first.run_id != second.run_id
    assert runlog.log_path() == isolated_data_dir / "runs" / "runs.parquet"
    runs = runlog.read_runs()
    assert runs["run_id"].to_list() == [first.run_id, second.run_id]
    assert runs.schema == pl.Schema(runlog.RUN_COLUMNS)
    row = runs.row(0, named=True)
    assert row["kind"] == "backtest"
    assert row["seasons"] == [2020, 2021]
    assert json.loads(row["params"]) == {"k": 20, "track": "independent"}
    assert json.loads(row["metrics"]) == {"overall": {"ece": None, "log_loss": 0.65}}
    assert row["seed"] == get_settings().seed
    assert row["n_games"] == 2
    assert row["created_at"].tzinfo is not None
    assert runlog.read_predictions(second.run_id)["home_win_prob"].to_list() == [0.7, 0.7]


def test_find_runs_matches_params_regardless_of_order() -> None:
    log("elo", 0.6, kind="holdout")
    found = runlog.find_runs("holdout", "elo", {"track": "independent", "k": 20})
    assert found.height == 1
    assert runlog.find_runs("holdout", "elo", {"k": 21}).height == 0
    assert runlog.find_runs("backtest", "elo", {"track": "independent", "k": 20}).height == 0


def test_git_unavailable_is_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTBALL_REPO_DIR", str(tmp_path / "not-a-repo"))
    get_settings.cache_clear()
    assert runlog.git_state() == (None, None)
    record = log("elo", 0.6)
    assert record.git_sha is None
    assert runlog.read_runs()["git_dirty"].to_list() == [None]


def test_git_state_in_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTBALL_REPO_DIR", str(Path(__file__).resolve().parents[2]))
    get_settings.cache_clear()
    sha, dirty = runlog.git_state()
    assert sha is not None
    assert len(sha) == 40
    assert isinstance(dirty, bool)


def test_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"football")
    assert runlog.file_sha256(path) == hashlib.sha256(b"football").hexdigest()
