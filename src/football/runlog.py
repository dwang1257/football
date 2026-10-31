import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from football import storage
from football.config import get_settings
from football.schemas import UTC_TS

RunKind = Literal["backtest", "holdout", "live"]
LOG_NAME = "runs"
PREDICTIONS_FILE = "predictions.parquet"

RUN_COLUMNS: dict[str, pl.DataType] = {
    "run_id": pl.String(),
    "kind": pl.String(),
    "model": pl.String(),
    "params": pl.String(),
    "track": pl.String(),
    "seasons": pl.List(pl.Int32()),
    "retrain": pl.String(),
    "min_train_season": pl.Int32(),
    "git_sha": pl.String(),
    "git_dirty": pl.Boolean(),
    "features_hash": pl.String(),
    "seed": pl.Int64(),
    "n_games": pl.Int64(),
    "metrics": pl.String(),
    "created_at": UTC_TS,
}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    kind: RunKind
    model: str
    params: dict[str, Any]
    track: str
    seasons: list[int]
    retrain: str
    min_train_season: int | None
    git_sha: str | None
    git_dirty: bool | None
    features_hash: str | None
    seed: int
    n_games: int
    metrics: dict[str, Any]
    created_at: datetime

    def row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "model": self.model,
            "params": canonical_json(self.params),
            "track": self.track,
            "seasons": self.seasons,
            "retrain": self.retrain,
            "min_train_season": self.min_train_season,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "features_hash": self.features_hash,
            "seed": self.seed,
            "n_games": self.n_games,
            "metrics": canonical_json(self.metrics),
            "created_at": self.created_at,
        }


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_finite(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=get_settings().repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def git_state() -> tuple[str | None, bool | None]:
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return None, None
    status = _git("status", "--porcelain")
    return sha.strip(), None if status is None else bool(status.strip())


def new_run_id(config: dict[str, Any], created_at: datetime) -> str:
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    payload = canonical_json(config) + created_at.isoformat() + os.urandom(8).hex()
    return f"{stamp}-{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def log_path() -> Path:
    return storage.table_path("runs", LOG_NAME)


def run_dir(run_id: str) -> Path:
    return storage.layer_dir("runs") / run_id


def predictions_path(run_id: str) -> Path:
    return run_dir(run_id) / PREDICTIONS_FILE


@contextmanager
def _locked() -> Iterator[None]:
    lock = storage.layer_dir("runs") / ".lock"
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    df.write_parquet(tmp)
    os.replace(tmp, path)


def empty_log() -> pl.DataFrame:
    return pl.DataFrame(schema=RUN_COLUMNS)


def read_runs() -> pl.DataFrame:
    path = log_path()
    return pl.read_parquet(path) if path.exists() else empty_log()


def read_predictions(run_id: str) -> pl.DataFrame:
    return pl.read_parquet(predictions_path(run_id))


def log_run(
    *,
    kind: RunKind,
    model: str,
    params: dict[str, Any],
    track: str,
    seasons: Sequence[int],
    retrain: str,
    min_train_season: int | None,
    features_hash: str | None,
    metrics: dict[str, Any],
    predictions: pl.DataFrame,
    seed: int | None = None,
) -> RunRecord:
    created_at = datetime.now(UTC)
    sha, dirty = git_state()
    config = {"kind": kind, "model": model, "params": params, "track": track}
    record = RunRecord(
        run_id=new_run_id(config, created_at),
        kind=kind,
        model=model,
        params=params,
        track=track,
        seasons=sorted(set(seasons)),
        retrain=retrain,
        min_train_season=min_train_season,
        git_sha=sha,
        git_dirty=dirty,
        features_hash=features_hash,
        seed=get_settings().seed if seed is None else seed,
        n_games=predictions.height,
        metrics=metrics,
        created_at=created_at,
    )
    row = pl.DataFrame([record.row()], schema=RUN_COLUMNS)
    with _locked():
        existing = read_runs()
        if record.run_id in existing["run_id"].to_list():
            raise RuntimeError(f"run id collision: {record.run_id}")
        _write_parquet(predictions, predictions_path(record.run_id))
        _write_parquet(pl.concat([existing, row], how="diagonal_relaxed"), log_path())
    return record


def find_runs(kind: RunKind, model: str, params: dict[str, Any]) -> pl.DataFrame:
    return read_runs().filter(
        (pl.col("kind") == kind)
        & (pl.col("model") == model)
        & (pl.col("params") == canonical_json(params))
    )
