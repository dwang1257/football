from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from football import runlog, storage
from football.backtest.metrics import metrics_by_split
from football.backtest.splits import Retrain, walk_forward
from football.models import registry
from football.models.base import Model
from football.schemas import FEATURES, validate

ModelFactory = Callable[[], Model]

LABEL_COLUMNS: tuple[str, ...] = ("home_win", "home_margin")
PREDICTION_SCHEMA = pl.Schema(
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "game_type": pl.String(),
        "home_win": pl.Float64(),
        "home_margin": pl.Float64(),
        "home_win_prob": pl.Float64(),
        "pred_margin": pl.Float64(),
    }
)


class ModelOutputError(ValueError):
    pass


@dataclass(frozen=True)
class BacktestResult:
    model_name: str
    track: str
    predictions: pl.DataFrame
    metrics: pl.DataFrame

    def metrics_dict(self) -> dict[str, dict[str, Any]]:
        return {row.pop("split"): row for row in self.metrics.to_dicts()}

    def overall(self) -> dict[str, Any]:
        return self.metrics_dict()["overall"]


def _checked_output(output: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    missing = [name for name in ("game_id", "home_win_prob") if name not in output.columns]
    if missing:
        raise ModelOutputError(f"model output is missing columns {missing}")
    margin = pl.col("home_margin") if "home_margin" in output.columns else pl.lit(None)
    out = output.select(
        "game_id",
        home_win_prob=pl.col("home_win_prob").cast(pl.Float64),
        pred_margin=margin.cast(pl.Float64),
    )
    if out["game_id"].is_duplicated().any():
        raise ModelOutputError("model output has duplicate game_id rows")
    if set(out["game_id"].to_list()) != set(test["game_id"].to_list()):
        raise ModelOutputError("model output game_ids do not match the test games")
    prob = pl.col("home_win_prob")
    valid = out.select((prob.is_finite() & prob.is_between(0.0, 1.0)).all()).item()
    if out["home_win_prob"].null_count() or not valid:
        raise ModelOutputError("home_win_prob must be finite and within [0, 1]")
    return out


def _predict(model: Model, test: pl.DataFrame) -> pl.DataFrame:
    output = _checked_output(model.predict(test.drop(LABEL_COLUMNS)), test)
    return test.select("game_id", "season", "week", "game_type", *LABEL_COLUMNS).join(
        output, on="game_id", how="left"
    )


def run_backtest(
    model_factory: ModelFactory,
    features: pl.DataFrame,
    seasons: Sequence[int],
    retrain: Retrain = "weekly",
    min_train_season: int | None = None,
) -> BacktestResult:
    model: Model | None = None
    frames: list[pl.DataFrame] = []
    for split in walk_forward(features, seasons, retrain, min_train_season):
        if model is None or split.refit:
            model = model_factory()
            model.fit(split.train)
        frames.append(_predict(model, split.test))
    if model is None:
        raise ValueError(f"no labeled games to test in seasons {sorted(set(seasons))}")
    predictions = (
        pl.concat(frames)
        .cast(PREDICTION_SCHEMA)
        .select(PREDICTION_SCHEMA.names())
        .sort("season", "week", "game_id")
    )
    return BacktestResult(
        model_name=model.name,
        track=model.track,
        predictions=predictions,
        metrics=metrics_by_split(predictions),
    )


def load_features() -> tuple[pl.DataFrame, str]:
    path = storage.table_path("features", FEATURES.name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run football build-features first")
    return validate(pl.read_parquet(path), FEATURES), runlog.file_sha256(path)


def registry_factory(name: str, params: dict[str, Any]) -> ModelFactory:
    return lambda: registry.build(name, **params)


def run_logged(
    kind: runlog.RunKind,
    name: str,
    params: dict[str, Any],
    seasons: Sequence[int],
    retrain: Retrain = "weekly",
    min_train_season: int | None = None,
) -> tuple[BacktestResult, runlog.RunRecord]:
    features, features_hash = load_features()
    result = run_backtest(
        registry_factory(name, params), features, seasons, retrain, min_train_season
    )
    record = runlog.log_run(
        kind=kind,
        model=name,
        params=params,
        track=result.track,
        seasons=seasons,
        retrain=retrain,
        min_train_season=min_train_season,
        features_hash=features_hash,
        metrics=result.metrics_dict(),
        predictions=result.predictions,
    )
    return result, record
