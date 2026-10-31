import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from football import runlog, storage
from football.features.build import build_features
from football.models import registry
from football.opinions.blend import BlendParams, blend
from football.opinions.files import OpinionSet, load_opinions
from football.opinions.learning import current_params
from football.predict.store import (
    PredictError,
    export_csv,
    final_rows,
    week_games,
    write_blended_rows,
    write_model_rows,
)
from football.schemas import PREDICTIONS_BLENDED, PREDICTIONS_MODEL, UTC_TS

LABEL_COLUMNS: tuple[str, ...] = ("home_win", "home_margin")
DEFAULT_LOCK_HOURS = 3.0
RUN_RECORD_DIR = "live"


@dataclass(frozen=True)
class PredictRequest:
    season: int
    week: int
    model: str = "ensemble"
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    final: bool = False
    lock_hours: float = DEFAULT_LOCK_HOURS
    train_from: int | None = None


@dataclass(frozen=True)
class PredictOutcome:
    run_id: str
    games: pl.DataFrame
    model_rows: pl.DataFrame
    blended_rows: pl.DataFrame
    opinions: OpinionSet | None
    locked: tuple[str, ...]
    refused: tuple[str, ...]
    exports: tuple[Path, ...]


def lock_plan(
    upcoming: pl.DataFrame, now: datetime, lock_hours: float
) -> tuple[list[str], list[str]]:
    window = upcoming.filter(pl.col("kickoff_utc") <= now + timedelta(hours=lock_hours))
    candidates = window["game_id"].to_list()
    already = set(final_rows(PREDICTIONS_MODEL.name, candidates)["game_id"].to_list())
    return [g for g in candidates if g not in already], [g for g in candidates if g in already]


def _model_output(output: pl.DataFrame, game_ids: list[str]) -> pl.DataFrame:
    missing = [name for name in ("game_id", "home_win_prob") if name not in output.columns]
    if missing:
        raise PredictError(f"model output is missing columns {missing}")
    margin = pl.col("home_margin") if "home_margin" in output.columns else pl.lit(None)
    out = output.select(
        "game_id",
        home_win_prob=pl.col("home_win_prob").cast(pl.Float64),
        home_margin=margin.cast(pl.Float64),
    )
    if sorted(out["game_id"].to_list()) != sorted(game_ids):
        raise PredictError("model output game_ids do not match the games to predict")
    prob = pl.col("home_win_prob")
    if not out.select((prob.is_not_null() & prob.is_between(0.0, 1.0)).all()).item():
        raise PredictError("model home_win_prob must be within [0, 1]")
    return out


def predict_model(
    request: PredictRequest, upcoming: pl.DataFrame, now: datetime
) -> tuple[str, str, pl.DataFrame]:
    first = request.train_from
    seasons = None if first is None else range(min(first, request.season), request.season + 1)
    features = build_features(as_of=now, seasons=seasons)
    game_ids = upcoming["game_id"].to_list()
    targets = features.filter(pl.col("game_id").is_in(game_ids))
    missing = sorted(set(game_ids) - set(targets["game_id"].to_list()))
    if missing:
        raise PredictError(f"no feature rows for {missing}")
    earliest = targets["cutoff"].min()
    train = features.filter(pl.col("home_win").is_not_null() & (pl.col("kickoff_utc") < earliest))
    if train.is_empty():
        raise PredictError(f"no labeled games before {earliest}")
    model = registry.build(request.model, **request.params)
    model.fit(train)
    output = _model_output(model.predict(targets.drop(LABEL_COLUMNS, strict=False)), game_ids)
    rows = output.join(targets.select("game_id", "cutoff"), on="game_id", how="left")
    return model.name, model.track, rows


def _model_rows(
    run_id: str,
    name: str,
    track: str,
    output: pl.DataFrame,
    locked: list[str],
    now: datetime,
) -> pl.DataFrame:
    return output.select(
        pl.lit(run_id).alias("run_id"),
        "game_id",
        pl.lit(name).alias("model"),
        pl.lit(track).alias("track"),
        "home_win_prob",
        "home_margin",
        pl.col("cutoff").cast(UTC_TS),
        pl.lit(now).cast(UTC_TS).alias("created_at"),
        pl.col("game_id").is_in(locked).alias("is_final"),
    ).sort("game_id")


def _blended_rows(
    model_rows: pl.DataFrame,
    opinions: OpinionSet | None,
    params: BlendParams,
    now: datetime,
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for row in model_rows.iter_rows(named=True):
        game_id = row["game_id"]
        entry = opinions.games.get(game_id) if opinions else None
        nudge = opinions.nudges.get(game_id, 0.0) if opinions else 0.0
        conf = entry.conf if entry else None
        result = blend(
            row["home_win_prob"], entry.user_prob if entry else None, nudge, conf, params
        )
        records.append(
            {
                "run_id": row["run_id"],
                "game_id": game_id,
                "model_prob": result.model_prob,
                "user_prob": result.user_prob,
                "team_nudge_points": nudge,
                "opinion_weight": params.weight,
                "delta_logit": result.delta_logit,
                "capped": result.capped,
                "blended_prob": result.blended_prob,
                "flipped": result.flipped,
                "created_at": now,
                "is_final": row["is_final"],
                "user_pick": entry.pick if entry else None,
                "user_conf": conf,
                "user_pick_prob": entry.pick_prob if entry else None,
                "user_why": entry.why if entry else None,
                "entered_at": opinions.entered_at if opinions else None,
                "opinion_late": bool(opinions and game_id in opinions.late),
            }
        )
    return pl.DataFrame(records, infer_schema_length=None).with_columns(
        pl.col("user_conf").cast(pl.Int32),
        pl.col("created_at", "entered_at").cast(UTC_TS),
        pl.col("user_prob", "user_pick_prob").cast(pl.Float64),
        pl.col("user_pick", "user_why").cast(pl.String),
    )


def unblended_finals(games: pl.DataFrame, run_id: str) -> pl.DataFrame:
    game_ids = games["game_id"].to_list()
    blended = final_rows(PREDICTIONS_BLENDED.name, game_ids)["game_id"].to_list()
    return (
        final_rows(PREDICTIONS_MODEL.name, game_ids)
        .filter(~pl.col("game_id").is_in(blended) & (pl.col("run_id") != run_id))
        .select(PREDICTIONS_MODEL.columns.keys())
        .sort("game_id")
    )


def _rows_sha256(df: pl.DataFrame) -> str:
    return hashlib.sha256(df.write_csv().encode()).hexdigest()


def _write_record(run_id: str, record: dict[str, Any]) -> Path:
    path = storage.layer_dir("runs") / RUN_RECORD_DIR / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _export(
    model_rows: pl.DataFrame,
    blended_rows: pl.DataFrame,
    games: pl.DataFrame,
    season: int,
    week: int,
) -> tuple[Path, ...]:
    info = games.select("game_id", "away_team", "home_team", "kickoff_utc")
    final_model = model_rows.filter(pl.col("is_final")).join(info, on="game_id", how="left")
    final_blended = blended_rows.filter(pl.col("is_final")).join(info, on="game_id", how="left")
    if final_model.is_empty():
        return ()
    return (
        export_csv(final_model, season, week, "model"),
        export_csv(final_blended, season, week, "blended"),
    )


def run_predict(request: PredictRequest, now: datetime | None = None) -> PredictOutcome:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    games = week_games(request.season, request.week)
    upcoming = games.filter(pl.col("kickoff_utc") > moment)
    if upcoming.is_empty():
        raise PredictError(f"every game of {request.season} week {request.week} has kicked off")
    locked, refused = lock_plan(upcoming, moment, request.lock_hours) if request.final else ([], [])
    name, track, output = predict_model(request, upcoming, moment)
    config = {"season": request.season, "week": request.week, "model": name, "track": track}
    run_id = runlog.new_run_id({**config, "params": request.params}, moment)
    model_rows = write_model_rows(_model_rows(run_id, name, track, output, locked, moment))
    git_sha, git_dirty = runlog.git_state()
    record: dict[str, Any] = {
        **config,
        "run_id": run_id,
        "params": request.params,
        "created_at": moment.isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "final_games": locked,
        "model_rows_sha256": _rows_sha256(model_rows),
    }
    _write_record(run_id, record)
    to_blend = pl.concat([model_rows, unblended_finals(games, run_id)], how="vertical")
    params = current_params()
    opinions = load_opinions(request.season, request.week, games, params)
    blended_rows = write_blended_rows(_blended_rows(to_blend, opinions, params, moment))
    record.update(
        opinions_file=str(opinions.path) if opinions else None,
        opinions_sha256=opinions.sha256 if opinions else None,
        opinions_entered_at=opinions.entered_at.isoformat() if opinions else None,
        opinion_weight=params.weight,
        opinion_team_weight=params.team_weight,
    )
    _write_record(run_id, record)
    return PredictOutcome(
        run_id=run_id,
        games=games,
        model_rows=to_blend,
        blended_rows=blended_rows,
        opinions=opinions,
        locked=tuple(locked),
        refused=tuple(refused),
        exports=_export(to_blend, blended_rows, games, request.season, request.week),
    )
