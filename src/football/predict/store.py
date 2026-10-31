from datetime import datetime
from pathlib import Path

import polars as pl

from football import storage
from football.config import get_settings
from football.schemas import (
    GAMES,
    PREDICTIONS_BLENDED,
    PREDICTIONS_MODEL,
    UTC_TS,
    TableSpec,
    validate,
)

LAYER = "predictions"

BLENDED_STORED = TableSpec(
    PREDICTIONS_BLENDED.name,
    PREDICTIONS_BLENDED.keys,
    {
        **PREDICTIONS_BLENDED.columns,
        "user_pick": pl.String(),
        "user_conf": pl.Int32(),
        "user_pick_prob": pl.Float64(),
        "user_why": pl.String(),
        "entered_at": UTC_TS,
        "opinion_late": pl.Boolean(),
    },
)


class PredictError(RuntimeError):
    pass


class FinalLockError(PredictError):
    pass


def clean_games() -> pl.DataFrame:
    if not storage.exists("clean", GAMES.name):
        raise PredictError("clean games table is missing; run football ingest first")
    return storage.read("clean", GAMES.name)


def week_games(season: int, week: int) -> pl.DataFrame:
    games = clean_games().filter((pl.col("season") == season) & (pl.col("week") == week))
    if games.is_empty():
        raise PredictError(f"no games found for {season} week {week}")
    return games.sort("kickoff_utc", "game_id")


def next_week(season: int, now: datetime) -> int:
    upcoming = clean_games().filter((pl.col("season") == season) & (pl.col("kickoff_utc") > now))
    if upcoming.is_empty():
        raise PredictError(f"no upcoming games in {season}; pass --week")
    return int(upcoming.select(pl.col("week").min()).item())


def read_table(name: str) -> pl.DataFrame | None:
    return storage.read(LAYER, name) if storage.exists(LAYER, name) else None


def final_rows(name: str, game_ids: list[str]) -> pl.DataFrame:
    table = read_table(name)
    if table is None:
        return pl.DataFrame(schema={"run_id": pl.String(), "game_id": pl.String()})
    return table.filter(pl.col("is_final") & pl.col("game_id").is_in(game_ids))


def ensure_unlocked(name: str, game_ids: list[str]) -> None:
    locked = final_rows(name, game_ids).sort("game_id")
    if locked.is_empty():
        return
    lines = [
        f"refusing to overwrite final prediction for {row['game_id']}: locked by run "
        f"{row['run_id']} at {row['created_at'].isoformat(timespec='seconds')}"
        for row in locked.iter_rows(named=True)
    ]
    raise FinalLockError("\n".join(lines))


def _append_only(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    rows = validate(df, spec)
    existing = read_table(spec.name)
    if existing is not None:
        clash = existing.join(rows.select(spec.keys), on=list(spec.keys), how="semi")
        if not clash.is_empty():
            raise PredictError(
                f"predictions/{spec.name} already has rows for run {clash['run_id'][0]}"
            )
    ensure_unlocked(spec.name, rows.filter(pl.col("is_final"))["game_id"].to_list())
    storage.upsert(rows, LAYER, spec.name, spec.keys)
    return rows


def write_model_rows(df: pl.DataFrame) -> pl.DataFrame:
    return _append_only(df, PREDICTIONS_MODEL)


def write_blended_rows(df: pl.DataFrame) -> pl.DataFrame:
    return _append_only(df, BLENDED_STORED)


def latest_model_probs(game_ids: list[str]) -> dict[str, float]:
    model = read_table(PREDICTIONS_MODEL.name)
    if model is None:
        return {}
    latest = (
        model.filter(pl.col("game_id").is_in(game_ids))
        .sort("is_final", "created_at", descending=True)
        .unique("game_id", keep="first")
    )
    return dict(zip(latest["game_id"].to_list(), latest["home_win_prob"].to_list(), strict=True))


def export_dir() -> Path:
    path = get_settings().repo_dir / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_csv(df: pl.DataFrame, season: int, week: int, kind: str) -> Path:
    path = export_dir() / f"{season}-w{week:02d}-{kind}.csv"
    if path.exists():
        header = path.read_text().splitlines()[0].split(",")
        if header != df.columns:
            raise PredictError(f"{path} has columns {header}, expected {df.columns}")
        with path.open("a") as handle:
            df.write_csv(handle, include_header=False)
    else:
        df.write_csv(path)
    return path
