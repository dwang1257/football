import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from football.opinions.blend import (
    CONF_LEVELS,
    BlendParams,
    blend,
    default_conf_prob,
)
from football.storage import layer_dir

WEIGHTS_FILE = "opinion_weights.json"
MIN_GAMES = 40
PRIOR_GAMES = 100
WEIGHT_GRID_MAX = 0.5
WEIGHT_CLAMP_MAX = 0.30
TEAM_WEIGHT_GRID_MAX = 2.0
TEAM_WEIGHT_CLAMP_MAX = 1.25
GRID_POINTS = 101
CONF_PRIOR_GAMES = 2
CONF_STEP = 0.001
CONF_BOUNDS = (0.5, 0.99)
LOSS_EPS = 1e-12

HISTORY_COLUMNS: tuple[str, ...] = (
    "model_prob",
    "user_pick_prob",
    "user_conf",
    "user_picked_home",
    "team_nudge_points",
    "home_win",
)


@dataclass(frozen=True)
class FittedWeights:
    weight: float
    team_weight: float
    conf_table: dict[int, float]
    n_games: int
    n_team_games: int
    fitted_at: datetime


def weights_path() -> Path:
    return layer_dir("runs") / WEIGHTS_FILE


def save_weights(fitted: FittedWeights) -> Path:
    payload = asdict(fitted)
    payload["conf_table"] = {str(k): v for k, v in fitted.conf_table.items()}
    payload["fitted_at"] = fitted.fitted_at.isoformat()
    path = weights_path()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_weights() -> FittedWeights | None:
    path = weights_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return FittedWeights(
        weight=float(payload["weight"]),
        team_weight=float(payload["team_weight"]),
        conf_table={int(k): float(v) for k, v in payload["conf_table"].items()},
        n_games=int(payload["n_games"]),
        n_team_games=int(payload["n_team_games"]),
        fitted_at=datetime.fromisoformat(payload["fitted_at"]),
    )


def current_params() -> BlendParams:
    base = BlendParams.from_settings()
    fitted = load_weights()
    if fitted is None:
        return base
    return BlendParams(
        weight=fitted.weight,
        team_weight=fitted.team_weight,
        cap=base.cap,
        cap_max_conf=base.cap_max_conf,
        conf_table=fitted.conf_table,
    )


def log_loss(y: float, p: float) -> float:
    q = min(max(p, LOSS_EPS), 1.0 - LOSS_EPS)
    return -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))


def _isotonic(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    blocks: list[tuple[float, float, int]] = []
    for value, weight in zip(values, weights, strict=True):
        blocks.append((value, weight, 1))
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append(((v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2))
    return [value for value, _, count in blocks for _ in range(count)]


def fit_conf_table(history: pl.DataFrame) -> dict[int, float]:
    rated = history.filter(pl.col("user_conf").is_not_null())
    raw: list[float] = []
    weights: list[float] = []
    for conf in CONF_LEVELS:
        level = rated.filter(pl.col("user_conf") == conf)
        hit = (
            pl.when(pl.col("user_picked_home"))
            .then(pl.col("home_win"))
            .otherwise(1.0 - pl.col("home_win"))
        )
        hits = float(level.select(hit.sum()).item() or 0.0)
        n = level.height
        raw.append((hits + CONF_PRIOR_GAMES * default_conf_prob(conf)) / (n + CONF_PRIOR_GAMES))
        weights.append(float(n + CONF_PRIOR_GAMES))
    low, high = CONF_BOUNDS
    table: dict[int, float] = {}
    previous = -math.inf
    for conf, value in zip(CONF_LEVELS, _isotonic(raw, weights), strict=True):
        previous = max(min(max(value, low), high), previous + CONF_STEP)
        table[conf] = previous
    return table


def _user_probs(history: pl.DataFrame, table: dict[int, float]) -> list[float | None]:
    probs: list[float | None] = []
    for row in history.iter_rows(named=True):
        conf = row["user_conf"]
        pick_prob = table[conf] if conf is not None else row["user_pick_prob"]
        if pick_prob is None:
            probs.append(None)
        else:
            probs.append(pick_prob if row["user_picked_home"] else 1.0 - pick_prob)
    return probs


def _total_loss(
    history: pl.DataFrame, user_probs: list[float | None], params: BlendParams
) -> float:
    total = 0.0
    for row, user_prob in zip(history.iter_rows(named=True), user_probs, strict=True):
        result = blend(
            row["model_prob"], user_prob, row["team_nudge_points"] or 0.0, row["user_conf"], params
        )
        total += log_loss(row["home_win"], result.blended_prob)
    return total


def fit_shrunk(
    loss: Callable[[float], float], n: int, default: float, grid_max: float, clamp_max: float
) -> float:
    if n < MIN_GAMES:
        return default
    grid = np.linspace(0.0, grid_max, GRID_POINTS)
    best = float(grid[int(np.argmin([loss(float(value)) for value in grid]))])
    shrunk = (n * best + PRIOR_GAMES * default) / (n + PRIOR_GAMES)
    return min(max(shrunk, 0.0), clamp_max)


def fit_weights(history: pl.DataFrame, defaults: BlendParams, now: datetime) -> FittedWeights:
    resolved = history.select(HISTORY_COLUMNS).filter(pl.col("home_win").is_not_null())
    table = fit_conf_table(resolved)
    params = BlendParams(
        weight=defaults.weight,
        team_weight=defaults.team_weight,
        cap=defaults.cap,
        cap_max_conf=defaults.cap_max_conf,
        conf_table=table,
    )
    games = resolved.filter(pl.col("user_pick_prob").is_not_null())
    game_probs = _user_probs(games, table)
    weight = fit_shrunk(
        lambda w: _total_loss(games, game_probs, params.with_weights(weight=w)),
        games.height,
        defaults.weight,
        WEIGHT_GRID_MAX,
        WEIGHT_CLAMP_MAX,
    )
    nudged = resolved.filter(pl.col("team_nudge_points").fill_null(0.0) != 0.0)
    nudged_probs = _user_probs(nudged, table)
    team_weight = fit_shrunk(
        lambda wt: _total_loss(
            nudged, nudged_probs, params.with_weights(weight=weight, team_weight=wt)
        ),
        nudged.height,
        defaults.team_weight,
        TEAM_WEIGHT_GRID_MAX,
        TEAM_WEIGHT_CLAMP_MAX,
    )
    return FittedWeights(
        weight=weight,
        team_weight=team_weight,
        conf_table=table,
        n_games=games.height,
        n_team_games=nudged.height,
        fitted_at=now.astimezone(UTC),
    )
