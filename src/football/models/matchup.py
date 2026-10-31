from dataclasses import dataclass, field
from typing import Any, Literal, overload

import numpy as np
import polars as pl

from football.models.base import Track
from football.models.common import (
    check_choice,
    check_track,
    game_seed,
    margin_rows,
    margins,
    normal_win_prob,
    prediction_frame,
    residual_sd,
    resolve_seed,
    season_folds,
    with_sources,
)
from football.schemas import MATCHUP_STATS

Combine = Literal["league_relative", "average"]
Mode = Literal["simulate", "analytic"]
COMBINES: tuple[Combine, ...] = ("league_relative", "average")
MODES: tuple[Mode, ...] = ("simulate", "analytic")
SIDES = ("home", "away")
WIND_COLUMN = "fcst_wind_mph"
WIND_STAT = "pass_yards_per_play"
DEFAULT_ALPHA = 10.0


@overload
def expected_stat(
    off: float, allowed: float, league: float, combine: Combine, multiplicative: bool = True
) -> float: ...
@overload
def expected_stat(
    off: np.ndarray,
    allowed: np.ndarray,
    league: np.ndarray,
    combine: Combine,
    multiplicative: bool = True,
) -> np.ndarray: ...
@overload
def expected_stat(
    off: pl.Expr, allowed: pl.Expr, league: pl.Expr, combine: Combine, multiplicative: bool = True
) -> pl.Expr: ...
def expected_stat(
    off: Any, allowed: Any, league: Any, combine: Combine, multiplicative: bool = True
) -> Any:
    if combine == "average":
        return (off + allowed) / 2
    if multiplicative:
        return off * allowed / league
    return off + allowed - league


def combined_sd(
    off: np.ndarray,
    allowed: np.ndarray,
    league: np.ndarray,
    sd_off: np.ndarray,
    sd_allowed: np.ndarray,
    combine: Combine,
    multiplicative: bool = True,
) -> np.ndarray:
    if combine == "league_relative" and multiplicative:
        sd_off = sd_off * allowed / league
        sd_allowed = sd_allowed * off / league
    return np.sqrt((sd_off**2 + sd_allowed**2) / 2)


def stat_columns(stat: str) -> tuple[str, ...]:
    return (*(f"{side}_{unit}_{stat}" for side in SIDES for unit in ("off", "def")), f"lg_{stat}")


def sd_columns(stat: str) -> tuple[str, ...]:
    return tuple(f"{side}_{unit}_{stat}_sd" for side in SIDES for unit in ("off", "def"))


def _usable(train: pl.DataFrame, columns: tuple[str, ...]) -> str | None:
    missing = [c for c in columns if c not in train.columns]
    if missing:
        return f"missing columns {missing}"
    empty = [c for c in columns if train[c].null_count() == train.height]
    if empty:
        return f"all null in train {empty}"
    return None


def usable_stats(train: pl.DataFrame) -> tuple[list[str], dict[str, str]]:
    stats: list[str] = []
    dropped: dict[str, str] = {}
    for stat in MATCHUP_STATS:
        reason = _usable(train, stat_columns(stat))
        if reason is None:
            stats.append(stat)
        else:
            dropped[stat] = reason
    return stats, dropped


def multiplicative_stats(train: pl.DataFrame, stats: list[str]) -> frozenset[str]:
    out: set[str] = set()
    for stat in stats:
        cols = stat_columns(stat)
        team = train.select(pl.col(c).cast(pl.Float64) for c in cols[:4]).to_numpy()
        league = train[cols[4]].cast(pl.Float64).to_numpy()
        if np.nanmin(league) > 0 and np.nanmin(team) >= 0:
            out.add(stat)
    return frozenset(out)


def matchup_expressions(
    stats: list[str], multiplicative: frozenset[str], combine: Combine
) -> dict[str, pl.Expr]:
    exprs: dict[str, pl.Expr] = {}
    for stat in stats:
        h_off, h_def, a_off, a_def, lg = (pl.col(c).cast(pl.Float64) for c in stat_columns(stat))
        mult = stat in multiplicative
        home = expected_stat(h_off, a_def, lg, combine, mult)
        away = expected_stat(a_off, h_def, lg, combine, mult)
        exprs[f"mx_{stat}_diff"] = home - away
    return exprs


@dataclass(frozen=True)
class Inputs:
    home: np.ndarray
    away: np.ndarray
    home_sd: np.ndarray
    away_sd: np.ndarray
    context: np.ndarray


@dataclass
class PointsModel:
    combine: Combine
    alpha: float
    stats: list[str] = field(default_factory=list[str])
    multiplicative: frozenset[str] = frozenset()
    context: list[str] = field(default_factory=list[str])
    dropped: dict[str, str] = field(default_factory=dict[str, str])
    medians: dict[str, float] = field(default_factory=dict[str, float])
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    context_scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def stat_weights(self) -> np.ndarray:
        return self.coef[: len(self.stats)] / self.scale

    def fit(self, rows: pl.DataFrame) -> "PointsModel":
        self.stats, self.dropped = usable_stats(rows)
        self.multiplicative = multiplicative_stats(rows, self.stats)
        sources = [c for s in self.stats for c in (*stat_columns(s), *sd_columns(s))]
        for col in sources:
            if col not in rows.columns or rows[col].null_count() == rows.height:
                self.dropped.setdefault(col, "missing or all null in train, sd taken as 0")
        self.medians = _medians(rows, [*sources, *_context_sources(rows)])
        partial = self._stat_inputs(rows)
        both = np.vstack([partial.home, partial.away])
        scale = np.std(both, axis=0) if both.size else np.ones(0)
        self.scale = np.where(scale > 0, scale, 1.0)
        raw_context, names = self._raw_context(rows, partial)
        keep = [i for i in range(len(names)) if np.any(raw_context[:, i] != 0)]
        for i, name in enumerate(names):
            if i not in keep:
                self.dropped[name] = "missing or all zero in train"
        self.context = [names[i] for i in keep]
        context_scale = np.sqrt(np.mean(raw_context[:, keep] ** 2, axis=0))
        self.context_scale = np.where(context_scale > 0, context_scale, 1.0)
        design = self._design(self.inputs(rows))
        self.coef = _ridge(design, margins(rows), self.alpha)
        return self

    def inputs(self, df: pl.DataFrame) -> Inputs:
        partial = self._stat_inputs(df)
        raw_context, names = self._raw_context(df, partial)
        index = [names.index(n) for n in self.context]
        return Inputs(
            partial.home,
            partial.away,
            partial.home_sd,
            partial.away_sd,
            raw_context[:, index] / self.context_scale,
        )

    def mu(self, inputs: Inputs) -> np.ndarray:
        return self._design(inputs) @ self.coef

    def context_term(self, inputs: Inputs) -> np.ndarray:
        return inputs.context @ self.coef[len(self.stats) :]

    def _design(self, inputs: Inputs) -> np.ndarray:
        return np.hstack([(inputs.home - inputs.away) / self.scale, inputs.context])

    def _stat_inputs(self, df: pl.DataFrame) -> Inputs:
        n = df.height
        k = len(self.stats)
        values = self._values(
            df, [c for s in self.stats for c in (*stat_columns(s), *sd_columns(s))]
        )
        home, away = np.zeros((n, k)), np.zeros((n, k))
        home_sd, away_sd = np.zeros((n, k)), np.zeros((n, k))
        for j, stat in enumerate(self.stats):
            h_off, h_def, a_off, a_def, lg = (values[c] for c in stat_columns(stat))
            sh_off, sh_def, sa_off, sa_def = (values[c] for c in sd_columns(stat))
            mult = stat in self.multiplicative
            home[:, j] = expected_stat(h_off, a_def, lg, self.combine, mult)
            away[:, j] = expected_stat(a_off, h_def, lg, self.combine, mult)
            home_sd[:, j] = combined_sd(h_off, a_def, lg, sh_off, sa_def, self.combine, mult)
            away_sd[:, j] = combined_sd(a_off, h_def, lg, sa_off, sh_def, self.combine, mult)
        return Inputs(home, away, home_sd, away_sd, np.zeros((n, 0)))

    def _values(self, df: pl.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
        frame = with_sources(df, columns).select(
            pl.col(c).cast(pl.Float64).fill_nan(None).fill_null(self.medians.get(c, 0.0))
            for c in dict.fromkeys(columns)
        )
        return {c: frame[c].to_numpy().astype(float) for c in frame.columns}

    def _raw_context(self, df: pl.DataFrame, partial: Inputs) -> tuple[np.ndarray, list[str]]:
        values = self._values(df, [c for c in _CONTEXT_SOURCES if c != "neutral_site"])
        neutral = with_sources(df, ["neutral_site"])["neutral_site"].cast(pl.Float64)
        columns: dict[str, np.ndarray] = {"home_field": 1.0 - neutral.fill_null(0.0).to_numpy()}
        columns["rest_diff"] = values["rest_diff"]
        columns["travel_diff"] = values["travel_km_away"] - values["travel_km_home"]
        columns["qb_value_diff"] = values["qb_value_diff"]
        if WIND_STAT in self.stats:
            j = self.stats.index(WIND_STAT)
            wind_pass = values[WIND_COLUMN] * (partial.home[:, j] - partial.away[:, j])
            columns["wind_pass_diff"] = wind_pass
        names = list(columns)
        return np.column_stack([columns[n] for n in names]), names


_CONTEXT_SOURCES = (
    "neutral_site",
    "rest_diff",
    "travel_km_home",
    "travel_km_away",
    "qb_value_diff",
    WIND_COLUMN,
)


def _context_sources(rows: pl.DataFrame) -> list[str]:
    return [c for c in _CONTEXT_SOURCES if c in rows.columns]


def _medians(rows: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    present = [c for c in dict.fromkeys(columns) if c in rows.columns]
    if not present or rows.height == 0:
        return {}
    medians = rows.select(pl.col(c).cast(pl.Float64).fill_nan(None).median() for c in present)
    return {c: float(v) for c, v in medians.row(0, named=True).items() if v is not None}


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def oof_sigma(rows: pl.DataFrame, combine: Combine, alpha: float) -> float:
    residuals: list[np.ndarray] = []
    for fit_rows, hold in season_folds(rows):
        model = PointsModel(combine, alpha).fit(fit_rows)
        residuals.append(margins(hold) - model.mu(model.inputs(hold)))
    if residuals:
        return residual_sd(np.concatenate(residuals))
    model = PointsModel(combine, alpha).fit(rows)
    return residual_sd(margins(rows) - model.mu(model.inputs(rows)), model.coef.size)


class MatchupModel:
    def __init__(
        self,
        track: str = "independent",
        combine: str = "league_relative",
        mode: str = "simulate",
        n_sims: int = 10000,
        alpha: float = DEFAULT_ALPHA,
        seed: int | None = None,
    ) -> None:
        self.name = "matchup"
        self.track: Track = check_track(track)
        self.combine: Combine = check_choice(combine, COMBINES, "combine")
        self.mode: Mode = check_choice(mode, MODES, "mode")
        if n_sims < 1:
            raise ValueError("n_sims must be positive")
        self.n_sims = int(n_sims)
        self.alpha = float(alpha)
        self.seed = resolve_seed(seed)
        self.points = PointsModel(self.combine, self.alpha)
        self.sigma = 13.5
        self.dropped: dict[str, str] = {}

    def fit(self, train: pl.DataFrame) -> None:
        rows = margin_rows(train)
        self.points = PointsModel(self.combine, self.alpha).fit(rows)
        self.sigma = oof_sigma(rows, self.combine, self.alpha)
        self.dropped = dict(self.points.dropped)

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        inputs = self.points.inputs(games)
        mu = self.points.mu(inputs)
        if self.mode == "analytic":
            probs = normal_win_prob(mu, self.sigma)
        else:
            probs = self._simulate(games["game_id"].to_list(), inputs)
        return prediction_frame(games, probs, mu)

    def _simulate(self, game_ids: list[str], inputs: Inputs) -> np.ndarray:
        weights = self.points.stat_weights
        context = self.points.context_term(inputs)
        stat_var = ((inputs.home_sd**2 + inputs.away_sd**2) * weights**2).sum(axis=1)
        shrink = np.sqrt(np.minimum(1.0, self.sigma**2 / np.maximum(stat_var, 1e-12)))
        noise_sd = np.sqrt(np.maximum(self.sigma**2 - stat_var * shrink**2, 0.0))
        probs = np.empty(len(game_ids))
        k = weights.size
        for i, gid in enumerate(game_ids):
            rng = np.random.default_rng(game_seed(self.seed, gid))
            home_sd = shrink[i] * inputs.home_sd[i]
            away_sd = shrink[i] * inputs.away_sd[i]
            home = inputs.home[i] + home_sd * rng.standard_normal((self.n_sims, k))
            away = inputs.away[i] + away_sd * rng.standard_normal((self.n_sims, k))
            noise = noise_sd[i] * rng.standard_normal(self.n_sims)
            sims = (home - away) @ weights + context[i] + noise
            probs[i] = np.mean(sims > 0) + 0.5 * np.mean(sims == 0)
        return probs
