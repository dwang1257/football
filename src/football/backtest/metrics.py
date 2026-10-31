import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import binomtest

from football.config import get_settings

Array = NDArray[np.float64]

PROB_EPS = 1e-15
ECE_BINS = 10
BOOTSTRAP_DRAWS = 10_000
ADOPT_SEASON_SHARE = 0.7
REGULAR = "REG"
EARLY_WEEKS = 4

SUMMARY_KEYS: tuple[str, ...] = ("n", "ties", "accuracy", "log_loss", "brier", "ece")


def _arrays(df: pl.DataFrame, prob_col: str = "home_win_prob") -> tuple[Array, Array]:
    labeled = df.filter(pl.col("home_win").is_not_null())
    y = labeled["home_win"].cast(pl.Float64).to_numpy()
    p = labeled[prob_col].cast(pl.Float64).to_numpy()
    return y, p


def pick_credit(y: Array, p: Array) -> Array:
    credit = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1.0)).astype(np.float64))
    return np.where(y == 0.5, np.nan, credit)


def accuracy(y: Array, p: Array) -> float:
    credit = pick_credit(y, p)
    credit = credit[~np.isnan(credit)]
    return float(credit.mean()) if credit.size else math.nan


def log_losses(y: Array, p: Array) -> Array:
    q = np.clip(p, PROB_EPS, 1.0 - PROB_EPS)
    return -(y * np.log(q) + (1.0 - y) * np.log1p(-q))


def brier_scores(y: Array, p: Array) -> Array:
    return (p - y) ** 2


def log_loss(y: Array, p: Array) -> float:
    return float(log_losses(y, p).mean()) if y.size else math.nan


def brier(y: Array, p: Array) -> float:
    return float(brier_scores(y, p).mean()) if y.size else math.nan


def reliability(y: Array, p: Array, bins: int = ECE_BINS) -> pl.DataFrame:
    index = np.minimum((p * bins).astype(np.int64), bins - 1)
    frame = pl.DataFrame({"bin": index, "y": y, "p": p})
    counts = (
        frame.group_by("bin")
        .agg(
            n=pl.len(),
            mean_prob=pl.col("p").mean(),
            observed=pl.col("y").mean(),
        )
        .with_columns(n=pl.col("n").cast(pl.Int64))
    )
    grid = pl.DataFrame({"bin": np.arange(bins, dtype=np.int64)}).with_columns(
        lower=pl.col("bin") / bins, upper=(pl.col("bin") + 1) / bins
    )
    return (
        grid.join(counts, on="bin", how="left")
        .with_columns(pl.col("n").fill_null(0))
        .select("bin", "lower", "upper", "n", "mean_prob", "observed")
        .sort("bin")
    )


def expected_calibration_error(y: Array, p: Array, bins: int = ECE_BINS) -> float:
    if not y.size:
        return math.nan
    table = reliability(y, p, bins).filter(pl.col("n") > 0)
    gaps = (table["mean_prob"] - table["observed"]).abs() * table["n"]
    return float(gaps.sum()) / y.size


def summary(df: pl.DataFrame, prob_col: str = "home_win_prob") -> dict[str, Any]:
    y, p = _arrays(df, prob_col)
    return {
        "n": int(y.size),
        "ties": int((y == 0.5).sum()),
        "accuracy": accuracy(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": expected_calibration_error(y, p),
    }


def split_filters(df: pl.DataFrame) -> list[tuple[str, pl.Expr]]:
    regular = pl.col("game_type") == REGULAR
    named = [
        ("overall", pl.lit(True)),
        ("weeks 1-4", regular & (pl.col("week") <= EARLY_WEEKS)),
        ("weeks 5+", regular & (pl.col("week") > EARLY_WEEKS)),
        ("playoffs", ~regular),
    ]
    seasons = sorted(df["season"].unique().to_list())
    return named + [(str(season), pl.col("season") == season) for season in seasons]


def metrics_by_split(df: pl.DataFrame, prob_col: str = "home_win_prob") -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, condition in split_filters(df):
        part = df.filter(condition)
        if part.filter(pl.col("home_win").is_not_null()).height:
            rows.append({"split": name, **summary(part, prob_col)})
    schema = {
        "split": pl.String(),
        "n": pl.Int64(),
        "ties": pl.Int64(),
        "accuracy": pl.Float64(),
        "log_loss": pl.Float64(),
        "brier": pl.Float64(),
        "ece": pl.Float64(),
    }
    return pl.DataFrame(rows, schema=schema)


def paired(candidate: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    left = candidate.select("game_id", p_candidate=pl.col("home_win_prob"))
    right = reference.select(
        "game_id", "season", "week", "game_type", "home_win", p_reference=pl.col("home_win_prob")
    )
    return (
        right.join(left, on="game_id", how="inner")
        .filter(pl.col("home_win").is_not_null())
        .sort("season", "week", "game_id")
    )


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float

    def excludes_zero(self) -> bool:
        return self.hi < 0.0 or self.lo > 0.0


@dataclass(frozen=True)
class BootstrapResult:
    n_games: int
    n_weeks: int
    draws: int
    log_loss: Interval
    brier: Interval


def paired_bootstrap(
    pairs: pl.DataFrame, draws: int = BOOTSTRAP_DRAWS, seed: int | None = None
) -> BootstrapResult:
    if not pairs.height:
        raise ValueError("paired bootstrap needs at least one common labeled game")
    y, p_ref = _arrays(pairs, "p_reference")
    _, p_cand = _arrays(pairs, "p_candidate")
    diffs = pairs.select("season", "week").with_columns(
        d_log_loss=log_losses(y, p_cand) - log_losses(y, p_ref),
        d_brier=brier_scores(y, p_cand) - brier_scores(y, p_ref),
    )
    weekly = (
        diffs.group_by("season", "week")
        .agg(n=pl.len(), d_log_loss=pl.col("d_log_loss").sum(), d_brier=pl.col("d_brier").sum())
        .sort("season", "week")
    )
    counts = weekly["n"].cast(pl.Float64).to_numpy()
    rng = np.random.default_rng(get_settings().seed if seed is None else seed)
    index = rng.integers(0, weekly.height, size=(draws, weekly.height))
    totals = counts[index].sum(axis=1)

    def interval(column: str) -> Interval:
        sums = weekly[column].to_numpy()
        samples = sums[index].sum(axis=1) / totals
        lo, hi = np.quantile(samples, [0.025, 0.975])
        observed = float(sums.sum() / counts.sum())
        return Interval(mean=observed, lo=float(lo), hi=float(hi))

    return BootstrapResult(
        n_games=pairs.height,
        n_weeks=weekly.height,
        draws=draws,
        log_loss=interval("d_log_loss"),
        brier=interval("d_brier"),
    )


@dataclass(frozen=True)
class McNemarResult:
    candidate_only: int
    reference_only: int
    p_value: float


def mcnemar(pairs: pl.DataFrame) -> McNemarResult:
    y, p_ref = _arrays(pairs, "p_reference")
    _, p_cand = _arrays(pairs, "p_candidate")
    cand = pick_credit(y, p_cand)
    ref = pick_credit(y, p_ref)
    candidate_only = int(((cand == 1.0) & (ref == 0.0)).sum())
    reference_only = int(((cand == 0.0) & (ref == 1.0)).sum())
    discordant = candidate_only + reference_only
    p_value = binomtest(candidate_only, discordant, 0.5).pvalue if discordant else 1.0
    return McNemarResult(candidate_only, reference_only, float(p_value))


@dataclass(frozen=True)
class SeasonsWon:
    won: int
    total: int

    @property
    def share(self) -> float:
        return self.won / self.total if self.total else math.nan


def seasons_won(pairs: pl.DataFrame) -> SeasonsWon:
    won = 0
    seasons = sorted(pairs["season"].unique().to_list())
    for season in seasons:
        part = pairs.filter(pl.col("season") == season)
        y, p_ref = _arrays(part, "p_reference")
        _, p_cand = _arrays(part, "p_candidate")
        won += int(log_loss(y, p_cand) < log_loss(y, p_ref))
    return SeasonsWon(won=won, total=len(seasons))


@dataclass(frozen=True)
class Comparison:
    candidate: dict[str, Any]
    reference: dict[str, Any]
    bootstrap: BootstrapResult
    mcnemar: McNemarResult
    seasons: SeasonsWon

    @property
    def adoptable(self) -> bool:
        ci_better = self.bootstrap.log_loss.hi < 0.0
        return ci_better or self.seasons.share >= ADOPT_SEASON_SHARE


def compare(
    candidate: pl.DataFrame,
    reference: pl.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int | None = None,
) -> Comparison:
    pairs = paired(candidate, reference)
    return Comparison(
        candidate=summary(pairs, "p_candidate"),
        reference=summary(pairs, "p_reference"),
        bootstrap=paired_bootstrap(pairs, draws, seed),
        mcnemar=mcnemar(pairs),
        seasons=seasons_won(pairs),
    )
