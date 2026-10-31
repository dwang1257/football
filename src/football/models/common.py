import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from scipy.special import logit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from football.config import get_settings
from football.features.columns import BENCHMARK_COLUMNS, MARKET_FEATURES, OWNER_FEATURES
from football.models.base import Track

PROB_EPS = 1e-3
WIN_TOTAL_COLUMNS: tuple[str, ...] = tuple(c for c in OWNER_FEATURES if "win_total" in c)
MARKET_COLUMNS: tuple[str, ...] = (*MARKET_FEATURES, *WIN_TOTAL_COLUMNS)

Calibration = Literal["none", "platt", "isotonic"]


def check_track(track: str) -> Track:
    if track == "independent":
        return "independent"
    if track == "market":
        return "market"
    raise ValueError(f"unknown track {track!r}; use 'independent' or 'market'")


def check_choice[T: str](value: str, choices: Sequence[T], what: str) -> T:
    for choice in choices:
        if value == choice:
            return choice
    raise ValueError(f"unknown {what} {value!r}; choose from {list(choices)}")


def forbidden_columns(track: Track) -> frozenset[str]:
    if track == "independent":
        return frozenset((*BENCHMARK_COLUMNS, *MARKET_COLUMNS))
    return frozenset(BENCHMARK_COLUMNS)


def resolve_seed(seed: int | None) -> int:
    return get_settings().seed if seed is None else seed


def game_seed(seed: int, game_id: str) -> list[int]:
    return [seed, zlib.crc32(game_id.encode())]


def classifier_rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("home_win").is_in([0.0, 1.0]))


def margin_rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("home_margin").is_not_null())


def labels(df: pl.DataFrame) -> np.ndarray:
    return df["home_win"].cast(pl.Float64).to_numpy()


def margins(df: pl.DataFrame) -> np.ndarray:
    return df["home_margin"].cast(pl.Float64).to_numpy()


def clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(probs, nan=0.5), PROB_EPS, 1 - PROB_EPS)


def normal_win_prob(mu: np.ndarray, sigma: float) -> np.ndarray:
    return norm.cdf(mu / sigma)


def prediction_frame(
    games: pl.DataFrame, probs: np.ndarray, home_margin: np.ndarray | None = None
) -> pl.DataFrame:
    margin_values = [None] * games.height if home_margin is None else home_margin
    return pl.DataFrame(
        {
            "game_id": games["game_id"],
            "home_win_prob": pl.Series(clip_probs(np.asarray(probs, dtype=float))),
            "home_margin": pl.Series(margin_values, dtype=pl.Float64),
        },
        schema={"game_id": pl.String, "home_win_prob": pl.Float64, "home_margin": pl.Float64},
    )


def seasons_of(df: pl.DataFrame) -> list[int]:
    return sorted(int(s) for s in df["season"].unique().to_list())


def season_folds(df: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    seasons = seasons_of(df)
    if len(seasons) < 2:
        return
    for season in seasons:
        yield df.filter(pl.col("season") != season), df.filter(pl.col("season") == season)


def split_last_season(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    seasons = seasons_of(df)
    if len(seasons) < 2:
        return None
    last = seasons[-1]
    return df.filter(pl.col("season") < last), df.filter(pl.col("season") == last)


def residual_sd(residuals: np.ndarray, n_params: int = 0) -> float:
    n = residuals.size
    dof = max(n - n_params, 1)
    return float(np.sqrt(np.sum(residuals**2) / dof))


def home_rate(df: pl.DataFrame) -> float:
    rows = classifier_rows(df)
    if rows.height == 0:
        return 0.5
    return float((labels(rows).sum() + 1.0) / (rows.height + 2.0))


@dataclass(frozen=True)
class Feature:
    name: str
    expr: pl.Expr
    sources: tuple[str, ...]


def column(name: str) -> Feature:
    return Feature(name, pl.col(name).cast(pl.Float64), (name,))


def difference(name: str, home: str, away: str) -> Feature:
    return Feature(
        name, pl.col(home).cast(pl.Float64) - pl.col(away).cast(pl.Float64), (home, away)
    )


def paired_features(columns: Sequence[str]) -> list[Feature]:
    present = set(columns)
    features: list[Feature] = []
    names: set[str] = set()
    for col in columns:
        feature = _pair_feature(col, present)
        if feature is None or feature.name in names:
            continue
        names.add(feature.name)
        features.append(feature)
    return features


def _pair_feature(col: str, present: set[str]) -> Feature | None:
    if col.startswith("home_") and f"away_{col[5:]}" in present:
        name = f"{col[5:]}_diff"
        return None if name in present else difference(name, col, f"away_{col[5:]}")
    if col.endswith("_home") and f"{col[:-5]}_away" in present:
        name = f"{col[:-5]}_diff"
        return None if name in present else difference(name, col, f"{col[:-5]}_away")
    if col.startswith("away_") and f"home_{col[5:]}" in present:
        return None
    if col.endswith("_away") and f"{col[:-5]}_home" in present:
        return None
    return column(col)


def with_sources(df: pl.DataFrame, sources: Sequence[str]) -> pl.DataFrame:
    missing = [s for s in dict.fromkeys(sources) if s not in df.columns]
    if not missing:
        return df
    return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(s) for s in missing)


@dataclass
class Design:
    candidates: list[Feature]
    track: Track
    standardize: bool = True
    impute: bool = True
    features: list[Feature] = field(default_factory=list[Feature])
    dropped: dict[str, str] = field(default_factory=dict[str, str])
    _center: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    _medians: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    def fit(self, train: pl.DataFrame) -> np.ndarray:
        forbidden = forbidden_columns(self.track)
        leaked = sorted({s for f in self.candidates for s in f.sources} & forbidden)
        if leaked:
            raise ValueError(f"track {self.track} may not use columns {leaked}")
        self.features, self.dropped = [], {}
        for feature in self.candidates:
            reason = self._drop_reason(feature, train)
            if reason is None:
                self.features.append(feature)
            else:
                self.dropped[feature.name] = reason
        raw = self._raw(train)
        self._medians = np.nanmedian(raw, axis=0) if raw.shape[1] else np.zeros(0)
        filled = self._fill(raw)
        self._center = np.mean(filled, axis=0) if raw.shape[1] else np.zeros(0)
        scale = np.std(filled, axis=0) if raw.shape[1] else np.ones(0)
        self._scale = np.where(scale > 0, scale, 1.0)
        return self._finish(filled)

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        return self._finish(self._fill(self._raw(df)))

    def _drop_reason(self, feature: Feature, train: pl.DataFrame) -> str | None:
        missing = [s for s in feature.sources if s not in train.columns]
        if missing:
            return f"missing columns {missing}"
        values = train.select(feature.expr.alias("v"))["v"].drop_nulls().drop_nans()
        if values.len() == 0:
            return "all null in train"
        if values.n_unique() < 2:
            return "constant in train"
        return None

    def _raw(self, df: pl.DataFrame) -> np.ndarray:
        if not self.features:
            return np.zeros((df.height, 0))
        frame = with_sources(df, [s for f in self.features for s in f.sources])
        out = frame.select(f.expr.cast(pl.Float64).alias(f.name) for f in self.features)
        return out.to_numpy().astype(float)

    def _fill(self, raw: np.ndarray) -> np.ndarray:
        if not self.impute:
            return raw
        return np.where(np.isnan(raw), self._medians, raw)

    def _finish(self, filled: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return filled
        return (filled - self._center) / self._scale


class Calibrator:
    def __init__(self, method: Calibration) -> None:
        self.method: Calibration = method
        self._platt: LogisticRegression | None = None
        self._isotonic: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "platt":
            self._platt = LogisticRegression(C=1e6, max_iter=1000)
            self._platt.fit(_logit_column(probs), y.astype(int))
        elif self.method == "isotonic":
            self._isotonic = IsotonicRegression(
                y_min=PROB_EPS, y_max=1 - PROB_EPS, out_of_bounds="clip"
            )
            self._isotonic.fit(clip_probs(probs), y)
        return self

    def apply(self, probs: np.ndarray) -> np.ndarray:
        if self._platt is not None:
            return self._platt.predict_proba(_logit_column(probs))[:, 1]
        if self._isotonic is not None:
            return np.asarray(self._isotonic.predict(clip_probs(probs)), dtype=float)
        return probs


def _logit_column(probs: np.ndarray) -> np.ndarray:
    return logit(clip_probs(probs)).reshape(-1, 1)
