import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from football.models.base import Track
from football.models.common import (
    Calibrator,
    check_track,
    classifier_rows,
    home_rate,
    labels,
    prediction_frame,
)

ELO_SCALE = 400.0


def _float_column(df: pl.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.full(df.height, np.nan)
    return df[name].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)


class HomeModel:
    def __init__(self, track: str = "independent") -> None:
        self.name = "home"
        self.track: Track = check_track(track)
        self.rate = 0.5

    def fit(self, train: pl.DataFrame) -> None:
        self.rate = home_rate(train)

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        return prediction_frame(games, np.full(games.height, self.rate))


class RecordModel:
    def __init__(self, track: str = "independent") -> None:
        self.name = "record"
        self.track: Track = check_track(track)
        self.home_pick_rate = 0.5
        self.away_pick_rate = 0.5

    def fit(self, train: pl.DataFrame) -> None:
        rows = classifier_rows(train)
        picks = self._home_pick(rows)
        self.home_pick_rate = home_rate(rows.filter(pl.Series(picks)))
        self.away_pick_rate = home_rate(rows.filter(pl.Series(~picks)))

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        probs = np.where(self._home_pick(games), self.home_pick_rate, self.away_pick_rate)
        return prediction_frame(games, probs)

    def _home_pick(self, df: pl.DataFrame) -> np.ndarray:
        diff = _float_column(df, "season_win_pct_diff")
        pair = _float_column(df, "home_season_win_pct") - _float_column(df, "away_season_win_pct")
        diff = np.where(np.isnan(diff), pair, diff)
        return np.nan_to_num(diff, nan=0.0) >= 0


class EloModel:
    def __init__(self, track: str = "independent", recalibrate: bool = False) -> None:
        self.name = "elo"
        self.track: Track = check_track(track)
        self.recalibrate = recalibrate
        self.fallback = 0.5
        self.calibrator = Calibrator("platt" if recalibrate else "none")

    def fit(self, train: pl.DataFrame) -> None:
        rows = classifier_rows(train)
        raw = self._raw(rows)
        self.fallback = float(np.nanmedian(raw)) if np.any(~np.isnan(raw)) else home_rate(rows)
        if self.recalibrate and rows.height:
            self.calibrator.fit(self._filled(raw), labels(rows))

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        probs = self.calibrator.apply(self._filled(self._raw(games)))
        return prediction_frame(games, probs)

    def _raw(self, df: pl.DataFrame) -> np.ndarray:
        prob = _float_column(df, "elo_home_prob")
        from_diff = 1.0 / (1.0 + 10.0 ** (-_float_column(df, "elo_diff") / ELO_SCALE))
        return np.where(np.isnan(prob), from_diff, prob)

    def _filled(self, raw: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(raw), self.fallback, raw)


class MarketModel:
    def __init__(self, track: str = "market", recalibrate: bool = False) -> None:
        self.name = "market"
        self.track: Track = check_track(track)
        if self.track != "market":
            raise ValueError("the market model only runs on track=market")
        self.recalibrate = recalibrate
        self.fallback = 0.5
        self.spread_model: LogisticRegression | None = None
        self.calibrator = Calibrator("platt" if recalibrate else "none")

    def fit(self, train: pl.DataFrame) -> None:
        rows = classifier_rows(train)
        self.fallback = home_rate(rows)
        spread = _float_column(rows, "market_home_spread")
        known = ~np.isnan(spread)
        y = labels(rows)
        self.spread_model = None
        if known.sum() >= 10 and len(np.unique(y[known])) == 2:
            self.spread_model = LogisticRegression(C=1e6, max_iter=1000)
            self.spread_model.fit(spread[known].reshape(-1, 1), y[known].astype(int))
        if self.recalibrate and rows.height:
            self.calibrator.fit(self._raw(rows), y)

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        return prediction_frame(games, self.calibrator.apply(self._raw(games)))

    def _raw(self, df: pl.DataFrame) -> np.ndarray:
        prob = _float_column(df, "market_home_prob")
        spread = _float_column(df, "market_home_spread")
        if self.spread_model is not None:
            known = ~np.isnan(spread)
            from_spread = np.full(df.height, np.nan)
            if known.any():
                fitted = self.spread_model.predict_proba(spread[known].reshape(-1, 1))[:, 1]
                from_spread[known] = fitted
            prob = np.where(np.isnan(prob), from_spread, prob)
        return np.where(np.isnan(prob), self.fallback, prob)
