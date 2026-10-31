from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from football.models.base import Track
from football.models.common import (
    Design,
    check_choice,
    check_track,
    classifier_rows,
    labels,
    prediction_frame,
    resolve_seed,
    split_last_season,
)
from football.models.featuresets import DEFAULT_GROUPS, feature_set, parse_groups
from football.models.matchup import COMBINES, Combine

MONOTONE_INCREASING = frozenset({"elo_diff", "elo_home_prob"})


class LgbmModel:
    def __init__(
        self,
        track: str = "independent",
        features: str = DEFAULT_GROUPS,
        num_leaves: int = 7,
        min_data_in_leaf: int = 50,
        learning_rate: float = 0.03,
        lambda_l2: float = 1.0,
        feature_fraction: float = 0.8,
        max_rounds: int = 2000,
        early_stopping_rounds: int = 100,
        default_rounds: int = 300,
        combine: str = "league_relative",
        seed: int | None = None,
    ) -> None:
        self.name = "lgbm"
        self.track: Track = check_track(track)
        self.groups = parse_groups(features, self.track)
        self.combine: Combine = check_choice(combine, COMBINES, "combine")
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.learning_rate = float(learning_rate)
        self.lambda_l2 = float(lambda_l2)
        self.feature_fraction = float(feature_fraction)
        self.max_rounds = int(max_rounds)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.default_rounds = int(default_rounds)
        self.seed = resolve_seed(seed)
        self.design = Design([], self.track, standardize=False, impute=False)
        self.booster: lgb.Booster | None = None
        self.rounds = self.default_rounds
        self.dropped: dict[str, str] = {}

    def fit(self, train: pl.DataFrame) -> None:
        rows = classifier_rows(train)
        candidates = feature_set(self.groups, rows, self.combine)
        self.design = Design(candidates, self.track, standardize=False, impute=False)
        x = self.design.fit(rows)
        self.dropped = dict(self.design.dropped)
        params = self._params()
        self.rounds = self._best_rounds(rows, params)
        self.booster = lgb.train(params, self._dataset(x, labels(rows)), self.rounds)

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        if self.booster is None:
            raise RuntimeError("fit before predict")
        probs = np.asarray(self.booster.predict(self.design.transform(games)), dtype=float)
        return prediction_frame(games, probs)

    def _best_rounds(self, rows: pl.DataFrame, params: dict[str, Any]) -> int:
        split = split_last_season(rows)
        if split is None or split[0].height == 0 or split[1].height == 0:
            return self.default_rounds
        fit_rows, valid_rows = split
        train_set = self._dataset(self.design.transform(fit_rows), labels(fit_rows))
        valid_set = self._dataset(self.design.transform(valid_rows), labels(valid_rows))
        booster = lgb.train(
            params,
            train_set,
            self.max_rounds,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return max(int(booster.best_iteration), 1)

    def _dataset(self, x: np.ndarray, y: np.ndarray) -> lgb.Dataset:
        return lgb.Dataset(
            x, label=y, feature_name=self.design.names or "auto", free_raw_data=False
        )

    def _params(self) -> dict[str, Any]:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "lambda_l2": self.lambda_l2,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": 1.0,
            "monotone_constraints": [
                1 if name in MONOTONE_INCREASING else 0 for name in self.design.names
            ],
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "seed": self.seed,
            "verbose": -1,
        }
