from collections.abc import Callable

import numpy as np
import polars as pl

from football.models.base import Model, Track
from football.models.common import (
    Calibration,
    Calibrator,
    check_choice,
    check_track,
    classifier_rows,
    labels,
    margin_rows,
    prediction_frame,
    split_last_season,
)

DEFAULT_COMPONENTS = "elo,matchup,logreg,margin"
CALIBRATIONS: tuple[Calibration, ...] = ("none", "platt", "isotonic")

Builder = Callable[[str, Track], Model]


class EnsembleModel:
    def __init__(
        self,
        builder: Builder,
        track: str = "independent",
        components: str = DEFAULT_COMPONENTS,
        calibration: str = "none",
    ) -> None:
        self.name = "ensemble"
        self.track: Track = check_track(track)
        self.builder = builder
        self.components = [c.strip() for c in components.split(",") if c.strip()]
        if not self.components or "ensemble" in self.components:
            raise ValueError("ensemble needs one or more non-ensemble components")
        self.calibration: Calibration = check_choice(calibration, CALIBRATIONS, "calibration")
        self.models: list[Model] = self._build()
        self.calibrator = Calibrator("none")
        self.dropped: dict[str, str] = {}

    def fit(self, train: pl.DataFrame) -> None:
        self.calibrator = Calibrator("none")
        self.dropped = {}
        split = split_last_season(margin_rows(train))
        if self.calibration != "none":
            if split is None:
                self.dropped["calibration"] = "needs at least two train seasons"
            else:
                self._fit_calibrator(*split)
        self.models = self._build()
        for model in self.models:
            model.fit(train)

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        probs, margin = self._average(self.models, games)
        return prediction_frame(games, self.calibrator.apply(probs), margin)

    def _fit_calibrator(self, fit_rows: pl.DataFrame, hold: pl.DataFrame) -> None:
        hold = classifier_rows(hold)
        if hold.height == 0:
            self.dropped["calibration"] = "no decided games in the last train season"
            return
        models = self._build()
        for model in models:
            model.fit(fit_rows)
        probs, _ = self._average(models, hold)
        self.calibrator = Calibrator(self.calibration).fit(probs, labels(hold))

    def _build(self) -> list[Model]:
        return [self.builder(name, self.track) for name in self.components]

    @staticmethod
    def _average(models: list[Model], games: pl.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        frames = [m.predict(games) for m in models]
        probs = np.mean([f["home_win_prob"].to_numpy() for f in frames], axis=0)
        margins = [
            f["home_margin"].to_numpy() for f in frames if f["home_margin"].null_count() == 0
        ]
        margin = np.mean(margins, axis=0) if margins else None
        return probs, margin
