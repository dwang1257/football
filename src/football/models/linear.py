import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge

from football.models.base import Track
from football.models.common import (
    Design,
    check_choice,
    check_track,
    classifier_rows,
    labels,
    margin_rows,
    margins,
    normal_win_prob,
    prediction_frame,
    residual_sd,
    season_folds,
)
from football.models.featuresets import (
    DEFAULT_GROUPS,
    MATCHUP_MU,
    feature_set,
    parse_groups,
    with_matchup_mu,
)
from football.models.matchup import COMBINES, DEFAULT_ALPHA, Combine, PointsModel


def _points(rows: pl.DataFrame, combine: Combine) -> PointsModel:
    return PointsModel(combine, DEFAULT_ALPHA).fit(margin_rows(rows))


def _with_mu(games: pl.DataFrame, points: PointsModel) -> pl.DataFrame:
    mu = points.mu(points.inputs(games))
    return games.with_columns(pl.Series(MATCHUP_MU, mu, dtype=pl.Float64))


def cross_fit_matchup_mu(train: pl.DataFrame, combine: Combine) -> pl.DataFrame | None:
    parts = [_with_mu(hold, _points(fit_rows, combine)) for fit_rows, hold in season_folds(train)]
    if not parts:
        return None
    return pl.concat(parts)


class LogRegModel:
    def __init__(
        self,
        track: str = "independent",
        c: float = 0.01,
        features: str = DEFAULT_GROUPS,
        use_matchup: bool = False,
        combine: str = "league_relative",
    ) -> None:
        self.name = "logreg"
        self.track: Track = check_track(track)
        self.c = float(c)
        self.groups = parse_groups(features, self.track)
        self.use_matchup = use_matchup
        self.combine: Combine = check_choice(combine, COMBINES, "combine")
        self.design = Design([], self.track)
        self.model = LogisticRegression(C=self.c, max_iter=2000)
        self.points: PointsModel | None = None
        self.dropped: dict[str, str] = {}

    def fit(self, train: pl.DataFrame) -> None:
        candidates = feature_set(self.groups, train, self.combine)
        frame = train
        self.points = None
        self.dropped = {}
        if self.use_matchup:
            crossed = cross_fit_matchup_mu(train, self.combine)
            if crossed is None:
                self.dropped[MATCHUP_MU] = "cross-fitting needs at least two train seasons"
            else:
                frame, self.points = crossed, _points(train, self.combine)
                candidates = with_matchup_mu(candidates)
        rows = classifier_rows(frame)
        self.design = Design(candidates, self.track)
        x = self.design.fit(rows)
        self.dropped = {**self.dropped, **self.design.dropped}
        self.model = LogisticRegression(C=self.c, max_iter=2000)
        self.model.fit(x, labels(rows).astype(int))

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        frame = games if self.points is None else _with_mu(games, self.points)
        probs = self.model.predict_proba(self.design.transform(frame))[:, 1]
        return prediction_frame(games, probs)

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.design.names, self.model.coef_[0].tolist(), strict=True))


class MarginModel:
    def __init__(
        self,
        track: str = "independent",
        alpha: float = 10.0,
        features: str = DEFAULT_GROUPS,
        combine: str = "league_relative",
    ) -> None:
        self.name = "margin"
        self.track: Track = check_track(track)
        self.alpha = float(alpha)
        self.groups = parse_groups(features, self.track)
        self.combine: Combine = check_choice(combine, COMBINES, "combine")
        self.design = Design([], self.track)
        self.model = Ridge(alpha=self.alpha)
        self.sigma = 13.5
        self.dropped: dict[str, str] = {}

    def fit(self, train: pl.DataFrame) -> None:
        rows = margin_rows(train)
        self.design, self.model = self._fit(rows)
        self.dropped = dict(self.design.dropped)
        residuals = [
            margins(hold) - model.predict(design.transform(hold))
            for fit_rows, hold in season_folds(rows)
            for design, model in [self._fit(fit_rows)]
        ]
        if residuals:
            self.sigma = residual_sd(np.concatenate(residuals))
        else:
            fitted = self.model.predict(self.design.transform(rows))
            self.sigma = residual_sd(margins(rows) - fitted, len(self.design.features) + 1)

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        mu = self.model.predict(self.design.transform(games))
        return prediction_frame(games, normal_win_prob(mu, self.sigma), mu)

    def _fit(self, rows: pl.DataFrame) -> tuple[Design, Ridge]:
        design = Design(feature_set(self.groups, rows, self.combine), self.track)
        x = design.fit(rows)
        model = Ridge(alpha=self.alpha)
        model.fit(x, margins(rows))
        return design, model
