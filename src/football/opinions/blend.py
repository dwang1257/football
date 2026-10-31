import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from football.config import get_settings

CONF_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)
MAX_CONF = 5
TEAM_POINT_LOGIT = 0.12
PROB_EPS = 1e-6


def default_conf_prob(conf: int) -> float:
    return 0.50 + 0.08 * conf


def default_conf_table() -> dict[int, float]:
    return {conf: default_conf_prob(conf) for conf in CONF_LEVELS}


def logit(p: float) -> float:
    q = min(max(p, PROB_EPS), 1.0 - PROB_EPS)
    return math.log(q / (1.0 - q))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class BlendParams:
    weight: float
    team_weight: float
    cap: float
    cap_max_conf: float
    conf_table: Mapping[int, float]

    @classmethod
    def from_settings(cls) -> "BlendParams":
        settings = get_settings()
        return cls(
            weight=settings.opinion_weight,
            team_weight=settings.opinion_team_weight,
            cap=settings.opinion_cap,
            cap_max_conf=settings.opinion_cap_max_conf,
            conf_table=default_conf_table(),
        )

    def with_weights(
        self, weight: float | None = None, team_weight: float | None = None
    ) -> "BlendParams":
        return replace(
            self,
            weight=self.weight if weight is None else weight,
            team_weight=self.team_weight if team_weight is None else team_weight,
        )

    def pick_prob(self, conf: int | None, prob: float | None) -> float:
        if prob is not None:
            return prob
        if conf is None:
            raise ValueError("an opinion needs conf or prob")
        return self.conf_table[conf]


@dataclass(frozen=True)
class BlendResult:
    model_prob: float
    user_prob: float | None
    nudge_points: float
    delta_logit: float
    capped: bool
    blended_prob: float

    @property
    def delta_pp(self) -> float:
        return 100.0 * (self.blended_prob - self.model_prob)

    @property
    def flipped(self) -> bool:
        return (self.model_prob >= 0.5) != (self.blended_prob >= 0.5)


def blend(
    model_prob: float,
    user_prob: float | None,
    nudge_points: float,
    conf: int | None,
    params: BlendParams,
) -> BlendResult:
    base = logit(model_prob)
    game_delta = 0.0 if user_prob is None else params.weight * (logit(user_prob) - base)
    team_delta = params.team_weight * TEAM_POINT_LOGIT * nudge_points
    raw = game_delta + team_delta
    cap = params.cap_max_conf if conf == MAX_CONF else params.cap
    delta = min(max(raw, -cap), cap)
    return BlendResult(
        model_prob=model_prob,
        user_prob=user_prob,
        nudge_points=nudge_points,
        delta_logit=delta,
        capped=abs(raw) > cap,
        blended_prob=sigmoid(base + delta),
    )
