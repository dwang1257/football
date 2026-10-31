import inspect
from collections.abc import Callable
from typing import Any

from football.models.base import Model, Track
from football.models.baselines import EloModel, HomeModel, MarketModel, RecordModel
from football.models.ensemble import EnsembleModel
from football.models.gbm import LgbmModel
from football.models.linear import LogRegModel, MarginModel
from football.models.matchup import MatchupModel


def _component(name: str, track: Track) -> Model:
    return build(name, track=track)


def _ensemble(**params: Any) -> Model:
    return EnsembleModel(_component, **params)


_BUILDERS: dict[str, Callable[..., Model]] = {
    "home": HomeModel,
    "record": RecordModel,
    "elo": EloModel,
    "market": MarketModel,
    "matchup": MatchupModel,
    "logreg": LogRegModel,
    "margin": MarginModel,
    "lgbm": LgbmModel,
    "ensemble": _ensemble,
}

MODEL_NAMES: tuple[str, ...] = tuple(_BUILDERS)


def parse_param(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


def parameters(name: str) -> list[str]:
    target = EnsembleModel if name == "ensemble" else _BUILDERS[name]
    return [p for p in inspect.signature(target).parameters if p not in ("self", "builder")]


def build(name: str, **params: Any) -> Model:
    if name not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_NAMES)}")
    allowed = parameters(name)
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(f"model {name!r} has no parameters {unknown}; allowed: {allowed}")
    return _BUILDERS[name](**{key: parse_param(value) for key, value in params.items()})
