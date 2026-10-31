from typing import Any

from football.models.base import Model

MODEL_NAMES: tuple[str, ...] = ()


def build(name: str, **params: Any) -> Model:
    raise NotImplementedError(name)
