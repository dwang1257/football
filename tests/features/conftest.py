import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from football.features.inputs import Tables

SYNTHETIC = Path(__file__).resolve().parents[1] / "leakage" / "synthetic.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("features_synthetic", SYNTHETIC)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load()


@pytest.fixture
def make_tables() -> Callable[..., Tables]:
    return _MODULE.make_tables


@pytest.fixture
def required() -> Callable[[pl.DataFrame | None], pl.DataFrame]:
    return _MODULE.required
