from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestOptions:
    seasons: list[int]
    refresh: bool = False


IngestFn = Callable[[IngestOptions], None]
