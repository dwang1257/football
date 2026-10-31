from typing import Literal, Protocol

import polars as pl

Track = Literal["independent", "market"]


class Model(Protocol):
    name: str
    track: Track

    def fit(self, train: pl.DataFrame) -> None: ...

    def predict(self, games: pl.DataFrame) -> pl.DataFrame: ...
