from collections.abc import Sequence
from datetime import datetime

import polars as pl


def build_features(
    as_of: datetime | None = None, seasons: Sequence[int] | None = None
) -> pl.DataFrame:
    raise NotImplementedError
