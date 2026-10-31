import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from football.config import get_settings

Layer = Literal["raw", "clean", "features", "predictions", "runs", "archive"]


def layer_dir(layer: Layer) -> Path:
    path = get_settings().data_dir / layer
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_path(layer: Layer, name: str) -> Path:
    return layer_dir(layer) / f"{name}.parquet"


def exists(layer: Layer, name: str) -> bool:
    return table_path(layer, name).exists()


def read(layer: Layer, name: str) -> pl.DataFrame:
    return pl.read_parquet(table_path(layer, name))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    os.replace(tmp, path)


def write(df: pl.DataFrame, layer: Layer, name: str) -> Path:
    path = table_path(layer, name)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    df.write_parquet(tmp)
    os.replace(tmp, path)
    return path


def upsert(df: pl.DataFrame, layer: Layer, name: str, keys: Sequence[str]) -> pl.DataFrame:
    combined = df
    if exists(layer, name):
        old = read(layer, name).join(df.select(keys), on=list(keys), how="anti", nulls_equal=True)
        combined = pl.concat([old, df], how="diagonal_relaxed")
    combined = combined.sort(list(keys))
    write(combined, layer, name)
    return combined


def write_raw(source: str, name: str, payload: bytes) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = layer_dir("raw") / source / f"{stamp}_{name}"
    _atomic_write_bytes(path, payload)
    return path
