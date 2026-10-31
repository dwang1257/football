import os
import tempfile
import time
from pathlib import Path

import httpx
import polars as pl

from football.config import get_settings
from football.storage import layer_dir

RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/"
RAW_SOURCE = "nflverse"
ATTEMPTS = 4
THROTTLE_S = 0.5


class NotPublished(LookupError):
    pass


def raw_path(asset: str) -> Path:
    return layer_dir("raw") / RAW_SOURCE / asset


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    os.replace(tmp, path)


def _download(url: str) -> bytes:
    settings = get_settings()
    headers = {"User-Agent": settings.user_agent}
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        time.sleep(THROTTLE_S if attempt == 0 else 2.0 * 2**attempt)
        try:
            response = httpx.get(
                url,
                headers=headers,
                follow_redirects=True,
                timeout=max(settings.http_timeout_s, 120.0),
            )
        except httpx.TransportError as error:
            last_error = error
            continue
        if response.status_code == 404:
            raise NotPublished(url)
        if response.status_code >= 500 or response.status_code == 429:
            last_error = httpx.HTTPStatusError(
                f"{response.status_code} for {url}", request=response.request, response=response
            )
            continue
        response.raise_for_status()
        return response.content
    raise ConnectionError(f"failed to download {url}: {last_error}")


def fetch(asset: str, refresh: bool = False, url: str | None = None) -> Path:
    path = raw_path(asset)
    if path.exists() and not refresh:
        return path
    _atomic_write(path, _download(url or RELEASE_URL + asset))
    return path


def fetch_frame(asset: str, refresh: bool = False, url: str | None = None) -> pl.DataFrame:
    path = fetch(asset, refresh, url)
    if path.suffix == ".csv":
        return pl.read_csv(
            path, null_values=["NA", "NULL", ""], infer_schema_length=None, try_parse_dates=False
        )
    return pl.read_parquet(path)


def fetch_optional(asset: str, refresh: bool = False) -> pl.DataFrame | None:
    try:
        return fetch_frame(asset, refresh)
    except NotPublished:
        return None
