import fcntl
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import nflreadpy
import polars as pl
from rich.console import Console

from football.config import get_settings
from football.storage import layer_dir

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SCHEDULE_MAX_AGE = timedelta(hours=12)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
EASTERN = "America/New_York"
DEFAULT_GAMETIME = "13:00"

console = Console()


class FetchError(RuntimeError):
    pass


def static_path(name: str) -> Path:
    return STATIC_DIR / name


def raw_dir(source: str) -> Path:
    path = layer_dir("raw") / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> datetime:
    return datetime.now(UTC)


def load_schedules(refresh: bool = False) -> pl.DataFrame:
    path = raw_dir("venues") / "schedules.parquet"
    fresh = path.exists() and (
        utc_now() - datetime.fromtimestamp(path.stat().st_mtime, UTC) < SCHEDULE_MAX_AGE
    )
    if fresh and not refresh:
        return pl.read_parquet(path)
    frame = nflreadpy.load_schedules()
    frame.write_parquet(path)
    return frame


def kickoff_expr() -> pl.Expr:
    local = pl.concat_str(
        pl.col("gameday"), pl.lit(" "), pl.col("gametime").fill_null(DEFAULT_GAMETIME)
    ).str.to_datetime("%Y-%m-%d %H:%M", time_unit="us")
    return (
        local.dt.replace_time_zone(EASTERN, ambiguous="latest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .alias("kickoff_utc")
    )


def parse_utc(name: str, fmt: str) -> pl.Expr:
    text = pl.col(name).str.strip_chars()
    return pl.coalesce(
        text.str.to_datetime(fmt, time_unit="us", time_zone="UTC", strict=False),
        text.str.to_datetime("%Y-%m-%d", time_unit="us", time_zone="UTC", strict=False),
    ).alias(name)


class Throttle:
    def __init__(self, name: str, min_interval_s: float) -> None:
        self.path = raw_dir("throttle") / f"{name}.lock"
        self.min_interval_s = min_interval_s

    def wait(self) -> None:
        with self.path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            text = handle.read().strip()
            last = float(text) if text else 0.0
            delay = last + self.min_interval_s - time.time()
            if delay > 0:
                time.sleep(delay)
            handle.seek(0)
            handle.truncate()
            handle.write(f"{time.time():.3f}")
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)


class Http:
    def __init__(
        self,
        throttle: Throttle,
        *,
        attempts: int = 5,
        backoff_s: float = 10.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        settings = get_settings()
        self.throttle = throttle
        self.attempts = attempts
        self.backoff_s = backoff_s
        self.client = httpx.Client(
            timeout=max(settings.http_timeout_s, 120.0),
            headers={"User-Agent": settings.user_agent, **(headers or {})},
            follow_redirects=True,
        )

    def get(
        self,
        url: str,
        params: httpx.QueryParams | Mapping[str, str | int | float] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        last_error = ""
        for attempt in range(self.attempts):
            self.throttle.wait()
            try:
                response = self.client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code not in RETRY_STATUSES:
                    return response
                last_error = f"HTTP {response.status_code}"
            if attempt + 1 < self.attempts:
                time.sleep(self.backoff_s * 2**attempt)
        raise FetchError(f"{url} failed after {self.attempts} attempts: {last_error}")

    def close(self) -> None:
        self.client.close()


def iem_http() -> Http:
    return Http(Throttle("iem", 2.5))
