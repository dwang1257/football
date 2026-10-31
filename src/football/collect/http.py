import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from football.config import get_settings

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fetched:
    url: str
    status: int
    text: str
    fetched_at: datetime

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        return json.loads(self.text)


class Fetcher:
    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout_s: float | None = None,
        min_interval_s: float = 0.25,
        host_intervals_s: Mapping[str, float] | None = None,
        retries: int = 3,
        backoff_s: float = 2.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self._client = httpx.Client(
            timeout=timeout_s or settings.http_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": user_agent or settings.user_agent},
            transport=transport,
        )
        self._min_interval_s = min_interval_s
        self._host_intervals_s = dict(host_intervals_s or {})
        self._retries = retries
        self._backoff_s = backoff_s
        self._lock = threading.Lock()
        self._next_slot: dict[str, float] = {}

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _wait_turn(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, now))
            interval = self._host_intervals_s.get(host, self._min_interval_s)
            self._next_slot[host] = slot + interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Fetched:
        host = urlsplit(url).netloc
        error = "no attempt made"
        for attempt in range(self._retries + 1):
            if attempt:
                time.sleep(self._backoff_s * 2 ** (attempt - 1))
            self._wait_turn(host)
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
                continue
            fetched = Fetched(
                url=str(response.url),
                status=response.status_code,
                text=response.text,
                fetched_at=datetime.now(UTC),
            )
            if response.status_code in RETRY_STATUSES and attempt < self._retries:
                error = f"HTTP {response.status_code}"
                continue
            return fetched
        raise FetchError(f"{url}: {error}")
