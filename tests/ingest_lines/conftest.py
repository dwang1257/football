from collections.abc import Callable

import pytest
from lines_fixtures import FETCHED_AT, schedule_frame

from football.ingest import lines_common
from football.ingest.lines_common import Schedule


@pytest.fixture
def schedule() -> Schedule:
    return Schedule(schedule_frame(), FETCHED_AT)


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, bytes]], list[str]]:
    requested: list[str] = []

    def install(routes: dict[str, bytes]) -> list[str]:
        def fake_get(url: str, **_: object) -> bytes:
            requested.append(url)
            for prefix, payload in routes.items():
                if url.startswith(prefix):
                    return payload
            raise AssertionError(f"unexpected url {url}")

        monkeypatch.setattr(lines_common, "http_get", fake_get)
        return requested

    return install


@pytest.fixture
def use_schedule(monkeypatch: pytest.MonkeyPatch, schedule: Schedule) -> Schedule:
    from football.ingest import benchmarks, lines, lines_espn, lines_sbro, lines_wintotals

    for module in (lines, lines_sbro, lines_espn, lines_wintotals, benchmarks):
        monkeypatch.setattr(module, "load_schedule", lambda *_, **__: schedule)
    return schedule
