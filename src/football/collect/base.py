from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from football.collect.archive import Snapshot, snapshot_from_fetch, write_snapshot
from football.collect.http import Fetched, Fetcher


@dataclass
class CollectorResult:
    collector: str
    files: list[Path] = field(default_factory=list)
    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        if not self.ok and self.failed:
            return "failed"
        if self.failed:
            return "partial"
        return "ok"


@dataclass
class Context:
    fetcher: Fetcher
    run_id: str
    started_at: datetime
    cache: dict[str, Any] = field(default_factory=dict)

    def save(
        self,
        result: CollectorResult,
        name: str,
        fetched: Fetched,
        body: Any,
        meta: dict[str, Any] | None = None,
    ) -> Snapshot:
        snapshot = snapshot_from_fetch(result.collector, name, fetched, self.run_id, body, meta)
        result.files.append(write_snapshot(snapshot))
        return snapshot

    def save_json(
        self,
        result: CollectorResult,
        name: str,
        fetched: Fetched,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        try:
            body = fetched.json()
        except ValueError:
            self.save(result, name, fetched, fetched.text[:20000], meta)
            raise
        self.save(result, name, fetched, body, meta)
        return body


Collector = Callable[[Context, CollectorResult], None]
