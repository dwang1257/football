import gzip
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from football.collect.http import Fetched
from football.storage import layer_dir

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_DAY_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILE_NAME = re.compile(r"^\d{8}T\d{6}Z_[A-Za-z0-9_.-]+\.json\.gz$")


@dataclass(frozen=True)
class Snapshot:
    collector: str
    name: str
    url: str
    fetched_at: datetime
    status: int
    run_id: str
    body: Any
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def archive_root() -> Path:
    return layer_dir("archive")


def collector_dir(collector: str) -> Path:
    return archive_root() / collector


def _safe(name: str) -> str:
    return _SAFE_NAME.sub("-", name).strip("-") or "item"


def snapshot_path(collector: str, name: str, fetched_at: datetime) -> Path:
    moment = fetched_at.astimezone(UTC)
    day = moment.strftime("%Y-%m-%d")
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    return collector_dir(_safe(collector)) / day / f"{stamp}_{_safe(name)}.json.gz"


def _envelope(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "collector": snapshot.collector,
        "name": snapshot.name,
        "url": snapshot.url,
        "fetched_at": snapshot.fetched_at.astimezone(UTC).isoformat(),
        "status": snapshot.status,
        "run_id": snapshot.run_id,
        "meta": snapshot.meta,
        "body": snapshot.body,
    }


def write_snapshot(snapshot: Snapshot) -> Path:
    path = snapshot_path(snapshot.collector, snapshot.name, snapshot.fetched_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_envelope(snapshot), ensure_ascii=False, separators=(",", ":"))
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle, gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as gz:
            gz.write(payload.encode("utf-8"))
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
    return path


def snapshot_from_fetch(
    collector: str,
    name: str,
    fetched: Fetched,
    run_id: str,
    body: Any,
    meta: dict[str, Any] | None = None,
) -> Snapshot:
    return Snapshot(
        collector=collector,
        name=name,
        url=fetched.url,
        fetched_at=fetched.fetched_at,
        status=fetched.status,
        run_id=run_id,
        body=body,
        meta=meta or {},
    )


def read_snapshot(path: Path) -> Snapshot:
    with gzip.open(path, "rb") as handle:
        envelope = json.loads(handle.read().decode("utf-8"))
    return Snapshot(
        collector=str(envelope["collector"]),
        name=str(envelope["name"]),
        url=str(envelope["url"]),
        fetched_at=datetime.fromisoformat(envelope["fetched_at"]).astimezone(UTC),
        status=int(envelope["status"]),
        run_id=str(envelope["run_id"]),
        body=envelope.get("body"),
        meta=dict(envelope.get("meta") or {}),
        path=path,
    )


def snapshot_paths(collector: str) -> list[Path]:
    root = collector_dir(collector)
    if not root.is_dir():
        return []
    return sorted(
        path
        for day in root.iterdir()
        if day.is_dir() and _DAY_DIR.match(day.name)
        for path in day.iterdir()
        if _FILE_NAME.match(path.name)
    )


def iter_snapshots(collector: str, prefix: str = "") -> Iterator[Snapshot]:
    for path in snapshot_paths(collector):
        if prefix and not path.name.split("_", 1)[1].startswith(prefix):
            continue
        yield read_snapshot(path)


def collectors_present() -> list[str]:
    root = archive_root()
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def snapshot_days(collector: str) -> list[date]:
    root = collector_dir(collector)
    if not root.is_dir():
        return []
    return sorted(
        date.fromisoformat(day.name)
        for day in root.iterdir()
        if day.is_dir() and _DAY_DIR.match(day.name) and any(day.glob("*.json.gz"))
    )


@dataclass(frozen=True)
class ImportResult:
    copied: int
    skipped: int
    ignored: int


def import_archive(source: Path) -> ImportResult:
    copied = skipped = ignored = 0
    root = archive_root()
    for collector in sorted(source.iterdir()):
        if collector.name.startswith(".") or collector.is_symlink() or not collector.is_dir():
            continue
        for day in sorted(collector.iterdir()):
            if day.is_symlink() or not day.is_dir() or not _DAY_DIR.match(day.name):
                ignored += 1
                continue
            for item in sorted(day.iterdir()):
                if item.is_symlink() or not item.is_file() or not _FILE_NAME.match(item.name):
                    ignored += 1
                    continue
                target = root / _safe(collector.name) / day.name / item.name
                if target.exists():
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
                os.close(fd)
                try:
                    shutil.copyfile(item, tmp)
                    os.link(tmp, target)
                finally:
                    os.unlink(tmp)
                copied += 1
    return ImportResult(copied=copied, skipped=skipped, ignored=ignored)
