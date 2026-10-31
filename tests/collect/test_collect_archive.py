import gzip
import json
from pathlib import Path

import pytest
from collect_fixtures import snapshot, store, ts

from football.collect.archive import (
    import_archive,
    iter_snapshots,
    read_snapshot,
    snapshot_days,
    snapshot_path,
    write_snapshot,
)


def test_layout_and_round_trip(isolated_data_dir: Path) -> None:
    item = snapshot("espn_fpi", "power index", {"a": 1}, "2026-10-31T18:36:28.5+00:00", {"w": 8})
    path = write_snapshot(item)
    assert (
        path
        == isolated_data_dir / "archive/espn_fpi/2026-10-31/20261031T183628Z_power-index.json.gz"
    )
    with gzip.open(path) as handle:
        envelope = json.loads(handle.read())
    assert set(envelope) == {
        "collector",
        "name",
        "url",
        "fetched_at",
        "status",
        "run_id",
        "meta",
        "body",
    }
    loaded = read_snapshot(path)
    assert loaded.body == {"a": 1}
    assert loaded.meta == {"w": 8}
    assert loaded.fetched_at == ts("2026-10-31T18:36:28.5+00:00")
    assert loaded.ok


def test_never_overwrites() -> None:
    item = snapshot("espn_fpi", "powerindex", {"a": 1}, "2026-10-31T18:36:28+00:00")
    path = write_snapshot(item)
    with pytest.raises(FileExistsError):
        write_snapshot(snapshot("espn_fpi", "powerindex", {"a": 2}, "2026-10-31T18:36:28+00:00"))
    assert read_snapshot(path).body == {"a": 1}
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_iter_snapshots_filters_prefix_and_sorts() -> None:
    store(
        snapshot("espn_odds", "odds_2", {}, "2026-10-31T10:00:00+00:00"),
        snapshot("espn_odds", "scoreboard", {}, "2026-10-30T10:00:00+00:00"),
        snapshot("espn_odds", "odds_1", {}, "2026-10-29T10:00:00+00:00"),
    )
    assert [s.name for s in iter_snapshots("espn_odds", prefix="odds_")] == ["odds_1", "odds_2"]
    assert [d.isoformat() for d in snapshot_days("espn_odds")] == [
        "2026-10-29",
        "2026-10-30",
        "2026-10-31",
    ]


def test_import_archive_copies_new_files_only(tmp_path: Path) -> None:
    source = tmp_path / "checkout"
    kept = write_snapshot(
        snapshot("forecasts", "BUF00", {"old": True}, "2026-10-31T10:00:00+00:00")
    )
    duplicate = source / "forecasts/2026-10-31/20261031T100000Z_BUF00.json.gz"
    fresh = source / "forecasts/2026-11-01/20261101T100000Z_BUF00.json.gz"
    for path in (duplicate, fresh):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps({"x": 1}).encode()))
    (source / ".git").mkdir()
    (source / ".git/config").write_text("x")
    (source / "forecasts/2026-11-01/notes.txt").write_text("ignored")
    (source / "forecasts/2026-11-01/link.json.gz").symlink_to(fresh)
    result = import_archive(source)
    assert (result.copied, result.skipped, result.ignored) == (1, 1, 2)
    assert read_snapshot(kept).body == {"old": True}
    assert snapshot_path("forecasts", "BUF00", ts("2026-11-01T10:00:00+00:00")).exists()
