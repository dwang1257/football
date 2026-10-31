from pathlib import Path

import httpx
import pytest
from collect_fixtures import snapshot, store
from typer.testing import CliRunner

from football.cli import app
from football.collect import commands, runner
from football.collect.base import CollectorResult, Context
from football.collect.http import Fetcher, FetchError
from football.config import get_settings


def _ok(ctx: Context, result: CollectorResult) -> None:
    result.ok.append("item")


def _boom(ctx: Context, result: CollectorResult) -> None:
    raise RuntimeError("site down")


def _fetcher() -> Fetcher:
    return Fetcher(transport=httpx.MockTransport(lambda r: httpx.Response(200)), retries=0)


def test_one_failure_does_not_stop_others(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "COLLECTORS", {"a": _boom, "b": _ok})
    results = runner.run_collectors(["a", "b"], _fetcher())
    assert [r.status for r in results] == ["failed", "ok"]
    assert results[0].failed == {"collector": "RuntimeError: site down"}
    assert not runner.all_failed(results)
    assert runner.all_failed(results[:1])


def test_cli_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "COLLECTORS", {"a": _boom, "b": _ok})
    monkeypatch.setattr(commands, "COLLECTORS", runner.COLLECTORS)
    cli = CliRunner()
    assert cli.invoke(app, ["collect"]).exit_code == 0
    failed = cli.invoke(app, ["collect", "--only", "a"])
    assert failed.exit_code == 1
    assert "failed" in failed.output
    assert cli.invoke(app, ["collect", "--only", "zzz"]).exit_code != 0


def test_archive_import_cli(
    tmp_path: Path, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store(snapshot("espn_fpi", "powerindex", {}, "2026-10-31T10:00:00+00:00"))
    source = isolated_data_dir / "archive"
    target = tmp_path / "other"
    monkeypatch.setenv("FOOTBALL_DATA_DIR", str(target))
    get_settings.cache_clear()
    result = CliRunner().invoke(app, ["archive-import", str(source)])
    assert result.exit_code == 0, result.output
    assert "copied 1" in result.output
    assert len(list((target / "archive/espn_fpi").rglob("*.json.gz"))) == 1


def test_fetcher_retries_transient_errors() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503 if len(calls) < 3 else 200, text="ok")

    fetcher = Fetcher(
        transport=httpx.MockTransport(handler), retries=3, backoff_s=0, min_interval_s=0
    )
    fetched = fetcher.get("https://example.test/x")
    assert fetched.ok
    assert len(calls) == 3


def test_fetcher_gives_up_and_returns_last_status() -> None:
    fetcher = Fetcher(
        transport=httpx.MockTransport(lambda r: httpx.Response(429)), retries=1, backoff_s=0
    )
    assert fetcher.get("https://example.test/x").status == 429

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("reset", request=request)

    fetcher = Fetcher(transport=httpx.MockTransport(broken), retries=1, backoff_s=0)
    with pytest.raises(FetchError, match="ConnectError"):
        fetcher.get("https://example.test/x")
