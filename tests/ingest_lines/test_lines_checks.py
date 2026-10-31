from datetime import UTC, datetime

import polars as pl
import pytest

from football import storage
from football.ingest import lines_checks
from football.ingest.lines_common import REJECTS, finalize, reject_frame, write_rejects
from football.schemas import LINES


def close_lines(source: str, spreads: list[float], kind: str = "close") -> pl.DataFrame:
    stamp = datetime(2020, 1, 1, tzinfo=UTC)
    n = len(spreads)
    return finalize(
        pl.DataFrame(
            {
                "game_id": [f"2020_{i:03d}_A_B" for i in range(n)],
                "book": ["consensus"] * n,
                "kind": [kind] * n,
                "captured_at": [stamp] * n,
                "available_at": [stamp] * n,
                "home_spread": spreads,
                "total": [None] * n,
                "home_ml": [None] * n,
                "away_ml": [None] * n,
            },
            schema_overrides={"total": pl.Float64, "home_ml": pl.Float64, "away_ml": pl.Float64},
        ),
        source,
    )


def store(*frames: pl.DataFrame) -> None:
    storage.write(pl.concat(frames), "clean", LINES.name)


def test_closing_agreement_passes_and_fails_on_threshold() -> None:
    base = [float(i % 14) for i in range(200)]
    near = [s + (0.5 if i % 2 else 0.0) for i, s in enumerate(base)]
    store(close_lines("lines_a", base), close_lines("lines_b", near))
    result = lines_checks.closing_spread_agreement()
    assert result.passed
    assert "a~b: n=200 exact=50.0% within0.5=100.0%" in result.detail
    far = [s + (1.0 if i % 5 == 0 else 0.0) for i, s in enumerate(base)]
    store(close_lines("lines_a", base), close_lines("lines_b", far))
    result = lines_checks.closing_spread_agreement()
    assert not result.passed
    assert "failing ['a~b']" in result.detail


def test_small_overlaps_are_reported_but_not_enforced() -> None:
    store(close_lines("lines_a", [3.0] * 10), close_lines("lines_b", [-3.0] * 10))
    assert not lines_checks.closing_spread_agreement().passed
    assert "n=10" in lines_checks.closing_spread_agreement().detail


def test_opening_agreement_always_passes() -> None:
    store(close_lines("lines_a", [3.0] * 150, "open"), close_lines("lines_b", [-3.0] * 150, "open"))
    result = lines_checks.opening_spread_agreement()
    assert result.passed
    assert "within0.5=0.0%" in result.detail


@pytest.mark.parametrize(("unmatched", "passed"), [(1, True), (5, False)])
def test_lines_unmatched_threshold(unmatched: int, passed: bool) -> None:
    store(close_lines("lines_a", [3.0] * 200))
    refs = [f"r{i}" for i in range(unmatched)]
    write_rejects(
        "lines_a", [2020], [reject_frame("lines_a", "unmatched", [2020] * unmatched, refs)]
    )
    write_rejects(
        "elo538",
        [2020],
        [reject_frame("elo538", "unmatched", [2020] * 50, [f"e{i}" for i in range(50)])],
    )
    result = lines_checks.lines_unmatched()
    assert result.passed is passed
    assert f"lines_a: rows=200 games=200 unmatched={unmatched}" in result.detail
    assert "elo538" not in result.detail


def test_write_rejects_replaces_only_processed_seasons() -> None:
    write_rejects(
        "lines_a", [2019, 2020], [reject_frame("lines_a", "parse", [2019, 2020], ["x", "y"])]
    )
    write_rejects("lines_a", [2020], [reject_frame("lines_a", "parse", [2020], ["z"])])
    table = storage.read("clean", REJECTS.name)
    assert sorted(table["ref"].to_list()) == ["x", "z"]
