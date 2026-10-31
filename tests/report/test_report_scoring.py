import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from typer.testing import CliRunner

from football import storage
from football.cli import app
from football.config import get_settings
from football.opinions.learning import load_weights
from football.predict.store import write_blended_rows, write_model_rows
from football.report.scoring import calibration, flip_ledger, scorecard, scored_games
from football.schemas import UTC_TS

T0 = datetime(2026, 10, 20, tzinfo=UTC)
GAMES = [
    ("A", 8, "DAL", "PHI", 30, 20),
    ("B", 8, "MIA", "NE", 10, 17),
    ("C", 8, "KC", "BUF", 27, 24),
    ("D", 8, "NYG", "ATL", 14, 21),
    ("E", 7, "LA", "SF", 21, 3),
    ("F", 8, "CHI", "GB", None, None),
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    monkeypatch.setenv("FOOTBALL_REPO_DIR", str(path))
    get_settings.cache_clear()
    return path


def model_row(run: str, game: str, prob: float, final: bool, minutes: int) -> dict[str, Any]:
    return {
        "run_id": run,
        "game_id": game,
        "model": "elo",
        "track": "independent",
        "home_win_prob": prob,
        "home_margin": None,
        "cutoff": T0,
        "created_at": T0 + timedelta(minutes=minutes),
        "is_final": final,
    }


def blended_row(
    run: str,
    game: str,
    model: float,
    blended: float,
    final: bool,
    pick: str | None = None,
    conf: int | None = None,
    pick_prob: float | None = None,
    user_prob: float | None = None,
    nudge: float = 0.0,
) -> dict[str, Any]:
    return {
        "run_id": run,
        "game_id": game,
        "model_prob": model,
        "user_prob": user_prob,
        "team_nudge_points": nudge,
        "opinion_weight": 0.12,
        "delta_logit": 0.0,
        "capped": False,
        "blended_prob": blended,
        "flipped": (model >= 0.5) != (blended >= 0.5),
        "created_at": T0,
        "is_final": final,
        "user_pick": pick,
        "user_conf": conf,
        "user_pick_prob": pick_prob,
        "user_why": "hunch" if pick else None,
        "entered_at": T0 if pick or nudge else None,
        "opinion_late": False,
    }


@pytest.fixture
def stored(repo: Path) -> None:
    names = ("game_id", "week", "away_team", "home_team", "home_score", "away_score")
    games = pl.DataFrame(
        [dict(zip(names, row, strict=True)) for row in GAMES],
        schema_overrides={"week": pl.Int32, "away_score": pl.Int32, "home_score": pl.Int32},
    ).with_columns(
        pl.lit(2026, dtype=pl.Int32).alias("season"),
        pl.lit(T0 + timedelta(days=1)).cast(UTC_TS).alias("kickoff_utc"),
    )
    storage.write(games, "clean", "games")
    model = [
        model_row("r1", "A", 0.8, True, 0),
        model_row("r1", "B", 0.6, True, 0),
        model_row("r1", "D", 0.45, True, 0),
        model_row("r1", "E", 0.55, True, 0),
        model_row("r1", "F", 0.5, True, 0),
        model_row("r0", "C", 0.2, False, 0),
        model_row("r2", "C", 0.7, False, 5),
        model_row("r2", "A", 0.1, False, 5),
    ]
    blended = [
        blended_row("r1", "A", 0.8, 0.8, True),
        blended_row("r1", "B", 0.6, 0.45, True, "MIA", 3, 0.74, 0.26),
        blended_row("r1", "D", 0.45, 0.52, True, nudge=1.0),
        blended_row("r1", "E", 0.55, 0.58, True, "SF", 2, 0.66, 0.66),
        blended_row("r2", "C", 0.7, 0.65, False, "KC", None, 0.6, 0.4),
        blended_row("r2", "A", 0.1, 0.1, False),
    ]
    write_model_rows(pl.DataFrame(model, schema_overrides={"home_margin": pl.Float64}))
    write_blended_rows(pl.DataFrame(blended, schema_overrides={"user_conf": pl.Int32}))


def ll(*probs: float) -> float:
    return -sum(math.log(p) for p in probs) / len(probs)


def test_week_scorecard_matches_hand_computation(stored: None) -> None:
    games = scored_games(2026)
    assert sorted(games["game_id"].to_list()) == ["A", "B", "C", "D", "E"]
    picked = dict(zip(games["game_id"].to_list(), games["model_prob"].to_list(), strict=True))
    assert picked["A"] == 0.8
    assert picked["C"] == 0.7

    card = scorecard(games.filter(pl.col("week") == 8), "Week 8")
    assert card.games == 4
    assert card.model.accuracy == pytest.approx(0.75)
    assert card.model.log_loss == pytest.approx(ll(0.8, 0.4, 0.7, 0.55))
    assert card.model.brier == pytest.approx((0.04 + 0.36 + 0.09 + 0.2025) / 4)
    assert card.blended.accuracy == pytest.approx(0.75)
    assert card.blended.log_loss == pytest.approx(ll(0.8, 0.55, 0.65, 0.48))
    assert card.blended.brier == pytest.approx((0.04 + 0.2025 + 0.1225 + 0.2704) / 4)

    assert card.opinion_games == 3
    assert card.model_on_opinions.log_loss == pytest.approx(ll(0.4, 0.7, 0.55))
    assert card.blended_on_opinions.log_loss == pytest.approx(ll(0.55, 0.65, 0.48))
    assert card.owner.n == 2
    assert card.owner.accuracy == pytest.approx(0.5)
    assert card.owner.log_loss == pytest.approx(ll(0.74, 0.4))
    assert card.owner.brier == pytest.approx((0.26**2 + 0.6**2) / 2)

    diffs = [
        math.log(0.4) - math.log(0.55),
        math.log(0.7) - math.log(0.65),
        math.log(0.55) - math.log(0.48),
    ]
    mean = sum(diffs) / 3
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / 2)
    assert card.paired.n == 3
    assert card.paired.mean == pytest.approx(mean)
    assert card.paired.se == pytest.approx(sd / math.sqrt(3))
    assert (card.flips_won, card.flips_lost) == (1, 1)


def test_flip_ledger_and_calibration(stored: None) -> None:
    games = scored_games(2026)
    ledger = flip_ledger(games).sort("game_id")
    assert ledger.select("game_id", "outcome").rows() == [("B", "won"), ("D", "lost")]
    table = {row["level"]: row for row in calibration(games).iter_rows(named=True)}
    assert set(table) == {"2", "3", "prob"}
    assert (table["3"]["n"], table["3"]["assumed"], table["3"]["actual"]) == (1, 0.74, 1.0)
    assert (table["prob"]["assumed"], table["prob"]["actual"]) == (0.6, 0.0)
    assert table["2"]["actual"] == 1.0


def test_score_and_report_commands(stored: None, repo: Path) -> None:
    runner = CliRunner()
    scored = runner.invoke(app, ["score", "--season", "2026", "--week", "8"])
    assert scored.exit_code == 0, scored.output
    assert "Season to date" in scored.output
    fitted = load_weights()
    assert fitted is not None
    assert fitted.weight == 0.12
    assert fitted.n_games == 3

    reported = runner.invoke(app, ["report", "--season", "2026"], env={"COLUMNS": "160"})
    assert reported.exit_code == 0, reported.output
    text = (repo / "reports" / "2026-w08.md").read_text()
    assert text.startswith("# 2026 week 8 report")
    assert "| MIA@NE | 17-10 | NE 60.0% | MIA c3 | MIA 55.0% | MIA | FLIPPED final |" in text
    assert "| 8 | NYG@ATL | NYG 55.0% | ATL 52.0% | NYG | lost |" in text
    assert "Week 8 | Owner only, picked games | 2 | 50.0%" in text
    assert "LA@SF" not in text.split("## Scorecard")[0]


def test_report_without_predictions_fails_cleanly(repo: Path) -> None:
    result = CliRunner().invoke(app, ["report", "--season", "2026"])
    assert result.exit_code == 1
    assert "no stored predictions" in result.output
