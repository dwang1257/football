import math

import numpy as np
import polars as pl
import pytest

from football.backtest.metrics import (
    accuracy,
    brier,
    compare,
    expected_calibration_error,
    log_loss,
    mcnemar,
    metrics_by_split,
    paired,
    paired_bootstrap,
    reliability,
    seasons_won,
    summary,
)

Y = np.array([1.0, 0.0, 0.5, 1.0])
P = np.array([0.8, 0.4, 0.5, 0.3])


def frame(
    y: list[float], p: list[float], season: int = 2020, weeks: list[int] | None = None
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"{season}_g{index}" for index in range(len(y))],
            "season": [season] * len(y),
            "week": weeks or [1] * len(y),
            "game_type": ["REG"] * len(y),
            "home_win": y,
            "home_win_prob": p,
        }
    )


def test_accuracy_excludes_ties() -> None:
    assert accuracy(Y, P) == pytest.approx(2 / 3)


def test_accuracy_gives_half_credit_for_coin_flip_predictions() -> None:
    assert accuracy(np.array([1.0, 0.0]), np.array([0.5, 0.2])) == pytest.approx(0.75)


def test_log_loss_scores_ties_as_half() -> None:
    expected = -(math.log(0.8) + math.log(0.6) + math.log(0.5) + math.log(0.3)) / 4
    assert log_loss(Y, P) == pytest.approx(expected)
    assert log_loss(np.array([0.5]), np.array([0.5])) == pytest.approx(math.log(2))


def test_log_loss_is_finite_for_certain_wrong_predictions() -> None:
    assert math.isfinite(log_loss(np.array([1.0]), np.array([0.0])))


def test_brier() -> None:
    assert brier(Y, P) == pytest.approx((0.04 + 0.16 + 0.0 + 0.49) / 4)


def test_ece_and_reliability_table() -> None:
    y = np.array([0.0, 1.0, 1.0, 1.0])
    p = np.array([0.15, 0.15, 0.85, 0.85])
    assert expected_calibration_error(y, p) == pytest.approx(0.25)
    table = reliability(y, p)
    assert table.height == 10
    filled = table.filter(pl.col("n") > 0)
    assert filled["bin"].to_list() == [1, 8]
    assert filled["observed"].to_list() == pytest.approx([0.5, 1.0])
    assert filled["mean_prob"].to_list() == pytest.approx([0.15, 0.85])
    edges = reliability(np.array([0.0, 1.0]), np.array([0.0, 1.0])).filter(pl.col("n") > 0)
    assert edges["bin"].to_list() == [0, 9]


def test_summary_ignores_unlabeled_rows() -> None:
    df = frame([1.0, 0.0, 0.5, 1.0], P.tolist()).vstack(
        frame([1.0], [0.9]).with_columns(home_win=pl.lit(None, pl.Float64), game_id=pl.lit("x"))
    )
    stats = summary(df)
    assert stats["n"] == 4
    assert stats["ties"] == 1
    assert stats["log_loss"] == pytest.approx(log_loss(Y, P))


def test_metrics_by_split() -> None:
    df = pl.concat(
        [
            frame([1.0, 0.0], [0.7, 0.4], weeks=[1, 6]),
            frame([1.0], [0.2], season=2021, weeks=[19]).with_columns(game_type=pl.lit("WC")),
        ]
    )
    table = metrics_by_split(df)
    assert table["split"].to_list() == [
        "overall",
        "weeks 1-4",
        "weeks 5+",
        "playoffs",
        "2020",
        "2021",
    ]
    rows = {row["split"]: row for row in table.iter_rows(named=True)}
    assert rows["overall"]["n"] == 3
    assert rows["weeks 1-4"]["log_loss"] == pytest.approx(-math.log(0.7))
    assert rows["weeks 5+"]["log_loss"] == pytest.approx(-math.log(0.6))
    assert rows["playoffs"]["accuracy"] == 0.0


def test_bootstrap_is_deterministic_and_seeded() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200).astype(float).tolist()
    weeks = [index // 10 + 1 for index in range(200)]
    reference = frame(y, rng.uniform(0.2, 0.8, 200).tolist(), weeks=weeks)
    candidate = frame(y, rng.uniform(0.2, 0.8, 200).tolist(), weeks=weeks)
    pairs = paired(candidate, reference)
    first = paired_bootstrap(pairs, draws=2000, seed=5)
    assert first == paired_bootstrap(pairs, draws=2000, seed=5)
    assert first != paired_bootstrap(pairs, draws=2000, seed=6)
    assert first.n_weeks == 20
    assert first.n_games == 200
    assert first.log_loss.lo < first.log_loss.mean < first.log_loss.hi


def test_bootstrap_resamples_whole_weeks() -> None:
    reference = frame([1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5])
    candidate = frame([1.0, 0.0, 1.0, 0.0], [0.9, 0.9, 0.6, 0.2])
    result = paired_bootstrap(paired(candidate, reference), draws=500, seed=1)
    y = np.array([1.0, 0.0, 1.0, 0.0])
    expected = log_loss(y, np.array([0.9, 0.9, 0.6, 0.2])) - math.log(2)
    assert result.n_weeks == 1
    assert result.log_loss.mean == pytest.approx(expected)
    assert result.log_loss.lo == pytest.approx(expected)
    assert result.log_loss.hi == pytest.approx(expected)


def test_bootstrap_identical_predictions_has_zero_interval() -> None:
    df = frame([1.0, 0.0, 1.0], [0.6, 0.3, 0.55], weeks=[1, 2, 3])
    result = paired_bootstrap(paired(df, df), draws=100, seed=1)
    assert (result.log_loss.lo, result.log_loss.mean, result.log_loss.hi) == (0.0, 0.0, 0.0)
    assert not result.brier.excludes_zero()


def test_mcnemar_known_case() -> None:
    y = [1.0] * 9 + [0.0] + [1.0, 0.0] + [0.5]
    candidate = frame(y, [0.7] * 9 + [0.6] + [0.8, 0.2] + [0.9])
    reference = frame(y, [0.3] * 9 + [0.4] + [0.8, 0.2] + [0.1])
    result = mcnemar(paired(candidate, reference))
    assert (result.candidate_only, result.reference_only) == (9, 1)
    assert result.p_value == pytest.approx(22 / 1024)


def test_mcnemar_without_disagreements() -> None:
    df = frame([1.0, 0.0], [0.6, 0.3])
    assert mcnemar(paired(df, df)).p_value == 1.0


def test_seasons_won_and_adoption() -> None:
    reference = pl.concat([frame([1.0, 0.0], [0.6, 0.4]), frame([1.0, 0.0], [0.6, 0.4], 2021)])
    candidate = pl.concat([frame([1.0, 0.0], [0.7, 0.3]), frame([1.0, 0.0], [0.5, 0.5], 2021)])
    won = seasons_won(paired(candidate, reference))
    assert (won.won, won.total, won.share) == (1, 2, 0.5)
    better = pl.concat([frame([1.0, 0.0], [0.7, 0.3]), frame([1.0, 0.0], [0.7, 0.3], 2021)])
    comparison = compare(better, reference, draws=200, seed=1)
    assert comparison.seasons.share == 1.0
    assert comparison.adoptable
    assert not compare(reference, better, draws=200, seed=1).adoptable
