from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from football.opinions.blend import BlendParams, default_conf_table
from football.opinions.learning import (
    current_params,
    fit_conf_table,
    fit_weights,
    load_weights,
    save_weights,
)

NOW = datetime(2026, 11, 3, tzinfo=UTC)
DEFAULTS = BlendParams.from_settings()


def history(
    n: int, owner_right: float, seed: int = 7, conf: int = 3, nudge: float = 0.0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    home_win = (rng.random(n) < 0.5).astype(float)
    right = rng.random(n) < owner_right
    picked_home = np.where(right, home_win == 1.0, home_win == 0.0)
    return pl.DataFrame(
        {
            "model_prob": np.full(n, 0.5),
            "user_pick_prob": np.full(n, 0.74),
            "user_conf": pl.Series([conf] * n, dtype=pl.Int32),
            "user_picked_home": picked_home,
            "team_nudge_points": np.where(picked_home, nudge, -nudge),
            "home_win": home_win,
        }
    )


def test_defaults_kept_below_40_games() -> None:
    fitted = fit_weights(history(39, 1.0), DEFAULTS, NOW)
    assert fitted.weight == DEFAULTS.weight
    assert fitted.team_weight == DEFAULTS.team_weight
    assert fitted.n_games == 39


def test_good_owner_raises_weight_with_shrinkage_and_clamp() -> None:
    fitted = fit_weights(history(60, 0.9), DEFAULTS, NOW)
    assert DEFAULTS.weight < fitted.weight <= 0.30
    assert fitted.weight <= (60 * 0.5 + 100 * 0.12) / 160 + 1e-9


def test_bad_owner_drifts_toward_zero() -> None:
    fitted = fit_weights(history(400, 0.2), DEFAULTS, NOW)
    assert fitted.weight == pytest.approx(400 * 0.0 / 500 + 100 * 0.12 / 500)


def test_team_weight_fits_on_nudge_games() -> None:
    games = history(80, 0.95, nudge=2.0).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("user_pick_prob"),
        pl.lit(None, dtype=pl.Int32).alias("user_conf"),
    )
    fitted = fit_weights(games, DEFAULTS, NOW)
    assert fitted.n_games == 0
    assert fitted.n_team_games == 80
    assert DEFAULTS.team_weight < fitted.team_weight <= 1.25


def test_conf_table_uses_prior_and_is_forced_to_increase() -> None:
    table = fit_conf_table(history(10, 1.0, conf=2))
    pooled = ((10 + 2 * 0.66) + 2 * 0.74 + 2 * 0.82 + 2 * 0.90) / 18
    assert table[1] == pytest.approx(0.58)
    assert table[2] == pytest.approx(pooled)
    assert [table[c] - table[c - 1] for c in range(3, 6)] == pytest.approx([0.001] * 3)
    assert fit_conf_table(history(0, 1.0)) == pytest.approx(default_conf_table())


def test_conf_table_never_drops_below_a_coin_flip() -> None:
    table = fit_conf_table(history(30, 0.0, conf=4))
    assert table[1] == pytest.approx(0.5)
    assert all(table[c] < table[c + 1] for c in range(1, 5))


def test_saved_weights_feed_current_params() -> None:
    assert current_params() == DEFAULTS
    fitted = fit_weights(history(60, 0.9), DEFAULTS, NOW)
    save_weights(fitted)
    assert load_weights() == fitted
    params = current_params()
    assert params.weight == fitted.weight
    assert params.conf_table == fitted.conf_table
    assert params.cap == DEFAULTS.cap
