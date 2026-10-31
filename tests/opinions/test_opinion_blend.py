import pytest
from hypothesis import given
from hypothesis import strategies as st

from football.opinions.blend import BlendParams, blend, default_conf_table, logit

PARAMS = BlendParams(
    weight=0.12, team_weight=0.5, cap=0.35, cap_max_conf=0.5, conf_table=default_conf_table()
)


def owner_home_prob(pick_home: bool, conf: int) -> float:
    p = PARAMS.pick_prob(conf, None)
    return p if pick_home else 1.0 - p


def test_conf_table_matches_plan() -> None:
    assert default_conf_table() == pytest.approx({1: 0.58, 2: 0.66, 3: 0.74, 4: 0.82, 5: 0.90})


def test_kc_at_buf_owner_buf_conf_4_keeps_kc() -> None:
    result = blend(0.30, owner_home_prob(True, 4), 0.0, 4, PARAMS)
    assert 1.0 - result.blended_prob == pytest.approx(0.637, abs=5e-4)
    assert not result.flipped
    assert not result.capped
    assert result.delta_pp == pytest.approx(6.27, abs=0.01)


def test_atl_owner_nyg_conf_5_flips() -> None:
    result = blend(0.55, owner_home_prob(False, 5), 0.0, 5, PARAMS)
    assert result.blended_prob == pytest.approx(0.478, abs=5e-4)
    assert result.flipped


def test_atl_owner_nyg_conf_3_keeps_atl() -> None:
    result = blend(0.55, owner_home_prob(False, 3), 0.0, 3, PARAMS)
    assert result.blended_prob == pytest.approx(0.513, abs=5e-4)
    assert not result.flipped


def test_given_prob_is_used_instead_of_conf() -> None:
    assert PARAMS.pick_prob(None, 0.61) == 0.61
    result = blend(0.5, 0.61, 0.0, None, PARAMS)
    assert result.delta_logit == pytest.approx(0.12 * logit(0.61))


def test_team_nudge_uses_k_and_team_weight() -> None:
    result = blend(0.5, None, 2.0, None, PARAMS)
    assert result.delta_logit == pytest.approx(0.5 * 0.12 * 2.0)
    assert result.user_prob is None


def test_cap_applies_to_sum_and_is_flagged() -> None:
    result = blend(0.5, 0.99, 3.0, 4, PARAMS)
    assert result.delta_logit == pytest.approx(0.35)
    assert result.capped
    max_conf = blend(0.5, 0.99, 3.0, 5, PARAMS)
    assert max_conf.delta_logit == pytest.approx(0.5)


def test_no_opinion_leaves_model_unchanged() -> None:
    result = blend(0.63, None, 0.0, None, PARAMS)
    assert result.blended_prob == pytest.approx(0.63)
    assert result.delta_pp == pytest.approx(0.0)


@given(
    model=st.floats(0.59, 0.999),
    conf=st.integers(1, 4),
    points=st.floats(-6.0, 6.0),
    home_model=st.booleans(),
)
def test_model_pick_at_59_percent_never_flips_below_conf_5(
    model: float, conf: int, points: float, home_model: bool
) -> None:
    home_prob = model if home_model else 1.0 - model
    against = owner_home_prob(not home_model, conf)
    nudge = -abs(points) if home_model else abs(points)
    assert not blend(home_prob, against, nudge, conf, PARAMS).flipped


@given(model=st.floats(0.623, 0.999), points=st.floats(-6.0, 6.0), home_model=st.booleans())
def test_model_pick_at_62_percent_never_flips_at_conf_5(
    model: float, points: float, home_model: bool
) -> None:
    home_prob = model if home_model else 1.0 - model
    against = owner_home_prob(not home_model, 5)
    nudge = -abs(points) if home_model else abs(points)
    assert not blend(home_prob, against, nudge, 5, PARAMS).flipped


def test_flip_thresholds_are_tight() -> None:
    assert blend(0.58, 0.01, -6.0, 4, PARAMS).flipped
    assert blend(0.615, 0.01, -6.0, 5, PARAMS).flipped
