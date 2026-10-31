from collections import Counter
from typing import Any

import polars as pl
import pytest

from football.backtest.splits import LeakageError, check_split, walk_forward

TEST_SEASONS = [2020, 2021, 2022]


def test_weekly_splits_never_leak_and_cover_each_game_once(features: pl.DataFrame) -> None:
    seen: Counter[str] = Counter()
    for split in walk_forward(features, TEST_SEASONS, "weekly"):
        assert split.refit
        last_train = split.train.select(pl.col("kickoff_utc").max()).item()
        assert last_train < split.test.select(pl.col("kickoff_utc").min()).item()
        assert split.train["home_win"].null_count() == 0
        assert not set(split.train["game_id"]) & set(split.test["game_id"])
        assert split.test["season"].unique().to_list() == [split.season]
        assert split.test["week"].unique().to_list() == [split.week]
        seen.update(split.test["game_id"].to_list())
    expected = features.filter(
        pl.col("season").is_in(TEST_SEASONS) & pl.col("home_win").is_not_null()
    )["game_id"]
    assert set(seen) == set(expected)
    assert set(seen.values()) == {1}


def test_weekly_training_window_expands(features: pl.DataFrame) -> None:
    sizes = [split.train.height for split in walk_forward(features, [2021], "weekly")]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_season_mode_fits_once_on_prior_seasons(features: pl.DataFrame) -> None:
    splits = list(walk_forward(features, [2021, 2022], "season"))
    for split in splits:
        assert split.train["season"].max() == split.season - 1
    refits = [(split.season, split.week) for split in splits if split.refit]
    assert refits == [(2021, 1), (2022, 1)]


def test_playoff_weeks_flagged_and_min_train_season(features: pl.DataFrame) -> None:
    splits = list(walk_forward(features, [2022], "weekly", min_train_season=2020))
    assert [split.playoff for split in splits] == [False] * 6 + [True]
    assert all(split.train["season"].min() == 2020 for split in splits)


def test_empty_training_window_is_an_error(features: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="no labeled training games"):
        list(walk_forward(features, [2018], "weekly"))


def test_check_split_detects_overlap(features: pl.DataFrame) -> None:
    week = features.filter((pl.col("season") == 2020) & (pl.col("week") == 3))
    earlier = features.filter((pl.col("season") == 2020) & (pl.col("week") == 2))
    with pytest.raises(LeakageError):
        check_split(week, earlier)
    with pytest.raises(LeakageError):
        check_split(week, week)


def test_unknown_retrain_mode(features: pl.DataFrame) -> None:
    mode: Any = "monthly"
    with pytest.raises(ValueError, match="retrain"):
        list(walk_forward(features, [2020], mode))
