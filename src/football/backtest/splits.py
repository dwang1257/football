from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, get_args

import polars as pl

Retrain = Literal["weekly", "season"]
RETRAIN_MODES: tuple[str, ...] = get_args(Retrain)
REGULAR = "REG"


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Split:
    season: int
    week: int
    playoff: bool
    refit: bool
    train: pl.DataFrame
    test: pl.DataFrame


def labeled(features: pl.DataFrame) -> pl.DataFrame:
    return features.filter(pl.col("home_win").is_not_null())


def check_split(train: pl.DataFrame, test: pl.DataFrame) -> None:
    if not train.height:
        return
    last_train = train.select(pl.col("kickoff_utc").max()).item()
    first_test = test.select(pl.col("kickoff_utc").min()).item()
    if last_train is None or first_test is None or not last_train < first_test:
        raise LeakageError(f"train kickoff {last_train} is not before test kickoff {first_test}")
    shared = set(train["game_id"].to_list()) & set(test["game_id"].to_list())
    if shared:
        raise LeakageError(f"games in both train and test: {sorted(shared)[:5]}")


def walk_forward(
    features: pl.DataFrame,
    seasons: Iterable[int],
    retrain: Retrain = "weekly",
    min_train_season: int | None = None,
) -> Iterator[Split]:
    if retrain not in RETRAIN_MODES:
        raise ValueError(f"retrain must be one of {RETRAIN_MODES}, got {retrain!r}")
    games = labeled(features).sort("kickoff_utc", "game_id")
    pool = games if min_train_season is None else games.filter(pl.col("season") >= min_train_season)
    for season in sorted(set(seasons)):
        season_games = games.filter(pl.col("season") == season)
        season_train = pool.filter(pl.col("season") < season)
        weeks = sorted(season_games["week"].unique().to_list())
        for position, week in enumerate(weeks):
            test = season_games.filter(pl.col("week") == week)
            first_kickoff = test["kickoff_utc"].min()
            if retrain == "weekly":
                train = pool.filter(pl.col("kickoff_utc") < first_kickoff)
            else:
                train = season_train
            if not train.height:
                raise ValueError(f"no labeled training games before season {season} week {week}")
            check_split(train, test)
            yield Split(
                season=season,
                week=week,
                playoff=bool((test["game_type"] != REGULAR).any()),
                refit=retrain == "weekly" or position == 0,
                train=train,
                test=test,
            )
