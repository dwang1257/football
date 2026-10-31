import polars as pl

from football.features.columns import (
    ELO_FEATURES,
    INJURY_FEATURES,
    OWNER_FEATURES,
    QB_FEATURES,
    SITUATIONAL_FEATURES,
    WEATHER_FEATURES,
)
from football.models.base import Track
from football.models.common import (
    MARKET_COLUMNS,
    WIN_TOTAL_COLUMNS,
    Feature,
    column,
    paired_features,
)
from football.models.matchup import (
    Combine,
    matchup_expressions,
    multiplicative_stats,
    stat_columns,
    usable_stats,
)

DEFAULT_GROUPS = "elo,matchup,situational,qb"
MATCHUP_MU = "matchup_mu"

_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "elo": ELO_FEATURES,
    "situational": SITUATIONAL_FEATURES,
    "qb": QB_FEATURES,
    "injury": INJURY_FEATURES,
    "weather": WEATHER_FEATURES,
    "owner": tuple(c for c in OWNER_FEATURES if c not in WIN_TOTAL_COLUMNS),
    "market": MARKET_COLUMNS,
}
GROUPS: tuple[str, ...] = (*_COLUMN_GROUPS, "matchup")


def parse_groups(text: str, track: Track) -> list[str]:
    groups = [g.strip() for g in text.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; choose from {list(GROUPS)}")
    if "market" in groups and track == "independent":
        raise ValueError("feature group 'market' requires track=market")
    if track == "market" and "market" not in groups:
        groups.append("market")
    return list(dict.fromkeys(groups))


def matchup_features(train: pl.DataFrame, combine: Combine) -> list[Feature]:
    stats, _ = usable_stats(train)
    exprs = matchup_expressions(stats, multiplicative_stats(train, stats), combine)
    return [
        Feature(name, expr, stat_columns(name.removeprefix("mx_").removesuffix("_diff")))
        for name, expr in exprs.items()
    ]


def feature_set(
    groups: list[str], train: pl.DataFrame, combine: Combine = "league_relative"
) -> list[Feature]:
    columns = [c for g in groups if g in _COLUMN_GROUPS for c in _COLUMN_GROUPS[g]]
    features = paired_features(columns)
    if "matchup" in groups:
        features.extend(matchup_features(train, combine))
    return features


def with_matchup_mu(features: list[Feature]) -> list[Feature]:
    return [*features, column(MATCHUP_MU)]
