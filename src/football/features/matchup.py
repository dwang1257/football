from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from football.features.asof import latest_before
from football.features.columns import MATCHUP_COLUMNS, UNITS
from football.features.common import (
    decay_factor,
    event_time,
    franchise_expr,
    last_per_time,
    latest_global,
    long_sides,
    safe_div,
    weighted_moments,
    widen,
)
from football.schemas import MATCHUP_STATS, UTC_TS

UNIT_STATS: tuple[tuple[str, str], ...] = tuple(
    (unit, stat) for unit in UNITS for stat in MATCHUP_STATS
)
SLOTS: tuple[str, ...] = tuple(f"{unit}_{stat}" for unit, stat in UNIT_STATS)
LG_COLS: tuple[str, ...] = tuple(f"lg_{stat}" for stat in MATCHUP_STATS)


@dataclass(frozen=True)
class MatchupParams:
    half_life_games: float = 6.0
    league_half_life: float = 256.0
    prior_games: float = 6.0
    regression: float = 1.0 / 3.0
    stat_regression: Mapping[str, float] = field(
        default_factory=lambda: {"turnovers_per_game": 2.0 / 3.0}
    )
    opponent_passes: int = 0

    def regression_for(self, stat: str) -> float:
        return self.stat_regression.get(stat, self.regression)


def stat_exprs(prefix: str = "") -> dict[str, pl.Expr]:
    def c(name: str) -> pl.Expr:
        return pl.col(f"{prefix}{name}").cast(pl.Float64)

    filtered = safe_div(c("wp_filtered_epa"), c("wp_filtered_plays"))
    return {
        "plays_per_game": c("plays"),
        "pass_rate": safe_div(c("pass_plays"), c("pass_plays") + c("rush_plays")),
        "pass_yards_per_play": safe_div(c("pass_yards"), c("pass_plays")),
        "rush_yards_per_play": safe_div(c("rush_yards"), c("rush_plays")),
        "epa_per_play": pl.coalesce(filtered, safe_div(c("total_epa"), c("plays"))),
        "success_rate": safe_div(c("success_plays"), c("plays")),
        "turnovers_per_game": c("giveaways"),
        "points_per_game": c("points_for"),
        "red_zone_td_rate": safe_div(c("red_zone_tds"), c("red_zone_trips")),
        "start_yardline": c("avg_start_yardline"),
    }


def observations(team_games: pl.DataFrame) -> pl.DataFrame:
    rows = team_games.with_columns(
        franchise_expr("team").alias("team_fr"), franchise_expr("opponent").alias("opp_fr")
    )
    opp = rows.select(
        "game_id",
        pl.col("team_fr").alias("opp_fr"),
        pl.col("available_at").alias("opp_available_at"),
        *(pl.col(c).alias(f"o_{c}") for c in team_games.columns if c not in ("game_id", "team")),
    )
    joined = rows.join(opp, on=["game_id", "opp_fr"], how="inner")
    off = stat_exprs()
    against = stat_exprs("o_")
    against["turnovers_per_game"] = pl.col("takeaways").cast(pl.Float64)
    against["points_per_game"] = pl.col("points_against").cast(pl.Float64)
    return joined.select(
        "game_id",
        "season",
        "kickoff_utc",
        pl.col("team_fr").alias("team"),
        pl.col("opp_fr").alias("opp"),
        event_time("available_at", "opp_available_at", "kickoff_utc"),
        *(off[stat].fill_nan(None).alias(f"off_{stat}") for stat in MATCHUP_STATS),
        *(against[stat].fill_nan(None).alias(f"def_{stat}") for stat in MATCHUP_STATS),
    ).sort("available_at", "kickoff_utc", "game_id", "team")


def _matrix(df: pl.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    return df.select(pl.col(c).cast(pl.Float64).fill_null(np.nan) for c in cols).to_numpy()


@dataclass
class _TeamState:
    season: int
    prior_mean: np.ndarray
    prior_var: np.ndarray
    n: int = 0
    w: np.ndarray = field(default_factory=lambda: np.zeros(len(SLOTS)))
    w2: np.ndarray = field(default_factory=lambda: np.zeros(len(SLOTS)))
    s1: np.ndarray = field(default_factory=lambda: np.zeros(len(SLOTS)))
    s2: np.ndarray = field(default_factory=lambda: np.zeros(len(SLOTS)))

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        return weighted_moments(self.w, self.w2, self.s1, self.s2)

    def next_season(self, season: int) -> "_TeamState":
        mean, var = self.moments()
        return _TeamState(
            season,
            np.where(np.isnan(mean), self.prior_mean, mean),
            np.where(np.isnan(var), self.prior_var, var),
        )

    def add(self, x: np.ndarray, d: float) -> None:
        mask = ~np.isnan(x)
        xv = np.where(mask, x, 0.0)
        self.w = self.w * d + mask
        self.w2 = self.w2 * (d * d) + mask
        self.s1 = self.s1 * d + xv
        self.s2 = self.s2 * d + xv * xv
        self.n += 1


def team_history(obs: pl.DataFrame, params: MatchupParams) -> pl.DataFrame:
    values = _matrix(obs, SLOTS)
    teams = obs["team"].to_list()
    seasons = obs["season"].to_list()
    n_rows, n_slots = values.shape
    out_season = np.zeros(n_rows, dtype=np.int32)
    out_n = np.zeros(n_rows, dtype=np.int32)
    blocks = {name: np.full((n_rows, n_slots), np.nan) for name in ("cm", "cv", "pm", "pv")}
    d = decay_factor(params.half_life_games)
    empty = np.full(n_slots, np.nan)
    state: dict[str, _TeamState] = {}
    for i in range(n_rows):
        team, season = teams[i], int(seasons[i])
        st = state.get(team)
        if st is None:
            st = _TeamState(season, empty, empty)
        elif st.season != season:
            st = st.next_season(season)
        state[team] = st
        st.add(values[i], d)
        out_season[i] = season
        out_n[i] = st.n
        blocks["cm"][i], blocks["cv"][i] = st.moments()
        blocks["pm"][i], blocks["pv"][i] = st.prior_mean, st.prior_var
    data: dict[str, pl.Series] = {
        "team": pl.Series(teams, dtype=pl.String),
        "available_at": obs["available_at"].cast(UTC_TS),
        "snap_season": pl.Series(out_season, dtype=pl.Int32),
        "snap_n": pl.Series(out_n, dtype=pl.Int32),
    }
    for name, block in blocks.items():
        for j, slot in enumerate(SLOTS):
            data[f"{name}_{slot}"] = pl.Series(block[:, j]).fill_nan(None)
    return last_per_time(pl.DataFrame(data), ["team"])


def league_history(obs: pl.DataFrame, params: MatchupParams) -> pl.DataFrame:
    cols = tuple(f"off_{stat}" for stat in MATCHUP_STATS)
    values = _matrix(obs, cols)
    n_rows, n_stats = values.shape
    d = decay_factor(params.league_half_life)
    w, w2, s1, s2 = (np.zeros(n_stats) for _ in range(4))
    means = np.full((n_rows, n_stats), np.nan)
    variances = np.full((n_rows, n_stats), np.nan)
    for i in range(n_rows):
        x = values[i]
        mask = ~np.isnan(x)
        xv = np.where(mask, x, 0.0)
        w = w * d + mask
        w2 = w2 * d * d + mask
        s1 = s1 * d + xv
        s2 = s2 * d + xv * xv
        means[i], variances[i] = weighted_moments(w, w2, s1, s2)
    data: dict[str, pl.Series] = {"available_at": obs["available_at"].cast(UTC_TS)}
    for j, stat in enumerate(MATCHUP_STATS):
        data[f"lg_{stat}"] = pl.Series(means[:, j]).fill_nan(None)
        data[f"lgv_{stat}"] = pl.Series(variances[:, j]).fill_nan(None)
    return last_per_time(pl.DataFrame(data), [])


def ratings_at(
    queries: pl.DataFrame,
    teams: pl.DataFrame,
    league: pl.DataFrame,
    params: MatchupParams,
) -> pl.DataFrame:
    joined = latest_global(latest_before(queries, teams, by=["team"]), league)
    same = pl.col("snap_season") == pl.col("season")
    n = pl.when(same).then(pl.col("snap_n")).otherwise(0).fill_null(0)
    w = (1.0 - n.cast(pl.Float64) / params.prior_games).clip(0.0, 1.0)
    exprs: list[pl.Expr] = [n.alias("games_played")]
    for unit, stat in UNIT_STATS:
        slot = f"{unit}_{stat}"
        lg, lgv = pl.col(f"lg_{stat}"), pl.col(f"lgv_{stat}")
        cur = pl.when(same).then(pl.col(f"cm_{slot}"))
        cur_var = pl.when(same).then(pl.col(f"cv_{slot}"))
        prior_raw = (
            pl.when(same)
            .then(pl.col(f"pm_{slot}"))
            .otherwise(pl.coalesce(f"cm_{slot}", f"pm_{slot}"))
        )
        prior_var_raw = (
            pl.when(same)
            .then(pl.col(f"pv_{slot}"))
            .otherwise(pl.coalesce(f"cv_{slot}", f"pv_{slot}"))
        )
        keep = 1.0 - params.regression_for(stat)
        prior = pl.coalesce(lg + keep * (prior_raw - lg), lg)
        prior_var = pl.coalesce(prior_var_raw, lgv)
        mean = pl.when(cur.is_null()).then(prior).otherwise(w * prior + (1.0 - w) * cur)
        var = (
            pl.when(cur_var.is_null())
            .then(prior_var)
            .otherwise(w * prior_var + (1.0 - w) * cur_var)
        )
        exprs.append(pl.coalesce(mean, cur).alias(slot))
        exprs.append(pl.coalesce(var, cur_var).sqrt().alias(f"{slot}_sd"))
    for stat in MATCHUP_STATS:
        exprs.append(pl.col(f"lg_{stat}"))
        exprs.append(pl.col(f"lgv_{stat}").sqrt().alias(f"lg_{stat}_sd"))
    return joined.select("qid", *exprs)


def _adjusted(
    obs: pl.DataFrame,
    teams: pl.DataFrame,
    league: pl.DataFrame,
    params: MatchupParams,
) -> pl.DataFrame:
    queries = obs.select(
        pl.int_range(pl.len()).alias("qid"),
        pl.col("opp").alias("team"),
        "season",
        pl.min_horizontal("kickoff_utc", "available_at").alias("cutoff"),
    )
    opp = ratings_at(queries.drop_nulls("team"), teams, league, params)
    opp = opp.select("qid", *(pl.col(slot).alias(f"{slot}_opp") for slot in SLOTS), *LG_COLS)
    merged = obs.with_columns(pl.int_range(pl.len()).alias("qid")).join(opp, on="qid", how="left")
    adjustments: list[pl.Expr] = []
    for unit, stat in UNIT_STATS:
        other = "def" if unit == "off" else "off"
        edge = (pl.col(f"{other}_{stat}_opp") - pl.col(f"lg_{stat}")).fill_null(0.0)
        adjustments.append((pl.col(f"{unit}_{stat}") - edge).alias(f"{unit}_{stat}"))
    return merged.with_columns(adjustments).select(obs.columns)


def matchup_histories(
    team_games: pl.DataFrame, params: MatchupParams
) -> tuple[pl.DataFrame, pl.DataFrame]:
    obs = observations(team_games)
    league = league_history(obs, params)
    teams = team_history(obs, params)
    for _ in range(params.opponent_passes):
        teams = team_history(_adjusted(obs, teams, league, params), params)
    return teams, league


def matchup_features(
    targets: pl.DataFrame,
    team_games: pl.DataFrame,
    params: MatchupParams | None = None,
) -> pl.DataFrame:
    params = params or MatchupParams()
    teams, league = matchup_histories(team_games, params)
    long = long_sides(targets).with_columns(pl.int_range(pl.len()).alias("qid"))
    rated = long.select("qid", "game_id", "side").join(
        ratings_at(long.select("qid", "team", "season", "cutoff"), teams, league, params),
        on="qid",
        how="left",
    )
    side_cols = [*SLOTS, *(f"{slot}_sd" for slot in SLOTS), "games_played"]
    wide = widen(rated, side_cols)
    league_cols = [c for c in MATCHUP_COLUMNS if c.startswith("lg_")]
    lg = rated.filter(pl.col("side") == "home").select("game_id", *league_cols)
    return (
        wide.join(lg, on="game_id", how="left")
        .with_columns(pl.col(c).cast(pl.Float64) for c in MATCHUP_COLUMNS)
        .select("game_id", *MATCHUP_COLUMNS)
    )
