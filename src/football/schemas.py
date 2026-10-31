from dataclasses import dataclass, field

import polars as pl

UTC_TS = pl.Datetime("us", "UTC")


@dataclass(frozen=True)
class TableSpec:
    name: str
    keys: tuple[str, ...]
    columns: dict[str, pl.DataType] = field(default_factory=dict)


class SchemaError(ValueError):
    pass


def validate(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    missing = [name for name in spec.columns if name not in df.columns]
    if missing:
        raise SchemaError(f"{spec.name}: missing columns {missing}")
    wrong = {
        name: (str(df.schema[name]), str(dtype))
        for name, dtype in spec.columns.items()
        if df.schema[name] != dtype
    }
    if wrong:
        raise SchemaError(f"{spec.name}: wrong dtypes (actual, expected) {wrong}")
    for key in spec.keys:
        if df[key].null_count():
            raise SchemaError(f"{spec.name}: null key column {key}")
    if df.select(spec.keys).is_duplicated().any():
        raise SchemaError(f"{spec.name}: duplicate keys {spec.keys}")
    if "available_at" in spec.columns and df["available_at"].null_count():
        raise SchemaError(f"{spec.name}: null available_at")
    ordered = list(spec.columns)
    extra = [name for name in df.columns if name not in spec.columns]
    return df.select(ordered + extra)


GAMES = TableSpec(
    "games",
    ("game_id",),
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "game_type": pl.String(),
        "kickoff_utc": UTC_TS,
        "home_team": pl.String(),
        "away_team": pl.String(),
        "home_score": pl.Int32(),
        "away_score": pl.Int32(),
        "neutral_site": pl.Boolean(),
        "stadium_id": pl.String(),
        "roof": pl.String(),
        "surface": pl.String(),
        "div_game": pl.Boolean(),
        "home_rest": pl.Int32(),
        "away_rest": pl.Int32(),
        "home_coach": pl.String(),
        "away_coach": pl.String(),
        "referee": pl.String(),
        "home_qb_id": pl.String(),
        "away_qb_id": pl.String(),
        "spread_line": pl.Float64(),
        "total_line": pl.Float64(),
        "home_moneyline": pl.Float64(),
        "away_moneyline": pl.Float64(),
        "temp_observed": pl.Float64(),
        "wind_observed": pl.Float64(),
        "available_at": UTC_TS,
    },
)

TEAM_GAMES = TableSpec(
    "team_games",
    ("game_id", "team"),
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "kickoff_utc": UTC_TS,
        "team": pl.String(),
        "opponent": pl.String(),
        "is_home": pl.Boolean(),
        "points_for": pl.Int32(),
        "points_against": pl.Int32(),
        "plays": pl.Int32(),
        "pass_plays": pl.Int32(),
        "rush_plays": pl.Int32(),
        "pass_yards": pl.Float64(),
        "rush_yards": pl.Float64(),
        "total_yards": pl.Float64(),
        "pass_epa": pl.Float64(),
        "rush_epa": pl.Float64(),
        "total_epa": pl.Float64(),
        "success_plays": pl.Int32(),
        "wp_filtered_plays": pl.Int32(),
        "wp_filtered_epa": pl.Float64(),
        "takeaways": pl.Int32(),
        "giveaways": pl.Int32(),
        "fumbles_lost": pl.Int32(),
        "interceptions_thrown": pl.Int32(),
        "drives": pl.Int32(),
        "red_zone_trips": pl.Int32(),
        "red_zone_tds": pl.Int32(),
        "avg_start_yardline": pl.Float64(),
        "available_at": UTC_TS,
    },
)

INJURIES = TableSpec(
    "injuries",
    ("season", "week", "team", "player_id"),
    {
        "season": pl.Int32(),
        "week": pl.Int32(),
        "game_type": pl.String(),
        "team": pl.String(),
        "player_id": pl.String(),
        "full_name": pl.String(),
        "position": pl.String(),
        "report_status": pl.String(),
        "practice_status": pl.String(),
        "available_at": UTC_TS,
    },
)

DEPTH_CHARTS = TableSpec(
    "depth_charts",
    ("season", "week", "team", "position", "depth_rank", "player_id", "captured_at"),
    {
        "season": pl.Int32(),
        "week": pl.Int32(),
        "team": pl.String(),
        "player_id": pl.String(),
        "full_name": pl.String(),
        "position": pl.String(),
        "depth_rank": pl.Int32(),
        "captured_at": UTC_TS,
        "available_at": UTC_TS,
    },
)

QB_GAMES = TableSpec(
    "qb_games",
    ("game_id", "team", "player_id"),
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "kickoff_utc": UTC_TS,
        "team": pl.String(),
        "player_id": pl.String(),
        "full_name": pl.String(),
        "is_starter": pl.Boolean(),
        "dropbacks": pl.Int32(),
        "qb_epa": pl.Float64(),
        "cpoe": pl.Float64(),
        "available_at": UTC_TS,
    },
)

LINES = TableSpec(
    "lines",
    ("game_id", "source", "book", "kind", "captured_at"),
    {
        "game_id": pl.String(),
        "source": pl.String(),
        "book": pl.String(),
        "kind": pl.String(),
        "captured_at": UTC_TS,
        "home_spread": pl.Float64(),
        "total": pl.Float64(),
        "home_ml": pl.Float64(),
        "away_ml": pl.Float64(),
        "available_at": UTC_TS,
    },
)

BENCHMARKS = TableSpec(
    "benchmarks",
    ("game_id", "source"),
    {
        "game_id": pl.String(),
        "source": pl.String(),
        "home_win_prob": pl.Float64(),
        "home_margin": pl.Float64(),
        "available_at": UTC_TS,
    },
)

TEAM_RATINGS = TableSpec(
    "team_ratings",
    ("source", "season", "week", "team", "captured_at"),
    {
        "source": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "team": pl.String(),
        "rating": pl.Float64(),
        "offense": pl.Float64(),
        "defense": pl.Float64(),
        "special_teams": pl.Float64(),
        "captured_at": UTC_TS,
        "available_at": UTC_TS,
    },
)

WIN_TOTALS = TableSpec(
    "win_totals",
    ("season", "team", "source"),
    {
        "season": pl.Int32(),
        "team": pl.String(),
        "source": pl.String(),
        "win_total": pl.Float64(),
        "over_prob": pl.Float64(),
        "available_at": UTC_TS,
    },
)

STADIUMS = TableSpec(
    "stadiums",
    ("stadium_id",),
    {
        "stadium_id": pl.String(),
        "name": pl.String(),
        "lat": pl.Float64(),
        "lon": pl.Float64(),
        "elevation_m": pl.Float64(),
        "tz": pl.String(),
        "roof_type": pl.String(),
        "surface": pl.String(),
        "weather_station": pl.String(),
        "is_international": pl.Boolean(),
    },
)

TEAM_HOMES = TableSpec(
    "team_homes",
    ("season", "team"),
    {
        "season": pl.Int32(),
        "team": pl.String(),
        "stadium_id": pl.String(),
        "lat": pl.Float64(),
        "lon": pl.Float64(),
        "tz": pl.String(),
        "roof_type": pl.String(),
    },
)

WEATHER_FORECASTS = TableSpec(
    "weather_forecasts",
    ("game_id", "source", "run_time"),
    {
        "game_id": pl.String(),
        "source": pl.String(),
        "station": pl.String(),
        "run_time": UTC_TS,
        "valid_time": UTC_TS,
        "temp_f": pl.Float64(),
        "wind_mph": pl.Float64(),
        "wind_gust_mph": pl.Float64(),
        "precip_prob": pl.Float64(),
        "available_at": UTC_TS,
    },
)

WEATHER_OBSERVED = TableSpec(
    "weather_observed",
    ("game_id", "source"),
    {
        "game_id": pl.String(),
        "source": pl.String(),
        "temp_f": pl.Float64(),
        "wind_mph": pl.Float64(),
        "precip_in": pl.Float64(),
        "available_at": UTC_TS,
    },
)

PRACTICE_REPORTS = TableSpec(
    "practice_reports",
    ("season", "week", "team", "player_name", "report_day", "captured_at"),
    {
        "season": pl.Int32(),
        "week": pl.Int32(),
        "team": pl.String(),
        "player_name": pl.String(),
        "position": pl.String(),
        "injury": pl.String(),
        "report_day": pl.String(),
        "participation": pl.String(),
        "game_status": pl.String(),
        "captured_at": UTC_TS,
        "available_at": UTC_TS,
    },
)

FEATURES = TableSpec(
    "games",
    ("game_id",),
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "game_type": pl.String(),
        "kickoff_utc": UTC_TS,
        "cutoff": UTC_TS,
        "home_team": pl.String(),
        "away_team": pl.String(),
        "home_win": pl.Float64(),
        "home_margin": pl.Float64(),
    },
)

PREDICTIONS_MODEL = TableSpec(
    "model",
    ("run_id", "game_id"),
    {
        "run_id": pl.String(),
        "game_id": pl.String(),
        "model": pl.String(),
        "track": pl.String(),
        "home_win_prob": pl.Float64(),
        "home_margin": pl.Float64(),
        "cutoff": UTC_TS,
        "created_at": UTC_TS,
        "is_final": pl.Boolean(),
    },
)

PREDICTIONS_BLENDED = TableSpec(
    "blended",
    ("run_id", "game_id"),
    {
        "run_id": pl.String(),
        "game_id": pl.String(),
        "model_prob": pl.Float64(),
        "user_prob": pl.Float64(),
        "team_nudge_points": pl.Float64(),
        "opinion_weight": pl.Float64(),
        "delta_logit": pl.Float64(),
        "capped": pl.Boolean(),
        "blended_prob": pl.Float64(),
        "flipped": pl.Boolean(),
        "created_at": UTC_TS,
        "is_final": pl.Boolean(),
    },
)

MATCHUP_STATS: tuple[str, ...] = (
    "plays_per_game",
    "pass_rate",
    "pass_yards_per_play",
    "rush_yards_per_play",
    "epa_per_play",
    "success_rate",
    "turnovers_per_game",
    "points_per_game",
    "red_zone_td_rate",
    "start_yardline",
)
