import polars as pl

from football.features.columns import SITUATIONAL_FEATURES
from football.features.common import franchise_expr, haversine_km, utc_offset_hours
from football.features.elo import BYE_REST_DAYS

SHORT_WEEK_DAYS = 5
PACIFIC_TZ = "America/Los_Angeles"
EASTERN_TZ = "America/New_York"
EARLY_KICKOFF_MINUTES_ET = 13 * 60
INDOOR_ROOFS = ("dome", "retractable")


HOME_FIELDS: dict[str, pl.DataType] = {
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "tz": pl.String(),
    "roof_type": pl.String(),
}


def team_home_context(targets: pl.DataFrame, team_homes: pl.DataFrame | None) -> pl.DataFrame:
    out = targets.select("game_id", "season", "home_fr", "away_fr")
    for side in ("home", "away"):
        names = {field: f"{side}_team_{field}" for field in HOME_FIELDS}
        if team_homes is None:
            out = out.with_columns(
                pl.lit(None, dtype).alias(names[field]) for field, dtype in HOME_FIELDS.items()
            )
            continue
        homes = team_homes.select(
            pl.col("season").cast(pl.Int32),
            franchise_expr("team").alias(f"{side}_fr"),
            *(
                pl.col(field).cast(dtype).alias(names[field])
                for field, dtype in HOME_FIELDS.items()
            ),
        ).unique(["season", f"{side}_fr"], keep="last")
        out = out.join(homes, on=["season", f"{side}_fr"], how="left")
    return out.drop("season", "home_fr", "away_fr")


def _flag(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Float64)


def situational_features(targets: pl.DataFrame, team_homes: pl.DataFrame | None) -> pl.DataFrame:
    df = targets.join(team_home_context(targets, team_homes), on="game_id", how="left")
    venue_offset = utc_offset_hours(df["venue_tz"], df["kickoff_utc"])
    away_offset = utc_offset_hours(df["away_team_tz"], df["kickoff_utc"])
    et = pl.col("kickoff_utc").dt.convert_time_zone(EASTERN_TZ)
    early = et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32) <= (
        EARLY_KICKOFF_MINUTES_ET
    )
    roof = pl.col("roof_type")
    rest = {side: pl.col(f"{side}_rest").cast(pl.Float64) for side in ("home", "away")}
    df = df.with_columns(
        (venue_offset - away_offset).alias("tz_shift_away"),
    ).with_columns(
        rest["home"].alias("home_rest"),
        rest["away"].alias("away_rest"),
        (rest["home"] - rest["away"]).alias("rest_diff"),
        _flag(rest["home"] >= BYE_REST_DAYS).alias("home_bye"),
        _flag(rest["away"] >= BYE_REST_DAYS).alias("away_bye"),
        _flag(rest["home"] <= SHORT_WEEK_DAYS).alias("home_short_week"),
        _flag(rest["away"] <= SHORT_WEEK_DAYS).alias("away_short_week"),
        _flag(pl.col("div_game")).alias("div_game"),
        _flag(pl.col("neutral_site")).alias("neutral_site"),
        _flag(pl.col("game_type") != "REG").alias("is_playoff"),
        haversine_km(
            pl.col("home_team_lat"),
            pl.col("home_team_lon"),
            pl.col("venue_lat"),
            pl.col("venue_lon"),
        ).alias("travel_km_home"),
        haversine_km(
            pl.col("away_team_lat"),
            pl.col("away_team_lon"),
            pl.col("venue_lat"),
            pl.col("venue_lon"),
        ).alias("travel_km_away"),
        (
            _flag(early & (pl.col("away_team_tz") == PACIFIC_TZ))
            - _flag(early & (pl.col("home_team_tz") == PACIFIC_TZ))
        ).alias("west_coast_early"),
        pl.col("elevation_m").cast(pl.Float64),
        _flag(roof == "dome").alias("is_dome"),
        _flag(roof == "retractable").alias("is_retractable"),
        _flag(pl.col("surface").str.contains("grass")).alias("is_grass"),
        _flag(pl.col("is_international")).alias("is_international"),
    )
    return df.select("game_id", *SITUATIONAL_FEATURES)
