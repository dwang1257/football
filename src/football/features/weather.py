import polars as pl

from football.features.asof import latest_before
from football.features.columns import WEATHER_FEATURES
from football.features.situational import INDOOR_ROOFS, team_home_context

COLD_F = 40.0
WINDY_MPH = 15.0
NEUTRAL_TEMP_F = 70.0


def latest_forecast(targets: pl.DataFrame, forecasts: pl.DataFrame | None) -> pl.DataFrame:
    schema = {
        "game_id": pl.String(),
        "temp_f": pl.Float64(),
        "wind_mph": pl.Float64(),
        "precip_prob": pl.Float64(),
    }
    if forecasts is None:
        return pl.DataFrame(schema=schema)
    runs = forecasts.sort("available_at", "run_time", "source").select(
        "game_id",
        "available_at",
        *(pl.col(c).cast(pl.Float64) for c in ("temp_f", "wind_mph", "precip_prob")),
    )
    runs = runs.unique(["game_id", "available_at"], keep="last", maintain_order=True)
    return latest_before(targets.select("game_id", "cutoff"), runs, by=["game_id"]).select(*schema)


def weather_features(
    targets: pl.DataFrame,
    forecasts: pl.DataFrame | None,
    team_homes: pl.DataFrame | None,
) -> pl.DataFrame:
    fc = latest_forecast(targets, forecasts)
    homes = team_home_context(targets, team_homes)
    indoor = pl.col("roof_type").is_in(INDOOR_ROOFS)
    df = (
        targets.select("game_id", "roof_type")
        .join(fc, on="game_id", how="left")
        .join(homes, on="game_id", how="left")
    )
    precip = pl.col("precip_prob")
    precip = pl.when(precip > 1.0).then(precip / 100.0).otherwise(precip)
    df = df.with_columns(
        pl.when(indoor).then(NEUTRAL_TEMP_F).otherwise(pl.col("temp_f")).alias("fcst_temp_f"),
        pl.when(indoor).then(0.0).otherwise(pl.col("wind_mph")).alias("fcst_wind_mph"),
        pl.when(indoor).then(0.0).otherwise(precip).alias("fcst_precip_prob"),
    ).with_columns(
        (pl.col("fcst_wind_mph") >= WINDY_MPH).cast(pl.Float64).alias("wind_15plus"),
        (pl.col("fcst_temp_f") < COLD_F).cast(pl.Float64).alias("cold_game"),
    )
    for side in ("home", "away"):
        home_roof = pl.col(f"{side}_team_roof_type").str.to_lowercase()
        dome_team = home_roof.str.contains("dome") | home_roof.str.contains("retract")
        df = df.with_columns(
            (pl.col("cold_game") * dome_team.cast(pl.Float64)).alias(f"{side}_dome_team_in_cold")
        )
    return df.select("game_id", *WEATHER_FEATURES)
