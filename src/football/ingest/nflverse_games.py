import polars as pl

from football import storage
from football.ingest.base import IngestOptions
from football.ingest.nflverse_common import eastern_to_utc, hours, log, replace_seasons
from football.ingest.nflverse_fetch import fetch_frame
from football.schemas import GAMES

SCHEDULES_ASSET = "schedules/games.parquet"
RESULT_DELAY = hours(4)
DEFAULT_GAMETIME = "13:00"
ROOF_VALUES = ("dome", "outdoors", "closed", "open")


def clean_schedules(raw: pl.DataFrame) -> pl.DataFrame:
    local = pl.concat_str(
        pl.col("gameday"), pl.lit(" "), pl.col("gametime").fill_null(DEFAULT_GAMETIME)
    ).str.to_datetime("%Y-%m-%d %H:%M", strict=True)
    kickoff = eastern_to_utc(local)
    return raw.select(
        pl.col("game_id"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_type"),
        kickoff.alias("kickoff_utc"),
        pl.col("home_team"),
        pl.col("away_team"),
        pl.col("home_score").cast(pl.Int32),
        pl.col("away_score").cast(pl.Int32),
        (pl.col("location") == "Neutral").alias("neutral_site"),
        pl.col("stadium_id"),
        pl.when(pl.col("roof").is_in(ROOF_VALUES)).then(pl.col("roof")).alias("roof"),
        pl.col("surface").str.strip_chars().replace("", None),
        (pl.col("div_game") == 1).alias("div_game"),
        pl.col("home_rest").cast(pl.Int32),
        pl.col("away_rest").cast(pl.Int32),
        pl.col("home_coach"),
        pl.col("away_coach"),
        pl.col("referee"),
        pl.col("home_qb_id"),
        pl.col("away_qb_id"),
        pl.col("spread_line").cast(pl.Float64),
        pl.col("total_line").cast(pl.Float64),
        pl.col("home_moneyline").cast(pl.Float64),
        pl.col("away_moneyline").cast(pl.Float64),
        pl.col("temp").cast(pl.Float64).alias("temp_observed"),
        pl.col("wind").cast(pl.Float64).alias("wind_observed"),
        (kickoff + RESULT_DELAY).alias("available_at"),
        pl.col("gameday"),
        pl.col("gametime"),
        pl.col("weekday"),
        pl.col("location"),
        (pl.col("overtime") == 1).alias("overtime"),
        pl.col("old_game_id"),
        pl.col("espn").alias("espn_id"),
        pl.col("pfr").alias("pfr_id"),
        pl.col("stadium").alias("stadium_name"),
        pl.col("home_qb_name"),
        pl.col("away_qb_name"),
        pl.col("home_spread_odds").cast(pl.Float64),
        pl.col("away_spread_odds").cast(pl.Float64),
        pl.col("over_odds").cast(pl.Float64),
        pl.col("under_odds").cast(pl.Float64),
    )


def ingest_schedules(options: IngestOptions) -> None:
    raw = fetch_frame(SCHEDULES_ASSET, options.refresh)
    games = clean_schedules(raw.filter(pl.col("season").is_in(options.seasons)))
    present = games["season"].unique().to_list()
    missing = sorted(set(options.seasons) - set(present))
    if missing:
        log(f"schedules: no games published for seasons {missing}")
    replace_seasons(games, GAMES, present)


def load_games(options: IngestOptions) -> pl.DataFrame:
    if not storage.exists("clean", GAMES.name):
        ingest_schedules(options)
    return storage.read("clean", GAMES.name)


def completed(games: pl.DataFrame) -> pl.DataFrame:
    return games.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())


def team_schedule(games: pl.DataFrame) -> pl.DataFrame:
    columns = ["game_id", "season", "week", "game_type", "kickoff_utc"]
    home = games.select(
        *columns,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("home_qb_id").alias("starter_qb_id"),
    )
    away = games.select(
        *columns,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        pl.col("away_qb_id").alias("starter_qb_id"),
    )
    return pl.concat([home, away])
