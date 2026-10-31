import polars as pl

from football.ingest.base import IngestOptions
from football.ingest.nflverse_common import hours, log, replace_seasons
from football.ingest.nflverse_fetch import NotPublished, fetch
from football.ingest.nflverse_games import completed, load_games, team_schedule
from football.schemas import QB_GAMES, TEAM_GAMES

PBP_DELAY = hours(12)
FIRST_PBP_SEASON = 1999
PBP_COLUMNS = (
    "game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "pass",
    "rush",
    "sack",
    "qb_scramble",
    "qb_dropback",
    "qb_kneel",
    "two_point_attempt",
    "epa",
    "qb_epa",
    "cpoe",
    "success",
    "wp",
    "yards_gained",
    "yardline_100",
    "interception",
    "fumble_lost",
    "fumbled_1_team",
    "fixed_drive",
    "fixed_drive_result",
    "passer_id",
    "passer",
    "total_home_score",
    "total_away_score",
)


def pbp_asset(season: int) -> str:
    return f"pbp/play_by_play_{season}.parquet"


def read_pbp(season: int, refresh: bool) -> pl.DataFrame | None:
    try:
        path = fetch(pbp_asset(season), refresh)
    except NotPublished:
        return None
    available = pl.read_parquet_schema(path)
    wanted = [c for c in PBP_COLUMNS if c in available]
    return pl.read_parquet(path, columns=wanted)


def _side(team_col: str) -> pl.Expr:
    return (
        pl.when(pl.col(team_col) == pl.col("home_team"))
        .then(pl.lit("home"))
        .when(pl.col(team_col) == pl.col("away_team"))
        .then(pl.lit("away"))
    )


def _flag(name: str) -> pl.Expr:
    return pl.col(name).fill_null(0) == 1


def scrimmage_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    return pbp.filter(
        (_flag("pass") | _flag("rush")) & pl.col("epa").is_not_null() & ~_flag("two_point_attempt")
    )


def kneel_yards(pbp: pl.DataFrame) -> pl.DataFrame:
    return (
        pbp.filter(_flag("qb_kneel") & ~_flag("two_point_attempt"))
        .with_columns(_side("posteam").alias("side"))
        .group_by("game_id", "side")
        .agg(pl.col("yards_gained").sum().cast(pl.Float64).alias("kneel_yards"))
    )


def offense_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    plays = scrimmage_plays(pbp).with_columns(
        _side("posteam").alias("side"),
        _flag("pass").alias("is_pass"),
    )
    plays = plays.filter(pl.col("side").is_not_null())
    in_wp = pl.col("wp").is_between(0.1, 0.9)
    per_game = plays.group_by("game_id", "side").agg(
        pl.len().cast(pl.Int32).alias("plays"),
        pl.col("is_pass").sum().cast(pl.Int32).alias("pass_plays"),
        (~pl.col("is_pass")).sum().cast(pl.Int32).alias("rush_plays"),
        pl.col("yards_gained").filter(pl.col("is_pass")).sum().cast(pl.Float64).alias("pass_yards"),
        pl.col("yards_gained")
        .filter(~pl.col("is_pass"))
        .sum()
        .cast(pl.Float64)
        .alias("rush_yards"),
        pl.col("epa").filter(pl.col("is_pass")).sum().alias("pass_epa"),
        pl.col("epa").filter(~pl.col("is_pass")).sum().alias("rush_epa"),
        pl.col("epa").sum().alias("total_epa"),
        pl.col("success").fill_null(0).sum().cast(pl.Int32).alias("success_plays"),
        in_wp.sum().cast(pl.Int32).alias("wp_filtered_plays"),
        pl.col("epa").filter(in_wp).sum().alias("wp_filtered_epa"),
        pl.col("yards_gained")
        .filter(_flag("qb_scramble"))
        .sum()
        .cast(pl.Float64)
        .alias("scramble_yards"),
        pl.col("yards_gained").filter(_flag("sack")).sum().cast(pl.Float64).alias("sack_yards"),
        _flag("sack").sum().cast(pl.Int32).alias("sacks"),
    )
    drives = (
        plays.filter(pl.col("fixed_drive").is_not_null())
        .sort("play_id")
        .group_by("game_id", "side", "fixed_drive")
        .agg(
            pl.col("yardline_100").first().alias("start_yardline_100"),
            (pl.col("yardline_100") <= 20).any().alias("red_zone"),
            (pl.col("fixed_drive_result").first() == "Touchdown").alias("touchdown"),
        )
        .group_by("game_id", "side")
        .agg(
            pl.len().cast(pl.Int32).alias("drives"),
            pl.col("red_zone").sum().cast(pl.Int32).alias("red_zone_trips"),
            (pl.col("red_zone") & pl.col("touchdown")).sum().cast(pl.Int32).alias("red_zone_tds"),
            (100 - pl.col("start_yardline_100")).mean().alias("avg_start_yardline"),
        )
    )
    return per_game.join(drives, on=["game_id", "side"], how="left").join(
        kneel_yards(pbp), on=["game_id", "side"], how="left"
    )


def turnover_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    interceptions = (
        pbp.filter(_flag("interception"))
        .with_columns(_side("posteam").alias("side"))
        .group_by("game_id", "side")
        .agg(pl.len().cast(pl.Int32).alias("interceptions_thrown"))
    )
    fumbles = (
        pbp.filter(_flag("fumble_lost"))
        .with_columns(_side("fumbled_1_team").alias("side"))
        .group_by("game_id", "side")
        .agg(pl.len().cast(pl.Int32).alias("fumbles_lost"))
    )
    return interceptions.join(fumbles, on=["game_id", "side"], how="full", coalesce=True).filter(
        pl.col("side").is_not_null()
    )


def final_scores(pbp: pl.DataFrame) -> pl.DataFrame:
    finals = pbp.group_by("game_id").agg(
        pl.col("total_home_score").max().cast(pl.Int32).alias("home"),
        pl.col("total_away_score").max().cast(pl.Int32).alias("away"),
    )
    return finals.unpivot(index="game_id", variable_name="side", value_name="pbp_points")


def build_team_games(pbp: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    played = completed(games).filter(pl.col("game_id").is_in(pbp["game_id"].unique().implode()))
    schedule = team_schedule(played).with_columns(
        pl.when(pl.col("is_home")).then(pl.lit("home")).otherwise(pl.lit("away")).alias("side")
    )
    points = played.select(
        "game_id", pl.col("home_score").alias("home"), pl.col("away_score").alias("away")
    ).unpivot(index="game_id", variable_name="side", value_name="points_for")
    turnovers = turnover_stats(pbp)
    zero_fill = ["interceptions_thrown", "fumbles_lost"]
    frame = (
        schedule.join(points, on=["game_id", "side"], how="left")
        .join(offense_stats(pbp), on=["game_id", "side"], how="left")
        .join(turnovers, on=["game_id", "side"], how="left")
        .join(final_scores(pbp), on=["game_id", "side"], how="left")
        .with_columns(pl.col(zero_fill).fill_null(0))
        .with_columns((pl.col("interceptions_thrown") + pl.col("fumbles_lost")).alias("giveaways"))
    )
    opponent = frame.select(
        "game_id",
        pl.col("opponent").alias("team"),
        pl.col("points_for").alias("points_against"),
        pl.col("giveaways").alias("takeaways"),
    )
    return frame.join(opponent, on=["game_id", "team"], how="left").select(
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "team",
        "opponent",
        "is_home",
        pl.col("points_for").cast(pl.Int32),
        pl.col("points_against").cast(pl.Int32),
        *(pl.col(c).fill_null(0).cast(pl.Int32) for c in ("plays", "pass_plays", "rush_plays")),
        *(pl.col(c).fill_null(0.0) for c in ("pass_yards", "rush_yards")),
        (pl.col("pass_yards").fill_null(0.0) + pl.col("rush_yards").fill_null(0.0)).alias(
            "total_yards"
        ),
        *(pl.col(c).fill_null(0.0) for c in ("pass_epa", "rush_epa", "total_epa")),
        pl.col("success_plays").fill_null(0).cast(pl.Int32),
        pl.col("wp_filtered_plays").fill_null(0).cast(pl.Int32),
        pl.col("wp_filtered_epa").fill_null(0.0),
        pl.col("takeaways").cast(pl.Int32),
        pl.col("giveaways").cast(pl.Int32),
        pl.col("fumbles_lost").cast(pl.Int32),
        pl.col("interceptions_thrown").cast(pl.Int32),
        pl.col("drives").fill_null(0).cast(pl.Int32),
        pl.col("red_zone_trips").fill_null(0).cast(pl.Int32),
        pl.col("red_zone_tds").fill_null(0).cast(pl.Int32),
        pl.col("avg_start_yardline"),
        (pl.col("kickoff_utc") + PBP_DELAY).alias("available_at"),
        "game_type",
        pl.col("scramble_yards").fill_null(0.0),
        pl.col("sack_yards").fill_null(0.0),
        pl.col("sacks").fill_null(0).cast(pl.Int32),
        pl.col("kneel_yards").fill_null(0.0),
        "pbp_points",
    )


def build_qb_games(pbp: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    played = completed(games)
    schedule = team_schedule(played).with_columns(
        pl.when(pl.col("is_home")).then(pl.lit("home")).otherwise(pl.lit("away")).alias("side")
    )
    dropbacks = pbp.filter(
        _flag("qb_dropback")
        & ~_flag("two_point_attempt")
        & pl.col("passer_id").is_not_null()
        & pl.col("epa").is_not_null()
    ).with_columns(_side("posteam").alias("side"))
    per_qb = dropbacks.group_by("game_id", "side", "passer_id").agg(
        pl.col("passer").drop_nulls().mode().sort().first().alias("full_name"),
        pl.len().cast(pl.Int32).alias("dropbacks"),
        pl.col("qb_epa").fill_null(pl.col("epa")).sum().alias("qb_epa"),
        pl.col("cpoe").mean().alias("cpoe"),
    )
    return per_qb.join(schedule, on=["game_id", "side"], how="inner").select(
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "team",
        pl.col("passer_id").alias("player_id"),
        "full_name",
        (pl.col("passer_id") == pl.col("starter_qb_id")).fill_null(False).alias("is_starter"),
        "dropbacks",
        "qb_epa",
        "cpoe",
        (pl.col("kickoff_utc") + PBP_DELAY).alias("available_at"),
        "game_type",
    )


def ingest_pbp(options: IngestOptions) -> None:
    games = load_games(options)
    for season in options.seasons:
        if season < FIRST_PBP_SEASON:
            continue
        pbp = read_pbp(season, options.refresh)
        if pbp is None:
            log(f"pbp: season {season} not published, skipped")
            continue
        season_games = games.filter(pl.col("season") == season)
        team_games = build_team_games(pbp, season_games)
        qb_games = build_qb_games(pbp, season_games)
        replace_seasons(team_games, TEAM_GAMES, [season])
        replace_seasons(qb_games, QB_GAMES, [season])
        del pbp
