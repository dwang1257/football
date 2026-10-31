import polars as pl

from football.ingest.base import IngestOptions
from football.ingest.nflverse_common import (
    eastern_wall_clock,
    hours,
    log,
    normalize_team,
    replace_seasons,
)
from football.ingest.nflverse_fetch import fetch_frame
from football.ingest.nflverse_games import load_games
from football.ingest.nflverse_specs import CONTRACTS, DRAFT_PICKS, QBR, TRADES
from football.schemas import UTC_TS

QBR_ASSET = "espn_data/qbr_week_level.csv"
QBR_DELAY = hours(48)
DRAFT_KNOWN_MONTH_DAY = (5, 15)


def clean_qbr(raw: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    frame = raw.with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("game_id").cast(pl.String).alias("espn_game_id"),
        pl.col("player_id").cast(pl.String).alias("espn_player_id"),
        pl.col("team_abb").alias("team"),
    ).drop("game_id")
    frame = normalize_team(frame, "team").join(
        games.select(
            "game_id",
            pl.col("espn_id").alias("espn_game_id"),
            "week",
            "kickoff_utc",
            "home_team",
            "away_team",
        ),
        on="espn_game_id",
        how="left",
    )
    in_game = (pl.col("team") == pl.col("home_team")) | (pl.col("team") == pl.col("away_team"))
    unmapped = frame.filter(pl.col("game_id").is_null() | ~in_game.fill_null(False)).height
    if unmapped:
        log(f"qbr: dropped {unmapped} rows whose ESPN game id or team does not match a game")
    return (
        frame.filter(pl.col("game_id").is_not_null() & in_game.fill_null(False))
        .select(
            "game_id",
            "season",
            pl.col("week").cast(pl.Int32),
            "season_type",
            "team",
            "espn_game_id",
            "espn_player_id",
            pl.col("name_display").alias("full_name"),
            *(
                pl.col(name).cast(pl.Float64)
                for name in ("qbr_total", "qbr_raw", "pts_added", "epa_total")
            ),
            pl.col("qb_plays").cast(pl.Int32),
            pl.col("qualified").cast(pl.Boolean),
            (pl.col("kickoff_utc") + QBR_DELAY).alias("available_at"),
        )
        .unique(["game_id", "team", "espn_player_id"], keep="last", maintain_order=True)
    )


def ingest_qbr(options: IngestOptions) -> None:
    games = load_games(options)
    raw = fetch_frame(QBR_ASSET, options.refresh)
    frame = clean_qbr(raw.filter(pl.col("season").is_in(options.seasons)), games)
    replace_seasons(frame, QBR, frame["season"].unique().to_list())


def clean_contracts(raw: pl.DataFrame) -> pl.DataFrame:
    frame = raw.filter(pl.col("year_signed") > 0).with_columns(
        pl.col("year_signed").cast(pl.Int32).alias("season"),
        pl.col("team").alias("team_raw"),
        pl.col("team").str.split("/").list.first().alias("team"),
    )
    frame = normalize_team(frame, "team")
    contract_id = pl.concat_str(
        pl.col("otc_id").cast(pl.String),
        pl.col("season").cast(pl.String),
        pl.col("team_raw"),
        pl.col("years").cast(pl.String),
        pl.col("value").cast(pl.String),
        pl.col("guaranteed").cast(pl.String),
        separator=":",
        ignore_nulls=True,
    )
    known = eastern_wall_clock(pl.date(pl.col("season") + 1, 1, 1), 0)
    return frame.select(
        contract_id.alias("contract_id"),
        "season",
        pl.col("otc_id").cast(pl.Int32),
        "gsis_id",
        pl.col("player").alias("full_name"),
        "position",
        "team",
        "team_raw",
        pl.col("years").cast(pl.Int32),
        *(pl.col(c).cast(pl.Float64) for c in ("value", "apy", "guaranteed", "apy_cap_pct")),
        known.alias("available_at"),
    ).unique("contract_id", keep="first", maintain_order=True)


def ingest_contracts(options: IngestOptions) -> None:
    raw = fetch_frame("contracts/historical_contracts.parquet", options.refresh)
    frame = clean_contracts(raw).filter(pl.col("season").is_in(options.seasons))
    replace_seasons(frame, CONTRACTS, frame["season"].unique().to_list())


def clean_draft_picks(raw: pl.DataFrame) -> pl.DataFrame:
    month, day = DRAFT_KNOWN_MONTH_DAY
    frame = normalize_team(raw.with_columns(pl.col("season").cast(pl.Int32)), "team")
    return frame.select(
        "season",
        pl.col("round").cast(pl.Int32),
        pl.col("pick").cast(pl.Int32),
        "team",
        "gsis_id",
        "pfr_player_id",
        pl.col("pfr_player_name").alias("full_name"),
        "position",
        "college",
        pl.col("age").cast(pl.Int32),
        eastern_wall_clock(pl.date(pl.col("season"), month, day), 0).alias("available_at"),
    ).unique(["season", "pick"], keep="first", maintain_order=True)


def ingest_draft_picks(options: IngestOptions) -> None:
    raw = fetch_frame("draft_picks/draft_picks.parquet", options.refresh)
    frame = clean_draft_picks(raw.filter(pl.col("season").is_in(options.seasons)))
    replace_seasons(frame, DRAFT_PICKS, frame["season"].unique().to_list())


def clean_trades(raw: pl.DataFrame) -> pl.DataFrame:
    frame = raw.with_columns(pl.col("season").cast(pl.Int32))
    frame = normalize_team(normalize_team(frame, "gave"), "received")
    return frame.select(
        pl.col("trade_id").cast(pl.Int32),
        (pl.int_range(pl.len()).over("trade_id") + 1).cast(pl.Int32).alias("asset_index"),
        "season",
        pl.col("trade_date").cast(pl.Date),
        "gave",
        "received",
        "pfr_id",
        "pfr_name",
        pl.col("pick_season").cast(pl.Int32),
        pl.col("pick_round").cast(pl.Int32),
        pl.col("pick_number").cast(pl.Int32),
        (pl.col("conditional") == 1).alias("conditional"),
        eastern_wall_clock(pl.col("trade_date") + pl.duration(days=1), 0)
        .cast(UTC_TS)
        .alias("available_at"),
    )


def ingest_trades(options: IngestOptions) -> None:
    raw = fetch_frame("trades/trades.parquet", options.refresh)
    frame = clean_trades(raw.filter(pl.col("season").is_in(options.seasons)))
    replace_seasons(frame, TRADES, frame["season"].unique().to_list())
