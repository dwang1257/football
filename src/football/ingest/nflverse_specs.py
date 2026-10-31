import polars as pl

from football.schemas import UTC_TS, TableSpec

SNAP_COUNTS = TableSpec(
    "snap_counts",
    ("game_id", "team", "player_id"),
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "game_type": pl.String(),
        "team": pl.String(),
        "player_id": pl.String(),
        "player_id_type": pl.String(),
        "pfr_player_id": pl.String(),
        "full_name": pl.String(),
        "position": pl.String(),
        "offense_snaps": pl.Float64(),
        "offense_pct": pl.Float64(),
        "defense_snaps": pl.Float64(),
        "defense_pct": pl.Float64(),
        "st_snaps": pl.Float64(),
        "st_pct": pl.Float64(),
        "available_at": UTC_TS,
    },
)

ROSTERS_WEEKLY = TableSpec(
    "rosters_weekly",
    ("season", "week", "team", "player_id"),
    {
        "season": pl.Int32(),
        "week": pl.Int32(),
        "game_type": pl.String(),
        "team": pl.String(),
        "game_id": pl.String(),
        "player_id": pl.String(),
        "player_id_type": pl.String(),
        "full_name": pl.String(),
        "position": pl.String(),
        "depth_chart_position": pl.String(),
        "status": pl.String(),
        "status_description_abbr": pl.String(),
        "years_exp": pl.Int32(),
        "rookie_year": pl.Int32(),
        "draft_number": pl.Int32(),
        "birth_date": pl.Date(),
        "available_at": UTC_TS,
    },
)

QBR = TableSpec(
    "qbr",
    ("game_id", "team", "espn_player_id"),
    {
        "game_id": pl.String(),
        "season": pl.Int32(),
        "week": pl.Int32(),
        "season_type": pl.String(),
        "team": pl.String(),
        "espn_game_id": pl.String(),
        "espn_player_id": pl.String(),
        "full_name": pl.String(),
        "qbr_total": pl.Float64(),
        "qbr_raw": pl.Float64(),
        "pts_added": pl.Float64(),
        "epa_total": pl.Float64(),
        "qb_plays": pl.Int32(),
        "qualified": pl.Boolean(),
        "available_at": UTC_TS,
    },
)

CONTRACTS = TableSpec(
    "contracts",
    ("contract_id",),
    {
        "contract_id": pl.String(),
        "season": pl.Int32(),
        "otc_id": pl.Int32(),
        "gsis_id": pl.String(),
        "full_name": pl.String(),
        "position": pl.String(),
        "team": pl.String(),
        "team_raw": pl.String(),
        "years": pl.Int32(),
        "value": pl.Float64(),
        "apy": pl.Float64(),
        "guaranteed": pl.Float64(),
        "apy_cap_pct": pl.Float64(),
        "available_at": UTC_TS,
    },
)

DRAFT_PICKS = TableSpec(
    "draft_picks",
    ("season", "pick"),
    {
        "season": pl.Int32(),
        "round": pl.Int32(),
        "pick": pl.Int32(),
        "team": pl.String(),
        "gsis_id": pl.String(),
        "pfr_player_id": pl.String(),
        "full_name": pl.String(),
        "position": pl.String(),
        "college": pl.String(),
        "age": pl.Int32(),
        "available_at": UTC_TS,
    },
)

TRADES = TableSpec(
    "trades",
    ("trade_id", "asset_index"),
    {
        "trade_id": pl.Int32(),
        "asset_index": pl.Int32(),
        "season": pl.Int32(),
        "trade_date": pl.Date(),
        "gave": pl.String(),
        "received": pl.String(),
        "pfr_id": pl.String(),
        "pfr_name": pl.String(),
        "pick_season": pl.Int32(),
        "pick_round": pl.Int32(),
        "pick_number": pl.Int32(),
        "conditional": pl.Boolean(),
        "available_at": UTC_TS,
    },
)
