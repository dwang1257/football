import io

import polars as pl

from football.ingest import lines_espn, lines_sbro, lines_wintotals
from football.ingest.base import IngestFn, IngestOptions
from football.ingest.lines_checks import CHECKS as LINE_CHECKS
from football.ingest.lines_common import (
    LIVE_MAX_AGE,
    UTC_TS,
    RawCache,
    close_available,
    load_schedule,
    match_by_date,
    match_by_ids,
    orient,
    publish,
    reject_frame,
    safe_franchise,
    tuesday_noon_before,
    wednesday_evening_before,
)
from football.validate.base import CheckFn

NFELO_URL = "https://raw.githubusercontent.com/greerreNFL/nfelomarket_data/main/Data/lines.csv"
NFLDATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/{name}"
HF_URL = "https://huggingface.co/datasets/minmaxxxer/{repo}/resolve/main/{file}"
HF_FILES: dict[int, tuple[str, str]] = {
    2025: ("nfl-closing-lines-ats-2025", "nfl_2025_closing_lines_ats.csv"),
    2026: ("nfl-odds-2026", "nfl_2026_odds.csv"),
}
NFELO_FIRST_REAL_OPEN = 2007
LINE_COLUMNS = [
    "game_id",
    "book",
    "kind",
    "captured_at",
    "available_at",
    "home_spread",
    "total",
    "home_ml",
    "away_ml",
]


def ingest_nflverse(options: IngestOptions) -> None:
    schedule = load_schedule(options.refresh)
    started = pl.col("kickoff_utc") <= pl.lit(schedule.fetched_at).cast(UTC_TS)
    games = schedule.games.filter(pl.col("season").is_in(options.seasons))
    captured = (
        pl.when(started)
        .then(close_available(pl.col("kickoff_utc")))
        .otherwise(pl.lit(schedule.fetched_at).cast(UTC_TS))
    )
    lines = games.select(
        "game_id",
        pl.lit("consensus").alias("book"),
        pl.when(started).then(pl.lit("close")).otherwise(pl.lit("snapshot")).alias("kind"),
        captured.alias("captured_at"),
        captured.alias("available_at"),
        pl.col("spread_line").alias("home_spread"),
        pl.col("total_line").alias("total"),
        pl.col("home_moneyline").alias("home_ml"),
        pl.col("away_moneyline").alias("away_ml"),
        pl.col("home_spread_odds").cast(pl.Float64),
        pl.col("away_spread_odds").cast(pl.Float64),
        pl.col("over_odds").cast(pl.Float64),
        pl.col("under_odds").cast(pl.Float64),
    )
    publish("lines_nflverse", lines, [], options.seasons)


def read_csv(payload: bytes) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(payload), infer_schema_length=None)


def parse_ts(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .str.to_datetime("%Y-%m-%d %H:%M:%S%.f%:z", strict=False, time_zone="UTC")
        .cast(UTC_TS)
    )


def split_game_id(df: pl.DataFrame, column: str = "game_id") -> pl.DataFrame:
    parts = pl.col(column).str.split("_")
    return df.with_columns(
        pl.col(column).alias("row_id"),
        parts.list.get(0).cast(pl.Int32).alias("season"),
        parts.list.get(1).cast(pl.Int32).alias("week"),
        parts.list.get(2).alias("away"),
        parts.list.get(3).alias("home"),
    ).drop(column)


def nfelo_lines(raw: pl.DataFrame, fetched_at: pl.Expr) -> tuple[pl.DataFrame, pl.DataFrame]:
    stamps = ["home_spread_last_timestamp", "ml_last_timestamp", "total_line_last_timestamp"]
    open_stamps = ["home_spread_open_timestamp", "ml_open_timestamp", "total_line_open_timestamp"]
    df = raw.with_columns(
        pl.max_horizontal([parse_ts(c) for c in stamps]).alias("last_ts"),
        pl.max_horizontal([parse_ts(c) for c in open_stamps]).alias("open_ts"),
        *[
            pl.col(c).cast(pl.Float64)
            for c in raw.columns
            if c.endswith(("_open", "_last", "_price", "_pct"))
        ],
    ).with_columns(
        (-pl.col("home_spread_last")).alias("neg_last"),
        (-pl.col("home_spread_open")).alias("neg_open"),
    )
    started = pl.col("kickoff_utc") <= fetched_at
    close_at = close_available(pl.col("kickoff_utc"))
    last = df.with_columns(
        *orient("swapped", "neg_last", "home_ml_last", "away_ml_last"),
        pl.col("total_line_last").alias("total"),
        pl.when(started).then(pl.lit("close")).otherwise(pl.lit("snapshot")).alias("kind"),
        pl.when(started)
        .then(pl.coalesce("last_ts", close_at))
        .otherwise(pl.coalesce("last_ts", fetched_at))
        .alias("captured_at"),
        pl.when(started)
        .then(pl.max_horizontal(close_at, pl.coalesce("last_ts", close_at)))
        .otherwise(pl.coalesce("last_ts", fetched_at))
        .alias("available_at"),
        pl.col("home_spread_last_source").alias("nfelo_book"),
        pl.when(pl.col("swapped"))
        .then(1 - pl.col("home_spread_tickets_pct"))
        .otherwise(pl.col("home_spread_tickets_pct"))
        .alias("home_tickets_pct"),
        pl.when(pl.col("swapped"))
        .then(1 - pl.col("home_spread_money_pct"))
        .otherwise(pl.col("home_spread_money_pct"))
        .alias("home_money_pct"),
    )
    real_open = (pl.col("season") >= NFELO_FIRST_REAL_OPEN) & pl.col(
        "home_spread_open"
    ).is_not_null()
    opened = (
        df.filter(real_open)
        .with_columns(
            *orient("swapped", "neg_open", "home_ml_open", "away_ml_open"),
            pl.col("total_line_open").alias("total"),
            pl.lit("open").alias("kind"),
            pl.coalesce("open_ts", tuesday_noon_before(pl.col("kickoff_utc"))).alias("captured_at"),
            pl.col("home_spread_open_source").alias("nfelo_book"),
        )
        .with_columns(pl.col("captured_at").alias("available_at"))
    )
    legacy = df.filter(~real_open & pl.col("home_spread_open").is_not_null())
    extras = ["nfelo_book"]
    lines = pl.concat(
        [
            last.select(
                *LINE_COLUMNS[:1], *LINE_COLUMNS[2:], *extras, "home_tickets_pct", "home_money_pct"
            ),
            opened.select(*LINE_COLUMNS[:1], *LINE_COLUMNS[2:], *extras),
        ],
        how="diagonal",
    ).with_columns(pl.lit("consensus").alias("book"))
    return lines, legacy


def ingest_nfelo(options: IngestOptions) -> None:
    source = "lines_nfelo"
    schedule = load_schedule(options.refresh)
    entry = RawCache(source).fetch(
        "lines.csv", NFELO_URL, refresh=options.refresh, max_age=LIVE_MAX_AGE
    )
    raw = split_game_id(read_csv(entry.read_bytes()).drop("", "season", "week", strict=False))
    result = match_by_ids(raw.filter(pl.col("season").is_in(options.seasons)), schedule.games)
    lines, legacy = nfelo_lines(result.matched, pl.lit(entry.fetched_at).cast(UTC_TS))
    rejects = [
        reject_frame(
            source,
            "unmatched",
            result.unmatched["season"].to_list(),
            result.unmatched["row_id"].to_list(),
        ),
        reject_frame(
            source,
            "open_equals_close_before_2007",
            legacy["season"].to_list(),
            legacy["row_id"].to_list(),
        ),
    ]
    publish(source, lines, rejects, options.seasons)


def same_franchise(left: str, right: str) -> pl.Expr:
    def code(name: str) -> pl.Expr:
        return pl.col(name).map_elements(safe_franchise, return_dtype=pl.String)

    return code(left) == code(right)


def nfldata_closing(raw: pl.DataFrame) -> pl.DataFrame:
    df = split_game_id(raw.drop("game_id").rename({"alt_game_id": "game_id"})).with_columns(
        same_franchise("side", "home").alias("home_side")
    )
    home_side = pl.col("home_side")
    return df.group_by("row_id", "season", "week", "away", "home").agg(
        (-pl.col("line").filter((pl.col("type") == "SPREAD") & home_side))
        .first()
        .alias("spread_raw"),
        pl.col("line")
        .filter((pl.col("type") == "TOTAL") & (pl.col("side") == "Over"))
        .first()
        .alias("total"),
        pl.col("odds")
        .filter((pl.col("type") == "MONEYLINE") & home_side)
        .first()
        .cast(pl.Float64)
        .alias("ml_h"),
        pl.col("odds")
        .filter((pl.col("type") == "MONEYLINE") & ~home_side)
        .first()
        .cast(pl.Float64)
        .alias("ml_a"),
    )


def nfldata_supercontest(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw.rename({"away_team": "away", "home_team": "home"})
        .with_columns(
            pl.concat_str(["season", "week", "away", "home"], separator="_").alias("row_id")
        )
        .group_by("row_id", "season", "week", "away", "home")
        .agg((-pl.col("line").filter(same_franchise("side", "home"))).first().alias("spread_raw"))
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    )


def ingest_nfldata(options: IngestOptions) -> None:
    source = "lines_nfldata"
    schedule = load_schedule(options.refresh)
    cache = RawCache(source)
    closing_raw = read_csv(
        cache.fetch(
            "closing_lines.csv",
            NFLDATA_URL.format(name="closing_lines.csv"),
            refresh=options.refresh,
        ).read_bytes()
    )
    sc_raw = read_csv(
        cache.fetch(
            "sc_lines.csv", NFLDATA_URL.format(name="sc_lines.csv"), refresh=options.refresh
        ).read_bytes()
    )
    closing = match_by_ids(
        nfldata_closing(closing_raw).filter(pl.col("season").is_in(options.seasons)), schedule.games
    )
    contest = match_by_ids(
        nfldata_supercontest(sc_raw).filter(pl.col("season").is_in(options.seasons)), schedule.games
    )
    close_lines = closing.matched.with_columns(
        *orient("swapped", "spread_raw", "ml_h", "ml_a"),
        pl.lit("consensus").alias("book"),
        pl.lit("close").alias("kind"),
        close_available(pl.col("kickoff_utc")).alias("captured_at"),
    )
    week_start = schedule.games.group_by("season", "week").agg(
        pl.col("kickoff_utc").min().alias("week_kickoff")
    )
    sc_lines = contest.matched.join(week_start, on=["season", "week"], how="left").with_columns(
        pl.when(pl.col("swapped"))
        .then(-pl.col("spread_raw"))
        .otherwise(pl.col("spread_raw"))
        .alias("home_spread"),
        pl.lit(None, dtype=pl.Float64).alias("total"),
        pl.lit(None, dtype=pl.Float64).alias("home_ml"),
        pl.lit(None, dtype=pl.Float64).alias("away_ml"),
        pl.lit("supercontest").alias("book"),
        pl.lit("wednesday").alias("kind"),
        wednesday_evening_before(pl.col("week_kickoff")).alias("captured_at"),
    )
    columns = [c for c in LINE_COLUMNS if c != "available_at"]
    lines = pl.concat([close_lines.select(columns), sc_lines.select(columns)]).with_columns(
        pl.col("captured_at").alias("available_at")
    )
    unmatched = pl.concat(
        [closing.unmatched.select("season", "row_id"), contest.unmatched.select("season", "row_id")]
    )
    rejects = [
        reject_frame(
            source, "unmatched", unmatched["season"].to_list(), unmatched["row_id"].to_list()
        )
    ]
    publish(source, lines, rejects, options.seasons)


def hf_lines(matched: pl.DataFrame) -> pl.DataFrame:
    flip = pl.col("swapped")

    def home_spread(column: str) -> pl.Expr:
        value = -pl.col(column).cast(pl.Float64)
        return pl.when(flip).then(-value).otherwise(value).alias("home_spread")

    def home_prob(column: str) -> pl.Expr:
        value = pl.col(column).cast(pl.Float64)
        return pl.when(flip).then(1 - value).otherwise(value).alias("home_prob_novig")

    nulls = [
        pl.lit(None, dtype=pl.Float64).alias("home_ml"),
        pl.lit(None, dtype=pl.Float64).alias("away_ml"),
    ]
    close = matched.with_columns(
        home_spread("close_spread_home"),
        pl.col("close_total").cast(pl.Float64).alias("total"),
        home_prob("close_home_win_prob"),
        *nulls,
        pl.lit("close").alias("kind"),
        close_available(pl.col("kickoff_utc")).alias("captured_at"),
    )
    opened = matched.with_columns(
        home_spread("open_spread_home"),
        pl.col("open_total").cast(pl.Float64).alias("total"),
        home_prob("open_home_win_prob"),
        *nulls,
        pl.lit("open").alias("kind"),
        tuesday_noon_before(pl.col("kickoff_utc")).alias("captured_at"),
    )
    columns = [c for c in LINE_COLUMNS if c not in ("available_at", "book")] + [
        "home_prob_novig",
        "bookmakers",
    ]
    return pl.concat([close.select(columns), opened.select(columns)]).with_columns(
        pl.lit("median").alias("book"),
        pl.col("captured_at").alias("available_at"),
        pl.col("bookmakers").cast(pl.Int32),
    )


def ingest_hf(options: IngestOptions) -> None:
    source = "lines_hf"
    seasons = [s for s in options.seasons if s in HF_FILES]
    if not seasons:
        return
    schedule = load_schedule(options.refresh)
    cache = RawCache(source)
    frames: list[pl.DataFrame] = []
    for season in seasons:
        repo, file = HF_FILES[season]
        live = season == max(HF_FILES)
        entry = cache.fetch(
            file,
            HF_URL.format(repo=repo, file=file),
            refresh=options.refresh,
            max_age=LIVE_MAX_AGE if live else None,
        )
        frames.append(read_csv(entry.read_bytes()).with_columns(pl.col("season").cast(pl.Int32)))
    raw = pl.concat(frames, how="diagonal_relaxed").with_columns(
        pl.col("game_id").alias("row_id"),
        pl.col("date_et").str.to_date("%Y-%m-%d").alias("game_date"),
        pl.col("home_team").alias("team_a"),
        pl.col("away_team").alias("team_b"),
    )
    result = match_by_date(raw.drop("game_id", "home_team", "away_team"), schedule.games)
    rejects = [
        reject_frame(
            source,
            "unmatched",
            result.unmatched["season"].to_list(),
            result.unmatched["row_id"].to_list(),
        )
    ]
    publish(source, hf_lines(result.matched), rejects, seasons)


SOURCES: dict[str, IngestFn] = {
    "lines_nflverse": ingest_nflverse,
    "lines_nfelo": ingest_nfelo,
    "lines_sbro": lines_sbro.ingest,
    "lines_nfldata": ingest_nfldata,
    "lines_hf": ingest_hf,
    "lines_espn": lines_espn.ingest,
    "win_totals": lines_wintotals.ingest,
}
CHECKS: dict[str, CheckFn] = dict(LINE_CHECKS)
