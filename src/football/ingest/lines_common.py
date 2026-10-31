import io
import math
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import nflreadpy
import polars as pl
from rich.console import Console

from football import storage
from football.config import get_settings
from football.schemas import LINES, UTC_TS, TableSpec, validate
from football.teams import franchise, game_id, team_code

ET = "America/New_York"
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
LIVE_MAX_AGE = timedelta(hours=6)
MIN_CONFLICT_SPREAD = 3.0
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_STAMP = re.compile(r"^(\d{8}T\d{6}Z)_(.+)$")
_last_request: dict[str, float] = {}
console = Console()

REJECTS = TableSpec(
    "lines_rejects",
    ("source", "reason", "ref"),
    {
        "source": pl.String(),
        "season": pl.Int32(),
        "reason": pl.String(),
        "ref": pl.String(),
        "recorded_at": UTC_TS,
    },
)


class RetryableStatus(Exception):
    pass


@cache
def client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=get_settings().http_timeout_s)


def http_get(
    url: str, *, browser: bool = False, min_interval: float = 0.3, attempts: int = 4
) -> bytes:
    host = urlsplit(url).hostname or ""
    headers = {"User-Agent": BROWSER_UA if browser else get_settings().user_agent}
    for attempt in range(attempts):
        wait = _last_request.get(host, 0.0) + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[host] = time.monotonic()
        try:
            response = client().get(url, headers=headers)
            if response.status_code in RETRY_STATUS:
                raise RetryableStatus(f"{response.status_code} for {url}")
            response.raise_for_status()
            return response.content
        except (httpx.TransportError, RetryableStatus):
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * 2**attempt)
    raise RuntimeError(f"unreachable: {url}")


@dataclass(frozen=True)
class Cached:
    path: Path
    fetched_at: datetime

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()


class RawCache:
    def __init__(self, source: str) -> None:
        self.source = source
        self.directory = storage.layer_dir("raw") / source
        self.index: dict[str, list[Cached]] = {}
        if self.directory.exists():
            for path in self.directory.iterdir():
                match = _STAMP.match(path.name)
                if not match:
                    continue
                stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                self.index.setdefault(match.group(2), []).append(Cached(path, stamp))
        for versions in self.index.values():
            versions.sort(key=lambda item: item.fetched_at)

    def versions(self, name: str) -> list[Cached]:
        return self.index.get(name, [])

    def latest(self, name: str) -> Cached | None:
        versions = self.versions(name)
        return versions[-1] if versions else None

    def store(self, name: str, payload: bytes) -> Cached:
        path = storage.write_raw(self.source, name, payload)
        match = _STAMP.match(path.name)
        assert match is not None
        stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        entry = Cached(path, stamp)
        self.index.setdefault(name, []).append(entry)
        return entry

    def is_fresh(self, name: str, refresh: bool, max_age: timedelta | None) -> Cached | None:
        entry = self.latest(name)
        if entry is None or refresh:
            return None
        if max_age is not None and datetime.now(UTC) - entry.fetched_at > max_age:
            return None
        return entry

    def fetch(
        self,
        name: str,
        url: str,
        *,
        refresh: bool = False,
        max_age: timedelta | None = None,
        browser: bool = False,
        min_interval: float = 0.3,
    ) -> Cached:
        entry = self.is_fresh(name, refresh, max_age)
        if entry is not None:
            return entry
        return self.store(name, http_get(url, browser=browser, min_interval=min_interval))


def kickoff_expr() -> pl.Expr:
    default_time = (
        pl.when(pl.col("weekday") == "Sunday")
        .then(pl.lit("13:00"))
        .when(pl.col("weekday") == "Monday")
        .then(pl.lit("21:00"))
        .otherwise(pl.lit("12:30"))
    )
    local = pl.concat_str(
        [pl.col("gameday"), pl.lit(" "), pl.coalesce(pl.col("gametime"), default_time)]
    ).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M")
    return local.dt.replace_time_zone(ET).dt.convert_time_zone("UTC").alias("kickoff_utc")


@dataclass(frozen=True)
class Schedule:
    games: pl.DataFrame
    fetched_at: datetime


def load_schedule(refresh: bool = False, max_age: timedelta | None = LIVE_MAX_AGE) -> Schedule:
    cache = RawCache("nflverse_schedules")
    entry = cache.is_fresh("schedules.parquet", refresh, max_age)
    if entry is None:
        buffer = io.BytesIO()
        nflreadpy.load_schedules().write_parquet(buffer)
        entry = cache.store("schedules.parquet", buffer.getvalue())
    return Schedule(prepare_schedule(pl.read_parquet(entry.path)), entry.fetched_at)


def prepare_schedule(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.with_columns(
        kickoff_expr(),
        pl.col("gameday").str.to_date("%Y-%m-%d").alias("game_date"),
        pl.col("home_team").map_elements(franchise, return_dtype=pl.String).alias("home_fr"),
        pl.col("away_team").map_elements(franchise, return_dtype=pl.String).alias("away_fr"),
    ).with_columns(
        pl.min_horizontal("home_fr", "away_fr").alias("team_lo"),
        pl.max_horizontal("home_fr", "away_fr").alias("team_hi"),
    )


def safe_franchise(team: str | None) -> str | None:
    if team is None:
        return None
    try:
        return franchise(team)
    except ValueError:
        return None


def safe_game_id(season: int, week: int, away: str, home: str) -> str | None:
    try:
        return game_id(season, week, away, home)
    except ValueError:
        return None


def safe_team_code(team: str | None, season: int) -> str | None:
    if team is None:
        return None
    try:
        return team_code(team, season)
    except ValueError:
        return None


@dataclass(frozen=True)
class Matched:
    matched: pl.DataFrame
    unmatched: pl.DataFrame


def match_by_ids(df: pl.DataFrame, games: pl.DataFrame) -> Matched:
    ids = df.with_columns(
        pl.struct("season", "week", "away", "home")
        .map_elements(
            lambda row: safe_game_id(row["season"], row["week"], row["away"], row["home"]),
            return_dtype=pl.String,
        )
        .alias("direct_id"),
        pl.struct("season", "week", "away", "home")
        .map_elements(
            lambda row: safe_game_id(row["season"], row["week"], row["home"], row["away"]),
            return_dtype=pl.String,
        )
        .alias("swapped_id"),
    )
    known = games.select("game_id")
    direct = ids.join(known, left_on="direct_id", right_on="game_id", how="semi")
    rest = ids.join(known, left_on="direct_id", right_on="game_id", how="anti")
    swapped = rest.join(known, left_on="swapped_id", right_on="game_id", how="semi")
    unmatched = rest.join(known, left_on="swapped_id", right_on="game_id", how="anti")
    matched = pl.concat(
        [
            direct.with_columns(
                pl.col("direct_id").alias("game_id"), pl.lit(False).alias("swapped")
            ),
            swapped.with_columns(
                pl.col("swapped_id").alias("game_id"), pl.lit(True).alias("swapped")
            ),
        ]
    ).drop("direct_id", "swapped_id")
    return Matched(attach_kickoff(matched, games), unmatched.drop("direct_id", "swapped_id"))


def match_by_date(df: pl.DataFrame, games: pl.DataFrame, tolerance_days: int = 1) -> Matched:
    keyed = df.with_columns(
        pl.col("team_a").map_elements(safe_franchise, return_dtype=pl.String).alias("fr_a"),
        pl.col("team_b").map_elements(safe_franchise, return_dtype=pl.String).alias("fr_b"),
    ).with_columns(
        pl.min_horizontal("fr_a", "fr_b").alias("team_lo"),
        pl.max_horizontal("fr_a", "fr_b").alias("team_hi"),
    )
    schedule = games.select(
        "game_id",
        "season",
        "team_lo",
        "team_hi",
        "home_fr",
        "away_fr",
        pl.col("game_date").alias("game_date_s"),
    )
    by_team = pl.concat(
        [
            schedule.select(
                "game_id", "season", "game_date_s", "home_fr", pl.col(side).alias("known")
            )
            for side in ("home_fr", "away_fr")
        ]
    )
    columns = [*keyed.columns, "game_id", "game_date_s", "home_fr"]
    paired = keyed.join(schedule, on=["season", "team_lo", "team_hi"], how="inner")
    rest = keyed.join(paired.select("row_id"), on="row_id", how="anti").with_columns(
        pl.when(pl.col("fr_b").is_null())
        .then(pl.col("fr_a"))
        .otherwise(pl.col("fr_b"))
        .alias("known")
    )
    one_sided = rest.filter(pl.col("fr_a").is_null() != pl.col("fr_b").is_null()).join(
        by_team, on=["season", "known"], how="inner"
    )
    candidates = (
        pl.concat([paired.select(columns), one_sided.select(columns)])
        .with_columns(
            (pl.col("game_date") - pl.col("game_date_s")).dt.total_days().abs().alias("gap")
        )
        .filter(pl.col("gap") <= tolerance_days)
        .filter(pl.len().over("row_id", "gap") == 1)
    )
    best = candidates.sort("row_id", "gap").unique("row_id", keep="first", maintain_order=True)
    matched = best.with_columns(
        pl.when(pl.col("fr_a").is_not_null())
        .then(pl.col("home_fr") != pl.col("fr_a"))
        .otherwise(pl.col("home_fr") == pl.col("fr_b"))
        .alias("swapped")
    ).drop("fr_a", "fr_b", "team_lo", "team_hi", "game_date_s", "home_fr", "gap")
    unmatched = keyed.join(best.select("row_id"), on="row_id", how="anti").drop(
        "fr_a", "fr_b", "team_lo", "team_hi"
    )
    return Matched(attach_kickoff(matched, games), unmatched)


def attach_kickoff(df: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    extra = [name for name in ("kickoff_utc", "home_team", "away_team") if name in df.columns]
    return df.drop(extra).join(
        games.select("game_id", "kickoff_utc", "home_team", "away_team"), on="game_id", how="left"
    )


def orient(swapped: str, spread: str, home_ml: str, away_ml: str) -> list[pl.Expr]:
    flip = pl.col(swapped)
    return [
        pl.when(flip).then(-pl.col(spread)).otherwise(pl.col(spread)).alias("home_spread"),
        pl.when(flip).then(pl.col(away_ml)).otherwise(pl.col(home_ml)).alias("home_ml"),
        pl.when(flip).then(pl.col(home_ml)).otherwise(pl.col(away_ml)).alias("away_ml"),
    ]


def parse_number(text: object) -> float | None:
    if text is None:
        return None
    if isinstance(text, int | float):
        value = float(text)
        return value if math.isfinite(value) else None
    cleaned = str(text).strip().lower().replace("½", ".5")
    if cleaned in {"pk", "pick", "pickem", "p", "even", "ev"}:
        return 0.0
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_american(text: object) -> float | None:
    if isinstance(text, str) and text.strip().lower() in {"even", "ev", "evs"}:
        return 100.0
    value = parse_number(text)
    if value is None or abs(value) < 100:
        return None
    return value


def american_to_prob(odds: float) -> float:
    return -odds / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def no_vig_prob(home_ml: float | None, away_ml: float | None) -> float | None:
    if home_ml is None or away_ml is None:
        return None
    home = american_to_prob(home_ml)
    away = american_to_prob(away_ml)
    return home / (home + away)


def no_vig_expr(home: str, away: str) -> pl.Expr:
    def implied(name: str) -> pl.Expr:
        odds = pl.col(name)
        return pl.when(odds < 0).then(-odds / (-odds + 100.0)).otherwise(100.0 / (odds + 100.0))

    return implied(home) / (implied(home) + implied(away))


def weekday_estimate(kickoff: pl.Expr, iso_weekday: int, hour: int) -> pl.Expr:
    local = kickoff.dt.convert_time_zone(ET)
    back = (local.dt.weekday() - iso_weekday) % 7
    day = local.dt.date() - pl.duration(days=back)
    stamp = (
        day.cast(pl.Datetime("us"))
        .dt.offset_by(f"{hour}h")
        .dt.replace_time_zone(ET, ambiguous="latest", non_existent="null")
        .dt.convert_time_zone("UTC")
    )
    return pl.when(stamp >= kickoff).then(stamp - pl.duration(days=7)).otherwise(stamp)


def tuesday_noon_before(kickoff: pl.Expr) -> pl.Expr:
    return weekday_estimate(kickoff, 2, 12)


def wednesday_evening_before(kickoff: pl.Expr) -> pl.Expr:
    return weekday_estimate(kickoff, 3, 21)


def close_available(kickoff: pl.Expr) -> pl.Expr:
    return kickoff - pl.duration(minutes=1)


def finalize(df: pl.DataFrame, source: str) -> pl.DataFrame:
    base = df.with_columns(
        pl.lit(source).alias("source"),
        pl.col("home_spread").cast(pl.Float64),
        pl.col("total").cast(pl.Float64),
        pl.col("home_ml").cast(pl.Float64),
        pl.col("away_ml").cast(pl.Float64),
        pl.col("captured_at").cast(UTC_TS),
        pl.col("available_at").cast(UTC_TS),
    )
    keep = [
        *LINES.columns,
        *[c for c in base.columns if c not in LINES.columns and c not in _INTERNAL],
    ]
    return base.select(keep).filter(
        pl.any_horizontal(
            pl.col("home_spread").is_not_null(),
            pl.col("total").is_not_null(),
            pl.col("home_ml").is_not_null(),
        )
    )


_INTERNAL = frozenset(
    {
        "row_id",
        "season",
        "week",
        "away",
        "home",
        "team_a",
        "team_b",
        "game_date",
        "swapped",
        "kickoff_utc",
        "home_team",
        "away_team",
        "ref",
    }
)


def split_open_conflicts(lines: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    closes = (
        lines.filter(pl.col("kind") == "close", pl.col("home_spread").is_not_null())
        .group_by("game_id")
        .agg(pl.col("home_spread").median().alias("close_spread"))
    )
    flagged = lines.join(closes, on="game_id", how="left").with_columns(
        (
            (pl.col("kind") == "open")
            & (pl.col("home_spread").abs() >= MIN_CONFLICT_SPREAD)
            & (pl.col("close_spread").abs() >= MIN_CONFLICT_SPREAD)
            & (pl.col("home_spread").sign() != pl.col("close_spread").sign())
        )
        .fill_null(False)
        .alias("conflict")
    )
    return (
        flagged.filter(~pl.col("conflict")).drop("close_spread", "conflict"),
        flagged.filter(pl.col("conflict")).drop("close_spread", "conflict"),
    )


def write_lines(df: pl.DataFrame, source: str, seasons: Sequence[int]) -> int:
    checked = validate(df, LINES)
    combined = checked
    if storage.exists("clean", LINES.name):
        season = pl.col("game_id").str.slice(0, 4).cast(pl.Int32, strict=False)
        replaced = (
            (pl.col("source") == source)
            & season.is_in(list(seasons))
            & (pl.col("kind") != "snapshot")
        )
        keys = list(LINES.keys)
        old = storage.read("clean", LINES.name).filter(~replaced)
        old = old.join(checked.select(keys), on=keys, how="anti", nulls_equal=True)
        combined = pl.concat([old, checked], how="diagonal_relaxed")
    storage.write(combined.sort(list(LINES.keys)), "clean", LINES.name)
    return checked.height


def reject_frame(
    source: str, reason: str, seasons: Iterable[int | None], refs: Iterable[str]
) -> pl.DataFrame:
    seasons_list = list(seasons)
    return pl.DataFrame(
        {
            "source": [source] * len(seasons_list),
            "season": seasons_list,
            "reason": [reason] * len(seasons_list),
            "ref": list(refs),
            "recorded_at": [datetime.now(UTC)] * len(seasons_list),
        },
        schema=REJECTS.columns,
    )


def write_rejects(source: str, seasons: Sequence[int], frames: Sequence[pl.DataFrame]) -> None:
    new = pl.concat([pl.DataFrame(schema=REJECTS.columns), *frames], how="vertical")
    if storage.exists("clean", REJECTS.name):
        replaced = (pl.col("source") == source) & (
            pl.col("season").is_in(list(seasons)) | pl.col("season").is_null()
        )
        new = pl.concat(
            [storage.read("clean", REJECTS.name).filter(~replaced), new], how="vertical"
        )
    new = new.unique(["source", "reason", "ref"], keep="last", maintain_order=True)
    storage.write(validate(new, REJECTS).sort("source", "reason", "ref"), "clean", REJECTS.name)


def publish(
    source: str, lines: pl.DataFrame, rejects: Sequence[pl.DataFrame], seasons: Sequence[int]
) -> None:
    kept, conflicts = split_open_conflicts(finalize(lines, source))
    season = pl.col("game_id").str.slice(0, 4).cast(pl.Int32)
    conflict_rejects = reject_frame(
        source,
        "open_favorite_conflicts_with_close",
        conflicts.select(season)["game_id"].to_list(),
        (conflicts["game_id"] + ":" + conflicts["book"]).to_list(),
    )
    all_rejects = [*rejects, conflict_rejects]
    write_rejects(source, seasons, all_rejects)
    report(source, write_lines(kept, source, seasons), pl.concat(all_rejects))


def report(source: str, rows: int, rejects: pl.DataFrame | None = None) -> None:
    detail = ""
    if rejects is not None and rejects.height:
        counts = rejects.group_by("reason").len().sort("reason")
        detail = " rejected " + ", ".join(f"{r}={n}" for r, n in counts.iter_rows())
    console.print(f"  {source}: {rows} rows{detail}")
