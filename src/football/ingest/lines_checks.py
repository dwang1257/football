from dataclasses import dataclass
from itertools import combinations

import polars as pl

from football import storage
from football.ingest.lines_common import REJECTS
from football.schemas import LINES
from football.validate.base import CheckFn, CheckResult

AGREEMENT_THRESHOLD = 0.85
MIN_OVERLAP = 100
UNMATCHED_THRESHOLD = 0.01
WORST = 3
OPEN_NOTE = (
    "report only: opens differ by definition (nfelo legacy and SBRO offshore Sunday-night opener, "
    "ESPN and nfelo 2024+ first book capture, Hugging Face multi-book median of first prices)"
)


def per_game(lines: pl.DataFrame, kind: str) -> pl.DataFrame:
    return (
        lines.filter(pl.col("kind") == kind, pl.col("home_spread").is_not_null())
        .group_by("game_id", "source")
        .agg(pl.col("home_spread").median())
    )


@dataclass(frozen=True)
class PairStats:
    pair: str
    n: int
    exact: float
    within: float
    worst: str

    def describe(self) -> str:
        return (
            f"{self.pair}: n={self.n} exact={self.exact:.1%} within0.5={self.within:.1%} "
            f"worst [{self.worst}]"
        )


def pair_stats(spreads: pl.DataFrame, left: str, right: str) -> PairStats | None:
    joined = (
        spreads.filter(pl.col("source") == left)
        .join(spreads.filter(pl.col("source") == right), on="game_id", suffix="_r")
        .with_columns((pl.col("home_spread") - pl.col("home_spread_r")).abs().alias("diff"))
    )
    if joined.is_empty():
        return None
    shares = joined.select(
        (pl.col("diff") < 0.01).mean().alias("exact"),
        (pl.col("diff") <= 0.5).mean().alias("within"),
    ).row(0, named=True)
    worst = joined.sort("diff", descending=True).head(WORST)
    return PairStats(
        pair=f"{left.removeprefix('lines_')}~{right.removeprefix('lines_')}",
        n=joined.height,
        exact=float(shares["exact"]),
        within=float(shares["within"]),
        worst=", ".join(
            f"{g} {a:+g}/{b:+g}"
            for g, a, b in worst.select("game_id", "home_spread", "home_spread_r").iter_rows()
        ),
    )


def agreement(kind: str) -> tuple[list[PairStats], str]:
    if not storage.exists("clean", LINES.name):
        return [], "no lines table"
    spreads = per_game(storage.read("clean", LINES.name), kind)
    sources = sorted(spreads["source"].unique().to_list())
    stats = [s for a, b in combinations(sources, 2) if (s := pair_stats(spreads, a, b)) is not None]
    return stats, "\n".join(s.describe() for s in stats)


def closing_spread_agreement() -> CheckResult:
    stats, detail = agreement("close")
    enforced = [s for s in stats if s.n >= MIN_OVERLAP]
    failing = [s.pair for s in enforced if s.within < AGREEMENT_THRESHOLD]
    passed = bool(enforced) and not failing
    head = (
        f"{len(enforced)} pairs with n>={MIN_OVERLAP}; failing {failing}"
        if enforced
        else "no overlapping sources"
    )
    return CheckResult("closing_spread_agreement", passed, f"{head}\n{detail}")


def opening_spread_agreement() -> CheckResult:
    _, detail = agreement("open")
    return CheckResult("opening_spread_agreement", True, f"{OPEN_NOTE}\n{detail}")


def lines_unmatched() -> CheckResult:
    if not storage.exists("clean", REJECTS.name) or not storage.exists("clean", LINES.name):
        return CheckResult("lines_unmatched", False, "missing lines or lines_rejects table")
    rejects = storage.read("clean", REJECTS.name).filter(pl.col("source").str.starts_with("lines_"))
    matched = (
        storage.read("clean", LINES.name)
        .group_by("source")
        .agg(pl.col("game_id").n_unique().alias("games"), pl.len().alias("rows"))
    )
    unmatched = rejects.filter(pl.col("reason") == "unmatched").group_by("source").len("unmatched")
    other = (
        rejects.filter(pl.col("reason") != "unmatched")
        .group_by("source", "reason")
        .len()
        .sort("source", "reason")
    )
    summary = (
        matched.join(unmatched, on="source", how="full", coalesce=True)
        .fill_null(0)
        .with_columns(
            (pl.col("unmatched") / (pl.col("unmatched") + pl.col("games"))).alias("share")
        )
        .sort("source")
    )
    failing = summary.filter(pl.col("share") > UNMATCHED_THRESHOLD)["source"].to_list()
    lines = [
        f"{src}: rows={rows} games={games} unmatched={un} ({share:.2%})"
        for src, games, rows, un, share in summary.select(
            "source", "games", "rows", "unmatched", "share"
        ).iter_rows()
    ]
    lines += [f"{src} {reason}: {n}" for src, reason, n in other.iter_rows()]
    examples = (
        rejects.filter(pl.col("reason") == "unmatched")
        .group_by("source")
        .agg(pl.col("ref").head(3))
    )
    lines += [f"{src} examples: {refs}" for src, refs in examples.iter_rows()]
    return CheckResult("lines_unmatched", not failing, f"failing {failing}\n" + "\n".join(lines))


CHECKS: dict[str, CheckFn] = {
    "closing_spread_agreement": closing_spread_agreement,
    "opening_spread_agreement": opening_spread_agreement,
    "lines_unmatched": lines_unmatched,
}
