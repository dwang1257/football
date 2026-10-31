import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import polars as pl

from football.opinions.blend import BlendParams
from football.opinions.learning import FittedWeights
from football.report.scoring import (
    Metrics,
    Scorecard,
    calibration,
    flip_ledger,
)


def _pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{100 * value:.1f}%"


def _num(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.4f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _side(home_prob: float, row: dict[str, Any]) -> str:
    if home_prob >= 0.5:
        return f"{row['home_team']} {100 * home_prob:.1f}%"
    return f"{row['away_team']} {100 * (1.0 - home_prob):.1f}%"


def _owner(row: dict[str, Any]) -> str:
    parts: list[str] = []
    if row["user_pick"] is not None:
        conf = row["user_conf"]
        strength = f"c{conf}" if conf is not None else f"{100 * row['user_pick_prob']:.0f}%"
        parts.append(f"{row['user_pick']} {strength}")
    if row["team_nudge_points"]:
        parts.append(f"{row['home_team']} {row['team_nudge_points']:+.1f} pts")
    return ", ".join(parts) or "-"


def _winner(row: dict[str, Any]) -> str:
    if row["home_win"] == 0.5:
        return "tie"
    return row["home_team"] if row["home_win"] == 1.0 else row["away_team"]


def _flags(row: dict[str, Any]) -> str:
    flags = [name for name, on in (("FLIPPED", row["flipped"]), ("cap", row["capped"])) if on]
    flags.append("final" if row["is_final"] else "latest")
    return " ".join(flags)


def games_section(df: pl.DataFrame) -> list[str]:
    rows = [
        [
            f"{row['away_team']}@{row['home_team']}",
            f"{row['away_score']}-{row['home_score']}",
            _side(row["model_prob"], row),
            _owner(row),
            _side(row["blended_prob"], row),
            _winner(row),
            _flags(row),
        ]
        for row in df.iter_rows(named=True)
    ]
    header = ("Game", "Score", "Model", "You", "Blend", "Winner", "Notes")
    lines = ["## Games", "", *_table(header, rows)]
    whys = [
        f"- {row['away_team']}@{row['home_team']}: {row['user_why']}"
        for row in df.iter_rows(named=True)
        if row["user_why"]
    ]
    return [*lines, "", "Owner notes:", "", *whys] if whys else lines


def _metric_row(scope: str, name: str, value: Metrics) -> list[str]:
    return [
        scope,
        name,
        str(value.n),
        _pct(value.accuracy),
        _num(value.log_loss),
        _num(value.brier),
    ]


def scorecard_section(cards: Sequence[Scorecard]) -> list[str]:
    rows: list[list[str]] = []
    for card in cards:
        rows += [
            _metric_row(card.label, "Model, all games", card.model),
            _metric_row(card.label, "Blended, all games", card.blended),
            _metric_row(card.label, "Model, opinion games", card.model_on_opinions),
            _metric_row(card.label, "Blended, opinion games", card.blended_on_opinions),
            _metric_row(card.label, "Owner only, picked games", card.owner),
        ]
    header = ("Scope", "Predictor", "Games", "Accuracy", "Log loss", "Brier")
    paired = [
        f"- {card.label}: blended minus model log loss on {card.paired.n} opinion games "
        f"{_signed(card.paired.mean)} (SE {_num(card.paired.se)}); "
        f"flips won {card.flips_won}, lost {card.flips_lost}."
        for card in cards
    ]
    return ["## Scorecard", "", *_table(header, rows), "", *paired]


def _signed(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:+.4f}"


def flips_section(df: pl.DataFrame) -> list[str]:
    ledger = flip_ledger(df)
    lines = ["## Flip ledger, season to date", ""]
    if ledger.is_empty():
        return [*lines, "No flipped picks yet."]
    rows = [
        [
            f"{row['week']}",
            f"{row['away_team']}@{row['home_team']}",
            _side(row["model_prob"], row),
            _side(row["blended_prob"], row),
            _winner(row),
            row["outcome"],
        ]
        for row in ledger.iter_rows(named=True)
    ]
    header = ("Week", "Game", "Model", "Blend", "Winner", "Flip")
    return [*lines, *_table(header, rows)]


def calibration_section(df: pl.DataFrame) -> list[str]:
    table = calibration(df)
    lines = ["## Calibration by confidence, season to date", ""]
    if table.is_empty():
        return [*lines, "No owner picks scored yet."]
    rows = [
        [row["level"], str(row["n"]), _pct(row["assumed"]), _pct(row["actual"])]
        for row in table.iter_rows(named=True)
    ]
    return [*lines, *_table(("Level", "Picks", "Assumed", "Actual"), rows)]


def weights_section(params: BlendParams, fitted: FittedWeights | None) -> list[str]:
    source = (
        f"fitted {fitted.fitted_at:%Y-%m-%d} on {fitted.n_games} opinion games "
        f"and {fitted.n_team_games} team nudge games"
        if fitted
        else "defaults"
    )
    table = ", ".join(
        f"c{conf} {100 * prob:.1f}%" for conf, prob in sorted(params.conf_table.items())
    )
    return [
        "## Opinion weights",
        "",
        f"- Source: {source}.",
        f"- Game weight w {params.weight:.3f}, team weight w_t {params.team_weight:.3f}.",
        f"- Confidence table: {table}.",
    ]


def render_report(
    season: int,
    week: int,
    season_games: pl.DataFrame,
    cards: Sequence[Scorecard],
    params: BlendParams,
    fitted: FittedWeights | None,
    now: datetime,
) -> str:
    week_games = season_games.filter(pl.col("week") == week)
    sections = [
        [f"# {season} week {week} report", "", f"Generated {now:%Y-%m-%d %H:%M} UTC."],
        games_section(week_games),
        scorecard_section(cards),
        flips_section(season_games),
        calibration_section(season_games),
        weights_section(params, fitted),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
