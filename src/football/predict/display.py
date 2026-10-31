from typing import Any

import polars as pl
from rich.table import Table

from football.predict.run import PredictOutcome


def side_prob(home_prob: float, home: str, away: str) -> tuple[str, float]:
    return (home, home_prob) if home_prob >= 0.5 else (away, 1.0 - home_prob)


def _you(row: dict[str, Any], team_points: dict[str, float]) -> str:
    parts: list[str] = []
    if row["user_pick"] is not None:
        strength = (
            f"c{row['user_conf']}"
            if row["user_conf"] is not None
            else f"{100 * row['user_pick_prob']:.0f}%"
        )
        parts.append(f"{row['user_pick']} {strength}")
    parts += [
        f"{team} {team_points[team]:+.1f} pts"
        for team in (row["away_team"], row["home_team"])
        if team_points.get(team)
    ]
    if row["opinion_late"]:
        parts.append("late, ignored")
    return ", ".join(parts) or "-"


def _flags(row: dict[str, Any]) -> str:
    flags = [
        label
        for label, on in (
            ("[bold red]FLIPPED[/bold red]", row["flipped"]),
            ("[yellow]cap[/yellow]", row["capped"]),
            ("[cyan]locked[/cyan]", row["is_final"]),
        )
        if on
    ]
    return " ".join(flags)


def why_lines(outcome: PredictOutcome) -> list[str]:
    lines: list[str] = []
    team_whys = outcome.opinions.team_whys if outcome.opinions else {}
    for row in weekly_frame(outcome).iter_rows(named=True):
        game = f"{row['away_team']}@{row['home_team']}"
        if row["user_why"]:
            lines.append(f"{game} {row['user_pick']}: {row['user_why']}")
        lines += [
            f"{game} {team}: {team_whys[team]}"
            for team in (row["away_team"], row["home_team"])
            if team in team_whys
        ]
    return lines


def weekly_frame(outcome: PredictOutcome) -> pl.DataFrame:
    info = outcome.games.select("game_id", "kickoff_utc", "away_team", "home_team")
    return (
        outcome.blended_rows.drop("created_at", "is_final")
        .join(outcome.model_rows.select("game_id", "is_final"), on="game_id", how="left")
        .join(info, on="game_id", how="left")
        .sort("kickoff_utc", "game_id")
    )


def _delta(points: float) -> str:
    rounded = round(points, 1)
    return "0.0" if rounded == 0 else f"{rounded:+.1f}"


def weekly_table(outcome: PredictOutcome, title: str) -> Table:
    table = Table(title=title)
    for name in ("Game", "Model", "You", "Blend", "delta pp", "Pick", "Notes"):
        table.add_column(name, justify="right" if name == "delta pp" else "left")
    team_points = outcome.opinions.team_points if outcome.opinions else {}
    for row in weekly_frame(outcome).iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        model_team, model_prob = side_prob(row["model_prob"], home, away)
        blend_team, blend_prob = side_prob(row["blended_prob"], home, away)
        towards_model = row["blended_prob"] if model_team == home else 1.0 - row["blended_prob"]
        table.add_row(
            f"{away}@{home}",
            f"{model_team} {100 * model_prob:.1f}%",
            _you(row, team_points),
            f"{blend_team} {100 * blend_prob:.1f}%",
            _delta(100 * (towards_model - model_prob)),
            blend_team,
            _flags(row),
        )
    return table
