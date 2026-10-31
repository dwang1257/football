from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from football.features.inputs import Tables
from football.schemas import (
    BENCHMARKS,
    DEPTH_CHARTS,
    GAMES,
    INJURIES,
    LINES,
    QB_GAMES,
    TEAM_GAMES,
    TEAM_HOMES,
    TEAM_RATINGS,
    WEATHER_FORECASTS,
    WEATHER_OBSERVED,
    WIN_TOTALS,
    TableSpec,
    validate,
)

TEAMS = ("KC", "BUF", "SF", "DEN", "MIA", "LAC")
STADIUMS = {
    "KC": ("KAN00", 39.05, -94.48, "America/Chicago", "outdoors", "grass", 260.0),
    "BUF": ("BUF00", 42.77, -78.79, "America/New_York", "outdoors", "a_turf", 180.0),
    "SF": ("SFO01", 37.40, -121.97, "America/Los_Angeles", "outdoors", "grass", 5.0),
    "DEN": ("DEN00", 39.74, -105.02, "America/Denver", "outdoors", "grass", 1609.0),
    "MIA": ("MIA00", 25.96, -80.24, "America/New_York", "outdoors", "grass", 3.0),
    "LAC": ("LAX01", 33.95, -118.34, "America/Los_Angeles", "dome", "matrixturf", 30.0),
}
LONDON = ("LON00", 51.56, -0.28, "Europe/London", "retractable", "grass", 40.0)
POSITIONS = ("T", "G", "CB", "S", "WR", "LB")
KICKOFF_SLOTS = (
    timedelta(hours=17),
    timedelta(hours=20, minutes=25),
    timedelta(days=1, hours=0, minutes=20),
)


def required(df: pl.DataFrame | None) -> pl.DataFrame:
    assert df is not None
    return df


def _frame(rows: list[dict[str, object]], spec: TableSpec) -> pl.DataFrame:
    df = pl.DataFrame(rows, schema=spec.columns) if rows else pl.DataFrame(schema=spec.columns)
    return validate(df, spec)


@dataclass(frozen=True)
class Slot:
    season: int
    week: int
    home: str
    away: str
    kickoff: datetime
    neutral: bool
    game_type: str


def _schedule(seasons: int, weeks: int, rng: np.random.Generator) -> list[Slot]:
    games: list[Slot] = []
    for s in range(seasons):
        season = 2020 + s
        start = datetime(season, 9, 10, tzinfo=UTC)
        for week in range(1, weeks + 1):
            order = list(rng.permutation(len(TEAMS)))
            for slot in range(len(TEAMS) // 2):
                games.append(
                    Slot(
                        season,
                        week,
                        TEAMS[order[2 * slot]],
                        TEAMS[order[2 * slot + 1]],
                        start + timedelta(days=7 * (week - 1)) + KICKOFF_SLOTS[slot],
                        week == 3 and slot == 0,
                        "REG" if week < weeks else "WC",
                    )
                )
    return games


def make_tables(seed: int = 0, seasons: int = 2, weeks: int = 6, unplayed: int = 3) -> Tables:
    rng = np.random.default_rng(seed)
    schedule = _schedule(seasons, weeks, rng)
    last_kickoff: dict[str, datetime] = {}
    games, team_games, qb_games, injuries, depth, snaps = [], [], [], [], [], []
    lines, benchmarks, ratings, forecasts, observed, venues = [], [], [], [], [], []
    starters = {team: [f"{team}_QB1", f"{team}_QB2", f"{team}_QB3"] for team in TEAMS}
    for index, g in enumerate(schedule):
        season, week, home, away, kickoff = g.season, g.week, g.home, g.away, g.kickoff
        game_id = f"{season}_{week:02d}_{away}_{home}"
        played = index < len(schedule) - unplayed
        neutral = g.neutral
        venue = LONDON if neutral else STADIUMS[home]
        rest = {
            team: 7 if team not in last_kickoff else (kickoff - last_kickoff[team]).days
            for team in (home, away)
        }
        last_kickoff[home] = last_kickoff[away] = kickoff
        hs = int(rng.integers(0, 42)) if played else None
        as_ = int(rng.integers(0, 42)) if played else None
        temp = float(rng.normal(55, 20))
        wind = float(abs(rng.normal(8, 6)))
        result_at = kickoff + timedelta(hours=4) + timedelta(minutes=int(rng.integers(0, 600)))
        games.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "game_type": g.game_type,
                "kickoff_utc": kickoff,
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "neutral_site": neutral,
                "stadium_id": venue[0],
                "roof": "closed" if venue[4] == "retractable" else venue[4],
                "surface": venue[5],
                "div_game": bool((TEAMS.index(home) + TEAMS.index(away)) % 2 == 0),
                "home_rest": rest[home],
                "away_rest": rest[away],
                "home_coach": f"coach_{home}",
                "away_coach": f"coach_{away}",
                "referee": "ref",
                "home_qb_id": starters[home][0] if played else None,
                "away_qb_id": starters[away][0] if played else None,
                "spread_line": float(rng.normal(0, 6)),
                "total_line": 44.5,
                "home_moneyline": -150.0,
                "away_moneyline": 130.0,
                "temp_observed": temp if played else None,
                "wind_observed": wind if played else None,
                "available_at": result_at,
            }
        )
        venues.append(
            {
                "game_id": game_id,
                "stadium_id": venue[0],
                "roof_type": venue[4],
                "surface": venue[5],
                "lat": venue[1],
                "lon": venue[2],
                "tz": venue[3],
                "elevation_m": venue[6],
                "is_international": neutral,
            }
        )
        for team, opp, pf, pa in ((home, away, hs, as_), (away, home, as_, hs)):
            for rank, qb in enumerate(starters[team], start=1):
                depth.append(
                    {
                        "season": season,
                        "week": week,
                        "team": team,
                        "player_id": qb,
                        "full_name": qb,
                        "position": "QB",
                        "depth_rank": rank,
                        "captured_at": kickoff - timedelta(days=5),
                        "available_at": kickoff - timedelta(days=5, hours=-int(rng.integers(0, 3))),
                    }
                )
            out_qb = rng.random() < 0.25
            injuries.append(
                {
                    "season": season,
                    "week": week,
                    "game_type": g.game_type,
                    "team": team,
                    "player_id": starters[team][0],
                    "full_name": starters[team][0],
                    "position": "QB",
                    "report_status": "Out" if out_qb else "Questionable",
                    "practice_status": "DNP",
                    "available_at": kickoff - timedelta(days=2),
                }
            )
            for pos_index, pos in enumerate(POSITIONS):
                if rng.random() < 0.5:
                    injuries.append(
                        {
                            "season": season,
                            "week": week,
                            "game_type": g.game_type,
                            "team": team,
                            "player_id": f"{team}_{pos}{pos_index}",
                            "full_name": f"{team} {pos}",
                            "position": pos,
                            "report_status": str(rng.choice(["Out", "Doubtful", "Questionable"])),
                            "practice_status": "Limited",
                            "available_at": kickoff
                            - timedelta(days=2, hours=int(rng.integers(-30, 30))),
                        }
                    )
            if not played:
                continue
            team_at = kickoff + timedelta(hours=int(rng.integers(5, 30)))
            plays = int(rng.integers(50, 75))
            pass_plays = int(rng.integers(20, plays - 15))
            trips = int(rng.integers(0, 6))
            team_games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "kickoff_utc": kickoff,
                    "team": team,
                    "opponent": opp,
                    "is_home": team == home,
                    "points_for": pf,
                    "points_against": pa,
                    "plays": plays,
                    "pass_plays": pass_plays,
                    "rush_plays": plays - pass_plays,
                    "pass_yards": float(rng.normal(230, 60)),
                    "rush_yards": float(rng.normal(110, 35)),
                    "total_yards": 340.0,
                    "pass_epa": float(rng.normal(0, 6)),
                    "rush_epa": float(rng.normal(0, 4)),
                    "total_epa": float(rng.normal(0, 8)),
                    "success_plays": int(rng.integers(15, 35)),
                    "wp_filtered_plays": int(rng.integers(0, plays)),
                    "wp_filtered_epa": float(rng.normal(0, 7)),
                    "takeaways": int(rng.integers(0, 4)),
                    "giveaways": int(rng.integers(0, 4)),
                    "fumbles_lost": int(rng.integers(0, 2)),
                    "interceptions_thrown": int(rng.integers(0, 2)),
                    "drives": 11,
                    "red_zone_trips": trips,
                    "red_zone_tds": int(rng.integers(0, trips + 1)),
                    "avg_start_yardline": float(rng.normal(28, 4)),
                    "available_at": team_at,
                }
            )
            starter = starters[team][1] if out_qb else starters[team][0]
            for qb, starts in ((starter, True), (starters[team][2], False)):
                qb_games.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "kickoff_utc": kickoff,
                        "team": team,
                        "player_id": qb,
                        "full_name": qb,
                        "is_starter": starts,
                        "dropbacks": int(rng.integers(1, 45)),
                        "qb_epa": float(rng.normal(1, 8)),
                        "cpoe": float(rng.normal(0, 5)),
                        "available_at": team_at,
                    }
                )
            for pos_index, pos in enumerate(POSITIONS):
                share = float(rng.random())
                snaps.append(
                    {
                        "game_id": game_id,
                        "team": team,
                        "player_id": f"{team}_{pos}{pos_index}",
                        "position": pos,
                        "offense_pct": share if pos in ("T", "G", "WR") else 0.0,
                        "defense_pct": share if pos not in ("T", "G", "WR") else 0.0,
                        "available_at": kickoff + timedelta(days=1, hours=int(rng.integers(0, 12))),
                    }
                )
            ratings.append(
                {
                    "source": "espn_fpi",
                    "season": season,
                    "week": week,
                    "team": team,
                    "rating": float(rng.normal(0, 5)),
                    "offense": 0.0,
                    "defense": 0.0,
                    "special_teams": 0.0,
                    "captured_at": kickoff + timedelta(days=2),
                    "available_at": kickoff + timedelta(days=2),
                }
            )
        for kind, delta in (
            ("open", timedelta(days=6)),
            ("snapshot", timedelta(days=2)),
            ("close", timedelta(minutes=1)),
        ):
            for source in ("lines_espn", "lines_sbro"):
                lines.append(
                    {
                        "game_id": game_id,
                        "source": source,
                        "book": "book",
                        "kind": kind,
                        "captured_at": kickoff - delta,
                        "home_spread": float(rng.normal(0, 6)),
                        "total": float(rng.normal(44, 4)),
                        "home_ml": -120.0,
                        "away_ml": 100.0,
                        "available_at": kickoff - delta,
                    }
                )
        for source in ("nfelo", "elo538"):
            benchmarks.append(
                {
                    "game_id": game_id,
                    "source": source,
                    "home_win_prob": float(rng.random()),
                    "home_margin": float(rng.normal(0, 7)),
                    "available_at": kickoff - timedelta(hours=int(rng.integers(1, 72))),
                }
            )
        for hours in (48, 24, 6, -2):
            run = kickoff - timedelta(hours=hours)
            forecasts.append(
                {
                    "game_id": game_id,
                    "source": "iem_mos",
                    "station": "KXXX",
                    "run_time": run,
                    "valid_time": kickoff,
                    "temp_f": float(rng.normal(55, 20)),
                    "wind_mph": float(abs(rng.normal(9, 6))),
                    "wind_gust_mph": 20.0,
                    "precip_prob": float(rng.random()),
                    "available_at": run + timedelta(hours=1),
                }
            )
        if played:
            observed.append(
                {
                    "game_id": game_id,
                    "source": "meteostat",
                    "temp_f": float(rng.normal(50, 20)),
                    "wind_mph": float(abs(rng.normal(10, 6))),
                    "precip_in": 0.0,
                    "available_at": kickoff + timedelta(days=int(rng.integers(0, 3)), hours=5),
                }
            )
    homes = [
        {
            "season": 2020 + s,
            "team": team,
            "stadium_id": STADIUMS[team][0],
            "lat": STADIUMS[team][1],
            "lon": STADIUMS[team][2],
            "tz": STADIUMS[team][3],
            "roof_type": STADIUMS[team][4],
        }
        for s in range(seasons)
        for team in TEAMS
    ]
    totals = [
        {
            "season": 2020 + s,
            "team": team,
            "source": "covers",
            "win_total": float(rng.integers(4, 13)) + 0.5,
            "over_prob": 0.5,
            "available_at": datetime(2020 + s, 8, 1, tzinfo=UTC),
        }
        for s in range(seasons)
        for team in TEAMS
    ]
    snap_schema = {
        "game_id": pl.String(),
        "team": pl.String(),
        "player_id": pl.String(),
        "position": pl.String(),
        "offense_pct": pl.Float64(),
        "defense_pct": pl.Float64(),
        "available_at": pl.Datetime("us", "UTC"),
    }
    return Tables(
        games=_frame(games, GAMES),
        team_games=_frame(team_games, TEAM_GAMES),
        qb_games=_frame(qb_games, QB_GAMES),
        injuries=_frame(injuries, INJURIES).unique(
            ["season", "week", "team", "player_id"], keep="first", maintain_order=True
        ),
        depth_charts=_frame(depth, DEPTH_CHARTS),
        snap_counts=pl.DataFrame(snaps, schema=snap_schema),
        lines=_frame(lines, LINES),
        benchmarks=_frame(benchmarks, BENCHMARKS),
        team_ratings=_frame(ratings, TEAM_RATINGS),
        weather_forecasts=_frame(forecasts, WEATHER_FORECASTS),
        weather_observed=_frame(observed, WEATHER_OBSERVED),
        game_venues=pl.DataFrame(venues),
        team_homes=_frame(homes, TEAM_HOMES),
        win_totals=_frame(totals, WIN_TOTALS),
    )
