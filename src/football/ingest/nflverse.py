from football.ingest.base import IngestFn
from football.ingest.nflverse_checks import coverage_by_season, scores_match_pbp, team_stats_match
from football.ingest.nflverse_games import ingest_schedules
from football.ingest.nflverse_misc import (
    ingest_contracts,
    ingest_draft_picks,
    ingest_qbr,
    ingest_trades,
)
from football.ingest.nflverse_pbp import ingest_pbp
from football.ingest.nflverse_players import (
    ingest_depth_charts,
    ingest_injuries,
    ingest_rosters_weekly,
    ingest_snap_counts,
)
from football.validate.base import CheckFn

SOURCES: dict[str, IngestFn] = {
    "schedules": ingest_schedules,
    "pbp": ingest_pbp,
    "injuries": ingest_injuries,
    "depth_charts": ingest_depth_charts,
    "snap_counts": ingest_snap_counts,
    "rosters_weekly": ingest_rosters_weekly,
    "qbr": ingest_qbr,
    "contracts": ingest_contracts,
    "draft_picks": ingest_draft_picks,
    "trades": ingest_trades,
}

CHECKS: dict[str, CheckFn] = {
    "scores_match_pbp": scores_match_pbp,
    "team_stats_match": team_stats_match,
    "coverage_by_season": coverage_by_season,
}
