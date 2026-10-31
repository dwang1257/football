from football.schemas import MATCHUP_STATS

SIDES: tuple[str, str] = ("home", "away")
UNITS: tuple[str, str] = ("off", "def")

ELO_FEATURES: tuple[str, ...] = ("home_elo", "away_elo", "elo_diff", "elo_home_prob")

MATCHUP_COLUMNS: tuple[str, ...] = (
    *(
        f"{side}_{unit}_{stat}{suffix}"
        for side in SIDES
        for unit in UNITS
        for stat in MATCHUP_STATS
        for suffix in ("", "_sd")
    ),
    *(f"lg_{stat}{suffix}" for stat in MATCHUP_STATS for suffix in ("", "_sd")),
    "home_games_played",
    "away_games_played",
)

SITUATIONAL_FEATURES: tuple[str, ...] = (
    "home_rest",
    "away_rest",
    "rest_diff",
    "home_bye",
    "away_bye",
    "home_short_week",
    "away_short_week",
    "div_game",
    "neutral_site",
    "is_playoff",
    "travel_km_home",
    "travel_km_away",
    "tz_shift_away",
    "west_coast_early",
    "elevation_m",
    "is_dome",
    "is_retractable",
    "is_grass",
    "is_international",
)

QB_FEATURES: tuple[str, ...] = (
    "home_qb_value",
    "away_qb_value",
    "qb_value_diff",
    "home_qb_change",
    "away_qb_change",
)

INJURY_FEATURES: tuple[str, ...] = tuple(
    f"{side}_inj_{group}" for side in SIDES for group in ("ol", "db", "other")
)

WEATHER_FEATURES: tuple[str, ...] = (
    "fcst_temp_f",
    "fcst_wind_mph",
    "fcst_precip_prob",
    "wind_15plus",
    "cold_game",
    "home_dome_team_in_cold",
    "away_dome_team_in_cold",
)

OWNER_FEATURES: tuple[str, ...] = (
    "home_season_win_pct",
    "away_season_win_pct",
    "season_win_pct_diff",
    "home_pythag",
    "away_pythag",
    "pythag_diff",
    "home_pd_per_game",
    "away_pd_per_game",
    "pd_per_game_diff",
    "home_stadium_win_pct",
    "home_stadium_games",
    "home_coach_vs_opp_win_pct",
    "home_coach_vs_opp_games",
    "away_coach_vs_opp_win_pct",
    "away_coach_vs_opp_games",
    "home_cold_win_pct",
    "away_cold_win_pct",
    "home_windy_win_pct",
    "away_windy_win_pct",
    "home_fumble_luck",
    "away_fumble_luck",
)

WIN_TOTAL_FEATURES: tuple[str, ...] = ("home_win_total", "away_win_total", "win_total_diff")

LINE_FEATURES: tuple[str, ...] = (
    "market_home_spread",
    "market_home_prob",
    "market_total",
    "market_spread_move",
)

MARKET_FEATURES: tuple[str, ...] = (*LINE_FEATURES, *WIN_TOTAL_FEATURES)

BENCHMARK_COLUMNS: tuple[str, ...] = ("bench_nfelo_prob", "bench_elo538_prob", "bench_fpi_diff")

INFO_COLUMNS: tuple[str, ...] = ("home_exp_qb_id", "away_exp_qb_id")

LABELS: tuple[str, str] = ("home_win", "home_margin")

SCORE_LABELS: tuple[str, str] = ("home_points", "away_points")

INDEPENDENT_FEATURES: tuple[str, ...] = (
    *ELO_FEATURES,
    *MATCHUP_COLUMNS,
    *SITUATIONAL_FEATURES,
    *QB_FEATURES,
    *INJURY_FEATURES,
    *WEATHER_FEATURES,
    *OWNER_FEATURES,
)

FEATURE_COLUMNS: tuple[str, ...] = (*INDEPENDENT_FEATURES, *MARKET_FEATURES)
