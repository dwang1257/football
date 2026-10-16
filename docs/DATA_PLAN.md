# Data Acquisition Plan

Status: draft, 2026-10-16, mid 2026 season (weeks 1-5 final, week 7 lines posted).
Scope: finding, verifying, and collecting free, accurate, in-depth NFL data for the predictor described in [PLAN.md](PLAN.md).

## 1. Rules for every source

- Free only: no paid subscriptions or paid API tiers.
- Point-in-time: every row gets an `available_at` timestamp for when it would have been known, and nothing known after kickoff may feed a prediction for that game.
- Verified: a source is only "in" once it has been downloaded from this environment and its row counts and coverage are recorded.
- Cross-checked: any field available from two sources is compared, and disagreements are measured before the field is trusted.
- Provenance: every row keeps its source name, fetch time, and source version or commit.
- Licensing respected: sources without a license, or with non-commercial terms, are used for personal research only and are never redistributed; sources whose terms forbid automated access or ML use are excluded.
- Polite access: throttle, cache, and never re-download unchanged data.

## 2. Verified source catalog

"Verified" means downloaded from this environment on 2026-10-16.

### 2.1 Games, results, schedule context

| Source | Coverage | Contents | Status |
|---|---|---|---|
| nflverse `load_schedules` | 1999-2026, 7,548 games | Scores, home/away, neutral site, `stadium_id`, roof, surface, temp, wind, rest days, div game, closing spread/total/ML, starting QBs, head coaches, referee | Verified, CC-BY 4.0 |
| nflverse/nfldata `games.csv` | same | Same data, raw GitHub mirror as a fallback | Verified |

### 2.2 Play-level and team stats (inputs for the matchup simulation model)

| Source | Coverage | Contents | Status |
|---|---|---|---|
| nflverse `load_pbp` | 1999-2026, 372 columns | Every play: yards, pass/rush, EPA, success, WPA, CPOE, turnovers, field position, red zone | Verified |
| nflverse `load_team_stats` | weekly | Team-game aggregates incl. EPA | Verified |
| nflverse `load_player_stats` | weekly | Player-game stats | Verified |
| nflverse `load_nextgen_stats` | 2016+ | Tracking-derived passing, rushing, receiving | Verified |
| nflverse `load_ftn_charting` | 2022+ | Charting: play action, blitz, motion | Verified, CC-BY-SA, credit FTN |
| nflverse `load_pfr_advstats` | 2018+ | Pressures, drops, broken tackles | Verified |
| ESPN QBR (nflverse release CSV) | 2006-2026 | Weekly QB rating | Verified |

The matchup model needs, per team-game and split into for and against: plays, passing yards, rushing yards, yards per play, EPA per play, success rate, takeaways, giveaways, red zone trips and conversions, and starting field position.
All of these are derived from `load_pbp`, then checked against `load_team_stats` and final scores (section 4).

### 2.3 Quarterbacks and injuries

| Source | Coverage | Contents | Timing | Status |
|---|---|---|---|---|
| nflverse `load_injuries` | 2009-2026 | Final weekly game status and practice status | Friday report | Verified |
| nflverse depth charts | 2001-2024 weekly, daily snapshots 2025+ | Depth chart by position | Early in the week, stale but leak-free | Verified |
| nflverse weekly rosters | 2019+ for inactives | `status=INA` game-day inactives | 90 min before kickoff | Verified |
| nfl.com/injuries/league/{year}/reg{N} | 2012-2026 | Same as `load_injuries` | Friday | Verified, cross-check only |
| NFL team site injury reports | current season only | Wed, Thu, Fri practice participation plus game status | Daily during the week | Verified, forward collection |
| ESPN core API team injuries | current only | Timestamped injury entries | Live | Verified, forward collection |

Expected starting QB rule: QB1 on that week's depth chart, overridden when the injury report lists him Out or Doubtful.
That rule matched the actual starter in 87-90% of team-games in 2015, 2023, and 2024, and uses only pre-game information.

Do not use: ESPN summary API injuries for past games (returns current injuries, which leaks the present into history), Pro-Football-Reference (terms forbid ML use and automated access).

### 2.4 Betting lines and market priors

| Source | Coverage | Open line | Close line | Status |
|---|---|---|---|---|
| nflverse `spread_line`, `total_line`, ML | 1999-2026 | No | Yes | Verified |
| greerreNFL/nfelomarket_data `Data/lines.csv` | 1999-2026, 7,369 games | Spread from 2007; ML and total from 2024 | Yes | Verified, no license |
| sportsbookreviewsonline season pages | 2007-2021, 4,025 games | Spread and total | Spread, total, ML | Verified (HTML parse), no license |
| nflverse/nfldata `closing_lines.csv` | 2006-2018 | No | Multi-book spread, total, ML | Verified |
| nflverse/nfldata `sc_lines.csv` | 2013-2020 | Wednesday SuperContest lines | - | Verified |
| ESPN core API competition odds | per-book from 2012, open/close objects from 2023 | 2023+ | Yes | Verified, validate oddities |
| Action Network public API | consensus from 2019, open book from 2023 | 2023+ | Yes | Verified, unofficial |
| Hugging Face `minmaxxxer` NFL odds | 2025-2026 | Yes, multi-book median, no-vig | Yes | Verified, CC-BY-4.0 |
| nflverse/nfldata `win_totals.csv` | 2003-2020 | Preseason win totals | - | Verified |
| covers.com sportsoddshistory win totals | 2020-2026 | Preseason win totals | - | Verified (HTML) |

Each source defines "open" differently (Sunday night opener, first scrape on Tuesday, Wednesday contest line), so every opening line stores its source and capture time, and models only compare lines captured at the same point in the week.

### 2.5 Weather

| Source | Coverage | Type | Status |
|---|---|---|---|
| IEM MOS archive (mesonet.agron.iastate.edu) | June 2000+ (GFS), NAM 2005+, NBS 2021+ | Archived point forecasts by airport station with run time | Verified, public domain, throttle (429 after ~15 fast requests) |
| Open-Meteo Previous Runs API | 2021-2024+ depending on model | Forecasts at 1-7 day lead | Docs checked |
| Open-Meteo Historical Forecast API | ECMWF 2017+, GFS 2021+ | Stitched near-analysis | Docs checked, rate limited on shared IP |
| NOAA NBM and NDFD on AWS | 2020+ | Gridded forecasts | Verified listing, heavy to process |
| Meteostat hourly | decades | Observations | Verified |
| NCEI GHCNh | decades | Observations | Verified |
| ThompsonJamesBliss/WeatherData | 2000-2021 | Hourly observations per game | Verified |
| NWS api.weather.gov | live | Forecasts | Verified |

Primary choice: IEM MOS forecasts, taking the latest model run before the prediction time, so training uses the same kind of forecast the live system will see.
Observations are used only to measure forecast error and as labels for weather effects, never as pre-game features.

### 2.6 Ratings and benchmarks

| Source | Coverage | Use | Status |
|---|---|---|---|
| ESPN FPI weekly (core API `powerindex`) | 2015 week 10 to 2026 | Benchmark and candidate feature; filter on `lastUpdated` before kickoff | Verified |
| nfelo `output_data/nfelo_games.csv` | 2009-2026, 4,723 games | Pregame Elo, adjustments, win probabilities; benchmark | Verified, no license |
| Sagarin | current week only | Occasional benchmark | Verified, all rights reserved |
| 538 Elo (DataHub mirror) | through 2017 | Historical benchmark only | Verified |

### 2.7 Rosters, talent, continuity

| Source | Coverage | Use | Status |
|---|---|---|---|
| nflverse `load_contracts` | OTC, 53k rows | Cap spent on active vs injured players, QB cost | Verified |
| nflverse `load_draft_picks` | 1980-2026 | Approximate value as a talent prior | Verified |
| nflverse `load_snap_counts` | 2013+ | Snap-weighted returning starters, injury weights | Verified |
| nflverse `load_rosters_weekly`, `load_players` | 1999+ | Experience, IDs across sources | Verified |
| nflverse `load_trades` | 2002-2026 | Midseason roster change flags | Verified |
| nflverse `load_combine` | 2000-2026 | Low value at team level | Verified |

Roster continuity is derived from snap counts and rosters, not from third-party articles.

### 2.8 Venues, travel, coaches, officials

| Source | Use | Status |
|---|---|---|
| nflverse `stadium_id`, roof, surface | Venue key; use "retractable" rather than the observed open/closed state | Verified |
| nflverse/nfldata `airports.csv` | Team lat/long and time zone for travel distance | Verified |
| Wikidata SPARQL | Stadium coordinates (CC0) | Verified |
| Open-Meteo elevation API, `timezonefinder` | Elevation and time zone | To build |
| Kaggle spreadspoke `nfl_stadiums.csv` mirror | Cross-check, incl. weather station IDs | Verified, partial |
| Schedules `home_coach`, `away_coach`, `referee`; `load_officials` (2015+) | Coach records, crew tendencies | Verified |
| Coordinators | No clean free dataset | Gap, hand curation |

## 3. Point-in-time timing per source

| Data | `available_at` |
|---|---|
| Results and all per-game stats | Game end, reliable by the next morning (nflverse updates nightly) |
| Depth chart, weekly | Capture time; treat as Tuesday-Wednesday of game week before 2025 |
| Practice participation | The report date (Wednesday, Thursday, Friday) |
| Final injury status | Friday, or two days before Thursday and Monday games |
| Inactives | Kickoff minus 90 minutes |
| Opening line | The source's capture time |
| Closing line | Kickoff |
| Weather forecast | Model run time |
| ESPN FPI | `lastUpdated` |
| Referee assignment | Unknown before about midweek; treat as unavailable for Wednesday predictions |

## 4. Accuracy validation

Each check runs automatically after ingestion and fails loudly when a threshold is broken.

| Check | Method | Threshold |
|---|---|---|
| Scores | nflverse schedules vs sum of scoring plays in pbp | 100% match |
| Team yards and turnovers | pbp aggregates vs `load_team_stats` | 99%+ within 1 yard and exact on turnovers |
| Closing spread | nflverse vs SBRO vs nfelomarket vs Hugging Face median | 85%+ within 0.5 points; investigate games off by 2+ |
| Opening spread | SBRO vs nfelomarket (2007-2021), Action Network vs ESPN vs Hugging Face (2023+) | Reported, not enforced, because definitions differ |
| Starting QB | depth chart rule vs actual starter | Track the hit rate per season |
| Injuries | nflverse vs nfl.com weekly pages | 99%+ row agreement |
| Weather | MOS forecast vs observed at kickoff | Track error; flag stations with missing runs |
| Venue | neutral-site games | Hand-fix copied roof and surface (e.g. London, Paris, Melbourne) |
| Coverage | rows per season per table | No unexplained drops vs the prior season |

Known data quirks to handle:
- For past games, nflverse lines are closing lines; for upcoming games, they are the current line.
- In nfelomarket data, opening and closing lines are identical for 1999-2006, and its closing totals are copied from nflverse, so they are not an independent check.
- Retractable roof state is only known after the game.
- Some nflverse temp and wind values are missing or late.
- Surface strings need trimming.

## 5. Forward collection: start this week

Several of the best point-in-time sources only exist for the current moment, so collectors run daily from now on to build our own archive:

| Collector | Frequency | Builds |
|---|---|---|
| NFL team site injury reports (all 32 clubs, one platform) | Daily Wed-Sat | Per-day practice participation history |
| ESPN core API injuries | Daily | Timestamped injury history |
| nflverse depth charts and rosters | Daily | Already timestamped, but cache locally |
| Action Network markets with per-book history | Daily, plus pre-kickoff | True opening-to-close line movement, before it is purged after the game |
| ESPN competition odds | Daily | Second line source |
| IEM MOS and NWS forecasts for each stadium | Every 6 hours in game week | Forecast archive at our own prediction times |
| ESPN FPI | Weekly | Benchmark snapshots |

Week 7 kicks off Thursday 2026-10-22, so collectors should be running by Wednesday 2026-10-21 to catch the first full week.

## 6. Search process for remaining gaps

### Open gaps

- Practice participation history before 2026 (only the final weekly status exists for free).
- Opening lines before 2007 and timestamped line history before 2023.
- Preseason win totals 2021 onward in machine-readable form (covers.com HTML covers 2020-2026).
- Offensive and defensive coordinators by team-season.
- Weather forecast archive gaps at international and some neutral sites.

### How to search

1. Check whether nflverse, nfelo, or other open-source NFL projects already publish it (GitHub, the nflverse Discord announcements, Open Source Football).
2. Search Hugging Face datasets and Kaggle (via mirrors when credentials are not available).
3. Check official sources (NFL Football Operations, team sites, NOAA) and their archives.
4. Check Wayback Machine snapshots from a network that can reach web.archive.org (blocked from this environment).
5. Read terms of use before writing any scraper; skip anything that forbids automated access or ML use.

### Scorecard for a new source

A new source is added only if it scores well on all of:
- Coverage: seasons and games covered, and missing rate.
- Point-in-time: whether the capture time is known.
- Accuracy: agreement with an existing source on overlapping fields.
- Stability: maintained, versioned, and likely to exist next season.
- Terms: allows personal research use.
- Marginal value: adds something the catalog above does not already have.

## 7. Deliberately excluded

| Source | Reason |
|---|---|
| Pro-Football-Reference / Sports Reference | Terms forbid automated access and ML use |
| PFF grades, FTN DVOA history | Paid |
| The Odds API history, paid timestamped odds sets | Paid |
| aussportsbetting.com, oddsportal | Bot protection or encrypted feeds; revisit manually only if a gap demands it |
| Kaggle Big Data Bowl tracking | Too sparse (selected weeks only) for team-level prediction |
| Massey | Bot-blocked; low marginal value |

## 8. Order of work

1. Ingest nflverse core tables (schedules, pbp, team stats, injuries, depth charts, rosters, snap counts, QBR) and run the score and stats cross-checks.
2. Start the forward collectors (section 5) before week 7.
3. Build the stadium table and fix neutral-site venues.
4. Ingest betting lines from nfelomarket, SBRO, and nfldata, and run the line cross-checks.
5. Backfill IEM MOS forecasts for every outdoor game since 2000, throttled.
6. Ingest ESPN FPI weekly and nfelo as benchmarks.
7. Ingest contracts, draft picks, and trades for talent and continuity features.
8. Work the open gaps in section 6 only where backtests show the missing data would matter.
