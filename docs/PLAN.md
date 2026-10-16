# NFL Weekly Game Predictor: General Plan

Status: draft, 2026-10-16, mid 2026 season (weeks 1-5 final, week 7 lines posted).

## 1. Goal and honest targets

Predict the straight-up winner of every NFL game each week, with a calibrated win probability, as accurately as possible.

Reference points, all to be recomputed on our own data with our own walk-forward splits:

| Predictor | Straight-up accuracy | Brier |
|---|---|---|
| Coin flip | 50% | 0.250 |
| Always pick home | 53-57% (shrinking) | - |
| Better record | 60-62% | - |
| Elo (538 style) | 63-65% | 0.21-0.22 |
| nfelo (best public model) | 66.5% | - |
| Closing market favorite | 66-68% | 0.205-0.21 |

The closing betting line is the practical ceiling.
Single seasons swing from about 63% to 72% on luck alone, because 272 games carry a standard error of about 3%.
Target for v1: beat Elo by about 2% and land within 1% of the market over many seasons.
Any backtest above about 68% across multiple seasons should be treated as a leakage bug until proven otherwise.

Decisions are made on log loss and Brier, which detect real improvements far sooner than accuracy, while accuracy is always reported because it is the headline goal.

## 2. Data

All sources below were checked from this environment unless marked otherwise.

### Primary: nflverse via `nflreadpy` (free, CC-BY 4.0)

| Loader | Coverage | Use |
|---|---|---|
| `load_schedules` | 1999-2026, 7,548 games | Results, home/away, neutral site, stadium, roof, surface, temp, wind, rest days, div game, spread, total, moneylines, starting QBs, head coaches, referee |
| `load_pbp` | 1999-2026, 372 columns | EPA, success rate, WPA, CPOE; basis for team strength ratings |
| `load_team_stats`, `load_player_stats` | weekly | Aggregates with EPA |
| `load_injuries` | 2009-2026 | Final weekly game status and practice status |
| `load_depth_charts` | 2001+, daily timestamped snapshots since 2025 | Point-in-time starters, especially QB |
| `load_rosters_weekly`, `load_snap_counts` (2013+) | weekly | Snap share for weighting injuries |
| `load_nextgen_stats` (2016+), `load_ftn_charting` (2022+), `load_pfr_advstats` (2018+) | weekly | Later-phase features |
| ESPN QBR CSV (nflverse release) | 2006-2026 | QB quality |

### Live current-week data: ESPN unofficial API (free, undocumented, retry on resets)

- Scoreboard and game summary endpoints give venue, current and opening lines, live injury reports, and ESPN FPI game projections.
- FPI is logged as an extra benchmark to compare against.

### Betting lines

- nflverse `spread_line` is effectively the closing line for past games and the current line for upcoming games.
- Kaggle spreadspoke (spreads since 1979, non-commercial license, needs Kaggle credentials) for deeper history and cross-checks.
- The Odds API (free 500 credits/month, paid for history) is optional.
- From day one we snapshot lines ourselves (Wednesday open-ish and pre-kickoff) so we build a true point-in-time line history.

### Weather

- Meteostat hourly station data for historical backfill (verified).
- ThompsonJamesBliss/WeatherData for per-game hourly weather 2000-2021 (verified).
- Open-Meteo forecast and Historical Forecast API so training can use forecasts as they stood before kickoff; the free archive is rate limited on this shared IP.
- NWS `api.weather.gov` for live US forecasts (verified).

### Hand-built tables

- Stadium table keyed by `stadium_id`: lat/long, time zone, elevation, roof type, surface.
- Corrections for neutral-site games where nflverse copies the home team's roof and surface.

### Not used

- 538 Elo archive (defunct; the mirror ends in 2017). We compute our own Elo.
- PFF (paid, no redistribution).

## 3. Features

Every feature is computed strictly from information available before kickoff and expressed as a home-minus-away difference.

### Core (expected high value)

1. Elo rating with margin-of-victory multiplier, tuned home-field term, and one-third regression to the mean between seasons.
2. Opponent-adjusted EPA per play, offense and defense, exponentially decayed (half-life 4-8 games), garbage time down-weighted, offense weighted above defense, carried across seasons with a prior.
3. QB adjustment: rolling QB rating for the expected starter versus the team's usual starter, using depth chart or injury status as of prediction time.
4. Early-season prior: last season's rating regressed toward the mean, adjusted for QB change, blended into current-season data by about week 6-8.
5. Market spread, used in a separate market-aware model track (see section 4).

### Situational (expected low to medium value)

- Home field as a league-wide era-dependent constant (about 1.5 points now), zero at neutral sites, reduced in division games.
- Rest differential, bye week, short week, post-international-trip.
- Division game flag.
- Travel distance, time zones crossed, West Coast team in early East Coast kickoff or late night body-clock games.
- Pythagorean expectation and point differential (regressed).
- Turnover luck correction (fumble recovery and interception rates versus expected).
- Forecast wind above about 15 mph, cold for dome teams, precipitation, Denver altitude.
- Weighted non-QB injuries (Out/Doubtful weighted by snap share), with OL and CB clusters.

### Features requested by the owner, tested honestly

These are included as candidate features and kept only if they improve walk-forward log loss:

| Feature | Prior expectation | Why |
|---|---|---|
| Season win/loss record | Low | Mostly redundant once point differential and EPA are in |
| Weather performance by team | Low | Weather mostly moves totals; team weather records are small samples |
| Injuries | High for QB, low otherwise | QB gaps are worth 1-7 points; most others are already in the line |
| Home vs away location | Medium | Real but shrinking league-wide effect |
| Team record in a specific stadium | Likely noise | Team-specific home field has near-zero predictive power year to year |
| Head coach record vs opponent | Likely noise | 2-20 game samples, different rosters, confounded by team quality |

### Leakage rules

- Week t features use data through week t-1 only; all rolling stats are shifted by one game.
- Never use post-game columns (`result`, scores, post-game win probability, actual starter learned after kickoff, retractable roof open/closed, observed game-time weather).
- Train with forecast weather, not observed weather, where available.
- Lines are tagged by when they were captured; models trained on closing lines are only compared fairly against closing lines.
- Scalers and imputers fit on training folds only.

## 4. Models

Two tracks are kept separate so we always know whether we add anything beyond copying Vegas:

- Independent track: no market inputs.
- Market-aware track: market spread as a feature.

Progression, each step promoted only if it beats the previous one on walk-forward log loss:

| Version | Model | Notes |
|---|---|---|
| v0 | Baselines + Elo | Home, record, Elo, market favorite. Cheap, strong, handles cold start. |
| v1 | Logistic regression | 5-10 standardized difference features, L2 regularized. Interpretable, well calibrated. |
| v1b | Margin regression | Ridge on point margin, win prob = Phi(margin / sigma). Usually beats direct classification on small samples. |
| v2 | LightGBM | Shallow trees, large min leaf size, early stopping, monotone constraint on Elo diff. |
| v3 | Ensemble | Average of Elo, logistic, margin, GBM, then calibrated on a held-out fold. |

About 6,500 games is a small dataset, so simple models are expected to stay competitive and complexity must earn its place.

## 5. Evaluation

- Walk-forward by season: for each test season S in 2012-2022, train on seasons before S and predict S week by week, updating Elo and refitting weekly.
- Final holdout 2023-2025 (about 850 games) is run once, after design choices are frozen.
- 2026 is a live forward test: predictions are locked before kickoff and scored after.
- Metrics: accuracy, log loss, Brier, calibration curve and expected calibration error, plus splits for weeks 1-4 and playoffs.
- Significance: paired bootstrap over weeks of the log loss difference and McNemar on disagreeing picks; a change is adopted only if its 95% interval excludes zero or it wins in at least 70% of walk-forward seasons.
- Tuning: nested walk-forward with a small search budget, preferring the simplest model within one standard error of the best.
- Ties are excluded from accuracy and scored as 0.5 in log loss.
- Every run logs git sha, config, data hash, seed, metrics, and per-game predictions so any two runs can be compared later without retraining.

## 6. Owner opinion layer ("just a little bit")

### Input

A weekly YAML file generated by `football opine --week N`, optionally in blind mode that hides model numbers to avoid anchoring:

```yaml
games:
  - game: KC@BUF
    pick: BUF
    conf: 4
    why: "KC OL injuries"
teams:
  - team: NYJ
    points: -1.5
    why: "QB looks done"
```

### Blending

Log-odds pooling with a small weight and a hard cap, with L = logit and S = sigmoid:

- Confidence 1-5 maps to p_user = 0.50 + 0.08 * conf (0.58 to 0.90), later refit from the owner's actual hit rate.
- Game delta: d_g = w * (L(p_user) - L(p_model)), default w = 0.12.
- Team nudge delta: d_t = 0.5 * 0.12 * (points_home - points_away).
- D = clip(d_g + d_t, -0.35, +0.35), or +-0.50 at confidence 5.
- p_final = S(L(p_model) + D).

The cap means an opinion can move a probability by at most about 8 percentage points and can never flip a model pick stronger than about 59% (62% at confidence 5).

Example: model KC 70%, owner BUF at confidence 4, blended KC 63.7%, pick stays KC.
Example: model ATL 55%, owner NYG at confidence 5, blended ATL 47.8%, pick flips to NYG.

### Learning whether the opinion helps

- The weight w is refit by minimizing log loss once at least 40 opinion games exist, shrunk toward 0.12 with a prior of 100 games and clamped to [0, 0.30].
- If opinions hurt, w drifts toward zero automatically.
- Model-only predictions are written and hashed before opinions are read and are never rewritten.
- The weekly report shows model-only, blended, and owner-only accuracy, log loss, and Brier, a flip ledger (flips won and lost), and calibration per confidence level.

## 7. Architecture

- Python 3.12, `uv`, `polars`, parquet storage with `duckdb` for ad hoc queries, `scikit-learn`, `lightgbm`, `typer` + `rich` CLI, `pydantic-settings` config, `ruff`, `pyright` strict, `pytest` + `hypothesis`.
- Layout under `src/football/`: `ingest/`, `features/` (with a single `asof.py` that owns every temporal join), `models/`, `backtest/`, `opinions/`, `predict/`, `report/`, plus `config.py`, `schemas.py`, `storage.py`, `runlog.py`, `cli.py`.
- Data flow: raw pulls, then clean tables where every row has a UTC `available_at`, then one point-in-time feature row per `game_id`, then `predictions_model`, then `predictions_blended`, then evaluation.

### Invariants enforced by tests

- No future leakage: truncation test (features built on all data equal features built on data cut at time T for games before T), perturbation test (changing any row available after kickoff never changes that game's features), and a boundary test.
- Deterministic runs with a fixed seed.
- Idempotent ingestion.
- Model-only predictions stored separately from blended ones; final pre-kickoff predictions are immutable.

### CLI

```
football ingest [--sources ...] [--season 2026]
football build-features [--seasons 2015-2026]
football backtest --model {elo,logreg,margin,lgbm} --seasons 2012-2022
football predict --season 2026 --week 7 [--final]
football opine --week 7 [--blind]
football score --season 2026 --week 6
football report [--season 2026 --week 7 | --backtest RUN_ID]
```

## 8. Weekly operating loop (ET)

- Tuesday: ingest results and play-by-play, score last week, report, retrain.
- Wednesday: first predictions for the coming week; owner fills in opinions.
- Wednesday to Saturday: daily refresh of injuries, depth charts, lines, and weather forecasts; re-predict and store each run with its build time.
- About 90 minutes before each kickoff window, after inactives: final refresh and `predict --final` to lock picks.
- After Monday night: score and report.

## 9. Roadmap

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Project scaffold, tooling, CI-style `check` command, AGENTS.md filled in | `uv run` check passes |
| 1 | Ingestion of schedules, pbp, injuries, depth charts, QBR, stadium table; line snapshotting starts | Clean tables with `available_at`, idempotency tests pass |
| 2 | Baselines and Elo, walk-forward backtester, metrics, run log | Baseline leaderboard on 2012-2022 matches published ranges |
| 3 | Core features and v1 logistic + v1b margin models, both tracks | Beats Elo on log loss with significance |
| 4 | Live weekly predictions, opinion layer, weekly report | First locked live week of 2026 scored |
| 5 | Situational and owner-requested features, each tested one at a time | Each kept feature improves walk-forward log loss |
| 6 | LightGBM and ensemble, calibration | Promoted only if it beats v1 |
| 7 | Final 2023-2025 holdout, then ongoing iteration | Holdout result reported once |

Phases 1-4 aim to have live picks running within the 2026 season so the forward test and the opinion weight start accumulating data as early as possible.

## 10. Open questions for the owner

- How do you want to enter opinions: per-game pick with confidence, team point nudges, or both?
- Is the market spread allowed as a model input, or should the headline model stay independent of Vegas?
- Are you willing to pay for any data (The Odds API history, Open-Meteo key, PFF), or free only?
- Should playoffs be predicted with the same model, or handled separately later?
