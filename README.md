# Football

Predicts the winner of every NFL game each week, with a calibrated win probability, from free data.
The owner's weekly opinions are blended in lightly and tracked so their value can be measured.

See [docs/PLAN.md](docs/PLAN.md) for the modeling plan and [docs/DATA_PLAN.md](docs/DATA_PLAN.md) for the data sources.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run `uv sync`.
Copy `.env.example` to `.env` to override any setting; every variable uses the `FOOTBALL_` prefix.

| Variable | Default | Meaning |
|---|---|---|
| `FOOTBALL_DATA_DIR` | `data` | Where parquet layers (raw, clean, features, predictions, runs, archive) live |
| `FOOTBALL_REPO_DIR` | `.` | Repo root for committed opinions, locked predictions, and reports |
| `FOOTBALL_SEED` | `20261016` | Seed for every random process |
| `FOOTBALL_HTTP_TIMEOUT_S` | `30` | HTTP timeout in seconds |
| `FOOTBALL_USER_AGENT` | project URL | User-Agent sent to data sources |
| `FOOTBALL_OPINION_WEIGHT` | `0.12` | Default weight of a per-game opinion in log-odds |
| `FOOTBALL_OPINION_TEAM_WEIGHT` | `0.5` | Damping of team point nudges |
| `FOOTBALL_OPINION_CAP` | `0.35` | Maximum opinion shift in log-odds |
| `FOOTBALL_OPINION_CAP_MAX_CONF` | `0.5` | Maximum shift at confidence 5 |

## First run

```
uv run football ingest --seasons 1999-2026
uv run football validate
uv run football build-features
uv run football backtest --model elo
uv run football backtest --model ensemble
uv run football leaderboard
```

The first ingest downloads about 1 GB of play-by-play data and backfills archived weather forecasts, which takes a couple of hours because the forecast archive is rate limited.
Later runs reuse the cache under `data/raw/`; pass `--refresh` to re-download the current season.

## Weekly loop

| When (ET) | Command |
|---|---|
| Tuesday | `uv run football ingest --seasons 2026 --refresh`, then `uv run football score` and `uv run football report` |
| Wednesday | `uv run football build-features`, `uv run football predict`, then `uv run football opine` and edit `opinions/<season>-w<WW>.yaml` |
| Wednesday to Saturday | `uv run football ingest --seasons 2026 --refresh` and `uv run football predict` as news arrives |
| About 90 minutes before each kickoff window | `uv run football predict --final` to lock picks |

`predict --final` locks only games kicking off within `--lock-hours` (default 3), so each kickoff window is locked separately.
Locked predictions are exported to `predictions/` and weekly reports to `reports/`, and both are committed.
The opinion file format is described in [opinions/README.md](opinions/README.md).

## Daily collection

Some data only exists in the present: per-day practice reports, line movement, and forecasts.
`.github/workflows/collect.yml` runs `football collect` daily and before kickoff windows and commits the snapshots to the `data-archive` branch.
Scheduled workflows only run from the default branch, so collection starts once this code is merged.
To use the snapshots locally, check out the `data-archive` branch somewhere and run `uv run football archive-import <dir>`, then `uv run football ingest --sources archive_practice,archive_lines,archive_fpi,archive_forecasts`.

## Models

| Name | Description |
|---|---|
| `home`, `record`, `elo`, `market` | Baselines |
| `matchup` | Base model: each team's expected stats from its offense against the opponent's defense, turned into points and simulated |
| `logreg`, `margin` | Logistic regression on win, ridge regression on margin |
| `lgbm` | Shallow LightGBM |
| `ensemble` | Average of elo, matchup, logreg, and margin |

Every model has an independent track and, where it makes sense, a market track (`--track market`) that may also use betting lines.
Seasons 2023-2025 are reserved for a one-time holdout (`football holdout`), so backtests default to 2012-2022.

## Development

`uv run ruff format --check && uv run ruff check && uv run pyright && uv run pytest` runs every check.
Network smoke tests are excluded by default; run them with `uv run pytest -m network`.
