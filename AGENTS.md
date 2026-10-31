# Agent guide

Football: predicts the winner of every NFL game each week from free data, blending in the owner's opinions lightly; for personal use.
Stack: Python 3.12+, uv, polars, parquet (duckdb for ad hoc SQL), scikit-learn, LightGBM, typer + rich CLI, pydantic-settings, ruff, pyright, pytest + hypothesis; GitHub Actions for daily data collection.

## General Guidelines

- Do not write any code comments.
- It is okay to delete code that is not needed; do not strive to add more lines.
- Never use the em dash (U+2014); use a plain hyphen "-" instead.
- Never add an agent name as co-author in commit messages.
- Never manually modify CHANGELOG.md files or any files marked as auto-generated.
- When writing or substantially editing long Markdown files, put each full sentence on its own line.
- When making technical decisions, give little weight to development cost and prefer quality, simplicity, robustness, scalability, and long term maintainability.
- When fixing bugs, start by reproducing the bug end to end, as close as possible to how a user would hit it.
- When testing the product end to end, be picky about the UI and fix anything that clearly looks off, even if it is unrelated to the current task.
- Hold the same bar for lint errors, test failures, and flaky tests: fix them even if the current task did not cause them.

## Orientation

- Plans: `docs/PLAN.md` (models, evaluation, opinions, architecture) and `docs/DATA_PLAN.md` (sources, timing, validation).
- Code lives in `src/football/`:
  - `ingest/` downloads sources into clean parquet tables; each module exposes `SOURCES` and `CHECKS`.
  - `collect/` snapshots data that only exists in the present (injury reports, line movement, forecasts) into `data/archive/`.
  - `features/` builds one point-in-time feature row per game; `features/asof.py` owns every time-based filter and join.
  - `models/` holds the baselines, the matchup simulation base model, regressions, LightGBM, and the ensemble behind `models/registry.py`.
  - `backtest/` runs walk-forward evaluation, metrics, significance tests, and the leaderboard; `runlog.py` records every run.
  - `opinions/`, `predict/`, `report/` handle owner opinions, live weekly predictions with locking, scoring, and reports.
  - `schemas.py` defines the shared table contracts, `storage.py` the parquet layers, `teams.py` team code normalization.
- The entry point is `src/football/cli.py`, which registers each area's `commands.register(app)`.
- Data sources: nflverse via nflreadpy (https://nflreadpy.nflverse.com, data dictionaries at https://nflreadr.nflverse.com/articles), ESPN unofficial APIs, nfelo market data, sportsbookreviewsonline, IEM MOS forecast archive (https://mesonet.agron.iastate.edu/mos/), NWS (https://www.weather.gov/documentation/services-web-api), Open-Meteo (https://open-meteo.com/en/docs); read their docs rather than guessing field names.
- Rules that must not break:
  - Every clean row has `available_at`, the UTC time it was first publicly known, and nothing at or after a game's cutoff may influence that game's features.
  - All time-based filtering in features goes through `features/asof.py`.
  - Backtests are walk-forward only; seasons 2023-2025 are a one-time holdout (`football holdout`).
  - Model-only predictions are written before opinions are read and are never modified; final predictions are immutable.
  - `home_spread` and `spread_line` are positive when the home team is favored.
  - Do not scrape Pro-Football-Reference; its terms forbid automated access and ML use.
- Use `rg` to find code.

## Commands

Run from the repo root.

- `uv sync` - install dependencies
- No build step.
- `uv run football --help` - run locally; the weekly flow is `ingest`, `build-features`, `predict`, `opine`, `score`, `report`
- `uv run pytest` - run the test suite (`-m network` for network smoke tests)
- `uv run ruff format --check && uv run ruff check && uv run pyright && uv run pytest` - format check, lint, typecheck, and tests together

## Done means

- Bug fixes: reproduce first, verify the same flow after.
- Lint, typecheck, and tests pass.
- Report what changed and what was actually verified, and say plainly what was not exercised.

## Commit & Pull Request Guidelines

Follow the concise conventional style: `feat:`, `fix:`, or `chore:` followed by an imperative summary.
Keep commits scoped to one coherent change and exclude local files such as `.DS_Store`.
Pull requests should explain the user-visible outcome, note important implementation choices, and list verification performed.

## Gotchas

- `data/` is gitignored; rebuild it with `uv run football ingest` (pbp for all seasons downloads about 1 GB).
- Env vars use the `FOOTBALL_` prefix (see `src/football/config.py` and `.env.example`).
- The IEM MOS archive returns HTTP 429 after about 15 fast requests; ingestion throttles it.
- ESPN and Action Network endpoints are unofficial and can change without notice.
- Action Network purges line history after a game, so line movement only exists if collected beforehand.
- Scheduled GitHub Actions only run from the default branch; collected snapshots live on the `data-archive` branch.
- Owner opinions (`opinions/`), locked predictions (`predictions/`), and reports (`reports/`) are committed to the repo.
- New env vars go in the config schema, `.env.example`, and the README.
