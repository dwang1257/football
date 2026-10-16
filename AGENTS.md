# Agent guide

Football: TODO one-line description of what this project does and who it is for.
Stack: TODO language, runtime, frameworks, and key external tools.

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

- TODO where the code lives and how it is split into modules.
- TODO the entry point and what it wires together.
- TODO external APIs or data sources, with links to their docs; read them rather than guessing field names.
- Rules that must not break:
  - TODO invariants that must hold, such as never auto-retrying billed or side-effectful calls.
- Use `rg` to find code.

## Commands

Run from the repo root.

- TODO install command - install dependencies
- TODO build command - build the project
- TODO run command - run locally
- TODO test command - run the test suite
- TODO check command - format check, lint, and typecheck together

## Done means

- Bug fixes: reproduce first, verify the same flow after.
- Lint, typecheck, and tests pass.
- Report what changed and what was actually verified, and say plainly what was not exercised.

## Commit & Pull Request Guidelines

Follow the concise conventional style: `feat:`, `fix:`, or `chore:` followed by an imperative summary.
Keep commits scoped to one coherent change and exclude local files such as `.DS_Store`.
Pull requests should explain the user-visible outcome, note important implementation choices, and list verification performed.

## Gotchas

- TODO non-obvious setup steps, env vars, rate limits, or costs.
- New env vars go in the config schema, `.env.example`, and the README.
