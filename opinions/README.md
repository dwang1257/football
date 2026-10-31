# Owner opinions

Each week gets one file named `<season>-w<WW>.yaml`, for example `2026-w09.yaml`.
Create it with `football opine --season 2026 --week 9`, or add `--blind` to leave out the model's numbers so they do not anchor you.
The command refuses to overwrite an existing file unless you pass `--force`.

## Format

```yaml
season: 2026
week: 9
games:
  - game: KC@BUF
    kickoff: 2026-11-08T18:00+00:00
    model_prob: 0.30
    pick: BUF
    conf: 4
    why: "KC OL injuries"
  - game: 2026_09_NYG_ATL
    pick: NYG
    prob: 0.60
  - game: LA@SF
    pick: null
    conf: null
    why: null
teams:
  - team: NYJ
    points: -1.5
    why: "QB looks done"
```

- `game` is `AWAY@HOME` or an nflverse game id, and must be a game of that week.
- `pick` is the team you expect to win; team names and common abbreviations are accepted.
- Give exactly one of `conf` (an integer from 1 to 5) or `prob` (your win probability for the pick, between 0 and 1).
- Leave `pick` empty to skip a game.
- `why` is optional and is printed under the game in the weekly table.
- `kickoff` and `model_prob` are written by `football opine` for reference and are ignored when reading; `model_prob` is the model's probability that the home team wins.
- `teams` nudges a team's strength by `points`, at most 3 in either direction, in every game it plays that week.
- Unknown games, unknown teams, teams that do not play that week, and duplicate entries are errors.

## How opinions are used

Confidence maps to a win probability of 0.50 + 0.08 * conf, so conf 1 is 58% and conf 5 is 90%, until the table is refit from your actual hit rate.
Your opinion moves the model's log-odds by a small weight (0.12 by default) and the total move is capped, so it shifts a probability by at most about 8 percentage points.
A model pick of about 59% or more can never be flipped, or about 62% or more when you use conf 5.
For example, if the model gives KC 70% and you pick BUF at conf 4, the blend is KC 63.7%.

## Timing

Opinions count for a game only if this file was last modified before that game's kickoff.
Editing the file after a Thursday kickoff therefore drops your opinions on the Thursday game from later runs, while locked final predictions keep what was entered in time.
Model-only predictions are stored before the file is read and are never changed by it.
