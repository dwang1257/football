import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from football.opinions.blend import BlendParams
from football.opinions.files import OpinionError, load_opinions, opinion_path
from football.opinions.template import write_template
from football.predict.store import write_model_rows
from football.schemas import UTC_TS

PARAMS = BlendParams.from_settings()
BEFORE_ALL = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


def write_opinions(text: str, modified: datetime = BEFORE_ALL) -> Path:
    path = opinion_path(2026, 9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (modified.timestamp(), modified.timestamp()))
    return path


def load(games: pl.DataFrame):
    return load_opinions(2026, 9, games, PARAMS)


def test_missing_file_means_no_opinions(repo: Path, games: pl.DataFrame) -> None:
    assert load(games) is None


def test_resolves_games_teams_and_nudges(repo: Path, games: pl.DataFrame) -> None:
    write_opinions(
        """
games:
  - game: KC@BUF
    pick: Bills
    conf: 4
    why: "KC OL injuries"
  - game: 2026_09_NYG_ATL
    pick: nyg
    prob: 0.6
  - game: LAR@SF
teams:
  - team: SF
    points: -1.5
    why: "QB looks done"
  - team: KC
    points: 1
"""
    )
    opinions = load(games)
    assert opinions is not None
    assert opinions.entered_at == BEFORE_ALL
    buf = opinions.games["2026_09_KC_BUF"]
    assert (buf.pick, buf.conf, buf.why) == ("BUF", 4, "KC OL injuries")
    assert buf.user_prob == pytest.approx(0.82)
    nyg = opinions.games["2026_09_NYG_ATL"]
    assert nyg.pick_prob == pytest.approx(0.6)
    assert nyg.user_prob == pytest.approx(0.4)
    assert "2026_09_LA_SF" not in opinions.games
    assert opinions.nudges == {"2026_09_KC_BUF": -1.0, "2026_09_LA_SF": -1.5}
    assert opinions.team_whys == {"SF": "QB looks done"}
    assert opinions.late == ()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("games:\n  - {game: KC@BUF, pick: BUF, conf: 6}", "games.0.conf"),
        ("games:\n  - {game: KC@BUF, pick: BUF, conf: 0}", "games.0.conf"),
        ("games:\n  - {game: KC@BUF, pick: BUF, conf: 3, prob: 0.7}", "exactly one of conf"),
        ("games:\n  - {game: KC@BUF, pick: BUF}", "exactly one of conf"),
        ("games:\n  - {game: KC@BUF, conf: 3}", "without a pick"),
        ("games:\n  - {game: KC@BUF, pick: BUF, prob: 1.2}", "games.0.prob"),
        ("games:\n  - {game: KC@BUF, pick: BUF, confidence: 3}", "games.0.confidence"),
        ("teams:\n  - {team: NYJ, points: 3.5}", "teams.0.points"),
        ("games:\n  - {game: KC@XYZ, pick: KC, conf: 2}", "unknown team 'XYZ'"),
        ("games:\n  - {game: DAL@NYG, pick: DAL, conf: 2}", "no game DAL@NYG in 2026 week 9"),
        ("games:\n  - {game: BUF@KC, pick: KC, conf: 2}", "did you mean KC@BUF?"),
        ("games:\n  - {game: 2026_08_KC_BUF, pick: KC, conf: 2}", "unknown game"),
        ("games:\n  - {game: KC@BUF, pick: DAL, conf: 2}", "pick DAL is not playing in KC@BUF"),
        (
            "games:\n  - {game: KC@BUF, pick: KC, conf: 2}\n  - {game: KC@BUF}",
            "listed more than once",
        ),
        ("teams:\n  - {team: Jets, points: 1}", "NYJ does not play in 2026 week 9"),
        ("teams:\n  - {team: KC, points: 1}\n  - {team: kc, points: 2}", "listed more than once"),
        ("season: 2025\ngames: []", "season 2025"),
        ("games: [unclosed", "not valid YAML"),
        ("- just a list", "expected a mapping"),
    ],
)
def test_invalid_files_raise_clear_errors(
    repo: Path, games: pl.DataFrame, text: str, message: str
) -> None:
    write_opinions(text)
    with pytest.raises(OpinionError, match=message.replace("?", r"\?").replace(".", r"\.")):
        load(games)


def test_new_orleans_is_not_read_as_a_yaml_boolean(repo: Path) -> None:
    games = pl.DataFrame(
        {
            "game_id": ["2026_09_NO_ATL"],
            "season": [2026],
            "week": [9],
            "away_team": ["NO"],
            "home_team": ["ATL"],
            "kickoff_utc": [datetime(2026, 11, 8, 18, 0, tzinfo=UTC)],
        }
    )
    write_opinions(
        "games:\n  - {game: NO@ATL, pick: NO, conf: 2}\nteams:\n  - {team: NO, points: 1}"
    )
    opinions = load(games)
    assert opinions is not None
    assert opinions.games["2026_09_NO_ATL"].pick == "NO"
    assert opinions.nudges == {"2026_09_NO_ATL": -1.0}


def test_late_entries_are_ignored_per_game(repo: Path, games: pl.DataFrame) -> None:
    after_thursday = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    write_opinions(
        """
games:
  - {game: LA@SF, pick: LA, conf: 3}
  - {game: KC@BUF, pick: KC, conf: 2}
teams:
  - {team: SF, points: 2}
""",
        modified=after_thursday,
    )
    opinions = load(games)
    assert opinions is not None
    assert opinions.late == ("2026_09_LA_SF",)
    assert list(opinions.games) == ["2026_09_KC_BUF"]
    assert opinions.nudges == {}
    warnings = opinions.warnings()
    assert len(warnings) == 1
    assert "after kickoff of 2026_09_LA_SF" in warnings[0]


def test_template_lists_games_with_model_probs(repo: Path, stored_games: pl.DataFrame) -> None:
    write_model_rows(
        pl.DataFrame(
            {
                "run_id": ["r1", "r2"],
                "game_id": ["2026_09_KC_BUF", "2026_09_KC_BUF"],
                "model": ["elo", "elo"],
                "track": ["independent", "independent"],
                "home_win_prob": [0.4, 0.3],
                "home_margin": [None, None],
                "cutoff": [BEFORE_ALL, BEFORE_ALL],
                "created_at": [BEFORE_ALL, datetime(2026, 11, 5, tzinfo=UTC)],
                "is_final": [False, False],
            },
            schema_overrides={"home_margin": pl.Float64, "cutoff": UTC_TS, "created_at": UTC_TS},
        )
    )
    path = write_template(2026, 9, blind=False, force=False)
    document = yaml.safe_load(path.read_text())
    assert [g["game"] for g in document["games"]] == ["LA@SF", "KC@BUF", "NYG@ATL"]
    assert document["games"][1]["model_prob"] == 0.3
    assert "model_prob" not in document["games"][0]
    assert load(stored_games) is not None


def test_blind_template_hides_model_and_refuses_overwrite(
    repo: Path, stored_games: pl.DataFrame
) -> None:
    path = write_template(2026, 9, blind=True, force=False)
    assert "model_prob" not in path.read_text()
    with pytest.raises(OpinionError, match="already exists"):
        write_template(2026, 9, blind=True, force=False)
    write_template(2026, 9, blind=True, force=True)
