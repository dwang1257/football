import os
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from conftest import BASE, Fakes
from football import storage
from football.opinions import files
from football.opinions.files import OpinionError, opinion_path
from football.predict import run
from football.predict.run import PredictRequest, run_predict
from football.predict.store import FinalLockError, PredictError, write_model_rows

OPINIONS = """
games:
  - {game: KC@BUF, pick: BUF, conf: 4, why: "KC OL injuries"}
  - {game: NYG@ATL, pick: NYG, conf: 5}
teams:
  - {team: SF, points: 1, why: "healthy again"}
"""
THURSDAY_LOCK = BASE + timedelta(days=1, hours=12)
SUNDAY_LOCK = BASE + timedelta(days=4, hours=4, minutes=30)


def write_opinions(text: str = OPINIONS, modified: datetime = BASE) -> Path:
    path = opinion_path(2026, 9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (modified.timestamp(), modified.timestamp()))
    return path


def request(final: bool = False) -> PredictRequest:
    return PredictRequest(
        season=2026, week=9, model="fake", params={"track": "market"}, final=final
    )


def by_game(df: pl.DataFrame, game_id: str) -> dict:
    return df.filter(pl.col("game_id") == game_id).row(0, named=True)


def test_trains_before_cutoff_and_blends_opinions(fakes: Fakes) -> None:
    write_opinions()
    outcome = run_predict(request(), BASE)
    model = fakes.models[0]
    assert model.train is not None
    assert sorted(model.train["game_id"].to_list()) == ["2026_08_DAL_PHI", "2026_08_MIA_NE"]
    assert fakes.feature_calls == [(BASE, None)]
    assert by_game(outcome.model_rows, "2026_09_KC_BUF")["track"] == "market"
    buf = by_game(outcome.blended_rows, "2026_09_KC_BUF")
    assert buf["model_prob"] == pytest.approx(0.30)
    assert 1.0 - buf["blended_prob"] == pytest.approx(0.637, abs=5e-4)
    assert (buf["user_pick"], buf["user_conf"], buf["flipped"]) == ("BUF", 4, False)
    atl = by_game(outcome.blended_rows, "2026_09_NYG_ATL")
    assert atl["blended_prob"] == pytest.approx(0.478, abs=5e-4)
    assert atl["flipped"]
    sf = by_game(outcome.blended_rows, "2026_09_LA_SF")
    assert sf["team_nudge_points"] == 1.0
    assert sf["user_prob"] is None
    assert sf["blended_prob"] > sf["model_prob"]
    assert outcome.locked == ()
    assert outcome.exports == ()


def test_model_rows_are_written_before_opinions_and_never_changed(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_opinions()
    seen: list[pl.DataFrame] = []
    real_load = files.load_opinions

    def spy(*args, **kwargs):
        seen.append(storage.read("predictions", "model"))
        assert not storage.exists("predictions", "blended")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(run, "load_opinions", spy)
    outcome = run_predict(request(), BASE)
    assert len(seen) == 1
    assert seen[0].filter(pl.col("run_id") == outcome.run_id).height == 3
    after = storage.read("predictions", "model")
    assert after.equals(seen[0])
    assert after["home_win_prob"].sort().to_list() == pytest.approx([0.30, 0.55, 0.62])


def test_final_locks_only_games_in_window_and_refuses_relock(fakes: Fakes, repo: Path) -> None:
    write_opinions()
    first = run_predict(request(final=True), THURSDAY_LOCK)
    assert first.locked == ("2026_09_LA_SF",)
    assert first.refused == ()
    model_csv = repo / "predictions" / "2026-w09-model.csv"
    blended_csv = repo / "predictions" / "2026-w09-blended.csv"
    assert pl.read_csv(model_csv)["game_id"].to_list() == ["2026_09_LA_SF"]
    assert pl.read_csv(blended_csv)["game_id"].to_list() == ["2026_09_LA_SF"]

    again = run_predict(request(final=True), THURSDAY_LOCK + timedelta(minutes=5))
    assert again.locked == ()
    assert again.refused == ("2026_09_LA_SF",)
    assert pl.read_csv(model_csv).height == 1

    locked_row = by_game(first.model_rows, "2026_09_LA_SF")
    with pytest.raises(
        FinalLockError, match="refusing to overwrite final prediction for 2026_09_LA_SF"
    ):
        write_model_rows(pl.DataFrame([{**locked_row, "run_id": "manual"}]))

    sunday = run_predict(request(final=True), SUNDAY_LOCK)
    assert sunday.locked == ("2026_09_KC_BUF",)
    exported = pl.read_csv(blended_csv)
    assert exported["game_id"].to_list() == ["2026_09_LA_SF", "2026_09_KC_BUF"]
    assert exported.filter(pl.col("game_id") == "2026_09_KC_BUF")["user_pick"].item() == "BUF"
    finals = storage.read("predictions", "model").filter(pl.col("is_final"))
    assert sorted(finals["game_id"].to_list()) == ["2026_09_KC_BUF", "2026_09_LA_SF"]
    assert finals["game_id"].is_unique().all()


def test_broken_opinion_file_after_lock_is_blended_on_rerun(fakes: Fakes) -> None:
    write_opinions("games:\n  - {game: LA@SF, pick: LA, conf: 9}")
    with pytest.raises(OpinionError, match="conf"):
        run_predict(request(final=True), THURSDAY_LOCK)
    locked = storage.read("predictions", "model").filter(pl.col("is_final"))
    assert locked["game_id"].to_list() == ["2026_09_LA_SF"]
    assert not storage.exists("predictions", "blended")

    write_opinions("games:\n  - {game: LA@SF, pick: LA, conf: 3}", modified=BASE)
    rerun = run_predict(request(final=True), THURSDAY_LOCK + timedelta(minutes=10))
    assert rerun.refused == ("2026_09_LA_SF",)
    blended = storage.read("predictions", "blended").filter(pl.col("is_final"))
    assert blended["run_id"].to_list() == locked["run_id"].to_list()
    assert blended["user_pick"].to_list() == ["LA"]


def test_opinions_entered_after_kickoff_are_ignored(fakes: Fakes) -> None:
    write_opinions("games:\n  - {game: LA@SF, pick: LA, conf: 3}")
    run_predict(request(final=True), THURSDAY_LOCK)
    write_opinions(modified=THURSDAY_LOCK + timedelta(days=1))
    outcome = run_predict(request(), THURSDAY_LOCK + timedelta(days=1, minutes=1))
    assert outcome.opinions is not None
    assert outcome.opinions.late == ("2026_09_LA_SF",)
    assert "after kickoff of 2026_09_LA_SF" in outcome.opinions.warnings()[0]
    assert set(outcome.model_rows["game_id"].to_list()) == {"2026_09_KC_BUF", "2026_09_NYG_ATL"}
    assert by_game(outcome.blended_rows, "2026_09_KC_BUF")["user_pick"] == "BUF"


def test_every_game_started_is_an_error(fakes: Fakes) -> None:
    with pytest.raises(PredictError, match="kicked off"):
        run_predict(request(), BASE + timedelta(days=10))
