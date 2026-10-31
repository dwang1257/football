from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from football.ingest.weather_base import kickoff_expr
from football.ingest.weather_mos import (
    KNOTS_TO_MPH,
    MODELS,
    Window,
    parse_mos,
    plan_windows,
    select_runs,
)
from football.schemas import UTC_TS, WEATHER_FORECASTS, validate

GFS = next(model for model in MODELS if model.model == "GFS")
NBS = next(model for model in MODELS if model.model == "NBS")
KICKOFF = datetime(2015, 12, 20, 18, tzinfo=UTC)


def games(kickoffs: dict[str, datetime], station: str = "KGRB") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": list(kickoffs),
            "season": [2015] * len(kickoffs),
            "kickoff_utc": list(kickoffs.values()),
            "station": [station] * len(kickoffs),
        },
        schema={
            "game_id": pl.String(),
            "season": pl.Int32(),
            "kickoff_utc": UTC_TS,
            "station": pl.String(),
        },
    )


def mos_csv(runs: list[datetime], hours: range, gust: bool = False) -> bytes:
    header = "runtime,ftime,model,tmp,wsp,p06,p12" + (",gst" if gust else "")
    lines = [header]
    for run in runs:
        for hour in hours:
            valid = run + timedelta(hours=hour)
            p06 = "40.0" if valid.hour % 6 == 0 else ""
            row = f"{run:%Y-%m-%d %H:%M:%S},{valid:%Y-%m-%d %H:%M:%S},GFS,{30 + hour},10,{p06},"
            lines.append(row + (",20" if gust else ""))
    return "\n".join(lines).encode()


def test_parse_mos_handles_midnight_dates_and_missing_columns() -> None:
    payload = b"runtime,ftime,model,tmp,wsp\n2000-12-24,2000-12-24 03:00:00,AVN,30, 5\n"
    frame = parse_mos(payload)
    assert frame["runtime"].to_list() == [datetime(2000, 12, 24, tzinfo=UTC)]
    assert frame["wsp"].to_list() == [5.0]
    assert frame["gst"].null_count() == 1
    assert parse_mos(b"").is_empty()
    assert parse_mos(b"<html>error</html>").is_empty()


def test_select_runs_keeps_point_in_time_runs_nearest_kickoff() -> None:
    runs = [KICKOFF - timedelta(hours=h) for h in (120, 96, 72, 30, 12, 6, 4)]
    mos = parse_mos(mos_csv(runs, range(3, 85, 3)))
    rows = validate(select_runs(mos, games({"g": KICKOFF}), GFS), WEATHER_FORECASTS)
    run_times = rows["run_time"].to_list()
    assert KICKOFF - timedelta(hours=120) not in run_times
    assert KICKOFF - timedelta(hours=4) not in run_times
    assert KICKOFF - timedelta(hours=6) in run_times
    assert KICKOFF - timedelta(hours=96) not in run_times
    assert (rows["valid_time"] == KICKOFF).all()
    assert (rows["available_at"] < KICKOFF).all()
    assert (rows["available_at"] == rows["run_time"] + GFS.lag).all()
    assert rows["wind_mph"][0] == pytest.approx(10 * KNOTS_TO_MPH)
    assert rows["precip_prob"].to_list() == [0.4] * rows.height
    assert rows["wind_gust_mph"].null_count() == rows.height


def test_select_runs_requires_valid_time_within_three_hours() -> None:
    kickoff = datetime(2015, 12, 20, 21, 25, tzinfo=UTC)
    run = datetime(2015, 12, 19, 6, tzinfo=UTC)
    mos = parse_mos(mos_csv([run], range(6, 80, 24)))
    assert select_runs(mos, games({"g": kickoff}), GFS).is_empty()
    mos = parse_mos(mos_csv([run], range(3, 80, 3)))
    rows = select_runs(mos, games({"g": kickoff}), NBS)
    assert rows["valid_time"].to_list() == [datetime(2015, 12, 20, 21, tzinfo=UTC)]
    assert rows["source"].to_list() == ["mos_nbs"]


def test_select_runs_converts_gusts_and_separates_games() -> None:
    other = KICKOFF + timedelta(days=1)
    runs = [KICKOFF - timedelta(hours=12), other - timedelta(hours=12)]
    mos = parse_mos(mos_csv(runs, range(3, 40, 3), gust=True))
    rows = select_runs(mos, games({"a": KICKOFF, "b": other}), NBS)
    assert rows.group_by("game_id").len().sort("game_id").rows() == [("a", 1), ("b", 2)]
    assert rows["wind_gust_mph"][0] == pytest.approx(20 * KNOTS_TO_MPH)


def test_plan_windows_season_and_month_granularity() -> None:
    done = games({"a": KICKOFF, "b": KICKOFF + timedelta(days=30)})
    windows = plan_windows(done, datetime(2016, 6, 1, tzinfo=UTC))
    assert [(w.label, w.settled) for w, _ in windows] == [("2015", True)]
    window = windows[0][0]
    assert window.start == KICKOFF - timedelta(hours=96)
    assert window.end == KICKOFF + timedelta(days=30, hours=1)
    now = datetime(2015, 12, 30, 18, 30, tzinfo=UTC)
    later = datetime(2016, 1, 2, 18, tzinfo=UTC)
    live = games({"a": KICKOFF, "b": later, "c": KICKOFF + timedelta(days=60)})
    planned = {w.label: (w, group) for w, group in plan_windows(live, now)}
    assert set(planned) == {"2015-12", "2016-01"}
    assert planned["2015-12"][0].settled
    assert not planned["2016-01"][0].settled
    assert planned["2016-01"][0].end == datetime(2015, 12, 30, 19, tzinfo=UTC)
    assert planned["2016-01"][1]["game_id"].to_list() == ["b"]


def test_window_cache_name_is_stable() -> None:
    window = Window("KGRB", "2015", KICKOFF, KICKOFF + timedelta(days=1), True)
    assert window.cache_name() == "KGRB_2015_2015122018_2015122118.csv"


def test_kickoff_expr_converts_eastern_time() -> None:
    frame = pl.DataFrame(
        {
            "gameday": ["2015-07-01", "2015-12-20", "2015-12-21"],
            "gametime": ["20:30", "13:00", None],
        }
    )
    kickoffs = frame.select(kickoff_expr())["kickoff_utc"].to_list()
    assert kickoffs == [
        datetime(2015, 7, 2, 0, 30, tzinfo=UTC),
        datetime(2015, 12, 20, 18, tzinfo=UTC),
        datetime(2015, 12, 21, 18, tzinfo=UTC),
    ]
