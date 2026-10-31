from datetime import UTC, datetime

import polars as pl
import pytest

from football import storage
from football.ingest.base import IngestOptions
from football.ingest.venues import (
    GAME_VENUES,
    ROOF_TYPES,
    game_venues_frame,
    load_overrides,
    load_stadiums,
    suspicious_neutral_games,
    team_homes_frame,
)
from football.schemas import STADIUMS, validate


def schedule(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "season": 2024,
        "game_type": "REG",
        "gameday": "2024-10-06",
        "gametime": "13:00",
        "location": "Home",
        "surface": "grass",
        "roof": "outdoors",
    }
    return pl.DataFrame(
        [defaults | row for row in rows],
        schema={
            "game_id": pl.String(),
            "season": pl.Int32(),
            "game_type": pl.String(),
            "gameday": pl.String(),
            "gametime": pl.String(),
            "home_team": pl.String(),
            "away_team": pl.String(),
            "location": pl.String(),
            "stadium_id": pl.String(),
            "surface": pl.String(),
            "roof": pl.String(),
        },
    )


def stadiums() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stadium_id": ["AAA00", "LON00", "BBB00"],
            "name": ["A Field", "Wembley", "B Dome"],
            "lat": [40.0, 51.5, 30.0],
            "lon": [-80.0, -0.28, -90.0],
            "elevation_m": [100.0, 60.0, 5.0],
            "tz": ["America/New_York", "Europe/London", "America/Chicago"],
            "roof_type": ["outdoors", "outdoors", "dome"],
            "surface": ["fieldturf", "grass", "sportturf"],
            "weather_station": ["KAAA", "EGLL", "KBBB"],
            "is_international": [False, True, False],
        }
    )


def overrides(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [g for g, _ in rows],
            "stadium_id": [s for _, s in rows],
            "note": ["x"] * len(rows),
        },
        schema={"game_id": pl.String(), "stadium_id": pl.String(), "note": pl.String()},
    )


def test_static_stadiums_are_complete() -> None:
    frame = validate(load_stadiums(), STADIUMS)
    assert frame.height >= 69
    assert set(frame["roof_type"]) <= set(ROOF_TYPES)
    for column in ("lat", "lon", "tz", "elevation_m", "weather_station", "surface"):
        assert frame[column].null_count() == 0, column
    assert frame["lat"].is_between(-60, 70).all()
    international = set(frame.filter(pl.col("is_international"))["stadium_id"])
    assert {"LON00", "LON01", "LON02", "MEX00", "MUN01", "FRA00", "DUB00", "BER00"} <= international
    assert set(load_overrides()["stadium_id"]) <= set(frame["stadium_id"])


def test_game_venues_kickoff_and_lead() -> None:
    games = schedule(
        [
            {"game_id": "g1", "home_team": "AAA", "away_team": "CCC", "stadium_id": "AAA00"},
            {
                "game_id": "g2",
                "home_team": "AAA",
                "away_team": "CCC",
                "stadium_id": "AAA00",
                "gameday": "2024-12-01",
                "gametime": None,
            },
        ]
    )
    venues = game_venues_frame(games, stadiums(), overrides([]))
    assert venues.schema == pl.Schema(GAME_VENUES.columns)
    kickoffs = dict(zip(venues["game_id"], venues["kickoff_utc"], strict=True))
    assert kickoffs["g1"] == datetime(2024, 10, 6, 17, tzinfo=UTC)
    assert kickoffs["g2"] == datetime(2024, 12, 1, 18, tzinfo=UTC)
    lead = venues.select((pl.col("kickoff_utc") - pl.col("available_at")).dt.total_days())
    assert lead.to_series().to_list() == [7, 7]


def test_overrides_and_neutral_surface_corrections() -> None:
    games = schedule(
        [
            {
                "game_id": "h1",
                "home_team": "AAA",
                "away_team": "CCC",
                "stadium_id": "AAA00",
                "surface": "a_turf",
            },
            {
                "game_id": "h2",
                "home_team": "AAA",
                "away_team": "DDD",
                "stadium_id": "AAA00",
                "surface": "a_turf ",
            },
            {
                "game_id": "n1",
                "home_team": "EEE",
                "away_team": "CCC",
                "stadium_id": "AAA00",
                "location": "Neutral",
                "surface": "grass",
            },
            {
                "game_id": "x1",
                "home_team": "AAA",
                "away_team": "FFF",
                "stadium_id": "AAA00",
                "location": "Neutral",
                "surface": "",
            },
            {
                "game_id": "d1",
                "home_team": "BBB",
                "away_team": "CCC",
                "stadium_id": "BBB00",
                "surface": "",
            },
        ]
    )
    venues = game_venues_frame(games, stadiums(), overrides([("x1", "LON00")]))
    by_id = {row["game_id"]: row for row in venues.iter_rows(named=True)}
    assert by_id["h1"]["surface"] == "a_turf"
    assert by_id["h2"]["surface"] == "a_turf"
    assert by_id["n1"]["surface"] == "a_turf"
    assert by_id["x1"]["stadium_id"] == "LON00"
    assert by_id["x1"]["nflverse_stadium_id"] == "AAA00"
    assert by_id["x1"]["corrected"]
    assert by_id["x1"]["is_international"]
    assert by_id["x1"]["surface"] == "grass"
    assert by_id["x1"]["tz"] == "Europe/London"
    assert by_id["d1"]["surface"] == "sportturf"
    assert by_id["d1"]["roof_type"] == "dome"


def test_team_homes_and_suspicious_neutral_games() -> None:
    games = schedule(
        [
            {"game_id": "a", "home_team": "AAA", "away_team": "CCC", "stadium_id": "AAA00"},
            {"game_id": "b", "home_team": "AAA", "away_team": "DDD", "stadium_id": "AAA00"},
            {"game_id": "c", "home_team": "AAA", "away_team": "EEE", "stadium_id": "BBB00"},
            {
                "game_id": "d",
                "home_team": "AAA",
                "away_team": "FFF",
                "stadium_id": "BBB00",
                "location": "Neutral",
            },
            {
                "game_id": "e",
                "home_team": "AAA",
                "away_team": "GGG",
                "stadium_id": "AAA00",
                "location": "Neutral",
            },
        ]
    )
    venues = game_venues_frame(games, stadiums(), overrides([]))
    homes = team_homes_frame(venues, stadiums())
    assert homes.select("season", "team", "stadium_id").rows() == [(2024, "AAA", "AAA00")]
    assert homes["tz"].to_list() == ["America/New_York"]
    assert suspicious_neutral_games(venues, stadiums())["game_id"].to_list() == ["e"]
    fixed = game_venues_frame(games, stadiums(), overrides([("e", "LON00")]))
    assert suspicious_neutral_games(fixed, stadiums()).is_empty()


def test_ingest_writes_clean_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    from football.ingest import venues as module

    games = schedule(
        [
            {
                "game_id": "2024_05_CCC_AAA",
                "home_team": "AAA",
                "away_team": "CCC",
                "stadium_id": "AAA00",
            }
        ]
    )
    monkeypatch.setattr(module, "load_schedules", lambda refresh=False: games)
    monkeypatch.setattr(module, "load_stadiums", stadiums)
    monkeypatch.setattr(module, "load_overrides", lambda: overrides([]))
    options = IngestOptions(seasons=[2024])
    for name in ("stadiums", "game_venues", "team_homes"):
        module.SOURCES[name](options)
        module.SOURCES[name](options)
    assert storage.read("clean", "stadiums").height == 3
    assert storage.read("clean", "game_venues")["game_id"].to_list() == ["2024_05_CCC_AAA"]
    assert storage.read("clean", "team_homes").height == 1
    result = module.check_stadium_coverage()
    assert result.passed, result.detail
