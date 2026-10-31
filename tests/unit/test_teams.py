import pytest

from football.teams import franchise, game_id, team_code


@pytest.mark.parametrize(
    ("raw", "season", "expected"),
    [
        ("LAR", 2015, "STL"),
        ("LAR", 2020, "LA"),
        ("San Diego", 2010, "SD"),
        ("Chargers", 2024, "LAC"),
        ("OAK", 2021, "LV"),
        ("Raiders", 2019, "OAK"),
        ("JAC", 2020, "JAX"),
        ("WSH", 2022, "WAS"),
        ("KansasCity", 2017, "KC"),
        ("NY Giants", 2017, "NYG"),
        ("NYJets", 2017, "NYJ"),
        ("Tampa Bay Buccaneers", 2017, "TB"),
    ],
)
def test_team_code(raw: str, season: int, expected: str) -> None:
    assert team_code(raw, season) == expected


def test_franchise_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        franchise("Gotham Rogues")


def test_game_id() -> None:
    assert game_id(2026, 8, "BAL", "BUF") == "2026_08_BAL_BUF"
    assert game_id(2015, 1, "SEA", "LAR") == "2015_01_SEA_STL"
