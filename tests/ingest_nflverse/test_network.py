import polars as pl
import pytest

from football import storage
from football.ingest.base import IngestOptions
from football.ingest.nflverse_fetch import NotPublished, fetch
from football.ingest.nflverse_games import ingest_schedules
from football.schemas import GAMES


@pytest.mark.network
def test_schedules_smoke() -> None:
    ingest_schedules(IngestOptions(seasons=[2023]))
    games = storage.read("clean", GAMES.name)
    assert games.filter(pl.col("game_type") == "REG").height == 272
    assert (games["available_at"] > games["kickoff_utc"]).all()


@pytest.mark.network
def test_missing_release_asset_is_not_published() -> None:
    with pytest.raises(NotPublished):
        fetch("pbp/play_by_play_1980.parquet")
