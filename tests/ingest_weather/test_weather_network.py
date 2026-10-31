from datetime import UTC, datetime

import pytest

from football.ingest.weather_base import iem_http
from football.ingest.weather_mos import MODELS, Window, cached_window, parse_mos

pytestmark = pytest.mark.network


def test_iem_mos_window_downloads_and_parses() -> None:
    gfs = next(model for model in MODELS if model.model == "GFS")
    window = Window(
        "KGRB",
        "smoke",
        datetime(2015, 12, 20, tzinfo=UTC),
        datetime(2015, 12, 20, 7, tzinfo=UTC),
        True,
    )
    http = iem_http()
    try:
        payload = cached_window(http, gfs, window, refresh=False)
    finally:
        http.close()
    assert payload is not None
    frame = parse_mos(payload)
    assert frame.height > 20
    assert frame["tmp"].null_count() < frame.height
