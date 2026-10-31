from collections.abc import Sequence
from datetime import UTC, datetime

from football.collect import action_network, espn, forecasts, team_reports
from football.collect.base import Collector, CollectorResult, Context
from football.collect.http import Fetcher

COLLECTORS: dict[str, Collector] = {
    team_reports.NAME: team_reports.collect,
    "espn_injuries": espn.collect_injuries,
    "action_network": action_network.collect,
    "espn_odds": espn.collect_odds,
    "espn_fpi": espn.collect_fpi,
    forecasts.NAME: forecasts.collect,
}

HOST_INTERVALS_S = {
    "sports.core.api.espn.com": 0.05,
    "api.actionnetwork.com": 0.5,
    "api.weather.gov": 0.5,
}


def run_collectors(names: Sequence[str], fetcher: Fetcher | None = None) -> list[CollectorResult]:
    started = datetime.now(UTC)
    owned = fetcher is None
    client = fetcher or Fetcher(host_intervals_s=HOST_INTERVALS_S)
    ctx = Context(fetcher=client, run_id=started.strftime("%Y%m%dT%H%M%SZ"), started_at=started)
    results: list[CollectorResult] = []
    try:
        for name in names:
            result = CollectorResult(collector=name)
            try:
                COLLECTORS[name](ctx, result)
            except Exception as exc:
                result.failed["collector"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
    finally:
        if owned:
            client.close()
    return results


def all_failed(results: Sequence[CollectorResult]) -> bool:
    return not any(result.status in {"ok", "partial"} for result in results)
