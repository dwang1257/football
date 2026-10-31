from datetime import UTC, datetime
from typing import Any

from football.collect.archive import Snapshot, write_snapshot


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def snapshot(
    collector: str,
    name: str,
    body: Any,
    fetched_at: str,
    meta: dict[str, Any] | None = None,
    status: int = 200,
    run_id: str | None = None,
) -> Snapshot:
    moment = ts(fetched_at)
    return Snapshot(
        collector=collector,
        name=name,
        url=f"https://example.test/{name}",
        fetched_at=moment,
        status=status,
        run_id=run_id or moment.strftime("%Y%m%dT%H%M%SZ"),
        body=body,
        meta=meta or {},
    )


def store(*snapshots: Snapshot) -> None:
    for item in snapshots:
        write_snapshot(item)


def _row(player: str, position: str, injury: str, days: list[str], status: str) -> str:
    cells = "".join(f"<td>{value}</td>" for value in days)
    return (
        f'<tr><td scope="row"><a href="/p/">{player}</a></td><td>{position}</td>'
        f"<td>{injury}</td>{cells}<td>{status}</td></tr>"
    )


def club_table(club: str, day_labels: list[str], rows: list[tuple[Any, ...]]) -> str:
    head = "".join(f"<th>{label}</th>" for label in day_labels)
    body = "".join(_row(*row) for row in rows)
    return (
        '<div class="d3-l-col__col-12 nfl-o-injury-report__container">'
        '<div class="nfl-o-injury-report__title">'
        f'<span class="nfl-o-injury-report__club-name">{club}</span></div>'
        '<div class="nfl-o-injury-report"><table><thead><tr>'
        '<th scope="col">Player</th><th scope="col">Position</th><th scope="col">Injury</th>'
        f'{head}<th scope="col">Game Status</th></tr></thead><tbody>{body}</tbody></table>'
        "</div></div>"
    )


def report_page(week_label: str, tables: list[str]) -> str:
    options = "".join(
        f"<option {'selected' if label == week_label else ''} "
        f'value="/team/injury-report/week/{label}"> {label} </option>'
        for label in dict.fromkeys([week_label, "REG-9", "REG-8", "REG-7"])
    )
    return (
        "<html><body><nav>menu</nav><main>"
        f'<div class="d3-o-season-selector"><select class="d3-o-dropdown">{options}</select></div>'
        f"{''.join(tables)}<footer>legend</footer></main></body></html>"
    )


def an_outcome(side: str, ticks: list[tuple[str, float]], key: str = "value") -> dict[str, Any]:
    return {
        "side": side,
        "period": "event",
        "is_live": False,
        "history": [{key: value, "updated_at": stamp} for stamp, value in ticks],
    }


def an_history(book_id: int, markets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {str(book_id): {"event": markets}}


def an_game(
    event_id: int, start: str, home: tuple[int, str], away: tuple[int, str], **odds: Any
) -> dict[str, Any]:
    return {
        "id": event_id,
        "season": 2026,
        "week": 8,
        "type": "reg",
        "status": "scheduled",
        "start_time": start,
        "home_team_id": home[0],
        "away_team_id": away[0],
        "teams": [{"id": home[0], "abbr": home[1]}, {"id": away[0], "abbr": away[1]}],
        "odds": [{"type": "game", "book_id": 15, **odds}],
    }
