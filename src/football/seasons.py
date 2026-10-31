from datetime import UTC, datetime


def current_season(now: datetime | None = None) -> int:
    moment = now or datetime.now(UTC)
    return moment.year if moment.month >= 3 else moment.year - 1


def parse_seasons(text: str) -> list[int]:
    seasons: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seasons.update(range(int(start), int(end) + 1))
        else:
            seasons.add(int(part))
    return sorted(seasons)
