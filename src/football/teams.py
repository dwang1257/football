import re

FRANCHISES: dict[str, tuple[str, ...]] = {
    "ARI": ("ARI",),
    "ATL": ("ATL",),
    "BAL": ("BAL",),
    "BUF": ("BUF",),
    "CAR": ("CAR",),
    "CHI": ("CHI",),
    "CIN": ("CIN",),
    "CLE": ("CLE",),
    "DAL": ("DAL",),
    "DEN": ("DEN",),
    "DET": ("DET",),
    "GB": ("GB",),
    "HOU": ("HOU",),
    "IND": ("IND",),
    "JAX": ("JAX",),
    "KC": ("KC",),
    "LA": ("STL", "LA"),
    "LAC": ("SD", "LAC"),
    "LV": ("OAK", "LV"),
    "MIA": ("MIA",),
    "MIN": ("MIN",),
    "NE": ("NE",),
    "NO": ("NO",),
    "NYG": ("NYG",),
    "NYJ": ("NYJ",),
    "PHI": ("PHI",),
    "PIT": ("PIT",),
    "SEA": ("SEA",),
    "SF": ("SF",),
    "TB": ("TB",),
    "TEN": ("TEN",),
    "WAS": ("WAS",),
}

_RELOCATIONS: dict[str, tuple[int, str, str]] = {
    "LA": (2016, "STL", "LA"),
    "LAC": (2017, "SD", "LAC"),
    "LV": (2020, "OAK", "LV"),
}

_ALIASES: dict[str, str] = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "GNB": "GB",
    "HST": "HOU",
    "JAC": "JAX",
    "KAN": "KC",
    "KCC": "KC",
    "LAR": "LA",
    "RAM": "LA",
    "LVR": "LV",
    "RAI": "LV",
    "NWE": "NE",
    "NEP": "NE",
    "NOR": "NO",
    "NOS": "NO",
    "SDG": "LAC",
    "SFO": "SF",
    "SFF": "SF",
    "TAM": "TB",
    "TBB": "TB",
    "WSH": "WAS",
    "WFT": "WAS",
}

_NAMES: dict[str, str] = {
    "arizona": "ARI",
    "cardinals": "ARI",
    "atlanta": "ATL",
    "falcons": "ATL",
    "baltimore": "BAL",
    "ravens": "BAL",
    "buffalo": "BUF",
    "bills": "BUF",
    "carolina": "CAR",
    "panthers": "CAR",
    "chicago": "CHI",
    "bears": "CHI",
    "cincinnati": "CIN",
    "bengals": "CIN",
    "cleveland": "CLE",
    "browns": "CLE",
    "dallas": "DAL",
    "cowboys": "DAL",
    "denver": "DEN",
    "broncos": "DEN",
    "detroit": "DET",
    "lions": "DET",
    "greenbay": "GB",
    "packers": "GB",
    "houston": "HOU",
    "texans": "HOU",
    "indianapolis": "IND",
    "colts": "IND",
    "jacksonville": "JAX",
    "jaguars": "JAX",
    "kansascity": "KC",
    "kansas": "KC",
    "chiefs": "KC",
    "larams": "LA",
    "losangelesrams": "LA",
    "stlouis": "LA",
    "stlouisrams": "LA",
    "rams": "LA",
    "lachargers": "LAC",
    "losangeleschargers": "LAC",
    "sandiego": "LAC",
    "sandiegochargers": "LAC",
    "chargers": "LAC",
    "lasvegas": "LV",
    "lasvegasraiders": "LV",
    "oakland": "LV",
    "oaklandraiders": "LV",
    "raiders": "LV",
    "miami": "MIA",
    "dolphins": "MIA",
    "minnesota": "MIN",
    "vikings": "MIN",
    "newengland": "NE",
    "patriots": "NE",
    "neworleans": "NO",
    "saints": "NO",
    "nygiants": "NYG",
    "newyorkgiants": "NYG",
    "giants": "NYG",
    "nyjets": "NYJ",
    "newyorkjets": "NYJ",
    "jets": "NYJ",
    "philadelphia": "PHI",
    "eagles": "PHI",
    "pittsburgh": "PIT",
    "steelers": "PIT",
    "seattle": "SEA",
    "seahawks": "SEA",
    "sanfrancisco": "SF",
    "49ers": "SF",
    "niners": "SF",
    "tampabay": "TB",
    "tampa": "TB",
    "buccaneers": "TB",
    "tennessee": "TEN",
    "titans": "TEN",
    "washington": "WAS",
    "commanders": "WAS",
    "redskins": "WAS",
    "footballteam": "WAS",
}

_ALL_CODES = {code for codes in FRANCHISES.values() for code in codes}


def franchise(team: str) -> str:
    key = team.strip().upper()
    if key in FRANCHISES:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    for franchise_code, codes in FRANCHISES.items():
        if key in codes:
            return franchise_code
    squashed = re.sub(r"[^a-z0-9]", "", team.lower())
    if squashed in _NAMES:
        return _NAMES[squashed]
    for name, code in sorted(_NAMES.items(), key=lambda item: -len(item[0])):
        if squashed.endswith(name) or squashed.startswith(name):
            return code
    raise ValueError(f"unknown team: {team!r}")


def team_code(team: str, season: int) -> str:
    code = franchise(team)
    if code in _RELOCATIONS:
        first_season, old, new = _RELOCATIONS[code]
        return new if season >= first_season else old
    return code


def is_team_code(team: str) -> bool:
    return team in _ALL_CODES


def game_id(season: int, week: int, away: str, home: str) -> str:
    return f"{season}_{week:02d}_{team_code(away, season)}_{team_code(home, season)}"
