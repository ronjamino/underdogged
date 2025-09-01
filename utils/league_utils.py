# underdogged/utils/league_utils.py

# Canonical codes used across the project
LEAGUE_ALIASES = {
    "PL": "PL", "ELC": "ELC", "BL1": "BL1", "SA": "SA", "PD": "PD",
    "CHAMP": "ELC", "EFL_CHAMPIONSHIP": "ELC", "CHAMPIONSHIP": "ELC",
    "PREMIER LEAGUE": "PL", "EPL": "PL",
    "BUNDESLIGA": "BL1",
    "SERIE A": "SA",
    "LA LIGA": "PD", "PRIMERA DIVISION": "PD",
}

# Map canonical -> label used by history fetcher (only ELC needs mapping today)
INTERNAL_FETCH_MAP = {
    "PL": "PL",
    "ELC": "CHAMP",
    "BL1": "BL1",
    "SA": "SA",
    "PD": "PD",
}

def _canon(x: str) -> str:
    if x is None:
        return "UNKNOWN"
    key = str(x).strip().upper()
    return LEAGUE_ALIASES.get(key, key)

def to_history_code(code: str) -> str:
    return INTERNAL_FETCH_MAP.get(code, code)
