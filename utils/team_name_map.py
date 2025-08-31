import re

TEAM_NAME_MAP = {
    # existing explicit mappings
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Tottenham Hotspur FC": "Tottenham",
    "Brighton & Hove Albion FC": "Brighton",
    "Brentford FC": "Brentford",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Leeds United FC": "Leeds",
    "Nottingham Forest FC": "Nottm Forest",
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Burnley FC": "Burnley",
    # add others as needed
}

_STRIP_WORDS = {"fc", "afc", "cfc"}
_SUFFIXES = [
    r"\s+fc$", r"\s+afc$", r"\s+cf$", r"\s+cf c$", r"\s+football club$",
    r"\s+united fc$", r"\s+city fc$", r"\s+town fc$", r"\s+athletic fc$",
]

def _strip_suffixes(name: str) -> str:
    n = name
    for pat in _SUFFIXES:
        n = re.sub(pat, "", n, flags=re.I)
    return n

def _squash_spaces(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()

def canonicalize_team(name: str) -> str:
    if not name:
        return name
    # explicit map first
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # generic cleanup
    n = _strip_suffixes(name)
    # remove standalone “FC/AF” noise tokens
    n = " ".join(w for w in n.split() if w.lower() not in _STRIP_WORDS)
    n = _squash_spaces(n)
    return TEAM_NAME_MAP.get(n, n)

def normalize_team(name: str) -> str:
    # keep old API but call canonicalize
    return canonicalize_team(name)
