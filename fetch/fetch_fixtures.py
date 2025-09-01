import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from utils.league_utils import _canon, LEAGUE_ALIASES

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# League codes we care about
LEAGUE_CODES = {
    # English
    "PL": "Premier League",
    "ELC": "Championship",
    # European top leagues
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "PD": "La Liga",
    # Optional: extend here
    "FL1": "Ligue 1",
    "PPL": "Primeira Liga",
    "DED": "Eredivisie",
}

def get_accessible_competitions():
    """
    Hit /competitions to see which league codes your API key can actually access.
    Returns a set of codes like {"PL","BL1","PD"}.
    """
    resp = requests.get(f"{BASE_URL}/competitions", headers=HEADERS)
    resp.raise_for_status()
    comps = resp.json().get("competitions", [])
    return {c.get("code") for c in comps if c.get("code")}

ACCESSIBLE = get_accessible_competitions()

def safe_get(url, max_retries=2):
    """
    Wrapper for API calls with backoff on 429 rate limits.
    """
    for attempt in range(max_retries):
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 429 and attempt < max_retries - 1:
            wait = 30 * (attempt + 1)
            print(f"⚠️ 429 rate limit. Waiting {wait}s…")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    return None

def fetch_upcoming_fixtures(league_codes=None, limit=10, date_from=None, date_to=None):
    """
    Fetch scheduled fixtures for one or more league codes.

    Args:
        league_codes: list or str of league codes ("PL","BL1",etc.)
        limit: max fixtures per league
        date_from/date_to: optional YYYY-MM-DD filters
    """
    if league_codes is None:
        league_codes = ["PL"]
    elif isinstance(league_codes, str):
        league_codes = [league_codes]

    # Filter down to what the API key can see
    filtered = [c for c in league_codes if c in ACCESSIBLE]
    missing = [c for c in league_codes if c not in ACCESSIBLE]
    if missing:
        print(f"⚠️ Not accessible with this key: {', '.join(missing)}")

    qs = ["status=SCHEDULED"]
    if date_from:
        qs.append(f"dateFrom={date_from}")
    if date_to:
        qs.append(f"dateTo={date_to}")
    query = "&".join(qs)

    fixtures = []
    for code in filtered:
        url = f"{BASE_URL}/competitions/{code}/matches?{query}"
        resp = safe_get(url)
        if not resp:
            print(f"❌ Skipped {code} due to repeated errors")
            continue

        matches = resp.json().get("matches", [])[:limit]
        for m in matches:
            fixtures.append({
                "utc_date": m["utcDate"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "matchday": m.get("matchday"),
                "league_code": code,
                "league_name": LEAGUE_CODES.get(code, code),
            })

        print(f"✅ {len(matches)} fixtures from {LEAGUE_CODES.get(code, code)}")

    df = pd.DataFrame(fixtures)
    if not df.empty:
        print(f"📊 Total: {len(df)} fixtures across {df['league_code'].nunique()} leagues")
    else:
        print("ℹ️ No fixtures found (maybe widen dateFrom/dateTo).")
    return df

# after building df from API
def map_league_column(df):
    if "league_code" in df.columns:
        df["league"] = df["league_code"].map(_canon)
    elif "league" in df.columns:
        df["league"] = df["league"].map(_canon)
    else:
        df["league"] = "UNKNOWN"
    return df

# Convenience wrappers
def fetch_fixtures_premier_league(limit=10):
    return fetch_upcoming_fixtures("PL", limit)

def fetch_fixtures_championship(limit=10):
    return fetch_upcoming_fixtures("ELC", limit)

def fetch_fixtures_bundesliga(limit=10):
    return fetch_upcoming_fixtures("BL1", limit)

def fetch_fixtures_serie_a(limit=10):
    return fetch_upcoming_fixtures("SA", limit)

def fetch_fixtures_la_liga(limit=10):
    return fetch_upcoming_fixtures("PD", limit)

def fetch_fixtures_top_5(limit=10):
    return fetch_upcoming_fixtures(["PL", "BL1", "SA", "PD", "FL1"], limit)

def fetch_fixtures_all_english(limit=10):
    return fetch_upcoming_fixtures(["PL", "ELC"], limit)

def fetch_fixtures_all_european(limit=10):
    return fetch_upcoming_fixtures(["BL1", "SA", "PD", "FL1", "PPL", "DED"], limit)

def fetch_fixtures_mega_combo(limit=10):
    return fetch_upcoming_fixtures(list(LEAGUE_CODES.keys()), limit)

if __name__ == "__main__":
    print("📅 Demo run: Premier League + Euro top 3")
    df = fetch_upcoming_fixtures(["PL", "BL1", "SA", "PD"], limit=5)
    df = map_league_column(df)
    if not df.empty:
        for _, row in df.iterrows():
            print(f"• {row['home_team']} vs {row['away_team']} ({row['league_name']})")
