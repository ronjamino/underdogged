# fetch/fetch_fixtures.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

HEADERS = {
    "X-Auth-Token": API_KEY
}

def fetch_upcoming_fixtures(league_code="PL", limit=10):
    """Fetch upcoming fixtures for a given league code (e.g. PL, CHAMP)"""
    url = f"{BASE_URL}/competitions/{league_code}/matches?status=SCHEDULED"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    matches = response.json().get("matches", [])[:limit]
    fixtures = []

    for match in matches:
        fixtures.append({
            "utc_date": match["utcDate"],
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "matchday": match.get("matchday"),
            "competition": match.get("competition", {}).get("name", league_code)
        })

    return pd.DataFrame(fixtures)

fetch_fixtures = fetch_upcoming_fixtures

if __name__ == "__main__":
    df = fetch_upcoming_fixtures()
    print(df)
