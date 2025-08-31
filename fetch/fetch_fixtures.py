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

# League code mapping
LEAGUE_CODES = {
    "PL": "Premier League",
    "ELC": "Championship",  # English League Championship 
    "EL1": "League One",    # Will add these later
    "EL2": "League Two"     # Will add these later
}

def fetch_upcoming_fixtures(league_codes=None, limit=10):
    """
    Fetch upcoming fixtures for given league codes.
    
    Args:
        league_codes: List of league codes (e.g., ["PL", "ELC"]) or single code
        limit: Max fixtures per league
    """
    if league_codes is None:
        league_codes = ["PL"]  # Default to Premier League only
    elif isinstance(league_codes, str):
        league_codes = [league_codes]  # Convert single string to list
    
    all_fixtures = []
    
    for league_code in league_codes:
        try:
            url = f"{BASE_URL}/competitions/{league_code}/matches?status=SCHEDULED"
            response = requests.get(url, headers=HEADERS)

            if response.status_code != 200:
                print(f"⚠️ API Error for {league_code}: {response.status_code} - {response.text}")
                continue

            matches = response.json().get("matches", [])[:limit]
            league_fixtures = []

            for match in matches:
                league_fixtures.append({
                    "utc_date": match["utcDate"],
                    "home_team": match["homeTeam"]["name"], 
                    "away_team": match["awayTeam"]["name"],
                    "matchday": match.get("matchday"),
                    "competition": match.get("competition", {}).get("name", league_code),
                    "league_code": league_code,
                    "league_name": LEAGUE_CODES.get(league_code, league_code)
                })
            
            all_fixtures.extend(league_fixtures)
            print(f"✅ Fetched {len(league_fixtures)} fixtures from {LEAGUE_CODES.get(league_code, league_code)}")
            
        except Exception as e:
            print(f"⚠️ Error fetching {league_code} fixtures: {e}")
            continue
    
    df = pd.DataFrame(all_fixtures)
    if not df.empty:
        print(f"📊 Total fixtures: {len(df)} across {df['league_code'].nunique()} leagues")
    
    return df

def fetch_fixtures_premier_league(limit=10):
    """Convenience function for Premier League only (backward compatibility)"""
    return fetch_upcoming_fixtures(["PL"], limit)

def fetch_fixtures_championship(limit=10): 
    """Convenience function for Championship only"""
    return fetch_upcoming_fixtures(["ELC"], limit)

def fetch_fixtures_both_top_divisions(limit=10):
    """Fetch from both Premier League and Championship"""
    return fetch_upcoming_fixtures(["PL", "ELC"], limit)

# Keep old function name for backward compatibility
fetch_fixtures = fetch_upcoming_fixtures

if __name__ == "__main__":
    print("Testing multi-league fixture fetching...\n")
    
    # Test Premier League only
    print("1. Premier League fixtures:")
    pl_fixtures = fetch_fixtures_premier_league(5)
    if not pl_fixtures.empty:
        print(pl_fixtures[["home_team", "away_team", "league_name"]].head())
    
    print("\n" + "="*50 + "\n")
    
    # Test Championship only  
    print("2. Championship fixtures:")
    champ_fixtures = fetch_fixtures_championship(5)
    if not champ_fixtures.empty:
        print(champ_fixtures[["home_team", "away_team", "league_name"]].head())
    
    print("\n" + "="*50 + "\n")
    
    # Test both together
    print("3. Both leagues combined:")
    both_fixtures = fetch_fixtures_both_top_divisions(5)
    if not both_fixtures.empty:
        print(both_fixtures[["home_team", "away_team", "league_name"]])
        print(f"\nLeague breakdown: {dict(both_fixtures['league_code'].value_counts())}")