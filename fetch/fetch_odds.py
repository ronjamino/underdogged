# fetch/fetch_odds.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

def fetch_match_odds(region="uk", market="h2h"):
    """Fetch head-to-head match odds from The Odds API"""
    params = {
        "apiKey": API_KEY,
        "regions": region,
        "markets": market,
        "oddsFormat": "decimal"
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    matches = response.json()
    odds_data = []

    for match in matches:
        if not match.get("bookmakers"):
            continue

        home = match["home_team"]
        away = match["away_team"]
        commence = match["commence_time"]

        # We'll just use the first listed bookmaker
        bookmaker = match["bookmakers"][0]
        outcomes = bookmaker["markets"][0]["outcomes"]

        odds_entry = {
            "commence_time": commence,
            "home_team": home,
            "away_team": away
        }

        for outcome in outcomes:
            if outcome["name"] == home:
                odds_entry["home_odds"] = outcome["price"]
            elif outcome["name"] == away:
                odds_entry["away_odds"] = outcome["price"]
            elif outcome["name"].lower() == "draw":
                odds_entry["draw_odds"] = outcome["price"]

        odds_data.append(odds_entry)

    return pd.DataFrame(odds_data)

if __name__ == "__main__":
    df = fetch_match_odds()
    print(df)
