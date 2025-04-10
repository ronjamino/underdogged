import pandas as pd
import os
from utils.team_name_map import normalize_team

def fetch_odds():
    # Fake/mock odds data for example (replace with real API call if needed)
    data = [
        {"commence_time": "2025-04-12T11:30:00Z", "home_team": "Manchester City", "away_team": "Crystal Palace", "home_odds": 1.50, "away_odds": 5.50, "draw_odds": 4.33},
        {"commence_time": "2025-04-12T14:00:00Z", "home_team": "Southampton", "away_team": "Aston Villa", "home_odds": 5.50, "away_odds": 1.53, "draw_odds": 4.20},
        # Add more rows as needed...
    ]
    df = pd.DataFrame(data)

    # Normalize team names
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)

    # Make sure directory exists
    os.makedirs("data/raw", exist_ok=True)

    # Save to CSV
    df.to_csv("data/raw/odds.csv", index=False)
    print("✅ Odds saved to data/raw/odds.csv")

    return df

if __name__ == "__main__":
    odds_df = fetch_odds()
    print(odds_df)
