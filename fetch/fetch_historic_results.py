import pandas as pd
from utils.team_name_map import normalize_team

# Mapping of seasons to their CSV URLs
SEASON_URLS = {
    "2324": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2223": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2122": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    # Add more seasons if needed
}

def fetch_historic_results(season_code):
    """Fetch and clean Premier League results for a given season."""
    url = SEASON_URLS[season_code]
    df = pd.read_csv(url)

    # Rename columns to standard names
    df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals"
    }, inplace=True)

    # Normalize team names
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)

    # Parse date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Drop rows missing required values
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    # Add the result column
    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"  # Home win
        elif row["away_goals"] > row["home_goals"]:
            return "A"  # Away win
        else:
            return "D"  # Draw

    df["result"] = df.apply(get_result, axis=1)

    return df[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]]

def fetch_historic_results_multi():
    """Fetch and combine results for multiple seasons."""
    dfs = []
    for season_code in SEASON_URLS.keys():
        try:
            df = fetch_historic_results(season_code)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping season {season_code} due to error: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs).sort_values("date").reset_index(drop=True)

if __name__ == "__main__":
    df = fetch_historic_results_multi()
    print(df.head())
