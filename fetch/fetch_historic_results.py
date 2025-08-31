import pandas as pd
from utils.team_name_map import normalize_team

# Mapping of seasons to their CSV URLs
SEASON_URLS = {
    "2526": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "2425": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
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
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce", utc=True)

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

def calculate_form(team, df, num_games=5):
    """Calculate the team's form over the last N matches."""
    # Filter the results for the specific team
    team_results_home = df[df['home_team'] == team].tail(num_games)
    team_results_away = df[df['away_team'] == team].tail(num_games)

    # Combine home and away results for the team
    team_results = pd.concat([team_results_home, team_results_away])

    # Calculate the form as total points in last N games
    points_map = {'H': 3, 'A': 0, 'D': 1}
    team_results["points"] = team_results["result"].map(points_map)
    
    return team_results["points"].sum()

def get_team_form_for_prediction(home_team, away_team, num_games=5):
    """Fetch current form for home and away teams."""
    df = fetch_historic_results_multi()
    home_form = calculate_form(home_team, df, num_games)
    away_form = calculate_form(away_team, df, num_games)
    return home_form, away_form

# Example: Fetch form for a match
if __name__ == "__main__":
    home_form, away_form = get_team_form_for_prediction('Tottenham Hotspur', 'Chelsea')
    print(f"Tottenham Hotspur form: {home_form} points")
    print(f"Chelsea form: {away_form} points")
