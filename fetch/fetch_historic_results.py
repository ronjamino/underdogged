import pandas as pd
from utils.team_name_map import normalize_team

# Mapping of seasons to their CSV URLs - now includes Championship (E1)
SEASON_URLS = {
    # Premier League (E0)
    "PL_2526": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "PL_2425": "https://www.football-data.co.uk/mmz4281/2425/E0.csv", 
    "PL_2324": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "PL_2223": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "PL_2122": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    
    # Championship (E1)
    "CHAMP_2526": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "CHAMP_2425": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "CHAMP_2324": "https://www.football-data.co.uk/mmz4281/2324/E1.csv", 
    "CHAMP_2223": "https://www.football-data.co.uk/mmz4281/2223/E1.csv",
    "CHAMP_2122": "https://www.football-data.co.uk/mmz4281/2122/E1.csv",
}

def parse_season_key(season_key):
    """Extract league and season from key like 'PL_2425'"""
    parts = season_key.split("_")
    if len(parts) == 2:
        league, season = parts
        return league, season
    return "UNKNOWN", season_key

def fetch_historic_results(season_key):
    """Fetch and clean results for a given season key (e.g., 'PL_2425', 'CHAMP_2324')."""
    url = SEASON_URLS[season_key]
    league, season = parse_season_key(season_key)
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"⚠️ Could not fetch {season_key}: {e}")
        return pd.DataFrame()

    # Rename columns to standard names
    df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals"
    }, inplace=True)

    # Add league and season info
    df["league"] = league
    df["season"] = season

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

    return df[["date", "home_team", "away_team", "home_goals", "away_goals", "result", "league", "season"]]

def fetch_historic_results_multi(leagues=None):
    """Fetch and combine results for multiple seasons and leagues."""
    if leagues is None:
        leagues = ["PL", "CHAMP"]  # Default to both Premier League and Championship
    
    dfs = []
    for season_key in SEASON_URLS.keys():
        league, season = parse_season_key(season_key)
        
        # Skip if not in requested leagues
        if league not in leagues:
            continue
            
        try:
            df = fetch_historic_results(season_key)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {season_key} due to error: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs).sort_values("date").reset_index(drop=True)
    print(f"✅ Loaded {len(combined)} matches across {combined['league'].nunique()} leagues")
    print(f"📊 League breakdown: {dict(combined['league'].value_counts())}")
    
    return combined

def calculate_form(team, df, num_games=5):
    """Calculate the team's form over the last N matches."""
    # Filter the results for the specific team in the same league
    team_league = df[df['home_team'] == team]['league'].iloc[0] if len(df[df['home_team'] == team]) > 0 else None
    if team_league is None:
        team_league = df[df['away_team'] == team]['league'].iloc[0] if len(df[df['away_team'] == team]) > 0 else None
    
    if team_league:
        df = df[df['league'] == team_league]  # Only consider same league
    
    team_results_home = df[df['home_team'] == team].tail(num_games)
    team_results_away = df[df['away_team'] == team].tail(num_games)

    # Combine home and away results for the team
    team_results = pd.concat([team_results_home, team_results_away])

    # Calculate the form as total points in last N games
    points_map = {'H': 3, 'A': 0, 'D': 1}
    team_results["points"] = team_results["result"].map(points_map)
    
    return team_results["points"].sum()

def get_team_form_for_prediction(home_team, away_team, num_games=5, league=None):
    """Fetch current form for home and away teams."""
    df = fetch_historic_results_multi(leagues=[league] if league else None)
    home_form = calculate_form(home_team, df, num_games)
    away_form = calculate_form(away_team, df, num_games)
    return home_form, away_form

# Example: Fetch form for a match
if __name__ == "__main__":
    # Test with both leagues
    df = fetch_historic_results_multi()
    print(f"\nTotal matches: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Test Championship teams if available
    champ_teams = df[df['league'] == 'CHAMP']['home_team'].unique()[:2]
    if len(champ_teams) >= 2:
        home_form, away_form = get_team_form_for_prediction(champ_teams[0], champ_teams[1], league='CHAMP')
        print(f"\nChampionship example:")
        print(f"{champ_teams[0]} form: {home_form} points")
        print(f"{champ_teams[1]} form: {away_form} points")