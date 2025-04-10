# fetch/fetch_historic_results.py

import pandas as pd
from utils.team_name_map import normalize_team

SEASON_URLS = {
    "2324": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2223": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2122": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
}


def fetch_historic_results(season_code):
    """Fetch and clean Premier League results for a given season."""
    url = SEASON_URLS[season_code]
    df = pd.read_csv(url)

    df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals"
    }, inplace=True)

    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    # Generate result labels
    def get_label(row):
        if row["home_goals"] > row["away_goals"]:
            return "home_win"
        elif row["home_goals"] < row["away_goals"]:
            return "away_win"
        else:
            return "draw"

    df["label"] = df.apply(get_label, axis=1)

    return df[["date", "home_team", "away_team", "home_goals", "away_goals", "label"]]


def fetch_historic_results_multi():
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
