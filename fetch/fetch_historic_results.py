# fetch/fetch_historic_results.py

import pandas as pd

def fetch_historic_results(season="2324", league_code="E0"):
    """
    Download historic match results from football-data.co.uk
    E0 = Premier League, E1 = Championship
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
    df = pd.read_csv(url)

    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
    df.columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]

    df["label"] = df["result"].map({"H": "home_win", "D": "draw", "A": "away_win"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    return df

if __name__ == "__main__":
    df = fetch_historic_results()
    print(df.tail())