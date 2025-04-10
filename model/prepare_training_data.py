# model/prepare_training_data.py

import pandas as pd
from fetch.fetch_historic_results import fetch_historic_results
from fetch.fetch_historic_results import fetch_historic_results_multi

def build_features(df, h2h_window=5):
    """
    Generate features from historic match results including head-to-head form.
    """
    df = df.sort_values("date")

    # Create result as 1 (home win), 0 (draw), -1 (away win)
    df["outcome_code"] = df["label"].map({"home_win": 1, "draw": 0, "away_win": -1})

    feature_rows = []

    for idx, row in df.iterrows():
        current_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        # Slice only past matches
        past_matches = df[df["date"] < current_date]

        # Head-to-head history
        h2h = past_matches[
            ((past_matches["home_team"] == home) & (past_matches["away_team"] == away)) |
            ((past_matches["home_team"] == away) & (past_matches["away_team"] == home))
        ].tail(h2h_window)

        # if len(h2h) < h2h_window: # Uncomment this line to enforce a minimum number of H2H matches
        if len(h2h) < 2:
            continue  # skip if not enough H2H history

        # Feature: average goal difference in past H2H
        h2h["goal_diff"] = h2h["home_goals"] - h2h["away_goals"]
        avg_goal_diff = h2h["goal_diff"].mean()

        # Feature: win rate for home team in H2H
        home_wins = 0
        for _, match in h2h.iterrows():
            if match["home_team"] == home and match["result"] == "H":
                home_wins += 1
            elif match["away_team"] == home and match["result"] == "A":
                home_wins += 1
        h2h_winrate = home_wins / len(h2h)

        feature_rows.append({
            "home_team": home,
            "away_team": away,
            "match_date": current_date,
            "avg_goal_diff_h2h": avg_goal_diff,
            "h2h_home_winrate": h2h_winrate,
            "label": row["label"]
        })

    return pd.DataFrame(feature_rows)

if __name__ == "__main__":
    print("📊 Building feature matrix from historic results...")
    df_raw = fetch_historic_results_multi()
    df_features = build_features(df_raw)

    print(df_features.head())

    # Save for training later
    df_features.to_csv("data/processed/training_data.csv", index=False)
    print("✅ Feature data saved to data/processed/training_data.csv")
