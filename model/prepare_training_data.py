import pandas as pd
from fetch.fetch_historic_results import fetch_historic_results_multi, get_team_form_for_prediction

def build_features(df, h2h_window=5, form_window=10):
    """
    Generate features from historic match results including head-to-head form and recent team form.
    """
    df = df.sort_values("date")

    # Create result as 1 (home win), 0 (draw), -1 (away win)
    df["outcome_code"] = df["result"].map({"H": 1, "D": 0, "A": -1})

    feature_rows = []

    for idx, row in df.iterrows():
        current_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        # Slice only past matches up to the current match date
        past_matches = df[df["date"] < current_date]

        # Limit to last 10 matches for each team
        home_past_matches = past_matches[past_matches["home_team"] == home].tail(form_window)
        away_past_matches = past_matches[past_matches["away_team"] == away].tail(form_window)

        # Combine home and away matches to get the last 10 matches for each team
        recent_form = pd.concat([home_past_matches, away_past_matches])

        # Head-to-head history (limiting to the last h2h_window matches)
        h2h = past_matches[
            ((past_matches["home_team"] == home) & (past_matches["away_team"] == away)) |
            ((past_matches["home_team"] == away) & (past_matches["away_team"] == home))
        ].tail(h2h_window)

        # If not enough H2H history, skip this row
        if len(h2h) < 2:
            continue

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

        # Feature: form-based win rate for home and away team in their last 10 games
        home_form = recent_form[recent_form["home_team"] == home]
        away_form = recent_form[recent_form["away_team"] == away]

        # Calculate home team's win rate in their last 10 games
        home_form_wins = home_form[home_form["result"] == "H"].shape[0]
        home_form_winrate = home_form_wins / len(home_form) if len(home_form) > 0 else 0

        # Calculate away team's win rate in their last 10 games
        away_form_wins = away_form[away_form["result"] == "A"].shape[0]
        away_form_winrate = away_form_wins / len(away_form) if len(away_form) > 0 else 0

        feature_rows.append({
            "home_team": home,
            "away_team": away,
            "match_date": current_date,
            "avg_goal_diff_h2h": avg_goal_diff,
            "h2h_home_winrate": h2h_winrate,
            "home_form_winrate": home_form_winrate,
            "away_form_winrate": away_form_winrate,
            "result": row["result"]  # Replace 'label' with 'result'
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
