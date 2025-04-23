import os
import pandas as pd
import joblib
from fetch.fetch_fixtures import fetch_upcoming_fixtures
from fetch.fetch_historic_results import fetch_historic_results_multi
from utils.team_name_map import normalize_team

MODEL_PATH = "models/random_forest_model.pkl"
CONFIDENCE_THRESHOLD = 0.0

LABELS = ["home_win", "draw", "away_win"]

def build_prediction_features(fixtures, history):
    fixtures = fixtures.copy()
    fixtures.rename(columns={"utc_date": "date"}, inplace=True)

    fixtures["home_team"] = fixtures["home_team"].apply(normalize_team)
    fixtures["away_team"] = fixtures["away_team"].apply(normalize_team)
    history["home_team"] = history["home_team"].apply(normalize_team)
    history["away_team"] = history["away_team"].apply(normalize_team)

    features = []
    for _, row in fixtures.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        fixture_date = pd.to_datetime(row["date"], utc=True)

        h2h = history[
            ((history["home_team"] == home) & (history["away_team"] == away)) |
            ((history["home_team"] == away) & (history["away_team"] == home))
        ]
        h2h = h2h[h2h["date"] < fixture_date]

        print(f"🔎 {home} vs {away} – Found {len(h2h)} H2H matches")

        if len(h2h) < 2:
            continue

        goal_diffs = []
        home_wins = 0
        for _, match in h2h.iterrows():
            if match["home_team"] == home:
                goal_diffs.append(match["home_goals"] - match["away_goals"])
                if match["result"] == "H":
                    home_wins += 1
            else:
                goal_diffs.append(match["away_goals"] - match["home_goals"])
                if match["result"] == "A":
                    home_wins += 1

        # Calculate recent form for the home and away teams
        home_form = history[(history["home_team"] == home) | (history["away_team"] == home)]
        away_form = history[(history["home_team"] == away) | (history["away_team"] == away)]

        # Select last 10 games for form calculation
        home_form = home_form.sort_values("date").tail(10)
        away_form = away_form.sort_values("date").tail(10)

        home_form_winrate = home_form[home_form["result"] == "H"].shape[0] / len(home_form)
        away_form_winrate = away_form[away_form["result"] == "A"].shape[0] / len(away_form)

        features.append({
            "match_date": fixture_date,
            "home_team": home,
            "away_team": away,
            "avg_goal_diff_h2h": sum(goal_diffs) / len(goal_diffs),
            "h2h_home_winrate": home_wins / len(h2h),
            "home_form_winrate": home_form_winrate,
            "away_form_winrate": away_form_winrate
        })

    return pd.DataFrame(features)

def predict_fixtures():
    print("⚽ Loading model and generating predictions...")

    model = joblib.load(MODEL_PATH)
    history = fetch_historic_results_multi()
    fixtures = fetch_upcoming_fixtures()

    print("\n📅 Upcoming Fixtures:")
    print(fixtures[["home_team", "away_team", "utc_date"]])

    features_df = build_prediction_features(fixtures, history)

    if features_df.empty:
        print("😕 No fixtures with sufficient H2H data to predict.")
        return

    # Update the feature set to include the new form-based features
    X = features_df[[
        "avg_goal_diff_h2h", "h2h_home_winrate", "home_form_winrate", "away_form_winrate"
    ]]
    
    predicted_classes = model.predict(X)
    predicted_probas = model.predict_proba(X)

    features_df["predicted_result"] = [LABELS[i] for i in predicted_classes]
    features_df["predicted_proba"] = predicted_probas.max(axis=1)
    features_df["confidence_label"] = [
        f"{proba:.2f} ({LABELS[i]})"
        for proba, i in zip(predicted_probas.max(axis=1), predicted_classes)
    ]

    confident_preds = features_df[features_df["predicted_proba"] >= CONFIDENCE_THRESHOLD]
    confident_preds = confident_preds.sort_values(by="predicted_proba", ascending=False)

    if confident_preds.empty:
        print(f"😐 No predictions above the confidence threshold of {CONFIDENCE_THRESHOLD}.")
    else:
        print("\n🎯 Confident Predictions (sorted by confidence):")
        print(confident_preds[["match_date", "home_team", "away_team", "predicted_result", "confidence_label"]])

        # Normalize datetimes to the same format
        confident_preds["match_date"] = pd.to_datetime(confident_preds["match_date"]).dt.floor("min")

        merged = confident_preds

    os.makedirs("data/predictions", exist_ok=True)
    merged.to_csv("data/predictions/latest_predictions.csv", index=False)
    print("✅ Predictions saved to data/predictions/latest_predictions.csv")

if __name__ == "__main__":
    predict_fixtures()
