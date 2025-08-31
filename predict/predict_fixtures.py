import os
import pandas as pd
import joblib
from fetch.fetch_fixtures import fetch_upcoming_fixtures
from fetch.fetch_historic_results import fetch_historic_results_multi
from utils.team_name_map import normalize_team

MODEL_PATH = "models/random_forest_model.pkl"

# Class label order used by the model
LABELS = ["home_win", "draw", "away_win"]

# Only show results above this confidence in the terminal & CSV
CONFIDENCE_THRESHOLD = 0.60

def _venue_aware_form(history: pd.DataFrame, home: str, away: str, league: str, window: int = 10):
    """Return (home_form_winrate, away_form_winrate) using venue-aware last-N games within same league."""
    # Filter to same league first
    league_history = history[history["league"] == league] if league != "UNKNOWN" else history
    
    home_last_home = league_history[league_history["home_team"] == home].sort_values("date").tail(window)
    away_last_away = league_history[league_history["away_team"] == away].sort_values("date").tail(window)

    home_form_winrate = (home_last_home["result"] == "H").mean() if len(home_last_home) else 0.5
    away_form_winrate = (away_last_away["result"] == "A").mean() if len(away_last_away) else 0.5
    return float(home_form_winrate), float(away_form_winrate)

def build_prediction_features(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Create prediction feature rows for each upcoming fixture."""
    fixtures = fixtures.copy()
    fixtures.rename(columns={"utc_date": "date"}, inplace=True)
    fixtures["date"] = pd.to_datetime(fixtures["date"], utc=True, errors="coerce")

    # Normalize names in BOTH datasets
    for col in ["home_team", "away_team"]:
        fixtures[col] = fixtures[col].apply(normalize_team)
        history[col] = history[col].apply(normalize_team)

    rows = []
    for _, f in fixtures.iterrows():
        fixture_date = pd.to_datetime(f["date"], utc=True, errors="coerce")
        if pd.isna(fixture_date):
            continue
        home = f["home_team"]
        away = f["away_team"]
        league_code = f.get("league_code", "PL")
        
        # Map league codes to our internal league names
        league_map = {"PL": "PL", "ELC": "CHAMP"}
        league = league_map.get(league_code, league_code)

        print(f"🔎 {home} vs {away} ({league})")

        # Prior head-to-head (strictly before fixture date, same league)
        league_history = history[history["league"] == league] if league in history["league"].values else history
        h2h = league_history[
            (
                ((league_history["home_team"] == home) & (league_history["away_team"] == away)) |
                ((league_history["home_team"] == away) & (league_history["away_team"] == home))
            ) & (league_history["date"] < fixture_date)
        ].sort_values("date")

        print(f"   Found {len(h2h)} H2H matches in {league}")

        # Compute H2H features with graceful fallback if low history
        if len(h2h) < 2:
            avg_goal_diff_h2h = 0.0
            h2h_home_winrate = 0.5
            print(f"   ⚠️ Limited H2H history, using defaults")
        else:
            goal_diffs = []
            home_wins = 0
            for _, m in h2h.iterrows():
                if m["home_team"] == home:
                    goal_diffs.append(m["home_goals"] - m["away_goals"])
                    if m["result"] == "H":
                        home_wins += 1
                else:
                    goal_diffs.append(m["away_goals"] - m["home_goals"])
                    if m["result"] == "A":
                        home_wins += 1
            avg_goal_diff_h2h = float(sum(goal_diffs) / len(goal_diffs)) if goal_diffs else 0.0
            h2h_home_winrate = float(home_wins / len(h2h)) if len(h2h) else 0.5
            print(f"   📊 H2H: avg_diff={avg_goal_diff_h2h:.2f}, home_winrate={h2h_home_winrate:.2f}")

        # Venue-aware recent form within same league
        home_form_winrate, away_form_winrate = _venue_aware_form(history, home, away, league, window=10)
        print(f"   📈 Form: home={home_form_winrate:.2f}, away={away_form_winrate:.2f}")

        rows.append({
            "match_date": fixture_date,
            "home_team": home,
            "away_team": away,
            "league": league,
            "league_code": league_code,
            "avg_goal_diff_h2h": avg_goal_diff_h2h,
            "h2h_home_winrate": h2h_home_winrate,
            "home_form_winrate": home_form_winrate,
            "away_form_winrate": away_form_winrate,
        })

    return pd.DataFrame(rows)

def predict_fixtures(leagues=None):
    """
    Generate predictions for upcoming fixtures.
    
    Args:
        leagues: List of league codes to predict (e.g., ["PL", "ELC"])
    """
    if leagues is None:
        leagues = ["PL", "ELC"]  # Default to both Premier League and Championship
    
    print(f"⚽ Loading model and generating predictions for: {leagues}")

    # Load model, history, fixtures
    model = joblib.load(MODEL_PATH)
    history = fetch_historic_results_multi(leagues=["PL", "CHAMP"])  # Internal league names
    fixtures = fetch_upcoming_fixtures(league_codes=leagues, limit=20)

    print(f"\n📅 Upcoming Fixtures ({len(fixtures)} total):")
    if not fixtures.empty:
        view_cols = ["home_team", "away_team", "utc_date", "league_name"]
        print(fixtures[view_cols].to_string(index=False))
    else:
        print("No upcoming fixtures found!")
        return

    features_df = build_prediction_features(fixtures, history)

    if features_df.empty:
        print("😕 No fixtures with sufficient data to predict.")
        return

    print(f"\n🔮 Making predictions for {len(features_df)} matches...")

    # Predict
    X = features_df[["avg_goal_diff_h2h", "h2h_home_winrate", "home_form_winrate", "away_form_winrate"]]
    predicted_classes = model.predict(X)
    predicted_probas = model.predict_proba(X)

    # Decorate outputs
    features_df["predicted_result"] = [LABELS[i] for i in predicted_classes]
    probas = pd.DataFrame(predicted_probas, columns=LABELS)
    features_df = pd.concat([features_df.reset_index(drop=True), probas], axis=1)

    # Confidence helpers
    features_df["max_proba"] = features_df[LABELS].max(axis=1)
    features_df["confidence_label"] = [
        f"{p:.2f} ({LABELS[c]})" for p, c in zip(features_df["max_proba"], predicted_classes)
    ]
    features_df["prob_label"] = (
        features_df["home_win"].map(lambda x: f"H:{x:.2f}") + " • " +
        features_df["draw"].map(lambda x: f"D:{x:.2f}") + " • " +
        features_df["away_win"].map(lambda x: f"A:{x:.2f}")
    )

    # Filter confident for terminal print; save all to CSV
    confident = features_df[features_df["max_proba"] >= CONFIDENCE_THRESHOLD].copy()
    if not confident.empty:
        confident = confident.sort_values("max_proba", ascending=False)
        print(f"\n🎯 Confident Predictions (≥{CONFIDENCE_THRESHOLD:.0%} confidence):")
        display_cols = ["match_date", "home_team", "away_team", "league", "predicted_result", "confidence_label"]
        print(confident[display_cols].to_string(index=False))
        
        # Show league breakdown of confident picks
        league_breakdown = confident["league"].value_counts()
        print(f"\n📊 Confident picks by league: {dict(league_breakdown)}")
    else:
        print(f"\nℹ️ No picks ≥ {CONFIDENCE_THRESHOLD:.2f} confidence.")
        print("📊 All predictions (top 10 by confidence):")
        top_picks = features_df.nlargest(10, "max_proba")
        display_cols = ["home_team", "away_team", "league", "predicted_result", "confidence_label"]
        print(top_picks[display_cols].to_string(index=False))

    # Save predictions
    os.makedirs("data/predictions", exist_ok=True)
    out_cols = [
        "match_date", "home_team", "away_team", "league", "league_code",
        "predicted_result", "confidence_label",
        "home_win", "draw", "away_win", "prob_label",
        "avg_goal_diff_h2h", "h2h_home_winrate", "home_form_winrate", "away_form_winrate",
    ]
    features_df[out_cols].to_csv("data/predictions/latest_predictions.csv", index=False)
    print(f"\n✅ All {len(features_df)} predictions saved to data/predictions/latest_predictions.csv")

def predict_premier_league_only():
    """Convenience function for Premier League predictions only"""
    predict_fixtures(["PL"])

def predict_championship_only():
    """Convenience function for Championship predictions only"""
    predict_fixtures(["ELC"])

def predict_both_leagues():
    """Convenience function for both leagues"""
    predict_fixtures(["PL", "ELC"])

if __name__ == "__main__":
    print("🚀 Multi-league prediction system")
    print("Predicting both Premier League and Championship...")
    predict_both_leagues()