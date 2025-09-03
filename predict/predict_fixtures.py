import os
import pandas as pd
import numpy as np
import joblib
from fetch.fetch_fixtures import fetch_upcoming_fixtures
from fetch.fetch_historic_results import fetch_historic_results_multi
from utils.team_name_map import normalize_team
from utils.league_utils import _canon, LEAGUE_ALIASES, to_history_code

MODEL_PATH = "models/ensemble_model.pkl"
SCALER_PATH = "models/scaler.pkl"
METADATA_PATH = "models/metadata.pkl"

# Class label order used by the model
LABELS = ["home_win", "draw", "away_win"]

# Only show results above this confidence in the terminal & CSV
CONFIDENCE_THRESHOLD = 0.60

# Some internal datasets use "CHAMP" instead of "ELC"
INTERNAL_FETCH_MAP = {
    "PL": "PL",
    "ELC": "CHAMP",
    "BL1": "BL1",
    "SA": "SA",
    "PD": "PD",
}

def _canon(x: str) -> str:
    if x is None:
        return "UNKNOWN"
    key = str(x).strip().upper()
    return LEAGUE_ALIASES.get(key, key)

def _to_history_code(code: str) -> str:
    """Map canonical code to the label used by the historic fetcher/history."""
    return INTERNAL_FETCH_MAP.get(code, code)

# ---------------- Enhanced Feature Builders ----------------
def _league_filter(history: pd.DataFrame, league_in_history: str) -> pd.DataFrame:
    """Return history restricted to a league label (as used in history), else original history."""
    if "league" in history.columns and league_in_history in history["league"].values:
        return history[history["league"] == league_in_history]
    return history

def build_prediction_features(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Create ENHANCED prediction feature rows for each upcoming fixture."""
    fixtures = fixtures.copy()
    
    # Ensure canonical league codes
    if "league_code" in fixtures.columns:
        fixtures["league_code"] = fixtures["league_code"].map(_canon)
    elif "league" in fixtures.columns:
        fixtures["league_code"] = fixtures["league"].map(_canon)
    else:
        fixtures["league_code"] = "PL"

    fixtures.rename(columns={"utc_date": "date"}, inplace=True)
    fixtures["date"] = pd.to_datetime(fixtures["date"], utc=True, errors="coerce")

    # Normalize team names in BOTH datasets
    for col in ["home_team", "away_team"]:
        fixtures[col] = fixtures[col].apply(normalize_team)
        history[col] = history[col].apply(normalize_team)

    rows = []
    h2h_window = 5
    form_window = 10
    
    for _, f in fixtures.iterrows():
        fixture_date = pd.to_datetime(f["date"], utc=True, errors="coerce")
        if pd.isna(fixture_date):
            continue
        home = f["home_team"]
        away = f["away_team"]

        league_code = str(f.get("league_code", "PL"))
        league_canon = _canon(league_code)
        league_hist = _to_history_code(league_canon)

        print(f"🔎 {home} vs {away} ({league_canon})")

        # Filter history to same league
        league_history = _league_filter(history, league_hist)
        past_matches = league_history[league_history["date"] < fixture_date]
        
        # --- H2H Features with draw focus ---
        h2h = past_matches[
            (
                ((past_matches["home_team"] == home) & (past_matches["away_team"] == away)) |
                ((past_matches["home_team"] == away) & (past_matches["away_team"] == home))
            )
        ].sort_values("date").tail(h2h_window)

        print(f"   Found {len(h2h)} H2H matches")

        if len(h2h) < 2:
            avg_goal_diff = 0.0
            h2h_home_winrate = 0.5
            h2h_draw_rate = 0.25
            h2h_total_goals = 2.5
        else:
            goal_diffs, home_wins, draws = [], 0, 0
            total_goals_h2h = []
            for _, match in h2h.iterrows():
                if match["home_team"] == home:
                    goal_diffs.append(match["home_goals"] - match["away_goals"])
                    total_goals_h2h.append(match["home_goals"] + match["away_goals"])
                    if match["result"] == "H":
                        home_wins += 1
                    elif match["result"] == "D":
                        draws += 1
                else:
                    goal_diffs.append(match["away_goals"] - match["home_goals"])
                    total_goals_h2h.append(match["home_goals"] + match["away_goals"])
                    if match["result"] == "A":
                        home_wins += 1
                    elif match["result"] == "D":
                        draws += 1
            avg_goal_diff = float(sum(goal_diffs) / len(goal_diffs)) if goal_diffs else 0.0
            h2h_home_winrate = float(home_wins / len(h2h)) if len(h2h) else 0.5
            h2h_draw_rate = float(draws / len(h2h)) if len(h2h) else 0.25
            h2h_total_goals = float(np.mean(total_goals_h2h)) if total_goals_h2h else 2.5

        # --- Venue-aware form ---
        home_recent = past_matches[past_matches["home_team"] == home].sort_values("date").tail(form_window)
        away_recent = past_matches[past_matches["away_team"] == away].sort_values("date").tail(form_window)

        home_form_wins = (home_recent["result"] == "H").sum()
        away_form_wins = (away_recent["result"] == "A").sum()
        home_form_winrate = float(home_form_wins / len(home_recent)) if len(home_recent) else 0.5
        away_form_winrate = float(away_form_wins / len(away_recent)) if len(away_recent) else 0.5

        # --- Draw rates ---
        home_all_matches = past_matches[
            (past_matches["home_team"] == home) | (past_matches["away_team"] == home)
        ].tail(form_window * 2)
        
        away_all_matches = past_matches[
            (past_matches["home_team"] == away) | (past_matches["away_team"] == away)
        ].tail(form_window * 2)
        
        home_draw_rate = float((home_all_matches["result"] == "D").mean()) if len(home_all_matches) else 0.25
        away_draw_rate = float((away_all_matches["result"] == "D").mean()) if len(away_all_matches) else 0.25
        combined_draw_rate = (home_draw_rate + away_draw_rate) / 2

        # --- Goals features ---
        if len(home_recent):
            home_avg_goals_scored = float(home_recent["home_goals"].mean())
            home_avg_goals_conceded = float(home_recent["away_goals"].mean())
            home_total_goals_avg = float((home_recent["home_goals"] + home_recent["away_goals"]).mean())
        else:
            home_avg_goals_scored = 1.0
            home_avg_goals_conceded = 1.0
            home_total_goals_avg = 2.5

        if len(away_recent):
            away_avg_goals_scored = float(away_recent["away_goals"].mean())
            away_avg_goals_conceded = float(away_recent["home_goals"].mean())
            away_total_goals_avg = float((away_recent["home_goals"] + away_recent["away_goals"]).mean())
        else:
            away_avg_goals_scored = 1.0
            away_avg_goals_conceded = 1.0
            away_total_goals_avg = 2.5

        # --- Differential features (draw indicators) ---
        form_differential = float(abs(home_form_winrate - away_form_winrate))
        goals_differential = float(abs(home_avg_goals_scored - away_avg_goals_scored))
        expected_total_goals = float(home_avg_goals_scored + away_avg_goals_scored)

        # --- League context ---
        league_recent = past_matches.tail(100)
        if len(league_recent):
            league_avg_goals = float((league_recent["home_goals"] + league_recent["away_goals"]).mean())
            league_draw_rate = float((league_recent["result"] == "D").mean())
            league_home_adv = float((league_recent["result"] == "H").mean()) - float((league_recent["result"] == "A").mean())
        else:
            league_avg_goals = 2.5
            league_draw_rate = 0.25
            league_home_adv = 0.1

        # --- Momentum ---
        home_last_3 = home_recent.tail(3)
        away_last_3 = away_recent.tail(3)
        
        if len(home_last_3):
            home_recent_points = sum([3 if r == "H" else (1 if r == "D" else 0) for r in home_last_3["result"]])
            home_momentum = float(home_recent_points / 9.0)
        else:
            home_momentum = 0.5
            
        if len(away_last_3):
            away_recent_points = sum([3 if r == "A" else (1 if r == "D" else 0) for r in away_last_3["result"]])
            away_momentum = float(away_recent_points / 9.0)
        else:
            away_momentum = 0.5
            
        momentum_differential = float(abs(home_momentum - away_momentum))

        # --- Low-scoring indicators ---
        is_low_scoring = float(expected_total_goals < league_avg_goals * 0.85)
        is_defensive_match = float(home_avg_goals_conceded < 1.2 and away_avg_goals_conceded < 1.2)

        print(f"   📈 Form: home={home_form_winrate:.2f}, away={away_form_winrate:.2f}, diff={form_differential:.2f}")
        print(f"   🎯 Draw indicators: combined_rate={combined_draw_rate:.2f}, h2h_draws={h2h_draw_rate:.2f}")

        rows.append({
            "match_date": fixture_date,
            "home_team": home,
            "away_team": away,
            "league": league_canon,
            "league_code": league_canon,
            # Core features
            "avg_goal_diff_h2h": avg_goal_diff,
            "h2h_home_winrate": h2h_home_winrate,
            "home_form_winrate": home_form_winrate,
            "away_form_winrate": away_form_winrate,
            "home_avg_goals_scored": home_avg_goals_scored,
            "home_avg_goals_conceded": home_avg_goals_conceded,
            "away_avg_goals_scored": away_avg_goals_scored,
            "away_avg_goals_conceded": away_avg_goals_conceded,
            # Draw features
            "h2h_draw_rate": h2h_draw_rate,
            "h2h_total_goals": h2h_total_goals,
            "home_draw_rate": home_draw_rate,
            "away_draw_rate": away_draw_rate,
            "combined_draw_rate": combined_draw_rate,
            "form_differential": form_differential,
            "goals_differential": goals_differential,
            "expected_total_goals": expected_total_goals,
            "home_total_goals_avg": home_total_goals_avg,
            "away_total_goals_avg": away_total_goals_avg,
            "league_avg_goals": league_avg_goals,
            "league_draw_rate": league_draw_rate,
            "league_home_adv": league_home_adv,
            "home_momentum": home_momentum,
            "away_momentum": away_momentum,
            "momentum_differential": momentum_differential,
            "is_low_scoring": is_low_scoring,
            "is_defensive_match": is_defensive_match,
        })

    return pd.DataFrame(rows)

def enhance_with_odds(features_df):
    """Add odds-based features to predictions."""
    try:
        # Try to load latest odds
        if os.path.exists("data/odds/latest_odds.csv"):
            odds_df = pd.read_csv("data/odds/latest_odds.csv")
            print(f"📊 Found {len(odds_df)} matches with odds data")
        else:
            print("⚠️ No odds data found - using default values")
            # Add default odds features
            features_df["home_true_prob"] = 0.40
            features_df["draw_true_prob"] = 0.27
            features_df["away_true_prob"] = 0.33
            features_df["market_draw_confidence"] = 0.27
            features_df["market_favorite_confidence"] = 0.40
            features_df["market_competitiveness"] = 0.5
            features_df["odds_spread"] = 1.0
            return features_df
        
        # Process odds
        merged_count = 0
        for idx, row in features_df.iterrows():
            # Try to find matching odds
            match_odds = odds_df[
                (odds_df["home_team"] == row["home_team"]) & 
                (odds_df["away_team"] == row["away_team"])
            ]
            
            if not match_odds.empty:
                odds_row = match_odds.iloc[0]
                
                # Calculate probabilities from odds
                if all(col in odds_row for col in ["home_odds", "draw_odds", "away_odds"]):
                    home_implied = 1 / odds_row["home_odds"] if odds_row["home_odds"] > 0 else 0.33
                    draw_implied = 1 / odds_row["draw_odds"] if odds_row["draw_odds"] > 0 else 0.33
                    away_implied = 1 / odds_row["away_odds"] if odds_row["away_odds"] > 0 else 0.33
                    
                    total = home_implied + draw_implied + away_implied
                    features_df.at[idx, "home_true_prob"] = home_implied / total
                    features_df.at[idx, "draw_true_prob"] = draw_implied / total
                    features_df.at[idx, "away_true_prob"] = away_implied / total
                    features_df.at[idx, "market_draw_confidence"] = draw_implied / total
                    features_df.at[idx, "market_favorite_confidence"] = max(home_implied, away_implied) / total
                    features_df.at[idx, "market_competitiveness"] = 1 - abs(home_implied - away_implied) / total
                    features_df.at[idx, "odds_spread"] = abs(odds_row["home_odds"] - odds_row["away_odds"])
                    merged_count += 1
                else:
                    # Use defaults if odds columns missing
                    features_df.at[idx, "home_true_prob"] = 0.40
                    features_df.at[idx, "draw_true_prob"] = 0.27
                    features_df.at[idx, "away_true_prob"] = 0.33
                    features_df.at[idx, "market_draw_confidence"] = 0.27
                    features_df.at[idx, "market_favorite_confidence"] = 0.40
                    features_df.at[idx, "market_competitiveness"] = 0.5
                    features_df.at[idx, "odds_spread"] = 1.0
            else:
                # No matching odds - use defaults
                features_df.at[idx, "home_true_prob"] = 0.40
                features_df.at[idx, "draw_true_prob"] = 0.27
                features_df.at[idx, "away_true_prob"] = 0.33
                features_df.at[idx, "market_draw_confidence"] = 0.27
                features_df.at[idx, "market_favorite_confidence"] = 0.40
                features_df.at[idx, "market_competitiveness"] = 0.5
                features_df.at[idx, "odds_spread"] = 1.0
        
        print(f"✅ Enhanced {merged_count}/{len(features_df)} predictions with live odds")
        
    except Exception as e:
        print(f"⚠️ Could not load odds data: {e}")
        # Add default odds features
        features_df["home_true_prob"] = 0.40
        features_df["draw_true_prob"] = 0.27
        features_df["away_true_prob"] = 0.33
        features_df["market_draw_confidence"] = 0.27
        features_df["market_favorite_confidence"] = 0.40
        features_df["market_competitiveness"] = 0.5
        features_df["odds_spread"] = 1.0
    
    return features_df

# ---------------- Main prediction entrypoint ----------------
def predict_fixtures(leagues=None):
    """Generate predictions for upcoming fixtures with enhanced features."""
    if leagues is None:
        leagues = ["PL", "ELC"]

    leagues_canon = sorted({_canon(l) for l in leagues})
    print(f"⚽ Loading model and generating predictions for: {leagues_canon}")

    # Load model and metadata
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
        expected_features = metadata.get("features", None)
        
        if expected_features:
            print(f"📋 Model expects {len(expected_features)} features")
            has_draw = any("draw" in f or "differential" in f for f in expected_features)
            has_odds = any("true_prob" in f or "market" in f for f in expected_features)
            print(f"   {'✅' if has_draw else '❌'} Draw features")
            print(f"   {'✅' if has_odds else '❌'} Odds features")
        
        # Load scaler if available
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load history
    fetch_leagues = sorted({_to_history_code(l) for l in leagues_canon})
    history = fetch_historic_results_multi(leagues=fetch_leagues, enhanced_output=False)

    # Fetch fixtures
    fixtures = fetch_upcoming_fixtures(league_codes=leagues_canon, limit=20)

    print(f"\n📅 Upcoming Fixtures ({len(fixtures)} total):")
    if not fixtures.empty:
        view_cols = [c for c in ["home_team", "away_team", "utc_date", "league_name"] if c in fixtures.columns]
        print(fixtures[view_cols].head(10).to_string(index=False))
    else:
        print("No upcoming fixtures found!")
        return

    # Build enhanced features
    features_df = build_prediction_features(fixtures, history)
    if features_df.empty:
        print("😕 No fixtures with sufficient data to predict.")
        return

    # Add odds features
    features_df = enhance_with_odds(features_df)

    print(f"\n🔮 Making predictions for {len(features_df)} matches...")
    
    # Use expected features if we know them, otherwise use all numeric columns
    if expected_features:
        # Ensure all expected features exist
        missing = [f for f in expected_features if f not in features_df.columns]
        if missing:
            print(f"⚠️ Missing features: {missing[:5]}...")
            print("   Adding with default values...")
            for feat in missing:
                features_df[feat] = 0.0  # Default value
        X = features_df[expected_features]
    else:
        # Fallback to core features
        feature_cols = [
            "avg_goal_diff_h2h", "h2h_home_winrate", "home_form_winrate", "away_form_winrate",
            "home_avg_goals_scored", "home_avg_goals_conceded", "away_avg_goals_scored", "away_avg_goals_conceded"
        ]
        X = features_df[feature_cols]

    # Scale if scaler available (for neural network)
    if scaler:
        X_scaled = scaler.transform(X)
        # Note: VotingClassifier handles this internally, but keeping for reference
    
    # Predict
    predicted_classes = model.predict(X)
    predicted_probas = model.predict_proba(X)

    # Add predictions to dataframe
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

    # Show summary
    print("\n📊 Predictions by league:", dict(features_df["league"].value_counts()))
    
    # Check draw predictions
    draw_preds = (features_df["predicted_result"] == "draw").sum()
    print(f"🎯 Draw predictions: {draw_preds}/{len(features_df)} ({draw_preds/len(features_df)*100:.1f}%)")

    # Filter confident predictions
    confident = features_df[features_df["max_proba"] >= CONFIDENCE_THRESHOLD].copy()
    if not confident.empty:
        confident = confident.sort_values("max_proba", ascending=False)
        print(f"\n🎯 Confident Predictions (≥{CONFIDENCE_THRESHOLD:.0%} confidence):")
        display_cols = ["match_date", "home_team", "away_team", "league", "predicted_result", "confidence_label"]
        print(confident[display_cols].to_string(index=False))

        # Show if any confident draws
        confident_draws = confident[confident["predicted_result"] == "draw"]
        if not confident_draws.empty:
            print(f"\n✨ Found {len(confident_draws)} confident DRAW predictions!")
    else:
        print(f"\nℹ️ No picks ≥ {CONFIDENCE_THRESHOLD:.2f} confidence.")
        print("📊 Top 10 predictions by confidence:")
        top_picks = features_df.nlargest(10, "max_proba")
        display_cols = ["home_team", "away_team", "league", "predicted_result", "confidence_label"]
        print(top_picks[display_cols].to_string(index=False))

    # Save predictions
    os.makedirs("data/predictions", exist_ok=True)
    features_df.to_csv("data/predictions/latest_predictions.csv", index=False)
    print(f"\n✅ All {len(features_df)} predictions saved to data/predictions/latest_predictions.csv")

# ---------------- Convenience wrappers ----------------
def predict_premier_league_only():
    predict_fixtures(["PL"])

def predict_championship_only():
    predict_fixtures(["ELC"])

def predict_both_leagues():
    predict_fixtures(["PL", "ELC"])

def predict_top5():
    """PL, ELC, BL1, SA, PD"""
    predict_fixtures(["PL", "ELC", "BL1", "SA", "PD"])

def predict_all_supported():
    predict_top5()

if __name__ == "__main__":
    print("🚀 Enhanced prediction system with draw features")
    print("Predicting Top 5 leagues...")
    predict_top5()