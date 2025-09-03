import os
import pandas as pd
import numpy as np
from fetch.fetch_historic_results import fetch_historic_results_multi

# ---- Canonical league codes and aliases ----
LEAGUE_ALIASES = {
    "PL": "PL",
    "ELC": "ELC",
    "BL1": "BL1",
    "SA": "SA",
    "PD": "PD",
    "CHAMP": "ELC",
    "EFL_CHAMPIONSHIP": "ELC",
    "CHAMPIONSHIP": "ELC",
    "PREMIER LEAGUE": "PL",
    "EPL": "PL",
    "BUNDESLIGA": "BL1",
    "SERIE A": "SA",
    "LA LIGA": "PD",
    "PRIMERA DIVISION": "PD",
}

def _canon(code_or_name: str) -> str:
    if code_or_name is None:
        return "UNKNOWN"
    key = str(code_or_name).strip().upper()
    return LEAGUE_ALIASES.get(key, key)

def build_features(df, h2h_window=5, form_window=10, separate_by_league=True):
    """
    ENHANCED: Generate features including draw-predictive features AND odds integration.
    
    This version includes:
    1. Original features (H2H, form, goals)
    2. NEW draw-predictive features
    3. Odds-based features (if available in the historical data)
    """
    df = df.sort_values("date").copy()
    if "league" not in df.columns:
        df["league"] = "UNKNOWN"
    else:
        df["league"] = df["league"].map(_canon)

    df["outcome_code"] = df["result"].map({"H": 1, "D": 0, "A": -1})

    feature_rows = []
    
    # Check if we have odds data in the historical results
    has_odds = all(col in df.columns for col in ['B365H', 'B365D', 'B365A'])
    if has_odds:
        print("✅ Found betting odds in historical data - will include odds features!")
    else:
        print("ℹ️ No betting odds in historical data - using form-based features only")

    for idx, row in df.iterrows():
        current_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        league = row.get("league", "UNKNOWN")

        # Only consider matches strictly before this match
        past_matches = df[df["date"] < current_date]

        # Optionally restrict to same league
        if separate_by_league and league != "UNKNOWN":
            past_matches = past_matches[past_matches["league"] == league]

        # --- H2H Features (existing + enhanced) ---
        h2h = past_matches[
            ((past_matches["home_team"] == home) & (past_matches["away_team"] == away)) |
            ((past_matches["home_team"] == away) & (past_matches["away_team"] == home))
        ].tail(h2h_window)

        if len(h2h) < 2:
            avg_goal_diff = 0.0
            h2h_home_winrate = 0.5
            h2h_draw_rate = 0.25  # NEW: Default draw rate
            h2h_total_goals = 2.5  # NEW: Average total goals in H2H
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
            avg_goal_diff = sum(goal_diffs) / len(goal_diffs) if goal_diffs else 0.0
            h2h_home_winrate = home_wins / len(h2h) if len(h2h) else 0.5
            h2h_draw_rate = draws / len(h2h) if len(h2h) else 0.25
            h2h_total_goals = np.mean(total_goals_h2h) if total_goals_h2h else 2.5

        # --- Venue-aware form (existing) ---
        home_recent = past_matches[past_matches["home_team"] == home].sort_values("date").tail(form_window)
        away_recent = past_matches[past_matches["away_team"] == away].sort_values("date").tail(form_window)

        home_form_wins = (home_recent["result"] == "H").sum()
        away_form_wins = (away_recent["result"] == "A").sum()

        home_form_winrate = (home_form_wins / len(home_recent)) if len(home_recent) else 0.5
        away_form_winrate = (away_form_wins / len(away_recent)) if len(away_recent) else 0.5

        # --- NEW: Draw rates for each team ---
        # How often does each team draw in general?
        home_all_matches = past_matches[
            (past_matches["home_team"] == home) | (past_matches["away_team"] == home)
        ].tail(form_window * 2)
        
        away_all_matches = past_matches[
            (past_matches["home_team"] == away) | (past_matches["away_team"] == away)
        ].tail(form_window * 2)
        
        home_draw_rate = (home_all_matches["result"] == "D").mean() if len(home_all_matches) else 0.25
        away_draw_rate = (away_all_matches["result"] == "D").mean() if len(away_all_matches) else 0.25
        combined_draw_rate = (home_draw_rate + away_draw_rate) / 2

        # --- Goals per match (enhanced) ---
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

        # --- NEW: Strength differential features ---
        # When teams are evenly matched, draws are more likely
        form_differential = abs(home_form_winrate - away_form_winrate)
        goals_differential = abs(home_avg_goals_scored - away_avg_goals_scored)
        expected_total_goals = home_avg_goals_scored + away_avg_goals_scored
        
        # --- NEW: League context ---
        league_recent = past_matches[past_matches["league"] == league].tail(100)
        if len(league_recent):
            league_avg_goals = float((league_recent["home_goals"] + league_recent["away_goals"]).mean())
            league_draw_rate = float((league_recent["result"] == "D").mean())
            league_home_adv = float((league_recent["result"] == "H").mean()) - float((league_recent["result"] == "A").mean())
        else:
            league_avg_goals = 2.5
            league_draw_rate = 0.25
            league_home_adv = 0.1

        # --- NEW: Recent form momentum ---
        home_last_3 = home_recent.tail(3)
        away_last_3 = away_recent.tail(3)
        
        if len(home_last_3):
            home_recent_points = sum([3 if r == "H" else (1 if r == "D" else 0) for r in home_last_3["result"]])
            home_momentum = home_recent_points / 9.0
        else:
            home_momentum = 0.5
            
        if len(away_last_3):
            away_recent_points = sum([3 if r == "A" else (1 if r == "D" else 0) for r in away_last_3["result"]])
            away_momentum = away_recent_points / 9.0
        else:
            away_momentum = 0.5
            
        momentum_differential = abs(home_momentum - away_momentum)

        # --- NEW: Low-scoring team indicators ---
        is_low_scoring = expected_total_goals < league_avg_goals * 0.85
        is_defensive_match = (home_avg_goals_conceded < 1.2 and away_avg_goals_conceded < 1.2)

        # --- ODDS-BASED FEATURES (if available) ---
        if has_odds and not pd.isna(row.get('B365H')):
            # Calculate implied probabilities from odds
            home_odds = row.get('B365H', 3.0)
            draw_odds = row.get('B365D', 3.3)
            away_odds = row.get('B365A', 3.0)
            
            # Convert to implied probabilities
            home_implied = 1 / home_odds if home_odds > 0 else 0.33
            draw_implied = 1 / draw_odds if draw_odds > 0 else 0.33
            away_implied = 1 / away_odds if away_odds > 0 else 0.33
            
            # Normalize (remove overround)
            total_implied = home_implied + draw_implied + away_implied
            home_true_prob = home_implied / total_implied
            draw_true_prob = draw_implied / total_implied
            away_true_prob = away_implied / total_implied
            
            # Market-derived features
            market_draw_confidence = draw_true_prob
            market_favorite_confidence = max(home_true_prob, away_true_prob)
            market_competitiveness = 1 - abs(home_true_prob - away_true_prob)
            odds_spread = max(home_odds, away_odds) - min(home_odds, away_odds)
        else:
            # Fallback values when no odds available
            home_true_prob = 0.45 if home_form_winrate > away_form_winrate else 0.30
            away_true_prob = 0.45 if away_form_winrate > home_form_winrate else 0.30
            draw_true_prob = 1.0 - home_true_prob - away_true_prob
            market_draw_confidence = draw_true_prob
            market_favorite_confidence = max(home_true_prob, away_true_prob)
            market_competitiveness = 1 - abs(home_true_prob - away_true_prob)
            odds_spread = 1.0

        feature_rows.append({
            "home_team": home,
            "away_team": away,
            "match_date": current_date,
            "league": league,
            
            # Original features
            "avg_goal_diff_h2h": avg_goal_diff,
            "h2h_home_winrate": h2h_home_winrate,
            "home_form_winrate": home_form_winrate,
            "away_form_winrate": away_form_winrate,
            "home_avg_goals_scored": home_avg_goals_scored,
            "home_avg_goals_conceded": home_avg_goals_conceded,
            "away_avg_goals_scored": away_avg_goals_scored,
            "away_avg_goals_conceded": away_avg_goals_conceded,
            
            # NEW draw-focused features
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
            "is_low_scoring": float(is_low_scoring),
            "is_defensive_match": float(is_defensive_match),
            
            # Odds-based features
            "home_true_prob": home_true_prob,
            "draw_true_prob": draw_true_prob,
            "away_true_prob": away_true_prob,
            "market_draw_confidence": market_draw_confidence,
            "market_favorite_confidence": market_favorite_confidence,
            "market_competitiveness": market_competitiveness,
            "odds_spread": odds_spread,
            
            # Target
            "result": row["result"],
        })

        if len(feature_rows) % 1000 == 0:
            print(f"  📊 Processed {len(feature_rows)} matches...")

    return pd.DataFrame(feature_rows)

def build_features_by_league(leagues=None):
    """Build enhanced features for specific leagues."""
    if leagues is None:
        leagues = ["PL", "ELC", "BL1", "SA", "PD"]  # All leagues by default

    # Normalize requested leagues to canonical codes
    leagues_canon = sorted({_canon(l) for l in leagues if l})

    print(f"🔄 Building ENHANCED features with draw + odds focus...")
    print(f"📋 Leagues: {leagues_canon}")

    # Fetch raw data
    df_raw = fetch_historic_results_multi(leagues=leagues_canon)

    if df_raw.empty:
        print("❌ No historical data found!")
        return pd.DataFrame()

    # Normalize league codes
    if "league" in df_raw.columns:
        df_raw["league"] = df_raw["league"].map(_canon)
    else:
        df_raw["league"] = "UNKNOWN"

    print(f"📊 Raw data: {len(df_raw)} matches across {df_raw['league'].nunique()} leagues")
    
    # Show result distribution
    result_dist = df_raw["result"].value_counts()
    total = len(df_raw)
    print(f"📈 Overall result distribution:")
    print(f"   Home wins: {result_dist.get('H', 0)} ({result_dist.get('H', 0)/total*100:.1f}%)")
    print(f"   Draws: {result_dist.get('D', 0)} ({result_dist.get('D', 0)/total*100:.1f}%)")
    print(f"   Away wins: {result_dist.get('A', 0)} ({result_dist.get('A', 0)/total*100:.1f}%)")

    # Filter to requested leagues
    df_raw = df_raw[df_raw["league"].isin(leagues_canon)]

    if df_raw.empty:
        print("❌ No data for requested leagues.")
        return pd.DataFrame()

    # Build enhanced features
    print("\n🚀 Building enhanced feature set with draw + odds features...")
    df_features = build_features(df_raw, separate_by_league=True)

    print(f"✅ Generated {len(df_features)} feature rows")
    
    # Analyze feature quality
    if not df_features.empty:
        print(f"\n📊 Feature statistics:")
        print(f"   Draw-focused features:")
        print(f"     • Avg combined draw rate: {df_features['combined_draw_rate'].mean():.3f}")
        print(f"     • Avg H2H draw rate: {df_features['h2h_draw_rate'].mean():.3f}")
        print(f"     • Avg form differential: {df_features['form_differential'].mean():.3f}")
        print(f"     • Avg expected total goals: {df_features['expected_total_goals'].mean():.2f}")
        print(f"   Odds-based features:")
        print(f"     • Avg market draw confidence: {df_features['market_draw_confidence'].mean():.3f}")
        print(f"     • Avg market competitiveness: {df_features['market_competitiveness'].mean():.3f}")
        
        # Analyze draw correlation
        draw_matches = df_features[df_features["result"] == "D"]
        non_draw_matches = df_features[df_features["result"] != "D"]
        
        print(f"\n🎯 Draw indicator analysis:")
        print(f"   When match ENDS in draw:")
        print(f"     • Form differential: {draw_matches['form_differential'].mean():.3f}")
        print(f"     • Combined draw rate: {draw_matches['combined_draw_rate'].mean():.3f}")
        print(f"     • Market draw confidence: {draw_matches['market_draw_confidence'].mean():.3f}")
        print(f"   When match DOESN'T end in draw:")
        print(f"     • Form differential: {non_draw_matches['form_differential'].mean():.3f}")
        print(f"     • Combined draw rate: {non_draw_matches['combined_draw_rate'].mean():.3f}")
        print(f"     • Market draw confidence: {non_draw_matches['market_draw_confidence'].mean():.3f}")
        
        # Show which features will be most useful
        print(f"\n💡 Feature insights:")
        form_diff_gap = abs(draw_matches['form_differential'].mean() - non_draw_matches['form_differential'].mean())
        draw_rate_gap = abs(draw_matches['combined_draw_rate'].mean() - non_draw_matches['combined_draw_rate'].mean())
        market_gap = abs(draw_matches['market_draw_confidence'].mean() - non_draw_matches['market_draw_confidence'].mean())
        
        print(f"   • Form differential discriminative power: {form_diff_gap:.3f}")
        print(f"   • Draw rate discriminative power: {draw_rate_gap:.3f}")
        print(f"   • Market confidence discriminative power: {market_gap:.3f}")

    return df_features

if __name__ == "__main__":
    print("📊 Building ENHANCED feature matrix with draw + odds features...\n")

    # Include all leagues for better training
    requested = ["PL", "ELC", "BL1", "SA", "PD"]

    df_features = build_features_by_league(requested)

    if df_features.empty:
        print("❌ No features generated!")
        raise SystemExit(1)

    print(f"\n📋 Feature columns ({len(df_features.columns)} total):")
    
    feature_groups = {
        "Original": ["avg_goal_diff_h2h", "h2h_home_winrate", "home_form_winrate", 
                    "away_form_winrate", "home_avg_goals_scored", "home_avg_goals_conceded",
                    "away_avg_goals_scored", "away_avg_goals_conceded"],
        "Draw-focused": ["h2h_draw_rate", "combined_draw_rate", "form_differential",
                        "goals_differential", "expected_total_goals", "league_draw_rate",
                        "momentum_differential", "is_low_scoring", "is_defensive_match"],
        "Odds-based": ["home_true_prob", "draw_true_prob", "away_true_prob",
                      "market_draw_confidence", "market_competitiveness", "odds_spread"]
    }
    
    for group_name, features in feature_groups.items():
        available = [f for f in features if f in df_features.columns]
        print(f"\n   {group_name} ({len(available)}/{len(features)}):")
        for feat in available:
            if feat in df_features.columns:
                print(f"     ✅ {feat}")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/training_data.csv"
    df_features.to_csv(out_path, index=False)
    print(f"\n✅ Enhanced feature data saved to {out_path}")
    print(f"📊 Total feature rows: {len(df_features)}")
    print(f"🎯 Total features: {len([c for c in df_features.columns if c not in ['home_team', 'away_team', 'match_date', 'league', 'result']])} predictive features!")