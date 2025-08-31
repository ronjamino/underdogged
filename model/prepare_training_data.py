import pandas as pd
from fetch.fetch_historic_results import fetch_historic_results_multi

def build_features(df, h2h_window=5, form_window=10, separate_by_league=True):
    """
    Generate features from historic match results including head-to-head form and recent team form.
    
    Args:
        df: Historic match results dataframe
        h2h_window: Number of previous H2H matches to consider
        form_window: Number of recent matches for form calculation  
        separate_by_league: If True, only consider matches within same league for H2H and form
    """
    df = df.sort_values("date")

    # Create result as 1 (home win), 0 (draw), -1 (away win)
    df["outcome_code"] = df["result"].map({"H": 1, "D": 0, "A": -1})

    feature_rows = []

    for idx, row in df.iterrows():
        current_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        league = row.get("league", "UNKNOWN")

        # Slice only past matches up to the current match date
        past_matches = df[df["date"] < current_date]
        
        # If separating by league, only consider matches in the same league
        if separate_by_league and league != "UNKNOWN":
            past_matches = past_matches[past_matches["league"] == league]

        # Head-to-head history (limiting to the last h2h_window matches)
        h2h = past_matches[
            ((past_matches["home_team"] == home) & (past_matches["away_team"] == away)) |
            ((past_matches["home_team"] == away) & (past_matches["away_team"] == home))
        ].tail(h2h_window)

        # If not enough H2H history, skip this row
        if len(h2h) < 2:
            continue

        # Feature: average goal difference in past H2H (from home team's perspective)
        goal_diffs = []
        home_wins = 0
        
        for _, match in h2h.iterrows():
            if match["home_team"] == home:
                # Home team is playing at home in this H2H match
                goal_diffs.append(match["home_goals"] - match["away_goals"])
                if match["result"] == "H":
                    home_wins += 1
            else:
                # Home team is playing away in this H2H match  
                goal_diffs.append(match["away_goals"] - match["home_goals"])
                if match["result"] == "A":
                    home_wins += 1
        
        avg_goal_diff = sum(goal_diffs) / len(goal_diffs) if goal_diffs else 0
        h2h_winrate = home_wins / len(h2h) if len(h2h) else 0.5

        # Feature: form-based win rate for home and away team in their last N games
        # Home team recent form (when playing at home)
        home_recent = past_matches[past_matches["home_team"] == home].tail(form_window)
        home_form_wins = len(home_recent[home_recent["result"] == "H"])
        home_form_winrate = home_form_wins / len(home_recent) if len(home_recent) > 0 else 0.5

        # Away team recent form (when playing away)  
        away_recent = past_matches[past_matches["away_team"] == away].tail(form_window)
        away_form_wins = len(away_recent[away_recent["result"] == "A"])
        away_form_winrate = away_form_wins / len(away_recent) if len(away_recent) > 0 else 0.5

        feature_rows.append({
            "home_team": home,
            "away_team": away,
            "match_date": current_date,
            "league": league,
            "avg_goal_diff_h2h": avg_goal_diff,
            "h2h_home_winrate": h2h_winrate,
            "home_form_winrate": home_form_winrate,
            "away_form_winrate": away_form_winrate,
            "result": row["result"]
        })

        # Progress tracking for large datasets
        if len(feature_rows) % 1000 == 0:
            print(f"  📊 Processed {len(feature_rows)} matches...")

    return pd.DataFrame(feature_rows)

def build_features_by_league(leagues=None):
    """Build features for specific leagues"""
    if leagues is None:
        leagues = ["PL", "CHAMP"]
    
    print(f"🔄 Building features for leagues: {leagues}")
    df_raw = fetch_historic_results_multi(leagues=leagues)
    
    if df_raw.empty:
        print("❌ No historical data found!")
        return pd.DataFrame()
    
    print(f"📊 Raw data: {len(df_raw)} matches across {df_raw['league'].nunique()} leagues")
    print(f"League breakdown: {dict(df_raw['league'].value_counts())}")
    
    # Build features
    df_features = build_features(df_raw, separate_by_league=True)
    
    print(f"✅ Generated {len(df_features)} feature rows")
    if not df_features.empty:
        print(f"League breakdown in features: {dict(df_features['league'].value_counts())}")
    
    return df_features

if __name__ == "__main__":
    print("📊 Building feature matrix from historic results...")
    
    # Build features for both Premier League and Championship
    df_features = build_features_by_league(["PL", "CHAMP"])
    
    if df_features.empty:
        print("❌ No features generated!")
        exit(1)
    
    print("\n🔍 Sample of generated features:")
    print(df_features.head())
    
    print(f"\n📈 Feature statistics by league:")
    for league in df_features['league'].unique():
        league_data = df_features[df_features['league'] == league]
        print(f"\n{league}:")
        print(f"  Matches: {len(league_data)}")
        print(f"  Date range: {league_data['match_date'].min()} to {league_data['match_date'].max()}")
        print(f"  Results: {dict(league_data['result'].value_counts())}")

    # Save for training later
    import os
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv("data/processed/training_data.csv", index=False)
    print(f"\n✅ Feature data saved to data/processed/training_data.csv")
    print(f"📊 Total feature rows: {len(df_features)}")