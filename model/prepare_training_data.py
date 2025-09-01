import os
import pandas as pd
from fetch.fetch_historic_results import fetch_historic_results_multi

# ---- Canonical league codes and aliases ----
# Use these canonical codes everywhere downstream:
#   PL (Premier League), ELC (Championship), BL1 (Bundesliga), SA (Serie A), PD (La Liga)
LEAGUE_ALIASES = {
    # Canonical pass-through
    "PL": "PL",
    "ELC": "ELC",
    "BL1": "BL1",
    "SA": "SA",
    "PD": "PD",

    # Common alternates you might pass in:
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
    Generate features from historic match results including head-to-head form,
    recent venue-aware form, and venue-aware goals per match.

    Expects df columns:
      ['date','home_team','away_team','home_goals','away_goals','result','league']
      - 'result' in {'H','D','A'}
      - 'league' will be normalized to canonical codes before this is called.
    """
    # sort and ensure canonical league labels
    df = df.sort_values("date").copy()
    if "league" not in df.columns:
        df["league"] = "UNKNOWN"
    else:
        df["league"] = df["league"].map(_canon)

    df["outcome_code"] = df["result"].map({"H": 1, "D": 0, "A": -1})

    feature_rows = []

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

        # --- H2H ---
        h2h = past_matches[
            ((past_matches["home_team"] == home) & (past_matches["away_team"] == away)) |
            ((past_matches["home_team"] == away) & (past_matches["away_team"] == home))
        ].tail(h2h_window)

        if len(h2h) < 2:
            avg_goal_diff = 0.0
            h2h_home_winrate = 0.5
        else:
            goal_diffs, home_wins = [], 0
            for _, match in h2h.iterrows():
                if match["home_team"] == home:
                    goal_diffs.append(match["home_goals"] - match["away_goals"])
                    if match["result"] == "H":
                        home_wins += 1
                else:
                    goal_diffs.append(match["away_goals"] - match["home_goals"])
                    if match["result"] == "A":
                        home_wins += 1
            avg_goal_diff = sum(goal_diffs) / len(goal_diffs) if goal_diffs else 0.0
            h2h_home_winrate = home_wins / len(h2h) if len(h2h) else 0.5

        # --- Venue-aware form (last N at same venue) ---
        home_recent = past_matches[past_matches["home_team"] == home].sort_values("date").tail(form_window)
        away_recent = past_matches[past_matches["away_team"] == away].sort_values("date").tail(form_window)

        home_form_wins = (home_recent["result"] == "H").sum()
        away_form_wins = (away_recent["result"] == "A").sum()

        home_form_winrate = (home_form_wins / len(home_recent)) if len(home_recent) else 0.5
        away_form_winrate = (away_form_wins / len(away_recent)) if len(away_recent) else 0.5

        # --- Venue-aware goals per match ---
        if len(home_recent):
            home_avg_goals_scored = float(home_recent["home_goals"].mean())
            home_avg_goals_conceded = float(home_recent["away_goals"].mean())
        else:
            home_avg_goals_scored = 1.0
            home_avg_goals_conceded = 1.0

        if len(away_recent):
            away_avg_goals_scored = float(away_recent["away_goals"].mean())
            away_avg_goals_conceded = float(away_recent["home_goals"].mean())
        else:
            away_avg_goals_scored = 1.0
            away_avg_goals_conceded = 1.0

        feature_rows.append({
            "home_team": home,
            "away_team": away,
            "match_date": current_date,
            "league": league,  # already canonicalized
            "avg_goal_diff_h2h": avg_goal_diff,
            "h2h_home_winrate": h2h_home_winrate,
            "home_form_winrate": home_form_winrate,
            "away_form_winrate": away_form_winrate,
            "home_avg_goals_scored": home_avg_goals_scored,
            "home_avg_goals_conceded": home_avg_goals_conceded,
            "away_avg_goals_scored": away_avg_goals_scored,
            "away_avg_goals_conceded": away_avg_goals_conceded,
            "result": row["result"],
        })

        if len(feature_rows) % 1000 == 0:
            print(f"  📊 Processed {len(feature_rows)} matches...")

    return pd.DataFrame(feature_rows)

def build_features_by_league(leagues=None):
    """
    Build features for specific leagues.

    `leagues` can contain any of:
      Canonical codes: ["PL","ELC","BL1","SA","PD"]
      Aliases/names:   ["CHAMP","Premier League","La Liga",...]
    """
    if leagues is None:
        leagues = ["PL", "ELC"]  # defaults (canonical)

    # Normalize requested leagues to canonical codes
    leagues_canon = sorted({_canon(l) for l in leagues if l})

    print(f"🔄 Building features for leagues (requested): {leagues}")
    print(f"✅ Canonicalized leagues: {leagues_canon}")

    # Fetch raw
    df_raw = fetch_historic_results_multi(leagues=leagues_canon)

    if df_raw.empty:
        print("❌ No historical data found from fetcher!")
        return pd.DataFrame()

    # Normalize df_raw['league'] to canonical codes BEFORE any filtering
    if "league" in df_raw.columns:
        df_raw["league"] = df_raw["league"].map(_canon)
    else:
        df_raw["league"] = "UNKNOWN"

    print(f"📊 Raw data: {len(df_raw)} matches across {df_raw['league'].nunique()} leagues")
    print(f"   League breakdown (canonical): {dict(df_raw['league'].value_counts())}")

    # Only keep the leagues we actually asked for (in canonical form)
    df_raw = df_raw[df_raw["league"].isin(leagues_canon)]

    if df_raw.empty:
        print("❌ After canonical filtering, no rows remain for requested leagues.")
        return pd.DataFrame()

    # Build features
    df_features = build_features(df_raw, separate_by_league=True)

    print(f"✅ Generated {len(df_features)} feature rows")
    if not df_features.empty:
        print(f"   League breakdown in features: {dict(df_features['league'].value_counts())}")

    return df_features

if __name__ == "__main__":
    print("📊 Building feature matrix from historic results...")

    # >>> Include Euro leagues here <<<
    requested = ["PL", "ELC", "BL1", "SA", "PD"]  # PL, Championship, Bundesliga, Serie A, La Liga

    df_features = build_features_by_league(requested)

    if df_features.empty:
        print("❌ No features generated!")
        raise SystemExit(1)

    print("\n🔍 Sample of generated features:")
    print(df_features.head())

    print(f"\n📈 Feature statistics by league:")
    for league in df_features['league'].unique():
        league_data = df_features[df_features['league'] == league]
        print(f"\n{league}:")
        print(f"  Matches: {len(league_data)}")
        print(f"  Date range: {league_data['match_date'].min()} to {league_data['match_date'].max()}")
        print(f"  Results: {dict(league_data['result'].value_counts())}")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/training_data.csv"
    df_features.to_csv(out_path, index=False)
    print(f"\n✅ Feature data saved to {out_path}")
    print(f"📊 Total feature rows: {len(df_features)}")
