# fetch/fetch_odds.py - FIXED VERSION (no import issues)
import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
import time

# Fix Python path issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import your utilities
try:
    from utils.team_name_map import normalize_team
except ImportError:
    print("⚠️ Could not import normalize_team, using fallback")
    def normalize_team(team_name):
        """Fallback team normalization"""
        if not team_name:
            return team_name
        
        # Basic cleanup
        team_name = str(team_name).strip()
        
        # Common substitutions for your model
        substitutions = {
            "Manchester City FC": "Man City",
            "Manchester United FC": "Man United",
            "Newcastle United FC": "Newcastle", 
            "Tottenham Hotspur FC": "Tottenham",
            "Brighton & Hove Albion FC": "Brighton",
            "West Ham United FC": "West Ham",
            "Wolverhampton Wanderers FC": "Wolves",
            "Nottingham Forest FC": "Nottm Forest",
            "AFC Bournemouth": "Bournemouth",
            "Sheffield United FC": "Sheffield Utd",
            "West Bromwich Albion FC": "West Brom",
            "Queens Park Rangers FC": "QPR",
            "Sheffield Wednesday FC": "Sheffield Wed"
        }
        
        return substitutions.get(team_name, team_name)

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

# Correct league mappings (confirmed working)
LEAGUE_TO_ODDS_API = {
    "PL": "soccer_epl",
    "ELC": "soccer_efl_champ", 
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a", 
    "PD": "soccer_spain_la_liga"
}

def check_api_usage():
    """Check remaining credits without costing any"""
    url = f"{ODDS_BASE_URL}/sports"
    params = {'api_key': ODDS_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        used = response.headers.get('x-requests-used', 'Unknown')
        remaining = response.headers.get('x-requests-remaining', 'Unknown')
        print(f"💰 API Usage: {used} used, {remaining} remaining")
        return int(remaining) if remaining != 'Unknown' else None
    except:
        return None

def fetch_odds_for_league(league_code, region="uk"):
    """
    Fetch current betting odds for a specific league
    Cost: 1 credit per call
    
    Returns DataFrame with normalized team names and average odds
    """
    if not ODDS_API_KEY:
        print("❌ ODDS_API_KEY not found in .env file!")
        return pd.DataFrame()
        
    if league_code not in LEAGUE_TO_ODDS_API:
        print(f"⚠️ League {league_code} not supported")
        return pd.DataFrame()
    
    sport_key = LEAGUE_TO_ODDS_API[league_code]
    
    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds"
    params = {
        'api_key': ODDS_API_KEY,
        'regions': region,
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    
    try:
        print(f"🔄 Fetching {league_code} odds...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Track usage
        remaining = response.headers.get('x-requests-remaining', 'Unknown')
        print(f"💰 Credits remaining: {remaining}")
        
        matches = response.json()
        if not matches:
            print(f"ℹ️ No matches for {league_code} (might be off-season)")
            return pd.DataFrame()
        
        rows = []
        for match in matches:
            # Basic match info
            row = {
                'match_id': match['id'],
                'commence_time': match['commence_time'],
                'home_team_raw': match['home_team'],  # Keep original
                'away_team_raw': match['away_team'],
                'home_team': normalize_team(match['home_team']),  # Normalized
                'away_team': normalize_team(match['away_team']),
                'league': league_code,
                'sport_key': match['sport_key']
            }
            
            # Extract and average odds across bookmakers
            home_odds_list, away_odds_list, draw_odds_list = [], [], []
            
            for bookmaker in match['bookmakers']:
                h2h_market = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
                
                if h2h_market and len(h2h_market['outcomes']) == 3:
                    for outcome in h2h_market['outcomes']:
                        if outcome['name'] == match['home_team']:
                            home_odds_list.append(outcome['price'])
                        elif outcome['name'] == match['away_team']:
                            away_odds_list.append(outcome['price'])
                        else:  # Draw
                            draw_odds_list.append(outcome['price'])
            
            # Calculate average odds (more stable than single bookmaker)
            if home_odds_list and away_odds_list and draw_odds_list:
                row['home_odds'] = round(sum(home_odds_list) / len(home_odds_list), 2)
                row['away_odds'] = round(sum(away_odds_list) / len(away_odds_list), 2)
                row['draw_odds'] = round(sum(draw_odds_list) / len(draw_odds_list), 2)
                row['num_bookmakers'] = len(home_odds_list)
            else:
                # Skip matches without complete odds
                continue
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            print(f"✅ {league_code}: {len(df)} matches from {df.iloc[0]['num_bookmakers']} bookmakers")
            
            # Show sample for verification
            if league_code == "PL":  # Show Premier League teams to verify
                print("📋 Sample teams:")
                for _, row in df.head(3).iterrows():
                    print(f"   {row['home_team']} vs {row['away_team']}")
        
        return df
        
    except requests.RequestException as e:
        print(f"❌ Error fetching {league_code}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return pd.DataFrame()

def fetch_all_odds(leagues=None, save_to_csv=True, pipeline_mode=False):
    """
    Fetch odds for multiple leagues

    Args:
        leagues: List of league codes to fetch
        save_to_csv: Whether to save results
        pipeline_mode: If True, skip interactive prompts (always proceed)

    Returns:
        Combined DataFrame with all odds
    """
    if leagues is None:
        leagues = ["PL", "ELC", "BL1", "SA", "PD"]

    print(f"🎯 Fetching odds for {len(leagues)} leagues...")
    print(f"💰 Estimated cost: {len(leagues)} credits")

    # Check budget first
    remaining = check_api_usage()
    if remaining and remaining < len(leagues):
        print(f"⚠️ Warning: Only {remaining} credits remaining, need {len(leagues)}")
        if not pipeline_mode:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return pd.DataFrame()
        else:
            print("   Pipeline mode: proceeding anyway")
    
    all_odds = []
    total_matches = 0
    successful_leagues = 0
    
    for i, league in enumerate(leagues):
        print(f"\n[{i+1}/{len(leagues)}] {league}:")
        
        odds_df = fetch_odds_for_league(league)
        if not odds_df.empty:
            all_odds.append(odds_df)
            total_matches += len(odds_df)
            successful_leagues += 1
        else:
            print(f"⚠️ No data for {league}")
        
        # Rate limiting to be respectful
        if i < len(leagues) - 1:
            time.sleep(0.5)
    
    if not all_odds:
        print("❌ No odds data retrieved from any league")
        return pd.DataFrame()
    
    # Combine all leagues
    combined = pd.concat(all_odds, ignore_index=True)
    combined['fetch_timestamp'] = pd.Timestamp.now(tz='UTC')
    
    print(f"\n📊 SUCCESS! {total_matches} matches from {successful_leagues}/{len(leagues)} leagues")
    
    # Show breakdown
    league_summary = combined['league'].value_counts()
    for league, count in league_summary.items():
        print(f"   • {league}: {count} matches")
    
    if save_to_csv:
        # Create directories
        os.makedirs("data/odds", exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        
        # Save with timestamp for history
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        timestamped_file = f"data/odds/odds_{timestamp}.csv"
        combined.to_csv(timestamped_file, index=False)
        
        # Save as latest for easy access
        combined.to_csv("data/odds/latest_odds.csv", index=False)
        
        # IMPORTANT: Replace old test data
        combined.to_csv("data/raw/odds.csv", index=False)
        
        print(f"💾 Saved to:")
        print(f"   • {timestamped_file}")
        print(f"   • data/odds/latest_odds.csv")
        print(f"   • data/raw/odds.csv (replaced old test data)")

    # Persist to PostgreSQL
    try:
        from db.connection import engine
        from sqlalchemy import text as _text

        insert_sql = _text("""
            INSERT INTO odds (
                match_id, commence_time, home_team_raw, away_team_raw,
                home_team, away_team, league, sport_key,
                home_odds, away_odds, draw_odds, num_bookmakers, fetch_timestamp
            ) VALUES (
                :match_id, :commence_time, :home_team_raw, :away_team_raw,
                :home_team, :away_team, :league, :sport_key,
                :home_odds, :away_odds, :draw_odds, :num_bookmakers, :fetch_timestamp
            )
            ON CONFLICT (match_id) DO UPDATE SET
                home_odds       = EXCLUDED.home_odds,
                away_odds       = EXCLUDED.away_odds,
                draw_odds       = EXCLUDED.draw_odds,
                num_bookmakers  = EXCLUDED.num_bookmakers,
                fetch_timestamp = EXCLUDED.fetch_timestamp,
                updated_at      = NOW()
        """)

        odds_df = combined.copy()
        if "fetch_timestamp" in odds_df.columns:
            odds_df["fetch_timestamp"] = pd.to_datetime(odds_df["fetch_timestamp"], utc=True, errors="coerce")
        if "commence_time" in odds_df.columns:
            odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], utc=True, errors="coerce")

        rows = odds_df.to_dict("records")
        with engine.begin() as conn:
            conn.execute(insert_sql, rows)
        print(f"✅ {len(rows)} odds rows upserted to PostgreSQL")
    except Exception as exc:
        print(f"⚠️  PostgreSQL write failed (CSV still saved): {exc}")

    return combined

def add_odds_features(df):
    """Add market-based features to predictions"""
    required_cols = ['home_odds', 'draw_odds', 'away_odds']
    if not all(col in df.columns for col in required_cols):
        print("⚠️ Missing odds columns, cannot add odds features")
        return df
    
    print("🧮 Calculating odds-based features...")
    
    # Market implied probabilities
    df['home_implied_prob'] = 1 / df['home_odds']
    df['draw_implied_prob'] = 1 / df['draw_odds']
    df['away_implied_prob'] = 1 / df['away_odds']
    
    # Remove bookmaker overround (normalize to sum to 1.0)
    total_prob = df['home_implied_prob'] + df['draw_implied_prob'] + df['away_implied_prob']
    df['home_true_prob'] = df['home_implied_prob'] / total_prob
    df['draw_true_prob'] = df['draw_implied_prob'] / total_prob
    df['away_true_prob'] = df['away_implied_prob'] / total_prob
    
    # Derived market features
    df['odds_home_advantage'] = df['away_implied_prob'] - df['home_implied_prob']
    df['match_competitiveness'] = 1 - abs(df['home_true_prob'] - df['away_true_prob'])
    df['market_uncertainty'] = df['draw_true_prob']
    
    print("✅ Added odds features:")
    print("   • Market probabilities (home/draw/away_true_prob)")
    print("   • Home advantage estimate (odds_home_advantage)")
    print("   • Match competitiveness and uncertainty")
    
    return df

def show_odds_summary(df):
    """Display summary of fetched odds"""
    if df.empty:
        print("❌ No odds data to summarize")
        return
    
    print(f"\n📊 ODDS SUMMARY")
    print(f"=" * 30)
    print(f"Total matches: {len(df)}")
    print(f"Leagues: {df['league'].nunique()} ({', '.join(df['league'].unique())})")
    print(f"Average home odds: {df['home_odds'].mean():.2f}")
    print(f"Average away odds: {df['away_odds'].mean():.2f}")
    print(f"Average draw odds: {df['draw_odds'].mean():.2f}")
    
    if 'home_true_prob' in df.columns:
        home_adv = df['odds_home_advantage'].mean()
        competitiveness = df['match_competitiveness'].mean()
        print(f"Market home advantage: {home_adv:.3f}")
        print(f"Average competitiveness: {competitiveness:.3f}")
    
    print(f"\n📋 Sample matches:")
    display_cols = ['league', 'home_team', 'away_team', 'home_odds', 'draw_odds', 'away_odds']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head().to_string(index=False))

# Main execution — non-interactive when run as part of the pipeline
if __name__ == "__main__":
    print("🎯 Fetching live odds for all leagues...")

    if not ODDS_API_KEY:
        print("❌ Missing ODDS_API_KEY in .env file")
        exit(1)

    all_odds = fetch_all_odds(pipeline_mode=True)

    if not all_odds.empty:
        show_odds_summary(all_odds)
        print("✅ Odds fetch complete!")
    else:
        print("⚠️ No odds data retrieved — predictions will run without live odds")