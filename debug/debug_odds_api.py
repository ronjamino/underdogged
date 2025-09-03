# debug_odds_api.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

def check_api_key_and_sports():
    """Step 1: Check if API key works and see available sports"""
    print("🔍 Step 1: Checking API key and available sports...")
    
    if not ODDS_API_KEY:
        print("❌ ODDS_API_KEY not found in .env file!")
        print("   Add: ODDS_API_KEY=your_actual_key_here")
        return False
    
    url = f"{ODDS_BASE_URL}/sports"
    params = {'api_key': ODDS_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        sports = response.json()
        
        # Check for soccer sports
        soccer_sports = [s for s in sports if 'soccer' in s['key']]
        print(f"✅ API key working! Found {len(sports)} total sports")
        print(f"⚽ Found {len(soccer_sports)} soccer leagues:")
        
        for sport in soccer_sports:
            status = "🟢 ACTIVE" if sport.get('active', False) else "🔴 INACTIVE"
            print(f"   {sport['key']}: {sport['title']} {status}")
        
        # Check specifically for EPL
        epl_sport = next((s for s in sports if s['key'] == 'soccer_epl'), None)
        if epl_sport:
            print(f"\n⚽ EPL found: {epl_sport}")
            return True
        else:
            print(f"\n❌ soccer_epl not found in available sports")
            print("   Available soccer leagues:")
            for sport in soccer_sports:
                print(f"   - {sport['key']}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

def test_epl_odds():
    """Step 2: Test EPL odds specifically"""
    print("\n🔍 Step 2: Testing EPL odds...")
    
    url = f"{ODDS_BASE_URL}/sports/soccer_epl/odds"
    params = {
        'api_key': ODDS_API_KEY,
        'regions': 'uk',
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    
    try:
        print(f"📡 Calling: {url}")
        print(f"📋 Parameters: {params}")
        
        response = requests.get(url, params=params)
        print(f"📡 Response status: {response.status_code}")
        
        # Check headers for API usage
        used = response.headers.get('x-requests-used', 'Unknown')
        remaining = response.headers.get('x-requests-remaining', 'Unknown')
        print(f"💰 API Usage: {used} used, {remaining} remaining")
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return None
        
        matches = response.json()
        print(f"📊 Received {len(matches)} matches")
        
        if not matches:
            print("⚠️ No matches returned - EPL might be out of season")
            return None
        
        # Show first match details
        first_match = matches[0]
        print(f"\n📋 First match example:")
        print(f"   ID: {first_match.get('id')}")
        print(f"   Sport: {first_match.get('sport_key')}")
        print(f"   Date: {first_match.get('commence_time')}")
        print(f"   Home: {first_match.get('home_team')}")
        print(f"   Away: {first_match.get('away_team')}")
        print(f"   Bookmakers: {len(first_match.get('bookmakers', []))}")
        
        return matches
        
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return None

def check_which_api_you_used():
    """Step 3: Check what data you actually have"""
    print("\n🔍 Step 3: Analyzing your existing odds data...")
    
    try:
        df = pd.read_csv("data/raw/odds.csv")
        print(f"📊 Your odds.csv contains {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"\n📋 Sample teams:")
        print(df[['home_team', 'away_team']].head(10))
        
        # Analyze team names to guess the source
        home_teams = df['home_team'].unique()
        
        print(f"\n🔍 Analysis of team names:")
        
        # Check for different sports
        hockey_indicators = ['Wild', 'Sharks', 'Ducks', 'Flames', 'Oilers', 'Blues']
        soccer_indicators = ['City', 'United', 'Arsenal', 'Chelsea', 'Liverpool']
        american_football = ['Patriots', 'Cowboys', 'Giants', '49ers']
        
        hockey_count = sum(1 for team in home_teams if any(indicator in team for indicator in hockey_indicators))
        soccer_count = sum(1 for team in home_teams if any(indicator in team for indicator in soccer_indicators))
        american_football_count = sum(1 for team in home_teams if any(indicator in team for indicator in american_football))
        
        print(f"   🏒 Hockey-like teams: {hockey_count}")
        print(f"   ⚽ Soccer-like teams: {soccer_count}")
        print(f"   🏈 American Football-like teams: {american_football_count}")
        
        if hockey_count > 0:
            print("   ➜ This looks like HOCKEY data!")
        if any('Cruzeiro' in team or 'Flamengo' in team for team in home_teams):
            print("   ➜ This includes South American SOCCER!")
            
    except FileNotFoundError:
        print("❌ data/raw/odds.csv not found")
    except Exception as e:
        print(f"❌ Error reading odds.csv: {e}")

def get_correct_epl_data():
    """Step 4: Get actual EPL data"""
    print("\n🔍 Step 4: Fetching actual EPL data...")
    
    matches = test_epl_odds()
    if not matches:
        print("❌ Could not get EPL data")
        return
    
    # Convert to DataFrame
    rows = []
    for match in matches:
        row = {
            'match_id': match['id'],
            'sport_key': match['sport_key'],
            'commence_time': match['commence_time'],
            'home_team': match['home_team'],
            'away_team': match['away_team']
        }
        
        # Extract odds
        if match['bookmakers']:
            bookmaker = match['bookmakers'][0]
            h2h_market = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
            
            if h2h_market:
                for outcome in h2h_market['outcomes']:
                    if outcome['name'] == match['home_team']:
                        row['home_odds'] = outcome['price']
                    elif outcome['name'] == match['away_team']:
                        row['away_odds'] = outcome['price']
                    else:
                        row['draw_odds'] = outcome['price']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print(f"✅ Converted to DataFrame: {len(df)} matches")
    print(f"📋 Sample EPL matches:")
    display_cols = ['home_team', 'away_team', 'home_odds', 'draw_odds', 'away_odds']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head())
    
    # Save correct data
    os.makedirs("data/odds", exist_ok=True)
    df.to_csv("data/odds/correct_epl_odds.csv", index=False)
    print(f"💾 Saved correct EPL data to data/odds/correct_epl_odds.csv")

def main():
    print("🚨 ODDS API DEBUGGING SCRIPT 🚨")
    print("="*50)
    
    # Step 1: Check API key and available sports
    if not check_api_key_and_sports():
        print("\n❌ Cannot proceed - fix API key first")
        return
    
    # Step 2: Test EPL odds
    test_epl_odds()
    
    # Step 3: Analyze existing data
    check_which_api_you_used()
    
    # Step 4: Get correct data
    get_correct_epl_data()
    
    print("\n" + "="*50)
    print("🎯 DIAGNOSIS COMPLETE!")
    print("\nPossible issues:")
    print("1. ❌ Wrong API being called")
    print("2. ❌ EPL out of season (no current matches)")
    print("3. ❌ Wrong sport key in your code")
    print("4. ❌ Your existing odds.csv is from different source")
    
    print("\n💡 Next steps:")
    print("1. ✅ Use the correct_epl_odds.csv generated above")
    print("2. ✅ Check that your fetch_odds.py uses correct URL")
    print("3. ✅ Verify EPL is in season (matches available)")

if __name__ == "__main__":
    main()