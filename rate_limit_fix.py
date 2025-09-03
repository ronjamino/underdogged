# ==========================================
# COMPLETE RATE LIMIT SOLUTION
# ==========================================

import time
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps

class RateLimitManager:
    """
    Smart rate limit manager that tracks API calls and prevents 429 errors
    """
    
    def __init__(self, calls_per_minute=6, calls_per_hour=50, calls_per_day=100):
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour  
        self.calls_per_day = calls_per_day
        
        # Track API calls
        self.call_history = []
        self.cache_file = "api_call_history.pkl"
        self.load_call_history()
    
    def load_call_history(self):
        """Load previous API call history"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.call_history = pickle.load(f)
                # Remove old calls (older than 24 hours)
                cutoff = datetime.now() - timedelta(hours=24)
                self.call_history = [call for call in self.call_history if call > cutoff]
            except:
                self.call_history = []
    
    def save_call_history(self):
        """Save API call history"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.call_history, f)
    
    def record_call(self):
        """Record an API call"""
        self.call_history.append(datetime.now())
        self.save_call_history()
    
    def can_make_call(self):
        """Check if we can make an API call without hitting limits"""
        now = datetime.now()
        
        # Check minute limit
        minute_ago = now - timedelta(minutes=1)
        calls_last_minute = len([call for call in self.call_history if call > minute_ago])
        
        # Check hour limit  
        hour_ago = now - timedelta(hours=1)
        calls_last_hour = len([call for call in self.call_history if call > hour_ago])
        
        # Check day limit
        day_ago = now - timedelta(hours=24)
        calls_last_day = len([call for call in self.call_history if call > day_ago])
        
        print(f"📊 API Usage: {calls_last_minute}/{self.calls_per_minute} per min, {calls_last_hour}/{self.calls_per_hour} per hour, {calls_last_day}/{self.calls_per_day} per day")
        
        return (calls_last_minute < self.calls_per_minute and 
                calls_last_hour < self.calls_per_hour and 
                calls_last_day < self.calls_per_day)
    
    def wait_if_needed(self):
        """Wait if we're approaching rate limits"""
        if not self.can_make_call():
            # Calculate wait time
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            calls_last_minute = [call for call in self.call_history if call > minute_ago]
            
            if calls_last_minute:
                oldest_call = min(calls_last_minute)
                wait_time = 65 - (now - oldest_call).seconds  # Wait for oldest call to age out + buffer
                
                if wait_time > 0:
                    print(f"⏳ Rate limit protection: waiting {wait_time} seconds...")
                    time.sleep(wait_time)
    
    def get_recommended_delay(self):
        """Get recommended delay between calls"""
        if len(self.call_history) == 0:
            return 10  # Start with 10 seconds
        
        # Dynamic delay based on recent call frequency
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        recent_calls = len([call for call in self.call_history if call > minute_ago])
        
        if recent_calls >= self.calls_per_minute - 1:
            return 70  # Near limit, be very conservative
        elif recent_calls >= self.calls_per_minute // 2:
            return 45  # Moderate usage
        else:
            return 15  # Light usage

# Global rate limit manager
rate_manager = RateLimitManager(calls_per_minute=6, calls_per_hour=50, calls_per_day=100)

def rate_limited_api_call(func):
    """Decorator to add rate limiting to any API function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check rate limits before call
        rate_manager.wait_if_needed()
        
        try:
            # Make the API call
            print(f"📡 Making API call to {func.__name__}...")
            result = func(*args, **kwargs)
            
            # Record successful call
            rate_manager.record_call()
            
            # Smart delay after call
            delay = rate_manager.get_recommended_delay()
            print(f"✅ API call successful. Next call in {delay} seconds...")
            time.sleep(delay)
            
            return result
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"⚠️ Rate limit hit! Implementing emergency backoff...")
                
                # Emergency backoff - wait 2 minutes
                time.sleep(120)
                
                # Try once more
                try:
                    print(f"🔄 Retrying API call...")
                    result = func(*args, **kwargs)
                    rate_manager.record_call()
                    time.sleep(90)  # Conservative delay after recovery
                    return result
                except:
                    print(f"❌ API call failed even after backoff")
                    raise Exception(f"Rate limit exceeded and retry failed: {e}")
            else:
                raise e
    
    return wrapper

# ==========================================
# ENHANCED FIXTURE FETCHING WITH RATE LIMITING
# ==========================================

@rate_limited_api_call
def safe_fetch_single_league(league_code, limit=5):
    """Safely fetch fixtures for a single league"""
    import requests
    from dotenv import load_dotenv
    
    load_dotenv()
    
    API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    BASE_URL = "https://api.football-data.org/v4"
    
    headers = {"X-Auth-Token": API_KEY}
    url = f"{BASE_URL}/competitions/{league_code}/matches?status=SCHEDULED"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    
    matches = response.json().get("matches", [])[:limit]
    
    league_names = {
        'PL': 'Premier League',
        'ELC': 'Championship',
        'BL1': 'Bundesliga', 
        'SA': 'Serie A',
        'PD': 'La Liga'
    }
    
    fixtures = []
    for match in matches:
        fixtures.append({
            "utc_date": match["utcDate"],
            "home_team": match["homeTeam"]["name"], 
            "away_team": match["awayTeam"]["name"],
            "matchday": match.get("matchday"),
            "competition": match.get("competition", {}).get("name", league_code),
            "league_code": league_code,
            "league_name": league_names.get(league_code, league_code)
        })
    
    return pd.DataFrame(fixtures)

def fetch_fixtures_with_smart_rate_limiting(leagues=['PL', 'ELC'], limit=3):
    """
    Fetch fixtures with intelligent rate limiting and fallback strategies
    """
    print(f"🚀 Fetching fixtures with smart rate limiting...")
    print(f"   📋 Target leagues: {leagues}")
    print(f"   📊 Limit per league: {limit}")
    
    all_fixtures = []
    successful_leagues = []
    failed_leagues = []
    
    for i, league_code in enumerate(leagues):
        league_names = {
            'PL': 'Premier League', 'ELC': 'Championship', 'BL1': 'Bundesliga', 
            'SA': 'Serie A', 'PD': 'La Liga'
        }
        league_name = league_names.get(league_code, league_code)
        
        print(f"\n📡 [{i+1}/{len(leagues)}] Fetching {league_name}...")
        
        try:
            fixtures = safe_fetch_single_league(league_code, limit)
            
            if not fixtures.empty:
                all_fixtures.append(fixtures)
                successful_leagues.append(league_name)
                print(f"✅ Success: {len(fixtures)} fixtures from {league_name}")
            else:
                print(f"⚠️ No fixtures available for {league_name}")
                
        except Exception as e:
            failed_leagues.append(league_name)
            print(f"❌ Failed to fetch {league_name}: {e}")
            
            # Don't continue if we hit rate limits
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"⚠️ Rate limit encountered. Stopping further requests.")
                break
    
    # Combine results
    if all_fixtures:
        combined = pd.concat(all_fixtures, ignore_index=True)
        
        print(f"\n🎉 FETCH COMPLETE!")
        print(f"   ✅ Successful: {', '.join(successful_leagues)}")
        if failed_leagues:
            print(f"   ❌ Failed: {', '.join(failed_leagues)}")
        print(f"   📊 Total fixtures: {len(combined)}")
        
        return combined
    else:
        print(f"\n⚠️ No fixtures retrieved!")
        return pd.DataFrame()

# ==========================================
# CACHING SYSTEM TO REDUCE API CALLS
# ==========================================

def get_cached_fixtures(cache_hours=2):
    """Get cached fixtures if available and fresh"""
    cache_file = "fixtures_cache.pkl"
    
    if os.path.exists(cache_file):
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_hours = file_age.total_seconds() / 3600
        
        if age_hours <= cache_hours:
            print(f"📂 Using cached fixtures (age: {age_hours:.1f} hours)")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    return None

def save_fixtures_cache(fixtures):
    """Save fixtures to cache"""
    cache_file = "fixtures_cache.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(fixtures, f)
    print(f"💾 Fixtures cached for future use")

def fetch_fixtures_with_caching(leagues=['PL', 'ELC'], limit=3, cache_hours=2, force_refresh=False):
    """
    Fetch fixtures with caching to minimize API calls
    """
    print(f"🔄 Fetching fixtures with caching (cache_hours={cache_hours})...")
    
    # Try cache first (unless force refresh)
    if not force_refresh:
        cached = get_cached_fixtures(cache_hours)
        if cached is not None and not cached.empty:
            return cached
    
    # Cache miss - fetch fresh data
    print(f"📡 Cache miss or forced refresh - fetching fresh data...")
    fixtures = fetch_fixtures_with_smart_rate_limiting(leagues, limit)
    
    # Save to cache if successful
    if not fixtures.empty:
        save_fixtures_cache(fixtures)
    
    return fixtures

# ==========================================
# MAIN FUNCTION FOR YOUR PIPELINE
# ==========================================

def fetch_fixtures_for_pipeline(leagues=['PL', 'ELC'], limit=3, use_cache=True):
    """
    Main function to replace your fixture fetching in run_pipeline.py
    
    This function handles:
    - Rate limiting
    - Caching  
    - Error recovery
    - Fallback strategies
    """
    print(f"🎯 PIPELINE FIXTURE FETCHING")
    print(f"=" * 50)
    
    try:
        if use_cache:
            fixtures = fetch_fixtures_with_caching(leagues, limit, cache_hours=1)
        else:
            fixtures = fetch_fixtures_with_smart_rate_limiting(leagues, limit)
        
        if not fixtures.empty:
            print(f"\n✅ SUCCESS: Retrieved {len(fixtures)} fixtures")
            print(f"📊 Leagues covered: {', '.join(fixtures['league_name'].unique())}")
            return fixtures
        else:
            print(f"\n⚠️ No fixtures available")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"\n❌ Fixture fetching failed: {e}")
        print(f"💡 Try running again in 30+ minutes")
        return pd.DataFrame()

# ==========================================
# EASY INTEGRATION FOR YOUR CODE
# ==========================================

"""
TO FIX YOUR RATE LIMIT ISSUES:

1. Replace this line in your run_pipeline.py:
   
   OLD:
   fixtures = fetch_upcoming_fixtures(['PL', 'ELC'], limit=10)
   
   NEW:
   from rate_limit_fix import fetch_fixtures_for_pipeline
   fixtures = fetch_fixtures_for_pipeline(['PL', 'ELC'], limit=5, use_cache=True)

2. The new system will:
   - Track your API usage automatically
   - Prevent 429 errors with smart delays
   - Cache results to reduce API calls
   - Provide detailed logging
   - Handle errors gracefully

3. First run might be slower (due to delays) but subsequent runs will be much faster (due to caching)
"""

if __name__ == "__main__":
    print("🧪 Testing complete rate limit solution...")
    
    # Test the system
    print(f"\n" + "=" * 60)
    print("TEST 1: Checking current rate limit status")
    print("=" * 60)
    
    print(f"Can make call: {rate_manager.can_make_call()}")
    print(f"Recommended delay: {rate_manager.get_recommended_delay()} seconds")
    
    print(f"\n" + "=" * 60)
    print("TEST 2: Safe fixture fetching")
    print("=" * 60)
    
    # Test with conservative settings
    test_fixtures = fetch_fixtures_for_pipeline(['PL'], limit=2, use_cache=True)
    
    if not test_fixtures.empty:
        print(f"✅ Rate limit fix working! Got {len(test_fixtures)} fixtures")
    else:
        print(f"⚠️ No fixtures - might need to wait or check API key")
    
    print(f"\n💡 INTEGRATION INSTRUCTIONS:")
    print(f"1. Save this code as 'rate_limit_fix.py' in your project")
    print(f"2. Replace fixture fetching in run_pipeline.py with:")
    print(f"   from rate_limit_fix import fetch_fixtures_for_pipeline")  
    print(f"   fixtures = fetch_fixtures_for_pipeline(['PL', 'ELC'], limit=5)")
    print(f"3. First run will be slower, subsequent runs will be faster (cached)")