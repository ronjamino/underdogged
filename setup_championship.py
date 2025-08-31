#!/usr/bin/env python3
"""
Quick setup script to test Championship expansion.
Run this to verify everything works before full integration.
"""

import os
import sys
import pandas as pd

def test_championship_data():
    """Test that we can fetch Championship historical data"""
    print("1. Testing Championship historical data...")
    try:
        from fetch.fetch_historic_results import fetch_historic_results_multi
        df = fetch_historic_results_multi(leagues=["CHAMP"])
        
        if df.empty:
            print("❌ No Championship data found")
            return False
            
        print(f"✅ Found {len(df)} Championship matches")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Show sample teams
        sample_teams = df['home_team'].unique()[:5]
        print(f"🏟️ Sample teams: {', '.join(sample_teams)}")
        
        return True
    except Exception as e:
        print(f"❌ Error fetching Championship data: {e}")
        return False

def test_championship_fixtures():
    """Test that we can fetch Championship fixtures"""
    print("\n2. Testing Championship fixtures...")
    try:
        from fetch.fetch_fixtures import fetch_fixtures_championship
        df = fetch_fixtures_championship(limit=5)
        
        if df.empty:
            print("⚠️ No Championship fixtures found (may be off-season)")
            return True  # Not a failure, just no current fixtures
            
        print(f"✅ Found {len(df)} Championship fixtures")
        for _, row in df.head(3).iterrows():
            print(f"   {row['home_team']} vs {row['away_team']}")
        
        return True
    except Exception as e:
        print(f"❌ Error fetching Championship fixtures: {e}")
        return False

def test_team_normalization():
    """Test team name normalization for Championship teams"""
    print("\n3. Testing team name normalization...")
    try:
        from utils.team_name_map import normalize_team
        
        test_cases = [
            ("Sheffield United FC", "Sheffield Utd"),
            ("West Bromwich Albion FC", "West Brom"),
            ("Queens Park Rangers FC", "QPR"),
            ("Sheffield Wednesday FC", "Sheffield Wed"),
            ("Leicester City FC", "Leicester"),
        ]
        
        all_passed = True
        for original, expected in test_cases:
            normalized = normalize_team(original)
            if normalized == expected:
                print(f"✅ {original} -> {normalized}")
            else:
                print(f"❌ {original} -> {normalized} (expected: {expected})")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ Error testing team normalization: {e}")
        return False

def test_feature_generation():
    """Test feature generation with Championship data"""
    print("\n4. Testing feature generation...")
    try:
        from model.prepare_training_data import build_features_by_league
        
        # Test with small sample
        df_features = build_features_by_league(["CHAMP"])
        
        if df_features.empty:
            print("❌ No features generated for Championship")
            return False
        
        print(f"✅ Generated {len(df_features)} Championship feature rows")
        
        # Check feature quality
        avg_h2h = df_features['avg_goal_diff_h2h'].mean()
        avg_form = df_features['home_form_winrate'].mean()
        
        print(f"📊 Average H2H goal diff: {avg_h2h:.2f}")
        print(f"📊 Average home form: {avg_form:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Error generating features: {e}")
        return False

def create_sample_predictions():
    """Create sample predictions to test the full pipeline"""
    print("\n5. Testing prediction pipeline...")
    try:
        # Check if model exists
        if not os.path.exists("models/random_forest_model.pkl"):
            print("⚠️ Model not found. Run training first with:")
            print("   python -m model.train_model")
            return False
        
        from predict.predict_fixtures import predict_championship_only
        
        print("🔮 Generating Championship predictions...")
        predict_championship_only()
        
        # Check if predictions were saved
        if os.path.exists("data/predictions/latest_predictions.csv"):
            df = pd.read_csv("data/predictions/latest_predictions.csv")
            champ_preds = df[df['league'] == 'CHAMP'] if 'league' in df.columns else df
            
            print(f"✅ Generated {len(champ_preds)} Championship predictions")
            if len(champ_preds) > 0:
                high_conf = len(champ_preds[champ_preds[['home_win', 'draw', 'away_win']].max(axis=1) >= 0.6])
                print(f"🎯 {high_conf} high-confidence predictions (≥60%)")
            
            return True
        else:
            print("❌ Predictions file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction pipeline: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Championship expansion setup...\n")
    
    tests = [
        test_championship_data,
        test_championship_fixtures, 
        test_team_normalization,
        test_feature_generation,
        create_sample_predictions,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    test_names = [
        "Championship historical data",
        "Championship fixtures", 
        "Team name normalization",
        "Feature generation",
        "Prediction pipeline"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Championship expansion is ready.")
        print("\nNext steps:")
        print("1. Run the full pipeline: python -m run_pipeline")
        print("2. Start the dashboard: streamlit run streamlit_app.py")
        print("3. Look for Championship predictions in the interface")
    else:
        print(f"\n⚠️ {len(tests) - passed} tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())