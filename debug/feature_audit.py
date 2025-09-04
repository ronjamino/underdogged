#!/usr/bin/env python3
"""
Quick audit script to verify current feature utilization
Run this to see exactly what features are being used vs generated
"""

import os
import pandas as pd
import joblib

def audit_current_system():
    """Audit the current feature utilization"""
    print("🔍 FOOTBALL PREDICTION FEATURE AUDIT")
    print("=" * 50)
    
    # Check if training data exists
    training_data_path = "data/processed/training_data.csv"
    if not os.path.exists(training_data_path):
        print("❌ No training data found. Run: python -m model.prepare_training_data")
        return
    
    # Load training data to see what features are generated
    print("📊 Loading training data...")
    df = pd.read_csv(training_data_path)
    print(f"✅ Loaded {len(df)} training samples")
    
    # Identify feature columns (exclude meta columns)
    meta_columns = ['home_team', 'away_team', 'match_date', 'league', 'result']
    feature_columns = [col for col in df.columns if col not in meta_columns]
    print(f"📈 Generated features: {len(feature_columns)}")
    
    # Check if model metadata exists
    metadata_path = "models/metadata.pkl"
    if os.path.exists(metadata_path):
        print("\n🤖 Loading model metadata...")
        metadata = joblib.load(metadata_path)
        
        if 'features' in metadata:
            model_features = metadata['features']
            print(f"✅ Model uses {len(model_features)} features")
            
            # Find orphaned features (generated but not used)
            orphaned = [f for f in feature_columns if f not in model_features]
            missing = [f for f in model_features if f not in feature_columns]
            
            print(f"\n⚠️ FEATURE GAPS:")
            print(f"   Generated but IGNORED: {len(orphaned)}")
            for i, feature in enumerate(orphaned):
                print(f"      {i+1}. {feature}")
            
            if missing:
                print(f"   Expected but MISSING: {len(missing)}")
                for i, feature in enumerate(missing):
                    print(f"      {i+1}. {feature}")
            else:
                print(f"   ✅ All model features are generated")
                
            # Calculate utilization rate
            utilization = len(model_features) / len(feature_columns) * 100
            print(f"\n📊 Feature utilization: {utilization:.1f}% ({len(model_features)}/{len(feature_columns)})")
            
            if utilization < 90:
                print(f"⚠️ LOW UTILIZATION - Many features are wasted!")
                print(f"💡 Fix: Update feature lists in model/train_model.py")
            
        else:
            print("⚠️ No feature list found in model metadata")
    else:
        print("⚠️ No model metadata found. Train a model first.")
    
    # Analyze feature categories
    print(f"\n📋 FEATURE ANALYSIS:")
    
    # Categorize features
    categories = {
        'Core': ['avg_goal_diff_h2h', 'h2h_home_winrate', 'home_form_winrate', 
                'away_form_winrate', 'home_avg_goals_scored', 'home_avg_goals_conceded',
                'away_avg_goals_scored', 'away_avg_goals_conceded'],
        
        'Draw-focused': [f for f in feature_columns if 'draw' in f.lower() or 
                        'differential' in f.lower() or 'momentum' in f.lower() or
                        'defensive' in f.lower() or 'low_scoring' in f.lower()],
        
        'Odds-based': [f for f in feature_columns if 'prob' in f.lower() or 
                      'market' in f.lower() or 'odds' in f.lower()],
        
        'League context': [f for f in feature_columns if 'league' in f.lower()],
        
        'Goals/Scoring': [f for f in feature_columns if 'goals' in f.lower() and 
                         'differential' not in f.lower()]
    }
    
    for category, expected_features in categories.items():
        available = [f for f in expected_features if f in feature_columns]
        print(f"   {category}: {len(available)} features")
        
        if len(available) != len(expected_features):
            missing_in_cat = [f for f in expected_features if f not in feature_columns]
            if missing_in_cat:
                print(f"      Missing: {missing_in_cat[:2]}...")
    
    # Check for odds data in historical results
    print(f"\n💰 ODDS DATA CHECK:")
    sample_row = df.iloc[0] if len(df) > 0 else None
    
    odds_features_present = [f for f in ['home_true_prob', 'draw_true_prob', 'away_true_prob'] 
                           if f in df.columns]
    
    if odds_features_present:
        print(f"✅ Odds features found: {len(odds_features_present)}")
        
        # Check if odds are actually calculated (not default values)
        sample_home_prob = df['home_true_prob'].iloc[0] if 'home_true_prob' in df.columns else None
        if sample_home_prob and sample_home_prob not in [0.33, 0.40, 0.45]:
            print(f"   ✅ Odds appear to be calculated from real data")
        else:
            print(f"   ⚠️ Odds might be default values - check historical data")
            
    else:
        print(f"❌ No odds features found")
    
    # Performance context
    print(f"\n📈 RECOMMENDATIONS:")
    
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        if 'draw_recall' in metadata:
            draw_recall = metadata['draw_recall']
            print(f"   Current draw recall: {draw_recall:.1%}")
            
            if draw_recall < 0.15:
                print(f"   💡 Draw prediction is low - orphaned features could help")
                print(f"   🎯 Focus on: home_draw_rate, away_draw_rate, h2h_total_goals")
            
        if 'accuracy' in metadata:
            accuracy = metadata['accuracy']
            print(f"   Current accuracy: {accuracy:.1%}")
    
    print(f"\n🔧 NEXT STEPS:")
    print(f"   1. Update feature lists in model/train_model.py")
    print(f"   2. Retrain model: python -m model.train_model") 
    print(f"   3. Compare performance before/after")
    print(f"   4. Focus on draw prediction improvements")

def check_odds_integration():
    """Check if odds are properly integrated"""
    print(f"\n💰 ODDS INTEGRATION CHECK:")
    
    # Check for live odds file
    live_odds_path = "data/odds/latest_odds.csv"
    if os.path.exists(live_odds_path):
        odds_df = pd.read_csv(live_odds_path)
        print(f"✅ Live odds found: {len(odds_df)} matches")
        
        # Check team normalization
        if 'home_team' in odds_df.columns:
            sample_teams = odds_df['home_team'].head(3).tolist()
            print(f"   Sample teams: {sample_teams}")
        
        # Check odds columns
        odds_cols = [col for col in odds_df.columns if 'odds' in col.lower()]
        print(f"   Odds columns: {odds_cols}")
        
    else:
        print(f"⚠️ No live odds file found")
        print(f"   Run: python -m fetch.fetch_odds")
    
    # Check historical odds
    print(f"\n📚 Historical odds check:")
    try:
        from fetch.fetch_historic_results import fetch_historic_results
        sample_data = fetch_historic_results("PL_2425")  # Recent season
        
        odds_columns = ['B365H', 'B365D', 'B365A']
        found_odds = [col for col in odds_columns if col in sample_data.columns]
        
        if found_odds:
            print(f"   ✅ Historical odds found: {found_odds}")
            
            # Check for valid odds
            for col in found_odds:
                valid_odds = sample_data[col].notna().sum()
                total = len(sample_data)
                print(f"   {col}: {valid_odds}/{total} valid ({valid_odds/total*100:.1f}%)")
        else:
            print(f"   ❌ No historical odds columns found")
            print(f"   Available columns: {list(sample_data.columns)}")
            
    except Exception as e:
        print(f"   ❌ Error checking historical data: {e}")

def main():
    """Run complete audit"""
    audit_current_system()
    check_odds_integration()
    
    print(f"\n" + "=" * 50)
    print(f"🎯 AUDIT COMPLETE")
    print(f"   See audit report for detailed fixes")
    print(f"   Priority: Fix feature utilization gap")

if __name__ == "__main__":
    main()