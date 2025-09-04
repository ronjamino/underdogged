# ==========================================
# VALUE BETTING ANALYSIS TOOL
# File: analyze_value_bets.py
# ==========================================

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_implied_probabilities(home_odds, draw_odds, away_odds):
    """Calculate bookmaker implied probabilities from decimal odds"""
    if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
        return None, None, None, None
    
    # Convert odds to implied probabilities
    home_implied = 1 / home_odds
    draw_implied = 1 / draw_odds  
    away_implied = 1 / away_odds
    
    # Calculate overround (bookmaker margin)
    total_implied = home_implied + draw_implied + away_implied
    overround = (total_implied - 1) * 100
    
    # True probabilities (remove overround)
    home_true = home_implied / total_implied
    draw_true = draw_implied / total_implied
    away_true = away_implied / total_implied
    
    return home_true, draw_true, away_true, overround

def calculate_value_score(model_prob, bookmaker_prob, min_edge=0.05):
    """Calculate value betting score"""
    if pd.isna(model_prob) or pd.isna(bookmaker_prob) or bookmaker_prob == 0:
        return 0
    
    # Edge = difference between model and bookmaker probabilities
    edge = model_prob - bookmaker_prob
    
    # Value score = edge / bookmaker_prob (percentage overperformance)
    value_score = edge / bookmaker_prob if bookmaker_prob > 0 else 0
    
    # Only consider significant edges
    if abs(edge) < min_edge:
        return 0
    
    return value_score

def analyze_value_bets():
    """Main value betting analysis"""
    print("🎯 VALUE BETTING ANALYSIS")
    print("=" * 50)
    
    # Load predictions and odds
    try:
        predictions_df = pd.read_csv("data/predictions/latest_predictions.csv")
        print(f"📊 Loaded {len(predictions_df)} predictions")
    except:
        print("❌ Could not load predictions. Run predict_fixtures first.")
        return
    
    try:
        odds_df = pd.read_csv("data/odds/latest_odds.csv")
        print(f"💰 Loaded {len(odds_df)} odds entries")
    except:
        print("❌ Could not load odds data. Run fetch_odds first.")
        return
    
    # Merge predictions with odds data
    value_bets = []
    
    for _, pred_row in predictions_df.iterrows():
        # Find matching odds
        odds_match = odds_df[
            (odds_df['home_team'] == pred_row['home_team']) & 
            (odds_df['away_team'] == pred_row['away_team'])
        ]
        
        if odds_match.empty:
            continue
            
        odds_row = odds_match.iloc[0]
        
        # Get model probabilities
        model_home_prob = pred_row['home_win']
        model_draw_prob = pred_row['draw'] 
        model_away_prob = pred_row['away_win']
        
        # Calculate bookmaker probabilities
        home_true, draw_true, away_true, overround = calculate_implied_probabilities(
            odds_row['home_odds'], odds_row['draw_odds'], odds_row['away_odds']
        )
        
        if home_true is None:
            continue
        
        # Calculate value scores for each outcome
        home_value = calculate_value_score(model_home_prob, home_true)
        draw_value = calculate_value_score(model_draw_prob, draw_true)  
        away_value = calculate_value_score(model_away_prob, away_true)
        
        # Determine best value bet
        values = [
            ('home_win', home_value, model_home_prob, home_true, odds_row['home_odds']),
            ('draw', draw_value, model_draw_prob, draw_true, odds_row['draw_odds']),
            ('away_win', away_value, model_away_prob, away_true, odds_row['away_odds'])
        ]
        
        # Sort by value score
        values.sort(key=lambda x: abs(x[1]), reverse=True)
        best_outcome, best_value, model_prob, bookie_prob, odds = values[0]
        
        # Only include if significant value found
        if abs(best_value) > 0.10:  # 10% edge minimum
            
            # Calculate Kelly Criterion suggested bet size
            if best_value > 0 and odds > 1:
                # Kelly = (bp - q) / b, where b = odds-1, p = model_prob, q = 1-model_prob
                b = odds - 1
                p = model_prob
                q = 1 - model_prob
                kelly_fraction = (b * p - q) / b if b > 0 else 0
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0
            
            value_bets.append({
                'match_date': pred_row['match_date'],
                'home_team': pred_row['home_team'],
                'away_team': pred_row['away_team'], 
                'league': pred_row['league'],
                'best_bet': best_outcome,
                'model_prob': model_prob,
                'bookmaker_prob': bookie_prob,
                'odds': odds,
                'value_score': best_value,
                'edge_pct': (model_prob - bookie_prob) * 100,
                'kelly_pct': kelly_fraction * 100,
                'overround': overround,
                'match_confidence': pred_row[['home_win', 'draw', 'away_win']].max()
            })
    
    if not value_bets:
        print("\n📊 No significant value bets found.")
        print("💡 This could mean:")
        print("   • Bookmaker odds are well-calibrated")  
        print("   • Model needs more refinement")
        print("   • Markets are efficient for these fixtures")
        return
    
    # Convert to DataFrame and sort by value
    value_df = pd.DataFrame(value_bets)
    value_df = value_df.sort_values('value_score', key=abs, ascending=False)
    
    print(f"\n🎯 FOUND {len(value_df)} POTENTIAL VALUE BETS")
    print("=" * 60)
    
    # Display top value bets
    print("\n📈 TOP VALUE OPPORTUNITIES:")
    
    for _, row in value_df.head(10).iterrows():
        fixture = f"{row['home_team']} vs {row['away_team']}"
        bet_type = row['best_bet'].replace('_', ' ').title()
        
        # Color coding based on value
        if abs(row['value_score']) > 0.30:
            status = "🔥 STRONG"
        elif abs(row['value_score']) > 0.20:
            status = "⚡ GOOD" 
        else:
            status = "💡 MILD"
        
        # Direction indicator
        direction = "📈 BACK" if row['value_score'] > 0 else "📉 LAY"
        
        print(f"\n{status} {direction} • {fixture} ({row['league']})")
        print(f"   🎲 Bet: {bet_type} @ {row['odds']:.2f}")
        print(f"   🧠 Model: {row['model_prob']:.1%} | 📊 Bookie: {row['bookmaker_prob']:.1%}")
        print(f"   💰 Edge: {row['edge_pct']:+.1f}% | Value: {row['value_score']:+.1%}")
        print(f"   📏 Kelly: {row['kelly_pct']:.1f}% | Confidence: {row['match_confidence']:.1%}")
    
    # Summary statistics
    print(f"\n📊 VALUE BETTING SUMMARY:")
    print("=" * 30)
    
    positive_value = value_df[value_df['value_score'] > 0]
    negative_value = value_df[value_df['value_score'] < 0]
    
    print(f"📈 Back bets (model higher): {len(positive_value)}")
    print(f"📉 Lay bets (model lower): {len(negative_value)}")
    
    if len(positive_value) > 0:
        print(f"💰 Avg positive edge: {positive_value['edge_pct'].mean():.1f}%")
        print(f"🏆 Best opportunity: {positive_value['value_score'].max():.1%} value")
    
    if len(negative_value) > 0:
        print(f"⚠️ Avg negative edge: {negative_value['edge_pct'].mean():.1f}%")
    
    print(f"📊 Avg bookmaker margin: {value_df['overround'].mean():.1f}%")
    
    # League breakdown
    print(f"\n🏆 VALUE BY LEAGUE:")
    league_summary = value_df.groupby('league').agg({
        'value_score': ['count', 'mean'],
        'edge_pct': 'mean'
    }).round(2)
    
    for league in league_summary.index:
        count = league_summary.loc[league, ('value_score', 'count')]
        avg_value = league_summary.loc[league, ('value_score', 'mean')]
        avg_edge = league_summary.loc[league, ('edge_pct', 'mean')]
        print(f"   {league}: {count} bets, {avg_value:+.1%} value, {avg_edge:+.1f}% edge")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"data/value_bets_{timestamp}.csv"
    value_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to {output_file}")
    
    # Risk warnings
    print(f"\n" + "⚠️" * 20)
    print("RISK MANAGEMENT REMINDERS:")
    print("⚠️" * 20)
    print("• This is analytical research, not financial advice")
    print("• Sports betting involves risk of loss")  
    print("• Never bet more than you can afford to lose")
    print("• Model predictions can be wrong")
    print("• Past performance doesn't guarantee future results")
    print("• Consider Kelly percentages as maximum, not recommendations")
    print("• Bookmaker odds change frequently")
    print("• Always verify odds before placing any bets")

def analyze_historical_value_performance():
    """Analyze how well value bets would have performed historically"""
    print("\n🔍 HISTORICAL VALUE BET ANALYSIS")
    print("=" * 40)
    print("💡 This would require historical results to validate the model")
    print("💡 Suggestion: Track predictions vs actual results over time")
    print("💡 Calculate: ROI, hit rate, Kelly performance, etc.")

if __name__ == "__main__":
    analyze_value_bets()
    
    print(f"\n🎯 USAGE INSTRUCTIONS:")
    print("=" * 25)
    print("1. Run this after generating predictions and fetching odds")
    print("2. Focus on bets with >20% value scores")
    print("3. Cross-reference with match confidence levels")  
    print("4. Consider bookmaker limits and market liquidity")
    print("5. Track results to validate model performance")
    
    print(f"\n💡 INTEGRATION TIP:")
    print("Add this to your pipeline:")
    print("python -m analyze_value_bets")
    print("Or call analyze_value_bets() from your main script")