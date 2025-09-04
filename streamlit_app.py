import streamlit as st
import pandas as pd
import subprocess
import numpy as np

st.set_page_config(layout="wide", page_title="Underdogged - Multi-League Analysis")

# Helper functions for value betting analysis
def calculate_implied_probabilities(home_odds, draw_odds, away_odds):
    """Calculate bookmaker implied probabilities from decimal odds"""
    if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
        return None, None, None, None
    
    home_implied = 1 / home_odds
    draw_implied = 1 / draw_odds  
    away_implied = 1 / away_odds
    
    total_implied = home_implied + draw_implied + away_implied
    overround = (total_implied - 1) * 100
    
    home_true = home_implied / total_implied
    draw_true = draw_implied / total_implied
    away_true = away_implied / total_implied
    
    return home_true, draw_true, away_true, overround

def calculate_value_score(model_prob, bookmaker_prob, min_edge=0.05):
    """Calculate value betting score"""
    if pd.isna(model_prob) or pd.isna(bookmaker_prob) or bookmaker_prob == 0:
        return 0
    
    edge = model_prob - bookmaker_prob
    value_score = edge / bookmaker_prob if bookmaker_prob > 0 else 0
    
    if abs(edge) < min_edge:
        return 0
    
    return value_score

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv(
            "data/predictions/latest_predictions.csv",
            parse_dates=["match_date"]
        )
    except Exception:
        return pd.DataFrame()

@st.cache_data 
def load_odds():
    try:
        return pd.read_csv("data/odds/latest_odds.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def calculate_value_bets():
    """Calculate value bets for the app"""
    predictions_df = load_predictions()
    odds_df = load_odds()
    
    if predictions_df.empty or odds_df.empty:
        return pd.DataFrame()
    
    value_bets = []
    
    for _, pred_row in predictions_df.iterrows():
        odds_match = odds_df[
            (odds_df['home_team'] == pred_row['home_team']) & 
            (odds_df['away_team'] == pred_row['away_team'])
        ]
        
        if odds_match.empty:
            continue
            
        odds_row = odds_match.iloc[0]
        
        model_home_prob = pred_row['home_win']
        model_draw_prob = pred_row['draw'] 
        model_away_prob = pred_row['away_win']
        
        home_true, draw_true, away_true, overround = calculate_implied_probabilities(
            odds_row['home_odds'], odds_row['draw_odds'], odds_row['away_odds']
        )
        
        if home_true is None:
            continue
        
        home_value = calculate_value_score(model_home_prob, home_true)
        draw_value = calculate_value_score(model_draw_prob, draw_true)  
        away_value = calculate_value_score(model_away_prob, away_true)
        
        values = [
            ('home_win', home_value, model_home_prob, home_true, odds_row['home_odds']),
            ('draw', draw_value, model_draw_prob, draw_true, odds_row['draw_odds']),
            ('away_win', away_value, model_away_prob, away_true, odds_row['away_odds'])
        ]
        
        values.sort(key=lambda x: abs(x[1]), reverse=True)
        best_outcome, best_value, model_prob, bookie_prob, odds = values[0]
        
        if abs(best_value) > 0.10:  # 10% edge minimum
            if best_value > 0 and odds > 1:
                b = odds - 1
                p = model_prob
                q = 1 - model_prob
                kelly_fraction = (b * p - q) / b if b > 0 else 0
                kelly_fraction = max(0, min(kelly_fraction, 0.25))
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
    
    return pd.DataFrame(value_bets)

# Header
st.title("⚽ Underdogged — Multi-League Analysis")
st.caption("Advanced football prediction system with statistical modeling and market analysis")

# Create tabs
tab1, tab2 = st.tabs(["🎯 Predictions", "💰 Value Analysis"])

# Tab 1: Original Predictions
with tab1:
    df = load_predictions()

    if df.empty:
        st.warning("No predictions found. Run the pipeline first with:")
        st.code("python -m run_pipeline", language="bash")
    else:
        # Calculate max_proba if it doesn't exist
        if "max_proba" not in df.columns:
            df["max_proba"] = df[["home_win", "draw", "away_win"]].max(axis=1)

        # Show summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            high_conf = len(df[df["max_proba"] >= 0.70])
            st.metric("High Confidence (≥70%)", high_conf)
        with col3:
            leagues = df["league"].nunique() if "league" in df.columns else 1
            st.metric("Leagues", leagues)
        with col4:
            if "league" in df.columns:
                league_counts = dict(df["league"].value_counts())
                league_split = " | ".join([f"{k}: {v}" for k, v in league_counts.items()])
                st.metric("League Split", league_split)

        # Sidebar controls
        st.sidebar.header("Prediction Filters")

        # League filter
        if "league" in df.columns:
            available_leagues = ["All"] + list(df["league"].unique())
            selected_league = st.sidebar.selectbox("League", available_leagues)
            
            if selected_league != "All":
                df = df[df["league"] == selected_league]

        # Confidence filter
        min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.60, 0.01)

        # Date filter
        if not df.empty:
            date_min = df["match_date"].min()
            date_max = df["match_date"].max()
            if pd.notna(date_min) and pd.notna(date_max):
                date_window = st.sidebar.date_input(
                    "Filter by date",
                    value=(date_min.date(), date_max.date())
                )
            else:
                date_window = None
        else:
            date_window = None

        # Refresh button
        if st.sidebar.button("🔄 Refresh Predictions", help="Re-run the full prediction pipeline"):
            with st.spinner("Running prediction pipeline... This may take a few minutes"):
                try:
                    result = subprocess.run(
                        ["python", "-m", "run_pipeline"], 
                        capture_output=True, 
                        text=True, 
                        timeout=300
                    )
                    if result.returncode == 0:
                        st.success("✅ Predictions updated successfully!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ Pipeline failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("⏰ Pipeline timed out. Try running manually.")
                except Exception as e:
                    st.error(f"❌ Error running pipeline: {e}")

        # Apply filters
        filtered = df.copy()

        if "max_proba" not in filtered.columns:
            filtered["max_proba"] = filtered[["home_win", "draw", "away_win"]].max(axis=1)

        # Date filter
        if isinstance(date_window, tuple) and len(date_window) == 2:
            start, end = date_window
            filtered = filtered[(filtered["match_date"].dt.date >= start) & (filtered["match_date"].dt.date <= end)]

        # Confidence filter
        filtered = filtered[filtered["max_proba"] >= min_conf]

        # Main predictions table
        st.subheader("🎯 Predictions")

        if filtered.empty:
            st.info("No matches meet the current filters. Try lowering the confidence threshold or changing the date range.")
        else:
            # Sort and display
            filtered = filtered.sort_values(["league", "match_date", "max_proba"], ascending=[True, True, False])
            
            if "league" in filtered.columns:
                filtered = filtered.copy()
                league_mapping = {
                    "PL": "⚽ Premier League", 
                    "ELC": "🏆 Championship",
                    "BL1": "🇩🇪 Bundesliga",
                    "SA": "🇮🇹 Serie A", 
                    "PD": "🇪🇸 La Liga"
                }
                filtered["league_display"] = filtered["league"].map(league_mapping).fillna(filtered["league"])
            
            display_cols = [
                "match_date", "home_team", "away_team", "predicted_result", 
                "confidence_label", "prob_label"
            ]
            
            if "league_display" in filtered.columns:
                display_cols.insert(3, "league_display")
            
            st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "match_date": st.column_config.DatetimeColumn("Match Date", format="MMM DD, HH:mm"),
                    "home_team": st.column_config.TextColumn("Home Team", width="medium"),
                    "away_team": st.column_config.TextColumn("Away Team", width="medium"),
                    "league_display": st.column_config.TextColumn("League", width="medium"),
                    "predicted_result": st.column_config.TextColumn("Prediction", width="small"),
                    "confidence_label": st.column_config.TextColumn("Confidence", width="small"),
                    "prob_label": st.column_config.TextColumn("All Probabilities", width="large"),
                }
            )

            # Summary statistics
            if not filtered.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Predictions by League")
                    if "league" in filtered.columns:
                        league_summary = filtered["league"].value_counts()
                        for league_name, league_count in league_summary.items():
                            league_data = filtered[filtered["league"] == league_name]
                            avg_conf = league_data["max_proba"].mean()
                            league_display = league_mapping.get(league_name, league_name)
                            st.write(f"**{league_display}**: {league_count} matches (avg confidence: {avg_conf:.1%})")
                
                with col2:
                    st.subheader("🎲 Prediction Distribution")
                    result_summary = filtered["predicted_result"].value_counts()
                    for result_type, result_count in result_summary.items():
                        pct = result_count / len(filtered) * 100
                        emoji_map = {"home_win": "🏠", "away_win": "✈️", "draw": "🤝"}
                        emoji = emoji_map.get(result_type, "")
                        st.write(f"**{emoji} {result_type.replace('_', ' ').title()}**: {result_count} ({pct:.1f}%)")

# Tab 2: Value Betting Analysis
with tab2:
    st.header("💰 Value Betting Analysis")
    
    # Important disclaimers at the top
    st.warning("""
    **⚠️ IMPORTANT DISCLAIMERS**
    - This is for analytical/educational purposes only
    - Sports betting involves significant risk of financial loss
    - Model predictions may not be accurate
    - Always verify odds independently before making any decisions
    - Never bet more than you can afford to lose
    """)
    
    value_df = calculate_value_bets()
    
    if value_df.empty:
        st.info("""
        No value betting opportunities found. This could mean:
        - No odds data available 
        - Bookmaker odds are well-calibrated
        - Model confidence thresholds not met
        
        Make sure you have both predictions and odds data loaded.
        """)
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Value Opportunities", len(value_df))
        with col2:
            positive_value = len(value_df[value_df['value_score'] > 0])
            st.metric("Positive Value Bets", positive_value)
        with col3:
            if positive_value > 0:
                avg_edge = value_df[value_df['value_score'] > 0]['edge_pct'].mean()
                st.metric("Avg Positive Edge", f"{avg_edge:.1f}%")
            else:
                st.metric("Avg Positive Edge", "N/A")
        with col4:
            avg_overround = value_df['overround'].mean()
            st.metric("Avg Bookmaker Margin", f"{avg_overround:.1f}%")
        
        # Value betting controls
        st.sidebar.header("Value Betting Filters")
        
        min_value = st.sidebar.slider("Minimum Value Score", 0.0, 2.0, 0.2, 0.1)
        min_confidence = st.sidebar.slider("Minimum Match Confidence", 0.0, 1.0, 0.4, 0.05)
        
        # League filter for value bets
        if "league" in value_df.columns:
            available_leagues_value = ["All"] + list(value_df["league"].unique())
            selected_league_value = st.sidebar.selectbox("League (Value)", available_leagues_value)
            
            if selected_league_value != "All":
                value_df = value_df[value_df["league"] == selected_league_value]
        
        # Apply filters
        value_filtered = value_df[
            (value_df['value_score'].abs() >= min_value) & 
            (value_df['match_confidence'] >= min_confidence)
        ].sort_values('value_score', key=abs, ascending=False)
        
        if value_filtered.empty:
            st.info("No value bets meet the current filter criteria. Try adjusting the filters.")
        else:
            st.subheader("🎯 Value Betting Opportunities")
            
            # Format the data for display
            display_value = value_filtered.copy()
            
            # Add formatted columns
            display_value['fixture'] = display_value['home_team'] + ' vs ' + display_value['away_team']
            display_value['bet_type'] = display_value['best_bet'].str.replace('_', ' ').str.title()
            display_value['model_prob_fmt'] = (display_value['model_prob'] * 100).round(1).astype(str) + '%'
            display_value['bookmaker_prob_fmt'] = (display_value['bookmaker_prob'] * 100).round(1).astype(str) + '%'
            display_value['edge_fmt'] = display_value['edge_pct'].round(1).astype(str) + '%'
            display_value['value_fmt'] = (display_value['value_score'] * 100).round(1).astype(str) + '%'
            display_value['kelly_fmt'] = display_value['kelly_pct'].round(1).astype(str) + '%'
            display_value['odds_fmt'] = display_value['odds'].round(2).astype(str)
            
            # Add status indicators
            def get_value_status(value_score):
                abs_value = abs(value_score)
                if abs_value > 0.5:
                    return "🔥 Strong"
                elif abs_value > 0.3:
                    return "⚡ Good"
                else:
                    return "💡 Mild"
            
            display_value['status'] = display_value['value_score'].apply(get_value_status)
            
            # Display table
            st.dataframe(
                display_value[[
                    'status', 'fixture', 'league', 'bet_type', 'odds_fmt',
                    'model_prob_fmt', 'bookmaker_prob_fmt', 'edge_fmt', 'value_fmt', 'kelly_fmt'
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "fixture": st.column_config.TextColumn("Fixture", width="large"),
                    "league": st.column_config.TextColumn("League", width="small"),
                    "bet_type": st.column_config.TextColumn("Bet Type", width="small"),
                    "odds_fmt": st.column_config.TextColumn("Odds", width="small"),
                    "model_prob_fmt": st.column_config.TextColumn("Model %", width="small"),
                    "bookmaker_prob_fmt": st.column_config.TextColumn("Bookie %", width="small"),
                    "edge_fmt": st.column_config.TextColumn("Edge", width="small"),
                    "value_fmt": st.column_config.TextColumn("Value", width="small"),
                    "kelly_fmt": st.column_config.TextColumn("Kelly", width="small"),
                }
            )
            
            # Analysis by league
            if len(value_filtered) > 0:
                st.subheader("📊 Value Analysis by League")
                
                league_analysis = value_filtered.groupby('league').agg({
                    'value_score': ['count', 'mean'],
                    'edge_pct': 'mean',
                    'match_confidence': 'mean'
                }).round(2)
                
                for league in league_analysis.index:
                    count = int(league_analysis.loc[league, ('value_score', 'count')])
                    avg_value = league_analysis.loc[league, ('value_score', 'mean')]
                    avg_edge = league_analysis.loc[league, ('edge_pct', 'mean')]
                    avg_conf = league_analysis.loc[league, ('match_confidence', 'mean')]
                    
                    st.write(f"**{league}**: {count} opportunities, {avg_value:+.1%} avg value, {avg_edge:+.1f}% avg edge, {avg_conf:.1%} avg confidence")
    
    # Model calibration warning
    with st.expander("⚠️ Model Calibration Notice", expanded=False):
        st.warning("""
        **Important Model Limitations:**
        
        The value betting analysis is only as reliable as the underlying prediction model. 
        Current observations suggest the model may need calibration:
        
        - Very high value scores (>100%) are unusual in efficient betting markets
        - The model has not been validated against actual match results
        - Bookmakers have access to information not in the model
        
        **Recommendations:**
        - Treat this as academic analysis, not betting advice
        - Track model performance over time to assess accuracy
        - Start with very small stakes if exploring these insights
        - Focus on smaller value edges (10-20%) which are more realistic
        """)

# Footer
st.markdown("---")
st.caption("""
**Underdogged Multi-League Analysis** | 
Advanced statistical modeling for football predictions | 
For educational and analytical purposes only
""")