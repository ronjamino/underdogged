import streamlit as st
import pandas as pd
import subprocess

st.set_page_config(layout="wide", page_title="Underdogged - Multi-League")

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv(
            "data/predictions/latest_predictions.csv",
            parse_dates=["match_date"]
        )
    except Exception:
        return pd.DataFrame()

# Header with league info
st.title("⚽ Underdogged — Multi-League Predictions")
st.caption("Premier League & Championship predictions using head-to-head analysis and recent form")

df = load_predictions()

if df.empty:
    st.warning("No predictions found. Run the pipeline first with:")
    st.code("python -m run_pipeline", language="bash")
    st.stop()

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
st.sidebar.header("Filters & Controls")

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
                timeout=300  # 5 minute timeout
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

# Calculate max_proba if it doesn't exist
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
    # Sort by league, then match date, then confidence
    filtered = filtered.sort_values(["league", "match_date", "max_proba"], ascending=[True, True, False])
    
    # Add league emoji for visual distinction
    if "league" in filtered.columns:
        filtered = filtered.copy()
        filtered["league_display"] = filtered["league"].map({
            "PL": "⚽ Premier League", 
            "CHAMP": "🏆 Championship"
        }).fillna(filtered["league"])
    
    # Prepare display columns
    display_cols = [
        "match_date", "home_team", "away_team", "predicted_result", 
        "confidence_label", "prob_label"
    ]
    
    if "league_display" in filtered.columns:
        display_cols.insert(3, "league_display")  # Insert league after away_team
    
    # Display with native dark theme compatible styling
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

    # Summary of filtered results
    if not filtered.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Predictions by League")
            if "league" in filtered.columns:
                league_summary = filtered["league"].value_counts()
                for league_name, league_count in league_summary.items():
                    league_data = filtered[filtered["league"] == league_name]
                    avg_conf = league_data["max_proba"].mean()
                    emoji = "⚽" if league_name == "PL" else "🏆"
                    league_display = "Premier League" if league_name == "PL" else "Championship"
                    st.write(f"**{emoji} {league_display}**: {league_count} matches (avg confidence: {avg_conf:.1%})")
        
        with col2:
            st.subheader("🎲 Prediction Distribution")
            result_summary = filtered["predicted_result"].value_counts()
            for result_type, result_count in result_summary.items():
                pct = result_count / len(filtered) * 100
                emoji_map = {"home_win": "🏠", "away_win": "✈️", "draw": "🤝"}
                emoji = emoji_map.get(result_type, "")
                st.write(f"**{emoji} {result_type.replace('_', ' ').title()}**: {result_count} ({pct:.1f}%)")

# Advanced stats section
with st.expander("📈 Advanced Statistics", expanded=False):
    if not filtered.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Analysis")
            
            # Show average features by league
            if "league" in filtered.columns:
                for league_name in filtered["league"].unique():
                    league_data = filtered[filtered["league"] == league_name]
                    emoji = "⚽" if league_name == "PL" else "🏆"
                    league_display = "Premier League" if league_name == "PL" else "Championship"
                    st.write(f"**{emoji} {league_display} Averages:**")
                    st.write(f"• H2H Goal Diff: {league_data['avg_goal_diff_h2h'].mean():.2f}")
                    st.write(f"• H2H Home Win Rate: {league_data['h2h_home_winrate'].mean():.2f}")
                    st.write(f"• Home Form: {league_data['home_form_winrate'].mean():.2f}")
                    st.write(f"• Away Form: {league_data['away_form_winrate'].mean():.2f}")
                    st.write("")
        
        with col2:
            st.subheader("Confidence Distribution")
            
            # Confidence bins
            filtered["conf_bin"] = pd.cut(
                filtered["max_proba"], 
                bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
                labels=["<50%", "50-60%", "60-70%", "70-80%", "80%+"]
            )
            conf_dist = filtered["conf_bin"].value_counts().sort_index()
            
            confidence_emojis = {"<50%": "🔴", "50-60%": "🟡", "60-70%": "🟠", "70-80%": "🟢", "80%+": "⭐"}
            for bin_name, bin_count in conf_dist.items():
                emoji = confidence_emojis.get(str(bin_name), "")
                st.write(f"**{emoji} {bin_name}**: {bin_count} matches")

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Higher confidence predictions in Championship matches may offer better value due to less efficient odds.")

if "league" in df.columns and not df.empty:
    total_pl = len(df[df["league"] == "PL"])
    total_champ = len(df[df["league"] == "CHAMP"])
    st.caption(f"📊 **Data**: {total_pl} Premier League + {total_champ} Championship predictions loaded")