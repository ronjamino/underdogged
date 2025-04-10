import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# Load predictions
@st.cache_data
def load_predictions():
    return pd.read_csv("data/predictions/latest_predictions.csv", parse_dates=["match_date"])

# Title
st.title("⚽ Underdogged: Football Match Predictions")

# Sidebar Filters
st.sidebar.header("Filters")
min_confidence = st.sidebar.slider("Minimum model confidence", 0.0, 1.0, 0.6, 0.01)
team_filter = st.sidebar.text_input("Filter by team name (optional)").lower()

# Load data
df = load_predictions()

# Apply filters
filtered = df[df["predicted_proba"] >= min_confidence]
if team_filter:
    filtered = filtered[
        filtered["home_team"].str.lower().str.contains(team_filter) |
        filtered["away_team"].str.lower().str.contains(team_filter)
    ]

# Highlight potential mismatches
def highlight_mismatch(row):
    if row["predicted_result"] == "home_win" and row.get("home_odds", 1) > row.get("away_odds", 1):
        return "🔺 Underdog Home"
    if row["predicted_result"] == "away_win" and row.get("away_odds", 1) > row.get("home_odds", 1):
        return "🔻 Underdog Away"
    return ""

filtered["underdog_mismatch"] = filtered.apply(highlight_mismatch, axis=1)

# Display
st.subheader("Predictions")
st.dataframe(
    filtered[[
        "match_date", "home_team", "away_team",
        "predicted_result", "confidence_label",
        "home_odds", "draw_odds", "away_odds",
        "underdog_mismatch"
    ]].sort_values(by="match_date"),
    use_container_width=True
)

# Placeholder for bet simulation
st.markdown("---")
st.subheader("🧪 Experimental: Strategy Simulator (coming soon!)")
st.markdown("Imagine testing different betting strategies here...")
