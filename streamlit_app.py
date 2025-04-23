import streamlit as st
import pandas as pd
import subprocess 

st.set_page_config(layout="wide")

# Load predictions
@st.cache_data
def load_predictions():
    return pd.read_csv("data/predictions/latest_predictions.csv", parse_dates=["match_date"])

# Load historic head-to-head data
@st.cache_data
def load_historic_data():
    h2h_data = pd.read_csv("data/processed/training_data.csv")  # Assuming this file contains the historical data
    return h2h_data

# Title
st.title("⚽ Underdogged: Football Match Predictions")

# Sidebar Filters
st.sidebar.header("Filters")
min_confidence = st.sidebar.slider("Minimum model confidence", 0.0, 1.0, 0.6, 0.01)
team_filter = st.sidebar.text_input("Filter by team name (optional)").lower()

# Sidebar Refresh Button
if st.button("🔁 Refresh Predictions"):
    with st.spinner("Running model pipeline and refreshing predictions..."):
        try:
            subprocess.run(["python", "-m", "run_pipeline"], check=True)
            st.success("Predictions updated successfully!")
            st.cache_data.clear()
            df = load_predictions()  # Force reload right after clearing cache
        except subprocess.CalledProcessError:
            st.error("There was an error running the prediction pipeline.")

# Load data
df = load_predictions()
h2h_data = load_historic_data()

# Apply filters to predictions
filtered = df[df["predicted_proba"] >= min_confidence]
if team_filter:
    filtered = filtered[
        filtered["home_team"].str.lower().str.contains(team_filter) |
        filtered["away_team"].str.lower().str.contains(team_filter)
    ]

# Display Predictions
st.subheader("Predictions")
st.dataframe(
    filtered[[
        "match_date", "home_team", "away_team",
        "predicted_result", "confidence_label"
    ]].sort_values(by="match_date"),
    use_container_width=True
)

st.markdown("### Head-to-Head (Last 10 Matches)")

if not h2h_data.empty and not filtered.empty:
    selected_match = st.selectbox(
        "Select a match to view recent head-to-head results",
        filtered.apply(lambda row: f"{row['home_team']} vs {row['away_team']} on {row['match_date'].date()}", axis=1)
    )

    # Extract selected home and away teams
    selected_row = filtered[
        filtered.apply(lambda row: f"{row['home_team']} vs {row['away_team']} on {row['match_date'].date()}", axis=1)
        == selected_match
    ].iloc[0]

    home = selected_row["home_team"]
    away = selected_row["away_team"]

    # Filter for any match between the two teams (regardless of home/away)
    h2h_filtered = h2h_data[
        ((h2h_data["home_team"] == home) & (h2h_data["away_team"] == away)) |
        ((h2h_data["home_team"] == away) & (h2h_data["away_team"] == home))
    ].sort_values(by="match_date", ascending=False).head(10)

    if not h2h_filtered.empty:
        st.dataframe(h2h_filtered[["match_date", "home_team", "away_team", "result"]])
    else:
        st.write("No recent head-to-head matches found for these teams.")
else:
    st.write("No predictions or head-to-head data available.")

