import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="Underdogged")

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv(
            "data/predictions/latest_predictions.csv",
            parse_dates=["match_date"]
        )
    except Exception:
        return pd.DataFrame()

st.title("Underdogged — Upcoming Picks")

df = load_predictions()
if df.empty:
    st.warning("No predictions found. Run the pipeline first.")
    st.stop()

# Controls
left, right = st.columns([2, 1])
with right:
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.60, 0.01)
with left:
    date_min = df["match_date"].min()
    date_max = df["match_date"].max()
    date_window = st.date_input(
        "Filter by date",
        value=(date_min.date(), date_max.date()) if pd.notna(date_min) else None
    )

# Filter by controls
filtered = df.copy()
if isinstance(date_window, tuple) and len(date_window) == 2:
    start, end = date_window
    filtered = filtered[(filtered["match_date"].dt.date >= start) & (filtered["match_date"].dt.date <= end)]

filtered = filtered[filtered[["home_win", "draw", "away_win"]].max(axis=1) >= min_conf]

st.subheader("Predictions")
if filtered.empty:
    st.info("No matches meet the current filters.")
else:
    # Nice compact table with probs and confidence
    show_cols = [
        "match_date", "home_team", "away_team",
        "predicted_result", "confidence_label", "prob_label",
        "home_win", "draw", "away_win",
        "avg_goal_diff_h2h", "h2h_home_winrate", "home_form_winrate", "away_form_winrate",
    ]
    # Sort by kickoff time then confidence
    filtered = filtered.sort_values(["match_date", "home_team", "away_team", "confidence_label"])
    st.dataframe(filtered[show_cols], use_container_width=True)

st.caption("Tip: adjust the confidence slider to narrow down to stronger edges.")
