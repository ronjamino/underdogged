import os
import time
import subprocess
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Underdogged - Multi-League Analysis")

# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    {
        "key":   "fetch_fixtures",
        "label": "📅 Fetch Fixtures",
        "module": "fetch.fetch_fixtures",
        "description": "Download upcoming match fixtures from the API.",
        "expected_lines": 25,
        "timeout": 120,
    },
    {
        "key":   "fetch_odds",
        "label": "💰 Fetch Odds",
        "module": "fetch.fetch_odds",
        "description": "Download latest bookmaker odds.",
        "expected_lines": 30,
        "timeout": 120,
    },
    {
        "key":   "prepare_data",
        "label": "🧱 Prepare Training Data",
        "module": "model.prepare_training_data",
        "description": "Build feature matrix from ~5,000 historical matches. Takes 1–3 min.",
        "expected_lines": 120,
        "timeout": 600,
    },
    {
        "key":   "train_model",
        "label": "🧠 Train Model",
        "module": "model.train_model",
        "description": "Train stacking ensemble with Optuna tuning (RF + XGB + MLP). Takes 10–15 min.",
        "expected_lines": 200,
        "timeout": 1200,
    },
    {
        "key":   "backtest",
        "label": "🔁 Backtest",
        "module": "predict.backtest",
        "description": "Walk-forward backtest — trains full ensemble per window. Takes 5–10 min.",
        "expected_lines": 100,
        "timeout": 900,
    },
    {
        "key":   "predict",
        "label": "🎯 Generate Predictions",
        "module": "predict.predict_fixtures",
        "description": "Score upcoming fixtures with the trained model.",
        "expected_lines": 60,
        "timeout": 180,
    },
]

_STEP_BY_KEY = {s["key"]: s for s in PIPELINE_STEPS}

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
for _k, _v in {
    "pending_step": None,   # key of the step to run this script execution
    "step_results": {},     # key → {"success": bool, "lines": [...], "elapsed": float}
    "run_all": False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Value-betting helpers
# ---------------------------------------------------------------------------
def calculate_implied_probabilities(home_odds, draw_odds, away_odds):
    if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
        return None, None, None, None
    home_implied = 1 / home_odds
    draw_implied = 1 / draw_odds
    away_implied = 1 / away_odds
    total_implied = home_implied + draw_implied + away_implied
    overround = (total_implied - 1) * 100
    return home_implied / total_implied, draw_implied / total_implied, away_implied / total_implied, overround

def calculate_value_score(model_prob, bookmaker_prob, min_edge=0.05):
    if pd.isna(model_prob) or pd.isna(bookmaker_prob) or bookmaker_prob == 0:
        return 0
    edge = model_prob - bookmaker_prob
    if abs(edge) < min_edge:
        return 0
    return edge / bookmaker_prob

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv("data/predictions/latest_predictions.csv", parse_dates=["match_date"])
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
        home_true, draw_true, away_true, overround = calculate_implied_probabilities(
            odds_row['home_odds'], odds_row['draw_odds'], odds_row['away_odds']
        )
        if home_true is None:
            continue
        home_value = calculate_value_score(pred_row['home_win'], home_true)
        draw_value = calculate_value_score(pred_row['draw'], draw_true)
        away_value = calculate_value_score(pred_row['away_win'], away_true)
        values = [
            ('home_win', home_value, pred_row['home_win'], home_true, odds_row['home_odds']),
            ('draw',     draw_value, pred_row['draw'],     draw_true, odds_row['draw_odds']),
            ('away_win', away_value, pred_row['away_win'], away_true, odds_row['away_odds']),
        ]
        values.sort(key=lambda x: abs(x[1]), reverse=True)
        best_outcome, best_value, model_prob, bookie_prob, odds = values[0]
        if abs(best_value) > 0.10:
            b = odds - 1
            p = model_prob
            q = 1 - model_prob
            kelly_fraction = max(0, min((b * p - q) / b if b > 0 else 0, 0.25)) if best_value > 0 and odds > 1 else 0
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
                'match_confidence': pred_row[['home_win', 'draw', 'away_win']].max(),
            })
    return pd.DataFrame(value_bets)

# ---------------------------------------------------------------------------
# Pipeline execution helper
# ---------------------------------------------------------------------------
def _run_step(step: dict, progress_bar, status_text, log_area):
    """
    Run one pipeline step as a subprocess, streaming stdout line-by-line
    into the provided Streamlit placeholders.

    Returns (success: bool, lines: list[str], elapsed: float).
    """
    module = step["module"]
    expected = max(step["expected_lines"], 1)
    timeout  = step["timeout"]

    lines = []
    start = time.time()

    try:
        proc = subprocess.Popen(
            ["python3", "-u", "-m", module],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
        )

        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip()
            if not line:
                continue
            lines.append(line)
            elapsed = time.time() - start

            # Hard timeout guard
            if elapsed > timeout:
                proc.kill()
                status_text.error(f"⏰ Timed out after {timeout}s")
                return False, lines, elapsed

            pct = min(len(lines) / expected, 0.95)
            short = line if len(line) <= 72 else line[:69] + "..."
            progress_bar.progress(pct, text=short)
            log_area.code("\n".join(lines[-20:]), language="bash")

        proc.wait()
        elapsed = time.time() - start

        if proc.returncode == 0:
            progress_bar.progress(1.0, text=f"✅ Complete ({elapsed:.0f}s)")
            return True, lines, elapsed
        else:
            progress_bar.progress(1.0, text=f"❌ Failed — exit code {proc.returncode}")
            return False, lines, elapsed

    except Exception as exc:
        elapsed = time.time() - start
        status_text.error(f"❌ Error: {exc}")
        return False, lines, elapsed


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚽ Underdogged — Multi-League Analysis")
st.caption("Advanced football prediction system with statistical modeling and market analysis")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "💰 Value Analysis", "⚙️ Pipeline"])

# ===========================================================================
# Tab 1: Predictions
# ===========================================================================
with tab1:
    df = load_predictions()

    if df.empty:
        st.warning("No predictions found. Use the ⚙️ Pipeline tab to generate them.")
    else:
        if "max_proba" not in df.columns:
            df["max_proba"] = df[["home_win", "draw", "away_win"]].max(axis=1)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            st.metric("High Confidence (≥70%)", len(df[df["max_proba"] >= 0.70]))
        with col3:
            st.metric("Leagues", df["league"].nunique() if "league" in df.columns else 1)
        with col4:
            if "league" in df.columns:
                st.metric("League Split", " | ".join(f"{k}: {v}" for k, v in df["league"].value_counts().items()))

        st.sidebar.header("Prediction Filters")

        if "league" in df.columns:
            selected_league = st.sidebar.selectbox("League", ["All"] + list(df["league"].unique()))
            if selected_league != "All":
                df = df[df["league"] == selected_league]

        min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.60, 0.01)

        date_window = None
        if not df.empty:
            date_min = df["match_date"].min()
            date_max = df["match_date"].max()
            if pd.notna(date_min) and pd.notna(date_max):
                date_window = st.sidebar.date_input("Filter by date", value=(date_min.date(), date_max.date()))

        filtered = df.copy()
        if "max_proba" not in filtered.columns:
            filtered["max_proba"] = filtered[["home_win", "draw", "away_win"]].max(axis=1)
        if isinstance(date_window, tuple) and len(date_window) == 2:
            start, end = date_window
            filtered = filtered[(filtered["match_date"].dt.date >= start) & (filtered["match_date"].dt.date <= end)]
        filtered = filtered[filtered["max_proba"] >= min_conf]

        st.subheader("🎯 Predictions")
        league_mapping = {
            "PL": "⚽ Premier League", "ELC": "🏆 Championship",
            "BL1": "🇩🇪 Bundesliga", "SA": "🇮🇹 Serie A", "PD": "🇪🇸 La Liga",
        }

        if filtered.empty:
            st.info("No matches meet the current filters.")
        else:
            filtered = filtered.sort_values(["league", "match_date", "max_proba"], ascending=[True, True, False])
            if "league" in filtered.columns:
                filtered = filtered.copy()
                filtered["league_display"] = filtered["league"].map(league_mapping).fillna(filtered["league"])

            display_cols = ["match_date", "home_team", "away_team", "predicted_result", "confidence_label", "prob_label"]
            if "league_display" in filtered.columns:
                display_cols.insert(3, "league_display")

            st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "match_date":        st.column_config.DatetimeColumn("Match Date", format="MMM DD, HH:mm"),
                    "home_team":         st.column_config.TextColumn("Home Team", width="medium"),
                    "away_team":         st.column_config.TextColumn("Away Team", width="medium"),
                    "league_display":    st.column_config.TextColumn("League", width="medium"),
                    "predicted_result":  st.column_config.TextColumn("Prediction", width="small"),
                    "confidence_label":  st.column_config.TextColumn("Confidence", width="small"),
                    "prob_label":        st.column_config.TextColumn("All Probabilities", width="large"),
                },
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Predictions by League")
                if "league" in filtered.columns:
                    for league_name, count in filtered["league"].value_counts().items():
                        avg_conf = filtered[filtered["league"] == league_name]["max_proba"].mean()
                        st.write(f"**{league_mapping.get(league_name, league_name)}**: {count} matches (avg {avg_conf:.1%})")
            with col2:
                st.subheader("🎲 Prediction Distribution")
                emoji_map = {"home_win": "🏠", "away_win": "✈️", "draw": "🤝"}
                for result_type, count in filtered["predicted_result"].value_counts().items():
                    pct = count / len(filtered) * 100
                    st.write(f"**{emoji_map.get(result_type, '')} {result_type.replace('_', ' ').title()}**: {count} ({pct:.1f}%)")

# ===========================================================================
# Tab 2: Value Analysis
# ===========================================================================
with tab2:
    st.header("💰 Value Betting Analysis")
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
        st.info("No value betting opportunities found. Make sure predictions and odds data are loaded.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        positive_value = len(value_df[value_df['value_score'] > 0])
        with col1: st.metric("Value Opportunities", len(value_df))
        with col2: st.metric("Positive Value Bets", positive_value)
        with col3:
            avg_edge = value_df[value_df['value_score'] > 0]['edge_pct'].mean() if positive_value > 0 else float('nan')
            st.metric("Avg Positive Edge", f"{avg_edge:.1f}%" if not np.isnan(avg_edge) else "N/A")
        with col4:
            st.metric("Avg Bookmaker Margin", f"{value_df['overround'].mean():.1f}%")

        st.sidebar.header("Value Betting Filters")
        min_value      = st.sidebar.slider("Minimum Value Score", 0.0, 2.0, 0.2, 0.1)
        min_confidence = st.sidebar.slider("Minimum Match Confidence", 0.0, 1.0, 0.4, 0.05)
        if "league" in value_df.columns:
            sel_v_league = st.sidebar.selectbox("League (Value)", ["All"] + list(value_df["league"].unique()))
            if sel_v_league != "All":
                value_df = value_df[value_df["league"] == sel_v_league]

        value_filtered = value_df[
            (value_df['value_score'].abs() >= min_value) &
            (value_df['match_confidence'] >= min_confidence)
        ].sort_values('value_score', key=abs, ascending=False)

        if value_filtered.empty:
            st.info("No value bets meet the current filter criteria.")
        else:
            st.subheader("🎯 Value Betting Opportunities")
            dv = value_filtered.copy()
            dv['fixture']             = dv['home_team'] + ' vs ' + dv['away_team']
            dv['bet_type']            = dv['best_bet'].str.replace('_', ' ').str.title()
            dv['model_prob_fmt']      = (dv['model_prob'] * 100).round(1).astype(str) + '%'
            dv['bookmaker_prob_fmt']  = (dv['bookmaker_prob'] * 100).round(1).astype(str) + '%'
            dv['edge_fmt']            = dv['edge_pct'].round(1).astype(str) + '%'
            dv['value_fmt']           = (dv['value_score'] * 100).round(1).astype(str) + '%'
            dv['kelly_fmt']           = dv['kelly_pct'].round(1).astype(str) + '%'
            dv['odds_fmt']            = dv['odds'].round(2).astype(str)
            dv['status'] = dv['value_score'].apply(
                lambda v: "🔥 Strong" if abs(v) > 0.5 else ("⚡ Good" if abs(v) > 0.3 else "💡 Mild")
            )
            st.dataframe(
                dv[['status','fixture','league','bet_type','odds_fmt',
                    'model_prob_fmt','bookmaker_prob_fmt','edge_fmt','value_fmt','kelly_fmt']],
                use_container_width=True, hide_index=True,
                column_config={
                    "status":             st.column_config.TextColumn("Status", width="small"),
                    "fixture":            st.column_config.TextColumn("Fixture", width="large"),
                    "league":             st.column_config.TextColumn("League", width="small"),
                    "bet_type":           st.column_config.TextColumn("Bet Type", width="small"),
                    "odds_fmt":           st.column_config.TextColumn("Odds", width="small"),
                    "model_prob_fmt":     st.column_config.TextColumn("Model %", width="small"),
                    "bookmaker_prob_fmt": st.column_config.TextColumn("Bookie %", width="small"),
                    "edge_fmt":           st.column_config.TextColumn("Edge", width="small"),
                    "value_fmt":          st.column_config.TextColumn("Value", width="small"),
                    "kelly_fmt":          st.column_config.TextColumn("Kelly", width="small"),
                },
            )

            if len(value_filtered) > 0:
                st.subheader("📊 Value Analysis by League")
                league_analysis = value_filtered.groupby('league').agg(
                    count=('value_score', 'count'),
                    avg_value=('value_score', 'mean'),
                    avg_edge=('edge_pct', 'mean'),
                    avg_conf=('match_confidence', 'mean'),
                ).round(2)
                for league, row in league_analysis.iterrows():
                    st.write(f"**{league}**: {int(row['count'])} opportunities, "
                             f"{row['avg_value']:+.1%} avg value, "
                             f"{row['avg_edge']:+.1f}% avg edge, "
                             f"{row['avg_conf']:.1%} avg confidence")

    with st.expander("⚠️ Model Calibration Notice", expanded=False):
        st.warning("""
        **Important Model Limitations:**
        - Very high value scores (>100%) are unusual in efficient betting markets
        - The model has not been validated against actual match results
        - Bookmakers have access to information not in the model

        **Recommendations:** Treat this as academic analysis, not betting advice.
        Focus on smaller value edges (10–20%) which are more realistic.
        """)

# ===========================================================================
# Tab 3: Pipeline
# ===========================================================================
with tab3:
    st.header("⚙️ Pipeline Controls")
    st.caption(
        "Run individual pipeline steps independently. "
        "Each step streams live output below its progress bar. "
        "**Tip:** You rarely need to re-run Fetch Fixtures / Fetch Odds and Train Model together — "
        "train once, then just re-run Generate Predictions when new fixtures appear."
    )

    # -- Run All button (top) -----------------------------------------------
    st.divider()
    col_all, col_note = st.columns([1, 3])
    with col_all:
        run_all_clicked = st.button("▶️ Run Full Pipeline", type="primary", use_container_width=True)
    with col_note:
        st.info("Runs all 6 steps in order. Expect 15–30 min total due to model training and backtest.")

    if run_all_clicked:
        st.session_state.run_all = True
        st.session_state.pending_step = None

    st.divider()

    # -- Individual step cards -----------------------------------------------
    # Render two columns of step cards
    left_steps  = PIPELINE_STEPS[:3]
    right_steps = PIPELINE_STEPS[3:]

    col_l, col_r = st.columns(2)

    def _render_step_card(step, container):
        with container:
            result = st.session_state.step_results.get(step["key"])
            status_icon = ""
            if result is not None:
                status_icon = " ✅" if result["success"] else " ❌"

            st.markdown(f"#### {step['label']}{status_icon}")
            st.caption(step["description"])

            btn_clicked = st.button(
                f"Run {step['label']}",
                key=f"btn_{step['key']}",
                use_container_width=True,
            )
            if btn_clicked:
                st.session_state.pending_step = step["key"]
                st.session_state.run_all = False

            # Show previous result summary if available
            if result is not None:
                if result["success"]:
                    st.success(f"Last run: {result['elapsed']:.0f}s — {len(result['lines'])} lines of output")
                else:
                    st.error(f"Last run failed after {result['elapsed']:.0f}s")
                with st.expander("📋 Last output", expanded=False):
                    st.code("\n".join(result["lines"][-30:]), language="bash")

            st.markdown("---")

    for step in left_steps:
        _render_step_card(step, col_l)
    for step in right_steps:
        _render_step_card(step, col_r)

    # -- Execute the pending step or run-all ---------------------------------
    steps_to_run = []

    if st.session_state.run_all:
        steps_to_run = list(PIPELINE_STEPS)
        st.session_state.run_all = False
    elif st.session_state.pending_step:
        key = st.session_state.pending_step
        if key in _STEP_BY_KEY:
            steps_to_run = [_STEP_BY_KEY[key]]
        st.session_state.pending_step = None

    if steps_to_run:
        st.divider()
        overall_bar = st.progress(0, text="Starting pipeline…")
        n = len(steps_to_run)

        for i, step in enumerate(steps_to_run):
            overall_pct = i / n
            overall_bar.progress(overall_pct, text=f"Step {i+1}/{n}: {step['label']}")

            st.markdown(f"**{step['label']}**")
            progress_bar = st.progress(0, text="Initialising…")
            status_text  = st.empty()
            log_area     = st.empty()

            success, lines, elapsed = _run_step(step, progress_bar, status_text, log_area)

            st.session_state.step_results[step["key"]] = {
                "success": success,
                "lines":   lines,
                "elapsed": elapsed,
            }

            if not success:
                overall_bar.progress((i + 1) / n, text=f"❌ {step['label']} failed — pipeline halted")
                st.error(f"Step '{step['label']}' failed. Fix the error before continuing.")
                break
        else:
            overall_bar.progress(1.0, text="✅ All steps complete!")
            st.success("Pipeline finished! Reloading predictions…")
            st.cache_data.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "**Underdogged Multi-League Analysis** | "
    "Advanced statistical modeling for football predictions | "
    "For educational and analytical purposes only"
)
