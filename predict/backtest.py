"""
Walk-forward backtesting for the Underdogged prediction pipeline.

Splits historical feature data chronologically into N test windows.
For each window: trains on all preceding data using the full stacking
ensemble (train_pipeline), predicts the window, compares predictions
to actual results, and (where raw bookmaker odds are available)
calculates simulated ROI using a flat £1 stake per bet.

Usage:
    python -m predict.backtest
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from model.train_model import train_pipeline

LABEL_MAP  = {"home_win": 0, "draw": 1, "away_win": 2}
RESULT_MAP = {"H": "home_win", "D": "draw", "A": "away_win"}
IDX_TO_LABEL = {0: "home_win", 1: "draw", 2: "away_win"}

# Minimum training samples before we attempt a prediction window
MIN_TRAIN_SAMPLES = 300

# Confidence threshold — predictions below this are skipped for ROI calc
CONFIDENCE_THRESHOLD = 0.55

# Columns that are metadata / raw odds — not model features
_NON_FEATURE_COLS = {
    "home_team", "away_team", "match_date", "league", "result", "y",
    "raw_home_odds", "raw_draw_odds", "raw_away_odds",
}


def _feature_cols(df):
    """Return all numeric feature columns (exclude metadata, target, and raw odds)."""
    return [c for c in df.columns
            if c not in _NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def run_backtest(n_splits: int = 5, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    Walk-forward backtest over data/processed/training_data.csv.

    Each window trains the full stacking ensemble via train_pipeline()
    (no Optuna tuning for speed), then predicts the test window.

    ROI is simulated using raw bookmaker odds (raw_home/draw/away_odds)
    rather than inverting stored true probabilities, giving a realistic
    estimate that includes the bookmaker's overround margin.

    Parameters
    ----------
    n_splits : int
        Number of chronological test windows.
    confidence_threshold : float
        Min predicted probability to count as a 'confident' bet for ROI.

    Returns
    -------
    summary_df : pd.DataFrame  (one row per window)
    bets_df    : pd.DataFrame  (one row per simulated bet)
    """
    path = "data/processed/training_data.csv"
    if not os.path.exists(path):
        print(f"❌ Training data not found at {path}. Run prepare_training_data.py first.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df = df.dropna(subset=["result"])

    # Map result to encoded label
    df["y"] = df["result"].map(RESULT_MAP).map(LABEL_MAP)
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    features = _feature_cols(df)
    # Fill remaining NaNs with 0 (consistent with has_odds=0 approach in train_model)
    df[features] = df[features].fillna(0.0)

    has_raw_odds = all(c in df.columns for c in ["raw_home_odds", "raw_draw_odds", "raw_away_odds"])

    print(f"📊 Loaded {len(df)} matches  ({df['match_date'].min().date()} → {df['match_date'].max().date()})")
    print(f"📋 Features: {len(features)}")
    print(f"💰 Raw bookmaker odds available: {'Yes' if has_raw_odds else 'No'}")

    # --- Build time-based splits ---
    window_size = len(df) // (n_splits + 1)

    windows = []
    for i in range(n_splits):
        test_start_idx = window_size * (i + 1)
        test_end_idx   = min(window_size * (i + 2), len(df))
        train_mask = df.index < test_start_idx
        test_mask  = (df.index >= test_start_idx) & (df.index < test_end_idx)
        if train_mask.sum() < MIN_TRAIN_SAMPLES or test_mask.sum() == 0:
            continue
        windows.append((train_mask, test_mask))

    print(f"\n🔄 Running {len(windows)}-window walk-forward backtest (full ensemble per window)...\n")

    summary_rows = []
    all_bets = []

    for w_idx, (train_mask, test_mask) in enumerate(windows):
        train_df = df[train_mask]
        test_df  = df[test_mask]

        X_train = train_df[features]
        y_train = train_df["y"]
        X_test  = test_df[features]
        y_test  = test_df["y"].values

        window_start = test_df["match_date"].min().date()
        window_end   = test_df["match_date"].max().date()

        print(f"  Window {w_idx + 1}/{len(windows)}: {window_start} → {window_end}  "
              f"(train={len(train_df)}, test={len(test_df)})")
        print(f"    Training full stacking ensemble...")

        # Train the full pipeline (no Optuna tuning for backtest speed)
        model = train_pipeline(X_train, y_train, n_trials=0)

        y_proba = model.predict_proba(X_test)
        y_pred  = y_proba.argmax(axis=1)
        confidence = y_proba.max(axis=1)

        # --- Per-class metrics ---
        accuracy = (y_pred == y_test).mean()

        draw_mask_test = y_test == 1
        draw_recall = float(
            ((y_pred == 1) & draw_mask_test).sum() / draw_mask_test.sum()
        ) if draw_mask_test.sum() else 0.0
        draw_precision_denom = (y_pred == 1).sum()
        draw_precision = float(
            ((y_pred == 1) & draw_mask_test).sum() / draw_precision_denom
        ) if draw_precision_denom else 0.0

        # --- Simulated ROI using raw bookmaker odds (Issue 10) ---
        # Raw odds (B365H/D/A) preserve the actual bookmaker margin; inverting
        # stored true probabilities would inflate ROI by ~5-8%.
        roi_records = []
        raw_odds_map = {
            0: "raw_home_odds",
            1: "raw_draw_odds",
            2: "raw_away_odds",
        }

        for i, (pred, prob, actual) in enumerate(zip(y_pred, confidence, y_test)):
            if prob < confidence_threshold:
                continue

            outcome_label = IDX_TO_LABEL[pred]
            row = test_df.iloc[i]

            if has_raw_odds:
                raw_col = raw_odds_map[pred]
                raw_val = row.get(raw_col, np.nan)
                decimal_odds = float(raw_val) if not pd.isna(raw_val) and float(raw_val) > 1.0 else np.nan
            else:
                decimal_odds = np.nan

            won = int(pred == actual)
            profit = (decimal_odds - 1.0) * won - (1.0 - won) if not np.isnan(decimal_odds) else np.nan

            roi_records.append({
                "window": w_idx + 1,
                "match_date": row["match_date"],
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "league": row.get("league", ""),
                "predicted": outcome_label,
                "actual": IDX_TO_LABEL[actual],
                "correct": bool(won),
                "confidence": round(float(prob), 3),
                "decimal_odds": round(decimal_odds, 2) if not np.isnan(decimal_odds) else None,
                "profit": round(float(profit), 2) if not np.isnan(profit) else None,
            })

        window_bets = pd.DataFrame(roi_records)
        all_bets.append(window_bets)

        n_bets = len(window_bets)
        hit_rate = window_bets["correct"].mean() if n_bets else np.nan
        total_profit = window_bets["profit"].sum() if n_bets and window_bets["profit"].notna().any() else np.nan
        roi_pct = (total_profit / n_bets * 100) if (n_bets and not np.isnan(total_profit)) else np.nan

        summary_rows.append({
            "window": w_idx + 1,
            "start_date": str(window_start),
            "end_date": str(window_end),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "accuracy": round(accuracy, 3),
            "draw_recall": round(draw_recall, 3),
            "draw_precision": round(draw_precision, 3),
            "n_confident_bets": n_bets,
            "hit_rate": round(hit_rate, 3) if not np.isnan(hit_rate) else None,
            "total_profit": round(total_profit, 2) if not np.isnan(total_profit) else None,
            "roi_pct": round(roi_pct, 1) if not np.isnan(roi_pct) else None,
        })

        if not np.isnan(roi_pct):
            print(f"    Accuracy: {accuracy:.1%}  Draw recall: {draw_recall:.1%}  "
                  f"Bets: {n_bets}  ROI: {roi_pct:+.1f}%")
        else:
            print(f"    Accuracy: {accuracy:.1%}  Draw recall: {draw_recall:.1%}  "
                  f"Bets: {n_bets}  (no raw odds for ROI)")

    summary_df = pd.DataFrame(summary_rows)
    bets_df = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()

    # --- Print overall summary ---
    print("\n" + "=" * 60)
    print("📊 BACKTEST SUMMARY")
    print("=" * 60)
    if not summary_df.empty:
        print(f"  Windows:          {len(summary_df)}")
        print(f"  Avg accuracy:     {summary_df['accuracy'].mean():.1%}  "
              f"(range {summary_df['accuracy'].min():.1%}–{summary_df['accuracy'].max():.1%})")
        print(f"  Avg draw recall:  {summary_df['draw_recall'].mean():.1%}  "
              f"(range {summary_df['draw_recall'].min():.1%}–{summary_df['draw_recall'].max():.1%})")

        roi_rows = summary_df.dropna(subset=["roi_pct"])
        if not roi_rows.empty:
            total_bets   = int(summary_df["n_confident_bets"].sum())
            total_profit = summary_df["total_profit"].sum()
            overall_roi  = total_profit / total_bets * 100 if total_bets else 0
            print(f"  Total bets:       {total_bets}")
            print(f"  Total profit:     £{total_profit:+.2f}")
            print(f"  Overall ROI:      {overall_roi:+.1f}%  (based on raw bookmaker odds)")
        else:
            print("  ROI: N/A (no raw bookmaker odds in feature data)")

    # --- Save outputs ---
    os.makedirs("data/backtest", exist_ok=True)
    summary_path = "data/backtest/summary.csv"
    bets_path    = "data/backtest/bets.csv"
    summary_df.to_csv(summary_path, index=False)
    if not bets_df.empty:
        bets_df.to_csv(bets_path, index=False)
    print(f"\n💾 Saved: {summary_path}")
    if not bets_df.empty:
        print(f"💾 Saved: {bets_path}")

    return summary_df, bets_df


if __name__ == "__main__":
    run_backtest(n_splits=5)
