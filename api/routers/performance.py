"""
Model performance endpoints.

GET /performance   — backtest summary stats + per-window breakdown
"""

import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import DB

router = APIRouter()

_SUMMARY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "backtest" / "summary.csv"


class WindowResult(BaseModel):
    window: int
    start_date: str
    end_date: str
    test_size: int
    accuracy: float
    draw_recall: float
    draw_precision: float
    n_confident_bets: int
    hit_rate: float | None
    roi_pct: float | None


class PerformanceSummary(BaseModel):
    avg_accuracy: float
    avg_hit_rate: float
    overall_roi_pct: float
    total_bets: int
    total_matches_tested: int
    windows: list[WindowResult]


class LiveRecord(BaseModel):
    total_predicted: int
    correct: int
    incorrect: int
    pending: int
    accuracy: float | None
    # confusion[predicted][actual] = count  e.g. confusion["H"]["D"] = 3
    confusion: dict


@router.get("", response_model=PerformanceSummary, summary="Backtest performance summary")
def get_performance():
    """
    Return walk-forward backtest results: per-window accuracy, hit rate, and
    ROI, plus aggregate stats across all windows.
    """
    if not _SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Backtest data unavailable. Run predict.backtest to generate it.",
        )

    df = pd.read_csv(_SUMMARY_PATH)

    windows = []
    for _, r in df.iterrows():
        windows.append(WindowResult(
            window=int(r["window"]),
            start_date=str(r["start_date"]),
            end_date=str(r["end_date"]),
            test_size=int(r["test_size"]),
            accuracy=round(float(r["accuracy"]), 3),
            draw_recall=round(float(r["draw_recall"]), 3),
            draw_precision=round(float(r["draw_precision"]), 3),
            n_confident_bets=int(r["n_confident_bets"]),
            hit_rate=round(float(r["hit_rate"]), 3) if pd.notna(r.get("hit_rate")) else None,
            roi_pct=round(float(r["roi_pct"]), 1) if pd.notna(r.get("roi_pct")) else None,
        ))

    avg_accuracy = round(df["accuracy"].mean(), 3)
    avg_hit_rate = round(df["hit_rate"].dropna().mean(), 3) if df["hit_rate"].notna().any() else 0.0
    total_bets = int(df["n_confident_bets"].sum())
    total_profit = df["total_profit"].dropna().sum()
    overall_roi = round(float(total_profit / total_bets * 100), 1) if total_bets else 0.0
    total_matches = int(df["test_size"].sum())

    return PerformanceSummary(  # type: ignore[call-arg]
        avg_accuracy=avg_accuracy,
        avg_hit_rate=avg_hit_rate,
        overall_roi_pct=overall_roi,
        total_bets=total_bets,
        total_matches_tested=total_matches,
        windows=windows,
    )


@router.get("/live", response_model=LiveRecord, summary="Live prediction accuracy record")
def get_live_record(db: DB):
    """
    Compute accuracy over predictions that have actual results recorded.
    """
    df = db.predictions
    if df.empty:
        return LiveRecord(total_predicted=0, correct=0, incorrect=0, pending=0, accuracy=None, by_outcome={})

    # Only past matches
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    past = df[df["match_date"] < now].copy()

    resolved   = past[past["actual_result"].notna()]
    pending_n  = int(past["actual_result"].isna().sum())

    outcome_remap = {"home_win": "H", "draw": "D", "away_win": "A"}
    resolved = resolved.copy()
    resolved["pred_code"]   = resolved["predicted_result"].map(outcome_remap)
    resolved["actual_code"] = resolved["actual_result"].map(outcome_remap)
    resolved["correct"]     = resolved["pred_code"] == resolved["actual_code"]

    total     = len(resolved)
    correct_n = int(resolved["correct"].sum())
    accuracy  = round(correct_n / total, 3) if total else None

    # Build full 3×3 confusion matrix: confusion[predicted][actual] = count
    labels = ["H", "D", "A"]
    confusion: dict = {p: {a: 0 for a in labels} for p in labels}
    for _, row in resolved.iterrows():
        p = row["pred_code"]
        a = row["actual_code"]
        if p in labels and a in labels:
            confusion[p][a] += 1

    return LiveRecord(
        total_predicted=total,
        correct=correct_n,
        incorrect=total - correct_n,
        pending=pending_n,
        accuracy=accuracy,
        confusion=confusion,
    )
