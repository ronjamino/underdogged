"""
Prediction endpoints.

GET /predictions             — list, filter by league / date range
GET /predictions/top         — top 5 highest-confidence picks (next gameweek)
GET /predictions/{match_id}  — single prediction by match_id
"""

import math
from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.database import DataStore, _value_bet
from api.dependencies import DB, VALID_LEAGUES
from api.schemas.prediction import PredictionOut

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_to_none(v):
    """Convert NaN / inf floats to None so Pydantic can serialise them."""
    if v is None:
        return None
    try:
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
    except TypeError:
        return None


def _row_to_prediction(row: pd.Series) -> PredictionOut:
    prob_home = float(row["home_win"])
    prob_draw = float(row["draw"])
    prob_away = float(row["away_win"])

    odds_home = _nan_to_none(row.get("bk_home_odds"))
    odds_draw = _nan_to_none(row.get("bk_draw_odds"))
    odds_away = _nan_to_none(row.get("bk_away_odds"))

    value = _value_bet(prob_home, prob_draw, prob_away, odds_home, odds_draw, odds_away)

    # Normalise match_date to ISO-8601 UTC string
    md = row["match_date"]
    if hasattr(md, "isoformat"):
        match_date_str = md.isoformat()
    else:
        match_date_str = str(md)

    def _f(key: str) -> float | None:
        v = row.get(key)
        try:
            f = float(v)
            return round(f, 4) if (f == f and abs(f) != float("inf")) else None
        except (TypeError, ValueError):
            return None

    return PredictionOut(
        match_id=str(row["match_id"]),
        home_team=row["home_team"],
        away_team=row["away_team"],
        league=row.get("league") or row.get("league_code", ""),
        match_date=match_date_str,
        prob_home=round(prob_home, 4),
        prob_draw=round(prob_draw, 4),
        prob_away=round(prob_away, 4),
        predicted_outcome=str(row.get("predicted_outcome", "")),
        confidence=round(float(row["max_proba"]), 4),
        odds_home=round(odds_home, 2) if odds_home else None,
        odds_draw=round(odds_draw, 2) if odds_draw else None,
        odds_away=round(odds_away, 2) if odds_away else None,
        value_bet=value,
        home_form_winrate=_f("home_form_winrate"),
        away_form_winrate=_f("away_form_winrate"),
        home_momentum=_f("home_momentum"),
        away_momentum=_f("away_momentum"),
        home_venue_draw_rate=_f("home_venue_draw_rate"),
        away_venue_draw_rate=_f("away_venue_draw_rate"),
        h2h_home_winrate=_f("h2h_home_winrate"),
        h2h_draw_rate=_f("h2h_draw_rate"),
        h2h_total_goals=_f("h2h_total_goals"),
        home_avg_goals_scored=_f("home_avg_goals_scored"),
        home_avg_goals_conceded=_f("home_avg_goals_conceded"),
        away_avg_goals_scored=_f("away_avg_goals_scored"),
        away_avg_goals_conceded=_f("away_avg_goals_conceded"),
        expected_total_goals=_f("expected_total_goals"),
    )


def _get_df(db: DataStore) -> pd.DataFrame:
    df = db.get_merged()
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="Predictions data is unavailable. Run the pipeline to generate predictions.",
        )
    return df


# ---------------------------------------------------------------------------
# Routes — /top MUST be defined before /{match_id} to avoid path collision
# ---------------------------------------------------------------------------

@router.get("/top", response_model=list[PredictionOut], summary="Top 5 high-confidence picks")
def get_top_predictions(db: DB):
    """
    Return the 5 upcoming predictions with the highest model confidence,
    across all leagues.
    """
    df = _get_df(db)
    top = df.nlargest(5, "max_proba")
    return [_row_to_prediction(r) for _, r in top.iterrows()]


@router.get("", response_model=list[PredictionOut], summary="List predictions")
def list_predictions(
    db: DB,
    league: str | None = Query(
        default=None,
        description=f"Filter by league code. Valid values: {sorted(VALID_LEAGUES)}",
    ),
    from_date: date | None = Query(
        default=None,
        description="Include matches on or after this date (YYYY-MM-DD). Defaults to today.",
    ),
    to_date: date | None = Query(
        default=None,
        description="Include matches on or before this date (YYYY-MM-DD). Defaults to today + 7 days.",
    ),
):
    """
    Return upcoming predictions, optionally filtered by league and/or date range.
    """
    # Validate league
    if league is not None:
        league_upper = league.upper()
        if league_upper not in VALID_LEAGUES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid league '{league}'. Valid codes: {sorted(VALID_LEAGUES)}",
            )
        league = league_upper

    df = _get_df(db)

    # Default date window
    today = date.today()
    _from = from_date or today
    _to   = to_date   or (today + timedelta(days=7))

    # Date filter — match_date is timezone-aware; compare to date only
    if "match_date" in df.columns:
        df["_date_only"] = df["match_date"].dt.date
        df = df[(df["_date_only"] >= _from) & (df["_date_only"] <= _to)]
        df = df.drop(columns=["_date_only"])

    if league:
        df = df[df["league"].str.upper() == league]

    if df.empty:
        return []

    df = df.sort_values("max_proba", ascending=False)
    return [_row_to_prediction(r) for _, r in df.iterrows()]


@router.get("/value", response_model=list[PredictionOut], summary="Value bets across all leagues")
def get_value_bets(db: DB):
    """
    Return upcoming predictions (today → today+7) where the model has a
    positive edge (≥5%) over bookmaker odds, across all leagues.
    """
    df = _get_df(db)

    today = date.today()
    _to = today + timedelta(days=7)
    if "match_date" in df.columns:
        df["_date_only"] = df["match_date"].dt.date
        df = df[(df["_date_only"] >= today) & (df["_date_only"] <= _to)]
        df = df.drop(columns=["_date_only"])

    rows = []
    for _, r in df.iterrows():
        pred = _row_to_prediction(r)
        if pred.value_bet is not None:
            rows.append((pred, r["max_proba"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [pred for pred, _ in rows]


@router.get("/{match_id}", response_model=PredictionOut, summary="Single prediction")
def get_prediction(match_id: str, db: DB):
    """
    Return the prediction for a specific match identified by its `match_id`.
    """
    df = _get_df(db)
    row = df[df["match_id"] == match_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found.")
    return _row_to_prediction(row.iloc[0])
