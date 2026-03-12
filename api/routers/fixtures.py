"""
Fixture endpoints.

GET /fixtures              — list upcoming fixtures, filter by league / date range
GET /fixtures/{match_id}   — single fixture detail
"""

from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.database import DataStore
from api.dependencies import DB, VALID_LEAGUES
from api.schemas.fixture import FixtureOut

router = APIRouter()


def _row_to_fixture(row: pd.Series) -> FixtureOut:
    md = row["match_date"]
    match_date_str = md.isoformat() if hasattr(md, "isoformat") else str(md)

    return FixtureOut(
        match_id=str(row["match_id"]),
        home_team=row["home_team"],
        away_team=row["away_team"],
        league=row.get("league") or row.get("league_code", ""),
        match_date=match_date_str,
        matchweek=None,     # not present in current predictions CSV
        status="upcoming",  # all rows in latest_predictions.csv are upcoming
    )


def _get_df(db: DataStore) -> pd.DataFrame:
    df = db.predictions
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="Fixture data is unavailable. Run the pipeline to generate predictions.",
        )
    return df


@router.get("", response_model=list[FixtureOut], summary="List upcoming fixtures")
def list_fixtures(
    db: DB,
    league: str | None = Query(
        default=None,
        description=f"Filter by league code. Valid values: {sorted(VALID_LEAGUES)}",
    ),
    from_date: date | None = Query(
        default=None,
        description="Include fixtures on or after this date (YYYY-MM-DD). Defaults to today.",
    ),
    to_date: date | None = Query(
        default=None,
        description="Include fixtures on or before this date (YYYY-MM-DD). Defaults to today + 14 days.",
    ),
):
    """
    Return upcoming fixtures derived from the latest predictions CSV.
    All returned fixtures have status='upcoming'.
    """
    if league is not None:
        league_upper = league.upper()
        if league_upper not in VALID_LEAGUES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid league '{league}'. Valid codes: {sorted(VALID_LEAGUES)}",
            )
        league = league_upper

    df = _get_df(db).copy()

    today = date.today()
    _from = from_date or today
    _to   = to_date   or (today + timedelta(days=14))

    if "match_date" in df.columns:
        df["_date_only"] = df["match_date"].dt.date
        df = df[(df["_date_only"] >= _from) & (df["_date_only"] <= _to)]
        df = df.drop(columns=["_date_only"])

    if league:
        df = df[df["league"].str.upper() == league]

    df = df.sort_values("match_date")
    return [_row_to_fixture(r) for _, r in df.iterrows()]


@router.get("/{match_id}", response_model=FixtureOut, summary="Single fixture")
def get_fixture(match_id: str, db: DB):
    """
    Return fixture details for a specific match identified by its `match_id`.
    """
    df = _get_df(db)
    row = df[df["match_id"] == match_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Fixture '{match_id}' not found.")
    return _row_to_fixture(row.iloc[0])
