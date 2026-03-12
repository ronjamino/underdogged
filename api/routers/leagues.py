"""
League endpoints.

GET /leagues                          — list all supported leagues with metadata
GET /leagues/{league_id}/predictions  — predictions for a single league
"""

from fastapi import APIRouter, HTTPException

from api.dependencies import DB, VALID_LEAGUES
from api.schemas.league import LeagueOut
from api.schemas.prediction import PredictionOut
from api.routers.predictions import _get_df, _row_to_prediction

router = APIRouter()

# Static metadata for the five supported leagues
_LEAGUE_META: dict[str, dict] = {
    "PL":  {"name": "Premier League", "country": "England"},
    "ELC": {"name": "Championship",   "country": "England"},
    "BL1": {"name": "Bundesliga",     "country": "Germany"},
    "SA":  {"name": "Serie A",        "country": "Italy"},
    "PD":  {"name": "La Liga",        "country": "Spain"},
}


@router.get("", response_model=list[LeagueOut], summary="List supported leagues")
def list_leagues(db: DB):
    """
    Return all five supported leagues with their current fixture counts
    and the timestamp of the most recently updated data file.
    """
    df = db.predictions
    last_updated = db.get_last_updated()

    results = []
    for code, meta in _LEAGUE_META.items():
        count = int((df["league"] == code).sum()) if not df.empty else 0
        results.append(
            LeagueOut(
                id=code,
                name=meta["name"],
                country=meta["country"],
                match_count=count,
                last_updated=last_updated,
            )
        )
    return results


@router.get(
    "/{league_id}/predictions",
    response_model=list[PredictionOut],
    summary="Predictions for one league",
)
def league_predictions(league_id: str, db: DB):
    """
    Convenience route: all predictions filtered to a single league.
    Equivalent to GET /predictions?league={league_id}.
    """
    league_upper = league_id.upper()
    if league_upper not in VALID_LEAGUES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown league '{league_id}'. Valid codes: {sorted(VALID_LEAGUES)}",
        )

    df = _get_df(db)
    df = df[df["league"].str.upper() == league_upper]

    if df.empty:
        return []

    df = df.sort_values("max_proba", ascending=False)
    return [_row_to_prediction(r) for _, r in df.iterrows()]
