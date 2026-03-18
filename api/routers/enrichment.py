"""
LLM enrichment endpoints.

GET /enrichment/predictions  — AI verdicts for today's confident predictions
GET /enrichment/value-bets   — AI verdicts for today's value bets
"""

from datetime import date as date_type

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from db.connection import engine

router = APIRouter()


class EnrichmentItem(BaseModel):
    match_id:         str | None
    home_team:        str
    away_team:        str
    section:          str
    verdict:          str          # "BACK" | "MONITOR" | "SKIP"
    commentary:       str
    model_confidence: float | None
    edge_pct:         float | None
    market:           str | None


class EnrichmentResponse(BaseModel):
    run_date: str
    items:    list[EnrichmentItem]


_VERDICT_ORDER = {"BACK": 1, "MONITOR": 2, "SKIP": 3}

_SQL = text("""
    SELECT match_id, home_team, away_team, section, verdict,
           commentary, model_confidence, edge_pct, market
    FROM llm_enrichment
    WHERE run_date = :run_date AND section = :section
    ORDER BY CASE verdict
        WHEN 'BACK'    THEN 1
        WHEN 'MONITOR' THEN 2
        WHEN 'SKIP'    THEN 3
        ELSE 4 END
""")


def _fetch(section: str, run_date: str) -> EnrichmentResponse:
    with engine.connect() as conn:
        rows = conn.execute(_SQL, {"run_date": run_date, "section": section}).mappings().all()
    items = [EnrichmentItem(**dict(r)) for r in rows]
    return EnrichmentResponse(run_date=run_date, items=items)


@router.get("/predictions", response_model=EnrichmentResponse, summary="AI verdicts for predictions")
def get_predictions_enrichment(run_date: str = None):
    d = run_date or str(date_type.today())
    return _fetch("predictions", d)


@router.get("/value-bets", response_model=EnrichmentResponse, summary="AI verdicts for value bets")
def get_value_bets_enrichment(run_date: str = None):
    d = run_date or str(date_type.today())
    return _fetch("value_bets", d)
