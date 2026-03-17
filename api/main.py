"""
Underdogged FastAPI application entry point.

Run locally:
    uvicorn api.main:app --reload

API docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

import os

from dotenv import load_dotenv

load_dotenv()  # ensure DATABASE_URL is available before db.connection imports it

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.routers import fixtures, leagues, predictions, performance
from db.connection import get_db

app = FastAPI(
    title="Underdogged API",
    description=(
        "Football match prediction API. "
        "Exposes ML-generated Home / Draw / Away probabilities for upcoming fixtures "
        "across the Premier League, Championship, Bundesliga, Serie A, and La Liga."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the Next.js frontend and local dev
# ---------------------------------------------------------------------------
_frontend_url = os.getenv("FRONTEND_URL", "")
_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://underdogged.com",
    "https://www.underdogged.com",
    "https://underdogged.vercel.app",
]
if _frontend_url:
    _origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET"],               # read-only API
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(predictions.router,  prefix="/predictions",  tags=["Predictions"])
app.include_router(fixtures.router,     prefix="/fixtures",     tags=["Fixtures"])
app.include_router(leagues.router,      prefix="/leagues",      tags=["Leagues"])
app.include_router(performance.router,  prefix="/performance",  tags=["Performance"])


# ---------------------------------------------------------------------------
# Health check — always returns 200 so Railway considers the deploy live.
# DB status is surfaced in the body for observability.
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health(db: Session = Depends(get_db)):
    """Railway health check. Always 200; DB connectivity reported in body."""
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        db_status = f"unreachable: {exc}"
    return {"status": "ok", "database": db_status}


@app.get("/ping", tags=["Health"])
def ping():
    """Lightweight liveness probe — no DB dependency."""
    return {"pong": True}


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "Underdogged API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
