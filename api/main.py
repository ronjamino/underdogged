"""
Underdogged FastAPI application entry point.

Run locally:
    uvicorn api.main:app --reload

API docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import fixtures, leagues, predictions

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
# CORS — allow the future Next.js frontend and local dev
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://underdogged.com",        # update with real production domain
        "https://www.underdogged.com",
    ],
    allow_credentials=True,
    allow_methods=["GET"],               # read-only API
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(fixtures.router,    prefix="/fixtures",    tags=["Fixtures"])
app.include_router(leagues.router,     prefix="/leagues",     tags=["Leagues"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    """Returns {"status": "ok"} — used by Railway and load balancers."""
    return {"status": "ok"}


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "Underdogged API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
