"""
Data access layer for the Underdogged API.

Reads from PostgreSQL (Supabase) via SQLAlchemy.
Returns pandas DataFrames so the existing routers need no changes.
"""

import hashlib
import math
from typing import Generator

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.connection import SessionLocal, engine

# Map internal predicted_result labels → standard outcome codes
_OUTCOME_MAP = {"home_win": "H", "draw": "D", "away_win": "A"}


def _make_match_id(home: str, away: str, date: str) -> str:
    """Stable 16-char hex ID derived from home team, away team, and match date."""
    return hashlib.sha1(f"{home}|{away}|{date}".encode()).hexdigest()[:16]


def _value_bet(
    prob_home: float, prob_draw: float, prob_away: float,
    odds_home: float | None, odds_draw: float | None, odds_away: float | None,
    min_edge: float = 0.05,
) -> str | None:
    """
    Return the outcome code ("H", "D", "A") with the best positive expected
    value against bookmaker odds, or None if no edge exceeds min_edge.
    """
    if not all(isinstance(o, float) and o > 1.0 for o in (odds_home, odds_draw, odds_away)
               if o is not None):
        return None

    try:
        inv_h = 1.0 / odds_home if odds_home else 0.0
        inv_d = 1.0 / odds_draw  if odds_draw  else 0.0
        inv_a = 1.0 / odds_away  if odds_away  else 0.0
        total = inv_h + inv_d + inv_a
        if total <= 0:
            return None
        fair_h = inv_h / total
        fair_d = inv_d / total
        fair_a = inv_a / total

        candidates = [
            ("H", prob_home, fair_h),
            ("D", prob_draw, fair_d),
            ("A", prob_away, fair_a),
        ]
        best = max(candidates, key=lambda x: x[1] - x[2])
        label, model_p, fair_p = best
        return label if (model_p - fair_p) >= min_edge else None
    except (ZeroDivisionError, TypeError):
        return None


class DataStore:
    """
    Loads predictions and odds from PostgreSQL into DataFrames.
    A new instance is created per request via get_db().
    """

    def __init__(self, session: Session):
        self._session = session
        self.predictions: pd.DataFrame = pd.DataFrame()
        self.odds: pd.DataFrame = pd.DataFrame()
        self._load()

    def _load(self):
        # --- predictions ---
        preds_sql = text("""
            SELECT
                id, match_date, home_team, away_team, league_code,
                avg_goal_diff_h2h, h2h_home_winrate, home_form_winrate, away_form_winrate,
                home_avg_goals_scored, home_avg_goals_conceded,
                away_avg_goals_scored, away_avg_goals_conceded,
                h2h_draw_rate, h2h_total_goals,
                home_draw_rate, away_draw_rate, combined_draw_rate,
                home_venue_draw_rate, away_venue_draw_rate, current_season_draw_rate,
                form_differential, goals_differential, expected_total_goals,
                home_total_goals_avg, away_total_goals_avg,
                league_avg_goals, league_draw_rate, league_home_adv,
                home_momentum, away_momentum, momentum_differential,
                is_low_scoring, is_defensive_match, has_odds,
                form_x_goals, momentum_interaction, draw_affinity,
                home_true_prob, draw_true_prob, away_true_prob,
                market_draw_confidence, market_favorite_confidence,
                market_competitiveness, odds_spread,
                predicted_result,
                prob_home, prob_draw, prob_away,
                max_proba, confidence_label, prob_label,
                actual_result, home_score, away_score,
                updated_at
            FROM predictions
            ORDER BY match_date ASC
        """)
        rows = self._session.execute(preds_sql).mappings().all()
        if rows:
            df = pd.DataFrame(rows)
            df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")

            # Aliases expected by the routers
            df["home_win"] = df["prob_home"]
            df["draw"]     = df["prob_draw"]
            df["away_win"] = df["prob_away"]
            df["league"]   = df["league_code"]

            # Stable match_id (routers use this for /predictions/{match_id})
            df["match_id"] = df.apply(
                lambda r: _make_match_id(
                    r["home_team"], r["away_team"],
                    r["match_date"].strftime("%Y-%m-%d") if hasattr(r["match_date"], "strftime") else str(r["match_date"])[:10],
                ),
                axis=1,
            )

            df["predicted_outcome"] = df["predicted_result"].map(_OUTCOME_MAP)
            self.predictions = df

        # --- odds ---
        odds_sql = text("""
            SELECT match_id, commence_time,
                   home_team, away_team, league,
                   home_odds, away_odds, draw_odds, num_bookmakers, fetch_timestamp
            FROM odds
        """)
        odds_rows = self._session.execute(odds_sql).mappings().all()
        if odds_rows:
            self.odds = pd.DataFrame(odds_rows)

    def get_merged(self) -> pd.DataFrame:
        """Predictions DataFrame enriched with bookmaker odds columns."""
        df = self.predictions.copy()

        if not self.odds.empty:
            cols_needed = {"home_team", "away_team", "home_odds", "draw_odds", "away_odds"}
            if cols_needed.issubset(self.odds.columns):
                odds_slim = (
                    self.odds[list(cols_needed)]
                    .drop_duplicates(subset=["home_team", "away_team"])
                    .rename(columns={
                        "home_odds": "bk_home_odds",
                        "draw_odds": "bk_draw_odds",
                        "away_odds": "bk_away_odds",
                    })
                )
                df = df.merge(odds_slim, on=["home_team", "away_team"], how="left")
            else:
                df["bk_home_odds"] = None
                df["bk_draw_odds"] = None
                df["bk_away_odds"] = None
        else:
            df["bk_home_odds"] = None
            df["bk_draw_odds"] = None
            df["bk_away_odds"] = None

        return df

    def get_last_updated(self) -> str:
        """ISO timestamp of the most recently updated prediction row."""
        result = self._session.execute(
            text("SELECT MAX(updated_at) FROM predictions")
        ).scalar()
        if result is None:
            return "unknown"
        import datetime
        if hasattr(result, "strftime"):
            if result.tzinfo is None:
                result = result.replace(tzinfo=datetime.timezone.utc)
            return result.strftime("%Y-%m-%dT%H:%M:%SZ")
        return str(result)


def get_db() -> Generator[DataStore, None, None]:
    """
    FastAPI dependency. Yields a DataStore backed by a fresh SQLAlchemy session.
    """
    session = SessionLocal()
    try:
        yield DataStore(session)
    finally:
        session.close()
