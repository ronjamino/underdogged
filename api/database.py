"""
Data access layer for the Underdogged API.

Note: This project is CSV-based rather than SQLite. This module provides
the DataStore class and get_db() dependency that the routers use to read
predictions, odds, and backtest data from the existing CSV files.

CSV files consumed (read-only):
  data/predictions/latest_predictions.csv  — model predictions for upcoming fixtures
  data/odds/latest_odds.csv                — bookmaker odds from The Odds API
  data/backtest/summary.csv               — walk-forward backtest summary
"""

import hashlib
import pandas as pd
from pathlib import Path
from typing import Generator

DATA_DIR         = Path(__file__).parent.parent / "data"
PREDICTIONS_PATH = DATA_DIR / "predictions" / "latest_predictions.csv"
ODDS_PATH        = DATA_DIR / "odds"        / "latest_odds.csv"
BACKTEST_PATH    = DATA_DIR / "backtest"    / "summary.csv"

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
        # Normalise bookmaker implied probabilities (remove overround)
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
    In-memory view of the CSV files. A new instance is created per request
    so the API always serves the latest pipeline output without restart.
    """

    def __init__(self):
        self.predictions: pd.DataFrame = pd.DataFrame()
        self.odds: pd.DataFrame = pd.DataFrame()
        self.backtest: pd.DataFrame = pd.DataFrame()
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        if PREDICTIONS_PATH.exists():
            df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["match_date"])

            # Stable match_id
            df["match_id"] = df.apply(
                lambda r: _make_match_id(
                    r["home_team"],
                    r["away_team"],
                    str(r["match_date"].date())
                    if hasattr(r["match_date"], "date")
                    else str(r["match_date"])[:10],
                ),
                axis=1,
            )

            # Standardise outcome code
            df["predicted_outcome"] = df["predicted_result"].map(_OUTCOME_MAP)

            self.predictions = df

        if ODDS_PATH.exists():
            self.odds = pd.read_csv(ODDS_PATH)

        if BACKTEST_PATH.exists():
            self.backtest = pd.read_csv(BACKTEST_PATH)

    # ------------------------------------------------------------------
    def get_merged(self) -> pd.DataFrame:
        """
        Return the predictions DataFrame enriched with bookmaker odds columns.
        Merged on (home_team, away_team); unmatched rows get NaN odds.
        """
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

    # ------------------------------------------------------------------
    def get_last_updated(self) -> str:
        """ISO timestamp of the most recently modified source CSV."""
        paths = [PREDICTIONS_PATH, ODDS_PATH]
        mtimes = [p.stat().st_mtime for p in paths if p.exists()]
        if not mtimes:
            return "unknown"
        import datetime
        ts = datetime.datetime.fromtimestamp(max(mtimes), tz=datetime.timezone.utc)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_db() -> DataStore:
    """
    FastAPI dependency. Returns a fresh DataStore so every request
    reflects the latest pipeline output.
    """
    return DataStore()
