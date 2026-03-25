"""
LLM enrichment step — runs after the predictions pipeline.

Web-searches for current team context (injuries, form, motivation) and
produces BACK / MONITOR / SKIP verdicts with brief commentary.

Results are upserted to the llm_enrichment table and served by
api/routers/enrichment.py to the frontend summary cards.

Usage:
    python llm_enrichment.py [--dry-run] [--batch-size N]

    --batch-size N   Process at most N un-enriched matches then stop.
                     Defaults to all un-enriched matches.
                     Use small values (e.g. 3) when called from a frequent
                     cron job to spread API calls over time.
"""

import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import anthropic
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

_OUTCOME_MAP   = {"home_win": "H", "draw": "D", "away_win": "A"}
_OUTCOME_LABEL = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
_MODEL           = "claude-haiku-4-5-20251001"
_CALL_DELAY_S    = 60   # seconds between API calls (30K TPM rate limit)
_MIN_CONFIDENCE  = 0.55  # only enrich predictions with ≥55% model confidence
_MIN_EDGE        = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan(v):
    try:
        return None if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else v
    except TypeError:
        return None

def _make_match_id(home: str, away: str, date_str: str) -> str:
    return hashlib.sha1(f"{home}|{away}|{date_str}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Already-enriched check
# ---------------------------------------------------------------------------

def _already_enriched(engine, run_date: date) -> set[tuple[str, str, str]]:
    """
    Return a set of (home_team, away_team, section) tuples that already have
    an enrichment record for today. Used to skip completed matches so reruns
    and batch crons don't duplicate API calls.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT home_team, away_team, section FROM llm_enrichment WHERE run_date = :d"),
            {"d": run_date},
        ).fetchall()
    return {(r[0], r[1], r[2]) for r in rows}


# ---------------------------------------------------------------------------
# Data loading  (uses existing DataStore so merging / odds logic is shared)
# ---------------------------------------------------------------------------

def _load_data(session) -> tuple[list[dict], list[dict]]:
    """Return (confident_predictions, value_bets) as lists of dicts."""
    from api.database import DataStore, _value_bet

    store = DataStore(session)
    df = store.get_merged()

    if df.empty:
        return [], []

    now = datetime.now(timezone.utc)

    # Upcoming fixtures only
    df = df[df["match_date"] >= now].copy()
    df["match_date_str"] = df["match_date"].dt.strftime("%Y-%m-%d")
    df["predicted_outcome"] = df["predicted_result"].map(_OUTCOME_MAP)

    # Compute value_bet flag
    df["value_bet"] = df.apply(lambda r: _value_bet(
        float(r["prob_home"]), float(r["prob_draw"]), float(r["prob_away"]),
        _nan(r.get("bk_home_odds")), _nan(r.get("bk_draw_odds")), _nan(r.get("bk_away_odds")),
        _MIN_EDGE,
    ), axis=1)

    # --- Confident predictions (≥60%, no cap) ---
    preds_df = df[df["max_proba"] >= _MIN_CONFIDENCE].sort_values("max_proba", ascending=False)
    predictions = []
    for _, r in preds_df.iterrows():
        predictions.append({
            "match_id":   _make_match_id(r["home_team"], r["away_team"], r["match_date_str"]),
            "home_team":  r["home_team"],
            "away_team":  r["away_team"],
            "league":     r["league_code"],
            "match_date": r["match_date_str"],
            "prediction": _OUTCOME_LABEL.get(r["predicted_outcome"], r["predicted_outcome"]),
            "confidence": round(float(r["max_proba"]) * 100, 1),
        })

    # --- Value bets (all, no cap) ---
    vb_df = df[df["value_bet"].notna()].sort_values("max_proba", ascending=False)
    value_bets = []
    for _, r in vb_df.iterrows():
        vb = r["value_bet"]
        model_prob = float(r["prob_home"] if vb == "H" else r["prob_draw"] if vb == "D" else r["prob_away"])
        odds       = _nan(r.get(f"bk_{'home' if vb == 'H' else 'draw' if vb == 'D' else 'away'}_odds"))
        implied    = round(100 / odds, 1) if odds else None
        edge_pct   = round((model_prob - (1 / odds)) * 100, 1) if odds else None
        value_bets.append({
            "match_id":    _make_match_id(r["home_team"], r["away_team"], r["match_date_str"]),
            "home_team":   r["home_team"],
            "away_team":   r["away_team"],
            "league":      r["league_code"],
            "match_date":  r["match_date_str"],
            "market":      _OUTCOME_LABEL.get(vb, vb),
            "model_prob":  round(model_prob * 100, 1),
            "implied_prob": implied,
            "edge":        edge_pct,
            "confidence":  round(float(r["max_proba"]) * 100, 1),
        })

    return predictions, value_bets


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _prompt_predictions(match: dict) -> str:
    return f"""You are a football betting analyst. Use web search to find current information about this match, then give a verdict.

Match: {match['home_team']} vs {match['away_team']}
League: {match['league']}
Date: {match['match_date']}
Model prediction: {match['prediction']} ({match['confidence']}% confidence)

Search for: injury news, lineup concerns, suspension lists, recent form, motivation factors, and any tactical context that could affect this prediction.

Respond ONLY with valid JSON in this exact format (no markdown, no explanation outside the JSON):
{{
  "verdict": "BACK" | "MONITOR" | "SKIP",
  "commentary": "1-2 sentence plain English rationale referencing what you found"
}}

BACK    = current context supports the model signal
MONITOR = mixed signals — worth watching lineup news before betting
SKIP    = context contradicts model or significant risk factor found"""


def _prompt_value_bets(match: dict) -> str:
    return f"""You are a football value betting analyst. Use web search to find current information, then assess this bet.

Match: {match['home_team']} vs {match['away_team']}
League: {match['league']}
Date: {match['match_date']}
Bet: {match['market']}
Model probability: {match['model_prob']}% vs bookmaker implied {match['implied_prob']}%
Edge: +{match['edge']}%

Search for: injury news, lineup concerns, suspensions, current form, motivation factors, and anything that could affect the value here.

Respond ONLY with valid JSON in this exact format (no markdown, no explanation outside the JSON):
{{
  "verdict": "BACK" | "MONITOR" | "SKIP",
  "commentary": "1-2 sentence plain English rationale referencing what you found"
}}

BACK    = context supports the value — edge looks real
MONITOR = mixed signals — check lineup closer to kickoff
SKIP    = context undermines the model edge"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _enrich_one(client: anthropic.Anthropic, match: dict, section: str) -> dict | None:
    prompt = _prompt_predictions(match) if section == "predictions" else _prompt_value_bets(match)
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=400,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract the last text block (after any tool-use blocks)
        text_block = next(
            (b.text for b in reversed(response.content) if hasattr(b, "text")),
            None,
        )
        if not text_block:
            logger.warning("No text block in response for %s vs %s", match["home_team"], match["away_team"])
            return None

        # Strip possible markdown fences
        clean = text_block.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()

        parsed = json.loads(clean)
        verdict = parsed.get("verdict", "MONITOR").upper()
        if verdict not in ("BACK", "MONITOR", "SKIP"):
            verdict = "MONITOR"

        return {
            "run_date":         date.today(),
            "match_id":         match.get("match_id"),
            "home_team":        match["home_team"],
            "away_team":        match["away_team"],
            "section":          section,
            "verdict":          verdict,
            "commentary":       parsed.get("commentary", ""),
            "model_confidence": match.get("confidence"),
            "edge_pct":         match.get("edge"),
            "market":           match.get("market"),
        }
    except (json.JSONDecodeError, anthropic.APIError) as exc:
        logger.error("Enrichment failed for %s vs %s: %s", match["home_team"], match["away_team"], exc)
        return None


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(engine, records: list[dict]) -> int:
    sql = text("""
        INSERT INTO llm_enrichment
            (run_date, match_id, home_team, away_team, section, verdict,
             commentary, model_confidence, edge_pct, market)
        VALUES
            (:run_date, :match_id, :home_team, :away_team, :section, :verdict,
             :commentary, :model_confidence, :edge_pct, :market)
        ON CONFLICT (run_date, home_team, away_team, section)
        DO UPDATE SET
            verdict           = EXCLUDED.verdict,
            commentary        = EXCLUDED.commentary,
            model_confidence  = EXCLUDED.model_confidence,
            edge_pct          = EXCLUDED.edge_pct,
            market            = EXCLUDED.market,
            created_at        = NOW()
    """)
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False, batch_size: int | None = None) -> int:
    """
    Run LLM enrichment. Returns number of records upserted.

    Parameters
    ----------
    dry_run    : If True, log what would be enriched without calling the API.
    batch_size : Maximum number of un-enriched matches to process in this
                 invocation. None means process all. Use a small value (e.g. 3)
                 when called from a frequent cron so API calls are spread over
                 time without exceeding rate limits.
    """
    from db.connection import engine, SessionLocal

    session = SessionLocal()
    try:
        predictions, value_bets = _load_data(session)
    finally:
        session.close()

    if not predictions and not value_bets:
        logger.info("Nothing to enrich — no confident predictions or value bets found.")
        return 0

    # Filter out matches already enriched today
    done = _already_enriched(engine, date.today())
    all_matches: list[tuple[str, dict]] = (
        [("predictions", m) for m in predictions] +
        [("value_bets",  m) for m in value_bets]
    )
    pending = [
        (section, m) for section, m in all_matches
        if (m["home_team"], m["away_team"], section) not in done
    ]

    logger.info(
        "Enrichment status: %d total, %d already done, %d pending",
        len(all_matches), len(done), len(pending),
    )

    if not pending:
        logger.info("All matches already enriched for today.")
        return 0

    # Apply batch cap
    batch = pending[:batch_size] if batch_size else pending
    logger.info("Processing %d match(es) this run (batch_size=%s)", len(batch), batch_size)

    if dry_run:
        logger.info("[DRY RUN] Would enrich: %s", [
            f"{m['home_team']} v {m['away_team']} [{s}]" for s, m in batch
        ])
        return 0

    client = anthropic.Anthropic()
    records = []

    for i, (section, match) in enumerate(batch):
        logger.info("Enriching [%d/%d] %s: %s vs %s", i + 1, len(batch), section, match["home_team"], match["away_team"])
        result = _enrich_one(client, match, section)
        if result:
            records.append(result)
        if i < len(batch) - 1:
            time.sleep(_CALL_DELAY_S)

    if not records:
        logger.warning("All enrichment calls failed.")
        return 0

    n = _upsert(engine, records)
    logger.info("Upserted %d enrichment record(s). %d match(es) still pending.", n, len(pending) - len(batch))
    return n


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true", help="Print matches without calling API")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="Process at most N un-enriched matches then stop")
    args = parser.parse_args()
    run(dry_run=args.dry_run, batch_size=args.batch_size)
