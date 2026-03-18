#!/usr/bin/env python3
"""
Daily pipeline runner.
Runs: fetch odds → generate predictions → write to DB + CSV.

Usage:
    python scripts/run_daily_pipeline.py

Called by Railway cron job (0 6 * * *) or triggered manually.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so fetch.*, predict.* etc. are importable
# regardless of which directory the script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run():
    logger.info(f"Pipeline started at {datetime.now(timezone.utc).isoformat()}")

    # Step 0: Fetch actual results for past predictions → PostgreSQL
    logger.info("Step 0/2 — Fetching results for past predictions …")
    try:
        from fetch.fetch_results import fetch_results
        n = fetch_results()
        logger.info(f"Results fetch complete — {n} predictions updated.")
    except Exception as exc:
        logger.warning(f"Results fetch failed (non-fatal): {exc}", exc_info=True)
        # Non-fatal: pipeline continues even if results fetch fails

    # Step 1: Fetch latest odds from The Odds API → CSV + PostgreSQL
    logger.info("Step 1/2 — Fetching latest odds …")
    try:
        from fetch.fetch_odds import fetch_all_odds
        fetch_all_odds(pipeline_mode=True)
        logger.info("Odds fetch complete.")
    except Exception as exc:
        logger.error(f"Odds fetch failed: {exc}", exc_info=True)
        sys.exit(1)

    # Step 2: Generate predictions for all top-5 leagues → CSV + PostgreSQL
    logger.info("Step 2/2 — Generating predictions …")
    try:
        from predict.predict_fixtures import predict_top5
        predict_top5()
        logger.info("Predictions complete.")
    except Exception as exc:
        logger.error(f"Prediction generation failed: {exc}", exc_info=True)
        sys.exit(1)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    run()
