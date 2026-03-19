#!/usr/bin/env python3
"""
Enrichment batch runner — called by the Railway cron job every 5 minutes.

Processes up to 3 un-enriched matches per invocation, respecting the
60-second delay between API calls (~3 minutes per batch). The skip-if-done
logic in llm_enrichment.py ensures parallel/overlapping cron runs are safe.

Railway cron schedule: */5 * * * *
Command:               python scripts/run_enrichment_batch.py
"""

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

if __name__ == "__main__":
    import llm_enrichment
    n = llm_enrichment.run(batch_size=3)
    logging.getLogger(__name__).info("Batch enrichment done — %d record(s) upserted.", n)
