"""
Fetch actual match results for past predictions and update the DB.

For each prediction where match_date < NOW() and actual_result IS NULL,
queries football-data.org for the finished match and writes the result back.

Usage:
    python -m fetch.fetch_results
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# Map our league codes to football-data.org competition codes
LEAGUE_MAP = {
    "PL":  "PL",
    "ELC": "ELC",
    "BL1": "BL1",
    "SA":  "SA",
    "PD":  "PD",
}

# Map score outcome to our result labels
def _outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    elif away_score > home_score:
        return "away_win"
    return "draw"


def _normalize(name: str) -> str:
    """Lowercase + strip for fuzzy matching."""
    return name.lower().strip()


def fetch_results(dry_run: bool = False) -> int:
    """
    Fetch results for all unresolved past predictions.
    Returns the number of rows updated.
    """
    from db.connection import engine

    # --- 1. Load unresolved past predictions from DB ---
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, home_team, away_team, league_code, match_date
            FROM predictions
            WHERE match_date < NOW()
              AND actual_result IS NULL
            ORDER BY match_date ASC
        """)).mappings().all()

    if not rows:
        logger.info("No unresolved past predictions found.")
        return 0

    logger.info(f"Found {len(rows)} unresolved past predictions.")

    # --- 2. Group by league ---
    by_league: dict[str, list] = {}
    for r in rows:
        code = r["league_code"]
        by_league.setdefault(code, []).append(r)

    updated = 0

    for league_code, preds in by_league.items():
        api_code = LEAGUE_MAP.get(league_code)
        if not api_code:
            logger.warning(f"No API mapping for league {league_code}, skipping.")
            continue

        # Date range: earliest unresolved to today
        dates = [p["match_date"] for p in preds]
        date_from = min(dates).strftime("%Y-%m-%d")
        date_to   = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

        url = (f"{BASE_URL}/competitions/{api_code}/matches"
               f"?status=FINISHED&dateFrom={date_from}&dateTo={date_to}")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited — waiting 60s")
                time.sleep(60)
                resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            logger.error(f"Failed to fetch {league_code} results: {exc}")
            continue

        finished = resp.json().get("matches", [])
        logger.info(f"{league_code}: {len(finished)} finished matches from API")

        # Build lookup: (norm_home, norm_away) → match data
        lookup: dict[tuple, dict] = {}
        for m in finished:
            score = m.get("score", {}).get("fullTime", {})
            if score.get("home") is None:
                continue
            key = (_normalize(m["homeTeam"]["name"]), _normalize(m["awayTeam"]["name"]))
            lookup[key] = {
                "home_score": int(score["home"]),
                "away_score": int(score["away"]),
                "utc_date":   m["utcDate"],
            }

        # --- 3. Match predictions to results ---
        with engine.begin() as conn:
            for pred in preds:
                key = (_normalize(pred["home_team"]), _normalize(pred["away_team"]))
                result_data = lookup.get(key)

                if not result_data:
                    # Try partial match — first word of each team name
                    for (ah, aa), data in lookup.items():
                        ph = _normalize(pred["home_team"]).split()[0]
                        pa = _normalize(pred["away_team"]).split()[0]
                        if ph in ah and pa in aa:
                            result_data = data
                            break

                if not result_data:
                    logger.debug(f"No result found for {pred['home_team']} vs {pred['away_team']}")
                    continue

                outcome = _outcome(result_data["home_score"], result_data["away_score"])

                if dry_run:
                    logger.info(
                        f"[DRY RUN] {pred['home_team']} vs {pred['away_team']}: "
                        f"{result_data['home_score']}-{result_data['away_score']} → {outcome}"
                    )
                    updated += 1
                    continue

                conn.execute(text("""
                    UPDATE predictions
                    SET actual_result     = :actual_result,
                        home_score        = :home_score,
                        away_score        = :away_score,
                        result_fetched_at = NOW(),
                        updated_at        = NOW()
                    WHERE id = :id
                """), {
                    "actual_result": outcome,
                    "home_score":    result_data["home_score"],
                    "away_score":    result_data["away_score"],
                    "id":            pred["id"],
                })
                logger.info(
                    f"✅ {pred['home_team']} vs {pred['away_team']}: "
                    f"{result_data['home_score']}-{result_data['away_score']} → {outcome}"
                )
                updated += 1

        time.sleep(0.5)  # be polite between league requests

    logger.info(f"Results fetch complete — {updated} predictions updated.")
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fetch_results()
