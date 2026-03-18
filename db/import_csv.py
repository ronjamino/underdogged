"""
One-time import of existing CSV data into PostgreSQL.

Usage:
    python -m db.import_csv
"""

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db.connection import engine

_PREDICTIONS_CSV = Path("data/predictions/latest_predictions.csv")
_ODDS_CSV        = Path("data/odds/latest_odds.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_odds(conn) -> int:
    df = pd.read_csv(_ODDS_CSV)

    df["fetch_timestamp"] = pd.to_datetime(df["fetch_timestamp"], utc=True, errors="coerce")
    df["commence_time"]   = pd.to_datetime(df["commence_time"],   utc=True, errors="coerce")

    insert_sql = text("""
        INSERT INTO odds (
            match_id, commence_time, home_team_raw, away_team_raw,
            home_team, away_team, league, sport_key,
            home_odds, away_odds, draw_odds, num_bookmakers, fetch_timestamp
        ) VALUES (
            :match_id, :commence_time, :home_team_raw, :away_team_raw,
            :home_team, :away_team, :league, :sport_key,
            :home_odds, :away_odds, :draw_odds, :num_bookmakers, :fetch_timestamp
        )
        ON CONFLICT (match_id) DO UPDATE SET
            home_odds       = EXCLUDED.home_odds,
            away_odds       = EXCLUDED.away_odds,
            draw_odds       = EXCLUDED.draw_odds,
            num_bookmakers  = EXCLUDED.num_bookmakers,
            fetch_timestamp = EXCLUDED.fetch_timestamp,
            updated_at      = NOW()
    """)

    rows = df.to_dict("records")
    conn.execute(insert_sql, rows)
    return len(rows)


def _import_predictions(conn) -> int:
    df = pd.read_csv(_PREDICTIONS_CSV)

    df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")

    # Rename CSV columns → DB columns
    df = df.rename(columns={
        "home_win": "prob_home",
        "draw":     "prob_draw",
        "away_win": "prob_away",
    })

    # 'league_code' is already in the CSV; use it as the DB league_code column.
    # Drop the 'league' column (full name string) to avoid confusion.
    if "league" in df.columns:
        df = df.drop(columns=["league"])

    insert_sql = text("""
        INSERT INTO predictions (
            match_date, home_team, away_team, league_code,
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
            predicted_result, prob_home, prob_draw, prob_away,
            max_proba, confidence_label, prob_label,
            home_form, away_form, h2h_form
        ) VALUES (
            :match_date, :home_team, :away_team, :league_code,
            :avg_goal_diff_h2h, :h2h_home_winrate, :home_form_winrate, :away_form_winrate,
            :home_avg_goals_scored, :home_avg_goals_conceded,
            :away_avg_goals_scored, :away_avg_goals_conceded,
            :h2h_draw_rate, :h2h_total_goals,
            :home_draw_rate, :away_draw_rate, :combined_draw_rate,
            :home_venue_draw_rate, :away_venue_draw_rate, :current_season_draw_rate,
            :form_differential, :goals_differential, :expected_total_goals,
            :home_total_goals_avg, :away_total_goals_avg,
            :league_avg_goals, :league_draw_rate, :league_home_adv,
            :home_momentum, :away_momentum, :momentum_differential,
            :is_low_scoring, :is_defensive_match, :has_odds,
            :form_x_goals, :momentum_interaction, :draw_affinity,
            :home_true_prob, :draw_true_prob, :away_true_prob,
            :market_draw_confidence, :market_favorite_confidence,
            :market_competitiveness, :odds_spread,
            :predicted_result, :prob_home, :prob_draw, :prob_away,
            :max_proba, :confidence_label, :prob_label,
            :home_form, :away_form, :h2h_form
        )
        ON CONFLICT (home_team, away_team, match_date) DO UPDATE SET
            predicted_result            = EXCLUDED.predicted_result,
            prob_home                   = EXCLUDED.prob_home,
            prob_draw                   = EXCLUDED.prob_draw,
            prob_away                   = EXCLUDED.prob_away,
            max_proba                   = EXCLUDED.max_proba,
            confidence_label            = EXCLUDED.confidence_label,
            prob_label                  = EXCLUDED.prob_label,
            home_form                   = EXCLUDED.home_form,
            away_form                   = EXCLUDED.away_form,
            h2h_form                    = EXCLUDED.h2h_form,
            updated_at                  = NOW()
    """)

    rows = df.to_dict("records")
    conn.execute(insert_sql, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def import_csv() -> None:
    with engine.begin() as conn:
        print("Importing odds …")
        n_odds = _import_odds(conn)
        print(f"  {n_odds} odds rows upserted")

        print("Importing predictions …")
        n_preds = _import_predictions(conn)
        print(f"  {n_preds} prediction rows upserted")

    print("Import complete.")


if __name__ == "__main__":
    import_csv()
