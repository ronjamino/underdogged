-- Underdogged PostgreSQL schema
-- Run once against Supabase via db/migrate.py

-- -----------------------------------------------------------------------
-- predictions
-- One row per upcoming fixture with ML probability outputs.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                      SERIAL PRIMARY KEY,

    -- fixture identity
    match_date              TIMESTAMPTZ NOT NULL,
    home_team               TEXT        NOT NULL,
    away_team               TEXT        NOT NULL,
    league_code             TEXT        NOT NULL,   -- 'PL','ELC','BL1','SA','PD'

    -- raw form / H2H features (stored for auditability, not queried heavily)
    avg_goal_diff_h2h           REAL,
    h2h_home_winrate            REAL,
    home_form_winrate           REAL,
    away_form_winrate           REAL,
    home_avg_goals_scored       REAL,
    home_avg_goals_conceded     REAL,
    away_avg_goals_scored       REAL,
    away_avg_goals_conceded     REAL,
    h2h_draw_rate               REAL,
    h2h_total_goals             REAL,
    home_draw_rate              REAL,
    away_draw_rate              REAL,
    combined_draw_rate          REAL,
    home_venue_draw_rate        REAL,
    away_venue_draw_rate        REAL,
    current_season_draw_rate    REAL,
    form_differential           REAL,
    goals_differential          REAL,
    expected_total_goals        REAL,
    home_total_goals_avg        REAL,
    away_total_goals_avg        REAL,
    league_avg_goals            REAL,
    league_draw_rate            REAL,
    league_home_adv             REAL,
    home_momentum               REAL,
    away_momentum               REAL,
    momentum_differential       REAL,
    is_low_scoring              REAL,
    is_defensive_match          REAL,
    has_odds                    REAL,
    form_x_goals                REAL,
    momentum_interaction        REAL,
    draw_affinity               REAL,

    -- odds-derived market features
    home_true_prob              REAL,
    draw_true_prob              REAL,
    away_true_prob              REAL,
    market_draw_confidence      REAL,
    market_favorite_confidence  REAL,
    market_competitiveness      REAL,
    odds_spread                 REAL,

    -- ML outputs
    predicted_result    TEXT NOT NULL,  -- 'home_win','draw','away_win'
    prob_home           REAL NOT NULL,
    prob_draw           REAL NOT NULL,
    prob_away           REAL NOT NULL,
    max_proba           REAL NOT NULL,
    confidence_label    TEXT,
    prob_label          TEXT,

    -- housekeeping
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (home_team, away_team, match_date)
);

CREATE INDEX IF NOT EXISTS idx_predictions_league_code ON predictions (league_code);
CREATE INDEX IF NOT EXISTS idx_predictions_match_date   ON predictions (match_date);

-- -----------------------------------------------------------------------
-- odds
-- One row per fixture from The Odds API.
-- match_id is the 32-char hex supplied by The Odds API.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds (
    id              SERIAL PRIMARY KEY,
    match_id        TEXT        NOT NULL UNIQUE,  -- Odds API 32-char hex
    commence_time   TIMESTAMPTZ NOT NULL,
    home_team_raw   TEXT        NOT NULL,
    away_team_raw   TEXT        NOT NULL,
    home_team       TEXT        NOT NULL,
    away_team       TEXT        NOT NULL,
    league          TEXT        NOT NULL,
    sport_key       TEXT        NOT NULL,
    home_odds       REAL,
    away_odds       REAL,
    draw_odds       REAL,
    num_bookmakers  INTEGER,
    fetch_timestamp TIMESTAMPTZ NOT NULL,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_odds_home_away ON odds (home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_odds_league    ON odds (league);
