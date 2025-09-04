# ==========================================
# SQLite Database Migration Plan
# ==========================================

import sqlite3
import pandas as pd
from datetime import datetime
import os

DATABASE_PATH = "football_predictions.db"

# ==========================================
# 1. Database Schema Design
# ==========================================

DATABASE_SCHEMA = """
-- Matches table (master data)
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    league TEXT NOT NULL,
    match_date TIMESTAMP NOT NULL,
    actual_result TEXT,  -- H/D/A when known
    home_goals INTEGER,
    away_goals INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table (model outputs)
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_result TEXT NOT NULL,  -- home_win/draw/away_win
    home_win_prob REAL NOT NULL,
    draw_prob REAL NOT NULL,
    away_win_prob REAL NOT NULL,
    max_confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
);

-- Odds table (bookmaker data)
CREATE TABLE IF NOT EXISTS odds (
    odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    home_odds REAL,
    draw_odds REAL,
    away_odds REAL,
    home_implied_prob REAL,
    draw_implied_prob REAL,
    away_implied_prob REAL,
    overround REAL,
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
);

-- Value bets table (analysis results)
CREATE TABLE IF NOT EXISTS value_bets (
    value_bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    prediction_id INTEGER NOT NULL,
    bet_type TEXT NOT NULL,  -- home_win/draw/away_win
    model_prob REAL NOT NULL,
    bookmaker_prob REAL NOT NULL,
    odds_value REAL NOT NULL,
    value_score REAL NOT NULL,
    edge_percent REAL NOT NULL,
    kelly_percent REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches (match_id),
    FOREIGN KEY (prediction_id) REFERENCES predictions (prediction_id)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    evaluation_date DATE NOT NULL,
    total_predictions INTEGER,
    correct_predictions INTEGER,
    accuracy REAL,
    draw_recall REAL,
    draw_precision REAL,
    home_win_accuracy REAL,
    away_win_accuracy REAL,
    calibration_score REAL,  -- How well probabilities match reality
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches (league);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions (match_id);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds (match_id);
CREATE INDEX IF NOT EXISTS idx_value_bets_match ON value_bets (match_id);
"""

# ==========================================
# 2. Database Connection Manager
# ==========================================

class FootballDB:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection with foreign keys enabled"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def init_database(self):
        """Initialize database with schema"""
        with self.get_connection() as conn:
            conn.executescript(DATABASE_SCHEMA)
            print(f"Database initialized at {self.db_path}")
    
    def execute_query(self, query, params=None):
        """Execute a query and return results as DataFrame"""
        with self.get_connection() as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            else:
                return pd.read_sql_query(query, conn)
    
    def execute_insert(self, table, data_dict):
        """Insert a single record"""
        columns = ', '.join(data_dict.keys())
        placeholders = ', '.join(['?' for _ in data_dict])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, list(data_dict.values()))
            return cursor.lastrowid
    
    def execute_bulk_insert(self, table, df):
        """Bulk insert DataFrame to table"""
        with self.get_connection() as conn:
            df.to_sql(table, conn, if_exists='append', index=False)

# ==========================================
# 3. Migration Functions
# ==========================================

def migrate_csv_to_database():
    """Migrate existing CSV data to SQLite"""
    db = FootballDB()
    
    print("Starting CSV to SQLite migration...")
    
    # 1. Migrate predictions
    try:
        predictions_df = pd.read_csv("data/predictions/latest_predictions.csv")
        predictions_df['match_date'] = pd.to_datetime(predictions_df['match_date'])
        
        # Create match_id from teams and date
        predictions_df['match_id'] = (
            predictions_df['home_team'] + '_vs_' + predictions_df['away_team'] + '_' +
            predictions_df['match_date'].dt.strftime('%Y%m%d_%H%M')
        ).str.replace(' ', '_')
        
        # Insert matches first
        matches_data = predictions_df[[
            'match_id', 'home_team', 'away_team', 'league', 'match_date'
        ]].drop_duplicates()
        
        db.execute_bulk_insert('matches', matches_data)
        print(f"Migrated {len(matches_data)} matches")
        
        # Insert predictions
        predictions_clean = predictions_df[[
            'match_id', 'predicted_result', 'home_win', 'draw', 'away_win', 'max_proba'
        ]].copy()
        predictions_clean['model_version'] = 'v1.0_enhanced'
        predictions_clean.rename(columns={
            'home_win': 'home_win_prob',
            'draw': 'draw_prob', 
            'away_win': 'away_win_prob',
            'max_proba': 'max_confidence'
        }, inplace=True)
        
        db.execute_bulk_insert('predictions', predictions_clean)
        print(f"Migrated {len(predictions_clean)} predictions")
        
    except FileNotFoundError:
        print("No predictions CSV found, skipping...")
    
    # 2. Migrate odds data
    try:
        odds_df = pd.read_csv("data/odds/latest_odds.csv")
        
        # Create match_id for odds (may need fuzzy matching)
        odds_df['match_id'] = (
            odds_df['home_team'] + '_vs_' + odds_df['away_team'] + '_' +
            pd.to_datetime(odds_df.get('commence_time', datetime.now())).dt.strftime('%Y%m%d_%H%M')
        ).str.replace(' ', '_')
        
        odds_clean = odds_df[[
            'match_id', 'home_odds', 'draw_odds', 'away_odds'
        ]].copy()
        odds_clean['bookmaker'] = 'aggregated'
        
        # Calculate implied probabilities
        odds_clean['home_implied_prob'] = 1 / odds_clean['home_odds']
        odds_clean['draw_implied_prob'] = 1 / odds_clean['draw_odds']
        odds_clean['away_implied_prob'] = 1 / odds_clean['away_odds']
        odds_clean['overround'] = (
            odds_clean['home_implied_prob'] + 
            odds_clean['draw_implied_prob'] + 
            odds_clean['away_implied_prob'] - 1
        ) * 100
        
        db.execute_bulk_insert('odds', odds_clean)
        print(f"Migrated {len(odds_clean)} odds entries")
        
    except FileNotFoundError:
        print("No odds CSV found, skipping...")
    
    print("Migration completed!")

# ==========================================
# 4. Updated Data Access Layer
# ==========================================

class PredictionDataAccess:
    def __init__(self):
        self.db = FootballDB()
    
    def save_prediction(self, match_data, prediction_data, odds_data=None):
        """Save a complete prediction with match and odds data"""
        
        # 1. Insert/update match
        match_id = match_data['match_id']
        self.db.execute_insert('matches', match_data)
        
        # 2. Insert prediction
        prediction_data['match_id'] = match_id
        prediction_id = self.db.execute_insert('predictions', prediction_data)
        
        # 3. Insert odds if available
        if odds_data:
            odds_data['match_id'] = match_id
            self.db.execute_insert('odds', odds_data)
        
        return prediction_id
    
    def update_match_result(self, match_id, actual_result, home_goals, away_goals):
        """Update match with actual result"""
        query = """
        UPDATE matches 
        SET actual_result = ?, home_goals = ?, away_goals = ?, updated_at = CURRENT_TIMESTAMP
        WHERE match_id = ?
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (actual_result, home_goals, away_goals, match_id))
    
    def get_predictions_with_results(self, days_back=30):
        """Get predictions with actual results for validation"""
        query = """
        SELECT 
            m.match_id,
            m.home_team,
            m.away_team,
            m.league,
            m.match_date,
            m.actual_result,
            p.predicted_result,
            p.home_win_prob,
            p.draw_prob,
            p.away_win_prob,
            p.max_confidence,
            CASE 
                WHEN m.actual_result = 'H' AND p.predicted_result = 'home_win' THEN 1
                WHEN m.actual_result = 'D' AND p.predicted_result = 'draw' THEN 1
                WHEN m.actual_result = 'A' AND p.predicted_result = 'away_win' THEN 1
                ELSE 0
            END as correct_prediction
        FROM matches m
        JOIN predictions p ON m.match_id = p.match_id
        WHERE m.actual_result IS NOT NULL
        AND m.match_date >= datetime('now', '-{} days')
        ORDER BY m.match_date DESC
        """.format(days_back)
        
        return self.db.execute_query(query)
    
    def calculate_model_performance(self, model_version="v1.0_enhanced"):
        """Calculate and store model performance metrics"""
        results_df = self.get_predictions_with_results(days_back=365)  # Last year
        
        if len(results_df) == 0:
            print("No results available for performance calculation")
            return
        
        # Calculate metrics
        total_predictions = len(results_df)
        correct_predictions = results_df['correct_prediction'].sum()
        accuracy = correct_predictions / total_predictions
        
        # Draw-specific metrics
        draw_predictions = results_df[results_df['predicted_result'] == 'draw']
        actual_draws = results_df[results_df['actual_result'] == 'D']
        
        draw_recall = len(draw_predictions[draw_predictions['correct_prediction'] == 1]) / len(actual_draws) if len(actual_draws) > 0 else 0
        draw_precision = draw_predictions['correct_prediction'].mean() if len(draw_predictions) > 0 else 0
        
        # Save performance metrics
        performance_data = {
            'model_version': model_version,
            'evaluation_date': datetime.now().date(),
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'draw_recall': draw_recall,
            'draw_precision': draw_precision
        }
        
        self.db.execute_insert('model_performance', performance_data)
        
        print(f"Model Performance Calculated:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Draw Recall: {draw_recall:.2%}")
        print(f"  Draw Precision: {draw_precision:.2%}")
        
        return performance_data

# ==========================================
# 5. Integration Examples
# ==========================================

def update_existing_code_example():
    """Example of how to update existing code to use database"""
    
    # OLD: CSV approach
    # df = pd.read_csv("data/predictions/latest_predictions.csv")
    
    # NEW: Database approach
    db_access = PredictionDataAccess()
    
    # Get recent predictions
    df = db_access.db.execute_query("""
        SELECT m.*, p.predicted_result, p.home_win_prob, p.draw_prob, p.away_win_prob
        FROM matches m 
        JOIN predictions p ON m.match_id = p.match_id
        WHERE m.match_date >= datetime('now', '-7 days')
        ORDER BY m.match_date
    """)
    
    # Get value betting opportunities
    value_bets = db_access.db.execute_query("""
        SELECT vb.*, m.home_team, m.away_team, m.league
        FROM value_bets vb
        JOIN matches m ON vb.match_id = m.match_id
        WHERE vb.value_score > 0.2
        ORDER BY vb.value_score DESC
    """)
    
    return df, value_bets

# ==========================================
# 6. Results Tracking Integration
# ==========================================

def setup_results_tracking():
    """Set up automatic results tracking"""
    
    print("""
    To enable automatic results tracking, you'll need to:
    
    1. Add a results fetcher that runs after matches:
       - Use the same APIs (football-data.org) to get results
       - Match by team names and dates
       - Update matches table with actual_result
    
    2. Set up a daily cron job:
       - python -m update_results.py
       - Fetches results for matches in last 3 days
       - Updates database automatically
    
    3. Model validation dashboard:
       - Shows prediction accuracy over time
       - Calibration curves (predicted vs actual probabilities)
       - Value bet ROI tracking
    """)

if __name__ == "__main__":
    print("Football Prediction Database Migration")
    print("=====================================")
    
    # Initialize database
    db = FootballDB()
    
    # Run migration
    migrate_csv_to_database()
    
    # Test data access
    data_access = PredictionDataAccess()
    
    # Show some example queries
    print("\nExample: Recent predictions")
    recent = data_access.db.execute_query("""
        SELECT m.home_team, m.away_team, p.predicted_result, p.max_confidence
        FROM matches m 
        JOIN predictions p ON m.match_id = p.match_id
        ORDER BY m.match_date DESC
        LIMIT 5
    """)
    print(recent)