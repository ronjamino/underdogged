import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise RuntimeError(
        "xgboost is not installed. Run:\n\n  pip install xgboost\n"
    ) from e


# These are the actual features your prepare_training_data.py generates
FEATURES = [
    "avg_goal_diff_h2h",
    "h2h_home_winrate",
    "home_form_winrate",
    "away_form_winrate",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
]

RESULT_MAP = {"A": "away_win", "H": "home_win", "D": "draw"}
LABEL_MAP  = {"home_win": 0, "draw": 1, "away_win": 2}

    
def _load_training_data(path: str) -> pd.DataFrame:
    """Load and validate training data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}. Run prepare_training_data.py first.")
    
    df = pd.read_csv(path)
    print(f"📊 Loaded {len(df)} rows from {path}")
    
    # Show what columns we actually have
    print(f"📋 Available columns: {', '.join(df.columns.tolist())}")
    
    # Check for required columns
    if "result" not in df.columns:
        raise ValueError("Training data missing 'result' column")
    
    df = df.dropna(subset=["result"])
    
    # Check for required features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Training data missing features: {missing}")
    
    # Check for NaNs in features
    feature_nans = df[FEATURES].isna().sum()
    if feature_nans.any():
        print("⚠️ NaNs detected in features:")
        print(feature_nans[feature_nans > 0])
        print("🔧 Dropping rows with NaN values...")
        df = df.dropna(subset=FEATURES)
        print(f"📊 Rows after cleaning: {len(df)}")
    
    return df


def train_model():
    """Train the ensemble model."""
    print("🚀 Starting model training...")
    
    # Load data
    df = _load_training_data("data/processed/training_data.csv")
    
    # Show class distribution
    print("\n📊 Class distribution:")
    print(df["result"].value_counts())
    print(f"Total samples: {len(df)}")

    # Prepare features and labels
    X = df[FEATURES]
    y_raw = df["result"].map(RESULT_MAP)
    
    # Check for unmapped results
    if y_raw.isna().any():
        bad_results = df[y_raw.isna()]["result"].unique()
        raise ValueError(f"Unmappable result values found: {bad_results}")
    
    y_encoded = y_raw.map(LABEL_MAP)

    # Split data
    print("\n🔄 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")

    # === Define base models ===
    print("\n🤖 Building ensemble model...")
    
    rf = RandomForestClassifier(
        n_estimators=250, 
        random_state=42, 
        n_jobs=-1,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        verbosity=0  # Suppress warnings
    )

    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    # Build estimators list
    estimators = [("rf", rf), ("xgb", xgb), ("nn", nn)]
    print(f"👥 Ensemble members: {[name for name, _ in estimators]}")

    # === Create soft-voting ensemble ===
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1
    )

    # Train the ensemble
    print("\n🏋️ Training ensemble model...")
    ensemble.fit(X_train, y_train)
    print("✅ Training complete!")

    # Evaluate on test set
    print("\n📈 Evaluating model performance...")
    y_pred = ensemble.predict(X_test)
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=["Home Win", "Draw", "Away Win"],
        digits=3
    ))

    # Confusion matrix
    print("\n🧮 Confusion Matrix:")
    print("    Predicted: Home | Draw | Away")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Home", "Draw", "Away"]
    for i, row in enumerate(cm):
        print(f"Actual {labels[i]:5}: {row[0]:5} | {row[1]:5} | {row[2]:5}")

    # Save models
    print("\n💾 Saving models...")
    os.makedirs("models", exist_ok=True)
    
    # Save ensemble
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    print("✅ Ensemble model saved to models/ensemble_model.pkl")
    
    # Also save individual models for analysis
    joblib.dump(rf, "models/rf_model.pkl")
    joblib.dump(xgb, "models/xgb_model.pkl")
    joblib.dump(nn, "models/mlp_model.pkl")
    print("✅ Individual models also saved (rf/xgb/mlp)")
    
    # Feature importance from Random Forest (need to access the fitted one from ensemble)
    print("\n📊 Feature Importance (from Random Forest):")
    try:
        # Get the fitted Random Forest from the ensemble
        rf_fitted = ensemble.named_estimators_['rf']
        feature_importance = pd.DataFrame({
            'feature': FEATURES,
            'importance': rf_fitted.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']:25} {row['importance']:.4f}")
    except Exception as e:
        print(f"  Could not extract feature importance: {e}")
    
    print("\n🎉 Model training completed successfully!")
    print(f"   Total samples used: {len(df)}")
    print(f"   Model accuracy: {accuracy:.2%}")
    print(f"   Models saved in: models/")


if __name__ == "__main__":
    train_model()