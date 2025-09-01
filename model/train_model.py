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


FEATURES = ["avg_goal_diff_h2h", "h2h_home_winrate", "home_form_winrate", "away_form_winrate"]
RESULT_MAP = {"A": "away_win", "H": "home_win", "D": "draw"}
LABEL_MAP  = {"home_win": 0, "draw": 1, "away_win": 2}


def _load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["result"])
    # basic sanity: ensure features are present
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Training data missing features: {missing}")
    if df.isna().sum().any():
        # be strict so we don't propagate NaNs into training
        print("⚠️ NaNs detected after cleaning; printing per-column counts:")
        print(df.isna().sum())
        df = df.dropna()
    return df


def train_model():
    # Load data
    df = _load_training_data("data/processed/training_data.csv")

    # X / y
    X = df[FEATURES]
    y = df["result"].map(RESULT_MAP)
    if y.isna().any():
        bad = df[y.isna()]
        raise ValueError(f"Unmappable result labels found in rows:\n{bad[['result']].head()}")

    y_encoded = y.map(LABEL_MAP)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # === Base models ===
    rf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)

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
        num_class=3,
    )

    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42
    )

    # 👉 Build the list here
    estimators = [("rf", rf), ("xgb", xgb), ("nn", nn)]

    # 👉 Now you can log it
    print("👥 Ensemble members:", [name for name, _ in estimators])

    # === Soft-voting ensemble ===
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        flatten_transform=True
    )

    # Train
    ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    print("📊 Classification Report (Ensemble):")
    print(classification_report(y_test, y_pred, target_names=LABEL_MAP.keys()))

    print("🧮 Confusion Matrix (Ensemble):")
    print(confusion_matrix(y_test, y_pred))

    # Persist
    os.makedirs("models", exist_ok=True)
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    print("✅ Ensemble model saved to models/ensemble_model.pkl")

    # (Optional) also save base learners for later analysis
    joblib.dump(rf,  "models/rf_model.pkl")
    joblib.dump(xgb, "models/xgb_model.pkl")
    joblib.dump(nn,  "models/mlp_model.pkl")
    print("ℹ️ Base learners also saved (rf/xgb/mlp).")


if __name__ == "__main__":
    train_model()
