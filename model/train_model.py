# model/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_model():
    # Load your features
    df = pd.read_csv("data/processed/training_data.csv")

    # Select input features and target
    X = df[["avg_goal_diff_h2h", "h2h_home_winrate"]]
    y = df["label"]

    # Encode target labels as numbers
    label_map = {"home_win": 0, "draw": 1, "away_win": 2}
    y_encoded = y.map(label_map)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train baseline model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_map.keys()))

    print("🧮 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/random_forest_model.pkl")
    print("✅ Model saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    train_model()
