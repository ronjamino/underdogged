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

    # Drop rows where 'result' is NaN and check if there are still NaN values
    df = df.dropna(subset=['result'])

    # Check for any NaN values in the dataframe after dropping 'result' NaNs
    if df.isna().sum().any():
        print("⚠️ There are still NaN values in the dataset after cleaning!")
        print(df.isna().sum())
        return

    # Select input features and target
    X = df[["avg_goal_diff_h2h", "h2h_home_winrate"]]
    y = df["result"]

    # Map shorthand values to full labels
    result_map = {"A": "away_win", "H": "home_win", "D": "draw"}
    y = y.map(result_map)

    # Check if there are any NaN values after mapping
    if y.isna().sum() > 0:
        print(f"⚠️ There are {y.isna().sum()} NaN values after mapping the result column.")
        print("🚨 Rows with NaN in result after mapping:")
        print(y[y.isna()])

        # Optionally, drop the NaN values or handle them as needed
        df = df.dropna(subset=["result"])
        y = df["result"].map(result_map)

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
