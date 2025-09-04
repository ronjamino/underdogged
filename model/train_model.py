import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise RuntimeError(
        "xgboost is not installed. Run:\n\n  pip install xgboost\n"
    ) from e

# Define feature groups - model will auto-detect which are available
CORE_FEATURES = [
    "avg_goal_diff_h2h",
    "h2h_home_winrate",
    "home_form_winrate",
    "away_form_winrate",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
]

DRAW_FEATURES = [
    "h2h_draw_rate",
    "combined_draw_rate", 
    "form_differential",
    "goals_differential",
    "expected_total_goals",
    "league_draw_rate",
    "momentum_differential",
    "is_low_scoring",
    "is_defensive_match",
    "h2h_total_goals",           
    "home_draw_rate",            
    "away_draw_rate",            
    "home_total_goals_avg",      
    "away_total_goals_avg",      
    "league_avg_goals",          
    "league_home_adv",           
    "home_momentum",             
    "away_momentum",             
]

ODDS_FEATURES = [
    "home_true_prob",
    "draw_true_prob",
    "away_true_prob",
    "market_draw_confidence",
    "market_competitiveness",
    "odds_spread",
    "market_favorite_confidence",
]

RESULT_MAP = {"A": "away_win", "H": "home_win", "D": "draw"}
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}


def _load_training_data(path: str) -> tuple:
    """Load and validate training data, auto-detecting available features."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}. Run prepare_training_data.py first.")
    
    df = pd.read_csv(path)
    print(f"📊 Loaded {len(df)} rows from {path}")
    
    # Auto-detect which features are available
    available_features = []
    feature_groups = {
        "Core": CORE_FEATURES,
        "Draw-focused": DRAW_FEATURES,
        "Odds-based": ODDS_FEATURES
    }
    
    print("\n📋 Feature availability check:")
    for group_name, features in feature_groups.items():
        available_in_group = [f for f in features if f in df.columns]
        available_features.extend(available_in_group)
        print(f"   {group_name}: {len(available_in_group)}/{len(features)} features")
        if len(available_in_group) < len(features):
            missing = [f for f in features if f not in df.columns]
            print(f"      Missing: {', '.join(missing[:3])}...")
    
    if len(available_features) < len(CORE_FEATURES):
        raise ValueError(f"Missing core features! Need at least: {CORE_FEATURES}")
    
    print(f"\n✅ Using {len(available_features)} total features")
    
    # Check for required columns
    if "result" not in df.columns:
        raise ValueError("Training data missing 'result' column")
    
    df = df.dropna(subset=["result"])
    df = df.dropna(subset=available_features)
    
    print(f"📊 Rows after cleaning: {len(df)}")
    
    return df, available_features


def calculate_class_weights(y):
    """Calculate balanced class weights with extra boost for draws."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weight_dict = dict(zip(classes, weights))
    
    # Boost draw weight significantly (draws are hardest to predict)
    class_weight_dict[1] *= 2.0  # Double the draw weight
    
    return class_weight_dict


def create_sample_weights(y, class_weights):
    """Convert class weights to sample weights for XGBoost."""
    return np.array([class_weights[int(val)] for val in y])


def train_model():
    """Train the enhanced ensemble model with draw focus."""
    print("🚀 Starting ENHANCED model training with draw focus...\n")
    
    # Load data
    df, features = _load_training_data("data/processed/training_data.csv")
    
    # Show class distribution
    print("\n📊 Class distribution:")
    result_counts = df["result"].value_counts()
    total = len(df)
    for result, count in result_counts.items():
        pct = count / total * 100
        label = {"H": "Home", "D": "Draw", "A": "Away"}[result]
        print(f"   {label:5}: {count:5} ({pct:5.1f}%)")
    print(f"   Total: {total}")
    
    # Check if we have draw features
    has_draw_features = any(f in features for f in DRAW_FEATURES)
    has_odds_features = any(f in features for f in ODDS_FEATURES)
    
    print(f"\n🎯 Feature set:")
    print(f"   ✅ Core features: Yes")
    print(f"   {'✅' if has_draw_features else '❌'} Draw-focused features: {'Yes' if has_draw_features else 'No'}")
    print(f"   {'✅' if has_odds_features else '❌'} Odds-based features: {'Yes' if has_odds_features else 'No'}")

    # Prepare features and labels
    X = df[features]
    y_raw = df["result"].map(RESULT_MAP)
    
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
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    print(f"\n⚖️ Class weights (to handle imbalance):")
    for class_id, weight in class_weights.items():
        class_name = ["Home Win", "Draw", "Away Win"][class_id]
        print(f"   {class_name}: {weight:.2f}")

    # === Define base models with class balancing ===
    print("\n🤖 Building ensemble model with class balancing...")
    
    # Random Forest with class weights
    rf = RandomForestClassifier(
        n_estimators=350,  # More trees for complex features
        random_state=42,
        n_jobs=-1,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=7,
        class_weight=class_weights,
        max_features='sqrt'
    )

    # XGBoost with custom objective for draws
    sample_weights = create_sample_weights(y_train, class_weights)
    
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,  # Very low learning rate
        max_depth=5,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=3.0,
        reg_alpha=2.0,
        min_child_weight=5,
        gamma=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        verbosity=0
    )

    # Neural Network - deeper for complex patterns
    nn = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32, 16),  # Deeper network for complex patterns
        activation="relu",
        alpha=1e-3,
        learning_rate="adaptive",
        learning_rate_init=5e-4,
        max_iter=1500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-4
    )

    # === Train individual models ===
    print("\n🏋️ Training individual models...")
    
    # Train RF
    print("   Training Random Forest...")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_score = (rf_pred == y_test).mean()
    
    # Train XGBoost with sample weights
    print("   Training XGBoost with weighted samples...")
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_pred = xgb.predict(X_test)
    xgb_score = (xgb_pred == y_test).mean()
    
    # Train NN on scaled data
    print("   Training Neural Network...")
    nn.fit(X_train_scaled, y_train)
    nn_pred = nn.predict(X_test_scaled)
    nn_score = (nn_pred == y_test).mean()
    
    print(f"\n📊 Individual model accuracies:")
    print(f"   Random Forest: {rf_score:.2%}")
    print(f"   XGBoost: {xgb_score:.2%}")
    print(f"   Neural Network: {nn_score:.2%}")
    
    # Check individual model draw performance
    print(f"\n🎯 Individual model DRAW recall:")
    for name, preds in [("RF", rf_pred), ("XGB", xgb_pred), ("NN", nn_pred)]:
        draw_mask = y_test == 1  # Draw class
        if draw_mask.sum() > 0:
            draw_recall = ((preds == 1) & draw_mask).sum() / draw_mask.sum()
            print(f"   {name}: {draw_recall:.1%} ({((preds == 1) & draw_mask).sum()}/{draw_mask.sum()})")
    
    # === Create weighted voting ensemble ===
    # Adjust weights based on performance
    # Quick fix: Favor RF and XGB since they're excellent at draws
    if nn_score < 0.40:  # NN performing poorly
        weights = [0.5, 0.5, 0.0]  # Exclude NN completely
        print(f"🚫 Excluding Neural Network (poor performance)")
    else:
        weights = [0.45, 0.45, 0.10]  # Heavily favor RF and XGB
        print(f"🎯 Draw-focused weights: RF=45%, XGB=45%, NN=10%")
        
    weights = [w / sum(weights) for w in weights]
    
    print(f"\n🎯 Ensemble weights: RF={weights[0]:.2f}, XGB={weights[1]:.2f}, NN={weights[2]:.2f}")
    
    # Need to recreate estimators for ensemble (sklearn requirement)
    rf_ensemble = RandomForestClassifier(**rf.get_params())
    xgb_ensemble = XGBClassifier(**xgb.get_params())
    nn_ensemble = MLPClassifier(**nn.get_params())
    
    ensemble = VotingClassifier(
        estimators=[("rf", rf_ensemble), ("xgb", xgb_ensemble), ("nn", nn_ensemble)],
        voting="soft",
        weights=weights,
        n_jobs=-1
    )
    
    # Train ensemble (it needs to retrain all models)
    print("\n🏋️ Training weighted ensemble...")
    
    # For XGBoost in ensemble, we need a workaround for sample weights
    # Train the ensemble with fit
    ensemble.fit(X_train, y_train)
    
    # === Evaluate ensemble ===
    print("\n📈 Evaluating ensemble performance...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")
    
    # Detailed classification report
    print("\n📊 Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=["Home Win", "Draw", "Away Win"],
        digits=3,
        output_dict=True
    )
    
    # Print formatted report
    print(f"{'Class':12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 55)
    for class_name in ["Home Win", "Draw", "Away Win"]:
        metrics = report[class_name]
        print(f"{class_name:12} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1-score']:>10.3f} {int(metrics['support']):>10}")
    
    print("-" * 55)
    print(f"{'Accuracy':12} {' ':>10} {' ':>10} {report['accuracy']:>10.3f} {int(report['macro avg']['support']):>10}")
    print(f"{'Macro avg':12} {report['macro avg']['precision']:>10.3f} "
          f"{report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}")
    print(f"{'Weighted avg':12} {report['weighted avg']['precision']:>10.3f} "
          f"{report['weighted avg']['recall']:>10.3f} {report['weighted avg']['f1-score']:>10.3f}")

    # Confusion matrix
    print("\n🧮 Confusion Matrix:")
    print("    Predicted: Home | Draw | Away")
    print("    " + "-" * 30)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Home", "Draw", "Away"]
    for i, row in enumerate(cm):
        print(f"Actual {labels[i]:5}: {row[0]:5} | {row[1]:5} | {row[2]:5}")
    
    # Calculate and highlight draw improvement
    draw_recall = report["Draw"]["recall"]
    draw_precision = report["Draw"]["precision"]
    draw_f1 = report["Draw"]["f1-score"]
    
    print(f"\n✨ DRAW PREDICTION PERFORMANCE:")
    print(f"   Recall: {draw_recall:.1%} (ability to find draws)")
    print(f"   Precision: {draw_precision:.1%} (accuracy when predicting draw)")
    print(f"   F1-Score: {draw_f1:.3f}")
    
    if draw_recall > 0.15:
        print(f"   ✅ EXCELLENT! Draw recall above 15%")
    elif draw_recall > 0.10:
        print(f"   ✅ GOOD! Draw recall above 10%")
    elif draw_recall > 0.05:
        print(f"   ⚠️ OK - Draw recall above 5% but could be better")
    else:
        print(f"   ❌ POOR - Draw recall below 5%, needs improvement")

    # Save models and metadata
    print("\n💾 Saving models and metadata...")
    os.makedirs("models", exist_ok=True)
    
    # Save ensemble
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    print("✅ Ensemble model saved")
    
    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Feature scaler saved")
    
    # Save feature list and metadata
    metadata = {
        "features": features,
        "n_features": len(features),
        "has_draw_features": has_draw_features,
        "has_odds_features": has_odds_features,
        "accuracy": accuracy,
        "draw_recall": draw_recall,
        "draw_precision": draw_precision,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    joblib.dump(metadata, "models/metadata.pkl")
    
    with open("models/features.txt", "w") as f:
        f.write("\n".join(features))
    print("✅ Feature list and metadata saved")
    
    # Also save individual models for analysis
    joblib.dump(rf, "models/rf_model.pkl")
    joblib.dump(xgb, "models/xgb_model.pkl")
    joblib.dump(nn, "models/mlp_model.pkl")
    print("✅ Individual models saved")
    
    # Feature importance analysis
    print("\n📊 Top 15 Feature Importance (from Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    max_importance = feature_importance['importance'].max()
    for _, row in feature_importance.iterrows():
        # Visual bar chart in terminal
        bar_length = int((row['importance'] / max_importance) * 30)
        bar = '█' * bar_length
        
        # Mark feature type
        if row['feature'] in DRAW_FEATURES:
            marker = "🎯"  # Draw feature
        elif row['feature'] in ODDS_FEATURES:
            marker = "💰"  # Odds feature
        else:
            marker = "📊"  # Core feature
            
        print(f"  {marker} {row['feature']:28} {row['importance']:.4f} {bar}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE - MODEL SUMMARY")
    print("=" * 60)
    print(f"   Total samples: {len(df):,}")
    print(f"   Features used: {len(features)}")
    print(f"   Overall accuracy: {accuracy:.2%}")
    print(f"   Draw recall: {draw_recall:.1%}")
    print(f"   Draw precision: {draw_precision:.1%}")
    
    if has_draw_features and has_odds_features:
        print(f"\n   ✅ FULL FEATURE SET - Using draw + odds features")
    elif has_draw_features:
        print(f"\n   ✅ Using draw features (no odds available)")
    else:
        print(f"\n   ⚠️ Basic features only - run enhanced prepare_training_data.py")
    
    print(f"\n   Models saved in: models/")
    print(f"   Ready for predictions!")


if __name__ == "__main__":
    train_model()