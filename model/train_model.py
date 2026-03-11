import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

INTERACTION_FEATURES = [
    "form_x_goals",
    "momentum_interaction",
    "draw_affinity",
]

RESULT_MAP = {"A": "away_win", "H": "home_win", "D": "draw"}
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}


class StackingEnsemble:
    """
    Stacking ensemble: RF + XGB + NN base models feed into a logistic
    regression meta-learner. Drop-in replacement for VotingClassifier —
    exposes the same predict() / predict_proba() interface.
    """

    def __init__(self, rf, xgb, nn, scaler, meta_learner):
        self.rf = rf
        self.xgb = xgb
        self.nn = nn
        self.scaler = scaler
        self.meta_learner = meta_learner

    def _meta_features(self, X):
        X_scaled = self.scaler.transform(X)
        return np.hstack([
            self.rf.predict_proba(X),
            self.xgb.predict_proba(X),
            self.nn.predict_proba(X_scaled),
        ])

    def predict(self, X):
        return self.meta_learner.predict(self._meta_features(X))

    def predict_proba(self, X):
        return self.meta_learner.predict_proba(self._meta_features(X))


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
        "Odds-based": ODDS_FEATURES,
        "Interactions": INTERACTION_FEATURES,
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


def tune_hyperparameters(X_train, y_train, class_weights, n_trials=50):
    """
    Use Optuna to search RF and XGB hyperparameters, optimising draw F1.
    Returns (best_rf_params, best_xgb_params). If Optuna is not installed,
    returns empty dicts so callers fall back to hardcoded defaults.
    Reduce n_trials for faster runs (e.g. n_trials=20).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("⚠️  optuna not installed — skipping tuning (pip install optuna to enable)")
        return {}, {}

    from sklearn.metrics import make_scorer

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # --- Tune Random Forest ---
    def rf_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 5, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 5, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 15),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        )
        model = RandomForestClassifier(**params)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            y_pred = model.predict(X_train.iloc[val_idx])
            scores.append(f1_score(y_train.iloc[val_idx], y_pred, labels=[1], average="macro", zero_division=0))
        return float(np.mean(scores))

    print(f"   Tuning Random Forest ({n_trials} trials)...")
    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(rf_objective, n_trials=n_trials, show_progress_bar=False)
    best_rf_params = rf_study.best_params
    best_rf_params.update({"class_weight": class_weights, "random_state": 42, "n_jobs": -1})
    print(f"   RF best draw F1: {rf_study.best_value:.3f}")

    # --- Tune XGBoost ---
    def xgb_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 700),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.6, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.5, 3.0),
            min_child_weight=trial.suggest_int("min_child_weight", 3, 10),
            gamma=trial.suggest_float("gamma", 0.0, 0.3),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
        model = XGBClassifier(**params)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            sw = create_sample_weights(y_train.iloc[tr_idx], class_weights)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], sample_weight=sw)
            y_pred = model.predict(X_train.iloc[val_idx])
            scores.append(f1_score(y_train.iloc[val_idx], y_pred, labels=[1], average="macro", zero_division=0))
        return float(np.mean(scores))

    print(f"   Tuning XGBoost ({n_trials} trials)...")
    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
    best_xgb_params = xgb_study.best_params
    best_xgb_params.update({
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
    })
    print(f"   XGB best draw F1: {xgb_study.best_value:.3f}")

    return best_rf_params, best_xgb_params


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

    # Impute NaN odds features with column medians (rows without real odds data)
    odds_cols_present = [f for f in ODDS_FEATURES if f in df.columns]
    feature_medians = {}
    if odds_cols_present:
        medians = df[odds_cols_present].median()
        feature_medians = medians.to_dict()
        nan_count = df[odds_cols_present].isna().sum().sum()
        if nan_count > 0:
            df[odds_cols_present] = df[odds_cols_present].fillna(medians)
            print(f"📊 Imputed {nan_count} NaN odds values with column medians")

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

    # === Hyperparameter tuning ===
    print("\n🔍 Hyperparameter tuning with Optuna...")
    best_rf_params, best_xgb_params = tune_hyperparameters(
        X_train, y_train, class_weights, n_trials=50
    )

    # === Define base models (tuned params if available, else defaults) ===
    print("\n🤖 Building base models...")

    rf_defaults = dict(
        n_estimators=350, random_state=42, n_jobs=-1,
        max_depth=12, min_samples_split=15, min_samples_leaf=7,
        class_weight=class_weights, max_features="sqrt",
    )
    rf = RandomForestClassifier(**(best_rf_params if best_rf_params else rf_defaults))

    sample_weights = create_sample_weights(y_train, class_weights)

    xgb_defaults = dict(
        n_estimators=500, learning_rate=0.02, max_depth=5,
        subsample=0.75, colsample_bytree=0.75, reg_lambda=3.0,
        reg_alpha=2.0, min_child_weight=5, gamma=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", random_state=42, verbosity=0,
    )
    xgb = XGBClassifier(**(best_xgb_params if best_xgb_params else xgb_defaults))

    # Neural Network — not tuned (least impactful, scaling handled internally)
    nn = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32, 16),
        activation="relu",
        alpha=1e-3,
        learning_rate="adaptive",
        learning_rate_init=5e-4,
        max_iter=1500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-4,
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
    
    print(f"\n📊 Individual model accuracies (hold-out test):")
    print(f"   Random Forest: {rf_score:.2%}")
    print(f"   XGBoost: {xgb_score:.2%}")
    print(f"   Neural Network: {nn_score:.2%}")

    # K-fold cross-validation for robust accuracy estimate
    print(f"\n🔄 5-fold cross-validation (Random Forest)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        RandomForestClassifier(**rf.get_params()), X, y_encoded, cv=skf, scoring="accuracy", n_jobs=-1
    )
    print(f"   CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    
    # Check individual model draw performance
    print(f"\n🎯 Individual model DRAW recall:")
    for name, preds in [("RF", rf_pred), ("XGB", xgb_pred), ("NN", nn_pred)]:
        draw_mask = y_test == 1  # Draw class
        if draw_mask.sum() > 0:
            draw_recall = ((preds == 1) & draw_mask).sum() / draw_mask.sum()
            print(f"   {name}: {draw_recall:.1%} ({((preds == 1) & draw_mask).sum()}/{draw_mask.sum()})")
    
    # === Build stacking ensemble via out-of-fold meta-features ===
    print("\n🤖 Building stacking ensemble (5-fold out-of-fold meta-features)...")
    skf_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_train = np.zeros((len(X_train), 9))  # 3 models × 3 classes

    for fold_idx, (tr_idx, val_idx) in enumerate(skf_stack.split(X_train, y_train)):
        print(f"   Fold {fold_idx + 1}/5...")
        X_ftr  = X_train.iloc[tr_idx];  X_fval  = X_train.iloc[val_idx]
        y_ftr  = y_train.iloc[tr_idx]
        fold_sw = create_sample_weights(y_ftr, class_weights)

        fold_scaler = StandardScaler()
        X_ftr_sc  = fold_scaler.fit_transform(X_ftr)
        X_fval_sc = fold_scaler.transform(X_fval)

        rf_f = RandomForestClassifier(**rf.get_params())
        rf_f.fit(X_ftr, y_ftr)
        meta_train[val_idx, :3] = rf_f.predict_proba(X_fval)

        xgb_f = XGBClassifier(**xgb.get_params())
        xgb_f.fit(X_ftr, y_ftr, sample_weight=fold_sw)
        meta_train[val_idx, 3:6] = xgb_f.predict_proba(X_fval)

        nn_f = MLPClassifier(**nn.get_params())
        nn_f.fit(X_ftr_sc, y_ftr)
        meta_train[val_idx, 6:9] = nn_f.predict_proba(X_fval_sc)

    print("   Training meta-learner (Logistic Regression)...")
    meta_learner = LogisticRegression(
        C=1.0, max_iter=1000, random_state=42,
        multi_class="multinomial", solver="lbfgs"
    )
    meta_learner.fit(meta_train, y_train)

    print("   Meta-learner learned weights:")
    for i, class_name in enumerate(["Home Win", "Draw", "Away Win"]):
        rf_w  = meta_learner.coef_[i][:3].mean()
        xgb_w = meta_learner.coef_[i][3:6].mean()
        nn_w  = meta_learner.coef_[i][6:9].mean()
        print(f"     {class_name}: RF={rf_w:+.2f}  XGB={xgb_w:+.2f}  NN={nn_w:+.2f}")

    ensemble = StackingEnsemble(rf, xgb, nn, scaler, meta_learner)

    # === Evaluate stacking ensemble ===
    print("\n📈 Evaluating stacking ensemble performance...")
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

    # Optimize confidence threshold on test set
    print("\n🎯 Optimizing confidence threshold...")
    best_threshold, best_threshold_f1 = 0.60, 0.0
    for threshold in np.arange(0.40, 0.80, 0.025):
        mask = y_proba.max(axis=1) >= threshold
        if mask.sum() < 10:
            continue
        y_true_t = y_test.values[mask]
        y_pred_t = y_proba[mask].argmax(axis=1)
        draw_f1 = f1_score(y_true_t, y_pred_t, labels=[1], average="macro", zero_division=0)
        if draw_f1 > best_threshold_f1:
            best_threshold_f1 = draw_f1
            best_threshold = float(threshold)
    print(f"   Optimal threshold: {best_threshold:.2f} (draw F1: {best_threshold_f1:.3f})")

    # Save models and metadata
    print("\n💾 Saving models and metadata...")
    os.makedirs("models", exist_ok=True)
    
    # Save stacking ensemble (wraps all base models + meta-learner)
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    joblib.dump(meta_learner, "models/meta_learner.pkl")
    print("✅ Stacking ensemble saved")
    
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
        "test_samples": len(X_test),
        "confidence_threshold": best_threshold,
        "feature_medians": feature_medians,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "tuned_rf_params": best_rf_params,
        "tuned_xgb_params": {k: v for k, v in best_xgb_params.items() if k != "class_weight"},
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
    print("🎉 TRAINING COMPLETE - STACKING ENSEMBLE SUMMARY")
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