import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from model.ensemble import StackingEnsemble  # shared module so pickle can resolve it

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
    "home_venue_draw_rate",       # venue-specific draw rate (Issue 4)
    "away_venue_draw_rate",       # venue-specific draw rate (Issue 4)
    "current_season_draw_rate",   # in-season league draw rate (Issue 8)
    "home_total_goals_avg",
    "away_total_goals_avg",
    "league_avg_goals",
    "league_home_adv",
    "home_momentum",
    "away_momentum",
]

ODDS_FEATURES = [
    "has_odds",                   # binary indicator (Issue 5)
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

# Metadata columns — present in CSV but NOT used as model features
METADATA_COLS = {"home_team", "away_team", "match_date", "league", "result",
                 "raw_home_odds", "raw_draw_odds", "raw_away_odds"}

RESULT_MAP = {"A": "away_win", "H": "home_win", "D": "draw"}
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}


def _load_training_data(path: str) -> tuple:
    """Load and validate training data, auto-detecting available features."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}. Run prepare_training_data.py first.")

    df = pd.read_csv(path)
    print(f"📊 Loaded {len(df)} rows from {path}")

    # Fill NaN in odds features with 0 BEFORE dropna.
    # has_odds=0 rows are valid training examples — median imputation was masking
    # the difference between 'no odds' and 'odds predict 33% each'.
    odds_fill_cols = [f for f in ODDS_FEATURES if f in df.columns and f != "has_odds"]
    if odds_fill_cols:
        nan_count = df[odds_fill_cols].isna().sum().sum()
        if nan_count > 0:
            df[odds_fill_cols] = df[odds_fill_cols].fillna(0.0)
            print(f"📊 Filled {nan_count} NaN odds values with 0.0 (has_odds column flags these rows)")

    # Auto-detect which features are available
    all_feature_lists = {
        "Core": CORE_FEATURES,
        "Draw-focused": DRAW_FEATURES,
        "Odds-based": ODDS_FEATURES,
        "Interactions": INTERACTION_FEATURES,
    }
    available_features = []
    print("\n📋 Feature availability check:")
    for group_name, feat_list in all_feature_lists.items():
        available_in_group = [f for f in feat_list if f in df.columns]
        available_features.extend(available_in_group)
        print(f"   {group_name}: {len(available_in_group)}/{len(feat_list)} features")
        if len(available_in_group) < len(feat_list):
            missing = [f for f in feat_list if f not in df.columns]
            print(f"      Missing: {', '.join(missing[:3])}...")

    if len([f for f in CORE_FEATURES if f in available_features]) < len(CORE_FEATURES):
        raise ValueError(f"Missing core features! Need at least: {CORE_FEATURES}")

    print(f"\n✅ Using {len(available_features)} total features")

    if "result" not in df.columns:
        raise ValueError("Training data missing 'result' column")

    df = df.dropna(subset=["result"])
    df = df.dropna(subset=available_features)

    print(f"📊 Rows after cleaning: {len(df)}")

    return df, available_features


def tune_hyperparameters(X_train, y_train, class_weights, n_trials=50):
    """
    Use Optuna to search RF, XGB, and MLP hyperparameters, optimising draw F1.
    Returns (best_rf_params, best_xgb_params, best_mlp_params).
    If Optuna is not installed or n_trials=0, returns empty dicts so callers
    fall back to hardcoded defaults.
    """
    if n_trials == 0:
        return {}, {}, {}

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("⚠️  optuna not installed — skipping tuning (pip install optuna to enable)")
        return {}, {}, {}

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

    # --- Tune MLP (Issue 9) ---
    # Fit a temporary scaler for MLP tuning
    scaler_tmp = StandardScaler()
    X_train_sc = scaler_tmp.fit_transform(X_train)

    def mlp_objective(trial):
        hidden = trial.suggest_categorical("mlp_hidden", [(64, 32), (128, 64), (64, 32, 16)])
        alpha = trial.suggest_float("mlp_alpha", 1e-4, 0.1, log=True)
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            activation="relu",
            learning_rate="adaptive",
            learning_rate_init=5e-4,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        scores = []
        for tr_idx, val_idx in cv.split(X_train_sc, y_train):
            model.fit(X_train_sc[tr_idx], y_train.iloc[tr_idx])
            y_pred = model.predict(X_train_sc[val_idx])
            scores.append(f1_score(y_train.iloc[val_idx], y_pred, labels=[1], average="macro", zero_division=0))
        return float(np.mean(scores))

    print(f"   Tuning MLP ({n_trials} trials)...")
    mlp_study = optuna.create_study(direction="maximize")
    mlp_study.optimize(mlp_objective, n_trials=n_trials, show_progress_bar=False)
    best_mlp_params = mlp_study.best_params
    print(f"   MLP best draw F1: {mlp_study.best_value:.3f}")

    return best_rf_params, best_xgb_params, best_mlp_params


def calculate_class_weights(y):
    """Calculate balanced class weights (no manual amplification — meta-learner handles draw boost)."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def create_sample_weights(y, class_weights):
    """Convert class weights to sample weights for XGBoost."""
    return np.array([class_weights[int(val)] for val in y])


def train_pipeline(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 0) -> StackingEnsemble:
    """
    Fit the full stacking ensemble (RF + XGB + MLP → LogReg meta-learner).

    Parameters
    ----------
    X_train : pd.DataFrame  — feature matrix
    y_train : pd.Series     — integer-encoded labels (0=home, 1=draw, 2=away)
    n_trials : int          — Optuna trials per model (0 = skip tuning, use defaults)

    Returns
    -------
    StackingEnsemble ready for predict() / predict_proba()

    Used by both train_model() (n_trials=50) and backtest.py (n_trials=0).
    """
    class_weights = calculate_class_weights(y_train)
    sample_weights = create_sample_weights(y_train, class_weights)

    # --- Hyperparameter tuning ---
    if n_trials > 0:
        print(f"\n🔍 Hyperparameter tuning with Optuna ({n_trials} trials each)...")
        best_rf_params, best_xgb_params, best_mlp_params = tune_hyperparameters(
            X_train, y_train, class_weights, n_trials=n_trials
        )
    else:
        best_rf_params, best_xgb_params, best_mlp_params = {}, {}, {}

    # --- Define base models ---
    rf_defaults = dict(
        n_estimators=350, random_state=42, n_jobs=-1,
        max_depth=12, min_samples_split=15, min_samples_leaf=7,
        class_weight=class_weights, max_features="sqrt",
    )
    rf = RandomForestClassifier(**(best_rf_params if best_rf_params else rf_defaults))

    xgb_defaults = dict(
        n_estimators=500, learning_rate=0.02, max_depth=5,
        subsample=0.75, colsample_bytree=0.75, reg_lambda=3.0,
        reg_alpha=2.0, min_child_weight=5, gamma=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", random_state=42, verbosity=0,
    )
    xgb = XGBClassifier(**(best_xgb_params if best_xgb_params else xgb_defaults))

    # MLP: reduced to (64, 32) as default; Optuna may choose a different architecture
    mlp_hidden = best_mlp_params.get("mlp_hidden", (64, 32)) if best_mlp_params else (64, 32)
    mlp_alpha = best_mlp_params.get("mlp_alpha", 1e-3) if best_mlp_params else 1e-3
    nn = MLPClassifier(
        hidden_layer_sizes=mlp_hidden,
        activation="relu",
        alpha=mlp_alpha,
        learning_rate="adaptive",
        learning_rate_init=5e-4,
        max_iter=1500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-4,
    )

    # Scaler for NN (fitted on full training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- OOF stacking to build meta-training features ---
    skf_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_train = np.zeros((len(X_train), 9))  # 3 models × 3 classes

    for fold_idx, (tr_idx, val_idx) in enumerate(skf_stack.split(X_train, y_train)):
        X_ftr = X_train.iloc[tr_idx];  X_fval = X_train.iloc[val_idx]
        y_ftr = y_train.iloc[tr_idx]
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

    # --- Train final base models on full training data ---
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    nn.fit(X_train_scaled, y_train)

    # --- Train meta-learner ---
    meta_learner = LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, solver="lbfgs",
        class_weight="balanced"
    )
    meta_learner.fit(meta_train, y_train)

    return StackingEnsemble(rf, xgb, nn, scaler, meta_learner)


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

    has_draw_features = any(f in features for f in DRAW_FEATURES)
    has_odds_features = any(f in features for f in ODDS_FEATURES)

    print(f"\n🎯 Feature set:")
    print(f"   ✅ Core features: Yes")
    print(f"   {'✅' if has_draw_features else '❌'} Draw-focused features: {'Yes' if has_draw_features else 'No'}")
    print(f"   {'✅' if has_odds_features else '❌'} Odds-based features: {'Yes' if has_odds_features else 'No'}")

    # --- Prepare features and labels ---
    X = df[features]
    y_raw = df["result"].map(RESULT_MAP)

    if y_raw.isna().any():
        bad_results = df[y_raw.isna()]["result"].unique()
        raise ValueError(f"Unmappable result values found: {bad_results}")

    y_encoded = y_raw.map(LABEL_MAP)

    # --- Chronological train/test split (Issue 1) ---
    # Sort by match_date so the last 20% in time is the test set.
    print("\n🔄 Chronological split (80% train, 20% test by match date)...")
    if "match_date" in df.columns:
        sort_idx = df["match_date"].argsort().values
        X = X.iloc[sort_idx].reset_index(drop=True)
        y_encoded = y_encoded.iloc[sort_idx].reset_index(drop=True)
    else:
        print("   ⚠️  No match_date column — falling back to row order")

    split_idx = int(len(X) * 0.8)
    X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_full, y_test = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]

    print(f"📊 Training (full): {len(X_train_full)} samples")
    print(f"📊 Test set:        {len(X_test)} samples")

    # --- Carve validation set from training data for threshold tuning (Issue 6) ---
    val_split_idx = int(len(X_train_full) * 0.85)
    X_train = X_train_full.iloc[:val_split_idx]
    y_train = y_train_full.iloc[:val_split_idx]
    X_val   = X_train_full.iloc[val_split_idx:]
    y_val   = y_train_full.iloc[val_split_idx:]

    print(f"📊 Train (core):    {len(X_train)} samples")
    print(f"📊 Validation set:  {len(X_val)} samples  (for threshold tuning only)")
    print(f"📊 Test set:        {len(X_test)} samples  (held out until final eval)")

    # --- Train full stacking ensemble ---
    print("\n🤖 Training stacking ensemble via train_pipeline()...")
    ensemble = train_pipeline(X_train, y_train, n_trials=50)

    # --- Show per-model metrics on test set ---
    print("\n📊 Individual model accuracies (hold-out test):")
    X_test_scaled = ensemble.scaler.transform(X_test)
    for name, model, X_eval in [
        ("Random Forest", ensemble.rf, X_test),
        ("XGBoost",       ensemble.xgb, X_test),
        ("Neural Network", ensemble.nn, X_test_scaled),
    ]:
        preds = model.predict(X_eval)
        score = (preds == y_test).mean()
        draw_mask = y_test == 1
        draw_rec = ((preds == 1) & draw_mask).sum() / draw_mask.sum() if draw_mask.sum() else 0.0
        print(f"   {name}: accuracy={score:.2%}  draw_recall={draw_rec:.1%}")

    # Show meta-learner weights
    meta_learner = ensemble.meta_learner
    print("\n   Meta-learner learned weights:")
    for i, class_name in enumerate(["Home Win", "Draw", "Away Win"]):
        rf_w  = meta_learner.coef_[i][:3].mean()
        xgb_w = meta_learner.coef_[i][3:6].mean()
        nn_w  = meta_learner.coef_[i][6:9].mean()
        print(f"     {class_name}: RF={rf_w:+.2f}  XGB={xgb_w:+.2f}  NN={nn_w:+.2f}")

    # --- K-fold CV on X_train (not X_test!) ---
    print(f"\n🔄 5-fold cross-validation (Random Forest, on training data)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        RandomForestClassifier(**ensemble.rf.get_params()), X_train, y_train,
        cv=skf, scoring="accuracy", n_jobs=-1
    )
    print(f"   CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # --- Evaluate stacking ensemble on test set (no threshold) ---
    print("\n📈 Evaluating stacking ensemble on hold-out test set...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)

    accuracy = (y_pred == y_test).mean()
    print(f"\n🎯 Overall Accuracy (raw, no threshold): {accuracy:.2%}")

    report = classification_report(
        y_test, y_pred,
        target_names=["Home Win", "Draw", "Away Win"],
        digits=3,
        output_dict=True
    )

    print(f"\n📊 Classification Report:")
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

    print("\n🧮 Confusion Matrix:")
    print("    Predicted: Home | Draw | Away")
    print("    " + "-" * 30)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Home", "Draw", "Away"]
    for i, row in enumerate(cm):
        print(f"Actual {labels[i]:5}: {row[0]:5} | {row[1]:5} | {row[2]:5}")

    draw_recall = report["Draw"]["recall"]
    draw_precision = report["Draw"]["precision"]
    draw_f1 = report["Draw"]["f1-score"]

    print(f"\n✨ DRAW PREDICTION PERFORMANCE (before threshold):")
    print(f"   Recall: {draw_recall:.1%}")
    print(f"   Precision: {draw_precision:.1%}")
    print(f"   F1-Score: {draw_f1:.3f}")

    # --- Optimize confidence threshold on VALIDATION SET (Issue 6) ---
    print("\n🎯 Optimizing confidence threshold on validation set (not test set)...")
    y_proba_val = ensemble.predict_proba(X_val)
    best_threshold, best_threshold_f1 = 0.60, 0.0
    for threshold in np.arange(0.40, 0.80, 0.025):
        mask = y_proba_val.max(axis=1) >= threshold
        if mask.sum() < 10:
            continue
        y_true_t = y_val.values[mask]
        y_pred_t = y_proba_val[mask].argmax(axis=1)
        t_f1 = f1_score(y_true_t, y_pred_t, labels=[1], average="macro", zero_division=0)
        if t_f1 > best_threshold_f1:
            best_threshold_f1 = t_f1
            best_threshold = float(threshold)
    print(f"   Optimal threshold: {best_threshold:.2f} (val draw F1: {best_threshold_f1:.3f})")

    # Report test set metrics with optimal threshold applied
    thresh_mask = y_proba.max(axis=1) >= best_threshold
    if thresh_mask.sum() >= 10:
        y_pred_thresh = y_proba[thresh_mask].argmax(axis=1)
        y_test_thresh = y_test.values[thresh_mask]
        accuracy_thresh = (y_pred_thresh == y_test_thresh).mean()
        report_thresh = classification_report(
            y_test_thresh, y_pred_thresh,
            target_names=["Home Win", "Draw", "Away Win"],
            digits=3, output_dict=True
        )
        print(f"\n📊 Test set WITH threshold ≥ {best_threshold:.2f} ({thresh_mask.sum()} / {len(y_test)} matches):")
        print(f"   Accuracy:       {accuracy_thresh:.2%}")
        print(f"   Draw recall:    {report_thresh['Draw']['recall']:.1%}")
        print(f"   Draw precision: {report_thresh['Draw']['precision']:.1%}")
        print(f"   Draw F1:        {report_thresh['Draw']['f1-score']:.3f}")

    # --- Save models and metadata ---
    print("\n💾 Saving models and metadata...")
    os.makedirs("models", exist_ok=True)

    joblib.dump(ensemble, "models/ensemble_model.pkl")
    joblib.dump(ensemble.meta_learner, "models/meta_learner.pkl")
    print("✅ Stacking ensemble saved")

    joblib.dump(ensemble.scaler, "models/scaler.pkl")
    print("✅ Feature scaler saved")

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
        "feature_medians": {},  # no longer used — kept for API compatibility
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
    }

    joblib.dump(metadata, "models/metadata.pkl")

    with open("models/features.txt", "w") as f:
        f.write("\n".join(features))
    print("✅ Feature list and metadata saved")

    joblib.dump(ensemble.rf, "models/rf_model.pkl")
    joblib.dump(ensemble.xgb, "models/xgb_model.pkl")
    joblib.dump(ensemble.nn, "models/mlp_model.pkl")
    print("✅ Individual models saved")

    # Feature importance
    print("\n📊 Top 15 Feature Importance (from Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': ensemble.rf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    max_importance = feature_importance['importance'].max()
    for _, row in feature_importance.iterrows():
        bar_length = int((row['importance'] / max_importance) * 30)
        bar = '█' * bar_length
        if row['feature'] in DRAW_FEATURES:
            marker = "🎯"
        elif row['feature'] in ODDS_FEATURES:
            marker = "💰"
        else:
            marker = "📊"
        print(f"  {marker} {row['feature']:30} {row['importance']:.4f} {bar}")

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE - STACKING ENSEMBLE SUMMARY")
    print("=" * 60)
    print(f"   Total samples: {len(df):,}")
    print(f"   Features used: {len(features)}")
    print(f"   Overall accuracy: {accuracy:.2%}")
    print(f"   Draw recall: {draw_recall:.1%}")
    print(f"   Draw precision: {draw_precision:.1%}")
    print(f"   Confidence threshold: {best_threshold:.2f} (tuned on validation set)")
    print(f"\n   Models saved in: models/")
    print(f"   Ready for predictions!")


if __name__ == "__main__":
    train_model()
