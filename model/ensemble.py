"""
Shared StackingEnsemble class.

Defined here (not in train_model.py) so that joblib/pickle can resolve the
class when loading the saved model from any module (predict, backtest, etc.).
"""

import numpy as np


class StackingEnsemble:
    """
    Stacking ensemble: RF + XGB + NN base models feed into a logistic
    regression meta-learner. Drop-in replacement for VotingClassifier —
    exposes the same predict() / predict_proba() interface.

    Optionally holds per-class isotonic calibrators (fitted in train_model.py
    on a held-out validation set) to correct the systematic draw over-prediction
    caused by balanced class weights.
    """

    def __init__(self, rf, xgb, nn, scaler, meta_learner):
        self.rf = rf
        self.xgb = xgb
        self.nn = nn
        self.scaler = scaler
        self.meta_learner = meta_learner
        self.calibrators = None  # set after training via attach_calibrators()

    def attach_calibrators(self, calibrators):
        """Store a list of 3 fitted IsotonicRegression objects (one per class)."""
        self.calibrators = calibrators

    def _meta_features(self, X):
        X_scaled = self.scaler.transform(X)
        return np.hstack([
            self.rf.predict_proba(X),
            self.xgb.predict_proba(X),
            self.nn.predict_proba(X_scaled),
        ])

    def predict_proba(self, X):
        raw = self.meta_learner.predict_proba(self._meta_features(X))
        if not self.calibrators:
            return raw
        # Apply per-class isotonic calibration then renormalize rows to sum to 1
        cal = np.column_stack([
            self.calibrators[c].predict(raw[:, c]) for c in range(raw.shape[1])
        ])
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
