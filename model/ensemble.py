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
