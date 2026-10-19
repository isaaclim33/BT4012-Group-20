"""Model zoo. Every model exposes fit(X, y) and score(X) -> higher = riskier.

All fitted transforms (log scaling, standardisation, rule thresholds) live inside
the model object, so they are learned from the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import SEED


class SingleSignal:
    """E0 baseline: rank by one column (e.g. total reimbursement). Nothing is fitted."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y):
        return self

    def score(self, X):
        return X[self.column].to_numpy(dtype=float)


class ExpertRules:
    """Rule-based screen of the kind a payer runs without ML (lecture 01).

    Each rule fires when a provider is above the training-fold 90th percentile of a
    red-flag measure; the score is the number of rules fired, with total
    reimbursement as tie-breaker so the queue has a defined order.
    """
    RULES = ["total_reimb", "reimb_per_claim", "los_mean", "claims_per_bene",
             "diag_per_claim", "ip_proc_per_claim"]

    def fit(self, X, y):
        self.cut_ = {c: np.quantile(X[c], 0.90) for c in self.RULES}
        return self

    def score(self, X):
        fired = sum((X[c] > t).astype(int) for c, t in self.cut_.items())
        tiebreak = X["total_reimb"].rank(pct=True) * 0.5
        return (fired + tiebreak).to_numpy(dtype=float)


class SklearnScorer:
    def __init__(self, est, sampler=None):
        self.est, self.sampler = est, sampler

    def fit(self, X, y):
        Xf, yf = X, y
        if self.sampler is not None:      # resampling happens inside the training fold only
            Xf, yf = self.sampler.fit_resample(X, y)
        self.est.fit(Xf, yf)
        return self

    def score(self, X):
        if hasattr(self.est, "predict_proba"):
            return self.est.predict_proba(X)[:, 1]
        return -self.est.score_samples(X)  # isolation forest: lower = more anomalous


def _log1p(X):
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


def logistic(C: float = 0.1, balanced: bool = True, sampler=None):
    est = make_pipeline(FunctionTransformer(_log1p), StandardScaler(),
                        LogisticRegression(C=C, class_weight="balanced" if balanced else None,
                                           max_iter=5000))
    return SklearnScorer(est, sampler)


def random_forest(min_samples_leaf: int = 3, max_features="sqrt", balanced: bool = True, sampler=None):
    est = RandomForestClassifier(n_estimators=500, min_samples_leaf=min_samples_leaf,
                                 max_features=max_features, n_jobs=-1, random_state=SEED,
                                 class_weight="balanced_subsample" if balanced else None)
    return SklearnScorer(est, sampler)


LGBM_DEFAULT = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0)


def lightgbm(params: dict | None = None, balanced: bool = True, sampler=None):
    p = {**LGBM_DEFAULT, **(params or {})}
    est = LGBMClassifier(**p, objective="binary", random_state=SEED, n_jobs=2, verbose=-1,
                         class_weight="balanced" if balanced else None)
    return SklearnScorer(est, sampler)


def isolation_forest():
    est = make_pipeline(FunctionTransformer(_log1p), StandardScaler(),
                        IsolationForest(n_estimators=500, random_state=SEED, n_jobs=-1))
    return SklearnScorer(est)


def lgbm_contributions(model: SklearnScorer, X: pd.DataFrame) -> pd.DataFrame:
    """Per-provider SHAP values from LightGBM's built-in TreeSHAP (pred_contrib)."""
    contrib = model.est.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns)
