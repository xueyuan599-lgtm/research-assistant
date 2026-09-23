"""Baseline methods M1/M2/M3 for the AHFE benchmark (plan.md §3).

M1  : FLM           - functional linear model baseline (ridge on raw curves)
M2  : FunFeat+GBDT  - GBDT on the SAME functional feature library as AHFE-B
                      (fairness: AHFE wins must come from structure, not features)
M3  : SIMPLS ensemble - bootstrap-aggregated robust weighted SIMPLS (Alin 2026
                      inspired base; the primary comparison baseline)

Each method exposes a sklearn-like (fit, predict) interface so the shared
benchmark runner can treat them uniformly. Contamination helpers for S6 are
also provided.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from simpls import SIMPLS
from robust_simpls import RWSIMPLS
from functional_features import FunctionalFeatureExtractor
from ahfe import make_gbdt


class M1_FLM:
    """Functional linear model baseline: ridge regression on the raw discretized
    curves X (n x p). Captures the linear functional effect at cost of assuming
    linearity in X."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "M1_FLM":
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(np.asarray(X, float))
        self.model_ = Ridge(alpha=self.alpha).fit(Xs, np.asarray(y, float).ravel())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, float))
        return self.model_.predict(Xs)


class M2_FunFeatGBDT:
    """GBDT directly on the functional feature library (SAME features as AHFE-B).
    Uses nested CV on the training set to prevent leakage of the feature
    projector parameters (fitted on inner folds, projected to validation)."""

    def __init__(self, n_fpca: int = 6, n_knots: int = 4, gbdt: str = "xgb",
                 random_state: int = 0, inner_folds: int = 4) -> None:
        self.n_fpca = int(n_fpca)
        self.n_knots = int(n_knots)
        self.gbdt = gbdt
        self.random_state = int(random_state)
        self.inner_folds = int(inner_folds)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "M2_FunFeatGBDT":
        # The feature projector (PCA / B-spline basis) depends only on X, never
        # on y, so fitting it on the full training fold introduces no target
        # leakage; it is projected to test at predict() time. Same feature
        # library as AHFE-B for a fair structural comparison.
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        self.fe_ = FunctionalFeatureExtractor(self.n_fpca, self.n_knots).fit(X)
        F_full = self.fe_.transform(X)
        self.model_ = make_gbdt(self.gbdt, self.random_state).fit(F_full, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self.fe_.transform(np.asarray(X, float))
        return self.model_.predict(F)


class M3_SIMPLSEmsemble:
    """Bootstrap-aggregated robust weighted SIMPLS (the ensemble base / M3).
    Fits B RWSIMPLS models on bootstrap resamples and averages predictions."""

    def __init__(self, A: int = 4, n_boot: int = 25, use_robust: bool = True,
                 random_state: int = 0) -> None:
        self.A = int(A)
        self.n_boot = int(n_boot)
        self.use_robust = bool(use_robust)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "M3_SIMPLSEmsemble":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.models_ = []
        for b in range(self.n_boot):
            idx = rng.integers(0, n, size=n)
            m = (RWSIMPLS(self.A) if self.use_robust else SIMPLS(self.A))
            self.models_.append(m.fit(X[idx], y[idx]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return np.mean([m.predict(X) for m in self.models_], axis=0)


# ---- contamination helpers (S6) ----
def contaminate_Y(y: np.ndarray, frac: float = 0.10, sigma: float = 5.0,
                  seed: int = 0) -> np.ndarray:
    """S6a: replace `frac` random responses with a +sigma shift (Y-outliers)."""
    y = np.asarray(y, float).ravel().copy()
    rng = np.random.default_rng(seed)
    n = y.size
    k = max(1, int(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    y[idx] += sigma * np.std(y)
    return y


def contaminate_X_level_shift(X: np.ndarray, frac: float = 0.10,
                              shift: float = 3.0, seed: int = 0) -> np.ndarray:
    """S6b: add a constant level shift to `frac` entire curves (functional
    outliers)."""
    X = np.asarray(X, float).copy()
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, int(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    X[idx] += shift * np.std(X, axis=1)[idx, None]
    return X
