"""AHFE — Adaptive Hybrid Functional Ensemble (gated residual correction).

yhat_AHFE(x) = yhat_A(x) + w(x) * rhat_B(x),  w(x) = sigmoid(alpha + z^T theta)

Components:
  A : linear robust SIMPLS (RWSIMPLS) base -> yhat_A, smooth interpretable beta
  B : residual GBDT (XGBoost/LightGBM/CatBoost) on functional features -> rhat_B
  g : sigmoid-link gate on prediction-time features z -> w(x) in [0,1]

Training (cross-fitting, plan.md §3.1 #2): inner K-fold on the training fold
generates out-of-fold (yhat_A, rhat_B, z); the gate is trained on those OOF
predictions so it never sees the outer-test target. A and B are then refit on
the full training fold; prediction is one pass on the test fold.

Distinct from Functional Mixtures-of-Experts: one linear base + one residual
corrector whose gating modulates correction *strength*, not multi-expert
probability mixing.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from simpls import SIMPLS
from robust_simpls import RWSIMPLS
from functional_features import FunctionalFeatureExtractor, GateFeatureExtractor
from gating import SigmoidGate


def make_gbdt(name: str = "xgb", random_state: int = 0):
    """Return a fresh GBDT regressor with a fixed hyperparameter space."""
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=random_state, verbosity=0)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=random_state, verbose=-1)
    if name == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=250, depth=4, learning_rate=0.05,
            random_seed=random_state, verbose=False, allow_writing_files=False)
    raise ValueError(name)


class AHFE:
    """Adaptive hybrid functional ensemble with cross-fitted gate.

    Parameters
    ----------
    A : int
        SIMPLS latent components.
    gbdt : str
        'xgb' | 'lgbm' | 'cat'.
    n_fpca, n_knots : int
        Functional feature dimensions.
    lambda_g : float
        Gate ridge regularization.
    inner_folds : int
        Inner cross-fitting folds for OOF gate training.
    use_robust : bool
        Use RWSIMPLS (True) or plain SIMPLS (False) as component A.
    random_state : int
        Seed for folds, GBDT, RWSIMPLS.
    """

    def __init__(self, A: int = 4, gbdt: str = "xgb", n_fpca: int = 6,
                 n_knots: int = 4, lambda_g: float = 1e-4, inner_folds: int = 4,
                 use_robust: bool = True, random_state: int = 0,
                 gate_mode: str = "adaptive") -> None:
        self.A = int(A)
        self.gbdt = gbdt
        self.n_fpca = int(n_fpca)
        self.n_knots = int(n_knots)
        self.lambda_g = float(lambda_g)
        self.inner_folds = int(inner_folds)
        self.use_robust = bool(use_robust)
        self.random_state = int(random_state)
        self.gate_mode = gate_mode  # 'adaptive' | 'fixed' (constant gamma ablation)

    def _fit_AB(self, X: np.ndarray, y: np.ndarray):
        """Fit linear component A and residual corrector B on a training set."""
        a_model = (RWSIMPLS(self.A) if self.use_robust else SIMPLS(self.A))
        a_model = a_model.fit(X, y)
        yhat_A = a_model.predict(X)
        feats = FunctionalFeatureExtractor(self.n_fpca, self.n_knots)
        F = feats.fit_transform(X)
        r = y - yhat_A
        b_model = make_gbdt(self.gbdt, self.random_state)
        b_model.fit(F, r)
        return a_model, b_model, feats

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AHFE":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n = X.shape[0]
        if n < self.inner_folds + 2:
            self.inner_folds = max(2, n // 2)
        inner = KFold(n_splits=self.inner_folds, shuffle=True,
                      random_state=self.random_state)
        # inner cross-fitting: collect OOF base predictions to train the gate
        z_oof, y_oof, ya_oof, rb_oof = [], [], [], []
        for tr, va in inner.split(X):
            a_m, b_m, fe = self._fit_AB(X[tr], y[tr])
            F_va = fe.transform(X[va])
            ya_va = a_m.predict(X[va])
            rb_va = b_m.predict(F_va)
            gfe = GateFeatureExtractor(self.n_fpca).fit(X[tr])
            z_va = gfe.transform(X[va], ya_va, rb_va, None)
            z_oof.append(z_va)
            y_oof.append(y[va])
            ya_oof.append(ya_va)
            rb_oof.append(rb_va)
        z_oof = np.vstack(z_oof)
        y_oof = np.concatenate(y_oof)
        ya_oof = np.concatenate(ya_oof)
        rb_oof = np.concatenate(rb_oof)
        if self.gate_mode == "adaptive":
            self.gate_ = SigmoidGate(self.lambda_g).fit(z_oof, y_oof, ya_oof, rb_oof)
            self.gamma_ = None
        else:  # fixed constant-gamma ablation (AHFE-fixed)
            eA = y_oof - ya_oof
            g = float(eA @ rb_oof / (rb_oof @ rb_oof + 1e-12))
            self.gamma_ = float(np.clip(g, 0.0, 1.0))
            self.gate_ = None
        # refit A, B, and gate-feature projector on the full training fold
        self.a_model_, self.b_model_, self.feats_ = self._fit_AB(X, y)
        self.gfe_ = GateFeatureExtractor(self.n_fpca).fit(X)
        return self

    def predict_with_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (yhat_A, rhat_B, w)."""
        X = np.asarray(X, float)
        yhat_A = self.a_model_.predict(X)
        F = self.feats_.transform(X)
        rhat_B = self.b_model_.predict(F)
        if self.gate_mode == "adaptive":
            z = self.gfe_.transform(X, yhat_A, rhat_B, None)
            w = self.gate_.predict(z)
        else:
            w = np.full(yhat_A.shape, self.gamma_)
        return yhat_A, rhat_B, w

    def predict(self, X: np.ndarray) -> np.ndarray:
        yhat_A, rhat_B, w = self.predict_with_components(X)
        return yhat_A + w * rhat_B
