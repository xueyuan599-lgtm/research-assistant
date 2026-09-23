"""SHAP explanation of the GBDT residual corrector (AHFE component B).

Q: which functional features drive the nonlinear residual correction r_hat_B?
This post-hoc explanation complements the linear, interpretable beta(t) of the
base SIMPLS component: SHAP on the tree ensemble surfaces the nonlinear
structure the linear latent space misses.

Only xgboost / lightgbm / catboost are supported (TreeExplainer). A reduce
backend renders an aggregate importance (mean |SHAP|) when shap plotting is
unavailable or for headless summary output.
"""

from __future__ import annotations

import numpy as np


def shap_values(model, F: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP (feature importance) of the GBDT corrector on features F.

    Parameters
    ----------
    model : fitted GBDT regressor (xgb/lgbm/cat)
    F : (n, d) functional feature matrix (the SAME library as AHFE-B)

    Returns (d,) mean |SHAP| per feature, order matching F columns.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(F)
    if isinstance(sv, list):               # older xgboost returns list
        sv = np.mean(sv, axis=0)
    sv = np.asarray(sv, float)
    return np.mean(np.abs(sv), axis=0)


def explain_features(F: np.ndarray) -> list[str]:
    """Column names for the functional feature matrix (matches
    FunctionalFeatureExtractor output order)."""
    d = F.shape[1]
    n_fpca = 6
    names = [f"FPCA{i + 1}" for i in range(n_fpca)]
    names += [f"Bspline{i + 1}" for i in range(d - n_fpca - 1)]
    names += ["Deriv"]
    return names[:d]
