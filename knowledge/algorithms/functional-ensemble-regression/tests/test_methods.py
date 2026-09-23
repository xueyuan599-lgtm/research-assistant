"""Layer-1 unit tests: baseline methods M1/M2/M3 + contamination + SHAP."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from methods import (M1_FLM, M2_FunFeatGBDT, M3_SIMPLSEmsemble,
                     contaminate_Y, contaminate_X_level_shift)
from shap_analysis import shap_values, explain_features
from ahfe import make_gbdt
from data_loader import generate_s5


def _r2(pred, yt):
    return 1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)


@pytest.fixture
def data():
    X, y, info = generate_s5(n=160, nonlinear=True, seed=3)
    return X, y, info


def test_m1_flm_fit_predict(data):
    X, y, _ = data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = M1_FLM(alpha=1.0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert pred.shape == yte.shape
    assert np.all(np.isfinite(pred))


def test_m2_funfeat_gbdt_fit_predict(data):
    X, y, _ = data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = M2_FunFeatGBDT(gbdt="xgb", random_state=0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert pred.shape == yte.shape
    assert np.all(np.isfinite(pred))


def test_m3_ensemble_fit_predict(data):
    X, y, _ = data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = M3_SIMPLSEmsemble(A=4, n_boot=10, random_state=0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert pred.shape == yte.shape
    assert np.all(np.isfinite(pred))


def test_m3_seed_reproducible(data):
    X, y, _ = data
    p1 = M3_SIMPLSEmsemble(A=4, n_boot=10, random_state=7).fit(X, y).predict(X)
    p2 = M3_SIMPLSEmsemble(A=4, n_boot=10, random_state=7).fit(X, y).predict(X)
    np.testing.assert_array_almost_equal(p1, p2, decimal=10)


def test_contaminate_Y_frac():
    y = np.arange(100.0)
    yc = contaminate_Y(y, frac=0.10, seed=0)
    changed = np.sum(yc != y)
    assert changed == 10
    assert np.all(yc >= y - 1e-12)       # only upward shifts


def test_contaminate_X_level_shift():
    X = np.arange(20 * 5, dtype=float).reshape(20, 5)
    Xc = contaminate_X_level_shift(X, frac=0.10, seed=0)
    changed = np.sum(np.abs(Xc - X).sum(axis=1) > 1e-9)
    assert changed == 2


def test_shap_values_shape(data):
    X, y, _ = data
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    from functional_features import FunctionalFeatureExtractor
    fe = FunctionalFeatureExtractor().fit(Xtr)
    F = fe.transform(Xtr)
    model = make_gbdt("xgb", random_state=0).fit(F, ytr)
    imp = shap_values(model, F)
    assert imp.shape == (F.shape[1],)
    assert np.all(imp >= 0)


def test_explain_features_names(data):
    X, y, _ = data
    from functional_features import FunctionalFeatureExtractor
    fe = FunctionalFeatureExtractor().fit(X)
    F = fe.transform(X)
    names = explain_features(F)
    assert len(names) == F.shape[1]
    assert names[0].startswith("FPCA")
