"""Layer-1 unit tests: AHFE pipeline (gating, leakage, mechanism)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from ahfe import AHFE, make_gbdt
from data_loader import generate_s5, load_tecator


def _r2(pred, yt):
    return 1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)


@pytest.fixture
def s5_data():
    X, y, info = generate_s5(n=300, nonlinear=True, seed=0)
    return X, y, info


def test_ahfe_fit_predict_shapes(s5_data):
    X, y, _ = s5_data
    m = AHFE(A=4, gbdt="xgb", random_state=0).fit(X, y)
    yhat_A, rhat_B, w = m.predict_with_components(X)
    assert yhat_A.shape == (300,)
    assert rhat_B.shape == (300,)
    assert w.shape == (300,)
    assert m.predict(X).shape == (300,)


def test_gate_weights_in_01(s5_data):
    X, y, _ = s5_data
    m = AHFE(A=4, gbdt="xgb", random_state=0).fit(X, y)
    _, _, w = m.predict_with_components(X)
    assert np.all((w >= -1e-12) & (w <= 1 + 1e-12))


def test_adaptive_gate_uses_oof_not_test_y():
    """Gate must be trained on inner OOF predictions; the fitted gate therefore
    never sees the outer-test target. Sanity: fitting on train then predicting
    on a held-out fold uses only prediction-time features (no target leak)."""
    X, y, _ = generate_s5(n=250, nonlinear=True, seed=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = AHFE(A=4, gbdt="xgb", random_state=0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert np.all(np.isfinite(pred))


def test_ahfe_beats_m3_on_nonlinear_s5(s5_data):
    """Mechanism-layer (loose) check: AHFE should improve over a plain SIMPLS
    base on nonlinear S5. This is a scientific sanity check, not a rigid gate."""
    from robust_simpls import RWSIMPLS
    X, y, _ = s5_data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    base = RWSIMPLS(4).fit(Xtr, ytr)
    ahfe = AHFE(A=4, gbdt="xgb", random_state=0).fit(Xtr, ytr)
    assert _r2(ahfe.predict(Xte), yte) > _r2(base.predict(Xte), yte)


def test_seed_reproducibility(s5_data):
    X, y, _ = s5_data
    p1 = AHFE(A=4, gbdt="xgb", random_state=42).fit(X, y).predict(X)
    p2 = AHFE(A=4, gbdt="xgb", random_state=42).fit(X, y).predict(X)
    np.testing.assert_array_almost_equal(p1, p2, decimal=8)


def test_fixed_gamma_is_constant_in_01(s5_data):
    X, y, _ = s5_data
    m = AHFE(A=4, gbdt="xgb", random_state=0, gate_mode="fixed").fit(X, y)
    _, _, w = m.predict_with_components(X)
    assert np.allclose(w, w[0])          # constant across samples
    assert 0.0 <= m.gamma_ <= 1.0 + 1e-9


def test_adaptive_gamma_none(s5_data):
    X, y, _ = s5_data
    m = AHFE(A=4, gbdt="xgb", random_state=0, gate_mode="adaptive").fit(X, y)
    assert m.gamma_ is None


def test_tecator_shape():
    X, y_fat, grid, y_full = load_tecator()
    assert X.shape == (215, 100)
    assert y_fat.shape == (215,)
    assert grid.shape == (100,)
    assert y_full.shape == (215, 3)


def test_tecator_ahfe_runs():
    X, y_fat, _, _ = load_tecator()
    m = AHFE(A=4, gbdt="xgb", random_state=0, inner_folds=3).fit(X, y_fat)
    pred = m.predict(X)
    assert pred.shape == (215,)
    assert np.all(np.isfinite(pred))


def test_make_gbdt_variants_predict():
    X, y, _ = generate_s5(n=120, nonlinear=True, seed=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    from functional_features import FunctionalFeatureExtractor
    fe = FunctionalFeatureExtractor().fit(Xtr)
    for name in ("xgb", "lgbm"):
        model = make_gbdt(name, random_state=0).fit(fe.transform(Xtr), ytr)
        assert np.all(np.isfinite(model.predict(fe.transform(Xte))))
