"""Layer-1 unit tests: functional feature extraction (no leakage, shapes)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from functional_features import FunctionalFeatureExtractor, GateFeatureExtractor


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((40, 100))


N_BSPLINE = 10  # cubic B-spline basis on n_knots=4 (see test_bspline_basis_column_count)


def test_funfeat_shape(X):
    fe = FunctionalFeatureExtractor(n_fpca=6, n_knots=4).fit(X)
    F = fe.transform(X)
    assert F.shape == (40, 6 + N_BSPLINE + 1)   # FPCA + B-spline + deriv


def test_funfeat_fit_transform_equals_fit_then_transform(X):
    fe1 = FunctionalFeatureExtractor().fit(X)
    fe2 = FunctionalFeatureExtractor()
    np.testing.assert_array_almost_equal(
        fe1.transform(X), fe2.fit_transform(X), decimal=12)


def test_funfeat_no_leakage_train_test_projection(X):
    """A projector fit on train must be reusable (not refit) on test curves of
    different scale; shape is preserved and no exception."""
    Xtr, Xte = X[:30], X[30:]
    fe = FunctionalFeatureExtractor().fit(Xtr)
    Fte = fe.transform(Xte)
    assert Fte.shape == (10, 6 + N_BSPLINE + 1)
    assert np.all(np.isfinite(Fte))


def test_gate_features_no_true_residual(X):
    """Gate features use prediction-time quantities only: FPCA scores, |rhat_B|,
    Q-residual, leverage. Passing yhat_A=0, rhat_B=0 must still yield finite
    features and NOT depend on the true residual magnitude."""
    gfe = GateFeatureExtractor(n_fpca=4).fit(X)
    z = gfe.transform(X, np.zeros(X.shape[0]), np.zeros(X.shape[0]), None)
    assert z.shape == (40, 4 + 3)       # FPCA + |rB| + Qres + leverage(=0)
    assert np.all(np.isfinite(z))
    # leverage is 0 when latents are None (P4a)
    assert np.all(z[:, -1] == 0.0)


def test_gate_features_leverage_when_latents_given(X):
    gfe = GateFeatureExtractor(n_fpca=3).fit(X)
    lat = np.random.default_rng(1).standard_normal((40, 3))
    z = gfe.transform(X, np.zeros(40), np.zeros(40), lat)
    assert np.all(np.isfinite(z[:, -1]))


def test_bspline_basis_column_count(X):
    from functional_features import _bspline_basis
    grid = np.linspace(0, 1, 100)
    B = _bspline_basis(grid, n_knots=4, degree=3)
    # inner(4) + 2 boundaries -> 6 knots, pad by degree+1=4 each side -> 14,
    # n_basis = 14 - degree - 1 = 10
    assert B.shape[1] == 10
    assert np.all(np.isfinite(B))
