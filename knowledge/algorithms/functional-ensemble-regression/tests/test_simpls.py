"""Layer-1 unit tests: SIMPLS / RWSIMPLS software correctness."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from simpls import SIMPLS
from robust_simpls import RWSIMPLS, _weighted_simpls, _tukey_bisquare


def _random_data(n=50, p=10, seed=0, A=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    return X, y


def _r2(pred, yt):
    return 1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)


def test_simpls_matches_sklearn_pls():
    """SIMPLS (de Jong) vs sklearn PLSRegression (NIPALS): predictions should
    agree closely on random data (RMSE < 1e-6 or corr > 0.999)."""
    X, y = _random_data(A=3)
    ours = SIMPLS(3).fit(X, y).predict(X)
    ref = PLSRegression(n_components=3).fit(X, y).predict(X)
    assert np.sqrt(np.mean((ours - ref) ** 2)) < 1e-6 or \
        np.corrcoef(ours, ref)[0, 1] > 0.999


def test_simpls_shape_and_no_nan():
    X, y = _random_data(A=4)
    m = SIMPLS(4).fit(X, y)
    assert m.beta_.shape == (X.shape[1],)
    assert np.all(np.isfinite(m.predict(X)))


def test_simpls_reproducible_seed():
    X, y = _random_data(A=2)
    p1 = SIMPLS(2).fit(X, y).predict(X)
    p2 = SIMPLS(2).fit(X, y).predict(X)
    np.testing.assert_array_almost_equal(p1, p2, decimal=10)


def test_weighted_simpls_degrades_to_ols_on_flat_weights():
    """Equal weights => OLS-like behavior (fit a linear latent model well)."""
    X, y = _random_data(A=3)
    beta, inter = _weighted_simpls(X, y, np.ones(X.shape[0]), 3)
    pred = X @ beta + inter
    assert _r2(pred, y) > 0.9


def test_tukey_bisquare_bounds():
    u = np.array([0.0, 1.0, 5.0, 100.0])
    w = _tukey_bisquare(u, c=4.685)
    assert w[0] == 1.0
    assert w[-1] == 0.0
    assert np.all((w >= 0) & (w <= 1))


def test_rwsimpls_robust_under_contamination():
    """RWSIMPLS should generalize better than plain SIMPLS to a CLEAN holdout
    when the training response is contaminated (robustness to Y-outliers).
    Evaluated on clean validation, not in-sample, so overfitting outliers is
    penalized rather than rewarded."""
    rng = np.random.default_rng(1)
    n = 90
    X = rng.standard_normal((n, 10))
    beta = rng.standard_normal(10)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    # contaminate a training split only; validation stays clean
    Xtr, ytr, Xva, yva = X[:70], y[:70], X[70:], y[70:]
    ytr = ytr.copy()
    idx = rng.choice(70, size=7, replace=False)
    ytr[idx] += 8 * np.std(ytr)          # 10% Y-outliers in train
    rw = RWSIMPLS(3).fit(Xtr, ytr)
    pl = SIMPLS(3).fit(Xtr, ytr)
    assert _r2(rw.predict(Xva), yva) > _r2(pl.predict(Xva), yva)


def test_rwsimpls_weights_in_01():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 8))
    y = X @ rng.standard_normal(8) + 0.1 * rng.standard_normal(60)
    m = RWSIMPLS(3).fit(X, y)
    assert np.all((m.weights_ >= 0) & (m.weights_ <= 1))
