"""Functional feature extraction shared by the nonlinear GBDT component B,
the M2 baseline (Functional Features + GBDT), and the gating network.

All extractors are fit/transform: parameters (PCA loadings, B-spline basis)
are fit on a training fold and projected to validation/test, preventing
leakage. This is the SAME feature library used by AHFE-B and M2 (fairness).

Feature block (fits to function-valued predictor, discretized on grid):
  - FPCA scores       : top-K PCA scores of the curve matrix
  - B-spline coefs    : projection onto a cubic B-spline basis
  - derivative stats  : L2-norm / mean-abs of the finite-difference derivative
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import BSpline


def _bspline_basis(grid: np.ndarray, n_knots: int, degree: int = 3):
    """Cubic B-spline basis over the grid using a uniform knot sequence."""
    grid = np.asarray(grid, float)
    n_inner = max(n_knots, 1)
    knots = np.linspace(grid.min(), grid.max(), n_inner + 2)
    knots = np.pad(knots, (degree + 1, degree + 1), mode="edge")
    t = np.asarray(knots, float)
    n_basis = len(t) - degree - 1
    # design matrix B[i, j] = N_j(grid[i])
    B = np.zeros((len(grid), n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        B[:, j] = BSpline(t, c, degree, extrapolate=True)(grid)
    return B


class FunctionalFeatureExtractor:
    """Concatenates FPCA scores, B-spline coefficients, and derivative stats.

    Parameters
    ----------
    n_fpca : int
        Number of FPCA (PCA) score components.
    n_knots : int
        Number of inner knots for the B-spline basis.
    """

    def __init__(self, n_fpca: int = 6, n_knots: int = 4) -> None:
        self.n_fpca = int(n_fpca)
        self.n_knots = int(n_knots)

    def fit(self, X: np.ndarray, grid: np.ndarray | None = None) -> "FunctionalFeatureExtractor":
        X = np.asarray(X, float)
        n, p = X.shape
        grid = np.linspace(0, 1, p) if grid is None else np.asarray(grid, float)
        self.grid_ = grid
        self.pca_ = PCA(n_components=min(self.n_fpca, n - 1, p))
        self.pca_.fit(X)
        self.B_ = _bspline_basis(grid, self.n_knots)   # (p, n_basis)
        # cache projected derivative weight (identity operator on derivative)
        d = np.gradient(grid)
        self._dW_ = 1.0 / np.sqrt((d ** 2).sum() / len(d) + 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        fpca = self.pca_.transform(X)                       # (n, n_fpca)
        bs = X @ self.B_                                    # (n, n_basis)
        dX = np.gradient(X, axis=1)
        deriv = np.sqrt((dX ** 2).mean(axis=1)) * self._dW_  # (n,)
        return np.concatenate([fpca, bs, deriv[:, None]], axis=1)

    def fit_transform(self, X: np.ndarray, grid=None) -> np.ndarray:
        return self.fit(X, grid).transform(X)

    @property
    def n_features(self) -> int:
        return self.n_fpca + self.B_.shape[1] + 1


class GateFeatureExtractor:
    """Prediction-time gate features z = [FPCA scores, |rhat_B|, Q-residual, leverage].

    Only quantities computable when y is unobserved are allowed (no true
    residual magnitude). Fitted on a training fold, projected to val/test.
    """

    def __init__(self, n_fpca: int = 6) -> None:
        self.n_fpca = int(n_fpca)

    def fit(self, X: np.ndarray) -> "GateFeatureExtractor":
        X = np.asarray(X, float)
        n, p = X.shape
        self.pca_ = PCA(n_components=min(self.n_fpca, n - 1, p))
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray, yhat_A: np.ndarray, rhat_B: np.ndarray,
                  latents: np.ndarray) -> np.ndarray:
        """z for each sample.

        latents : (n, A) SIMPLS latent scores T (for leverage). If None/empty,
        leverage column is set to 0.
        """
        X = np.asarray(X, float)
        yhat_A = np.asarray(yhat_A, float).ravel()
        rhat_B = np.asarray(rhat_B, float).ravel()
        fpca = self.pca_.transform(X)
        abs_rb = np.abs(rhat_B)[:, None]
        # Q-residual: squared reconstruction error onto FPCA subspace
        Xrec = self.pca_.inverse_transform(self.pca_.transform(X))
        qres = ((X - Xrec) ** 2).mean(axis=1)[:, None]
        # leverage from latent scores
        if latents is not None and latents.ndim == 2 and latents.shape[1] > 0:
            tt = latents.T @ latents
            h = np.einsum("ni,nj->n", latents, np.linalg.solve(tt, latents.T).T)
            lev = h[:, None]
        else:
            lev = np.zeros((X.shape[0], 1))
        return np.concatenate([fpca, abs_rb, qres, lev], axis=1)
