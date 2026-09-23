"""SIMPLS — partial least squares via SIMPLS algorithm (de Jong, 1993).

Scalar-on-function regression base model. For univariate response y and
discretized functional predictor X (n x p, p = grid points or basis coefs),
estimates linear coefficient beta (p,) with a latent-space factorization.

This is the standard SIMPLS computational path, distinct from sklearn's
PLSRegression (NIPALS). Equivalence is validated numerically on random data
(see tests/test_simpls.py): RMSE diff < 1e-6 or correlation > 0.999.
"""

from __future__ import annotations

import numpy as np


class SIMPLS:
    """SIMPLS regression for scalar response.

    Parameters
    ----------
    n_components : int
        Number of latent components A (<= min(n-1, p)).
    """

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = int(n_components)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SIMPLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        A = min(self.n_components, n - 1, p)
        self.A_ = A
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        self.xmean_ = X.mean(axis=0).copy()
        W = np.zeros((p, A))
        T = np.zeros((n, A))
        P = np.zeros((p, A))
        Q = np.zeros(A)
        V = np.zeros((p, A))
        S = Xc.T @ yc  # p-vector cross product
        for a in range(A):
            nS = np.linalg.norm(S)
            if nS < 1e-14:
                self.A_ = a
                break
            w = S / nS
            if a > 0:
                # orthogonalize weight against span of previous loadings
                w = w - V[:, :a] @ (V[:, :a].T @ w)
                nw = np.linalg.norm(w)
                if nw < 1e-12:
                    self.A_ = a
                    break
                w = w / nw
            t = Xc @ w
            tt = t @ t
            c = Xc.T @ t / tt          # loading p
            q = t @ yc / tt            # scalar loading
            S = S - c * (c @ S)        # deflate cross product
            W[:, a] = w
            T[:, a] = t
            P[:, a] = c
            Q[a] = q
            # Gram-Schmidt orthonormal basis of span(P) for next deflation
            v = c.copy()
            for j in range(a):
                v = v - (V[:, j] @ v) * V[:, j]
            nv = np.linalg.norm(v)
            if nv > 1e-12:
                V[:, a] = v / nv
            else:
                V[:, a] = v
        A_eff = self.A_
        if A_eff == 0:
            self.beta_ = np.zeros(p)
            self.intercept_ = float(y.mean())
            self.T_ = np.empty((n, 0))
            self.W_ = np.empty((p, 0))
            return self
        P_W = P[:, :A_eff].T @ W[:, :A_eff]
        self.beta_ = W[:, :A_eff] @ np.linalg.solve(P_W, Q[:A_eff])
        self.intercept_ = float(y.mean() - X.mean(axis=0) @ self.beta_)
        self.T_ = T[:, :A_eff]
        self.W_ = W[:, :A_eff]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.beta_ + self.intercept_

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).predict(X)
