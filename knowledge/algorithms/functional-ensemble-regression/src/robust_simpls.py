"""RWSIMPLS — robust weighted SIMPLS.

Adds robustness against response contamination on top of SIMPLS via
iteratively reweighted estimation: fit weighted SIMPLS, down-weight samples
with large standardized residuals (Tukey bisquare on MAD-scaled residuals),
iterate until weights stabilize.

Mirrors the robust weighted SIMPLS idea in Alin (2026) without claiming exact
reproduction (Inspired-by framing).
"""

from __future__ import annotations

import numpy as np


def _weighted_simpls(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                     A: int) -> tuple[np.ndarray, float]:
    """SIMPLS on sqrt(w)-scaled centered data (WLS-style latent regression)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    w = np.asarray(w, float).ravel()
    wsum = w.sum()
    if wsum <= 0:
        return np.zeros(X.shape[1]), float(y.mean())
    Xm = (w[:, None] * X).sum(0) / wsum
    ym = (w * y).sum() / wsum
    Xw = np.sqrt(w[:, None]) * (X - Xm)
    yw = np.sqrt(w) * (y - ym)
    n, p = Xw.shape
    A = min(int(A), n - 1, p)
    if A <= 0:
        return np.zeros(p), float(ym)
    W = np.zeros((p, A))
    T = np.zeros((n, A))
    P = np.zeros((p, A))
    Q = np.zeros(A)
    V = np.zeros((p, A))
    S = Xw.T @ yw
    for a in range(A):
        nS = np.linalg.norm(S)
        if nS < 1e-14:
            A = a
            break
        wvec = S / nS
        if a > 0:
            wvec = wvec - V[:, :a] @ (V[:, :a].T @ wvec)
            nw = np.linalg.norm(wvec)
            if nw < 1e-12:
                A = a
                break
            wvec = wvec / nw
        t = Xw @ wvec
        tt = t @ t
        c = Xw.T @ t / tt
        q = t @ yw / tt
        S = S - c * (c @ S)
        W[:, a] = wvec
        T[:, a] = t
        P[:, a] = c
        Q[a] = q
        v = c.copy()
        for j in range(a):
            v = v - (V[:, j] @ v) * V[:, j]
        nv = np.linalg.norm(v)
        V[:, a] = v / nv if nv > 1e-12 else v
    A_eff = max(A, 1)  # A already bounds n-1 and p inside the loop
    P_W = P[:, :A_eff].T @ W[:, :A_eff]
    beta = W[:, :A_eff] @ np.linalg.solve(P_W, Q[:A_eff])
    intercept = float(ym - Xm @ beta)
    return beta, intercept


def _tukey_bisquare(u: np.ndarray, c: float = 4.685) -> np.ndarray:
    return np.where(np.abs(u) <= c, (1 - (u / c) ** 2) ** 2, 0.0)


class RWSIMPLS:
    """Robust weighted SIMPLS (MAD-reweighted iteratively).

    Parameters
    ----------
    n_components : int
        Latent components A.
    max_iter : int
        IRLS iterations.
    tol : float
        Weight convergence tolerance.
    """

    def __init__(self, n_components: int = 3, max_iter: int = 8,
                 tol: float = 1e-6) -> None:
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RWSIMPLS":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n = X.shape[0]
        w = np.ones(n)
        for it in range(self.max_iter):
            beta, inter = _weighted_simpls(X, y, w, self.n_components)
            r = y - (X @ beta + inter)
            med = np.median(r)
            s = 1.4826 * np.median(np.abs(r - med))
            if s < 1e-10:
                break
            w_new = _tukey_bisquare((r - med) / s)
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new
        self.beta_ = beta
        self.intercept_ = inter
        self.weights_ = w
        self.residuals_ = r
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return X @ self.beta_ + self.intercept_
