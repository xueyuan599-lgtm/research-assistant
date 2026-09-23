"""Sigmoid-link residual-correction gate for AHFE.

w(z) = sigma(alpha + z^T theta) in [0,1] modulates the strength of the GBDT
residual corrector:  yhat_AHFE = yhat_A + w(z) * rhat_B.

Trained on out-of-fold (OOF) data from inner cross-fitting so it never sees
the outer-test target y. Minimizes sum_i (y_i - yhatA_i - w(z_i)*rhatB_i)^2
+ lambda_g * ||theta||^2. This is a sigmoid-parametrized gating regression
(logistic-link), NOT a classifier / binary-cross-entropy logistic regression.

The oracle gate is w*(z) = Pi_[0,1] E[eA*q|z] / E[q^2|z]; this learns an
estimator of it (see plan.md §2.3).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SigmoidGate:
    """w(z) = sigmoid(alpha + z^T theta)."""

    def __init__(self, lambda_g: float = 1e-4) -> None:
        self.lambda_g = float(lambda_g)

    def fit(self, z: np.ndarray, y: np.ndarray, yhat_A: np.ndarray,
            rhat_B: np.ndarray) -> "SigmoidGate":
        z = np.asarray(z, float)
        y = np.asarray(y, float).ravel()
        yhat_A = np.asarray(yhat_A, float).ravel()
        rhat_B = np.asarray(rhat_B, float).ravel()
        # standardize z columns for stable optimization
        self.zmean_ = z.mean(axis=0)
        self.zstd_ = z.std(axis=0)
        self.zstd_[self.zstd_ < 1e-12] = 1.0
        zs = (z - self.zmean_) / self.zstd_
        d = zs.shape[1]
        lam = self.lambda_g

        def loss(theta):
            a, wtheta = theta[0], theta[1:]
            w = _sigmoid(a + zs @ wtheta)
            resid = y - yhat_A - w * rhat_B
            return float(np.sum(resid ** 2) + lam * np.sum(wtheta ** 2))

        def grad(theta):
            a, wtheta = theta[0], theta[1:]
            lin = a + zs @ wtheta
            s = _sigmoid(lin)
            resid = y - yhat_A - s * rhat_B
            # dL/ds = -2 resid * rhat_B ; ds/dlin = s(1-s)
            g_lin = -2.0 * resid * rhat_B * s * (1.0 - s)
            ga = float(g_lin.sum())
            gw = zs.T @ g_lin + 2.0 * lam * wtheta
            return np.concatenate([[ga], gw])

        theta0 = np.zeros(d + 1)
        res = minimize(loss, theta0, jac=grad, method="L-BFGS-B")
        self.theta_ = res.x
        self.loss_ = res.fun
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        zs = (z - self.zmean_) / self.zstd_
        return _sigmoid(self.theta_[0] + zs @ self.theta_[1:])
