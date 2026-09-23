"""Data loading and the synthetic S5 DGP (centered quadratic interaction).

S5 (nonlinear): X(t) ~ GP(0, C) zero-mean Matern on grid [0,1];
   q_i = int X_i(t) phi(t) dt ;
   Y_i = alpha + int X_i(t) beta(t) dt + lambda [ q_i^2 - E(q^2) ] + eps_i
   beta(t)=sin(2pi t), phi(t)=cos(2pi t); phi orthogonalized against C*beta
   so the linear and centered-quadratic terms are near-orthogonal -> beta is a
   valid ground truth for the linear component (plan.md §3.1 #1).

S5b (linear positive control): Y = alpha + int X beta dt + eps.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import Matern


def _sample_gp(n: int, grid: np.ndarray, length_scale: float = 0.10,
               nu: float = 2.5, seed: int = 0) -> np.ndarray:
    C = Matern(length_scale=length_scale, nu=nu)(grid.reshape(-1, 1),
                                                 grid.reshape(-1, 1))
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(grid)))
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, len(grid))) @ L.T   # (n, p)


def _orthogonalize(v: np.ndarray, u: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return v - proj, making <v, C*u> ~ 0 (covariance-operator orthogonality)."""
    Cu = C @ u
    proj = (v @ Cu) / (u @ Cu) * u
    out = v - proj
    return out / (np.linalg.norm(out) + 1e-12)


def generate_s5(n: int = 200, p: int = 100, nl_share: float = 1.0,
                snr: float = 3.0, nonlinear: bool = True, seed: int = 0,
                length_scale: float = 0.35) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate the S5 synthetic scalar-on-function task.

    The quadratic coefficient lambda is chosen adaptively so the nonlinear
    term's variance equals `nl_share` x the linear term's variance (avoids the
    degenerate case where an orthogonalized phi leaves ~zero nonlinear power).

    Returns (X (n,p), y (n,), info dict with beta, phi, q, nl, grid, lambda).
    """
    grid = np.linspace(0, 1, p)
    X = _sample_gp(n, grid, length_scale, seed=seed)          # (n, p)
    beta = np.sin(2 * np.pi * grid)
    phi = np.cos(2 * np.pi * grid)
    C = Matern(length_scale=length_scale, nu=2.5)(
        grid.reshape(-1, 1), grid.reshape(-1, 1))
    phi = _orthogonalize(phi, beta, C)                        # <phi, C*beta> ~ 0
    phi = phi / (np.linalg.norm(phi) + 1e-12)                 # unit norm

    dt = grid[1] - grid[0]
    lin = (X @ beta) * dt                                      # int X beta
    q = (X @ phi) * dt                                        # int X phi
    nl_raw = q ** 2 - np.mean(q ** 2)                          # centered quadratic
    var_lin = lin.var()
    var_nl = nl_raw.var()
    lambda_nl = float(np.sqrt(nl_share * var_lin / var_nl)) if var_nl > 0 else 0.0
    nl = np.abs(lambda_nl * nl_raw)                            # per-sample nonlinear strength
    signal = lin + (lambda_nl * nl_raw) if nonlinear else lin
    var_s = signal.var()
    eps = np.random.default_rng(seed + 1).standard_normal(n)
    eps = eps / (eps.std() + 1e-12) * np.sqrt(var_s / snr ** 2)
    y = 5.0 + signal + eps
    info = dict(beta=beta, phi=phi, q=q, nl=nl, grid=grid, dt=dt,
                lambda_nl=lambda_nl, nonlinear=nonlinear, snr=snr,
                nl_share=nl_share)
    return X, y, info


def load_tecator(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Tecator NIR. Returns (X (215,100), y_fat (215,), grid (100,),
    y_full (215,3))."""
    from skfda.datasets import fetch_tecator
    X, y = fetch_tecator(return_X_y=True)
    X = np.squeeze(X.data_matrix, axis=2)                      # (215, 100)
    y_full = np.asarray(y, float)
    grid = np.linspace(0, 1, X.shape[1])
    return X, y_full[:, 0].ravel(), grid, y_full
