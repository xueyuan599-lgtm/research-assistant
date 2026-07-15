# Modelling Matrix Time Series via Tensor CP-Decomposition

**Source**: Chang, J., He, J., Yang, L., & Yao, Q. (2023). Modelling matrix time series via a tensor CP-decomposition. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 85(1), 127--148. https://doi.org/10.1093/jrsssb/qkac011

**Category**: Statistics / Time Series / High-Dimensional Inference

## Mathematical Setup

Consider a matrix time series $\{Y_t\}_{t=1}^T$ where each $Y_t$ is a $p \times q$ matrix. Traditional approaches either vectorize $Y_t$ (losing matrix structure) or use simple factor models that do not exploit the bilinear structure. Chang et al. (2023) propose a **tensor CP-decomposition** (canonical polyadic decomposition) approach that models the matrix time series as:

$$
Y_t = \sum_{r=1}^R a_{0r} \cdot \boldsymbol{u}_r \boldsymbol{v}_r^\top + \sum_{r=1}^R a_{tr} \cdot \boldsymbol{u}_r \boldsymbol{v}_r^\top + \varepsilon_t
$$

where:
- $\boldsymbol{u}_r \in \mathbb{R}^p$ and $\boldsymbol{v}_r \in \mathbb{R}^q$ are the loading vectors for the $r$-th factor (column and row factors)
- $a_{0r}$ are intercept terms
- $a_{tr}$ are latent factor processes capturing dynamics
- $\varepsilon_t$ is the noise matrix

The model is equivalently expressed as a tensor CP-decomposition of the $p \times q \times T$ array:

$$
\mathcal{Y} = \sum_{r=1}^R \boldsymbol{u}_r \circ \boldsymbol{v}_r \circ \boldsymbol{\alpha}_r + \mathcal{E}
$$

where $\circ$ denotes the outer product, and $\boldsymbol{\alpha}_r = (a_{1r}, \ldots, a_{Tr})^\top$ is the $r$-th factor process.

The key innovation is a **one-pass estimation procedure** based on a generalized eigenanalysis constructed from the serial dependence structure. Define the lag-$\ell$ autocovariance matrix of the vectorized process:

$$
\Sigma(\ell) = \frac{1}{T - \ell} \sum_{t = 1}^{T - \ell} \text{vec}(Y_t - \bar{Y}) \, \text{vec}(Y_{t+\ell} - \bar{Y})^\top
$$

The row and column factor spaces are estimated by solving a generalized eigenvalue problem involving the sum of squared autocovariances.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Factor structure | $R \ll \min(p,q)$, rank of the signal component is $R$ | Dimension reduction from $p \times q$ to $(p+q)R$ parameters |
| Serial dependence | $\{a_{tr}\}$ is a stationary, ergodic process with $\sum_{\ell} \|\text{Cov}(a_{t}, a_{t+\ell})\| < \infty$ | Autocovariance matrices have structure that reveals factors |
| Weak cross-correlation | $\varepsilon_t$ has bounded temporal and cross-sectional dependence | Noise does not dominate the autocovariance signal |
| Separability | Row and column factor spaces are identifiable | Factors can be uniquely recovered up to rotation |
| Regularity | $\frac{1}{T} \sum_{t=1}^T a_t a_t^\top \to \Sigma_A > 0$ in probability | Factor covariance is full rank |

Under these assumptions, the estimated loadings satisfy:

$$
\|\hat{\boldsymbol{u}}_r - \boldsymbol{u}_r\| = O_P\left(\frac{1}{\sqrt{T}} + \frac{1}{\sqrt{p}} + \frac{1}{\sqrt{q}}\right)
$$

$$
\|\hat{\boldsymbol{v}}_r - \boldsymbol{v}_r\| = O_P\left(\frac{1}{\sqrt{T}} + \frac{1}{\sqrt{p}} + \frac{1}{\sqrt{q}}\right)
$$

## Applicable Scenarios

- **When to use**: Matrix-valued time series (financial portfolios $\times$ asset characteristics, spatial-temporal data, EEG channels $\times$ frequency bands), when preserving matrix structure improves estimation, when $p$ and $q$ are moderate to large ($> 10$).
- **When NOT to use**: When $T$ is very small ($< 20$) relative to $p,q$, when the matrix structure is not meaningful (rows and columns are interchangeable categories), when $R$ is close to $\min(p,q)$.
- **Comparison with classical alternatives**: Vectorized VAR is infeasible for large $p,q$ ($p q$ parameters). Standard factor models ignore the bilinear structure. CP-decomposition achieves parsimony ($(p+q+T)R$ parameters vs $(pq)T$ for raw data).

## Method Details

1. **Centering**: Compute $\bar{Y} = \frac{1}{T} \sum_{t=1}^T Y_t$ and center the series.
2. **Autocovariance accumulation**: For a set of lags $L = \{1, \ldots, \ell_0\}$, compute $\Sigma(\ell)$ and form $M = \sum_{\ell \in L} \Sigma(\ell) \Sigma(\ell)^\top$.
3. **Column factor estimation**: Perform eigenvalue decomposition of $M$ to get column factor estimates $\{\hat{\boldsymbol{v}}_r\}_{r=1}^R$ (or a similar procedure using the row-side matrix).
4. **Row factor estimation**: Similarly, swap roles of rows and columns and estimate $\{\hat{\boldsymbol{u}}_r\}_{r=1}^R$.
5. **Factor process recovery**: Estimate $a_{tr}$ by projecting $Y_t$ onto the estimated loading spaces.
6. **Rank selection**: Use eigenvalue ratio method or information criterion to determine $R$.

**Theoretical guarantees**:
- **Consistency**: Loading vectors are estimated consistently as $T, p, q \to \infty$
- **Rate**: Convergence rates depend on $T, p, q$ jointly; double asymptotics
- **Automatic factor structure discovery**: The generalized eigenanalysis extracts factors without iterative CP algorithms (which have no convergence guarantees)

## Implementation Details

- **Key parameters**: Lag set $L$, number of factors $R$ (or automatic selection)
- **Computational considerations**: $O(\ell_0 p^2 q + \ell_0 p q^2)$ for autocovariances; eigenvalue decomposition is $O(p^3 + q^3)$. One-pass: no iteration required.
- **Software availability**: R package `HDTSA` includes the CP-decomposition method for matrix time series.

## Python Implementation

```python
import numpy as np
from numpy.linalg import eigh, svd
from typing import Tuple, Optional


class MatrixCPDecomposition:
    """Matrix time series modelling via tensor CP-decomposition.
    
    Implements the method of Chang, He, Yang & Yao (2023, JRSS-B)
    for estimating a CP-decomposition of matrix-variate time series
    using generalized eigenanalysis of serial dependence structure.
    
    Parameters
    ----------
    n_factors : int, default=None
        Number of CP factors (rank R). If None, estimated automatically 
        using eigenvalue ratio criterion.
    max_lag : int, default=5
        Maximum lag for autocovariance accumulation.
    var_threshold : float, default=0.85
        Variance explained threshold for automatic rank selection.
    """
    
    def __init__(
        self,
        n_factors: Optional[int] = None,
        max_lag: int = 5,
        var_threshold: float = 0.85,
    ):
        self.R = n_factors
        self.max_lag = max_lag
        self.var_threshold = var_threshold
        self._fitted = False
    
    def _center(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Center the matrix time series."""
        Y_mean = Y.mean(axis=0)
        return Y - Y_mean[np.newaxis, :, :], Y_mean
    
    def _autocov_accumulate(
        self, Y_c: np.ndarray, max_lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute accumulated autocovariance matrices.
        
        Returns matrices M_row and M_col whose eigenvectors
        reveal the row and column factor spaces.
        """
        T, p, q = Y_c.shape
        
        # Method 1: Use the sum of outer products of lagged autocovariances
        # Following the paper: construct M based on serial dependence
        
        # Compute autocovariance matrices (vectorized approach)
        M_row = np.zeros((p, p))
        M_col = np.zeros((q, q))
        
        for ell in range(1, max_lag + 1):
            # Lag-ell autocovariance: (p, q) -> (p, q)
            Gamma_ell = np.zeros((p, q))
            count = 0
            for t in range(T - ell):
                Gamma_ell += Y_c[t].T @ Y_c[t + ell] / (T - ell)
                count += 1
            
            if count > 0:
                Gamma_ell /= count
            
            # Row-side contribution: G_row = Gamma_ell @ Gamma_ell^T
            M_row += Gamma_ell @ Gamma_ell.T
            
            # Column-side contribution: G_col = Gamma_ell^T @ Gamma_ell
            M_col += Gamma_ell.T @ Gamma_ell
        
        return M_row, M_col
    
    def _estimate_rank(self, eigenvalues: np.ndarray) -> int:
        """Estimate rank using eigenvalue ratio criterion."""
        d = len(eigenvalues)
        
        if d <= 1:
            return 1
        
        # Eigenvalue ratio: r_hat = argmax_{i < d} lambda_i / lambda_{i+1}
        ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-15)
        
        # Find largest ratio, but require at least some variance explained
        cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        
        # Use ratio criterion with a minimum variance threshold
        r_candidate = np.argmax(ratios) + 1
        if cumvar[r_candidate - 1] >= self.var_threshold:
            return r_candidate
        
        # Fallback to variance threshold
        return int(np.searchsorted(cumvar, self.var_threshold) + 1)
    
    def fit(self, Y: np.ndarray) -> 'MatrixCPDecomposition':
        """Fit the CP-decomposition model.
        
        Parameters
        ----------
        Y : ndarray of shape (T, p, q)
            Matrix time series observations.
            
        Returns
        -------
        self : MatrixCPDecomposition
        """
        T, p, q = Y.shape
        
        # Center
        Y_c, self.Y_mean_ = self._center(Y)
        
        # Compute accumulated autocovariances
        M_row, M_col = self._autocov_accumulate(Y_c, self.max_lag)
        
        # Symmetrize
        M_row = (M_row + M_row.T) / 2
        M_col = (M_col + M_col.T) / 2
        
        # Eigen-decomposition for row loadings (u_r)
        evals_row, evecs_row = eigh(M_row)
        # Sort descending
        idx_row = np.argsort(-evals_row)
        evals_row = evals_row[idx_row]
        evecs_row = evecs_row[:, idx_row]
        
        # Eigen-decomposition for column loadings (v_r)
        evals_col, evecs_col = eigh(M_col)
        idx_col = np.argsort(-evals_col)
        evals_col = evals_col[idx_col]
        evecs_col = evecs_col[:, idx_col]
        
        # Determine rank
        if self.R is None:
            R_row = self._estimate_rank(evals_row)
            R_col = self._estimate_rank(evals_col)
            self.R = min(R_row, R_col)
        else:
            self.R = min(self.R, p, q)
        
        # Extract loading vectors
        self.U_ = evecs_row[:, :self.R]   # (p, R) row loadings
        self.V_ = evecs_col[:, :self.R]   # (q, R) column loadings
        
        if self.R == 0:
            self.factors_ = np.zeros((T, 0))
        else:
            # Estimate factor processes: a_t = diag(U^T @ Y_t @ V)
            # For the CP model, a_{tr} = u_r^T Y_t v_r
            self.factors_ = np.zeros((T, self.R))
            for t in range(T):
                for r in range(self.R):
                    u_r = self.U_[:, r]
                    v_r = self.V_[:, r]
                    self.factors_[t, r] = u_r @ Y_c[t] @ v_r
        
        self.eigenvalues_ = evals_row
        self._fitted = True
        return self
    
    def get_loadings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return estimated row and column loading matrices.
        
        Returns
        -------
        U : ndarray of shape (p, R)
            Row loading matrix.
        V : ndarray of shape (q, R)
            Column loading matrix.
        """
        if not self._fitted:
            raise RuntimeError("Must call .fit() first.")
        return self.U_, self.V_
    
    def get_factors(self) -> np.ndarray:
        """Return estimated latent factor processes.
        
        Returns
        -------
        factors : ndarray of shape (T, R)
            Latent factor time series.
        """
        if not self._fitted:
            raise RuntimeError("Must call .fit() first.")
        return self.factors_
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the time series from the CP decomposition.
        
        Returns
        -------
        Y_hat : ndarray of shape (T, p, q)
            Reconstructed matrix time series.
        """
        if not self._fitted:
            raise RuntimeError("Must call .fit() first.")
        
        T, p, q = len(self.factors_), self.U_.shape[0], self.V_.shape[0]
        Y_hat = np.zeros((T, p, q))
        
        for t in range(T):
            for r in range(self.R):
                Y_hat[t] += self.factors_[t, r] * np.outer(self.U_[:, r], self.V_[:, r])
        
        # Add back mean
        Y_hat += self.Y_mean_[np.newaxis, :, :]
        return Y_hat
    
    def forecast(self, h_steps: int, method: str = 'ar') -> np.ndarray:
        """Forecast future matrix observations.
        
        Uses univariate AR models on each factor process.
        
        Parameters
        ----------
        h_steps : int
            Number of steps to forecast.
        method : str, default='ar'
            Forecasting method. Currently only 'ar' is supported.
            
        Returns
        -------
        Y_forecast : ndarray of shape (h_steps, p, q)
            Forecasted matrix time series.
        """
        if not self._fitted:
            raise RuntimeError("Must call .fit() first.")
        
        # Fit AR(1) on each factor
        T = len(self.factors_)
        factor_forecast = np.zeros((h_steps, self.R))
        
        for r in range(self.R):
            f_r = self.factors_[:, r]
            # AR(1) via least squares
            phi = np.sum(f_r[1:] * f_r[:-1]) / (np.sum(f_r[:-1] ** 2) + 1e-10)
            f_hat = np.zeros(h_steps)
            f_last = f_r[-1]
            for h in range(h_steps):
                f_hat[h] = phi * f_last
                f_last = f_hat[h]
            factor_forecast[:, r] = f_hat
        
        # Convert factors to matrix observations
        Y_forecast = np.zeros((h_steps, self.U_.shape[0], self.V_.shape[0]))
        for h in range(h_steps):
            for r in range(self.R):
                Y_forecast[h] += factor_forecast[h, r] * np.outer(self.U_[:, r], self.V_[:, r])
            Y_forecast[h] += self.Y_mean_
        
        return Y_forecast


# ============================================================
# Example: Simulated matrix time series
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("=== Matrix Time Series via Tensor CP-Decomposition ===\n")
    
    # Generate matrix-variate time series with CP structure
    T_true, p_true, q_true = 300, 20, 15
    R_true = 3
    
    # True loading vectors
    U_true = np.random.randn(p_true, R_true)
    U_true, _ = np.linalg.qr(U_true)  # orthonormalize
    
    V_true = np.random.randn(q_true, R_true)
    V_true, _ = np.linalg.qr(V_true)  # orthonormalize
    
    # Latent factor processes: AR(1) with different persistence
    factor_true = np.zeros((T_true, R_true))
    for r in range(R_true):
        phi_r = 0.5 + 0.3 * r / R_true  # different persistence per factor
        e_r = np.random.randn(T_true) * 0.5
        f_r = np.zeros(T_true)
        f_r[0] = e_r[0]
        for t in range(1, T_true):
            f_r[t] = phi_r * f_r[t - 1] + e_r[t]
        factor_true[:, r] = f_r
    
    # Construct matrix observations
    Y_gen = np.zeros((T_true, p_true, q_true))
    for t in range(T_true):
        for r in range(R_true):
            Y_gen[t] += factor_true[t, r] * np.outer(U_true[:, r], V_true[:, r])
    
    # Add noise
    noise_scale = 0.3
    Y_obs = Y_gen + noise_scale * np.random.randn(T_true, p_true, q_true)
    
    print(f"True rank: {R_true}")
    print(f"Dimensions: T={T_true}, p={p_true}, q={q_true}")
    print(f"Signal-to-noise (approx): {1.0 / noise_scale:.1f}")
    
    # Fit CP-decomposition
    cp_model = MatrixCPDecomposition(n_factors=R_true, max_lag=3)
    cp_model.fit(Y_obs)
    
    U_est, V_est = cp_model.get_loadings()
    factors_est = cp_model.get_factors()
    
    # Evaluate loading recovery (up to sign/rotation): 
    # Compare estimated column space vs true
    proj_U = U_est @ U_est.T @ U_true
    subspace_error_U = np.linalg.norm(proj_U - U_true, 'fro') / np.sqrt(R_true)
    print(f"\nRow loading subspace error: {subspace_error_U:.4f}")
    
    proj_V = V_est @ V_est.T @ V_true
    subspace_error_V = np.linalg.norm(proj_V - V_true, 'fro') / np.sqrt(R_true)
    print(f"Column loading subspace error: {subspace_error_V:.4f}")
    
    # Factor process correlation (after aligning signs)
    print(f"\nFactor correlations (absolute):")
    for r in range(R_true):
        # Find best matching estimated factor
        corrs = [abs(np.corrcoef(factor_true[:, r], factors_est[:, s])[0, 1])
                 for s in range(R_true)]
        best = np.argmax(corrs)
        print(f"  Factor {r}: corr = {corrs[best]:.3f} (matches est. factor {best})")
    
    # Reconstruction error
    Y_recon = cp_model.reconstruct()
    recon_error = np.linalg.norm(Y_recon - Y_gen) / np.linalg.norm(Y_gen)
    print(f"\nReconstruction error (relative): {recon_error:.4f}")
    
    # Forecast example
    n_forecast = 10
    Y_forecast = cp_model.forecast(n_forecast)
    
    # Compare with naive factor model (vectorized PCA + AR)
    # This demonstrates the advantage of the matrix CP approach
    
    # ---- Comparison: PCA on vectorized data ----
    # Vectorize: (T, p*q)
    Y_vec = Y_obs.reshape(T_true, -1)
    Y_vec_centered = Y_vec - Y_vec.mean(axis=0)
    U_vec, S_vec, Vt_vec = np.linalg.svd(Y_vec_centered, full_matrices=False)
    factors_pca = U_vec[:, :R_true] * S_vec[:R_true]  # T x R
    
    # Align factors with truth
    corr_pca = []
    for r in range(R_true):
        c = [abs(np.corrcoef(factor_true[:, r], factors_pca[:, s])[0, 1])
             for s in range(R_true)]
        corr_pca.append(max(c))
    
    print(f"\nComparison: CP vs Vectorized PCA")
    print(f"  CP factor recovery: {np.mean([abs(np.corrcoef(factor_true[:, r], factors_est[:, r])[0, 1]) for r in range(R_true)]):.3f}")
    print(f"  PCA factor recovery: {np.mean(corr_pca):.3f}")
    
    print(f"\nModel parameter count:")
    print(f"  CP-decomposition: {(p_true + q_true + T_true) * R_true}")
    print(f"  Vectorized VAR(1): {(p_true * q_true) ** 2} (infeasible)")
    print(f"  Raw data: {T_true * p_true * q_true}")
```

## References

Chang, J., He, J., Yang, L., & Yao, Q. (2023). Modelling matrix time series via a tensor CP-decomposition. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 85(1), 127--148. https://doi.org/10.1093/jrsssb/qkac011

Chen, E. Y., & Fan, J. (2023). Statistical inference for high-dimensional matrix-variate factor models. *Journal of the American Statistical Association*, 118(542), 1038--1055. https://doi.org/10.1080/01621459.2021.1970568

Wang, D., Liu, X., & Chen, R. (2019). Factor models for matrix-valued high-dimensional time series. *Journal of Econometrics*, 208(1), 231--248. https://doi.org/10.1016/j.jeconom.2018.09.013

Lam, C., & Yao, Q. (2012). Factor modeling for high-dimensional time series: Inference for the number of factors. *The Annals of Statistics*, 40(2), 694--726. https://doi.org/10.1214/12-AOS970

Chang, J., Guo, B., & Yao, Q. (2018). Principal component analysis for second-order stationary vector time series. *The Annals of Statistics*, 46(5), 2095--2127. https://doi.org/10.1214/17-AOS1613
