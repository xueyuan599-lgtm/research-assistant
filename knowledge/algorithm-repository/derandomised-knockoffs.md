# Derandomised Knockoffs

**Source**: Ren, Z., & Barber, R. F. (2024). Derandomised knockoffs: leveraging e-values for false discovery rate control. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 86(1), 122--154. https://doi.org/10.1093/jrsssb/qkad085

**Category**: Statistics / Variable Selection / Multiple Testing

## Mathematical Setup

Model-X knockoffs (Candes et al., 2018) provides finite-sample false discovery rate (FDR) control for variable selection in high-dimensional settings. For each feature $X_j$, we construct a "knockoff" copy $\tilde{X}_j$ such that for any subset $S \subseteq \{1,\ldots,p\}$,

$$
(X_j, \tilde{X}_j)_{j \in S} \overset{d}{=} (X_j, \tilde{X}_j)_{j \in S} \text{ after swapping } X_j \text{ and } \tilde{X}_j
$$

The knockoff statistic $W_j = |Z_j| \cdot \text{sign}(Z_j - \tilde{Z}_j)$ (or similar) measures the importance of $X_j$ relative to its knockoff, where $Z_j, \tilde{Z}_j$ are coefficient estimates for $X_j$ and $\tilde{X}_j$.

The standard procedure selects variables with $W_j$ above a data-dependent threshold:

$$
\hat{S} = \{j : W_j \geq \tau\}, \quad \tau = \min\left\{t > 0 : \frac{\#\{j : W_j \leq -t\}}{\#\{j : W_j \geq t\}} \leq q\right\}
$$

This achieves $FDR \leq q$ in finite samples. However, the procedure is **randomized** (knockoffs depend on random sampling), leading to instability across different runs.

Ren & Barber (2024) show that knockoffs is equivalent to an **e-BH procedure** based on e-values:

$$
e_j = \frac{p \cdot \mathbb{1}\{W_j \geq \tau_{\text{max}}\}}{\#\{k : |W_k| \geq |W_j|\}}
$$

By averaging e-values across $M$ independent knockoff runs, they obtain derandomized e-values:

$$
\bar{e}_j = \frac{1}{M} \sum_{m=1}^M e_j^{(m)}
$$

Applying the e-BH procedure to $\{\bar{e}_j\}_{j=1}^p$ provably controls FDR with dramatically reduced selection variability.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Known covariate distribution | $P_X$ is known or can be accurately estimated | Knockoffs can be sampled from the true conditional distribution $X_j \mid X_{-j}$ |
| Exchangeability of knockoffs | $(X_j, \tilde{X}_j) \overset{d}{=} (X_j, \tilde{X}_j)$ under swap | Null statistics $W_j$ are symmetrically distributed around 0 |
| Independence of $Y \mid X$ | $Y_i \perp Y_{i'} \mid X_i, X_{i'}$ for $i \neq i'$ | Standard i.i.d. assumption |
| e-value validity | $E[e_j \mid \text{null}] \leq 1$ for each null hypothesis | e-BH controls FDR at level $q$ |

## Applicable Scenarios

- **When to use**: High-dimensional variable selection ($p \gg n$ or $p \approx n$), genome-wide association studies, feature selection in biomedical applications, any setting where FDR control is required and $P_X$ can be modeled.
- **When NOT to use**: When $P_X$ is completely unknown and cannot be estimated (although approximate knockoffs can be used with asymptotic guarantees), when $p$ is very small relative to $n$ (simpler methods like Bonferroni suffice), when features are deterministically related.
- **Comparison with classical alternatives**: LASSO+CV does not provide finite-sample FDR control. BH on p-values requires valid p-values (often unavailable in high dimensions). Standard knockoffs have high selection variability; derandomized knockoffs stabilizes selection while preserving FDR control.

## Method Details

1. **Knockoff construction**: For each feature $j$, construct knockoff $\tilde{X}_j$. For Gaussian data: $\tilde{X} = X(I - \Sigma^{-1} \text{diag}(s)) + \tilde{U}$, where $\tilde{U}$ is an orthogonal complement.
2. **Multiple knockoff runs**: Repeat knockoff construction and selection $M$ times (e.g., $M = 50$ or $M = 100$). Each run produces a knockoff matrix $\tilde{X}^{(m)}$.
3. **Feature statistics**: For each run $m$, compute knockoff statistics $W_j^{(m)}$. Common choices: LASSO coefficient difference, OLS after screening.
4. **E-value computation**: For each run $m$ and each feature $j$, compute $e_j^{(m)}$, the knockoff e-value.
5. **Averaging**: Compute $\bar{e}_j = \frac{1}{M} \sum_{m=1}^M e_j^{(m)}$.
6. **e-BH procedure**: Apply the e-BH procedure to $\{\bar{e}_j\}$ at level $q$ to select variables.

**Theoretical guarantees**:
- **Finite-sample FDR control**: $FDR \leq q$ for any $M \geq 1$
- **Consistency**: As $n \to \infty$, derandomized knockoffs selects the true support with probability approaching 1
- **Variance reduction**: Selection variability decreases as $1/M$
- **e-BH exactness**: If null $e_j$ are exactly super-uniform, $FDR = q \cdot \frac{|\mathcal{H}_0|}{p} \leq q$

## Implementation Details

- **Key parameters**: Target FDR level $q$, number of knockoff runs $M$, knockoff construction method (Gaussian, second-order, D-vine copula)
- **Computational considerations**: $M$ times more expensive than standard knockoffs, but embarrassingly parallel
- **Software availability**: R package `knockoff`; code from Ren & Barber at https://github.com/zhimeir/derandomized_knockoffs

## Python Implementation

```python
import numpy as np
from numpy.linalg import eigh, inv, cholesky
from typing import Tuple, Optional
from sklearn.linear_model import LassoCV, RidgeCV


class DerandomizedKnockoffs:
    """Derandomized model-X knockoffs with e-value aggregation.
    
    Implements the method of Ren & Barber (2024) for stable variable
    selection with finite-sample FDR control by aggregating e-values
    across multiple knockoff realizations.
    
    Parameters
    ----------
    fdr_level : float, default=0.1
        Target false discovery rate level q.
    n_runs : int, default=50
        Number of independent knockoff runs for derandomization.
    knockoff_method : str, default='gaussian'
        Method for constructing knockoffs: 'gaussian' for multivariate
        normal approximation, 'second_order' for general second-order.
    statistic : str, default='lasso'
        Knockoff statistic: 'lasso' for LASSO coefficient difference,
        'ridge' for ridge coefficient difference.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        fdr_level: float = 0.1,
        n_runs: int = 50,
        knockoff_method: str = 'gaussian',
        statistic: str = 'lasso',
        random_state: Optional[int] = None,
    ):
        self.q = fdr_level
        self.M = n_runs
        self.knockoff_method = knockoff_method
        self.statistic = statistic
        self.random_state = random_state
        self._fitted = False
        
    def _estimate_covariance(self, X: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix with shrinkage."""
        n, p = X.shape
        X_centered = X - X.mean(axis=0)
        S = (X_centered.T @ X_centered) / n
        
        # Simple shrinkage toward diagonal
        shrinkage = 0.2
        S_shrunk = (1 - shrinkage) * S + shrinkage * np.diag(np.diag(S))
        return S_shrunk
    
    def _construct_gaussian_knockoffs(
        self, X: np.ndarray, Sigma: np.ndarray
    ) -> np.ndarray:
        """Construct knockoffs for Gaussian covariates.
        
        Uses the equi-correlated construction which maximizes power
        subject to the SDP constraints.
        """
        n, p = X.shape
        X_centered = X - X.mean(axis=0)
        
        # Compute s vector: the equi-correlated knockoff construction
        # solves: minimize s subject to 0 <= s_j <= 1, 2*Sigma - diag(s)*Sigma^{-1}*diag(s) >= 0
        
        # Closed-form approximation: set all s_j equal to min eigenvalue of 2*Sigma
        # This is a simplified approximation (full SDP would be more powerful)
        evals = eigh(Sigma)[0]
        s_opt = min(1.0, 2.0 * evals[0])  # approximate equi-correlated
        s = np.full(p, s_opt)
        
        # Compute the knockoff covariance: G = 2*Sigma - diag(s)*inv(Sigma)*diag(s)
        Sigma_inv = inv(Sigma)
        s_mat = np.diag(s)
        G = 2 * Sigma - s_mat @ Sigma_inv @ s_mat
        
        # Ensure G is positive semi-definite
        evals_G, evecs_G = eigh(G)
        evals_G = np.maximum(evals_G, 1e-10)
        G_psd = evecs_G @ np.diag(evals_G) @ evecs_G.T
        G_sqrt = evecs_G @ np.diag(np.sqrt(evals_G)) @ evecs_G.T
        
        # Generate knockoffs
        U = np.random.randn(n, p)
        X_tilde = X_centered - X_centered @ Sigma_inv @ s_mat + U @ G_sqrt.T
        
        return X_tilde
    
    def _compute_lasso_statistics(
        self, X_aug: np.ndarray, y: np.ndarray, p: int
    ) -> np.ndarray:
        """Compute knockoff statistics using LASSO coefficient difference.
        
        $W_j = |\\hat{\\beta}_j| - |\\hat{\\beta}_{j+p}|$ where
        $\\hat{\\beta}$ is the LASSO estimate on [X, X_tilde].
        """
        # Fit cross-validated LASSO on augmented design
        lasso = LassoCV(cv=5, max_iter=5000, random_state=self.random_state)
        lasso.fit(X_aug, y)
        beta = lasso.coef_
        
        W = np.abs(beta[:p]) - np.abs(beta[p:])
        return W
    
    def _compute_e_values(self, W: np.ndarray, p: int) -> np.ndarray:
        """Compute knockoff e-values from feature statistics.
        
        Following Ren & Barber (2024), the e-value for feature j is:
        e_j = p * I(W_j >= tau_max) / |{k: |W_k| >= |W_j|}|
        where tau_max = max_{j: W_j < 0} -W_j (or 0 if none negative).
        """
        # Threshold for knockoff filter
        neg_W = W[W < 0]
        if len(neg_W) > 0:
            tau_max = np.max(-neg_W)
        else:
            tau_max = 0.0
        
        e_vals = np.zeros(p)
        for j in range(p):
            if W[j] >= tau_max:
                denominator = np.sum(np.abs(W) >= np.abs(W[j]))
                if denominator > 0:
                    e_vals[j] = p / denominator
        
        return e_vals
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DerandomizedKnockoffs':
        """Fit the derandomized knockoff procedure.
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Feature matrix.
        y : ndarray of shape (n,)
            Response vector.
            
        Returns
        -------
        self : DerandomizedKnockoffs
        """
        n, p = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Estimate covariance
        Sigma = self._estimate_covariance(X)
        
        # Run M knockoff replications
        e_values_all = np.zeros((self.M, p))
        
        for m in range(self.M):
            # Construct knockoffs
            X_tilde = self._construct_gaussian_knockoffs(X, Sigma)
            
            # Augmented design matrix
            X_aug = np.column_stack([X, X_tilde])
            
            # Standardize
            X_aug = (X_aug - X_aug.mean(axis=0)) / (X_aug.std(axis=0) + 1e-10)
            
            # Compute knockoff statistics
            W = self._compute_lasso_statistics(X_aug, y, p)
            
            # Convert to e-values
            e_values_all[m] = self._compute_e_values(W, p)
        
        # Average e-values across runs
        self.e_values_ = e_values_all.mean(axis=0)
        
        # Apply e-BH procedure
        # Sort by e-values descending, find largest k such that e_(k) >= p / (k * q)
        sorted_idx = np.argsort(-self.e_values_)
        sorted_e = self.e_values_[sorted_idx]
        
        threshold_e = np.arange(1, p + 1) * self.q / p
        reject = sorted_e >= 1.0 / threshold_e  # e-values are already scaled by 1/e-BH
        
        # e-BH: find largest k such that e_{(k)} * k / p >= 1/q
        # Equivalently: e_{(k)} >= p / (k * q)
        # This means: reject if e_j * p / (rank * q) >= 1
        # Equivalent to: e_j >= rank * q / p
        
        # Actually, e-BH rejects when e_(j) >= p / (j * q)
        # where e_(1) >= e_(2) >= ... >= e_(p)
        e_thresholds = np.arange(1, p + 1) * self.q / p
        
        # Find number of rejections: max k such that e_(k) * e_threshold_(k) >= 1
        # Wait, the standard e-BH: reject H_(j) if e_j >= p / (j * q)
        # where j is the rank. So we check: sorted_e[k] >= p / ((k+1) * q)
        
        comparisons = sorted_e >= p / (np.arange(1, p + 1) * self.q)
        if np.any(comparisons):
            k_max = np.where(comparisons)[0][-1] + 1
            # Threshold: smallest e-value among selected
            e_thresh = sorted_e[k_max - 1]
        else:
            k_max = 0
            e_thresh = np.inf
        
        self.selected_ = sorted_idx[:k_max]
        self.n_selected_ = k_max
        self.e_threshold_ = e_thresh
        self._fitted = True
        
        return self
    
    def get_selection(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices of selected variables and their e-values.
        
        Returns
        -------
        selected_idx : ndarray of shape (n_selected,)
            Indices of selected features (0-indexed).
        e_values : ndarray of shape (n_selected,)
            Derandomized e-values for selected features.
        """
        if not self._fitted:
            raise RuntimeError("Must call .fit() before .get_selection()")
        return self.selected_, self.e_values_[self.selected_]


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("=== Derandomized Knockoffs: Variable Selection with FDR Control ===\n")
    
    # Simulate high-dimensional data
    n, p = 200, 500
    k_true = 15  # number of truly relevant variables
    
    # Covariance: AR(1) structure
    rho = 0.3
    Sigma = np.array([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    L = cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    
    # Sparse coefficient vector
    beta = np.zeros(p)
    true_idx = np.random.choice(p, k_true, replace=False)
    beta[true_idx] = np.random.randn(k_true) * 2.0
    
    # Response
    y = X @ beta + 0.8 * np.random.randn(n)
    
    print(f"True support size: {k_true}")
    print(f"n = {n}, p = {p}")
    print(f"FDR level q = 0.1\n")
    
    # Derandomized knockoffs
    dk = DerandomizedKnockoffs(
        fdr_level=0.1,
        n_runs=30,
        knockoff_method='gaussian',
        statistic='lasso',
        random_state=42,
    )
    dk.fit(X, y)
    selected, e_vals = dk.get_selection()
    
    # Evaluate
    false_discoveries = len(set(selected) - set(true_idx))
    true_positives = len(set(selected) & set(true_idx))
    sensitivity = true_positives / k_true
    
    if len(selected) > 0:
        fdp = false_discoveries / len(selected)
    else:
        fdp = 0.0
    
    print(f"Number selected: {len(selected)}")
    print(f"True positives: {true_positives}")
    print(f"False discoveries: {false_discoveries}")
    print(f"False discovery proportion: {fdp:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    
    # Top selected features
    if len(selected) > 0:
        print(f"\nTop selected features:")
        top_k = min(10, len(selected))
        top_idx = selected[:top_k]
        for j, idx in enumerate(top_idx):
            is_true = "TRUE" if idx in true_idx else "FALSE"
            print(f"  Feature {idx}: e-value = {e_vals[j]:.4f} [{is_true}]")
    
    # Compare with standard knockoffs (single run)
    print("\n--- Comparison: Standard Knockoffs (single run) ---")
    np.random.seed(42)
    Sigma_s = dk._estimate_covariance(X)
    X_tilde_single = dk._construct_gaussian_knockoffs(X, Sigma_s)
    X_aug = np.column_stack([X, X_tilde_single])
    X_aug = (X_aug - X_aug.mean(axis=0)) / (X_aug.std(axis=0) + 1e-10)
    W_single = dk._compute_lasso_statistics(X_aug, y, p)
    
    # Knockoff threshold
    neg_W = W_single[W_single < 0]
    tau_kf = np.max(-neg_W) if len(neg_W) > 0 else 0
    selected_single = np.where(W_single >= tau_kf)[0]
    
    fp_single = len(set(selected_single) - set(true_idx))
    tp_single = len(set(selected_single) & set(true_idx))
    fdp_single = fp_single / len(selected_single) if len(selected_single) > 0 else 0
    sens_single = tp_single / k_true
    
    print(f"Number selected: {len(selected_single)}")
    print(f"True positives: {tp_single}")
    print(f"False discovery proportion: {fdp_single:.3f}")
    print(f"Sensitivity: {sens_single:.3f}")
```

## References

Ren, Z., & Barber, R. F. (2024). Derandomised knockoffs: leveraging e-values for false discovery rate control. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 86(1), 122--154. https://doi.org/10.1093/jrsssb/qkad085

Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 80(3), 551--577. https://doi.org/10.1111/rssb.12265

Barber, R. F., Candes, E. J., & Samworth, R. J. (2020). Robust inference with knockoffs. *The Annals of Statistics*, 48(3), 1409--1431. https://doi.org/10.1214/19-AOS1854

Wang, R., & Ramdas, A. (2022). False discovery rate control with e-values. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 84(3), 822--852. https://doi.org/10.1111/rssb.12489

Spector, A., & Janson, L. (2022). Powerful knockoffs via minimizing reconstructability. *The Annals of Statistics*, 50(1), 252--276. https://doi.org/10.1214/21-AOS2104
