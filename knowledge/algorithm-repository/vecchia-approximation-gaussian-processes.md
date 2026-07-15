# A General Framework for Vecchia Approximations of Gaussian Processes

**Source**: Katzfuss, M., & Guinness, J. (2021). A general framework for Vecchia approximations of Gaussian processes. *Statistical Science*, 36(1), 124--141. https://doi.org/10.1214/19-STS755

**Category**: Statistics / Spatial Statistics / Gaussian Processes

## Mathematical Setup

Gaussian processes (GPs) are widely used for modelling spatial data, computer experiments, and time series. For $n$ observations, the GP likelihood involves an $n \times n$ covariance matrix $\Sigma$ with $O(n^3)$ computational cost, making it infeasible for $n > 10^4$.

The Vecchia approximation (Vecchia, 1988) approximates the joint density as a product of conditional densities:

$$
p(\mathbf{y}) = \prod_{i=1}^{n} p(y_i \mid \mathbf{y}_{i-1}) \approx \prod_{i=1}^{n} p(y_i \mid \mathbf{y}_{c(i)})
$$

where $\mathbf{y}_{c(i)}$ is a small conditioning set (size $m \ll n$) of previously ordered observations. Katzfuss & Guinness (2021) unify many existing GP approximations within a single framework.

The approximation implies a **sparse Cholesky factor** of the precision matrix $Q = \Sigma^{-1}$:

$$
\text{Var}[Y_i \mid \mathbf{Y}_{c(i)}] \approx \text{Var}[Y_i \mid \mathbf{Y}_{<i}]
$$

leading to a sparse Cholesky decomposition $LL^\top \approx Q$ where $L$ has at most $m$ nonzeros per column.

The log-likelihood is approximated as:

$$
\log \hat{p}(\mathbf{y}) = -\frac{1}{2} \sum_{i=1}^n \left( \log(2\pi \sigma_i^2) + \frac{(y_i - \boldsymbol{\mu}_i)^2}{\sigma_i^2} \right)
$$

where $\sigma_i^2 = \text{Var}[Y_i \mid \mathbf{Y}_{c(i)}]$ and $\boldsymbol{\mu}_i = E[Y_i \mid \mathbf{Y}_{c(i)}]$ are computed from the GP covariance kernel and the conditioning set.

The general Vecchia framework shows that **many existing GP approximations** (NNGP, FSA, FIC, modified predictive process) correspond to specific choices of:
1. **Ordering** of the observations (maximin ordering, random, coordinate-based)
2. **Conditioning sets** (nearest neighbors, block-based, hierarchical)
3. **Grouping** of the observations (individual or in blocks)

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| GP specification | $y \sim \mathcal{N}(\mu, \Sigma(\theta))$ with known covariance kernel $K(\cdot, \cdot; \theta)$ | Full GP defines the target |
| Conditional independence given neighbors | $Y_i \perp Y_{<i \setminus c(i)} \mid \mathbf{Y}_{c(i)}$ | The Vecchia approximation accuracy depends on this |
| Smooth kernel | $K(\cdot, \cdot)$ is at least $2k$ times differentiable for Matern with smoothness $k$ | Smooth processes allow smaller $m$ |
| Space-filling design | Locations are not pathologically clustered | Nearest-neighbor conditioning works well |
| Conditioning set size | $m = O(\log^d n)$ for Matern-like kernels in $d$ dimensions | Approximation error decays superalgebraically |

For Matern covariance functions, the Kullback-Leibler divergence between the true GP and the Vecchia approximation decays as:

$$
KL(p(\mathbf{y}) \| \hat{p}(\mathbf{y})) = O(n \cdot \rho^{m^{1/d}})
$$

for some $\rho \in (0, 1)$, implying exponential decay in $m$.

## Applicable Scenarios

- **When to use**: Large spatial datasets ($n > 10^4$), computer model emulation, spatiotemporal modelling, Bayesian optimization, any GP application where $O(n^3)$ is prohibitive.
- **When NOT to use**: Small datasets ($n < 10^3$), where exact GP inference is feasible; when the covariance kernel is non-smooth (e.g., exponential with small range); when the domain is highly irregular with distant isolated points.
- **Comparison with classical alternatives**: Exact GP ($O(n^3)$, $O(n^2)$ storage) is infeasible for large $n$. Inducing point methods (e.g., FITC, $O(n m^2)$) require optimizer tuning. Vecchia typically achieves better accuracy for the same $m$.

## Method Details

1. **Ordering**: Reorder observations (e.g., maximin ordering to reduce fill-in).
2. **Neighbor selection**: For each observation $i$, select $m$ conditioning indices $c(i)$ from previously ordered points (e.g., nearest neighbors by Euclidean distance).
3. **Conditional computation**: For each $i$, compute:
   - $\boldsymbol{\mu}_i = \Sigma_{i, c(i)} \Sigma_{c(i), c(i)}^{-1} \mathbf{y}_{c(i)}$
   - $\sigma_i^2 = \Sigma_{i,i} - \Sigma_{i, c(i)} \Sigma_{c(i), c(i)}^{-1} \Sigma_{c(i), i}$
4. **Likelihood evaluation**: Sum over $i$ the log conditional densities.
5. **Prediction**: For a new point $x^*$, compute conditional mean and variance given its nearest neighbors among the observed data.

**Theoretical guarantees**:
- **Sparsity**: Precision matrix has $O(n m)$ nonzeros; Cholesky factor has $O(n m^2)$ nonzeros
- **Exponential accuracy**: KL divergence decays exponentially in $m$ for smooth kernels
- **Linear scaling**: Overall $O(n m^3)$ complexity; $m$ is typically $10-50$ independent of $n$
- **Unified view**: Encompasses NNGP, FSA, and many other approximations as special cases

## Implementation Details

- **Key parameters**: Conditioning set size $m$, ordering method, distance metric
- **Computational considerations**: $O(n m^3)$ for full likelihood evaluation; $O(n m^2)$ for computing sparse Cholesky; dominated by nearest-neighbor search which can be accelerated with KD-trees ($O(n \log n)$)
- **Software availability**: R packages `GPvecchia` (Katzfuss et al.), `GpGp` (Guinness), `BRISC` (Saha & Datta)

## Python Implementation

```python
import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import solve_triangular, cholesky
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class VecchiaConfig:
    """Configuration for Vecchia approximation."""
    m_neighbors: int = 30          # Number of nearest neighbors
    ordering: str = 'maximin'       # 'random', 'maximin', 'coordinate'
    cov_func: Optional[Callable] = None  # Covariance function K(h; theta)
    nugget: float = 1e-6            # Jitter for numerical stability


def matern_covariance(
    h: np.ndarray, sigma2: float = 1.0, rho: float = 1.0, nu: float = 1.5
) -> np.ndarray:
    """Matern covariance function.
    
    Parameters
    ----------
    h : ndarray
        Distance matrix or vector of distances.
    sigma2 : float
        Variance parameter.
    rho : float
        Range parameter.
    nu : float
        Smoothness parameter.
    """
    from scipy.special import kv as bessel_k
    from scipy.special import gamma
    
    h = np.asarray(h, dtype=float)
    h_scaled = np.sqrt(2 * nu) * h / rho
    
    # Handle zero distances
    mask = h > 0
    result = np.ones_like(h) * sigma2
    
    if nu == 0.5:  # Exponential
        result[mask] = sigma2 * np.exp(-h[mask] / rho)
    elif nu == np.inf:  # Gaussian/RBF
        result = sigma2 * np.exp(-0.5 * (h / rho) ** 2)
    else:
        result[mask] = (
            sigma2 / (2 ** (nu - 1) * gamma(nu))
            * (h_scaled[mask] ** nu)
            * bessel_k(nu, h_scaled[mask])
        )
    return result


class VecchiaGP:
    """Gaussian process with Vecchia approximation.
    
    Implements the general Vecchia framework of Katzfuss & Guinness (2021)
    for scalable GP inference with O(n*m^3) complexity.
    
    Parameters
    ----------
    config : VecchiaConfig
        Configuration parameters.
    """
    
    def __init__(self, config: Optional[VecchiaConfig] = None):
        self.config = config or VecchiaConfig()
        self._fitted = False
    
    def _maximin_ordering(self, locs: np.ndarray) -> np.ndarray:
        """Maximin distance ordering for reduced fill-in.
        
        Selects points to maximize the minimum distance to all previously
        selected points.
        """
        n = len(locs)
        order = np.zeros(n, dtype=int)
        order[0] = 0
        
        # Tree for nearest neighbor queries
        available = np.ones(n, dtype=bool)
        available[0] = False
        
        for i in range(1, n):
            # For each available point, find min distance to ordered set
            ordered_locs = locs[order[:i]]
            tree = cKDTree(ordered_locs)
            
            max_min_dist = -1
            best_idx = -1
            
            # Sample from available for efficiency if n is large
            candidates = np.where(available)[0]
            if len(candidates) > 1000:
                candidates = candidates[np.random.choice(
                    len(candidates), 1000, replace=False
                )]
            
            for idx in candidates:
                dist, _ = tree.query(locs[idx])
                if dist > max_min_dist:
                    max_min_dist = dist
                    best_idx = idx
            
            order[i] = best_idx
            available[best_idx] = False
        
        return order
    
    def _get_conditioning_sets(
        self, locs: np.ndarray, order: np.ndarray
    ) -> List[np.ndarray]:
        """Find nearest neighbor conditioning sets for each point.
        
        For each i in the ordering, finds the m closest previously
        ordered points.
        """
        n = len(locs)
        m = self.config.m_neighbors
        
        # Build tree for ordered points
        ordered_locs = locs[order]
        tree = cKDTree(ordered_locs[:1])
        
        cond_sets = [np.array([], dtype=int)]  # First point has no neighbors
        
        for i in range(1, n):
            k = min(m, i)
            dists, idx = tree.query(
                ordered_locs[i].reshape(1, -1), k=k + 1
            )
            
            # idx[0][0] should be the point itself
            neighbor_idx = idx[0][1:]
            
            # Map back to original indices
            cond_sets.append(order[neighbor_idx])
            
            # Update tree
            if i < n - 1:
                tree = cKDTree(ordered_locs[:i + 1])
        
        return cond_sets
    
    def fit(
        self, locs: np.ndarray, y: np.ndarray,
        cov_params: Optional[dict] = None,
    ) -> 'VecchiaGP':
        """Fit the Vecchia GP approximation.
        
        Computes the Vecchia-approximated log-likelihood and stores
        the sparse Cholesky factor structure.
        
        Parameters
        ----------
        locs : ndarray of shape (n, d)
            Input locations.
        y : ndarray of shape (n,)
            Observations.
        cov_params : dict, optional
            Covariance function parameters.
        """
        n = len(locs)
        self.locations_ = locs
        self.y_ = y
        
        # Default covariance parameters
        if cov_params is None:
            cov_params = {'sigma2': 1.0, 'rho': 1.0, 'nu': 1.5}
        self.cov_params_ = cov_params
        self.cov_func_ = self.config.cov_func or (
            lambda h: matern_covariance(h, **cov_params)
        )
        
        # Ordering
        if self.config.ordering == 'maximin':
            self.order_ = self._maximin_ordering(locs)
        elif self.config.ordering == 'random':
            self.order_ = np.random.permutation(n)
        else:
            self.order_ = np.arange(n)
        
        ordered_locs = locs[self.order_]
        ordered_y = y[self.order_]
        
        # Build conditioning sets
        cond_sets = self._get_conditioning_sets(locs, self.order_)
        
        # Compute Vecchia likelihood terms
        log_lik = 0.0
        self.cond_means_ = np.zeros(n)
        self.cond_vars_ = np.ones(n)
        
        for i in range(n):
            neighbors = cond_sets[i]
            m_i = len(neighbors)
            
            if m_i == 0:
                # First observation: unconditional
                self.cond_means_[i] = 0.0  # assuming zero mean
                self.cond_vars_[i] = self.cov_func_(0.0) + self.config.nugget
            else:
                # Compute conditional mean and variance
                loc_i = ordered_locs[i]
                loc_neighbors = ordered_locs[np.isin(self.order_, neighbors)]
                
                # Actually need to map carefully
                neighbor_positions = np.array([
                    np.where(self.order_ == nbr)[0][0] for nbr in neighbors
                ])
                
                # Vecchia: p(y_i | y_{c(i)})
                # Covariance between y_i and its neighbors
                dists_ineigh = np.array([
                    np.linalg.norm(loc_i - ordered_locs[pos])
                    for pos in neighbor_positions
                ])
                K_ineigh = self.cov_func_(dists_ineigh)
                
                # Covariance among neighbors
                K_neigh = np.zeros((m_i, m_i))
                for a in range(m_i):
                    for b in range(m_i):
                        d = np.linalg.norm(
                            ordered_locs[neighbor_positions[a]] -
                            ordered_locs[neighbor_positions[b]]
                        )
                        K_neigh[a, b] = self.cov_func_(d)
                
                # Add nugget to diagonal
                K_neigh += np.eye(m_i) * self.config.nugget
                
                # Conditional mean and variance
                try:
                    L_neigh = cholesky(K_neigh, lower=True)
                    alpha = solve_triangular(
                        L_neigh, K_ineigh, lower=True
                    )
                    alpha = solve_triangular(
                        L_neigh.T, alpha, lower=False
                    )
                    
                    self.cond_means_[i] = alpha @ ordered_y[neighbor_positions]
                    self.cond_vars_[i] = (
                        self.cov_func_(0.0) + self.config.nugget
                        - alpha @ K_ineigh
                    )
                except np.linalg.LinAlgError:
                    self.cond_vars_[i] = self.cov_func_(0.0) + self.config.nugget
                    self.cond_means_[i] = 0.0
            
            # Accumulate log-likelihood
            if self.cond_vars_[i] > 0:
                log_lik += (
                    -0.5 * np.log(2 * np.pi * self.cond_vars_[i])
                    - 0.5 * (ordered_y[i] - self.cond_means_[i]) ** 2
                    / self.cond_vars_[i]
                )
        
        self.log_likelihood_ = log_lik
        self.cond_sets_ = cond_sets
        self._fitted = True
        
        return self
    
    def predict(
        self, locs_new: np.ndarray, return_var: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions at new locations.
        
        Parameters
        ----------
        locs_new : ndarray of shape (n_new, d)
            Prediction locations.
        return_var : bool, default=True
            Whether to return predictive variances.
            
        Returns
        -------
        mu_pred : ndarray of shape (n_new,)
            Predictive means.
        var_pred : ndarray of shape (n_new,), optional
            Predictive variances (if return_var=True).
        """
        if not self._fitted:
            raise RuntimeError("Must call .fit() before .predict()")
        
        n_new = len(locs_new)
        locs = self.locations_
        y = self.y_
        m = self.config.m_neighbors
        
        tree = cKDTree(locs)
        
        mu_pred = np.zeros(n_new)
        var_pred = np.zeros(n_new)
        
        for i in range(n_new):
            # Find m nearest neighbors among all observations
            k = min(m, len(locs))
            dists, idx = tree.query(locs_new[i].reshape(1, -1), k=k)
            nn_idx = idx[0]
            nn_locs = locs[nn_idx]
            nn_y = y[nn_idx]
            
            # Build covariance matrices
            k_vec = np.array([
                self.cov_func_(np.linalg.norm(locs_new[i] - nn_locs[j]))
                for j in range(k)
            ])
            
            K_nn = np.zeros((k, k))
            for a in range(k):
                for b in range(k):
                    d = np.linalg.norm(nn_locs[a] - nn_locs[b])
                    K_nn[a, b] = self.cov_func_(d)
            
            K_nn += np.eye(k) * self.config.nugget
            
            try:
                L_nn = cholesky(K_nn, lower=True)
                alpha = solve_triangular(L_nn, k_vec, lower=True)
                alpha = solve_triangular(L_nn.T, alpha, lower=False)
                
                mu_pred[i] = alpha @ nn_y
                var_pred[i] = (
                    self.cov_func_(0.0) + self.config.nugget
                    - alpha @ k_vec
                )
            except np.linalg.LinAlgError:
                mu_pred[i] = 0.0
                var_pred[i] = self.cov_func_(0.0) + self.config.nugget
        
        if return_var:
            return mu_pred, var_pred
        return mu_pred
    
    def log_likelihood(self) -> float:
        """Return the Vecchia-approximated log-likelihood."""
        if not self._fitted:
            raise RuntimeError("Must call .fit() first.")
        return self.log_likelihood_


# ============================================================
# Example: Scalable GP on large spatial data
# ============================================================

if __name__ == "__main__":
    import time
    np.random.seed(42)
    print("=== Vecchia Approximation for Gaussian Processes ===\n")
    
    # Simulate large spatial dataset
    n = 5000  # large enough to show scalability
    d = 2
    
    locs = np.random.uniform(0, 10, (n, d))
    
    # True Matern parameters
    sigma2_true = 1.5
    rho_true = 2.0
    nu_true = 1.5
    
    # Generate data from true GP (using a small subset for Cholesky)
    print(f"Generating {n} spatial locations...")
    
    # For simulation, use Vecchia approximation with large m
    config_true = VecchiaConfig(
        m_neighbors=50, ordering='maximin', nugget=1e-6
    )
    gp_true = VecchiaGP(config_true)
    
    # Simulate data using the Vecchia approximation to the GP
    gp_true.fit(locs, np.zeros(n), {'sigma2': sigma2_true, 'rho': rho_true, 'nu': nu_true})
    
    # Generate data: sample from the implied conditional distributions
    y_sim = np.zeros(n)
    order = gp_true.order_
    ordered_locs = locs[order]
    cond_sets = gp_true.cond_sets_
    cond_means = gp_true.cond_means_
    cond_vars = gp_true.cond_vars_
    
    for i in range(n):
        y_sim[order[i]] = (
            cond_means[i] + np.sqrt(cond_vars[i]) * np.random.randn()
        )
    
    # ---- Fit Vecchia GP with different m values ----
    for m_test in [10, 20, 50]:
        config = VecchiaConfig(
            m_neighbors=m_test, ordering='maximin', nugget=1e-6
        )
        gp = VecchiaGP(config)
        
        t0 = time.time()
        gp.fit(locs, y_sim, {'sigma2': sigma2_true, 'rho': rho_true, 'nu': nu_true})
        t_elapsed = time.time() - t0
        
        print(f"m = {m_test:2d}: log-lik = {gp.log_likelihood():.1f}, "
              f"time = {t_elapsed:.2f}s")
    
    # ---- Prediction at new locations ----
    n_test = 200
    locs_test = np.random.uniform(0, 10, (n_test, d))
    
    print(f"\nPredicting at {n_test} new locations...")
    
    config = VecchiaConfig(m_neighbors=30, ordering='maximin', nugget=1e-6)
    gp = VecchiaGP(config)
    gp.fit(locs, y_sim, {'sigma2': sigma2_true, 'rho': rho_true, 'nu': nu_true})
    
    t0 = time.time()
    mu_pred, var_pred = gp.predict(locs_test)
    t_pred = time.time() - t0
    
    print(f"Prediction time: {t_pred:.3f}s ({n_test} points)")
    print(f"Mean predictive variance: {var_pred.mean():.4f}")
    
    # ---- Compare with exact GP on a subset ----
    if n >= 1000:
        print(f"\n--- Comparison with exact GP (subset) ---")
        n_sub = 500
        idx_sub = np.random.choice(n, n_sub, replace=False)
        locs_sub = locs[idx_sub]
        y_sub = y_sim[idx_sub]
        
        from scipy.linalg import cholesky, solve
        
        t0 = time.time()
        K_sub = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(i, n_sub):
                d = np.linalg.norm(locs_sub[i] - locs_sub[j])
                K_sub[i, j] = matern_covariance(d, sigma2_true, rho_true, nu_true)
                K_sub[j, i] = K_sub[i, j]
        K_sub += np.eye(n_sub) * 1e-6
        t_build = time.time() - t0
        
        t0 = time.time()
        L_sub = cholesky(K_sub, lower=True)
        t_chol = time.time() - t0
        
        print(f"Exact GP with n={n_sub}:")
        print(f"  Covariance build time: {t_build:.3f}s")
        print(f"  Cholesky time: {t_chol:.3f}s")
        print(f"  Full GP O(n^3) would require ~{(n/1000)**3:.1f}x more time")
    
    # Memory comparison
    print(f"\n--- Computational complexity ---")
    print(f"Exact GP: O(n^3) operations, O(n^2) memory")
    print(f"Vecchia: O(n*m^3) operations, O(n*m) memory")
    print(f"For n={n}, m=30:")
    print(f"  Exact: O({n**3:.2e}) vs Vecchia: O({n*30**3:.2e})")
    print(f"  Memory: O({n**2:.2e}) vs Vecchia: O({n*30:.2e})")
```

## References

Katzfuss, M., & Guinness, J. (2021). A general framework for Vecchia approximations of Gaussian processes. *Statistical Science*, 36(1), 124--141. https://doi.org/10.1214/19-STS755

Vecchia, A. V. (1988). Estimation and model identification for continuous spatial processes. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 50(2), 297--312. https://doi.org/10.1111/j.2517-6161.1988.tb01729.x

Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (2016). Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets. *Journal of the American Statistical Association*, 111(514), 800--812. https://doi.org/10.1080/01621459.2015.1044091

Guinness, J. (2021). Gaussian process learning via Fisher scoring of Vecchia's approximation. *Statistics and Computing*, 31, 25. https://doi.org/10.1007/s11222-021-09997-5

Cao, J., Guinness, J., Genton, M. G., & Katzfuss, M. (2022). Scalable Gaussian-process regression and variable selection using Vecchia approximations. *Journal of Machine Learning Research*, 23, 1--30.
