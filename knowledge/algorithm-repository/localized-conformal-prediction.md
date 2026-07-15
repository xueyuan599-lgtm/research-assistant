# Localized Conformal Prediction

**Source**: Guan, L. (2023). Localized conformal prediction: a generalized inference framework for conformal prediction. *Biometrika*, 110(1), 33--50. https://doi.org/10.1093/biomet/asac040

**Category**: Statistics / Uncertainty Quantification / Conformal Inference

## Mathematical Setup

Standard conformal prediction provides marginal coverage guarantees: $P(Y_{n+1} \in C_n(X_{n+1})) \geq 1-\alpha$, but conditional coverage $P(Y_{n+1} \in C_n(X_{n+1}) \mid X_{n+1} = x)$ can be far from $1-\alpha$ in regions with different data characteristics. Localized conformal prediction (Guan, 2023) addresses this by constructing prediction intervals that adapt to the local structure of the feature space.

Let $\{(X_i, Y_i)\}_{i=1}^n$ be calibration data and $V_i = s(X_i, Y_i)$ a nonconformity score. The standard prediction interval is:

$$
C_n(x) = \{y : V_{n+1} \leq Q_{1-\alpha}(V_1, \ldots, V_n)\}
$$

Localized conformal prediction assigns **localized weights** $w_i(x)$ to each calibration point based on its proximity to the test point $x$ in feature space. The localized prediction interval is:

$$
C_n^{\text{loc}}(x) = \left\{y : \sum_{i=1}^n w_i(x) \mathbb{1}\{V_i \geq V_{n+1}\} + \tau \cdot w_{n+1}(x) > (1-\alpha) \sum_{i=1}^{n+1} w_i(x) \right\}
$$

where $\tau \sim \text{Unif}[0,1]$ handles ties. A natural choice for the weights uses a kernel function $K(\cdot)$:

$$
w_i(x) = K\left(\frac{\|X_i - x\|}{h}\right)
$$

with bandwidth $h > 0$. This yields prediction intervals that are **localized** to the neighborhood of $x$ while maintaining the finite-sample marginal coverage guarantee.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Local exchangeability | For fixed $x$, $(X_i, Y_i)$ exchangeable with $(X_{n+1}, Y_{n+1})$ in a neighborhood of $x$ | Exact finite-sample local coverage |
| Kernel regularity | $K(\cdot)$ is nonnegative, bounded, and $K(u) \to 0$ as $\|u\| \to \infty$ | Localization is well-defined |
| Density continuity | The feature density $f_X(\cdot)$ is continuous and bounded away from 0 | Asymptotic conditional coverage |
| Smooth regression function | $E[Y \mid X = x]$ and $Var[Y \mid X = x]$ are Lipschitz | Localization accuracy |
| Bandwidth decay | $h \to 0$ and $nh^d \to \infty$ as $n \to \infty$ | Consistent conditional coverage |

If the weights are chosen such that each calibration point's contribution captures its relevance to $x$, the localized procedure guarantees:

$$
P(Y_{n+1} \in C_n^{\text{loc}}(X_{n+1})) \geq 1 - \alpha
$$

and under additional smoothness conditions, the conditional coverage converges to $1-\alpha$:

$$
P(Y_{n+1} \in C_n^{\text{loc}}(X_{n+1}) \mid X_{n+1} = x) \to 1 - \alpha \quad \text{as } n \to \infty
$$

## Applicable Scenarios

- **When to use**: Heteroscedastic regression where prediction uncertainty varies with $x$, multi-modal conditional distributions, data with varying signal-to-noise ratios across feature space, any setting where conditional coverage matters more than marginal.
- **When NOT to use**: Very low sample sizes ($n \ll 1000$), high-dimensional feature spaces ($d > 10$ without dimension reduction), settings where marginal coverage is sufficient (standard CP is simpler).
- **Comparison with classical alternatives**: Standard conformal prediction provides only marginal coverage. Quantile regression forests provide conditional quantiles but lack finite-sample guarantees. Localized CP combines finite-sample coverage guarantees with improved conditional coverage.

## Method Details

1. **Choice of nonconformity score**: Standard absolute residual $V_i = |Y_i - \hat{\mu}(X_i)|$ or more sophisticated scores.
2. **Kernel choice**: Gaussian kernel $K(u) = \exp(-u^2/(2h^2))$, uniform kernel, or adaptive nearest-neighbor kernel.
3. **Bandwidth selection**: Cross-validation to balance bias and variance, or use $k$-nearest neighbor distance for adaptive bandwidth.
4. **Weighted interval construction**: For each test point $x$, compute weights $w_i(x)$, then find the weighted $(1-\alpha)$-quantile of calibration scores.
5. **Tie-breaking**: Use randomized tie-breaking for exact coverage at discrete score values.

**Theoretical guarantees**:
- **Finite-sample marginal coverage**: $P(Y_{n+1} \in C_n^{\text{loc}}(X_{n+1})) \geq 1-\alpha$ unconditionally
- **Asymptotic conditional coverage**: Under smoothness, $P(Y_{n+1} \in C_n^{\text{loc}}(X_{n+1}) \mid X_{n+1} = x) \to 1-\alpha$
- **Adaptivity**: Interval widths automatically adapt to local variability, producing wider intervals in high-noise regions

## Implementation Details

- **Key parameters**: Significance level $\alpha$, kernel bandwidth $h$ (or $k$ for nearest-neighbor), kernel type
- **Computational considerations**: $O(n)$ per test point for naive implementation, $O(\log n)$ with KD-tree or ball-tree acceleration
- **Software availability**: R implementation in the supplementary material of Guan (2023)

## Python Implementation

```python
import numpy as np
from typing import Callable, Optional, Tuple
from sklearn.base import RegressorMixin
from sklearn.neighbors import NearestNeighbors


class LocalizedConformalPredictor:
    """Localized conformal prediction (Guan, 2023, Biometrika).
    
    Constructs prediction intervals that adapt to local structure
    in the feature space while maintaining finite-sample coverage
    guarantees.
    
    Parameters
    ----------
    model : sklearn RegressorMixin
        Fitted regression model with .predict() method.
    alpha : float, default=0.1
        Nominal miscoverage level.
    kernel : str or Callable, default='gaussian'
        Kernel function for localization. Options: 'gaussian', 
        'uniform', 'epanechnikov', or a custom Callable.
    bandwidth : float or str, default='auto'
        Kernel bandwidth. If 'auto', uses Scott's rule or
        k-nearest neighbor heuristic.
    n_neighbors : int, default=None
        Number of nearest neighbors for adaptive bandwidth.
        If None, uses global bandwidth.
    randomize : bool, default=True
        Whether to use randomized tie-breaking.
    """
    
    def __init__(
        self,
        model: RegressorMixin,
        alpha: float = 0.1,
        kernel: str = 'gaussian',
        bandwidth: float = 'auto',
        n_neighbors: Optional[int] = None,
        randomize: bool = True,
    ):
        self.model = model
        self.alpha = alpha
        self.kernel_name = kernel
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.randomize = randomize
        self._calibrated = False
        
        # Set kernel function
        if isinstance(kernel, str):
            self.kernel_func = self._get_kernel(kernel)
        else:
            self.kernel_func = kernel
    
    @staticmethod
    def _get_kernel(name: str) -> Callable:
        """Return kernel function by name."""
        kernels = {
            'gaussian': lambda u: np.exp(-0.5 * u ** 2),
            'uniform': lambda u: np.where(np.abs(u) <= 1.0, 0.5, 0.0),
            'epanechnikov': lambda u: np.where(
                np.abs(u) <= 1.0, 0.75 * (1 - u ** 2), 0.0
            ),
        }
        if name not in kernels:
            raise ValueError(f"Unknown kernel: {name}. Options: {list(kernels.keys())}")
        return kernels[name]
    
    def _compute_bandwidth(
        self, X: np.ndarray
    ) -> float:
        """Compute bandwidth using Scott's rule."""
        n, d = X.shape
        if isinstance(self.bandwidth, (int, float)):
            return float(self.bandwidth)
        
        # Scott's rule of thumb
        h = n ** (-1.0 / (d + 4))
        
        # Scale by median pairwise distance
        if n > 100:
            idx = np.random.choice(n, min(100, n), replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
        
        # Estimate characteristic length scale
        dists = []
        for i in range(min(20, len(X_sub))):
            for j in range(i + 1, min(20, len(X_sub))):
                dists.append(np.linalg.norm(X_sub[i] - X_sub[j]))
        
        if dists:
            scale = np.median(dists)
            h *= scale if scale > 0 else 1.0
        
        return h
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate using calibration data.
        
        Parameters
        ----------
        X_cal : ndarray of shape (n_cal, d)
            Calibration features.
        y_cal : ndarray of shape (n_cal,)
            Calibration responses.
        """
        self.X_cal = np.asarray(X_cal)
        self.y_cal = np.asarray(y_cal)
        n_cal = len(X_cal)
        
        # Compute nonconformity scores
        y_pred = self.model.predict(X_cal)
        self.scores_ = np.abs(y_cal - y_pred)
        
        # Fit nearest neighbors for adaptive bandwidth
        if self.n_neighbors is not None:
            self.nn_ = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, n_cal),
                metric='euclidean'
            )
            self.nn_.fit(self.X_cal)
        
        # Precompute bandwidth
        if self.bandwidth == 'auto':
            self.bandwidth_ = self._compute_bandwidth(self.X_cal)
        else:
            self.bandwidth_ = float(self.bandwidth)
        
        self._calibrated = True
    
    def _compute_weights(
        self, x_test: np.ndarray
    ) -> np.ndarray:
        """Compute localized weights for a test point.
        
        If n_neighbors is set, uses adaptive bandwidth equal to
        distance to the k-th nearest neighbor.
        """
        dists = np.linalg.norm(self.X_cal - x_test, axis=1)
        
        if self.n_neighbors is not None:
            # Adaptive bandwidth: distance to k-th nearest neighbor
            k = min(self.n_neighbors, len(self.X_cal) - 1)
            sorted_dists = np.sort(dists)
            h_local = sorted_dists[k] + 1e-10
        else:
            h_local = self.bandwidth_
        
        # Compute kernel weights
        u = dists / h_local
        weights = self.kernel_func(u)
        
        return weights
    
    def predict(
        self, X_test: np.ndarray, return_interval: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct localized conformal prediction intervals.
        
        Parameters
        ----------
        X_test : ndarray of shape (n_test, d)
            Test features.
        return_interval : bool, default=True
            If True, return (lower, upper) bounds.
            
        Returns
        -------
        lower : ndarray of shape (n_test,)
            Lower prediction bounds.
        upper : ndarray of shape (n_test,)
            Upper prediction bounds.
        """
        if not self._calibrated:
            raise RuntimeError("Must call .calibrate() before .predict()")
        
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        n_cal = len(self.scores_)
        thresholds = np.zeros(len(X_test))
        
        for t, x_t in enumerate(X_test):
            # Compute localized weights
            weights = self._compute_weights(x_t.ravel())
            
            # Ensure nonnegative and handle zeros
            weights = np.maximum(weights, 0.0)
            w_sum = weights.sum()
            if w_sum == 0:
                weights = np.ones(n_cal) / n_cal
            else:
                weights = weights / w_sum
            
            # Compute weighted quantile
            sorted_idx = np.argsort(self.scores_)
            sorted_scores = self.scores_[sorted_idx]
            sorted_weights = weights[sorted_idx]
            
            cumsum = np.cumsum(sorted_weights)
            idx = np.searchsorted(cumsum, 1.0 - self.alpha)
            
            if idx < len(sorted_scores):
                threshold = sorted_scores[idx]
            else:
                threshold = sorted_scores[-1]
            
            # Randomized tie-breaking
            if self.randomize:
                at_boundary = np.isclose(self.scores_, threshold)
                w_boundary = weights[at_boundary].sum()
                if w_boundary > 0:
                    cum_at_idx = cumsum[min(idx, len(cumsum) - 1)]
                    excess = cum_at_idx - (1.0 - self.alpha)
                    if excess > 0 and np.random.rand() > (w_boundary - excess) / w_boundary:
                        distinct = np.unique(sorted_scores)
                        next_idx = np.searchsorted(distinct, threshold) + 1
                        if next_idx < len(distinct):
                            threshold = distinct[next_idx]
            
            thresholds[t] = threshold
        
        # Construct intervals
        y_pred_test = self.model.predict(X_test)
        lower = y_pred_test - thresholds
        upper = y_pred_test + thresholds
        
        if return_interval:
            return lower, upper
        return thresholds


# ============================================================
# Example: Adaptive intervals for heteroscedastic data
# ============================================================

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # Generate heteroscedastic data: noise varies with X
    n_train, n_cal, n_test = 500, 500, 200
    
    X = np.random.uniform(-3, 3, (n_train + n_cal + n_test, 1))
    
    # Mean function: sinusoidal
    mu = np.sin(X).ravel()
    
    # Heteroscedastic noise: variance increases with |X|
    sigma = 0.2 + 0.3 * np.abs(X).ravel()
    y = mu + sigma * np.random.randn(len(X))
    
    # Split
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:n_train + n_cal], y[n_train:n_train + n_cal]
    X_test, y_test = X[n_train + n_cal:], y[n_train + n_cal:]
    
    # Train model
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    
    # ---- Standard conformal prediction ----
    std_cp = LocalizedConformalPredictor(
        model=rf, alpha=0.1,
        kernel='uniform', bandwidth=100.0,  # effectively global
        randomize=True,
    )
    std_cp.calibrate(X_cal, y_cal)
    lower_std, upper_std = std_cp.predict(X_test)
    
    # ---- Localized conformal prediction ----
    loc_cp = LocalizedConformalPredictor(
        model=rf, alpha=0.1,
        kernel='gaussian', bandwidth='auto',
        n_neighbors=100,
        randomize=True,
    )
    loc_cp.calibrate(X_cal, y_cal)
    lower_loc, upper_loc = loc_cp.predict(X_test)
    
    # ---- Evaluate ----
    cover_std = np.mean((y_test >= lower_std) & (y_test <= upper_std))
    cover_loc = np.mean((y_test >= lower_loc) & (y_test <= upper_loc))
    width_std = np.mean(upper_std - lower_std)
    width_loc = np.mean(upper_loc - lower_loc)
    
    print("=== Localized Conformal Prediction ===")
    print(f"Target coverage: 90%")
    print(f"\nStandard CP (marginal):")
    print(f"  Coverage: {cover_std:.1%}, Avg width: {width_std:.4f}")
    print(f"\nLocalized CP (adaptive):")
    print(f"  Coverage: {cover_loc:.1%}, Avg width: {width_loc:.4f}")
    
    # Conditional coverage by region
    print(f"\nConditional coverage by X region:")
    for region, mask in [("X < -1.5", X_test.ravel() < -1.5),
                         ("-1.5 < X < 1.5", np.abs(X_test.ravel()) < 1.5),
                         ("X > 1.5", X_test.ravel() > 1.5)]:
        if mask.sum() > 0:
            c_std = np.mean((y_test[mask] >= lower_std[mask]) & 
                          (y_test[mask] <= upper_std[mask]))
            c_loc = np.mean((y_test[mask] >= lower_loc[mask]) & 
                          (y_test[mask] <= upper_loc[mask]))
            print(f"  {region}:")
            print(f"    Standard CP: {c_std:.1%}")
            print(f"    Localized CP: {c_loc:.1%}")
    
    # Plot
    sort_idx = np.argsort(X_test.ravel())
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(X_test[sort_idx], y_test[sort_idx], 'k.', markersize=3, alpha=0.5)
    plt.fill_between(X_test[sort_idx].ravel(), 
                     lower_std[sort_idx], upper_std[sort_idx],
                     alpha=0.3, color='gray', label='Standard CP')
    plt.plot(X_test[sort_idx], rf.predict(X_test[sort_idx]), 'r-', linewidth=2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Standard Conformal Prediction')
    
    plt.subplot(1, 2, 2)
    plt.plot(X_test[sort_idx], y_test[sort_idx], 'k.', markersize=3, alpha=0.5)
    plt.fill_between(X_test[sort_idx].ravel(),
                     lower_loc[sort_idx], upper_loc[sort_idx],
                     alpha=0.3, color='steelblue', label='Localized CP')
    plt.plot(X_test[sort_idx], rf.predict(X_test[sort_idx]), 'r-', linewidth=2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Localized Conformal Prediction')
    
    plt.tight_layout()
    plt.savefig('localized_conformal_demo.png', dpi=100, bbox_inches='tight')
    plt.show()
```

## References

Guan, L. (2023). Localized conformal prediction: a generalized inference framework for conformal prediction. *Biometrika*, 110(1), 33--50. https://doi.org/10.1093/biomet/asac040

Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *The Annals of Statistics*, 51(2), 816--845. https://doi.org/10.1214/23-AOS2276

Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint*. https://arxiv.org/abs/2107.07511

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094--1111. https://doi.org/10.1080/01621459.2017.1307116

Papadopoulos, S. (2008). Inductive conformal prediction: Theory and application. *Neural Networks (IJCNN)*, 315--320. https://doi.org/10.1109/IJCNN.2008.4633824
