# Conformal Prediction Beyond Exchangeability

**Source**: Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *The Annals of Statistics*, 51(2), 816--845. https://doi.org/10.1214/23-AOS2276

**Category**: Statistics / Uncertainty Quantification / Conformal Inference

## Mathematical Setup

Conformal prediction constructs prediction sets with finite-sample marginal coverage guarantees under exchangeability. Let $(X_i, Y_i)_{i=1}^{n}$ be calibration data and $(X_{n+1}, Y_{n+1})$ a test point, all exchangeable. Define a nonconformity score $V_i = s(X_i, Y_i) \in \mathbb{R}$ measuring how unusual $(X_i, Y_i)$ appears relative to the fitted model. Under exchangeability, the ranks of the scores are uniform, yielding the standard split conformal prediction set:

$$
C_n(x_{n+1}) = \{y : V_{n+1} \leq Q_{1-\alpha}(V_1, \ldots, V_n)\}
$$

where $Q_{1-\alpha}$ is the $(1-\alpha)$-quantile of the calibration scores, guaranteeing $P(Y_{n+1} \in C_n(X_{n+1})) \geq 1-\alpha$.

Barber et al. (2023) generalize this to settings where exchangeability fails. The core idea is to assign weights $w_i$ to each calibration point reflecting its "relevance" to the test point. Let $w = (w_1, \ldots, w_n, w_{n+1})$ be nonnegative weights, and define the **weighted conformal prediction set**:

$$
C_n(x_{n+1}) = \left\{ y : \sum_{i=1}^{n} w_i \cdot \mathbb{1}\{V_i \geq V_{n+1}\} + w_{n+1} \cdot \tau > (1-\alpha) \sum_{i=1}^{n+1} w_i \right\}
$$

where $\tau \sim \text{Unif}[0,1]$ is a random tie-breaking variable.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Bounded weights | $\sum_{i=1}^{n+1} w_i^2 / (\sum_{i=1}^{n+1} w_i)^2 \to 0$ | Coverage converges to $1-\alpha$ as $n \to \infty$ |
| Weighted exchangeability | For permutation $\pi$ fixing test index, $w_{\pi(i)} V_i$ are distributionally identical up to weights | Generalizes exchangeability with known importance weights |
| Consistency of scores | The joint distribution of $(V_1, \ldots, V_{n+1})$ satisfies a weighted exchangeability condition with respect to $w$ | Required for exact finite-sample coverage |

The key theoretical result is a bound on the coverage gap:

$$
\left| P(Y_{n+1} \in C_n(X_{n+1})) - (1-\alpha) \right| \leq \frac{\sum_{i=1}^n w_i \cdot d_{\text{TV}}(R(Z), R(Z_i))}{1 + \sum_{i=1}^n w_i}
$$

where $d_{\text{TV}}$ measures total variation distance between score distributions, and $R(Z)$ denotes the distribution of the joint score vector.

## Applicable Scenarios

- **When to use**: Time series forecasting with distribution drift, online learning with concept shift, covariate shift (test points differ from calibration), A/B testing with non-stationary environments.
- **When NOT to use**: When exchangeability is expected to hold exactly (standard conformal prediction is simpler and tighter), when weights cannot be reasonably specified, or when extreme distribution shifts make all calibration points irrelevant.
- **Comparison with classical alternatives**: Standard split conformal prediction fails under distribution shift. Bootstrap and asymptotic methods rely on parametric assumptions. Weighted conformal prediction provides finite-sample guarantees under weaker conditions.

## Method Details

1. **Score computation**: Fit a model $\hat{\mu}$ on training data. Compute nonconformity scores $V_i = |Y_i - \hat{\mu}(X_i)|$ for calibration data.
2. **Weight assignment**: Choose weights $w_i$ reflecting the relevance of calibration point $i$ to the test point. Common choices: inverse propensity weights for covariate shift, exponential decay weights for time series, uniform weights for exchangeable data.
3. **Weighted quantile computation**: Compute the weighted $(1-\alpha)$-quantile of the scores.
4. **Prediction set**: Include all $y$ values whose score does not exceed the weighted quantile.
5. **Randomized tie-breaking**: If needed, add a uniform random variable $\tau$ to handle ties.

**Theoretical guarantees**:
- **Finite-sample marginal coverage** under weighted exchangeability: $P(Y_{n+1} \in C_n(X_{n+1})) \geq 1 - \alpha$
- **Asymptotic coverage** under distribution drift: coverage gap vanishes as $n \to \infty$ if weights are chosen appropriately
- **Adaptivity**: The method automatically adapts to the severity of the distribution shift

## Implementation Details

- **Key parameters**: Significance level $\alpha$, weight function $w(\cdot)$, nonconformity score $s(\cdot, \cdot)$
- **Computational considerations**: $O(n)$ for score computation and quantile search, dominated by model fitting
- **Software availability**: R package `conformalInference.fd`; Python implementations in `mapie` and `crepes`

## Python Implementation

```python
import numpy as np
from typing import Callable, Optional, Tuple
from sklearn.base import RegressorMixin


class WeightedConformalPredictor:
    """Weighted conformal prediction for non-exchangeable settings.
    
    Implements the method of Barber, Candes, Ramdas & Tibshirani (2023)
    for constructing prediction intervals with finite-sample coverage
    guarantees even when exchangeability is violated.
    
    Parameters
    ----------
    model : sklearn RegressorMixin
        A fitted regression model (must have .predict() method).
    alpha : float, default=0.1
        Nominal miscoverage level (1 - alpha is the target coverage).
    weight_function : Callable, default=None
        Function w(X_calib, X_test) -> array of weights for calibration points.
        If None, uses uniform weights (standard exchangeable conformal).
    randomize : bool, default=True
        Whether to use randomized tie-breaking (ensures exact 1-alpha coverage
        under weighted exchangeability).
    """
    
    def __init__(
        self,
        model: RegressorMixin,
        alpha: float = 0.1,
        weight_function: Optional[Callable] = None,
        randomize: bool = True,
    ):
        self.model = model
        self.alpha = alpha
        self.weight_function = weight_function or (lambda X_cal, X_test: np.ones(len(X_cal)))
        self.randomize = randomize
        self._calibrated = False
        
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate using calibration data.
        
        Parameters
        ----------
        X_cal : ndarray of shape (n_cal, d)
            Calibration features.
        y_cal : ndarray of shape (n_cal,)
            Calibration responses.
        """
        self.X_cal = X_cal
        self.y_cal = y_cal
        y_pred = self.model.predict(X_cal)
        self.scores_ = np.abs(y_cal - y_pred)  # absolute residual nonconformity score
        self._calibrated = True
        
    def predict(
        self, X_test: np.ndarray, return_interval: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct weighted conformal prediction intervals.
        
        Parameters
        ----------
        X_test : ndarray of shape (n_test, d)
            Test features.
        return_interval : bool, default=True
            If True, return (lower, upper) bounds; otherwise return only
            the weighted quantile threshold.
            
        Returns
        -------
        lower : ndarray of shape (n_test,)
            Lower prediction bounds (if return_interval=True).
        upper : ndarray of shape (n_test,)
            Upper prediction bounds (if return_interval=True).
        Or threshold : ndarray of shape (n_test,)
            Weighted quantile threshold for each test point.
        """
        if not self._calibrated:
            raise RuntimeError("Must call .calibrate() before .predict()")
        
        n_cal = len(self.scores_)
        thresholds = np.zeros(len(X_test))
        
        for t, x_t in enumerate(X_test):
            # Compute weights for this test point
            weights = self.weight_function(self.X_cal, x_t.reshape(1, -1))
            weights = np.asarray(weights, dtype=float).ravel()
            
            # Ensure nonnegative weights
            weights = np.maximum(weights, 0.0)
            if weights.sum() == 0:
                weights = np.ones(n_cal)  # fallback to uniform
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Compute weighted quantile threshold
            # Equivalent to solving: sum(weights * I(score <= thr)) >= 1 - alpha
            sorted_idx = np.argsort(self.scores_)
            sorted_scores = self.scores_[sorted_idx]
            sorted_weights = weights[sorted_idx]
            
            cumsum = np.cumsum(sorted_weights)
            idx = np.searchsorted(cumsum, 1.0 - self.alpha)
            
            if idx < len(sorted_scores):
                threshold = sorted_scores[idx]
            else:
                threshold = sorted_scores[-1]
                
            # Randomized tie-breaking for exact coverage
            if self.randomize:
                # Find all scores equal to threshold
                at_boundary = np.isclose(self.scores_, threshold)
                w_boundary = weights[at_boundary].sum()
                if w_boundary > 0:
                    excess = cumsum[min(idx, len(cumsum)-1)] - (1.0 - self.alpha)
                    if excess > 0 and np.random.rand() > (w_boundary - excess) / w_boundary:
                        # Find next distinct score above threshold
                        distinct_scores = np.unique(sorted_scores)
                        next_idx = np.searchsorted(distinct_scores, threshold) + 1
                        if next_idx < len(distinct_scores):
                            threshold = distinct_scores[next_idx]
            
            thresholds[t] = threshold
        
        # Construct intervals
        y_pred_test = self.model.predict(X_test)
        lower = y_pred_test - thresholds
        upper = y_pred_test + thresholds
        
        if return_interval:
            return lower, upper
        return thresholds


# ============================================================
# Example: Exponential decay weights for time series drift
# ============================================================

def exponential_decay_weights(
    X_cal: np.ndarray, X_test: np.ndarray, decay_rate: float = 0.01
) -> np.ndarray:
    """Weight calibration points by temporal proximity.
    
    Later calibration points receive higher weight.
    Useful when the data generating process drifts over time.
    
    Parameters
    ----------
    X_cal : ndarray of shape (n_cal, d)
        Calibration features (ordered by time).
    X_test : ndarray of shape (1, d)
        Single test point.
    decay_rate : float
        Exponential decay rate for temporal weighting.
        
    Returns
    -------
    weights : ndarray of shape (n_cal,)
        Exponential decay weights.
    """
    n_cal = len(X_cal)
    # Assign higher weight to more recent observations
    t = np.arange(n_cal)
    weights = np.exp(decay_rate * t)
    return weights


# ============================================================
# Example usage with simulated drifting data
# ============================================================

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # Generate data with distribution drift
    n_train, n_cal, n_test = 200, 200, 200
    n_total = n_train + n_cal + n_test
    
    # Time-varying mean: distribution drifts over time
    t = np.linspace(0, 2 * np.pi, n_total)
    X = np.column_stack([np.sin(t), np.cos(t)])
    
    # Response: linear in X with time-varying coefficients (drift)
    beta = np.column_stack([
        1.0 + 0.5 * np.sin(t / 5),   # beta_0 drifts
        0.5 + 0.3 * np.cos(t / 4),   # beta_1 drifts
    ])
    y = np.sum(X * beta, axis=1) + 0.5 * np.random.randn(n_total)
    
    # Split into train/cal/test (chronologically ordered)
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:n_train + n_cal], y[n_train:n_train + n_cal]
    X_test, y_test = X[n_train + n_cal:], y[n_train + n_cal:]
    
    # Fit a simple model (misspecified for drift)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # ---- Standard conformal (exchangeable assumption) ----
    std_predictor = WeightedConformalPredictor(
        model=model, alpha=0.1,
        weight_function=lambda Xc, Xt: np.ones(len(Xc)),
        randomize=True,
    )
    std_predictor.calibrate(X_cal, y_cal)
    lower_std, upper_std = std_predictor.predict(X_test)
    
    # ---- Weighted conformal (with drift-aware weights) ----
    weighted_predictor = WeightedConformalPredictor(
        model=model, alpha=0.1,
        weight_function=exponential_decay_weights,
        randomize=True,
    )
    weighted_predictor.calibrate(X_cal, y_cal)
    lower_wcp, upper_wcp = weighted_predictor.predict(X_test)
    
    # ---- Evaluate empirical coverage ----
    cover_std = np.mean((y_test >= lower_std) & (y_test <= upper_std))
    cover_wcp = np.mean((y_test >= lower_wcp) & (y_test <= upper_wcp))
    width_std = np.mean(upper_std - lower_std)
    width_wcp = np.mean(upper_wcp - lower_wcp)
    
    print("=== Conformal Prediction Beyond Exchangeability ===")
    print(f"Target coverage: 90%")
    print(f"Standard conformal (exchangeable assumption):")
    print(f"  Empirical coverage: {cover_std:.1%}, Avg width: {width_std:.3f}")
    print(f"Weighted conformal (drift-aware):")
    print(f"  Empirical coverage: {cover_wcp:.1%}, Avg width: {width_wcp:.3f}")
    
    # Baseline: Gaussian quantile
    residuals = y_train - model.predict(X_train)
    gaussian_threshold = 1.645 * np.std(residuals)  # 90% if normal
    lower_g = model.predict(X_test) - gaussian_threshold
    upper_g = model.predict(X_test) + gaussian_threshold
    cover_g = np.mean((y_test >= lower_g) & (y_test <= upper_g))
    print(f"Gaussian quantile (naive):")
    print(f"  Empirical coverage: {cover_g:.1%}, Avg width: {width_std:.3f}")
    
    # Plot results for a segment
    plt.figure(figsize=(12, 5))
    idx_plot = range(50)
    t_plot = np.arange(len(idx_plot))
    plt.plot(t_plot, y_test[:50], 'k.', label='True values', alpha=0.7)
    plt.plot(t_plot, model.predict(X_test[:50]), 'r-', label='Predicted', alpha=0.8)
    plt.fill_between(t_plot, lower_std[:50], upper_std[:50], 
                     alpha=0.3, color='gray', label='Standard CP')
    plt.fill_between(t_plot, lower_wcp[:50], upper_wcp[:50], 
                     alpha=0.3, color='blue', label='Weighted CP')
    plt.legend()
    plt.xlabel('Test index (time order)')
    plt.ylabel('Response')
    plt.title('Weighted Conformal Prediction Under Distribution Drift')
    plt.savefig('weighted_conformal_demo.png', dpi=100, bbox_inches='tight')
    plt.show()
```

## References

Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *The Annals of Statistics*, 51(2), 816--845. https://doi.org/10.1214/23-AOS2276

Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint*. https://arxiv.org/abs/2107.07511

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer.

Tibshirani, R. J., Barber, R. F., Candes, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *Advances in Neural Information Processing Systems*, 32.

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094--1111. https://doi.org/10.1080/01621459.2017.1307116
