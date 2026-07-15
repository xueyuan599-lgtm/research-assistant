# Double/Debiased Machine Learning (DML)

**Source**: Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097

**Category**: Causal Inference / Neyman-Orthogonal Estimation with Machine Learning

## Mathematical Setup

DML provides a framework for estimating causal parameters when the nuisance functions (propensity score, outcome regression) are estimated using high-dimensional or nonparametric machine learning methods. The key innovation is **Neyman orthogonality**: the moment condition used to estimate the causal parameter is insensitive to small perturbations in the nuisance estimates.

### Partially Linear Regression (PLR) Model

The canonical setting assumes:

$$Y = \theta_0 D + g_0(X) + U, \quad \mathbb{E}[U \mid X, D] = 0$$
$$D = m_0(X) + V, \quad \mathbb{E}[V \mid X] = 0$$

where:
- $Y$ is the outcome
- $D$ is the treatment
- $X$ are confounding covariates
- $\theta_0$ is the target causal parameter (ATE)
- $g_0(X)$ is the nuisance outcome regression
- $m_0(X) = \mathbb{E}[D \mid X]$ is the propensity score

The **orthogonal score** for $\theta_0$ is:

$$\psi(W; \theta, \eta) = (Y - \theta D - g(X))(D - m(X))$$

where $\eta = (g, m)$ are nuisance functions. This score satisfies the Neyman orthogonality condition:

$$\partial_{\eta} \mathbb{E}[\psi(W; \theta_0, \eta_0)] = 0$$

### Target Estimand

$$\theta_0 = \mathbb{E}[(D - \mathbb{E}[D \mid X])^{-1} (Y - \mathbb{E}[Y \mid X])]$$

i.e., the coefficient on $D$ after partialling out $X$ from both $Y$ and $D$.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Unconfoundedness | $\mathbb{E}[U \mid X, D] = 0$ | No unmeasured confounders |
| Overlap | $0 < \pi(x) = \mathbb{P}(D=1 \mid X=x) < 1$ | Treatment probability bounded away from 0 and 1 |
| Cross-fitting | Sample is split into $K$ folds; nuisance estimated on out-of-fold | Prevents overfitting bias |
| Nuisance convergence | $\|\hat{g} - g_0\|_2 \times \|\hat{m} - m_0\|_2 = o_p(n^{-1/2})$ | Product rate must beat $n^{-1/2}$; individual rates can be slow |

## Applicable Scenarios

**When to use:**
- High-dimensional covariates ($p > n$ or large $p$)
- Complex, nonlinear confounding that requires ML for adequate adjustment
- Settings where standard doubly robust estimators would suffer from regularization bias
- Binary, continuous, or multiple treatments
- Instrumental variable settings (DML-IV)

**When NOT to use:**
- When $n$ is very small (a few hundred or less)
- When overlap is severely violated
- When no good ML method exists for the nuisance functions
- When interpretability of the first-stage model is essential

## Method Details

### Step-by-Step Estimation (DML with Cross-Fitting)

**Step 1: Cross-fitting split.** Partition the sample into $K$ folds (typically $K = 5$).

**Step 2: Nuisance estimation.** For each fold $k$:
- Estimate $\hat{m}_{-k}(X)$ (propensity/treatment model) on all data except fold $k$
- Estimate $\hat{g}_{-k}(X)$ (outcome regression) on all data except fold $k$

**Step 3: Residual computation.** For observations in fold $k$:
- $\tilde{Y} = Y - \hat{g}_{-k}(X)$ (outcome residuals)
- $\tilde{D} = D - \hat{m}_{-k}(X)$ (treatment residuals)

**Step 4: Parameter estimation.** Regress $\tilde{Y}$ on $\tilde{D}$ (pooled across all folds):

$$\hat{\theta} = \frac{\sum_i \tilde{D}_i \tilde{Y}_i}{\sum_i \tilde{D}_i^2}$$

**Inference.** The estimator $\hat{\theta}$ is approximately $\mathcal{N}(\theta_0, \hat{\sigma}^2/n)$ where:

$$\hat{\sigma}^2 = \frac{1}{n} \sum_i \left(\hat{\psi}(W_i; \hat{\theta}, \hat{\eta})\right)^2 \bigg/ \left(\frac{1}{n} \sum_i \tilde{D}_i^2\right)^2$$

### Asymptotic Properties
- **Consistency**: $\hat{\theta} \xrightarrow{p} \theta_0$
- **Root-n normality**: $\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$
- **Semiparametric efficiency**: The variance attains the semiparametric efficiency bound under appropriate conditions

## Implementation Details

**Key hyperparameters:**
- Number of cross-fitting folds ($K = 5$ is default)
- ML method for nuisance estimation (choice affects finite-sample performance)
- Whether to use PLR, IV, or other DML variants

**Numerical considerations:**
- Cross-fitting is essential; do not use the same data for nuisance estimation and parameter estimation
- The product-rate condition $\|\hat{g} - g_0\| \times \|\hat{m} - m_0\| = o_p(n^{-1/2})$ allows flexible ML but still requires reasonable convergence

**Available software:**
- Python: [`econml`](https://econml.azurewebsites.net/) by Microsoft Research
- R: [`DoubleML`](https://docs.doubleml.org/) package
- Stata: `ddml` command

## Python Implementation

```python
"""
Double/Debiased Machine Learning (DML) for Partially Linear Regression

References:
    Chernozhukov et al. (2018). Double/debiased machine learning for treatment
    and structural parameters. The Econometrics Journal, 21(1), C1-C68.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold


class DMLPartiallyLinear:
    """Double/Debiased Machine Learning for the PLR model.

    Parameters
    ----------
    model_y : estimator object, default=RandomForestRegressor()
        ML estimator for outcome regression E[Y | X].
        Must implement .fit() and .predict().
    model_d : estimator object, default=RandomForestRegressor()
        ML estimator for treatment/propensity model E[D | X].
        Must implement .fit() and .predict().
    n_folds : int, default=5
        Number of folds for cross-fitting.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : float
        Estimated treatment effect theta.
    stderr_ : float
        Standard error of theta.
    pvalue_ : float
        Two-sided p-value for H0: theta = 0.
    ci_ : tuple
        95% confidence interval.
    """
    def __init__(self, model_y=None, model_d=None,
                 n_folds=5, random_state=42):
        self.model_y = model_y or RandomForestRegressor(
            n_estimators=500, min_samples_leaf=10, random_state=random_state)
        self.model_d = model_d or RandomForestRegressor(
            n_estimators=500, min_samples_leaf=10, random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, D, Y):
        """Estimate the treatment effect theta via DML with cross-fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Covariates.
        D : array-like, shape (n_samples,)
            Treatment assignment (binary or continuous).
        Y : array-like, shape (n_samples,)
            Outcome.

        Returns
        -------
        self : DMLPartiallyLinear
        """
        X = np.asarray(X)
        D = np.asarray(D).ravel()
        Y = np.asarray(Y).ravel()
        n = len(Y)

        # Cross-fitting split
        cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Store residuals
        Y_resid = np.empty(n)
        D_resid = np.empty(n)

        for train_idx, test_idx in cv.split(X):
            X_train, D_train, Y_train = X[train_idx], D[train_idx], Y[train_idx]
            X_test, D_test, Y_test = X[test_idx], D[test_idx], Y[test_idx]

            # Estimate nuisance functions on training fold
            model_y = clone(self.model_y).fit(X_train, Y_train)
            model_d = clone(self.model_d).fit(X_train, D_train)

            # Compute residuals on held-out fold
            Y_hat_test = model_y.predict(X_test)
            D_hat_test = model_d.predict(X_test)

            Y_resid[test_idx] = Y_test - Y_hat_test
            D_resid[test_idx] = D_test - D_hat_test

        # Parameter estimation via OLS on residuals
        theta = np.sum(D_resid * Y_resid) / np.sum(D_resid**2)
        residuals = Y_resid - theta * D_resid

        # Variance estimation (sandwich formula)
        n = len(Y)
        var_score = np.mean((D_resid * residuals)**2)
        var_hessian = np.mean(D_resid**2)
        var_theta = var_score / (n * var_hessian**2)

        se = np.sqrt(var_theta)
        t_stat = theta / se
        p_value = 2 * (1 - _normal_cdf(np.abs(t_stat)))

        self.coef_ = theta
        self.stderr_ = se
        self.tstat_ = t_stat
        self.pvalue_ = p_value
        self.ci_ = (theta - 1.96 * se, theta + 1.96 * se)

        return self


def _normal_cdf(x):
    """Standard normal CDF using the error function."""
    from scipy import special
    return 0.5 * (1 + special.erf(x / np.sqrt(2)))


def simulate_data(n=2000, p=10, seed=42):
    """Simulate data from the PLR model.

    Y = theta * D + g0(X) + U
    D = m0(X) + V

    Returns
    -------
    X : ndarray, shape (n, p)
    D : ndarray, shape (n,)
    Y : ndarray, shape (n,)
    true_theta : float
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)

    # Nuisance functions (nonlinear)
    g0 = np.sin(X[:, 0]) + np.abs(X[:, 1]) + 0.5 * X[:, 2]**2
    m0 = 0.5 * X[:, 0] + 0.3 * np.cos(X[:, 1]) + 0.2 * X[:, 2]

    true_theta = 1.5

    V = rng.randn(n) * 0.5
    U = rng.randn(n) * 0.5

    D = m0 + V
    Y = true_theta * D + g0 + U

    return X, D, Y, true_theta


if __name__ == "__main__":
    print("=" * 60)
    print("DML Partially Linear Regression — Simulation")
    print("=" * 60)

    # Simulate data
    X, D, Y, true_theta = simulate_data(n=2000, p=5, seed=42)
    print(f"True treatment effect: theta = {true_theta:.3f}")

    # DML estimation
    dml = DMLPartiallyLinear(n_folds=5, random_state=42)
    dml.fit(X, D, Y)

    print(f"\nDML estimate:        theta = {dml.coef_:.4f}")
    print(f"Standard error:             SE = {dml.stderr_:.4f}")
    print(f"t-statistic:                t  = {dml.tstat_:.2f}")
    print(f"p-value:                    p  = {dml.pvalue_:.2e}")
    print(f"95% CI: ({dml.ci_[0]:.4f}, {dml.ci_[1]:.4f})")
    print(f"Coverage: {'YES' if dml.ci_[0] < true_theta < dml.ci_[1] else 'NO'}")

    # Simple OLS (biased due to nonlinear confounding)
    ols_coef = np.linalg.lstsq(
        np.column_stack([D, X[:, :3]]), Y, rcond=None)[0][0]
    print(f"\nNaive OLS estimate:  theta = {ols_coef:.4f}  (likely biased)")

    # Sensitivity check: vary random seed
    print("\n--- Sensitivity to Random Seed (DML) ---")
    results = []
    for seed in range(5):
        X_s, D_s, Y_s, _ = simulate_data(n=2000, p=5, seed=seed)
        dml_s = DMLPartiallyLinear(n_folds=5, random_state=seed)
        dml_s.fit(X_s, D_s, Y_s)
        results.append(dml_s.coef_)
    print(f"Mean DML estimate across seeds: {np.mean(results):.4f}")
    print(f"Std DML estimate across seeds:  {np.std(results):.4f}")
```

## References

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097

Clarke, P., & Polselli, A. (2026). Double machine learning for static panel models with fixed effects. *The Econometrics Journal*, 29(1), 77–98. https://doi.org/10.1093/ectj/utae029

Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076

Bach, P., Chernozhukov, V., Kurz, M., & Spindler, M. (2021). DoubleML -- An object-oriented implementation of double machine learning in R. *Journal of Statistical Software*, 102(8), 1–26. https://doi.org/10.18637/jss.v102.i08

Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049. https://doi.org/10.1214/23-EJS2157
```

