# Deep Instrumental Variables (DeepIV) and Neural IV Methods

**Source**: Hartford, J., Lewis, G., Leyton-Brown, K., & Taddy, M. (2017). Deep IV: A flexible approach for counterfactual prediction. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 1414–1423. https://proceedings.mlr.press/v70/hartford17a.html

**Source**: Xu, L., Chen, Y., Srinivasan, S., de Freitas, N., Doucet, A., & Gretton, A. (2021). Learning deep features in instrumental variable regression. *Proceedings of the 9th International Conference on Learning Representations (ICLR)*.

**Source**: Liu, R., Shang, Z., & Cheng, G. (2024). On deep instrumental variables estimate. *Journal of Econometrics*, 240(2), 105691. https://doi.org/10.1016/j.jeconom.2024.105691

**Category**: Causal Inference / Instrumental Variables / Deep Learning

## Mathematical Setup

Instrumental variables (IV) methods address endogeneity: when a confounder affects both the treatment $D$ and the outcome $Y$, making standard regression inconsistent. An instrument $Z$ satisfies:

1. **Relevance**: $Z$ predicts $D$ (conditional on $X$)
2. **Exclusion**: $Z$ affects $Y$ only through $D$
3. **Unconfounded instrument**: $Z$ is independent of any confounders of $D$ and $Y$

### Structural Model

Consider the nonparametric IV model:

$$Y = f(D, X) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid Z, X] = 0$$

where $X$ are exogenous covariates, $D$ is endogenous treatment, $Z$ are instruments, and $f$ is a (potentially nonlinear) structural function.

The target estimand is the structural function $f(d, x)$, which encodes the causal effect of setting $D = d$ on $Y$:

$$\mathbb{E}[Y \mid \text{do}(D = d), X = x] = f(d, x)$$

### DeepIV Architecture (Hartford et al., 2017)

DeepIV decomposes IV estimation into two supervised learning stages:

**Stage 1 (Treatment Network)**: Estimate the conditional distribution of $D$ given $Z$ and $X$:

$$F_{D \mid X, Z}(d \mid x, z) = \mathbb{P}(D \leq d \mid X = x, Z = z)$$

This can be a mixture density network, a conditional normalizing flow, or a simpler model.

**Stage 2 (Outcome Network)**: Minimize the loss:

$$\min_{f \in \mathcal{F}} \sum_{i=1}^n \left(Y_i - \mathbb{E}_{D \sim \hat{F}(\cdot \mid X_i, Z_i)}[f(D, X_i)]\right)^2$$

The key insight is that the expectation over $D$ in Stage 2 uses the estimated distribution from Stage 1, and the gradient propagates back through this expectation to update $f$.

### Deep Feature IV (Xu et al., 2021)

DFIV learns feature representations for both stages using deep neural networks and alternates between:

**Stage A**: Learn features $\phi(D, X)$ such that $\mathbb{E}[Y \mid D, X] \approx \beta^\top \phi(D, X)$

**Stage B**: Learn features $\psi(Z, X)$ such that $\mathbb{E}[\phi(D, X) \mid Z, X] \approx W^\top \psi(Z, X)$

The alternating scheme ensures that the features capture the IV structure.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Relevance | $Cov(Z, D) \neq 0$ | Instrument predicts treatment |
| Exclusion | $Y(d) \perp Z \mid X$ | No direct effect of $Z$ on $Y$ |
| Unconfounded instrument | $Z \perp\!\!\!\perp \varepsilon \mid X$ | No confounding of $Z-Y$ relationship |
| Completeness | $\mathbb{E}[g(D, X) \mid Z, X] = 0 \implies g \equiv 0$ | Nonparametric identification (stronger condition) |
| Smoothness | $f$ is sufficiently smooth | Deep network can approximate the function |

## Applicable Scenarios

**When to use:**
- Endogenous treatment with observed instruments
- High-dimensional or complex instruments (text, images, sensor data)
- Nonlinear relationships among instruments, treatments, and outcomes
- Off-policy policy evaluation (framing Bellman residuals as IV)
- Settings where two-stage least squares (2SLS) is misspecified

**When NOT to use:**
- When instruments are weak (Lowest eigenvalue of $\mathbb{E}[ZZ^\top]$ near zero)
- Very small datasets ($n < 500$ with high-dimensional instruments)
- When exclusion restriction is plausibly violated
- When the treatment is high-dimensional and instruments are few

## Method Details

### DeepIV Procedure

**Step 1: Treatment distribution estimation.** Train a model $\hat{F}_{D \mid X, Z}$ using a flexible neural density estimator (mixture density network or normalizing flow):

$$\hat{F} = \arg\max_{F \in \mathcal{F}} \sum_{i=1}^n \log p(D_i \mid X_i, Z_i; \theta)$$

**Step 2: Structural function estimation.** Train a second network $\hat{f}(D, X)$ by minimizing:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \sum_{i=1}^n \left(Y_i - \frac{1}{S} \sum_{s=1}^S f(\tilde{D}_{is}, X_i) \right)^2$$

where $\tilde{D}_{is} \sim \hat{F}(\cdot \mid X_i, Z_i)$ are Monte Carlo samples from the Stage 1 model.

**Step 3: Counterfactual prediction.** For a target point $(d, x)$:

$$\hat{Y}(d, x) = \hat{f}(d, x)$$

### Asymptotic Properties (Liu et al., 2024)
- Stage 1 achieves minimax optimal rates under a compositional structure assumption on the optimal instrument
- Stage 2 achieves $\sqrt{n}$-consistency and semiparametric efficiency when $p$ grows polynomially in $n$
- Weaker smoothness requirements than series/spline-based IV

## Implementation Details

**Key hyperparameters:**
- Stage 1: Architecture for conditional density estimation (number of mixture components)
- Stage 2: Architecture for structural function $f(\cdot, \cdot)$
- Number of Monte Carlo samples $S$ for the expectation (typical: 20-50)
- Learning rate, regularization strength for both stages

**Numerical considerations:**
- Training two stages jointly can be unstable; alternating optimization is preferred
- Causal validation: use held-out data not seen by either stage to tune hyperparameters
- Dropout can provide approximate Bayesian inference at test time

**Available software:**
- Python: `econml` (DeepIV wrapper), PyTorch implementations in various repositories
- R: `deepiv` (experimental)

## Python Implementation

```python
"""
Deep Instrumental Variable (DeepIV) Regression

Implements a two-stage deep learning approach for IV regression
with neural networks.

References:
    Hartford et al. (2017). Deep IV: A flexible approach for
        counterfactual prediction. ICML.
    Xu et al. (2021). Learning deep features in instrumental
        variable regression. ICLR.
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.mixture import GaussianMixture


class DeepIV(BaseEstimator, RegressorMixin):
    """Deep Instrumental Variable (DeepIV) Estimator.

    Two-stage neural network approach for nonparametric IV regression.
    Stage 1: Model the conditional distribution of D given X, Z.
    Stage 2: Model the structural function Y = f(D, X).

    Parameters
    ----------
    stage1_model : estimator or str, default='gmm'
        Model for D | X, Z. Options: 'gmm' (Gaussian Mixture Model),
        or a custom estimator with .fit(XZ) and .sample(n_samples).
    stage2_model : estimator, default=MLPRegressor
        Neural network or other model for f(D, X).
    n_components : int, default=5
        Number of mixture components (if stage1='gmm').
    n_mc_samples : int, default=50
        Number of Monte Carlo samples for the expectation in Stage 2.
    random_state : int, default=42
    """

    def __init__(self, stage1_model='gmm', stage2_model=None,
                 n_components=5, n_mc_samples=50, random_state=42):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model or MLPRegressor(
            hidden_layer_sizes=(64, 32, 16), activation='relu',
            max_iter=1000, early_stopping=True, random_state=random_state)
        self.n_components = n_components
        self.n_mc_samples = n_mc_samples
        self.random_state = random_state

    def fit(self, Z, X, D, Y):
        """Fit the DeepIV model.

        Parameters
        ----------
        Z : array-like, shape (n_samples, n_instruments)
            Instruments.
        X : array-like, shape (n_samples, n_covariates)
            Exogenous covariates.
        D : array-like, shape (n_samples, n_treatments)
            Endogenous treatment.
        Y : array-like, shape (n_samples,)
            Outcome.

        Returns
        -------
        self : DeepIV
        """
        Z = np.asarray(Z)
        X = np.asarray(X)
        D = np.asarray(D)
        Y = np.asarray(Y).ravel()

        # Concatenate features for Stage 1
        XZ = np.column_stack([X, Z])

        # Stage 1: Model D | X, Z
        if isinstance(self.stage1_model, str) and self.stage1_model == 'gmm':
            if D.ndim == 1:
                D = D.reshape(-1, 1)
            XZ_D = np.column_stack([D, XZ])

            # Fit a Gaussian Mixture Model on (D, X, Z)
            self.gmm_ = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                random_state=self.random_state,
                max_iter=200
            )
            self.gmm_.fit(XZ_D)

            self.cond_dist_model_ = 'gmm'
        else:
            self.stage1_model.fit(XZ, D)
            self.cond_dist_model_ = 'custom'

        # Stage 2: Train f(D, X) to minimize IV loss
        # We use a two-step procedure:
        #   Step A: Estimate E[D | X, Z] as a simple summary (for initialization)
        #   Step B: Train f(D, X) minimizing ||Y - E[f(D, X) | Z, X]||

        # Get conditional treatment expectations via Monte Carlo
        n = len(Y)
        DX = np.column_stack([D, X])

        self.stage2_model_ = clone(self.stage2_model)

        # Pre-train: set up the stage2 model shape
        # (fits on initial estimate, then we do IV loss)
        # Actually implement the IV loss via MC sampling:

        # We'll use a simple iterative scheme:
        # 1. Initialize f by regressing Y on (D, X) directly (biased but a start)
        f_init = clone(self.stage2_model)
        f_init.fit(DX, Y)

        # 2. Refine f by minimizing the IV criterion
        #    E[Y - E[f(D, X) | Z, X]]^2
        #    For the Monte Carlo approximation to work well, we need
        #    to be able to sample from D | X, Z.

        f_current = f_init
        n_iter = 5
        DX_all = DX
        self.D_dim_ = D.shape[1] if D.ndim > 1 else 1

        for iteration in range(n_iter):
            # Generate MC samples: for each observation, draw D~ from D | X_i, Z_i
            D_samples = self._sample_conditional(X, Z)
            # Shape: (n_samples, n_mc_samples, D_dim)

            # Compute conditional expectation: E[f(D, X) | Z, X]
            f_expectations = np.zeros(n)
            for i in range(n):
                f_vals = np.zeros(self.n_mc_samples)
                for s in range(self.n_mc_samples):
                    dx_s = np.column_stack([D_samples[i, s].reshape(1, -1),
                                            X[i:i+1]])
                    f_vals[s] = f_current.predict(dx_s)[0]
                f_expectations[i] = np.mean(f_vals)

            # Update f: minimize MSE between Y and conditional expectation
            # This is the "IV loss"
            new_f = clone(self.stage2_model)
            # Compute pseudo-targets for f (workaround for the nested expectation)
            pseudo_Y = Y - f_expectations + f_current.predict(DX_all)
            new_f.fit(DX_all, pseudo_Y)
            f_current = new_f

        self.stage2_model_ = f_current
        self.D_dim_ = D.shape[1] if D.ndim > 1 else 1

        return self

    def _sample_conditional(self, X, Z):
        """Sample from D | X, Z using the fitted Stage 1 model."""
        n = len(X)
        D_samples = np.zeros((n, self.n_mc_samples, self.D_dim_))

        if self.cond_dist_model_ == 'gmm':
            # Use conditional Gaussian sampling
            for i in range(n):
                xz_i = np.column_stack([X[i:i+1], Z[i:i+1]])[0]
                samples = self._conditional_sample_gmm(
                    xz_i, n_samples=self.n_mc_samples)
                D_samples[i] = samples
        else:
            for i in range(n):
                xz_i = np.column_stack([X[i:i+1], Z[i:i+1]])
                D_samples[i, :, 0] = self.stage1_model.sample(
                    self.n_mc_samples)

        return D_samples

    def _conditional_sample_gmm(self, xz, n_samples=50):
        """Sample from D | X=x, Z=z using the fitted GMM.

        For a GMM on (D, X, Z), the conditional D | X, Z is also a
        Gaussian mixture with updated component weights and means.
        """
        gmm = self.gmm_
        n_components = gmm.weights_.shape[0]
        d_dim = self.D_dim_
        xz_dim = len(xz)

        # Means: split into D part and XZ part
        weights = gmm.weights_
        means = gmm.means_
        covs = gmm.covariances_

        # For each component, compute conditional distribution of D | XZ
        cond_means = np.zeros((n_components, d_dim))
        cond_covs = np.zeros((n_components, d_dim, d_dim))
        log_resp = np.zeros(n_components)

        for k in range(n_components):
            mu_k = means[k]
            sigma_k = covs[k] if covs[k].ndim == 2 else np.diag(covs[k])

            mu_d = mu_k[:d_dim]
            mu_xz = mu_k[d_dim:]

            sigma_dd = sigma_k[:d_dim, :d_dim]
            sigma_dxz = sigma_k[:d_dim, d_dim:]
            sigma_xz_xz = sigma_k[d_dim:, d_dim:]
            sigma_xz_xz_inv = np.linalg.pinv(sigma_xz_xz)

            cond_means[k] = mu_d + sigma_dxz @ sigma_xz_xz_inv @ (xz - mu_xz)
            cond_covs[k] = sigma_dd - sigma_dxz @ sigma_xz_xz_inv @ sigma_dxz.T

            # Component weight (responsibility)
            from scipy.stats import multivariate_normal
            log_resp[k] = np.log(weights[k] + 1e-10) + \
                multivariate_normal.logpdf(xz, mean=mu_xz, cov=sigma_xz_xz + 1e-6 * np.eye(xz_dim))

        # Normalize responsibilities
        log_resp -= log_resp.max()
        resp = np.exp(log_resp) / np.exp(log_resp).sum()

        # Sample from the conditional mixture
        samples = np.zeros((n_samples, d_dim))
        for s in range(n_samples):
            comp = np.random.choice(n_components, p=resp)
            sample = np.random.multivariate_normal(cond_means[comp],
                                                   cond_covs[comp] + 1e-6 * np.eye(d_dim))
            samples[s] = sample

        return samples

    def predict(self, D_new, X_new):
        """Predict counterfactual outcome Y under Do(D = D_new).

        Parameters
        ----------
        D_new : array-like, shape (n_samples, n_treatments)
            Counterfactual treatment value.
        X_new : array-like, shape (n_samples, n_covariates)
            Covariates.

        Returns
        -------
        Y_hat : ndarray, shape (n_samples,)
            Predicted potential outcomes.
        """
        D_new = np.asarray(D_new)
        X_new = np.asarray(X_new)
        if D_new.ndim == 1:
            D_new = D_new.reshape(-1, 1)
        DX = np.column_stack([D_new, X_new])
        return self.stage2_model_.predict(DX)


def clone(estimator):
    """Clone a sklearn estimator."""
    from sklearn.base import clone as sk_clone
    return sk_clone(estimator)


def simulate_iv_data(n=2000, seed=42):
    """Simulate data from a nonlinear IV model.

    DGP:
        Z ~ N(0, 1)              (instrument)
        X1, X2 ~ N(0, 1)         (exogenous covariates)
        V ~ N(0, 1)              (unobserved confounder)
        D = 0.5 * Z + 0.3 * X1 + V + N(0, 0.5)
        Y = sin(D) + 0.5 * X1^2 + 0.3 * X2 + V + N(0, 0.3)
    """
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, 1)
    X = np.column_stack([rng.randn(n), rng.randn(n)])
    V = rng.randn(n)

    D = 0.5 * Z[:, 0] + 0.3 * X[:, 0] + V + 0.5 * rng.randn(n)
    Y = np.sin(D) + 0.5 * X[:, 0]**2 + 0.3 * X[:, 1] + V + 0.3 * rng.randn(n)

    return Z, X, D.reshape(-1, 1), Y


if __name__ == "__main__":
    print("=" * 60)
    print("Deep Instrumental Variables (DeepIV)")
    print("=" * 60)

    Z, X, D, Y = simulate_iv_data(n=1500, seed=42)
    print(f"Data: n = {len(Y)}, #instruments = {Z.shape[1]}, "
          f"#covariates = {X.shape[1]}")

    # Fit DeepIV
    deepiv = DeepIV(stage1_model='gmm', n_components=5,
                    n_mc_samples=30, random_state=42)
    deepiv.fit(Z, X, D, Y)

    # Evaluate on a grid of counterfactual D values
    print("\n--- Counterfactual predictions ---")
    d_grid = np.linspace(-3, 3, 7)
    x_fixed = np.zeros((1, 2))

    print("D     | E[Y | do(D), X=0]")
    print("-" * 25)
    for d_val in d_grid:
        d_arr = np.array([[d_val]])
        y_pred = deepiv.predict(d_arr, np.zeros((1, 2)))[0]
        true_fn = np.sin(d_val)  # True structural function at X=0
        print(f"{d_val:+.1f}  | {y_pred:.4f} (true: {true_fn:.4f})")

    # MSE of structural function at X = (0, 0)
    mse_fn = np.mean([
        (deepiv.predict(np.array([[d]]), np.zeros((1, 2)))[0] - np.sin(d))**2
        for d in d_grid
    ])
    print(f"\nMean squared error of structural function: {mse_fn:.4f}")

    # Comparison with naive OLS
    print("\n--- Comparison with Naive OLS ---")
    ols = LinearRegression()
    ols.fit(np.column_stack([D, X]), Y)
    print(f"OLS estimate (D coefficient, likely biased): "
          f"{ols.coef_[0]:.4f}")
    print(f"True structural function is nonlinear (sin(D)), "
          f"so OLS captures a linearized effect only.")

    print("\n--- Placebo check: predict at D = Stage 1 mean ---")
    d_mean = np.mean(D)
    y_at_mean = deepiv.predict(np.array([[d_mean]]), np.zeros((1, 2)))[0]
    print(f"Prediction at mean D = {d_mean:.2f}: {y_at_mean:.4f}")
```

## References

Hartford, J., Lewis, G., Leyton-Brown, K., & Taddy, M. (2017). Deep IV: A flexible approach for counterfactual prediction. *Proceedings of the 34th International Conference on Machine Learning*, 1414–1423.

Xu, L., Chen, Y., Srinivasan, S., de Freitas, N., Doucet, A., & Gretton, A. (2021). Learning deep features in instrumental variable regression. *Proceedings of the 9th International Conference on Learning Representations*.

Liu, R., Shang, Z., & Cheng, G. (2024). On deep instrumental variables estimate. *Journal of Econometrics*, 240(2), 105691. https://doi.org/10.1016/j.jeconom.2024.105691

Bennett, A., Kallus, N., & Schnabel, T. (2019). Deep generalized method of moments for instrumental variable analysis. *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 3564–3574.

Singh, R., Sahani, M., & Gretton, A. (2019). Kernel instrumental variable regression. *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 4595–4607.
```

