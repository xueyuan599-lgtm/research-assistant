# CATE Meta-Learners: X-Learner, R-Learner, and DR-Learner

**Source**: Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116

**Source**: Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076

**Source**: Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049. https://doi.org/10.1214/23-EJS2157

**Category**: Causal Inference / Heterogeneous Treatment Effects / Meta-Learners

## Mathematical Setup

Meta-learners are modular estimation strategies for the Conditional Average Treatment Effect (CATE) that allow any supervised learning method to be used as a "base learner" in each component step.

### Potential Outcomes and CATE

Under the potential outcomes framework, the CATE is:

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

where $Y(1), Y(0)$ are potential outcomes and $X$ is the covariate vector. With unconfoundedness $(Y(0), Y(1)) \perp D \mid X$ and overlap $0 < \pi(x) < 1$, we have:

$$\tau(x) = \mathbb{E}[Y \mid X = x, D = 1] - \mathbb{E}[Y \mid X = x, D = 0]$$

### S-Learner (Single-Learner)

Fits a single model $\mu(x, d) = \mathbb{E}[Y \mid X = x, D = d]$ and computes:

$$\hat{\tau}_S(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$$

Simple but may fail to detect heterogeneity if $\mu(x, d)$ is dominated by the main effect of $X$.

### T-Learner (Two-Learner)

Fits separate outcome models for treated and control groups:

$$\hat{\mu}_1(x) = \mathbb{E}[Y \mid X = x, D = 1], \quad \hat{\mu}_0(x) = \mathbb{E}[Y \mid X = x, D = 0]$$

$$\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

Each group uses only its own data; may be inefficient if one group is small.

### X-Learner (Cross-Learner) — Kunzel et al. (2019)

1. Estimate $\hat{\mu}_0(x)$ and $\hat{\mu}_1(x)$ as in T-learner
2. Impute individual treatment effects:
   - For treated units: $\tilde{\tau}_i^1 = Y_i - \hat{\mu}_0(X_i)$
   - For control units: $\tilde{\tau}_i^0 = \hat{\mu}_1(X_i) - Y_i$
3. Fit CATE models on the imputed effects: $\hat{\tau}_1(x)$ from $\tilde{\tau}^1$, $\hat{\tau}_0(x)$ from $\tilde{\tau}^0$
4. Combine: $\hat{\tau}_X(x) = g(x) \hat{\tau}_0(x) + (1 - g(x)) \hat{\tau}_1(x)$

where $g(x)$ is typically the propensity score $\pi(x)$.

### R-Learner (Residual-Learner) — Nie & Wager (2021)

The R-learner exploits the Robinson decomposition:

$$Y_i - m(X_i) = \tau(X_i) (D_i - \pi(X_i)) + \varepsilon_i$$

where $m(x) = \mathbb{E}[Y \mid X = x]$ and $\pi(x) = \mathbb{E}[D \mid X = x]$. Estimation proceeds as:

1. Estimate $\hat{m}(x)$ and $\hat{\pi}(x)$ via cross-fitting
2. Compute residuals: $\tilde{Y}_i = Y_i - \hat{m}(X_i)$, $\tilde{D}_i = D_i - \hat{\pi}(X_i)$
3. Minimize the R-loss:

$$\hat{\tau}_R = \arg\min_{\tau \in \mathcal{T}} \frac{1}{n} \sum_{i=1}^n \left(\tilde{Y}_i - \tau(X_i) \tilde{D}_i \right)^2 + \Lambda(\tau)$$

where $\Lambda(\tau)$ is a regularizer. The R-learner achieves **quasi-oracle** rates: even with imperfect nuisance estimates, the CATE estimator converges at the oracle rate.

### DR-Learner (Doubly Robust Learner) — Kennedy (2023)

1. Estimate $\hat{\mu}_0(x)$, $\hat{\mu}_1(x)$, $\hat{\pi}(x)$ via cross-fitting
2. Construct pseudo-outcomes:

$$\phi_i = \frac{D_i - \hat{\pi}(X_i)}{\hat{\pi}(X_i)(1 - \hat{\pi}(X_i))} (Y_i - \hat{\mu}_{D_i}(X_i)) + \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)$$

3. Regress $\phi_i$ on $X_i$: $\hat{\tau}_{DR}(x) = \mathbb{E}[\phi \mid X = x]$

The DR-learner is **doubly robust**: it is consistent for $\tau(x)$ if either the propensity model or the outcome models are correctly specified. Kennedy (2023) proved it achieves **oracle efficiency** and is minimax optimal up to log factors.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Unconfoundedness | $(Y(1), Y(0)) \perp D \mid X$ | All confounders observed |
| Overlap | $0 < \pi(x) < 1$ | Common support |
| Nuisance convergence | Product rates $o_p(n^{-1/4})$ for R/DR learners | Allows flexible ML |
| Smoothness | $\tau(x)$ belongs to a Holder class | Determines nonparametric rate |

## Applicable Scenarios

**When to use each learner:**
- **S-Learner**: Quick baseline, low-dimensional $X$, treatment effect is simple
- **T-Learner**: When treatment and control groups have very different outcome mechanisms
- **X-Learner**: Best for unbalanced treatment assignment (e.g., rare treatment)
- **R-Learner**: General purpose, strong theoretical guarantees, Neyman-orthogonal
- **DR-Learner**: When either propensity or outcome model might be misspecified (double protection)

**When NOT to use:**
- S-learner with regularized models: may shrink CATE toward zero
- T-learner with highly imbalanced treatment groups
- R-learner when the treatment effect is nearly zero everywhere (R-loss may overfit)
- Any learner without cross-fitting for the nuisance functions

## Implementation Details

**Key hyperparameters:**
- Choice of base learner (regression method) for each stage
- Cross-fitting folds (typically $K = 5$)
- Regularization parameters for the final CATE regression

**Available software:**
- Python: `econml` (Microsoft), `causalml` (Uber), `CATENets` (Curth & van der Schaar)
- R: `grf` (causal forest), `metalearner` (dedicated package), `causalToolbox`

## Python Implementation

```python
"""
CATE Meta-Learners: S-Learner, T-Learner, X-Learner, R-Learner, DR-Learner

References:
    Kunzel et al. (2019). Metalearners for estimating heterogeneous
        treatment effects. PNAS, 116(10), 4156-4165.
    Nie & Wager (2021). Quasi-oracle estimation of heterogeneous
        treatment effects. Biometrika, 108(2), 299-319.
    Kennedy (2023). Towards optimal doubly robust estimation of
        heterogeneous causal effects. EJS, 17(2), 3008-3049.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def _fresh_model(base_learner):
    """Create a fresh unfitted copy of a model.

    Uses direct reconstruction to avoid sklearn clone issues with __len__.
    """
    if base_learner is None:
        return None
    try:
        return base_learner.__class__(**base_learner.get_params(deep=True))
    except Exception:
        # Fallback: for models with simple constructors
        return base_learner.__class__()


class SLearner:
    """Single-Learner (S-Learner) for CATE estimation.

    Fits a single model E[Y | X, D] with treatment as a feature.
    """
    def __init__(self, base_learner=None, random_state=42):
        self.base_learner = base_learner if base_learner is not None else RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=random_state)
        self.random_state = random_state

    def fit(self, X, D, Y):
        X_aug = np.column_stack([X, D])
        self.model_ = _fresh_model(self.base_learner).fit(X_aug, Y)
        return self

    def predict(self, X):
        X1 = np.column_stack([X, np.ones(len(X))])
        X0 = np.column_stack([X, np.zeros(len(X))])
        return self.model_.predict(X1) - self.model_.predict(X0)


class TLearner:
    """Two-Learner (T-Learner) for CATE estimation.

    Fits separate models for treated and control groups.
    """
    def __init__(self, base_learner=None, random_state=42):
        self.base_learner = base_learner if base_learner is not None else RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=random_state)
        self.random_state = random_state

    def fit(self, X, D, Y):
        self.model0_ = _fresh_model(self.base_learner).fit(X[D == 0], Y[D == 0])
        self.model1_ = _fresh_model(self.base_learner).fit(X[D == 1], Y[D == 1])
        return self

    def predict(self, X):
        return self.model1_.predict(X) - self.model0_.predict(X)


class XLearner:
    """Cross-Learner (X-Learner) for CATE estimation.

    Imputes individual treatment effects and combines via
    propensity-weighted averaging.

    Parameters
    ----------
    base_learner : estimator, optional
        Base model for outcome regression and CATE regression.
    propensity_learner : estimator, optional
        Model for propensity score estimation.
    random_state : int, default=42
    """
    def __init__(self, base_learner=None, propensity_learner=None,
                 random_state=42):
        self.base_learner = base_learner if base_learner is not None else RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=random_state)
        self.propensity_learner = propensity_learner if propensity_learner is not None else LogisticRegression(
            random_state=random_state)
        self.random_state = random_state

    def fit(self, X, D, Y):
        # Step 1: Estimate outcome regressions
        self.model0_ = _fresh_model(self.base_learner).fit(X[D == 0], Y[D == 0])
        self.model1_ = _fresh_model(self.base_learner).fit(X[D == 1], Y[D == 1])

        # Step 2: Impute individual treatment effects
        tau_hat1 = Y[D == 1] - self.model0_.predict(X[D == 1])
        tau_hat0 = self.model1_.predict(X[D == 0]) - Y[D == 0]

        # Step 3: Fit CATE models on imputed effects
        self.cate_model1_ = _fresh_model(self.base_learner).fit(X[D == 1], tau_hat1)
        self.cate_model0_ = _fresh_model(self.base_learner).fit(X[D == 0], tau_hat0)

        # Step 4: Estimate propensity score for weighting
        self.propensity_ = _fresh_model(self.propensity_learner).fit(X, D)

        return self

    def predict(self, X):
        pi = self.propensity_.predict_proba(X)[:, 1]
        tau0 = self.cate_model0_.predict(X)
        tau1 = self.cate_model1_.predict(X)
        return pi * tau0 + (1 - pi) * tau1


class RLearner:
    """Residual-Learner (R-Learner) for CATE estimation.

    Uses the Robinson decomposition: residualize both outcome and
    treatment w.r.t. covariates, then regress residual outcome on
    residual treatment weighted by CATE.

    Parameters
    ----------
    model_cate : estimator, optional
        Model for the CATE function tau(x).
    model_m : estimator, optional
        Model for the outcome regression E[Y | X].
    model_pi : estimator, optional
        Model for the propensity score E[D | X].
    n_folds : int, default=5
        Number of cross-fitting folds.
    random_state : int, default=42
    """
    def __init__(self, model_cate=None, model_m=None, model_pi=None,
                 n_folds=5, random_state=42):
        default_rf = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=random_state)
        default_rf_pi = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, random_state=random_state)
        self.model_cate = model_cate if model_cate is not None else default_rf
        self.model_m = model_m if model_m is not None else default_rf
        self.model_pi = model_pi if model_pi is not None else default_rf_pi
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, D, Y):
        X = np.asarray(X)
        D = np.asarray(D).ravel()
        Y = np.asarray(Y).ravel()
        n = len(Y)

        # Cross-fitting for nuisance functions
        cv = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)

        Y_resid = np.empty(n)
        D_resid = np.empty(n)

        for train_idx, test_idx in cv.split(X):
            X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y[train_idx]
            X_te, D_te, Y_te = X[test_idx], D[test_idx], Y[test_idx]

            m_hat = _fresh_model(self.model_m).fit(X_tr, Y_tr)
            pi_hat = _fresh_model(self.model_pi).fit(X_tr, D_tr)

            Y_resid[test_idx] = Y_te - m_hat.predict(X_te)
            D_resid[test_idx] = D_te - pi_hat.predict(X_te)

        # Fit CATE by minimizing weighted R-loss
        # (Y_resid - tau(X) * D_resid)^2
        # Via re-weighting: weight = D_resid^2, target = Y_resid / D_resid
        # (with clipping to avoid division by zero)

        epsilon = 1e-8
        weights = D_resid**2
        target = Y_resid / (np.clip(D_resid, -np.inf, -epsilon) +
                            np.clip(D_resid, epsilon, np.inf))

        # Use a weighted regression to minimize R-loss
        self.cate_model_ = _fresh_model(self.model_cate)
        self.cate_model_.fit(X, target, sample_weight=weights)

        self.Y_resid_ = Y_resid
        self.D_resid_ = D_resid

        return self

    def predict(self, X):
        return self.cate_model_.predict(np.asarray(X))


class DRLearner:
    """Doubly Robust Learner (DR-Learner) for CATE estimation.

    Constructs doubly robust pseudo-outcomes and regresses them
    on covariates to estimate CATE.

    Parameters
    ----------
    model_cate : estimator, optional
        Model for the last-stage CATE regression.
    model_m0 : estimator, optional
        Outcome model for control group.
    model_m1 : estimator, optional
        Outcome model for treated group.
    model_pi : estimator, optional
        Propensity score model.
    n_folds : int, default=5
    random_state : int, default=42
    """
    def __init__(self, model_cate=None, model_m0=None, model_m1=None,
                 model_pi=None, n_folds=5, random_state=42):
        default_rf = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=random_state)
        default_pi = LogisticRegression(random_state=random_state)
        self.model_cate = model_cate if model_cate is not None else default_rf
        self.model_m0 = model_m0 if model_m0 is not None else default_rf
        self.model_m1 = model_m1 if model_m1 is not None else default_rf
        self.model_pi = model_pi if model_pi is not None else default_pi
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, D, Y):
        X = np.asarray(X)
        D = np.asarray(D).ravel()
        Y = np.asarray(Y).ravel()
        n = len(Y)

        # Cross-fitting
        cv = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)

        pseudo_outcomes = np.empty(n)

        for train_idx, test_idx in cv.split(X):
            X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y[train_idx]
            X_te, D_te, Y_te = X[test_idx], D[test_idx], Y[test_idx]

            m0 = _fresh_model(self.model_m0).fit(X_tr[D_tr == 0], Y_tr[D_tr == 0])
            m1 = _fresh_model(self.model_m1).fit(X_tr[D_tr == 1], Y_tr[D_tr == 1])
            pi = _fresh_model(self.model_pi).fit(X_tr, D_tr)

            mu0_te = m0.predict(X_te)
            mu1_te = m1.predict(X_te)
            pi_te = pi.predict_proba(X_te)[:, 1]

            # DR pseudo-outcome
            pseudo_outcomes[test_idx] = (
                mu1_te - mu0_te
                + (D_te - pi_te) / (pi_te * (1 - pi_te) + 1e-8)
                * (Y_te - np.where(D_te == 1, mu1_te, mu0_te))
            )

        # Regress pseudo-outcomes on covariates
        self.cate_model_ = _fresh_model(self.model_cate).fit(X, pseudo_outcomes)
        self.pseudo_outcomes_ = pseudo_outcomes

        return self

    def predict(self, X):
        return self.cate_model_.predict(np.asarray(X))


def simulate_cate_data(n=2000, p=5, seed=42):
    """Simulate data with heterogeneous treatment effects.

    True CATE: tau(x) = 1 + 0.5 * x0 + sin(x1)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    tau = 1.0 + 0.5 * X[:, 0] + np.sin(X[:, 1])

    # Propensity (confounded)
    pi = 1 / (1 + np.exp(-(0.3 * X[:, 0] - 0.5 * X[:, 1])))
    D = rng.binomial(1, pi)

    # Outcome
    g = np.sin(X[:, 0]) + 0.3 * X[:, 1]**2 - 0.5 * X[:, 2]
    Y = tau * D + g + 0.3 * rng.randn(n)

    return X, D, Y, tau


def evaluate_cate(true_tau, pred_tau, name):
    """Evaluate CATE estimation accuracy."""
    mse = np.mean((pred_tau - true_tau)**2)
    corr = np.corrcoef(pred_tau, true_tau)[0, 1]
    return f"{name:>12s}: MSE = {mse:.4f}, Corr = {corr:.4f}"


if __name__ == "__main__":
    print("=" * 65)
    print("CATE Meta-Learners — Comparative Evaluation")
    print("=" * 65)

    X, D, Y, true_tau = simulate_cate_data(n=2000, p=5, seed=42)
    print(f"\nSimulated data: n={len(Y)}, p={X.shape[1]}")

    base_learner = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=10, random_state=42)

    learners = {
        "S-Learner": SLearner(base_learner),
        "T-Learner": TLearner(base_learner),
        "X-Learner": XLearner(base_learner),
        "R-Learner": RLearner(random_state=42),
        "DR-Learner": DRLearner(random_state=42),
    }

    print(f"\n--- CATE Estimation Performance ---")
    results = {}
    for name, learner in learners.items():
        learner.fit(X, D, Y)
        pred = learner.predict(X)
        results[name] = pred
        print(evaluate_cate(true_tau, pred, name))

    # Report best performer
    best_name = min(results.keys(),
                    key=lambda n: np.mean((results[n] - true_tau)**2))
    print(f"\nBest performer: {best_name}")

    # Average Treatment Effect
    print(f"\n--- Average Treatment Effect (ATE) ---")
    print(f"True ATE: {np.mean(true_tau):.4f}")
    for name, pred in results.items():
        print(f"  {name:>12s}: {np.mean(pred):.4f}")
```

## References

Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116

Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076

Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049. https://doi.org/10.1214/23-EJS2157

Curth, A., & van der Schaar, M. (2021). Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms. *Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 1810–1818.

Kunzel, S. R., Walter, S. J., & Sekhon, J. S. (2018). Causaltoolbox: A Python package for causal inference. *Journal of Open Source Software*, 3(26), 725. https://doi.org/10.21105/joss.00725
```

