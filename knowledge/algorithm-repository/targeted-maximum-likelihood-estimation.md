# Targeted Maximum Likelihood Estimation (TMLE)

**Source**: van der Laan, M. J., & Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

**Source**: van der Laan, M. J., & Rose, S. (2018). *Targeted Learning in Data Science: Causal Inference for Complex Longitudinal Studies*. Springer. https://doi.org/10.1007/978-3-319-65304-4

**Source**: van der Laan, M. J., & Rubin, D. (2006). Targeted maximum likelihood learning. *The International Journal of Biostatistics*, 2(1). https://doi.org/10.2202/1557-4679.1043

**Category**: Causal Inference / Semiparametric Estimation / Targeted Learning

## Mathematical Setup

TMLE is a general semiparametric estimation framework that **targets** the estimation of a specific parameter of interest (e.g., the ATE) by constructing a **fluctuation** of an initial estimate of the data-generating distribution. The fluctuation is designed to reduce bias for the target parameter while preserving the nonparametric consistency of the initial estimator.

### Target Parameter: ATE

Consider a binary treatment $D \in \{0, 1\}$ and outcome $Y$, with covariates $X$. The ATE is:

$$\psi_0 = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[\mathbb{E}[Y \mid X, D = 1] - \mathbb{E}[Y \mid X, D = 0]]$$

### The Efficient Influence Curve (EIC)

For the ATE, the efficient influence curve is:

$$IC(O \mid \eta) = \frac{D - \pi(X)}{\pi(X)(1 - \pi(X))} (Y - \bar{Q}(D, X)) + \bar{Q}(1, X) - \bar{Q}(0, X) - \psi$$

where $\pi(X) = \mathbb{P}(D = 1 \mid X)$ is the propensity score, $\bar{Q}(D, X) = \mathbb{E}[Y \mid D, X]$ is the outcome regression, and $\eta = (\pi, \bar{Q})$.

### TMLE Procedure (Two-Step)

**Step 1 (Initial estimation)**: Estimate $\bar{Q}^0(D, X)$ (e.g., using SuperLearner) and $\pi(X)$ (e.g., using logistic regression with SL).

**Step 2 (Targeting fluctuation)**: Construct a logistic regression of $Y$ on $X$ and $D$ with an **offset** (the initial estimate) and a **clever covariate**:

$$H(D, X) = \frac{D}{\hat{\pi}(X)} - \frac{1 - D}{1 - \hat{\pi}(X)}$$

The fluctuation model is:

$$\text{logit}(\bar{Q}^1(D, X)) = \text{logit}(\bar{Q}^0(D, X)) + \varepsilon \cdot H(D, X)$$

Estimate $\varepsilon$ via maximum likelihood (keeping the offset fixed, only the fluctuation parameter $\varepsilon$ is estimated). The targeted estimate is:

$$\bar{Q}^*(D, X) = \text{expit}\left(\text{logit}(\bar{Q}^0(D, X)) + \hat{\varepsilon} \cdot H(D, X)\right)$$

**Step 3 (Parameter estimation)**:

$$\psi^* = \frac{1}{n} \sum_{i=1}^n \left(\bar{Q}^*(1, X_i) - \bar{Q}^*(0, X_i)\right)$$

**Step 4 (Inference)**:

$$\text{Var}(\psi^*) = \frac{1}{n} \text{Var}\left(IC(O \mid \hat{\eta}^*)\right)$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Unconfoundedness | $Y(1), Y(0) \perp D \mid X$ | No unmeasured confounders |
| Positivity | $0 < \pi(x) < 1$ | Overlap between treatment groups |
| Consistency | $Y = D Y(1) + (1 - D) Y(0)$ | Well-defined potential outcomes |
| Nuisance convergence | $\|\hat{\pi} - \pi\| \|\hat{\bar{Q}} - \bar{Q}\| = o_p(n^{-1/2})$ | Product rate condition |

## Applicable Scenarios

**When to use:**
- Estimating ATE, ATT, or other low-dimensional causal parameters
- Settings with complex, high-dimensional confounding where flexible ML is needed
- Longitudinal data with time-varying treatments (LTMLE)
- Survival outcomes with censoring (survival TMLE)
- When **double robustness** and **asymptotic efficiency** are both desired
- When the target parameter is a smooth functional of the data distribution

**When NOT to use:**
- When positivity is severely violated (TMLE becomes unstable)
- Very small sample sizes ($n < 200$)
- When interest is in heterogeneous effects (CATE); TMLE focuses on average effects

**Comparison:**
- TMLE vs AIPW: Both are doubly robust. TMLE is generally more stable in finite samples because it respects the bounds of the outcome (through the logistic fluctuation) and can leverage ensemble learning (SuperLearner) for nuisance estimation.

## Method Details

### Step-by-Step TMLE for ATE

1. **Estimate the propensity score** $\hat{\pi}(X)$ using SuperLearner or logistic regression.

2. **Estimate the initial outcome regression** $\hat{\bar{Q}}_n^0(D, X)$ using SuperLearner (allowing separate models for $D=1$ and $D=0$).

3. **Compute the clever covariate**:

   $$H(1, X) = \frac{1}{\hat{\pi}(X)}, \quad H(0, X) = -\frac{1}{1 - \hat{\pi}(X)}$$

4. **Estimate the fluctuation parameter** $\varepsilon$: Fit a logistic regression of $Y$ on $H(D, X)$ with offset $\text{logit}(\hat{\bar{Q}}^0(D, X))$.

5. **Update the outcome regression**:

   $$\hat{\bar{Q}}^*(D, X) = \text{expit}\left(\text{logit}(\hat{\bar{Q}}^0(D, X)) + \hat{\varepsilon} \cdot H(D, X)\right)$$

6. **Compute the TMLE estimate**:

   $$\hat{\psi}^* = \frac{1}{n} \sum_{i=1}^n \left(\hat{\bar{Q}}^*(1, X_i) - \hat{\bar{Q}}^*(0, X_i)\right)$$

7. **Compute standard errors** using the influence curve.

### Asymptotic Properties
- **Double robustness**: Consistent if either $\pi$ or $\bar{Q}$ is consistently estimated (not necessarily both)
- **Semiparametric efficiency**: If both are consistent at sufficient rates, achieves the semiparametric efficiency bound
- **Asymptotic linearity**: $\sqrt{n}(\hat{\psi}^* - \psi_0) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$

### Longitudinal TMLE (LTMLE)

For time-varying treatments $D_1, \dots, D_T$, LTMLE uses sequential regression and targeting at each time point. The LTMLE algorithm:
1. Estimate treatment mechanisms at each time point
2. Iteratively compute and target the G-computation formula from the last time point backwards
3. The final estimator is consistent under weaker conditions than standard G-computation

## Implementation Details

**Key hyperparameters:**
- Choice of learners in SuperLearner (SL library)
- Whether to use logistic or linear fluctuation
- For LTMLE: number of time points, structural model for each time point

**Numerical considerations:**
- For the logistic fluctuation, outcomes must be bounded between 0 and 1. Rescale continuous outcomes to [0, 1] before fluctuation.
- Avoid predicted propensity scores too close to 0 or 1 (truncate at, e.g., 0.025 and 0.975)
- The fluctuation step uses MLE; ensure numerical stability by using glm with well-scaled covariates

**Available software:**
- R: `tmle` package (CRAN), `ltmle` package (CRAN), `SuperLearner` package
- Python: `tmle` (PyPI, limited), `zEcon` (experimental)
- Stata: `tmle` command

## Python Implementation

```python
"""
Targeted Maximum Likelihood Estimation (TMLE) for the ATE

References:
    van der Laan & Rose (2011). Targeted Learning. Springer.
    van der Laan & Rubin (2006). Targeted maximum likelihood learning.
"""

import numpy as np
from scipy import special, stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold


class TMLE:
    """Targeted Maximum Likelihood Estimation for the ATE.

    Implements the two-step TMLE procedure:
    1. Initial estimation of Q and g (using flexible ML)
    2. Logistic fluctuation targeting the ATE parameter

    Parameters
    ----------
    learner_Q : estimator, default=RandomForestRegressor
        Model for E[Y | D, X] (outcome regression).
    learner_g : estimator, default=RandomForestClassifier
        Model for P(D=1 | X) (propensity score).
    n_folds : int, default=5
        Cross-fitting folds for nuisance estimation.
    bound : float, default=0.025
        Truncation bound for propensity scores.
    random_state : int, default=42
    """
    def __init__(self, learner_Q=None, learner_g=None,
                 n_folds=5, bound=0.025, random_state=42):
        self.learner_Q = learner_Q or RandomForestRegressor(
            n_estimators=500, min_samples_leaf=10, random_state=random_state)
        self.learner_g = learner_g or RandomForestClassifier(
            n_estimators=500, min_samples_leaf=10, random_state=random_state)
        self.n_folds = n_folds
        self.bound = bound
        self.random_state = random_state

    def fit(self, X, D, Y):
        """Estimate the ATE using TMLE with cross-fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Covariates.
        D : array-like, shape (n_samples,)
            Binary treatment (0 or 1).
        Y : array-like, shape (n_samples,)
            Outcome (will be rescaled to [0, 1] for logistic fluctuation).

        Returns
        -------
        self : TMLE
        """
        X = np.asarray(X)
        D = np.asarray(D).ravel().astype(int)
        Y = np.asarray(Y).ravel()
        n = len(Y)

        # Rescale Y to [0, 1] for logistic fluctuation
        self.Y_min_ = Y.min()
        self.Y_max_ = Y.max()
        if self.Y_max_ > self.Y_min_:
            Y_scaled = (Y - self.Y_min_) / (self.Y_max_ - self.Y_min_)
        else:
            Y_scaled = Y - self.Y_min_

        # Cross-fitting
        cv = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)

        Q0_hat = np.empty(n)
        Q1_hat = np.empty(n)
        pi_hat = np.empty(n)

        for train_idx, test_idx in cv.split(X):
            X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y_scaled[train_idx]
            X_te, D_te, Y_te = X[test_idx], D[test_idx], Y_scaled[test_idx]

            # Propensity score model
            g_model = self.learner_g.fit(X_tr, D_tr)
            pi_hat[test_idx] = g_model.predict_proba(X_te)[:, 1]
            pi_hat[test_idx] = np.clip(pi_hat[test_idx],
                                       self.bound, 1 - self.bound)

            # Outcome regressions (separate models for D=0 and D=1)
            Q0_model = self.learner_Q.fit(X_tr[D_tr == 0], Y_tr[D_tr == 0])
            Q1_model = self.learner_Q.fit(X_tr[D_tr == 1], Y_tr[D_tr == 1])

            Q0_hat[test_idx] = Q0_model.predict(X_te)
            Q1_hat[test_idx] = Q1_model.predict(X_te)

        # Step 2: Targeting (logistic fluctuation)
        # The clever covariate
        H = D / pi_hat - (1 - D) / (1 - pi_hat)

        # Fit the fluctuation model
        # logit(Q^*) = logit(Q^0) + epsilon * H
        # where Q^0 is evaluated at the observed treatment level for each unit
        Q_obs = np.where(D == 1, Q1_hat, Q0_hat)

        # Clamp for logit transform
        eps = 1e-8
        Q_obs_clamped = np.clip(Q_obs, eps, 1 - eps)

        # The fluctuation model is: logit(Q) ~ offset(logit(Q^0)) + H
        # We estimate epsilon with no intercept
        logit_Q = special.logit(Q_obs_clamped)
        H_2d = H.reshape(-1, 1)

        # Use IRLS (logistic regression) for the fluctuation
        from sklearn.linear_model import LogisticRegression
        # Since Y_scaled is continuous in [0,1], we use a quasi-binomial approach
        # via weighted least squares on the logit scale
        # For simplicity: estimate epsilon by minimizing the log-likelihood
        # of a model with offset + single covariate H

        # Use a simple line search for epsilon
        def neg_log_lik(eps):
            logit_star = logit_Q + eps * H
            Q_star = special.expit(logit_star)
            # Bernoulli log-likelihood
            ll = np.sum(Y_scaled * np.log(Q_star + eps) +
                       (1 - Y_scaled) * np.log(1 - Q_star + eps))
            return -ll

        from scipy.optimize import minimize_scalar
        res = minimize_scalar(neg_log_lik, bounds=(-10, 10), method='bounded')
        epsilon = res.x

        # Update Q estimates
        Q0_star = special.expit(special.logit(
            np.clip(Q0_hat, eps, 1 - eps)) + epsilon * (-1 / (1 - pi_hat)))
        Q1_star = special.expit(special.logit(
            np.clip(Q1_hat, eps, 1 - eps)) + epsilon * (1 / pi_hat))

        # TMLE estimate
        psi = np.mean(Q1_star - Q0_star)

        # Influence curve for standard error
        IC = (D / pi_hat - (1 - D) / (1 - pi_hat)) * (Y_scaled - np.where(D == 1, Q1_star, Q0_star)) \
             + (Q1_star - Q0_star) - psi

        var_IC = np.var(IC) / n
        se = np.sqrt(var_IC)
        ci = (psi - 1.96 * se, psi + 1.96 * se)

        # Rescale back to original Y scale
        scale = self.Y_max_ - self.Y_min_
        self.ate_ = psi * scale
        self.se_ = se * scale
        self.ci_ = (ci[0] * scale, ci[1] * scale)
        self.z_stat_ = self.ate_ / self.se_
        self.p_value_ = 2 * (1 - stats.norm.cdf(abs(self.z_stat_)))
        self.epsilon_ = epsilon
        self.Q0_star_ = Q0_star * scale + self.Y_min_
        self.Q1_star_ = Q1_star * scale + self.Y_min_

        return self


def simulate_confounded_data(n=2000, seed=42):
    """Simulate observational data with confounding.

    DGP:
        X1, X2, X3 ~ N(0, 1)
        D = Bernoulli(logit(0.5*X1 - 0.3*X2 + 0.1*X3))
        Y = 2.0*D + X1 + 0.5*X2^2 + sin(X3) + N(0, 1)
    """
    rng = np.random.RandomState(seed)
    X = np.column_stack([rng.randn(n), rng.randn(n), rng.randn(n)])

    pi = 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2])))
    D = rng.binomial(1, pi)

    Y = 2.0 * D + X[:, 0] + 0.5 * X[:, 1]**2 + np.sin(X[:, 2]) + rng.randn(n)

    return X, D, Y


if __name__ == "__main__":
    print("=" * 60)
    print("Targeted Maximum Likelihood Estimation (TMLE)")
    print("=" * 60)

    X, D, Y = simulate_confounded_data(n=2000, seed=42)
    true_ate = 2.0
    print(f"True ATE: {true_ate}")
    print(f"Sample size: n = {len(Y)}, p = {X.shape[1]}")

    # TMLE estimation
    tmle = TMLE(n_folds=5, bound=0.025, random_state=42)
    tmle.fit(X, D, Y)

    print(f"\nTMLE Results:")
    print(f"  ATE estimate:        {tmle.ate_:.4f}")
    print(f"  Standard error:      {tmle.se_:.4f}")
    print(f"  z-statistic:         {tmle.z_stat_:.2f}")
    print(f"  p-value:             {tmle.p_value_:.2e}")
    print(f"  95% CI:              ({tmle.ci_[0]:.4f}, {tmle.ci_[1]:.4f})")
    print(f"  Fluctuation epsilon: {tmle.epsilon_:.4f}")
    print(f"  Coverage: {'YES' if tmle.ci_[0] < true_ate < tmle.ci_[1] else 'NO'}")

    # Compare with naive estimator
    print(f"\n--- Comparison ---")
    from sklearn.linear_model import LinearRegression
    naive = LinearRegression()
    naive.fit(np.column_stack([D, X]), Y)
    print(f"Naive OLS (D coeff):           {naive.coef_[0]:.4f}  (biased)")
    print(f"Difference (treatment vs no treatment):")
    print(f"  Treated mean: {np.mean(Y[D == 1]):.2f}")
    print(f"  Control mean: {np.mean(Y[D == 0]):.2f}")
    print(f"  Naive diff:   {np.mean(Y[D == 1]) - np.mean(Y[D == 0]):.2f}")

    # Sensitivity to the bound
    print(f"\n--- Sensitivity to positivity bound ---")
    for bound in [0.001, 0.01, 0.025, 0.05, 0.1]:
        tmle_b = TMLE(n_folds=3, bound=bound, random_state=42)
        tmle_b.fit(X, D, Y)
        print(f"  bound = {bound:.3f}: ATE = {tmle_b.ate_:.4f}"
              f" (SE = {tmle_b.se_:.4f})")
```

## References

van der Laan, M. J., & Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

van der Laan, M. J., & Rose, S. (2018). *Targeted Learning in Data Science: Causal Inference for Complex Longitudinal Studies*. Springer. https://doi.org/10.1007/978-3-319-65304-4

van der Laan, M. J., & Rubin, D. (2006). Targeted maximum likelihood learning. *The International Journal of Biostatistics*, 2(1), Article 11. https://doi.org/10.2202/1557-4679.1043

Gruber, S., & van der Laan, M. J. (2009). Targeted maximum likelihood estimation: A gentle introduction. *U.C. Berkeley Division of Biostatistics Working Paper Series*, 252.

Luque-Fernandez, M. A., Schomaker, M., Rachet, B., & Schnitzer, M. E. (2018). Targeted maximum likelihood estimation for a binary treatment: A tutorial. *Statistics in Medicine*, 37(16), 2530-2546. https://doi.org/10.1002/sim.7629
```

