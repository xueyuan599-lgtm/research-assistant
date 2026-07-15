# Sensitivity Analysis for Omitted Variable Bias

**Source**: Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 82(1), 39–67. https://doi.org/10.1111/rssb.12348

**Source**: Cinelli, C., & Hazlett, C. (2025). An omitted variable bias framework for sensitivity analysis of instrumental variables. *Biometrika*, 112(1), asaf004. https://doi.org/10.1093/biomet/asaf004

**Category**: Causal Inference / Sensitivity Analysis

## Mathematical Setup

Sensitivity analysis for omitted variable bias asks: **how strong would an unobserved confounder need to be to overturn a research conclusion?** The Cinelli and Hazlett (2020) framework extends the classic omitted variable bias (OVB) formula using a partial $R^2$ parameterization, providing intuitive summaries like the **Robustness Value (RV)**.

### Setting

Consider the linear regression model:

$$Y = \tau D + \beta_X^\top X + \beta_Z Z + \varepsilon$$

where $D$ is the treatment, $X$ are observed covariates, and $Z$ is an unobserved confounder (unobserved by the analyst but present in the true DGP). Since $Z$ is unobserved, the actual estimable regression is:

$$Y = \tau_{\text{obs}} D + \tilde{\beta}_X^\top X + \tilde{\varepsilon}$$

The omitted variable bias in $\tau_{\text{obs}}$ is:

$$\tau_{\text{obs}} - \tau_{\text{true}} = \gamma \cdot \delta$$

where $\gamma$ is the partial effect of $Z$ on $Y$ (controlling for $X, D$), and $\delta$ is the partial effect of $D$ on $Z$ (controlling for $X$).

### Partial $R^2$ Parameterization

The bias can be re-expressed in terms of partial $R^2$ values:

$$\tau_{\text{obs}} - \tau_{\text{true}} = \hat{\sigma}_{Y \sim D \mid X} \cdot \frac{R_{Z \sim Y \mid X, D}}{R_{Z \sim D \mid X}}$$

where:
- $R_{Y \sim D \mid X}^2$ is the partial $R^2$ of treatment with outcome
- $R_{Z \sim D \mid X}^2$ is the partial $R^2$ of confounder with treatment
- $R_{Z \sim Y \mid X, D}^2$ is the partial $R^2$ of confounder with outcome

### Robustness Value (RV)

The **Robustness Value** is the minimum strength (in terms of partial $R^2$) an unobserved confounder would need to have with both the treatment and the outcome to reduce the estimated effect to exactly zero:

$$RV(q=1) = \frac{1}{2} \left( \sqrt{fd^2 + 4f} - fd \right)$$

where $f = |\hat{\tau}_{\text{obs}}|$ and $d = \sqrt{\frac{1}{n-k-1}}$ for the test statistic $t$.

For the case of reducing the estimate to statistical non-significance (at level $\alpha$):

$$RV(\alpha) = \frac{1}{2} \left( \sqrt{fd_\alpha^2 + 4f_\alpha} - f_\alpha d_\alpha \right)$$

with adjusted $f$ and $d$ based on the critical $t$-value.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Linear DGP | $Y = \tau D + X\beta_X + Z\beta_Z + \varepsilon$ | True model is linear in parameters |
| Single confounder aggregation | Multiple unobserved confounders can be summarized by a single index | Bounds remain conservative for multiple confounders |
| Homogeneous effects | The confounder effect on treatment and outcome is constant | Interpreted as **minimum** strength needed |
| No other misspecification | No measurement error, simultaneity, etc. | Sensitivity analysis is confounder-specific |

## Applicable Scenarios

**When to use:**
- Any OLS regression with concerns about omitted variable bias
- Before publishing results to quantify robustness to unobserved confounding
- Benchmarking against observed covariates ("how strong would a confounder need to be relative to `age`?")
- IV settings (using the iv.sensemakr extension)

**When NOT to use:**
- Non-linear models (GLM, logit, etc.) without extension (see Cinelli et al. 2022 for extensions)
- When the primary threat is measurement error rather than confounding
- When the main concern is reverse causality rather than omitted variables

## Method Details

### Step-by-Step Procedure

1. **Run the baseline regression**: Regress $Y$ on $D$ and $X$, obtain $\hat{\tau}_{\text{obs}}$ and its standard error.

2. **Compute the robustness value**:
   - Compute the $t$-statistic: $t = \hat{\tau}_{\text{obs}} / \text{SE}(\hat{\tau}_{\text{obs}})$
   - Compute $RV(q=1)$ for zero-effect threshold
   - Compute $RV(\alpha = 0.05)$ for significance threshold

3. **Benchmarking against observed covariates**:
   - For each observed covariate, compute its partial $R^2$ with treatment and outcome
   - Compute how strong an unobserved confounder would need to be relative to that covariate to overturn the result

4. **Contour plot interpretation**: Create a contour plot showing combinations of $R_{Z \sim D \mid X}^2$ and $R_{Z \sim Y \mid X, D}^2$ that would reduce the estimate to zero or to non-significance.

### Interpretation Rules

- $RV > 0.10$: Relatively robust (confounder would need to explain 10%+ of residual variance)
- $RV < 0.01$: Fragile (trivially small confounder could overturn)
- Benchmark ratio $> 2$: Very robust (confounder must be > 2x as strong as the strongest observed covariate)
- Benchmark ratio $< 1$: Fragile (confounder weaker than observed covariates could overturn)

## Implementation Details

**Available software:**
- R: `sensemakr` package (CRAN)
- Stata: `sensemakr` command (SSC)
- Python: `sensemakr` available on PyPI

## Python Implementation

```python
"""
Sensitivity Analysis for Omitted Variable Bias (Cinelli & Hazlett, 2020)

References:
    Cinelli & Hazlett (2020). Making sense of sensitivity.
    JRSS-B, 82(1), 39-67.
"""

import numpy as np
import pandas as pd
from scipy import stats


class OVBSensitivity:
    """Sensitivity analysis for omitted variable bias in OLS.

    Implements the partial R^2 parameterization and Robustness Value
    framework of Cinelli & Hazlett (2020).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Covariates (including a constant column for intercept if needed).
    D : array-like, shape (n_samples,)
        Treatment variable.
    Y : array-like, shape (n_samples,)
        Outcome variable.
    """
    def __init__(self, X, D, Y):
        X = np.asarray(X)
        D = np.asarray(D).ravel()
        Y = np.asarray(Y).ravel()

        # Add intercept if not present
        if not np.allclose(X[:, 0], 1.0):
            X = np.column_stack([np.ones(len(X)), X])

        self.X = X
        self.D = D
        self.Y = Y
        self.n, self.k = X.shape

        # Fit full model
        coefs, self.residuals, rank, s = np.linalg.lstsq(
            np.column_stack([D, X]), Y, rcond=None)

        self.tau_obs = coefs[0]
        self.beta_x = coefs[1:]

        # Standard error of tau
        n, p = Y.shape[0], len(coefs)
        mse = np.sum(self.residuals**2) / (n - p)
        var_cov = mse * np.linalg.inv(
            np.column_stack([D, X]).T @ np.column_stack([D, X]))
        self.se_tau = np.sqrt(var_cov[0, 0])
        self.t_stat = self.tau_obs / self.se_tau
        self.p_value = 2 * (1 - stats.t.cdf(np.abs(self.t_stat),
                                            df=n - p))

    def partial_r2(self, y, x, covariates):
        """Compute partial R^2 of x in regression of y on x + covariates.

        Parameters
        ----------
        y : ndarray
            Outcome.
        x : ndarray
            Target variable.
        covariates : ndarray
            Other covariates to control for.

        Returns
        -------
        r2 : float
            Partial R^2.
        """
        n = len(y)
        # Regress y on covariates, get residuals
        if covariates.shape[1] > 0:
            beta_y = np.linalg.lstsq(covariates, y, rcond=None)[0]
            resid_y = y - covariates @ beta_y

            # Regress x on covariates, get residuals
            beta_x = np.linalg.lstsq(covariates, x, rcond=None)[0]
            resid_x = x - covariates @ beta_x
        else:
            resid_y = y - np.mean(y)
            resid_x = x - np.mean(x)

        # Partial R^2 = correlation of residuals squared
        r = np.corrcoef(resid_y, resid_x)[0, 1]
        return r**2

    def robustness_value(self, alpha=0.05, q=1):
        """Compute the Robustness Value (RV).

        The minimum strength (in partial R^2 terms) an unobserved
        confounder must have with both treatment and outcome to
        reduce the estimate to q (default: zero).

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for the RV(alpha) threshold.
        q : float, default=1
            Target: multiply estimate by q (q=1 -> zero).

        Returns
        -------
        rv : float
            Robustness value for zero effect.
        rv_alpha : float
            Robustness value for significance threshold.
        """
        df = self.n - self.k - 1
        t_val = self.t_stat
        se = self.se_tau

        # RV for zero effect (q=1)
        f = np.abs(self.t_stat)
        denom = np.sqrt(df + f**2)
        rv = 0.5 * (np.sqrt(f**4 + 4 * f**2 * df) - f**2) / (df)

        # RV for significance at alpha
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        if np.abs(self.t_stat) <= t_crit:
            rv_alpha = 0.0
        else:
            f_alpha = np.abs(self.t_stat) - t_crit
            rv_alpha = 0.5 * (np.sqrt(f_alpha**4 +
                                       4 * f_alpha**2 * df) -
                                f_alpha**2) / (df)

        self.rv_ = rv
        self.rv_alpha_ = rv_alpha

        return rv, rv_alpha

    def benchmark_covariate(self, covariate_idx, alpha=0.05):
        """Benchmark against an observed covariate.

        Computes how strong an unobserved confounder needs to be
        relative to a given observed covariate to overturn the result.

        Parameters
        ----------
        covariate_idx : int
            Index of the covariate in X to benchmark against.
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        result : dict
            Contains partial R^2 of the covariate with D and Y,
            and the adjusted RV benchmark.
        """
        # Full covariate set (excluding the treatment and this covariate)
        Z = self.D.reshape(-1, 1)
        W = np.delete(self.X, covariate_idx, axis=1)
        target = self.X[:, covariate_idx]

        # Partial R^2 of covariate with treatment (controlling for others)
        r2_d = self.partial_r2(Z.ravel(), target, W)

        # Partial R^2 of covariate with outcome (controlling for D and others)
        still_W = np.column_stack([Z, W])
        r2_y = self.partial_r2(self.Y, target, still_W)

        # Ratio to RV
        rv, rv_alpha = self.robustness_value(alpha=alpha)

        result = {
            "covariate_idx": covariate_idx,
            "r2_d": r2_d,
            "r2_y": r2_y,
            "rv_ratio": np.sqrt((r2_d * r2_y) / (rv**2)) if rv > 0 else np.inf,
        }
        return result

    def adjusted_estimate(self, r2_d, r2_y):
        """Compute the adjusted treatment estimate given a confounder
        with specific partial R^2 values.

        Parameters
        ----------
        r2_d : float
            Partial R^2 of confounder with treatment.
        r2_y : float
            Partial R^2 of confounder with outcome.

        Returns
        -------
        tau_adjusted : float
            Bias-adjusted treatment effect.
        """
        # Variance of treatment after partialling out X
        X_d = np.column_stack([np.ones(self.n), self.X])
        beta_d = np.linalg.lstsq(X_d, self.D, rcond=None)[0]
        resid_d = self.D - X_d @ beta_d
        var_d = np.var(resid_d)

        # Compute bias: sign = sign of tau_obs (assuming positive confounding)
        sign = np.sign(self.tau_obs)
        bias = sign * np.sqrt(r2_d * r2_y / (1 - r2_d) * var_d * np.var(self.residuals) / var_d)

        return self.tau_obs - bias * sign


def simulate_data(n=500, seed=42):
    """Simulate data with unobserved confounding.

    True DGP:
    Y = 0.5 * D + X1 + X2 + Z + noise
    D = X1 - X2 + Z + noise
    where Z is the unobserved confounder.
    """
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n)
    X2 = rng.randn(n)
    Z = rng.randn(n)  # unobserved confounder

    D = X1 - X2 + Z + 0.5 * rng.randn(n)
    Y = 0.5 * D + X1 + X2 + Z + rng.randn(n)

    return np.column_stack([X1, X2]), D, Y, Z


if __name__ == "__main__":
    print("=" * 60)
    print("Sensitivity Analysis for Omitted Variable Bias")
    print("Cinelli & Hazlett (2020) — JRSS-B")
    print("=" * 60)

    X_obs, D, Y, Z_unobs = simulate_data(n=1000, seed=42)
    print(f"\nTrue effect: tau = 0.5")
    print(f"(Unobserved confounder Z exists)")

    # Run sensitivity analysis
    sa = OVBSensitivity(X_obs, D, Y)
    print(f"\nObserved estimate: tau = {sa.tau_obs:.4f} "
          f"(SE = {sa.se_tau:.4f}, p = {sa.p_value:.2e})")

    # Compute robustness values
    rv, rv_alpha = sa.robustness_value(alpha=0.05)
    print(f"\nRobustness Value (zero effect):   RV = {rv:.4f}")
    print(f"Robustness Value (significance): RV(alpha) = {rv_alpha:.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    print(f"  An unobserved confounder would need to explain "
          f"{rv*100:.1f}% of the residual")
    print(f"  variance in BOTH treatment and outcome to reduce the")
    print(f"  estimated effect to exactly zero.")

    # Benchmarking against observed covariates
    print(f"\n--- Benchmarking against observed covariates ---")
    for i in range(X_obs.shape[1]):
        bm = sa.benchmark_covariate(i)
        print(f"  Covariate X{i}: partial R^2 with D = {bm['r2_d']:.4f}, "
              f"with Y = {bm['r2_y']:.4f}")
        print(f"    RV ratio: {bm['rv_ratio']:.2f}x "
              f"(confounder must be {bm['rv_ratio']:.1f}x as strong "
              f"as this covariate)")

    # Adjusted estimates under different confounder strengths
    print(f"\n--- Adjusted estimates under different confounder strengths ---")
    for r2_z in [0.01, 0.02, 0.05, 0.10, rv, rv_alpha]:
        tau_adj = sa.adjusted_estimate(r2_z, r2_z)
        still_significant = np.abs(tau_adj / sa.se_tau) > 1.96
        note = ""
        if abs(r2_z - rv) < 1e-4:
            note = " (= RV, zero effect)"
        elif abs(r2_z - rv_alpha) < 1e-4:
            note = " (= RV alpha, borderline significant)"
        print(f"  R^2_Z = {r2_z:.4f}{note}: "
              f"adjusted tau = {tau_adj:.4f}, "
              f"significant = {still_significant}")
```

## References

Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 82(1), 39–67. https://doi.org/10.1111/rssb.12348

Cinelli, C., Ferwerda, J., & Hazlett, C. (2020). sensemakr: Sensitivity analysis tools for OLS in R and Stata. *Observational Studies*, 10(2), 93–127. https://doi.org/10.2139/ssrn.3588978

Cinelli, C., & Hazlett, C. (2025). An omitted variable bias framework for sensitivity analysis of instrumental variables. *Biometrika*, 112(1), asaf004. https://doi.org/10.1093/biomet/asaf004

Rosenbaum, P. R. (2002). *Observational studies* (2nd ed.). Springer. https://doi.org/10.1007/978-1-4757-3692-2
```

