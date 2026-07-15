# Bayesian Distributionally Robust Optimization

**Source**: Shapiro, A., Zhou, E., & Lin, Y. (2023). Bayesian distributionally robust optimization. *SIAM Journal on Optimization*, 33(2), 1279-1304. https://doi.org/10.1137/21M1465548

**Category**: Operations Research / Optimization / Distributionally Robust Optimization

## Mathematical Setup

Consider a stochastic optimization problem:

$$
\min_{a \in \mathcal{A}} \quad \mathbb{E}_{\mathbb{P}^\star}[h(a, \xi)]
$$

where the true distribution $\mathbb{P}^\star$ is unknown and must be estimated from data. 

### Distributionally Robust Optimization (DRO)

The classical DRO approach constructs an ambiguity set $\mathcal{P}$ of plausible distributions and solves:

$$
\min_{a \in \mathcal{A}} \quad \sup_{\mathbb{P} \in \mathcal{P}} \mathbb{E}_{\mathbb{P}}[h(a, \xi)]
$$

### Bayesian-DRO

Shapiro, Zhou, and Lin (2023) propose a **Bayesian approach** to DRO:

1. **Prior**: Place a prior distribution $\pi(\mathbb{P})$ on the space of possible distributions
2. **Posterior**: After observing data $\hat{\xi}_1, \dots, \hat{\xi}_n$, compute the posterior $\pi(\mathbb{P} | \hat{\xi}_{1:n})$
3. **Ambiguity set**: Construct a credible region around the posterior mean:
   $$
   \mathcal{P}_{\alpha} = \left\{ \mathbb{P} : D(\mathbb{P} \| \hat{\mathbb{P}}_n) \leq \delta_\alpha \right\}
   $$
   where $D(\cdot \| \cdot)$ is a divergence measure (e.g., KL divergence, Wasserstein distance), and $\delta_\alpha$ is chosen so that $\mathbb{P}^\star \in \mathcal{P}_\alpha$ with posterior probability at least $1-\alpha$

4. **Optimization**: Solve the DRO problem using $\mathcal{P}_\alpha$

### Key Theoretical Result

Under a Dirichlet process prior (or more generally, a nonparametric Bayesian prior), the posterior distribution of the unknown distribution converges to a point mass at $\mathbb{P}^\star$ as $n \to \infty$, and the optimal value of the Bayesian-DRO problem converges to the true optimal value at rate $O_p(n^{-1/2})$.

The Bayesian-DRO problem can be reformulated as:

$$
\min_{a \in \mathcal{A}} \left\{ \mathbb{E}_{\hat{\mathbb{P}}_n}[h(a, \xi)] + \frac{\kappa}{\sqrt{n}} \sqrt{\text{Var}_{\hat{\mathbb{P}}_n}[h(a, \xi)]} \right\}
$$

where $\kappa$ is a constant determined by the confidence level and the prior, providing an interpretable **mean-variance trade-off**.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Nonparametric Bayesian prior | $\mathbb{P} \sim \text{DP}(\alpha, \mathbb{P}_0)$ (Dirichlet process) | The posterior is tractable; the Bayes estimator is the empirical distribution plus prior smoothing |
| Bounded or sub-Gaussian costs | $\mathbb{E}[e^{\lambda(h(a,\xi) - \mu)}] \leq e^{\lambda^2 \sigma^2/2}$ | Concentration inequalities hold |
| Compact decision set | $\mathcal{A}$ is compact in $\mathbb{R}^d$ | The DRO problem attains its minimum |
| Lipschitz continuity | $h$ is Lipschitz in $a$ for each $\xi$ | The optimal value is stable under distributional perturbations |
| Divergence convexity | $D(\mathbb{P} \| \mathbb{Q})$ is convex in $\mathbb{P}$ | The inner supremum is a convex optimization problem |

## Applicable Scenarios

**When to use:**
- Small-sample stochastic optimization where the prior can meaningfully regularize
- Sequential decision-making where the posterior is updated with each observation
- Risk-averse applications where the decision-maker wants Bayesian credible guarantees
- Problems where the Dirichlet process prior is a reasonable model (e.g., categorical data, mixture models)
- Portfolio optimization with return ambiguity

**When NOT to use:**
- When $n$ is very large and the prior influence is negligible (standard DRO is sufficient)
- When the dimension of $\xi$ is very high (the Dirichlet process suffers from the curse of dimensionality for continuous $\xi$)
- When computational cost of posterior inference is prohibitive
- When the decision-maker is ambiguity-averse but does not accept a Bayesian framework

**Comparison with alternatives:**
- **Frequentist DRO (Wasserstein/phi-divergence)**: Does not incorporate prior information; the Bayesian approach provides a principled way to include prior knowledge
- **Bayesian risk optimization (BRO)**: Minimizes posterior expected risk; Bayesian-DRO provides robustness against model misspecification within the posterior credible region
- **Robust Bayes**: Optimizes over a set of priors; Bayesian-DRO takes a single prior but optimizes over posteriors

## Algorithm / Method

### Bayesian-DRO Procedure

**Input**: Prior $\pi$, data $\hat{\xi}_{1:n}$, confidence $1-\alpha$, divergence $D$

1. **Compute posterior**: $\pi(\mathbb{P} | \hat{\xi}_{1:n}) \propto \pi(\mathbb{P}) \prod_{i=1}^n \mathbb{P}(\hat{\xi}_i)$

2. **Construct posterior credible region**: Find $\delta_\alpha$ such that:
   $$
   \pi\left( D(\mathbb{P} \| \hat{\mathbb{P}}_n) \leq \delta_\alpha \;|\; \hat{\xi}_{1:n} \right) = 1 - \alpha
   $$

3. **Solve Bayesian-DRO**:
   $$
   \hat{a}_n = \arg\min_{a \in \mathcal{A}} \sup_{\mathbb{P}: D(\mathbb{P} \| \hat{\mathbb{P}}_n) \leq \delta_\alpha} \mathbb{E}_{\mathbb{P}}[h(a, \xi)]
   $$

### Convergence Rate

Under standard conditions, as $n \to \infty$:
- $\hat{a}_n \to a^\star$ almost surely
- $|\hat{V}_n - V^\star| = O_p(n^{-1/2})$ where $\hat{V}_n$ is the optimal value and $V^\star$ is the true optimal value
- The radius $\delta_\alpha$ shrinks at rate $O_p(n^{-1/2})$

## Implementation Details

**Key parameters:**
- Prior precision: Controls the influence of the prior relative to the data
- Confidence level $\alpha$: Typically 0.05 or 0.10
- Divergence $D$: KL divergence for exponential families, Wasserstein for general continuous distributions

**Numerical considerations:**
- The inner supremum over $\mathbb{P}$ can often be reformulated as a finite-dimensional convex optimization via duality
- For KL divergence, the DRO problem is equivalent to an exponential tilting of the empirical distribution
- Sampling from the posterior can be done via the Dirichlet process stick-breaking construction

## Python Implementation

```python
import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.stats import dirichlet, norm, beta as beta_dist
from typing import Callable, Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")


class BayesianDRO:
    """
    Bayesian Distributionally Robust Optimization.
    
    Implements the framework of Shapiro, Zhou & Lin (2023).
    
    The key idea is to construct a posterior credible region around
    the empirical distribution and solve the resulting DRO problem.
    
    For tractability, we use a conjugate Dirichlet-multinomial model
    for discrete outcomes and a Gaussian-Wishart model for continuous.
    """
    
    def __init__(
        self,
        cost_function: Callable[[np.ndarray, np.ndarray], float],
        prior_strength: float = 1.0,
        confidence: float = 0.95,
        divergence: str = "kl"
    ):
        """
        Parameters
        ----------
        cost_function : Callable
            Cost function h(a, xi)
        prior_strength : float
            Strength of the prior (pseudo-counts). Higher = stronger prior.
        confidence : float
            Credible level for the ambiguity set (1 - alpha)
        divergence : str
            Divergence measure: "kl" (Kullback-Leibler) or "chi2" (chi-squared)
        """
        self.h = cost_function
        self.prior_strength = prior_strength
        self.confidence = confidence
        self.divergence = divergence
        
    def _compute_kl_radius(
        self, n: int, d: int
    ) -> float:
        """
        Compute the KL divergence radius for the posterior credible set.
        
        Under the Dirichlet-multinomial model, the posterior distribution
        of the multinomial probabilities follows a Dirichlet distribution.
        The (1-alpha) credible region for the KL divergence has radius
        approximately chi2_{d-1}(1-alpha) / (2n).
        """
        from scipy.stats import chi2
        alpha = 1.0 - self.confidence
        # Approximation for large n
        radius = chi2.ppf(1.0 - alpha, df=d-1) / (2.0 * n)
        return radius
    
    def _solve_kl_dro(
        self, a: np.ndarray, probs: np.ndarray, 
        xi_grid: np.ndarray, radius: float
    ) -> float:
        """
        Solve the inner maximization for KL-DRO:
            sup_{q: KL(q || p) <= radius} sum_i q_i * h(a, xi_i)
        
        Using the dual formulation:
            inf_{lambda >= 0} lambda * radius 
            + lambda * log( sum_i p_i * exp(h(a, xi_i) / lambda) )
        
        This is a 1D convex optimization in lambda.
        """
        costs = np.array([self.h(a, xi) for xi in xi_grid])
        
        # Dual function
        def dual(lmbda):
            if lmbda <= 0:
                return np.inf
            # log-sum-exp
            log_term = np.log(np.sum(probs * np.exp(costs / lmbda)))
            return lmbda * radius + lmbda * log_term
        
        # Optimize over lambda > 0
        result = minimize(
            dual, 
            x0=np.array([1.0]),
            bounds=[(1e-6, None)],
            method="L-BFGS-B"
        )
        
        return result.fun
    
    def fit_discrete(
        self,
        data: np.ndarray,
        xi_grid: np.ndarray
    ) -> dict:
        """
        Fit Bayesian-DRO for discrete outcomes.
        
        Parameters
        ----------
        data : np.ndarray of shape (n_samples,)
            Observed discrete outcomes (indices into xi_grid)
        xi_grid : np.ndarray of shape (n_categories, dim_xi)
            Grid of possible outcome values
            
        Returns
        -------
        model : dict
            Contains posterior probabilities, radius, etc.
        """
        n = len(data)
        d = len(xi_grid)
        
        # Count occurrences of each category
        counts = np.zeros(d)
        for val in data:
            counts[int(val)] += 1
        
        # Posterior Dirichlet parameters
        posterior_alpha = counts + self.prior_strength
        posterior_mean = posterior_alpha / np.sum(posterior_alpha)
        
        # KL radius for credible set
        radius = self._compute_kl_radius(n, d)
        
        print(f"Bayesian-DRO fit (discrete): n={n}, d={d}, "
              f"radius={radius:.4f}")
        print(f"Prior strength: {self.prior_strength}")
        print(f"Posterior mean probabilities: {posterior_mean[:5]}...")
        
        return {
            "type": "discrete",
            "posterior_alpha": posterior_alpha,
            "posterior_mean": posterior_mean,
            "radius": radius,
            "xi_grid": xi_grid,
            "n_samples": n
        }
    
    def optimize_discrete(
        self, model: dict, a0: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Solve the Bayesian-DRO problem for a discrete outcome space.
        
        Parameters
        ----------
        model : dict
            Fitted model from fit_discrete()
        a0 : np.ndarray
            Initial guess for the decision
            
        Returns
        -------
        a_opt : np.ndarray
            Optimal Bayesian-DRO decision
        obj_val : float
            Optimal objective value
        """
        probs = model["posterior_mean"]
        xi_grid = model["xi_grid"]
        radius = model["radius"]
        
        def objective(a):
            return self._solve_kl_dro(a, probs, xi_grid, radius)
        
        result = minimize(
            objective,
            a0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-8}
        )
        
        return result.x, result.fun
    
    def fit_continuous(
        self, data: np.ndarray
    ) -> dict:
        """
        Fit Bayesian-DRO for continuous outcomes using a Gaussian model
        with a Normal-Inverse-Wishart prior.
        
        Parameters
        ----------
        data : np.ndarray of shape (n_samples, dim_xi)
            Observed continuous outcomes
        """
        n, d = data.shape
        
        # Sample mean and covariance
        mu_emp = np.mean(data, axis=0)
        Sigma_emp = np.cov(data, rowvar=False)
        
        # Posterior under conjugate NIW prior
        # Prior: mu | Sigma ~ N(mu0, Sigma/kappa0), Sigma ~ IW(nu0, Psi0)
        # We use a weak prior: mu0 = mu_emp, kappa0 = prior_strength,
        #   nu0 = d + 2, Psi0 = (nu0 - d - 1) * Sigma_emp
        
        kappa0 = self.prior_strength
        nu0 = d + 2
        Psi0 = (nu0 - d - 1) * Sigma_emp
        
        # Posterior parameters
        kappa_n = kappa0 + n
        nu_n = nu0 + n
        mu_n = (kappa0 * mu_emp + n * mu_emp) / kappa_n  # simplified if mu0 = mu_emp
        # More generally, mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
        
        Psi_n = Psi0 + (n - 1) * Sigma_emp + (kappa0 * n / kappa_n) * np.outer(mu_emp - mu_emp, mu_emp - mu_emp)
        
        # The posterior predictive is a multivariate t-distribution
        # The ambiguity radius captures the posterior uncertainty
        
        # For the Wasserstein DRO with Gaussian data, the radius
        # relates to the posterior covariance
        # We use an approximation: radius = chi2_d(1-alpha) / sqrt(n)
        from scipy.stats import chi2
        alpha = 1.0 - self.confidence
        wasserstein_radius = np.sqrt(chi2.ppf(1.0 - alpha, df=d) / n)
        
        print(f"Bayesian-DRO fit (continuous): n={n}, d={d}")
        print(f"Posterior mean: {mu_n}")
        print(f"Wasserstein radius: {wasserstein_radius:.4f}")
        
        return {
            "type": "continuous",
            "mu_n": mu_n,
            "Sigma_n": Psi_n / (nu_n - d - 1),
            "nu_n": nu_n,
            "Psi_n": Psi_n,
            "radius": wasserstein_radius,
            "n_samples": n
        }
    
    def optimize_continuous(
        self, model: dict, a0: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Solve Bayesian-DRO for continuous outcomes.
        
        Uses the mean-variance approximation:
            min_a E_{post}[h(a, xi)] + (kappa/sqrt(n)) * sqrt(Var_{post}[h(a, xi)])
        """
        mu = model["mu_n"]
        Sigma = model["Sigma_n"]
        radius = model["radius"]
        n = model["n_samples"]
        
        def objective(a):
            # Sample from posterior predictive to estimate mean and variance
            n_mc = 500
            xi_samples = np.random.multivariate_normal(mu, Sigma, size=n_mc)
            costs = np.array([self.h(a, xi) for xi in xi_samples])
            
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            
            # Mean-variance DRO approximation
            kappa = np.sqrt(2 * np.log(1.0 / (1.0 - self.confidence)))
            return mean_cost + kappa * std_cost / np.sqrt(n)
        
        result = minimize(
            objective,
            a0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-8}
        )
        
        return result.x, result.fun


# ============================================================
# Example: Bayesian-DRO for Portfolio Optimization
# ============================================================
def demo_bayesian_dro():
    """
    Demonstrate Bayesian-DRO on a portfolio optimization problem.
    
    We have K assets with unknown expected returns. The investor
    allocates wealth to maximize the worst-case expected return
    within a posterior credible set.
    
    Portfolio optimization:
        max_{w >= 0, sum(w)=1}  min_{mu in credible_set}  w^T mu
    """
    np.random.seed(42)
    
    n_assets = 5
    n_samples = 30
    
    # True (unknown) expected returns
    mu_true = np.array([0.12, 0.08, 0.15, 0.06, 0.10])
    sigma_true = 0.15
    
    # Generate observed returns
    returns = np.random.normal(mu_true, sigma_true, size=(n_samples, n_assets))
    
    print("=" * 60)
    print("BAYESIAN DISTRIBUTIONALLY ROBUST PORTFOLIO OPTIMIZATION")
    print("=" * 60)
    print(f"\nTrue expected returns: {mu_true}")
    print(f"Number of assets: {n_assets}")
    print(f"Number of samples: {n_samples}")
    
    # ---- Bayesian-DRO ----
    
    # Negative expected return (for minimization)
    def neg_return(w, xi):
        return -np.dot(w, xi)
    
    bdro = BayesianDRO(
        cost_function=neg_return,
        prior_strength=1.0,
        confidence=0.90,
        divergence="kl"
    )
    
    model = bdro.fit_continuous(returns)
    w0 = np.ones(n_assets) / n_assets
    
    # Constrained optimization: w >= 0, sum(w) = 1
    def objective(w):
        w = np.array(w)
        # Project onto simplex
        w_sorted = np.sort(w)[::-1]
        cssv = np.cumsum(w_sorted)
        rho = np.where(w_sorted > (cssv - 1) / np.arange(1, len(w) + 1))[0][-1]
        tau = (cssv[rho] - 1) / (rho + 1)
        w_proj = np.maximum(w - tau, 0)
        w_proj = w_proj / np.sum(w_proj)
        
        # Evaluate objective
        mu = model["mu_n"]
        Sigma = model["Sigma_n"]
        
        # Mean-variance DRO objective
        mean_ret = np.dot(w_proj, mu)
        var_ret = w_proj @ Sigma @ w_proj
        kappa = np.sqrt(2 * np.log(1.0 / (1.0 - 0.90)))
        
        # We maximize the worst-case return, i.e., minimize negative worst-case
        obj = -(mean_ret - kappa * np.sqrt(var_ret) / np.sqrt(n_samples))
        return obj
    
    result = minimize(
        objective,
        w0,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8}
    )
    
    # Project result onto simplex
    w_bdro = result.x
    w_bdro = np.maximum(w_bdro, 0)
    w_bdro = w_bdro / np.sum(w_bdro)
    
    # Compute true performance
    true_ret_bdro = np.dot(w_bdro, mu_true)
    
    print(f"\n--- Bayesian-DRO Portfolio ---")
    for i in range(n_assets):
        print(f"  Asset {i+1}: {w_bdro[i]:.4f}")
    print(f"  True expected return: {true_ret_bdro:.4f}")
    
    # ---- Benchmark: Equal Weight ----
    w_equal = np.ones(n_assets) / n_assets
    true_ret_equal = np.dot(w_equal, mu_true)
    
    print(f"\n--- Equal Weight Portfolio ---")
    print(f"  True expected return: {true_ret_equal:.4f}")
    
    # ---- Benchmark: Sample Mean (Markowitz) ----
    mu_emp = np.mean(returns, axis=0)
    Sigma_emp = np.cov(returns, rowvar=False)
    
    # Maximum Sharpe ratio (simplified: max return for given risk level)
    # Here we maximize the sample mean return, constrained to sum(w)=1, w>=0
    def neg_sample_mean(w):
        w_proj = np.maximum(w, 0)
        w_proj = w_proj / np.sum(w_proj)
        return -np.dot(w_proj, mu_emp)
    
    res_mark = minimize(
        neg_sample_mean,
        w0,
        method="Nelder-Mead",
        options={"maxiter": 1000}
    )
    w_mark = np.maximum(res_mark.x, 0)
    w_mark = w_mark / np.sum(w_mark)
    true_ret_mark = np.dot(w_mark, mu_true)
    
    print(f"\n--- Sample Mean (Markowitz) Portfolio ---")
    for i in range(n_assets):
        print(f"  Asset {i+1}: {w_mark[i]:.4f}")
    print(f"  True expected return: {true_ret_mark:.4f}")
    
    # ---- Compare ----
    print(f"\n--- Summary ---")
    print(f"{'Strategy':<25} {'True Expected Return':>22}")
    print("-" * 48)
    print(f"{'Equal Weight':<25} {true_ret_equal:>22.4f}")
    print(f"{'Sample Mean (Markowitz)':<25} {true_ret_mark:>22.4f}")
    print(f"{'Bayesian-DRO':<25} {true_ret_bdro:>22.4f}")
    
    # The Bayesian-DRO should find a more robust portfolio that
    # avoids assets with high estimation uncertainty
    
    return bdro, w_bdro, w_mark, w_equal


if __name__ == "__main__":
    bdro, w_bdro, w_mark, w_equal = demo_bayesian_dro()
```

## References

Shapiro, A., Zhou, E., & Lin, Y. (2023). Bayesian distributionally robust optimization. *SIAM Journal on Optimization*, 33(2), 1279-1304. https://doi.org/10.1137/21M1465548

Shapiro, A. (2017). Distributionally robust stochastic programming. *SIAM Journal on Optimization*, 27(4), 2258-2275. https://doi.org/10.1137/16M108726X

Mohajerin Esfahani, P., & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric. *Operations Research*, 66(4), 917-939. https://doi.org/10.1287/opre.2017.1712

Kuhn, D., Esfahani, P. M., Nguyen, V. A., & Shafieezadeh-Abadeh, S. (2019). Wasserstein distributionally robust optimization: Theory and applications in machine learning. *INFORMS TutORials in Operations Research*, 130-166. https://doi.org/10.1287/educ.2019.0198

Rahimian, H., & Mehrotra, S. (2022). Frameworks and results in distributionally robust optimization. *Open Journal of Mathematical Optimization*, 3, 1-85. https://doi.org/10.5802/ojmo.15
