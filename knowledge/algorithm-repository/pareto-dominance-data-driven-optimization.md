# Pareto Dominance Principle for Data-Driven Optimization

**Source**: Sutter, T., Van Parys, B. P. G., & Kuhn, D. (2024). A Pareto dominance principle for data-driven optimization. *Operations Research*, 72(5), 1976-1999. https://doi.org/10.1287/opre.2021.0609

**Category**: Operations Research / Optimization / Data-Driven Optimization

## Mathematical Setup

Consider a stochastic optimization problem:

$$
\min_{a \in \mathcal{A}} \quad \mathbb{E}_{\mathbb{P}^\star}[h(a, \xi)] 
$$

where:
- $a \in \mathcal{A}$ is a decision (feasible set $\mathcal{A} \subseteq \mathbb{R}^d$)
- $\xi \in \Xi$ is a random vector with unknown true distribution $\mathbb{P}^\star$
- $h: \mathcal{A} \times \Xi \to \mathbb{R}$ is a cost function
- The decision $a$ must be chosen before $\xi$ is observed, based on training data $\hat{\xi}_1, \dots, \hat{\xi}_n \sim \mathbb{P}^\star$

A **data-driven decision** is a mapping $\hat{a}: \Xi^n \to \mathcal{A}$ that maps training data to a decision.

The **out-of-sample risk** of a data-driven decision $\hat{a}_n = \hat{a}(\hat{\xi}_1, \dots, \hat{\xi}_n)$ is:

$$
R(\hat{a}_n) = \mathbb{E}_{\mathbb{P}^\star}[h(\hat{a}_n, \xi)]
$$

The **out-of-sample disappointment** is:

$$
D(\hat{a}_n) = \mathbb{P}_{\mathbb{P}^\star}\left(R(\hat{a}_n) > \hat{V}_n\right)
$$

where $\hat{V}_n$ is the optimal value of the surrogate optimization model used to construct $\hat{a}_n$.

### Key Definition: Pareto Dominance

A data-driven decision $\hat{a}$ **Pareto dominates** another decision $\tilde{a}$ if:

$$
\mathbb{E}_{\mathbb{P}^\star}[D(\hat{a})] \leq \mathbb{E}_{\mathbb{P}^\star}[D(\tilde{a})] \quad \text{and} \quad \mathbb{E}_{\mathbb{P}^\star}[R(\hat{a})] \leq \mathbb{E}_{\mathbb{P}^\star}[R(\tilde{a})]
$$

with at least one inequality strict. A data-driven decision is **Pareto optimal** if no other decision Pareto dominates it.

### Main Result

Under the assumption that the unknown distribution $\mathbb{P}^\star$ belongs to a parametric ambiguity set $\mathcal{P}_\Theta = \{\mathbb{P}_\theta : \theta \in \Theta\}$ where $\theta$ admits a sufficient statistic $T_n$ satisfying a **large deviation principle**, the Pareto-optimal data-driven decision is obtained by solving:

$$
\hat{a}_n^\star = \arg\min_{a \in \mathcal{A}} \sup_{\mathbb{P} \in \mathcal{B}_n} \mathbb{E}_{\mathbb{P}}[h(a, \xi)]
$$

where $\mathcal{B}_n$ is a **distributionally robust ambiguity set** constructed from the sufficient statistic $T_n$ and its associated rate function $I(\cdot)$:

$$
\mathcal{B}_n = \left\{\mathbb{P}_\theta : I(\theta \| \hat{\theta}_n) \leq \frac{\log n}{n} \right\}
$$

Here $\hat{\theta}_n$ is the maximum likelihood estimator of $\theta$.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Parametric model | $\mathbb{P}^\star \in \{\mathbb{P}_\theta : \theta \in \Theta \subseteq \mathbb{R}^k\}$ | The data-generating process belongs to a known parametric family |
| Sufficient statistic | $T_n = t(\hat{\xi}_1, \dots, \hat{\xi}_n)$ is sufficient for $\theta$ | The data can be compressed without loss of information |
| Large deviation principle | $T_n$ satisfies a LDP with rate function $I$ | Tail probabilities decay exponentially; enables tight confidence sets |
| Compactness | $\mathcal{A}$ is compact, $h$ is continuous | The optimization problem is well-posed |
| Identifiability | The mapping $\theta \mapsto \mathbb{P}_\theta$ is injective | The parameter uniquely identifies the distribution |

## Applicable Scenarios

**When to use:**
- Stochastic optimization with limited data where distributional uncertainty is significant
- Problems where the decision-maker is ambiguity-averse and wants statistically optimal decisions
- Settings where the data-generating process is well-approximated by a parametric family
- Risk-sensitive applications (finance, healthcare, infrastructure) where disappointment matters

**When NOT to use:**
- When the parametric assumption is clearly violated and nonparametric approaches are needed
- When computational cost of solving DRO problems is prohibitive
- When $n$ is very small and the large deviation approximation may be poor
- When the sufficient statistic is high-dimensional (curse of dimensionality in the rate function)

**Comparison with alternatives:**
- **Sample Average Approximation (SAA)**: No robustness; can overfit in small samples. Pareto-dominated by DRO.
- **Classical DRO (Wasserstein/phi-divergence)**: Typically chooses the ambiguity set radius heuristically; the Pareto principle provides a principled, optimal choice.
- **Bayesian approach**: Requires a prior; the Pareto-DRO framework is prior-free.

## Algorithm / Method

### Constructing the Pareto-Optimal DRO Formulation

**Input**: Training data $\hat{\xi}_1, \dots, \hat{\xi}_n$, parametric family $\{\mathbb{P}_\theta\}$, significance level $\alpha$

1. **Estimate parameter**: Compute $\hat{\theta}_n$ via maximum likelihood estimation

2. **Identify sufficient statistic**: Find $T_n = t(\xi_1, \dots, \xi_n)$ sufficient for $\theta$, and derive its large deviation rate function $I(\cdot)$

3. **Construct ambiguity set**:
   $$
   \mathcal{B}_n(\delta_n) = \left\{\mathbb{P}_\theta : I(\theta \| \hat{\theta}_n) \leq \delta_n \right\}
   $$
   where $\delta_n = \frac{\log n}{n}$ gives Pareto optimality

4. **Solve DRO problem**:
   $$
   \hat{a}_n = \arg\min_{a \in \mathcal{A}} \sup_{\mathbb{P}_\theta \in \mathcal{B}_n} \mathbb{E}_{\mathbb{P}_\theta}[h(a, \xi)]
   $$

**Output**: Pareto-optimal data-driven decision $\hat{a}_n$

### Convergence Guarantees

- As $n \to \infty$, the radius $\delta_n \to 0$ and $\hat{a}_n \to a^\star$ (the true optimal decision under $\mathbb{P}^\star$)
- The DRO solution exhibits finite-sample optimality in the Pareto sense: no other decision procedure can improve both risk and disappointment simultaneously
- Rate of convergence is $O(n^{-1/2})$ under standard smoothness conditions

## Implementation Details

**Key parameters:**
- $\delta_n$: The size of the ambiguity set. The Pareto-optimal choice is $\delta_n = \log n / n$, but practitioners may use cross-validation to tune it
- Rate function $I(\cdot)$: May need to be estimated or approximated numerically

**Numerical considerations:**
- The inner maximization over $\theta$ in the DRO problem may be nonconvex
- For exponential families, the DRO problem often simplifies to a tractable convex optimization
- Use the MLE as a starting point for numerical optimization of the inner problem

## Python Implementation

```python
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats import norm, chi2
from typing import Callable, Optional, Tuple

class ParetoDRO:
    """
    Pareto-optimal data-driven optimization via distributionally robust
    optimization, following Sutter, Van Parys & Kuhn (2024).
    
    Solves:  min_{a in A}  max_{theta: I(theta || hat_theta) <= delta}
             E_{P_theta}[h(a, xi)]
    
    where the ambiguity set is constructed from a sufficient statistic
    and its large deviation rate function.
    """
    
    def __init__(
        self,
        decision_set_bounds: Tuple[float, float],
        cost_function: Callable,
        parametric_family: str = "gaussian",
    ):
        """
        Parameters
        ----------
        decision_set_bounds : tuple (low, high)
            Bounds for the decision variable a
        cost_function : Callable[[np.ndarray, np.ndarray], float]
            Cost function h(a, xi). Must be convex in a for tractability.
        parametric_family : str
            Parametric family for the DRO ambiguity set.
            Currently supported: "gaussian", "exponential"
        """
        self.bounds = decision_set_bounds
        self.h = cost_function
        self.family = parametric_family
        
    def _estimate_parameters(self, data: np.ndarray) -> dict:
        """Estimate parameters of the parametric family via MLE."""
        if self.family == "gaussian":
            mu_hat = np.mean(data, axis=0)
            sigma_hat = np.std(data, axis=0, ddof=1)
            return {"mu": mu_hat, "sigma": sigma_hat}
        elif self.family == "exponential":
            lam_hat = 1.0 / np.mean(data)
            return {"lambda": lam_hat}
        else:
            raise ValueError(f"Unknown family: {self.family}")
    
    def _rate_function(
        self, theta: dict, theta_hat: dict
    ) -> float:
        """
        Compute the rate function I(theta || theta_hat).
        
        For Gaussian N(mu, sigma^2), the rate function for the mean
        parameter (with known sigma) is:
            I(mu || mu_hat) = (mu - mu_hat)^2 / (2 * sigma^2)
        """
        if self.family == "gaussian":
            mu, mu_hat = theta["mu"], theta_hat["mu"]
            sigma = theta_hat["sigma"]
            return 0.5 * ((mu - mu_hat) / sigma) ** 2
        elif self.family == "exponential":
            lam, lam_hat = theta["lambda"], theta_hat["lambda"]
            return lam / lam_hat - 1 - np.log(lam / lam_hat)
        else:
            raise ValueError(f"Unknown family: {self.family}")
    
    def _worst_case_expectation(
        self, a: np.ndarray, theta_hat: dict, delta: float,
        n_samples: int = 1000
    ) -> float:
        """
        Approximate the worst-case expectation:
            sup_{theta: I <= delta} E_{P_theta}[h(a, xi)]
        
        Uses a sampling-based approach over the parameter space.
        """
        if self.family == "gaussian":
            mu_hat = theta_hat["mu"]
            sigma_hat = theta_hat["sigma"]
            
            # The ambiguity set for the mean is:
            #   |mu - mu_hat| <= sigma_hat * sqrt(2 * delta)
            # (for the univariate case)
            radius = sigma_hat * np.sqrt(2 * delta)
            
            # Sample theta values in the ambiguity set
            mu_values = np.linspace(mu_hat - radius, mu_hat + radius, n_samples)
            
            # For each mu, compute E[h(a, xi)] under N(mu, sigma_hat^2)
            expectations = []
            for mu in mu_values:
                # Sample from N(mu, sigma_hat^2)
                xi_samples = np.random.normal(mu, sigma_hat, size=1000)
                s = self.h(a, xi_samples)
                expectations.append(np.mean(s))
            
            # The worst case (maximization over the ambiguity set)
            return np.max(expectations)
        
        else:
            raise NotImplementedError(f"Family {self.family} not implemented yet")
    
    def fit(
        self,
        data: np.ndarray,
        delta: Optional[float] = None,
        a_init: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute the Pareto-optimal data-driven decision.
        
        Parameters
        ----------
        data : np.ndarray of shape (n_samples, n_features)
            Training data
        delta : float, optional
            Ambiguity set radius. If None, uses the Pareto-optimal
            delta = log(n) / n.
        a_init : np.ndarray, optional
            Initial guess for the decision variable
            
        Returns
        -------
        a_opt : np.ndarray
            Pareto-optimal decision
        info : dict
            Additional information (parameter estimates, etc.)
        """
        n = len(data)
        if delta is None:
            delta = np.log(n) / n  # Pareto-optimal radius
        
        # Estimate parameters
        theta_hat = self._estimate_parameters(data)
        
        # Inner function for optimization
        def objective(a):
            return self._worst_case_expectation(a, theta_hat, delta)
        
        # Initial guess
        if a_init is None:
            if self.family == "gaussian":
                a_init = np.array([theta_hat["mu"]])
            elif self.family == "exponential":
                a_init = np.array([1.0 / theta_hat["lambda"]])
        
        # Optimize
        bounds = [self.bounds]
        result = minimize(
            objective,
            a_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-8}
        )
        
        a_opt = result.x
        
        info = {
            "theta_hat": theta_hat,
            "delta": delta,
            "n_samples": n,
            "opt_val": result.fun,
            "success": result.success
        }
        
        return a_opt, info


# ============================================================
# Example: Pareto-Optimal Newsvendor Problem
# ============================================================
def demo_pareto_dro():
    """
    Demonstrate Pareto-optimal DRO on the classical newsvendor problem.
    
    The newsvendor must decide order quantity a to satisfy uncertain
    demand xi. The cost is:
        h(a, xi) = c_o * max(a - xi, 0) + c_u * max(xi - a, 0)
    
    where c_o is overage cost and c_u is underage cost.
    The optimal quantile solution under known distribution is:
        a* = F^{-1}(c_u / (c_o + c_u))
    
    We compare SAA, DRO with Pareto-optimal radius, and DRO with
    cross-validated radius.
    """
    np.random.seed(42)
    
    # True distribution: demand ~ N(100, 20)
    mu_true, sigma_true = 100.0, 20.0
    c_o, c_u = 1.0, 3.0
    optimal_quantile = c_u / (c_o + c_u)
    a_true = norm.ppf(optimal_quantile, mu_true, sigma_true)
    
    print(f"True optimal quantity (known distribution): {a_true:.2f}")
    print(f"Optimal quantile: {optimal_quantile:.2f}")
    
    # Generate training data
    n_train = 30
    data = np.random.normal(mu_true, sigma_true, size=n_train)
    
    # Newsvendor cost function
    def newsvendor_cost(a, xi):
        overage = c_o * np.maximum(a - xi, 0)
        underage = c_u * np.maximum(xi - a, 0)
        return overage + underage
    
    # 1. Sample Average Approximation
    a_saa = np.percentile(data, optimal_quantile * 100)
    cost_saa = np.mean(newsvendor_cost(a_saa, data))
    print(f"\nSAA optimal quantity: {a_saa:.2f}")
    print(f"SAA in-sample cost: {cost_saa:.4f}")
    
    # 2. Pareto-optimal DRO
    dro = ParetoDRO(
        decision_set_bounds=(0.0, 200.0),
        cost_function=newsvendor_cost,
        parametric_family="gaussian"
    )
    a_dro, info_dro = dro.fit(data)
    cost_dro = np.mean(newsvendor_cost(a_dro, data))
    print(f"\nPareto-DRO optimal quantity: {a_dro[0]:.2f}")
    print(f"Pareto-DRO in-sample cost: {cost_dro:.4f}")
    print(f"Pareto-DRO delta: {info_dro['delta']:.4f}")
    
    # 3. True out-of-sample performance
    n_test = 10000
    test_data = np.random.normal(mu_true, sigma_true, size=n_test)
    
    perf_saa = np.mean(newsvendor_cost(a_saa, test_data))
    perf_dro = np.mean(newsvendor_cost(a_dro[0], test_data))
    perf_opt = np.mean(newsvendor_cost(a_true, test_data))
    
    print(f"\n--- Out-of-sample performance ---")
    print(f"SAA test cost:        {perf_saa:.4f} "
          f"(gap: {100*(perf_saa-perf_opt)/perf_opt:.2f}%)")
    print(f"Pareto-DRO test cost: {perf_dro:.4f} "
          f"(gap: {100*(perf_dro-perf_opt)/perf_opt:.2f}%)")
    print(f"Oracle test cost:     {perf_opt:.4f}")
    
    # 4. Compare with cross-validated DRO
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    delta_candidates = np.logspace(-3, 0, 20)
    
    best_delta = None
    best_cv_cost = np.inf
    for delta in delta_candidates:
        cv_costs = []
        for train_idx, val_idx in kf.split(data):
            train_data = data[train_idx]
            val_data = data[val_idx]
            
            a_cv, _ = dro.fit(train_data, delta=delta)
            cost_cv = np.mean(newsvendor_cost(a_cv, val_data))
            cv_costs.append(cost_cv)
        
        mean_cv = np.mean(cv_costs)
        if mean_cv < best_cv_cost:
            best_cv_cost = mean_cv
            best_delta = delta
    
    a_dro_cv, _ = dro.fit(data, delta=best_delta)
    perf_dro_cv = np.mean(newsvendor_cost(a_dro_cv[0], test_data))
    
    print(f"\nCV-DRO best delta: {best_delta:.4f}")
    print(f"CV-DRO test cost:  {perf_dro_cv:.4f} "
          f"(gap: {100*(perf_dro_cv-perf_opt)/perf_opt:.2f}%)")
    
    return {
        "a_true": a_true, "a_saa": a_saa, "a_dro": a_dro,
        "perf_saa": perf_saa, "perf_dro": perf_dro, "perf_opt": perf_opt
    }


if __name__ == "__main__":
    results = demo_pareto_dro()
```

## References

Sutter, T., Van Parys, B. P. G., & Kuhn, D. (2024). A Pareto dominance principle for data-driven optimization. *Operations Research*, 72(5), 1976-1999. https://doi.org/10.1287/opre.2021.0609

Bertsimas, D., Gupta, V., & Kallus, N. (2018). Data-driven robust optimization. *Mathematical Programming*, 167(2), 235-292. https://doi.org/10.1007/s10107-017-1125-8

Mohajerin Esfahani, P., & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric. *Operations Research*, 66(4), 917-939. https://doi.org/10.1287/opre.2017.1712

Van Parys, B. P. G., Esfahani, P. M., & Kuhn, D. (2021). From data to decisions: Distributionally robust optimization is optimal. *Management Science*, 67(6), 3387-3402. https://doi.org/10.1287/mnsc.2020.3678

Shapiro, A. (2017). Distributionally robust stochastic programming. *SIAM Journal on Optimization*, 27(4), 2258-2275. https://doi.org/10.1137/16M108726X
