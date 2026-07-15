# Dynamic Optimization with Side Information

**Source**: Bertsimas, D., McCord, C., & Sturt, B. (2023). Dynamic optimization with side information. *European Journal of Operational Research*, 304(2), 634-651. https://doi.org/10.1016/j.ejor.2022.03.030

**Category**: Operations Research / Optimization / Stochastic Programming

## Mathematical Setup

Consider a **multi-stage stochastic optimization problem** where decisions are made sequentially over $T$ stages, and the decision-maker observes **side information** (covariates) before making each decision.

### Multi-Stage Problem with Side Information

$$
\begin{aligned}
\min_{a_1 \in \mathcal{A}_1} \quad & c_1^\top a_1 + \mathbb{E}\left[ \min_{a_2 \in \mathcal{A}_2(a_1, \xi_1)} c_2^\top a_2 + \cdots \right. \\
& \left. \cdots + \mathbb{E}\left[ \min_{a_T \in \mathcal{A}_T(a_{T-1}, \xi_{T-1})} c_T^\top a_T \mid z_T \right] \mid z_1 \right]
\end{aligned}
$$

where:
- $a_t$ is the decision at stage $t$
- $\xi_t$ is the random outcome realized after stage $t$
- $z_t$ is the **side information** (covariates) available at stage $t$
- $\mathcal{A}_t$ is the feasible set at stage $t$
- $c_t$ is the cost vector at stage $t$

The key challenge is that the conditional distribution $\mathbb{P}_{\xi_t | z_t}$ is unknown, and we only have historical data $(z_t^i, \xi_t^i)_{i=1}^n$.

### Bertsimas-McCord-Sturt Approach

The authors propose a **robust optimization approach** that incorporates side information through nonparametric machine learning:

1. **Feature-based ambiguity set**: For a given side信息 $z_t$, define a neighborhood $\mathcal{N}(z_t)$ of similar historical observations using a similarity measure (k-NN, kernel, random forest)

2. **Conditional robust optimization**: At each stage, solve:
   $$
   \min_{a_t} \max_{\xi \in \mathcal{U}_t(z_t)} c_t^\top a_t + \mathcal{Q}_{t+1}(a_t, \xi)
   $$
   where $\mathcal{U}_t(z_t)$ is an ambiguity set constructed from the historical outcomes of neighbors of $z_t$

### Linear Decision Rule Approximation

For tractability, the authors adopt **linear decision rules** (LDRs):
$$
a_t = A_t(z_t) + \sum_{s<t} B_{t,s} \xi_s
$$

Using LDRs, the multi-stage problem can be reformulated as a single convex optimization problem.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Side information | $z_t \in \mathbb{R}^d$ is observed before decision $a_t$ | Decisions can depend on covariates |
| Historical data | $(z_t^i, \xi_t^i)_{i=1}^n$ i.i.d. from the joint distribution | The conditional distribution can be estimated |
| Similarity measure | $d(z, z')$ is a metric or similarity function | Enables nearest-neighbor type methods |
| Bounded support | $\xi_t \in \Xi_t$ almost surely | The ambiguity set is bounded |
| Linear decision rules | $a_t = A_t(z_t) + \sum_{s<t} B_{t,s} \xi_s$ | The problem becomes a tractable LP/SOCP |
| Wasserstein concentration | $\mathbb{P}(W(\hat{\mathbb{P}}_{n,z}, \mathbb{P}_{\xi|z}) > \epsilon) \leq \text{decay}(\epsilon, n)$ | Asymptotic consistency of the conditional ambiguity set |

## Applicable Scenarios

**When to use:**
- Multi-stage inventory management with demand covariates (e.g., weather, economic indicators)
- Financial portfolio optimization with market features
- Shipment planning with traffic and weather covariates
- Energy systems with renewable generation forecasts
- Healthcare resource allocation with patient covariates

**When NOT to use:**
- When the number of stages $T$ is very large (the LDR approximation degrades for very long horizons)
- When the side information dimension $d$ is very high (curse of dimensionality in neighborhood estimation)
- When the relationship between covariates and outcomes is highly nonlinear and non-smooth (LDR may be restrictive)
- When the data generating process is nonstationary

**Comparison with alternatives:**
- **Stochastic programming with scenarios**: Does not incorporate side information naturally; scenarios are unconditional
- **Model predictive control (MPC)**: Uses point forecasts; the robust approach provides protection against forecast errors
- **Dynamic programming**: Suffers from the curse of dimensionality; the LDR approach is more scalable
- **Decision-focused learning**: Integrates prediction and optimization; the robust approach is more conservative but simpler

## Algorithm / Method

### Data-Driven Multi-Stage Optimization with Side Information

**Input**: Historical data $(z^i, \xi^i)_{i=1}^n$, similarity measure $d(\cdot, \cdot)$, confidence parameter $\alpha$

**At decision time** (given current side information $z_t$):

1. **Find neighbors**: Identify the $k$ nearest neighbors of $z_t$ in the historical data using $d$
   $$
   \mathcal{N}_k(z_t) = \{i : d(z_t, z^i) \leq d_{(k)}\}
   $$
   where $d_{(k)}$ is the $k$-th smallest distance

2. **Construct conditional ambiguity set**:
   $$
   \mathcal{U}_t(z_t) = \left\{\xi : \sum_{i \in \mathcal{N}_k(z_t)} w_i \|\xi - \xi^i\| \leq \epsilon_k \right\}
   $$
   where $w_i$ are similarity weights (e.g., inverse distance), and $\epsilon_k$ is the Wasserstein radius

3. **Solve the robust optimization problem** using LDRs:
   $$
   \min_{A, B} \quad \text{worst-case cost over } \mathcal{U}_t(z_t)
   $$
   subject to the LDR constraints

**Output**: Decision $a_t = A_t(z_t) + \sum_{s<t} B_{t,s} \xi_s$

### Theoretical Guarantees

- **Asymptotic optimality**: As $n \to \infty$, the optimal cost converges to the true optimal cost under mild conditions
- **Consistency**: The conditional ambiguity set shrinks around the true conditional distribution at rate $O(n^{-1/(d+2)})$ (for the k-NN case)
- **Finite-sample guarantee**: The optimal cost provides a high-probability upper bound on the true cost

## Implementation Details

**Key parameters:**
- $k$: Number of neighbors (typically $\sqrt{n}$ or cross-validated)
- $\epsilon_k$: Wasserstein radius (determines conservativeness)
- Similarity function $d$: Euclidean distance (scaled), Mahalanobis distance, or random forest proximity

**Numerical considerations:**
- Feature scaling is critical when the side information has different units
- The LDR approximation is a linear program (LP) for polyhedral costs, or a second-order cone program (SOCP) for quadratic costs
- For very large $n$, approximate nearest neighbor methods (e.g., KD-trees) are recommended

## Python Implementation

```python
import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple, Callable, List
import warnings
warnings.filterwarnings("ignore")


class DynamicOptimizationSideInfo:
    """
    Multi-stage dynamic optimization with side information using
    robust optimization and linear decision rules.
    
    Follows Bertsimas, McCord & Sturt (2023), EJOR.
    
    Implements a two-stage version:
        Stage 1: Decision a1 (before observing xi1)
        Stage 2: Decision a2(a1, xi1, z2) (after observing xi1 and side info z2)
    
    with the objective:
        min c1^T a1 + max_{xi1 in U(z1)} E[ min_{a2} c2^T a2 | z1, xi1 ]
    """
    
    def __init__(
        self,
        dim_a1: int,
        dim_a2: int,
        dim_xi: int,
        dim_z: int,
        c1: np.ndarray,
        c2: np.ndarray,
        bounds_a1: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        bounds_a2: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        k_neighbors: int = 10,
        epsilon: float = 0.1
    ):
        """
        Parameters
        ----------
        dim_a1 : int
            Dimension of first-stage decision
        dim_a2 : int
            Dimension of second-stage decision
        dim_xi : int
            Dimension of random outcomes
        dim_z : int
            Dimension of side information
        c1 : np.ndarray of shape (dim_a1,)
            First-stage cost coefficients
        c2 : np.ndarray of shape (dim_a2,)
            Second-stage cost coefficients
        bounds_a1 : tuple of np.ndarray, optional
            (lower, upper) bounds for a1
        bounds_a2 : tuple of np.ndarray, optional
            (lower, upper) bounds for a2
        k_neighbors : int
            Number of nearest neighbors for conditional ambiguity
        epsilon : float
            Wasserstein radius for ambiguity set
        """
        self.dim_a1 = dim_a1
        self.dim_a2 = dim_a2
        self.dim_xi = dim_xi
        self.dim_z = dim_z
        self.c1 = c1
        self.c2 = c2
        self.bounds_a1 = bounds_a1
        self.bounds_a2 = bounds_a2
        self.k = k_neighbors
        self.epsilon = epsilon
        
        # Placeholder for historical data
        self.X_hist = None  # historical xi values
        self.Z_hist = None  # historical z (side info) values
        
    def fit(self, Z: np.ndarray, X: np.ndarray):
        """
        Fit the model using historical data.
        
        Parameters
        ----------
        Z : np.ndarray of shape (n_samples, dim_z)
            Historical side information
        X : np.ndarray of shape (n_samples, dim_xi)
            Historical random outcomes
        """
        self.Z_hist = Z.copy()
        self.X_hist = X.copy()
        
        # Build nearest neighbors model
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.k, len(Z)),
            algorithm='kd_tree'
        )
        self.nn_model.fit(Z)
        
    def get_conditional_ambiguity_set(
        self, z: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Get the conditional ambiguity set for a given side information z.
        
        Returns the mean and radius of a Wasserstein ball around the
        empirical conditional distribution.
        """
        z = z.reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(z)
        
        # Get the neighboring xi values
        neighbor_xi = self.X_hist[indices[0]]
        
        # Mean of neighbors
        xi_mean = np.mean(neighbor_xi, axis=0)
        
        # Wasserstein radius: scaled by the spread of neighbors
        spread = np.mean(
            np.sqrt(np.sum((neighbor_xi - xi_mean) ** 2, axis=1))
        )
        radius = self.epsilon * (1.0 + spread)
        
        return xi_mean, radius
    
    def solve_two_stage(
        self, z1: np.ndarray, z2: np.ndarray
    ) -> dict:
        """
        Solve the two-stage robust optimization problem with side information.
        
        Uses a linear decision rule for the second stage:
            a2 = A * z2 + B * xi1 + d
        where A, B, d are coefficients to be optimized.
        
        Parameters
        ----------
        z1 : np.ndarray of shape (dim_z,)
            First-stage side information
        z2 : np.ndarray of shape (dim_z,)
            Second-stage side information
            
        Returns
        -------
        solution : dict
            Contains a1_opt, coefficients for a2, and costs
        """
        if self.Z_hist is None:
            raise ValueError("Must call fit() first")
        
        # Get conditional ambiguity for xi1 given z1
        xi1_mean, xi1_radius = self.get_conditional_ambiguity_set(z1)
        
        # The robust problem:
        #   min_{a1, A, B, d}  c1^T a1 + max_{xi1 in U(z1)} c2^T (A*z2 + B*xi1 + d)
        
        # With box ambiguity: xi1 in [xi1_mean - xi1_radius, xi1_mean + xi1_radius]
        # The worst case is at one of the extreme points.
        
        # For simplicity, we solve a deterministic approximation:
        #   Use the worst-case xi1 = xi1_mean + sign(c2^T B) * xi1_radius
        
        # For a demo, we simplify further: treat the worst-case xi1 as
        # xi1_mean + xi1_radius (assuming positive costs)
        
        xi1_wc = xi1_mean + xi1_radius  # worst-case scenario
        
        # Decision variables: a1 (dim_a1), then for a2 we have
        # a2 = A * z2 + B * xi1_wc + d
        # where A is (dim_a2 x dim_z), B is (dim_a2 x dim_xi), d is (dim_a2,)
        
        # Simple approach: optimize a1 and a2 jointly under worst-case xi1
        # by solving an LP
        
        # For demonstration: assume dim_a1 = dim_a2 = 1
        # a2 = alpha * z2 + beta * xi1_wc + gamma
        
        def solve_scalar():
            """Solve scalar version of the two-stage problem."""
            # Objective: c1 * a1 + c2 * (alpha * z2 + beta * xi1_wc + gamma)
            # We minimize over a1, alpha, beta, gamma
            
            # Bounds
            a1_low, a1_high = self.bounds_a1 if self.bounds_a1 else (0, 10)
            a2_low, a2_high = self.bounds_a2 if self.bounds_a2 else (0, 10)
            
            # Reformulate as an LP:
            #   min_{a1, alpha, beta, gamma} c1*a1 + c2*(alpha*z2 + beta*xi1_wc + gamma)
            #   s.t. a1 in [a1_low, a1_high]
            #        alpha*z2 + beta*xi1_wc + gamma in [a2_low, a2_high]
            
            c_obj = np.array([self.c1[0], 0.0, 0.0, 0.0])
            
            # A_eq * [a1, alpha, beta, gamma] = ...
            # The objective only explicitly depends on a1.
            # alpha, beta, gamma enter through the objective c2 * a2.
            
            # Full objective: c1*a1 + c2*alpha*z2 + c2*beta*xi1 + c2*gamma
            c_full = np.array([
                self.c1[0],
                self.c2[0] * z2[0] if np.ndim(z2) > 0 else self.c2[0] * z2,
                self.c2[0] * xi1_wc[0] if np.ndim(xi1_wc) > 0 else self.c2[0] * xi1_wc,
                self.c2[0]
            ])
            
            # Constraints: a2 in [low, high]
            # a2 = alpha * z2 + beta * xi1_wc + gamma
            a2_val = np.array([0, z2[0] if np.ndim(z2) > 0 else z2,
                               xi1_wc[0] if np.ndim(xi1_wc) > 0 else xi1_wc, 1])
            
            # a1 bound constraints
            bounds = [
                (a1_low, a1_high),   # a1
                (None, None),         # alpha
                (None, None),         # beta
                (None, None)          # gamma
            ]
            
            # Inequality: a2_low <= a2 <= a2_high
            A_ub = np.array([a2_val, -a2_val])
            b_ub = np.array([a2_high, -a2_low])
            
            result = linprog(
                c_full, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs'
            )
            
            if result.success:
                a1_opt = result.x[0]
                alpha_opt = result.x[1]
                beta_opt = result.x[2]
                gamma_opt = result.x[3]
                a2_opt = alpha_opt * z2 + beta_opt * xi1_wc + gamma_opt
                
                return {
                    "a1_opt": a1_opt,
                    "a2_opt": a2_opt,
                    "a2_coefficients": {
                        "alpha": alpha_opt,
                        "beta": beta_opt,
                        "gamma": gamma_opt
                    },
                    "total_cost": result.fun,
                    "xi1_worst_case": xi1_wc,
                    "success": True
                }
            else:
                return {
                    "a1_opt": None,
                    "a2_opt": None,
                    "total_cost": np.inf,
                    "success": False
                }
        
        return solve_scalar()
    
    def evaluate_policy(
        self, n_test: int = 1000
    ) -> Tuple[float, float]:
        """
        Evaluate the learned policy on test data.
        
        Returns
        -------
        mean_cost : float
            Average cost over test episodes
        std_cost : float
            Standard deviation of cost
        """
        if self.Z_hist is None or self.X_hist is None:
            raise ValueError("Must call fit() first")
        
        # Generate test episodes from the historical data
        n = len(self.Z_hist)
        costs = []
        
        for _ in range(n_test):
            idx = np.random.randint(n)
            z1 = self.Z_hist[idx]
            
            # Sample xi1 conditional on z1 (using a neighbor)
            _, neighbor_idx = self.nn_model.kneighbors(z1.reshape(1, -1))
            xi1_idx = np.random.choice(neighbor_idx[0])
            xi1 = self.X_hist[xi1_idx]
            
            # Second-stage side info (same as z1 for simplicity)
            z2 = z1
            
            # Solve for optimal decisions
            solution = self.solve_two_stage(z1, z2)
            
            if solution["success"]:
                a1 = solution["a1_opt"]
                a2 = solution["a2_opt"]
                
                # Compute actual cost
                cost = self.c1[0] * a1 + self.c2[0] * a2
                costs.append(cost)
        
        return np.mean(costs), np.std(costs)


# ============================================================
# Example: Inventory Management with Demand Covariates
# ============================================================
def demo_dynamic_side_info():
    """
    Demonstrate dynamic optimization with side information on an
    inventory management problem.
    
    Setting:
    - Stage 1: Order a1 units at cost c1 per unit
    - Stage 2: After observing demand xi1, order a2 units at cost c2
    - Side info z includes: weather index, economic indicator
    
    The goal is to minimize total ordering cost while meeting stochastic demand.
    """
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 200
    dim_z = 3
    
    # Side information: weather, economic, seasonal
    Z = np.random.randn(n_samples, dim_z)
    
    # Demand depends on side info with noise
    true_coeff = np.array([2.0, -1.5, 0.8])
    X = Z @ true_coeff + 0.5 * np.random.randn(n_samples)
    
    # Costs
    c1 = np.array([2.0])   # cheap advance order
    c2 = np.array([3.0])   # expensive rush order
    
    # Initialize optimizer
    optimizer = DynamicOptimizationSideInfo(
        dim_a1=1,
        dim_a2=1,
        dim_xi=1,
        dim_z=dim_z,
        c1=c1,
        c2=c2,
        bounds_a1=(0, 100),
        bounds_a2=(0, 50),
        k_neighbors=15,
        epsilon=0.2
    )
    
    # Fit on historical data
    optimizer.fit(Z, X.reshape(-1, 1))
    
    # ---- Compare with benchmark: no side info ----
    # Benchmark: ignore side info, use unconditional distribution
    benchmark_saa = np.mean(X)  # sample average
    
    print("=" * 60)
    print("DYNAMIC OPTIMIZATION WITH SIDE INFORMATION")
    print("=" * 60)
    print(f"\nCosts: advance={c1[0]}, rush={c2[0]}")
    print(f"True demand coefficient: {true_coeff}")
    print(f"Average demand (unconditional): {benchmark_saa:.2f}")
    
    # Test on out-of-sample data
    n_test = 100
    results_with_side_info = []
    results_without_side_info = []
    
    for t in range(n_test):
        # Generate new test point
        z_test = np.random.randn(dim_z)
        xi_true = z_test @ true_coeff + 0.5 * np.random.randn()
        
        # With side information
        solution = optimizer.solve_two_stage(z_test, z_test)
        if solution["success"]:
            a1_side = solution["a1_opt"]
            a2_side = solution["a2_opt"]
            # True second-stage need
            remaining = max(0, xi_true - a1_side)
            actual_a2 = min(remaining, 50)  # capped at max order
            cost_side = c1[0] * a1_side + c2[0] * actual_a2
            results_with_side_info.append(cost_side)
        
        # Without side information
        # Order the unconditional mean in stage 1
        remaining = max(0, xi_true - benchmark_saa)
        # In stage 2, we just order what's needed
        cost_no_side = c1[0] * benchmark_saa + c2[0] * remaining
        results_without_side_info.append(cost_no_side)
    
    avg_cost_side = np.mean(results_with_side_info)
    avg_cost_no_side = np.mean(results_without_side_info)
    improvement = 100 * (avg_cost_no_side - avg_cost_side) / avg_cost_no_side
    
    print(f"\n--- Out-of-sample cost comparison ---")
    print(f"With side information:    {avg_cost_side:.4f}")
    print(f"Without side information: {avg_cost_no_side:.4f}")
    print(f"Improvement:              {improvement:.2f}%")
    
    # Analyze the policy structure
    test_z = np.zeros(dim_z)
    sol_empty = optimizer.solve_two_stage(test_z, test_z)
    
    print(f"\n--- Policy structure ---")
    print(f"When z = (0, 0, 0):")
    print(f"  Advance order a1 = {sol_empty['a1_opt']:.2f}")
    print(f"  Rush order a2 = {sol_empty['a2_opt']:.2f}")
    if sol_empty['a2_coefficients']:
        coeffs = sol_empty['a2_coefficients']
        print(f"  a2 decision rule: alpha={coeffs['alpha']:.2f}, "
              f"beta={coeffs['beta']:.2f}, gamma={coeffs['gamma']:.2f}")
    
    return optimizer, results_with_side_info, results_without_side_info


if __name__ == "__main__":
    optimizer, costs_side, costs_no_side = demo_dynamic_side_info()
```

## References

Bertsimas, D., McCord, C., & Sturt, B. (2023). Dynamic optimization with side information. *European Journal of Operational Research*, 304(2), 634-651. https://doi.org/10.1016/j.ejor.2022.03.030

Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive analytics. *Management Science*, 66(3), 1025-1044. https://doi.org/10.1287/mnsc.2018.3253

Mohajerin Esfahani, P., & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric. *Operations Research*, 66(4), 917-939. https://doi.org/10.1287/opre.2017.1712

Ben-Tal, A., Goryashko, A., Guslitzer, E., & Nemirovski, A. (2004). Adjustable robust solutions of uncertain linear programs. *Mathematical Programming*, 99(2), 351-376. https://doi.org/10.1007/s10107-003-0454-y

Bertsimas, D., & Georghiou, A. (2015). Design of near optimal decision rules in multistage adaptive mixed-integer optimization. *Operations Research*, 63(3), 610-627. https://doi.org/10.1287/opre.2015.1365
