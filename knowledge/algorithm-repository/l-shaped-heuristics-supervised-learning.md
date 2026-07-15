# Fast Continuous and Integer L-Shaped Heuristics Through Supervised Learning

**Source**: Larsen, E., Frejinger, E., Gendron, B., & Lodi, A. (2024). Fast continuous and integer L-shaped heuristics through supervised learning. *INFORMS Journal on Computing*, 36(1), 203-223. https://doi.org/10.1287/ijoc.2022.0175

**Category**: Operations Research / Optimization / Stochastic Programming

## Mathematical Setup

Consider a **two-stage stochastic programming** problem:

$$
\begin{aligned}
\min_{x \in X} \quad & c^\top x + \mathbb{E}_{\xi}[Q(x, \xi)] \\
\end{aligned}
$$

where $Q(x, \xi)$ is the **second-stage value function**:

$$
Q(x, \xi) = \min_{y \in Y(x, \xi)} \{ q^\top y \}
$$

Here:
- $x$ are first-stage decisions (with feasible set $X$, possibly containing integer variables)
- $y$ are second-stage (recourse) decisions
- $\xi = (\xi_1, \dots, \xi_K)$ is a random vector with finite support $\{\xi^1, \dots, \xi^K\}$
- $c^\top x$ is the first-stage cost
- $q^\top y$ is the second-stage cost

Using the sample average approximation (SAA) with $K$ scenarios:

$$
\min_{x \in X} \quad c^\top x + \frac{1}{K} \sum_{k=1}^K Q(x, \xi^k)
$$

### The Integer L-Shaped Method

The classical integer L-shaped method (Laporte & Louveaux, 1993) solves this by:

1. **Relaxation**: Relax the second-stage costs, replacing $Q(x, \xi^k)$ with a lower bound $\theta_k$
2. **Master problem**: 
   $$
   \min_{x \in X, \theta} \quad c^\top x + \frac{1}{K} \sum_{k=1}^K \theta_k
   $$
3. **Optimality cuts**: Add cuts of the form $\theta_k \geq (Q(x^t, \xi^k) - L_k)(\sum_{i: x_i^t = 1} x_i - \sum_{i: x_i^t = 0} x_i + 1) + Q(x^t, \xi^k)$
   where $L_k$ is a lower bound on $Q(\cdot, \xi^k)$

### Larsen et al.'s Key Idea

The authors propose to **learn the second-stage value function** $\hat{Q}(x, \xi)$ using supervised learning. This replaces the expensive exact evaluation of $Q(x, \xi)$ with a fast approximation. The learned approximation is used as a **heuristic** within the L-shaped framework:

1. Train a predictor $\hat{Q}_\theta(x, \xi)$ on a dataset of $(x, \xi, Q(x, \xi))$ triples
2. Use $\hat{Q}_\theta$ as a **surrogate** for the exact second-stage value
3. The learned heuristic provides high-quality solutions in a fraction of the time

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Finite scenarios | $\xi$ has finite support $\{\xi^1, \dots, \xi^K\}$ | The expected value is a finite sum |
| Second-stage MIP | $Y(x, \xi^k)$ is a mixed-integer set | $Q(x, \xi^k)$ can be expensive to evaluate |
| Learning generalization | $\hat{Q}_\theta(x, \xi) \approx Q(x, \xi)$ for unseen $(x, \xi)$ | The surrogate can replace exact evaluation in heuristic search |
| Offline training | Training data can be generated before solving | Computational overhead is upfront |
| Repeatable instances | Similar problem instances are solved repeatedly | The cost of training is amortized over many solves |

## Applicable Scenarios

**When to use:**
- Stochastic programs where second-stage evaluation is computationally expensive
- Problems solved repeatedly with varying first-stage solutions (e.g., within a branch-and-bound tree)
- Fleet management, network design, and capacity planning with uncertainty
- Settings where exact optimality is less important than fast, high-quality solutions

**When NOT to use:**
- When second-stage problems are easy to solve (the overhead of ML is not justified)
- When the first-stage variable space is very high-dimensional (learning requires many training samples)
- When the problem instance is solved only once and training cost cannot be amortized

**Comparison with alternatives:**
- **Exact L-shaped method**: Guaranteed optimality but may be very slow; the ML heuristic provides fast approximate solutions
- **Sample average approximation (SAA)**: Solves a large deterministic equivalent; the ML-L-shaped approach can handle larger scenario sets
- **Progressive hedging**: Decomposition method that may require many iterations; the ML approach can provide a good warm start

## Algorithm / Method

### ML-Enhanced L-Shaped Heuristic

**Phase 1: Offline Training**

1. **Generate training data**: Sample first-stage solutions $x^1, \dots, x^N$ from the feasible set $X$
2. **Evaluate second-stage**: For each $x^i$ and scenario $\xi^k$, compute $Q(x^i, \xi^k)$ by solving the second-stage MIP
3. **Train predictor**: Learn $\hat{Q}_\theta(x, \xi)$ using supervised regression (e.g., random forest, neural network)

**Phase 2: Online Optimization**

1. **Initialize** with the ML surrogate
2. **Solve master problem** with the ML-approximated second-stage costs
3. **Optionally verify** candidate solutions with exact evaluation and add optimality cuts if needed
4. **Output** the best solution found

### Key Property

For problems where the exact L-shaped method is slow due to many optimality cuts, the ML heuristic can find near-optimal solutions ($<0.1\%$ gap) in less than $9\%$ of the time required by the exact method (Larsen et al., 2024).

## Implementation Details

**Key parameters:**
- Training set size: Typically $N = 500$--$2000$ first-stage solutions
- Predictor: Random forest (good out-of-the-box) or neural network (better with enough data)
- Feature engineering: Include both $x$ and scenario-specific features

**Numerical considerations:**
- The training data should cover the region of $X$ that is visited during optimization
- Active learning can reduce the number of expensive second-stage evaluations
- The ML predictor should be fast enough that evaluation is essentially free compared to MIP solves

## Python Implementation

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import linprog, minimize
from typing import Callable, Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")


class StochasticProgram:
    """
    Two-stage stochastic program:
    
    Stage 1: min_x c^T x + E[Q(x, xi)]
    Stage 2: Q(x, xi) = min_y q^T y   s.t. T x + W y = h(xi)
    
    For demonstration, we use a simple instance:
    - x in [0, 1] (continuous)
    - y is the recourse decision
    - xi follows a discrete distribution
    """
    
    def __init__(
        self,
        c: float,
        q: float,
        T: float,
        W: float,
        xi_scenarios: np.ndarray,
        probs: np.ndarray
    ):
        self.c = c        # first-stage cost coefficient
        self.q = q        # second-stage cost coefficient
        self.T = T        # technology matrix (scalar for simplicity)
        self.W = W        # recourse matrix
        self.xi = xi_scenarios
        self.probs = probs
        self.K = len(xi_scenarios)
    
    def second_stage_value(self, x: float, xi: float) -> float:
        """
        Solve the second-stage problem for given x and xi.
        
        Second-stage problem:
            min_y q * y
            s.t. W * y = xi - T * x
                 y >= 0
        """
        rhs = xi - self.T * x
        
        if self.W > 0:
            # y >= rhs / W if rhs > 0, otherwise y = 0
            if rhs > 0:
                y_opt = rhs / self.W
            else:
                y_opt = 0.0
            return self.q * y_opt
        else:
            return 0.0
    
    def evaluate(self, x: float) -> float:
        """Evaluate the full objective for a given first-stage decision."""
        first_cost = self.c * x
        second_cost = sum(
            self.probs[k] * self.second_stage_value(x, self.xi[k])
            for k in range(self.K)
        )
        return first_cost + second_cost


class MLSShapeHeuristic:
    """
    ML-enhanced L-shaped heuristic for two-stage stochastic programming.
    
    Follows Larsen et al. (2024) by training a supervised learning
    predictor to approximate the second-stage value function.
    """
    
    def __init__(
        self,
        stochastic_program: StochasticProgram,
        n_train_samples: int = 500,
        n_estimators: int = 100,
        test_size: float = 0.2
    ):
        """
        Parameters
        ----------
        stochastic_program : StochasticProgram
            The two-stage stochastic program to solve
        n_train_samples : int
            Number of first-stage solutions to sample for training
        n_estimators : int
            Number of trees in the random forest
        test_size : float
            Fraction of samples held out for testing
        """
        self.sp = stochastic_program
        self.n_train = n_train_samples
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.surrogate = None
        
    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data by sampling first-stage solutions
        and evaluating the second-stage value function.
        """
        X_train = []
        y_train = []
        
        # Sample first-stage solutions uniformly from [0, 1]
        x_samples = np.random.uniform(0, 1, self.n_train)
        
        for i, x in enumerate(x_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generating training sample {i+1}/{self.n_train}")
            
            # For each scenario, compute Q(x, xi^k)
            q_values = []
            for k in range(self.sp.K):
                q_val = self.sp.second_stage_value(x, self.sp.xi[k])
                q_values.append(q_val)
            
            # Average second-stage cost (expected value)
            avg_q = np.sum(
                [self.sp.probs[k] * q_values[k] for k in range(self.sp.K)]
            )
            
            # Features: x + scenario statistics
            features = np.array([
                x,
                np.mean(q_values),
                np.std(q_values),
                np.max(q_values),
                np.min(q_values),
                np.median(q_values)
            ])
            
            X_train.append(features)
            y_train.append(avg_q)
        
        return np.array(X_train), np.array(y_train)
    
    def train(self) -> dict:
        """
        Train the surrogate model for the second-stage value function.
        """
        print("Generating training data...")
        X, y = self._generate_training_data()
        
        print("Training random forest surrogate...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        self.surrogate = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.surrogate.fit(X_train, y_train)
        
        # Evaluate accuracy
        train_score = self.surrogate.score(X_train, y_train)
        test_score = self.surrogate.score(X_test, y_test)
        
        # Mean absolute error
        y_pred = self.surrogate.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-6))) * 100
        
        print(f"\nSurrogate model performance:")
        print(f"  Train R^2: {train_score:.4f}")
        print(f"  Test R^2:  {test_score:.4f}")
        print(f"  Test MAE:  {mae:.4f}")
        print(f"  Test MAPE: {mape:.2f}%")
        
        return {
            "train_r2": train_score,
            "test_r2": test_score,
            "mae": mae,
            "mape": mape,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }
    
    def surrogate_objective(self, x: float) -> float:
        """
        Compute the objective using the surrogate model.
        
        Surrogate objective: c*x + expected_second_stage_approx
        """
        if self.surrogate is None:
            raise ValueError("Must train the surrogate first")
        
        # Compute features for this x
        q_values = []
        for k in range(self.sp.K):
            q_val = self.sp.second_stage_value(x, self.sp.xi[k])
            q_values.append(q_val)
        
        features = np.array([[
            x,
            np.mean(q_values),
            np.std(q_values),
            np.max(q_values),
            np.min(q_values),
            np.median(q_values)
        ]])
        
        avg_q_pred = self.surrogate.predict(features)[0]
        return self.sp.c * x + avg_q_pred
    
    def solve_with_surrogate(self) -> Tuple[float, float]:
        """
        Solve the stochastic program using the surrogate objective.
        """
        # Simple 1D optimization over x in [0, 1]
        result = minimize_scalar(
            self.surrogate_objective,
            bounds=(0, 1),
            method="bounded"
        )
        
        x_opt = result.x
        obj_surrogate = result.fun
        
        # Verify with true objective
        obj_true = self.sp.evaluate(x_opt)
        
        return x_opt, obj_true, obj_surrogate
    
    def exact_solution(self) -> Tuple[float, float]:
        """
        Solve the stochastic program exactly (for comparison).
        Uses a fine grid search.
        """
        x_grid = np.linspace(0, 1, 10001)
        objs = [self.sp.evaluate(x) for x in x_grid]
        best_idx = np.argmin(objs)
        
        return x_grid[best_idx], objs[best_idx]


def minimize_scalar(fun, bounds, method="bounded"):
    """Minimize a scalar function of one variable on [bounds[0], bounds[1]]."""
    from scipy.optimize import minimize_scalar as _minimize_scalar
    return _minimize_scalar(fun, bounds=bounds, method=method)


# ============================================================
# Example: ML-Enhanced Stochastic Programming
# ============================================================
def demo_ml_lshape():
    """
    Demonstrate the ML-enhanced L-shaped heuristic on a simple
    two-stage stochastic program.
    
    Problem:
        min_{x in [0,1]}  2*x + E[ Q(x, xi) ]
        where Q(x, xi) = min_{y >= 0}  3*y  s.t.  2*y = xi - x
    
    The random variable xi follows a discrete distribution.
    
    We compare:
    1. Exact solution (grid search)
    2. ML-surrogate solution
    3. Comparison of computation time
    """
    np.random.seed(42)
    
    # Define stochastic program
    n_scenarios = 50
    xi_scenarios = np.random.exponential(scale=5.0, size=n_scenarios)
    xi_scenarios = np.sort(xi_scenarios)
    probs = np.ones(n_scenarios) / n_scenarios
    
    sp = StochasticProgram(
        c=2.0,
        q=3.0,
        T=1.0,
        W=2.0,
        xi_scenarios=xi_scenarios,
        probs=probs
    )
    
    print("=" * 60)
    print("ML-ENHANCED L-SHAPED HEURISTIC")
    print("=" * 60)
    
    print(f"\nProblem parameters:")
    print(f"  First-stage cost coefficient c = {sp.c}")
    print(f"  Second-stage cost coefficient q = {sp.q}")
    print(f"  Technology matrix T = {sp.T}")
    print(f"  Recourse matrix W = {sp.W}")
    print(f"  Number of scenarios: {sp.K}")
    print(f"  xi range: [{sp.xi.min():.2f}, {sp.xi.max():.2f}]")
    
    # --- Exact solution ---
    print("\n--- Exact solution ---")
    import time
    t0 = time.time()
    x_exact, obj_exact = MLSShapeHeuristic(sp).exact_solution()
    t_exact = time.time() - t0
    print(f"  Optimal x: {x_exact:.6f}")
    print(f"  Optimal cost: {obj_exact:.6f}")
    print(f"  Time: {t_exact:.4f}s")
    
    # --- ML solution ---
    print("\n--- ML-Enhanced solution ---")
    ml_solver = MLSShapeHeuristic(
        sp,
        n_train_samples=300,
        n_estimators=100
    )
    
    t0 = time.time()
    train_info = ml_solver.train()
    x_ml, obj_ml, obj_surr = ml_solver.solve_with_surrogate()
    t_ml = time.time() - t0
    
    print(f"  Optimal x (ML): {x_ml:.6f}")
    print(f"  True cost at ML solution: {obj_ml:.6f}")
    print(f"  Surrogate predicted cost: {obj_surr:.6f}")
    print(f"  Time: {t_ml:.4f}s")
    
    # --- Comparison ---
    optimality_gap = 100 * (obj_ml - obj_exact) / abs(obj_exact)
    speedup = t_exact / t_ml
    
    print(f"\n--- Comparison ---")
    print(f"  Exact optimal cost: {obj_exact:.6f}")
    print(f"  ML heuristic cost:  {obj_ml:.6f}")
    print(f"  Optimality gap:     {optimality_gap:.4f}%")
    print(f"  Speedup:            {speedup:.2f}x")
    
    # --- Sensitivity to training size ---
    print(f"\n--- Sensitivity to training size ---")
    train_sizes = [50, 100, 200, 500, 1000]
    
    for n_train in train_sizes:
        solver = MLSShapeHeuristic(sp, n_train_samples=n_train)
        solver.train()
        x_sol, obj_sol, _ = solver.solve_with_surrogate()
        gap = 100 * (obj_sol - obj_exact) / abs(obj_exact)
        print(f"  n_train={n_train:5d}: x={x_sol:.4f}, "
              f"cost={obj_sol:.4f}, gap={gap:.4f}%")
    
    return {
        "x_exact": x_exact,
        "obj_exact": obj_exact,
        "x_ml": x_ml,
        "obj_ml": obj_ml,
        "gap_pct": optimality_gap,
        "speedup": speedup
    }


if __name__ == "__main__":
    results = demo_ml_lshape()
```

## References

Larsen, E., Frejinger, E., Gendron, B., & Lodi, A. (2024). Fast continuous and integer L-shaped heuristics through supervised learning. *INFORMS Journal on Computing*, 36(1), 203-223. https://doi.org/10.1287/ijoc.2022.0175

Laporte, G., & Louveaux, F. V. (1993). The integer L-shaped method for stochastic integer programs with complete recourse. *Operations Research Letters*, 13(3), 133-142. https://doi.org/10.1016/0167-6377(93)90002-X

Bengio, Y., Lodi, A., & Prouvost, A. (2021). Machine learning for combinatorial optimization: A methodological tour d'horizon. *European Journal of Operational Research*, 290(2), 405-421. https://doi.org/10.1016/j.ejor.2020.07.063

Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive analytics. *Management Science*, 66(3), 1025-1044. https://doi.org/10.1287/mnsc.2018.3253

Birge, J. R., & Louveaux, F. V. (2011). *Introduction to stochastic programming* (2nd ed.). Springer. https://doi.org/10.1007/978-1-4614-0237-4
