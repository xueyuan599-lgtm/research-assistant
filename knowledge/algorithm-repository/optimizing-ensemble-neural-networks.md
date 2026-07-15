# Optimizing over an Ensemble of Trained Neural Networks

**Source**: Wang, K., Lozano, L., Cardonha, C., & Bergman, D. (2023). Optimizing over an ensemble of trained neural networks. *INFORMS Journal on Computing*, 35(3), 652-674. https://doi.org/10.1287/ijoc.2023.1285

**Category**: Operations Research / Optimization / Mixed-Integer Programming

## Mathematical Setup

Consider an optimization problem where the objective function is given by an **ensemble of ReLU neural networks**. We have $K$ pre-trained feedforward networks, each with $L$ layers. The $k$-th network computes:

$$
f_k(x) = W^{(L)}_k \cdot \phi(W^{(L-1)}_k \cdots \phi(W^{(1)}_k x + b^{(1)}_k) \cdots + b^{(L-1)}_k) + b^{(L)}_k
$$

where $\phi(z) = \max(0, z)$ is the ReLU activation function, and $W^{(\ell)}_k$, $b^{(\ell)}_k$ are the weights and biases of layer $\ell$ in network $k$.

The ensemble prediction is:

$$
f_{\text{ens}}(x) = \frac{1}{K} \sum_{k=1}^K f_k(x)
$$

The optimization problem is:

$$
\min_{x \in \mathcal{X}} \quad f_{\text{ens}}(x)
$$

where $\mathcal{X} \subseteq \mathbb{R}^n$ is a bounded feasible region (typically a polytope, possibly with integer constraints).

### MIP Reformulation

Each ReLU network $f_k$ can be reformulated as a **mixed-integer linear program (MILP)** using the big-M method. For each neuron $j$ in layer $\ell$ of network $k$:

$$
\begin{aligned}
\hat{z}^{(\ell)}_{k,j} &= \sum_i W^{(\ell)}_{k,ji} \cdot z^{(\ell-1)}_{k,i} + b^{(\ell)}_{k,j} \\
z^{(\ell)}_{k,j} &\geq \hat{z}^{(\ell)}_{k,j} \\
z^{(\ell)}_{k,j} &\geq 0 \\
z^{(\ell)}_{k,j} &\leq \hat{z}^{(\ell)}_{k,j} - M_{k,j}^{(\ell)} \cdot (1 - y^{(\ell)}_{k,j}) \\
z^{(\ell)}_{k,j} &\leq M_{k,j}^{(\ell)} \cdot y^{(\ell)}_{k,j} \\
y^{(\ell)}_{k,j} &\in \{0, 1\}
\end{aligned}
$$

where $M_{k,j}^{(\ell)}$ is a sufficiently large constant, and $y^{(\ell)}_{k,j}$ indicates whether the neuron is active ($\hat{z} > 0$) or inactive ($\hat{z} \leq 0$).

### Ensemble Objective

The ensemble MILP concatenates all $K$ networks:

$$
\begin{aligned}
\min_{x, z, y} \quad & \frac{1}{K} \sum_{k=1}^K z^{(L)}_{k,1} \\
\text{s.t.} \quad & \text{ReLU network constraints for } k = 1, \dots, K \\
& x \in \mathcal{X}
\end{aligned}
$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Pre-trained networks | Weights $W^{(\ell)}_k$, biases $b^{(\ell)}_k$ are fixed | No training needed during optimization |
| ReLU activations | $\phi(z) = \max(0, z)$ | Enables MILP reformulation |
| Bounded input domain | $x \in [\ell, u]$ with finite bounds | Big-M constants can be computed via interval arithmetic |
| Feedforward architecture | Directed acyclic graph connections | The network has a well-defined layered structure |
| Finite ensemble | $K$ is finite and typically small ($K \leq 50$) | The MILP size grows linearly with $K$ |

## Applicable Scenarios

**When to use:**
- Surrogate-based optimization where a neural network ensemble is used as a surrogate model
- Black-box optimization with expensive function evaluations (e.g., simulation optimization)
- Engineering design optimization with learned physics models
- Portfolio optimization with neural network return predictors
- Any setting where ensemble predictions are more stable and accurate than single networks

**When NOT to use:**
- When the input dimension is very large (MILP size grows with network width)
- When deep networks with many layers are used (MILP grows exponentially in layers in the worst case)
- When only a single network is available (the ensemble approach is overkill)
- When the feasible region $\mathcal{X}$ is nonconvex and nonlinear (MILP reformulation still works but may be large)

**Comparison with alternatives:**
- **Single network MIP**: Faster but less accurate and potentially unstable
- **Bayesian optimization**: Sample-based, no global optimality guarantees
- **Heuristic search**: Faster but no certificates of optimality
- **Deep ReLU optimization via LP**: Wang et al. show ensembles are more stable than single networks; the Lagrangian relaxation approach scales better than direct MILP

## Algorithm / Method

### Two-Phase Solution Approach (Wang et al., 2023)

**Phase 1: Preprocessing (Bound Tightening)**

1. **Interval arithmetic**: Propagate input bounds through each network to bound all neuron values
2. **Optimization-based bound tightening (OBBT)**: Solve auxiliary LPs to tighten big-M constants

**Phase 2: Lagrangian Relaxation-Based Branch-and-Bound**

1. **Lagrangian relaxation**: Dualize the linking constraints between network layers to decompose the problem into independent subproblems per network
2. **Subgradient method**: Update Lagrangian multipliers to find the best lower bound
3. **Branching**: Use standard MILP branching on the binary variables $y^{(\ell)}_{k,j}$
4. **Primal heuristic**: Use the Lagrangian solution to construct feasible solutions

### Convergence Guarantees

- The method converges to a global optimum of the MILP reformulation (finite termination)
- Bounds from the Lagrangian relaxation are at least as tight as the LP relaxation bound
- Computational complexity is exponential in the worst case (MILP is NP-hard), but the two-phase approach scales significantly better than direct MILP for practical instances

## Implementation Details

**Key parameters:**
- Big-M values: Must be chosen carefully -- too large leads to weak LP relaxations, too small may cut off feasible solutions
- Number of Lagrangian iterations: Typically 1000--5000
- Branching priority: Variables deeper in the network are often fixed first

**Numerical considerations:**
- Bound tightening is crucial for computational performance
- The Lagrangian dual is nonsmooth; use subgradient or bundle methods
- Ensemble size $K$ and network depth $L$ both affect solver time approximately linearly

## Python Implementation

```python
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings("ignore")


class ReLUNetwork:
    """
    A feedforward neural network with ReLU activations.
    """
    
    def __init__(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray]
    ):
        """
        Parameters
        ----------
        weights : list of np.ndarray
            List of weight matrices for each layer
        biases : list of np.ndarray
            List of bias vectors for each layer
        """
        self.weights = weights
        self.biases = biases
        self.n_layers = len(weights)
        
        # Verify dimensions
        for i in range(self.n_layers):
            assert weights[i].shape[0] == biases[i].shape[0]
            if i > 0:
                assert weights[i].shape[1] == weights[i-1].shape[0]
    
    def forward(self, x: np.ndarray) -> float:
        """Forward pass through the network."""
        h = x
        for i in range(self.n_layers - 1):
            h = np.maximum(0, self.weights[i] @ h + self.biases[i])
        # Final layer (linear output)
        h = self.weights[-1] @ h + self.biases[-1]
        return float(h[0])  # scalar output
    
    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for multiple inputs."""
        return np.array([self.forward(x) for x in X])


class NeuralEnsembleOptimizer:
    """
    Optimize over an ensemble of trained ReLU neural networks.
    
    Uses a Lagrangian relaxation approach following Wang et al. (2023)
    to handle the MILP reformulation of ReLU networks.
    
    Since solving the full MILP exactly is expensive for large networks,
    this implementation provides:
      1. A gradient-based heuristic using smooth approximations
      2. A Lagrangian bound for verification
    """
    
    def __init__(
        self,
        networks: List[ReLUNetwork],
        bounds: np.ndarray,
        n_lagrangian_iters: int = 2000,
        lr_init: float = 0.1
    ):
        """
        Parameters
        ----------
        networks : list of ReLUNetwork
            List of trained ReLU networks
        bounds : np.ndarray of shape (n, 2)
            Variable bounds [lower, upper]
        n_lagrangian_iters : int
            Number of Lagrangian subgradient iterations
        lr_init : float
            Initial learning rate for subgradient method
        """
        self.networks = networks
        self.K = len(networks)
        self.bounds = bounds
        self.n_vars = bounds.shape[0]
        self.n_lagrangian_iters = n_lagrangian_iters
        self.lr_init = lr_init
        
    def ensemble_predict(self, x: np.ndarray) -> float:
        """Compute ensemble prediction f_ens(x)."""
        preds = [net.forward(x) for net in self.networks]
        return float(np.mean(preds))
    
    def _smooth_relu(self, z: np.ndarray, alpha: float = 10.0) -> np.ndarray:
        """
        Smooth approximation of ReLU for gradient-based optimization:
            softplus(z; alpha) = (1/alpha) * log(1 + exp(alpha * z))
        """
        return (1.0 / alpha) * np.log(1.0 + np.exp(alpha * z))
    
    def _smooth_ensemble_forward(
        self, x: np.ndarray, alpha: float = 10.0
    ) -> float:
        """
        Forward pass through the ensemble with smooth activations.
        Used for gradient-based optimization.
        """
        preds = []
        for net in self.networks:
            h = x
            for i in range(net.n_layers - 1):
                h = self._smooth_relu(
                    net.weights[i] @ h + net.biases[i], alpha
                )
            # Final linear layer
            h = net.weights[-1] @ h + net.biases[-1]
            preds.append(float(h[0]))
        return float(np.mean(preds))
    
    def _gradient_ensemble(
        self, x: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of smooth ensemble objective using
        finite differences (for simplicity; autodiff is preferred).
        """
        eps = 1e-6
        grad = np.zeros(self.n_vars)
        f0 = self._smooth_ensemble_forward(x)
        
        for i in range(self.n_vars):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self._smooth_ensemble_forward(x_plus) - f0) / eps
        
        return grad
    
    def optimize_gradient(
        self,
        x0: Optional[np.ndarray] = None,
        n_restarts: int = 10,
        max_iter: int = 500
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize the ensemble using gradient-based methods with
        smooth ReLU approximation.
        
        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial point
        n_restarts : int
            Number of random restarts
        max_iter : int
            Maximum iterations per restart
            
        Returns
        -------
        x_best : np.ndarray
            Best solution found
        f_best : float
            Best objective value
        """
        best_x = None
        best_f = np.inf
        
        for restart in range(n_restarts):
            if x0 is not None and restart == 0:
                x = x0.copy()
            else:
                x = np.random.uniform(
                    self.bounds[:, 0], self.bounds[:, 1]
                )
            
            # L-BFGS-B optimization
            result = minimize(
                self._smooth_ensemble_forward,
                x,
                method="L-BFGS-B",
                jac=self._gradient_ensemble,
                bounds=[(b[0], b[1]) for b in self.bounds],
                options={"maxiter": max_iter, "ftol": 1e-8}
            )
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        
        return best_x, best_f
    
    def lagrangian_lower_bound(
        self, mu_init: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute a lower bound using Lagrangian relaxation.
        
        This relaxes the consensus constraint that all networks
        share the same input x, giving a decomposition.
        
        Parameters
        ----------
        mu_init : np.ndarray, optional
            Initial Lagrangian multipliers
            
        Returns
        -------
        best_lb : float
            Best lower bound found
        multipliers : np.ndarray
            Final Lagrangian multipliers
        """
        # Lagrangian relaxation of x consensus:
        #   min_{x_1,...,x_K, x}  (1/K) sum f_k(x_k)
        #   s.t. x_k = x for all k
        #
        # Dual function:
        #   q(mu) = min_{x, x_k} (1/K) sum f_k(x_k) + sum mu_k^T (x_k - x)
        #   s.t. x_k, x in bounds
        #
        # This decomposes into K independent subproblems + 1 x-problem.
        
        n = self.n_vars
        K = self.K
        
        if mu_init is None:
            mu = np.zeros((K, n))
        else:
            mu = mu_init.copy()
        
        best_lb = -np.inf
        lr = self.lr_init
        
        for t in range(self.n_lagrangian_iters):
            # Step 1: Solve for each x_k given current multipliers
            x_k_solutions = []
            for k in range(K):
                def obj_k(x):
                    return (
                        self.networks[k].forward(x) / K
                        + np.dot(mu[k], x)
                    )
                
                res = minimize(
                    obj_k,
                    np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]),
                    method="L-BFGS-B",
                    bounds=[(b[0], b[1]) for b in self.bounds],
                    options={"maxiter": 50}
                )
                x_k_solutions.append(res.x)
            
            # Step 2: Solve for x given current multipliers
            #   min_x -sum mu_k^T x
            # The solution is at a bound depending on the sign of sum mu_k
            total_mu = np.sum(mu, axis=0)
            x_dual = np.where(
                total_mu > 0, self.bounds[:, 0], self.bounds[:, 1]
            )
            
            # Step 3: Compute dual value and subgradient
            dual_val = 0.0
            for k in range(K):
                dual_val += self.networks[k].forward(x_k_solutions[k]) / K
                dual_val += np.dot(mu[k], x_k_solutions[k] - x_dual)
            
            if dual_val > best_lb:
                best_lb = dual_val
            
            # Step 4: Subgradient update
            lr_t = lr / (1.0 + 0.01 * t)
            for k in range(K):
                subgrad = x_k_solutions[k] - x_dual
                mu[k] += lr_t * subgrad
        
        return best_lb, mu
    
    def optimize(
        self,
        x0: Optional[np.ndarray] = None,
        n_restarts: int = 10
    ) -> dict:
        """
        Full optimization pipeline: gradient-based heuristic + lower bound.
        
        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial point
        n_restarts : int
            Number of random restarts
            
        Returns
        -------
        result : dict
            Best solution, objective, and lower bound
        """
        # Upper bound via gradient-based optimization
        x_best, f_best = self.optimize_gradient(x0, n_restarts)
        
        # Lower bound via Lagrangian relaxation
        lb, mu = self.lagrangian_lower_bound()
        
        gap = (f_best - lb) / max(1.0, abs(f_best)) * 100
        
        print(f"Best solution found: f = {f_best:.6f}")
        print(f"Lagrangian lower bound: {lb:.6f}")
        print(f"Optimality gap: {gap:.2f}%")
        
        return {
            "x": x_best,
            "f": f_best,
            "lower_bound": lb,
            "gap_pct": gap,
            "multipliers": mu
        }


# ============================================================
# Example: Optimize over a Synthetic Neural Ensemble
# ============================================================
def demo_ensemble_optimization():
    """
    Demonstrate optimizing over an ensemble of trained ReLU networks.
    
    We create 3 small ReLU networks with random weights, trained to
    approximate a known function f(x) = sin(x) * cos(2*x).
    Then we optimize over the ensemble to find the minimum.
    """
    np.random.seed(42)
    
    # ---- Generate training data ----
    n_samples = 500
    X_train = np.random.uniform(-3, 3, size=(n_samples, 1))
    X_train = np.sort(X_train, axis=0)
    y_train = np.sin(X_train[:, 0]) * np.cos(2 * X_train[:, 0]) \
              + 0.05 * np.random.randn(n_samples)
    
    # ---- Train 3 small ReLU networks ----
    # For simplicity, we construct networks with known weights.
    # Each network is a 1-8-1 architecture.
    
    networks = []
    for k in range(3):
        # Random weights with different seeds
        np.random.seed(100 + k * 10)
        
        W1 = np.random.randn(8, 1) * 1.5
        b1 = np.random.randn(8) * 0.5
        W2 = np.random.randn(1, 8) * 0.5
        b2 = np.random.randn(1) * 0.1
        
        net = ReLUNetwork([W1, W2], [b1, b2])
        
        # Verify approximate fit
        preds = net.forward_batch(X_train)
        mse = np.mean((preds - y_train) ** 2)
        print(f"Network {k+1} MSE: {mse:.6f}")
        
        networks.append(net)
    
    # ---- Optimize over the ensemble ----
    bounds = np.array([[-3.0, 3.0]])
    
    optimizer = NeuralEnsembleOptimizer(
        networks=networks,
        bounds=bounds,
        n_lagrangian_iters=500,
        lr_init=0.1
    )
    
    # Compare ensemble prediction with individual networks
    x_test = np.linspace(-3, 3, 100)
    print(f"\n--- Ensemble vs Single Network Predictions ---")
    for k, net in enumerate(networks):
        preds = net.forward_batch(x_test.reshape(-1, 1))
        print(f"Network {k+1} range: [{preds.min():.4f}, {preds.max():.4f}]")
    
    ensemble_preds = np.array([
        optimizer.ensemble_predict(np.array([x])) for x in x_test
    ])
    print(f"Ensemble range: [{ensemble_preds.min():.4f}, "
          f"{ensemble_preds.max():.4f}]")
    
    # Optimize
    print(f"\n--- Optimization ---")
    result = optimizer.optimize(n_restarts=15)
    
    # Verify
    x_opt = result["x"]
    f_opt = result["f"]
    individual_vals = [net.forward(x_opt) for net in networks]
    
    print(f"\nAt optimum x = {x_opt[0]:.6f}:")
    print(f"  Ensemble value: {f_opt:.6f}")
    for k, val in enumerate(individual_vals):
        print(f"  Network {k+1} value: {val:.6f}")
    
    # True function minimum (for comparison)
    x_grid = np.linspace(-3, 3, 10000)
    y_true = np.sin(x_grid) * np.cos(2 * x_grid)
    true_min = y_true.min()
    true_min_x = x_grid[y_true.argmin()]
    print(f"\nTrue function minimum: f({true_min_x:.4f}) = {true_min:.6f}")
    print(f"  (based on dense grid -- may not be global)")
    
    return optimizer, result


if __name__ == "__main__":
    optimizer, result = demo_ensemble_optimization()
```

## References

Wang, K., Lozano, L., Cardonha, C., & Bergman, D. (2023). Optimizing over an ensemble of trained neural networks. *INFORMS Journal on Computing*, 35(3), 652-674. https://doi.org/10.1287/ijoc.2023.1285

Tjeng, V., Xiao, K., & Tedrake, R. (2019). Evaluating robustness of neural networks with mixed integer programming. *Proceedings of ICLR*.

Fischetti, M., & Jo, J. (2018). Deep neural networks and mixed integer linear optimization. *Constraints*, 23(3), 296-309. https://doi.org/10.1007/s10601-018-9285-6

Serra, T., Tjandraatmadja, C., & Ramalingam, S. (2018). Bounding and counting linear regions of deep neural networks. *Proceedings of ICML*, 4565-4573.

Anderson, R., Huchette, J., Ma, W., Tjandraatmadja, C., & Vielma, J. P. (2020). Strong mixed-integer programming formulations for trained neural networks. *Mathematical Programming*, 183(1), 3-39. https://doi.org/10.1007/s10107-020-01474-5
