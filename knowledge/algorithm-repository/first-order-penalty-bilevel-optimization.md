# First-Order Penalty Methods for Bilevel Optimization

**Source**: Lu, Z., & Mei, S. (2024). First-order penalty methods for bilevel optimization. *SIAM Journal on Optimization*, 34(2), 1937-1969. https://doi.org/10.1137/23M1566753

**Category**: Operations Research / Optimization / Bilevel Optimization

## Mathematical Setup

A bilevel optimization problem takes the form:

$$
\begin{aligned}
\min_{x \in X, y \in Y} \quad & F(x, y) \\
\text{s.t.} \quad & G(x, y) \leq 0, \\
& y \in \mathcal{S}(x) := \arg\min_{y \in Y} \{ f(x, y) : g(x, y) \leq 0 \},
\end{aligned}
$$

where:
- $x \in \mathbb{R}^{n_x}$ is the upper-level (leader) decision variable
- $y \in \mathbb{R}^{n_y}$ is the lower-level (follower) decision variable
- $F(x,y)$ is the upper-level objective
- $f(x,y)$ is the lower-level objective
- $G(x,y) \leq 0$ are upper-level constraints
- $g(x,y) \leq 0$ are lower-level constraints
- $\mathcal{S}(x)$ is the solution set of the lower-level problem

Lu and Mei (2024) consider the case where the lower-level problem is **convex** and satisfies Slater's condition. The key insight is to reformulate the bilevel problem as a **penalized minimax problem**:

$$
\min_{x,y} \max_{u \geq 0} \left\{ F(x,y) + \frac{c}{2} \|y - \Pi_Y[y - \nabla_y f(x,y) - \nabla_y g(x,y)^\top u]\|^2 \right\}
$$

where $u$ are lower-level dual variables and $\Pi_Y$ denotes projection onto $Y$.

### Core Reformulation

Rather than solving the bilevel problem directly, Lu and Mei approximate it by solving a sequence of penalty problems:

$$
\min_{x,y} \Phi_\rho(x,y) := F(x,y) + \rho \Psi(x,y)
$$

where $\Psi(x,y)$ is a penalty function measuring the violation of the lower-level optimality conditions, and $\rho > 0$ is the penalty parameter.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Lower-level convexity | $f(x,\cdot)$ is convex for each $x$, $g(x,\cdot)$ is componentwise convex | The lower-level problem has a unique minimum or a well-defined solution set for each $x$ |
| Slater's condition | $\exists \hat{y} \in Y$ s.t. $g(x, \hat{y}) < 0$ for each $x$ | Strong duality holds for the lower-level problem, KKT conditions are necessary |
| Lipschitz gradients | $\nabla f$, $\nabla g$ are Lipschitz continuous | Gradient-based methods converge with established rates |
| Lower-level uniqueness | $\mathcal{S}(x)$ is a singleton for all $x$ | The bilevel problem has a well-defined implicit objective |
| Upper-level smoothness | $F$ and $G$ are smooth (possibly nonconvex) | First-order methods apply to the upper level |

## Applicable Scenarios

**When to use:**
- Hyperparameter optimization for machine learning models
- Meta-learning and few-shot learning
- Adversarial learning and game theory
- Optimal taxation and policy design
- Engineering design with nested optimization

**When NOT to use:**
- When the lower-level problem is nonconvex (the reformulation may break down)
- When high precision is required (first-order methods typically achieve moderate accuracy)
- When the lower-level solution set $\mathcal{S}(x)$ is multi-valued and nonconvex

**Comparison with alternatives:**
- **Classical KKT reformulation**: Requires solving large KKT systems; penalty methods are more scalable
- **Value-function approach**: Requires evaluating the optimal lower-level value, which can be nonsmooth; penalty methods provide a smooth approximation
- **Implicit differentiation**: Requires second-order information; penalty methods only use first-order information

## Algorithm / Method

### Lu-Mei First-Order Penalty Method

**Input**: Initial penalty $\rho_0 > 0$, penalty update factor $\tau > 1$, tolerance $\epsilon > 0$

1. **Initialize**: $k = 0$, $x^0$, $y^0$, $\rho_0$

2. **while** not converged **do**

3. **Solve penalty subproblem** approximately:
   $$
   (x^{k+1}, y^{k+1}) \approx \arg\min_{x,y} \Phi_{\rho_k}(x,y) := F(x,y) + \rho_k \Psi(x,y)
   $$
   using a gradient-based method (e.g., Adam, L-BFGS, or SGD)

4. **Check convergence**: If $\|\nabla \Phi_{\rho_k}(x^{k+1}, y^{k+1})\| \leq \epsilon$, break

5. **Update penalty**: $\rho_{k+1} = \tau \rho_k$

6. **Increment**: $k \leftarrow k + 1$

7. **end while**

**Output**: Approximately optimal $(x^*, y^*)$

### Convergence Guarantees

- **Global convergence**: Under standard assumptions, the method converges to an $\epsilon$-KKT point of the bilevel problem.
- **Complexity**: $O(\epsilon^{-4} \log \epsilon^{-1})$ for unconstrained bilevel problems and $O(\epsilon^{-7} \log \epsilon^{-1})$ for constrained bilevel problems.
- For the unconstrained case with strongly convex lower-level: $O(\epsilon^{-2} \log \epsilon^{-1})$.

## Implementation Details

**Key parameters:**
- $\rho_0$: Initial penalty (start small, e.g., 1.0)
- $\tau$: Penalty update factor (typically 1.5--5.0)
- Inner solver learning rate (depends on optimizer choice)
- Maximum inner iterations per penalty level

**Numerical considerations:**
- The penalty function $\Psi$ should be smooth for gradient-based optimization. A common choice is the squared residual of the lower-level KKT conditions.
- Warm-starting: Use $(x^k, y^k)$ as initial guess for the next penalty subproblem.
- Adaptive penalty: If the lower-level constraint violation is too large, increase $\rho$ more aggressively.

## Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, Optional

class BilevelPenaltyMethod:
    """
    First-order penalty method for bilevel optimization.
    
    Implements the Lu-Mei (2024) framework for solving bilevel problems
    of the form:
        min_{x,y} F(x,y)
        s.t.     y = argmin_{y'} f(x,y')
    
    where the lower-level problem f is convex in y.
    """
    
    def __init__(
        self,
        upper_obj: Callable,
        lower_obj: Callable,
        lower_grad: Callable,
        n_x: int,
        n_y: int,
        rho_0: float = 1.0,
        tau: float = 2.0,
        inner_lr: float = 0.01,
        inner_iters: int = 100,
        outer_iters: int = 20,
        tol: float = 1e-6
    ):
        """
        Parameters
        ----------
        upper_obj : Callable[[np.ndarray, np.ndarray], float]
            Upper-level objective F(x, y)
        lower_obj : Callable[[np.ndarray, np.ndarray], float]
            Lower-level objective f(x, y)
        lower_grad : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Gradient of lower-level objective w.r.t. y: ∇_y f(x, y)
        n_x : int
            Dimension of upper-level variable
        n_y : int
            Dimension of lower-level variable
        rho_0 : float
            Initial penalty parameter
        tau : float
            Penalty update multiplier (increase factor)
        inner_lr : float
            Learning rate for inner optimization
        inner_iters : int
            Maximum iterations per inner loop
        outer_iters : int
            Maximum outer iterations (penalty updates)
        tol : float
            Convergence tolerance
        """
        self.upper_obj = upper_obj
        self.lower_obj = lower_obj
        self.lower_grad = lower_grad
        self.n_x = n_x
        self.n_y = n_y
        self.rho = rho_0
        self.tau = tau
        self.inner_lr = inner_lr
        self.inner_iters = inner_iters
        self.outer_iters = outer_iters
        self.tol = tol
        
        # History tracking
        self.history = {
            'x': [], 'y': [], 'F': [], 'f': [], 'rho': []
        }
    
    def penalty_function(
        self, x: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Compute the penalty function Ψ(x, y) = ||∇_y f(x, y)||^2
        for unconstrained lower-level problems.
        
        For constrained lower-level, this would be ||KKT_residual||^2.
        """
        grad_y = self.lower_grad(x, y)
        return 0.5 * np.sum(grad_y ** 2)
    
    def penalized_objective(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Φ_ρ(x, y) = F(x, y) + ρ * Ψ(x, y)
        and its gradient w.r.t. (x, y).
        
        Uses finite differences for the gradient (in practice, use autodiff).
        
        Returns
        -------
        value : float
            Penalized objective value
        grad_x : np.ndarray
            Gradient w.r.t. x
        grad_y : np.ndarray
            Gradient w.r.t. y
        """
        eps = 1e-6
        
        # Value
        F_val = self.upper_obj(x, y)
        psi_val = self.penalty_function(x, y)
        phi_val = F_val + self.rho * psi_val
        
        # Gradient w.r.t. x (finite differences)
        grad_x = np.zeros(self.n_x)
        for i in range(self.n_x):
            x_plus = x.copy()
            x_plus[i] += eps
            F_plus = self.upper_obj(x_plus, y)
            psi_plus = self.penalty_function(x_plus, y)
            grad_x[i] = ((F_plus + self.rho * psi_plus) - phi_val) / eps
        
        # Gradient w.r.t. y (finite differences)
        grad_y = np.zeros(self.n_y)
        for i in range(self.n_y):
            y_plus = y.copy()
            y_plus[i] += eps
            F_plus = self.upper_obj(x, y_plus)
            psi_plus = self.penalty_function(x, y_plus)
            grad_y[i] = ((F_plus + self.rho * psi_plus) - phi_val) / eps
        
        return phi_val, grad_x, grad_y
    
    def solve_inner(
        self, x0: np.ndarray, y0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve min_{x,y} Φ_ρ(x,y) using gradient descent.
        """
        x, y = x0.copy(), y0.copy()
        
        for t in range(self.inner_iters):
            phi_val, grad_x, grad_y = self.penalized_objective(x, y)
            
            # Gradient descent step
            x = x - self.inner_lr * grad_x
            y = y - self.inner_lr * grad_y
            
            # Check local convergence
            grad_norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
            if grad_norm < self.tol:
                break
        
        return x, y
    
    def solve(
        self, x0: np.ndarray, y0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Main bilevel optimization loop.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial upper-level variable
        y0 : np.ndarray
            Initial lower-level variable
            
        Returns
        -------
        x_opt : np.ndarray
            Optimal upper-level variable
        y_opt : np.ndarray
            Optimal lower-level variable
        history : dict
            Optimization history
        """
        x, y = x0.copy(), y0.copy()
        
        for k in range(self.outer_iters):
            # Solve penalty subproblem
            x, y = self.solve_inner(x, y)
            
            # Record history
            F_val = self.upper_obj(x, y)
            f_val = self.lower_obj(x, y)
            self.history['x'].append(x.copy())
            self.history['y'].append(y.copy())
            self.history['F'].append(F_val)
            self.history['f'].append(f_val)
            self.history['rho'].append(self.rho)
            
            # Check convergence
            grad_norm = np.sqrt(np.sum(
                self.lower_grad(x, y) ** 2
            ))
            if grad_norm < self.tol:
                break
            
            # Increase penalty
            self.rho *= self.tau
        
        return x, y, self.history


# ============================================================
# Example: Hyperparameter Optimization for Ridge Regression
# ============================================================
def demo_bilevel_optimization():
    """
    Demonstrate bilevel optimization on a synthetic hyperparameter
    tuning problem:
    
    Upper-level (validation loss):
        min_λ (1/m) * ||X_val β*(λ) - y_val||^2
    
    Lower-level (training with ridge):
        β*(λ) = argmin_β (1/n) * ||X_train β - y_train||^2 + λ * ||β||^2
    
    where λ > 0 is the regularization strength (upper-level variable)
    and β are the regression coefficients (lower-level variable).
    
    This is a classic bilevel problem. We solve it using the penalty method,
    then compare against the analytical solution.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_train, n_val, n_features = 100, 50, 10
    true_beta = np.random.randn(n_features)
    
    X_train = np.random.randn(n_train, n_features)
    y_train = X_train @ true_beta + 0.1 * np.random.randn(n_train)
    
    X_val = np.random.randn(n_val, n_features)
    y_val = X_val @ true_beta + 0.1 * np.random.randn(n_val)
    
    # ----- Analytical baseline -----
    # For ridge regression, the bilevel problem has a closed-form
    # solution via leave-one-out or generalized cross-validation.
    # Here we do a grid search for the "true" optimal lambda.
    lambda_grid = np.logspace(-4, 2, 1000)
    val_errors = []
    for lam in lambda_grid:
        # Closed-form ridge solution
        I = np.eye(n_features)
        beta_hat = np.linalg.solve(
            X_train.T @ X_train + lam * I,
            X_train.T @ y_train
        )
        val_err = np.mean((X_val @ beta_hat - y_val) ** 2)
        val_errors.append(val_err)
    
    lambda_opt_grid = lambda_grid[np.argmin(val_errors)]
    val_err_opt = np.min(val_errors)
    
    print(f"Grid search optimal lambda: {lambda_opt_grid:.6f}")
    print(f"Grid search optimal val error: {val_err_opt:.6f}")
    
    # ----- Penalty method -----
    # We solve: min_λ F(λ, β) s.t. β minimizes f(λ, β)
    # where λ is 1D positive (we optimize log(λ) for unconstrained)
    # and β is n_features-dimensional.
    
    def upper_obj(log_lambda, beta):
        lam = np.exp(log_lambda[0])
        return float(np.mean((X_val @ beta - y_val) ** 2))
    
    def lower_obj(log_lambda, beta):
        lam = np.exp(log_lambda[0])
        return float(
            np.mean((X_train @ beta - y_train) ** 2) + lam * np.sum(beta ** 2)
        )
    
    def lower_grad(log_lambda, beta):
        lam = np.exp(log_lambda[0])
        # ∇_β f = (2/n) * X_train^T (X_train β - y_train) + 2*λ*β
        grad = (2.0 / n_train) * X_train.T @ (X_train @ beta - y_train) \
               + 2.0 * lam * beta
        return grad
    
    # Initial guess
    x0 = np.array([0.0])  # log(λ) = 0 => λ = 1
    y0 = np.linalg.lstsq(X_train, y_train, rcond=None)[0]  # OLS solution
    
    # Run bilevel optimization
    solver = BilevelPenaltyMethod(
        upper_obj=upper_obj,
        lower_obj=lower_obj,
        lower_grad=lower_grad,
        n_x=1,
        n_y=n_features,
        rho_0=0.1,
        tau=2.0,
        inner_lr=0.01,
        inner_iters=200,
        outer_iters=15,
        tol=1e-6
    )
    
    x_opt, y_opt, history = solver.solve(x0, y0)
    lambda_opt_penalty = np.exp(x_opt[0])
    val_err_penalty = upper_obj(x_opt, y_opt)
    
    print(f"\nPenalty method optimal lambda: {lambda_opt_penalty:.6f}")
    print(f"Penalty method val error: {val_err_penalty:.6f}")
    
    # Compare
    print(f"\nLambda relative difference: "
          f"{abs(lambda_opt_penalty - lambda_opt_grid) / lambda_opt_grid:.4f}")
    print(f"Val error relative difference: "
          f"{abs(val_err_penalty - val_err_opt) / val_err_opt:.4f}")
    
    # Print optimization history
    print(f"\nOuter iterations completed: {len(history['F'])}")
    print(f"Final penalty parameter: {history['rho'][-1]:.4f}")
    
    return solver, history


if __name__ == "__main__":
    solver, history = demo_bilevel_optimization()
```

## References

Lu, Z., & Mei, S. (2024). First-order penalty methods for bilevel optimization. *SIAM Journal on Optimization*, 34(2), 1937-1969. https://doi.org/10.1137/23M1566753

Ghadimi, S., & Wang, M. (2018). Approximation methods for bilevel programming. *arXiv preprint arXiv:1802.02246*.

Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., & Pontil, M. (2018). Bilevel programming for hyperparameter optimization and meta-learning. *Proceedings of ICML*, 1568-1577.

Colson, B., Marcotte, P., & Savard, G. (2007). An overview of bilevel optimization. *Annals of Operations Research*, 153(1), 235-256. https://doi.org/10.1007/s10479-007-0176-2

Hong, M., Wai, H.-T., Wang, Z., & Yang, Z. (2023). A two-timescale stochastic algorithm for bilevel optimization. *Mathematical Programming*, 198(2), 1125-1168. https://doi.org/10.1007/s10107-022-01812-9
