# An Adaptive Alternating Direction Method of Multipliers

**Source**: Bartz, S., Campoy, R., & Phan, H. M. (2022). An adaptive alternating direction method of multipliers. *Journal of Optimization Theory and Applications*, 195(3), 1019-1055. https://doi.org/10.1007/s10957-022-02098-9

**Category**: Operations Research / Optimization / Convex Optimization

## Mathematical Setup

Consider a **composite convex optimization problem**:

$$
\min_{x \in \mathbb{R}^n} \quad f(x) + g(x)
$$

where $f$ is a **smooth convex function** (with $L$-Lipschitz gradient) and $g$ is a **proximable convex function** (possibly nonsmooth).

ADMM solves this by introducing an auxiliary variable $z$ and a linear constraint:

$$
\begin{aligned}
\min_{x, z \in \mathbb{R}^n} \quad & f(x) + g(z) \\
\text{s.t.} \quad & x - z = 0
\end{aligned}
$$

The augmented Lagrangian is:

$$
\mathcal{L}_\rho(x, z, u) = f(x) + g(z) + u^\top (x - z) + \frac{\rho}{2} \|x - z\|^2
$$

where $\rho > 0$ is the penalty parameter and $u$ is the dual variable.

### Standard ADMM Updates

The standard ADMM iteration ($k = 0, 1, 2, \dots$) is:

$$
\begin{aligned}
x^{k+1} &= \arg\min_x \left\{ f(x) + \frac{\rho}{2} \|x - z^k + u^k\|^2 \right\} \\
z^{k+1} &= \arg\min_z \left\{ g(z) + \frac{\rho}{2} \|x^{k+1} - z + u^k\|^2 \right\} \\
u^{k+1} &= u^k + (x^{k+1} - z^{k+1})
\end{aligned}
$$

where the dual variable $u$ is scaled (the standard scaled form ADMM).

### Adaptive ADMM (Bartz, Campoy & Phan, 2022)

The key contribution is an **adaptive penalty parameter** that adjusts $\rho$ based on local curvature information. The adaptive ADMM (aADMM) updates:

$$
\rho_k = \frac{\|s^k\|}{\|r^k\|}
$$

where $r^k = x^k - z^k$ is the **primal residual** and $s^k = \rho_{k-1}(z^k - z^{k-1})$ is the **dual residual**.

This choice balances primal and dual convergence and can be motivated by optimizing the local convergence rate. The update rule connects ADMM to the **adaptive Douglas-Rachford algorithm** and ensures that primal and dual residuals converge at similar rates.

### Connection to Douglas-Rachford Splitting

Bartz et al. show that ADMM is equivalent to the Douglas-Rachford (DR) splitting algorithm applied to the dual problem. The adaptive penalty parameter in aADMM corresponds to an adaptive step size in DR splitting, providing a theoretical foundation for the update rule.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Convexity | $f, g$ are convex functions | The problem has a unique solution (possibly non-unique for $g$) |
| Smoothness | $\nabla f$ is $L$-Lipschitz continuous | The $x$-update can be computed efficiently |
| Proximability | $\text{prox}_g(z) = \arg\min_x \{ g(x) + \frac{1}{2}\|x - z\|^2 \}$ is computable | The $z$-update has a closed form or efficient algorithm |
| Existence of solution | The optimal set is nonempty | The problem is well-posed |
| Strong convexity (optional) | $f$ or $g$ is $\mu$-strongly convex | Linear convergence rate can be established |

## Applicable Scenarios

**When to use:**
- Composite optimization problems where $f$ is smooth and $g$ is proximable
- Problems where the standard ADMM penalty parameter $\rho$ requires tuning
- Applications where balanced primal-dual convergence is important
- LASSO, elastic net, and other regularized regression problems
- Image processing (denoising, deblurring, inpainting)
- Consensus optimization and distributed computing

**When NOT to use:**
- When both $f$ and $g$ are smooth (accelerated gradient methods are faster)
- When $g$ does not have a closed-form proximal operator
- When high-precision solutions are required (ADMM typically converges fast to moderate accuracy, slow to high accuracy)

**Comparison with alternatives:**
- **Standard ADMM**: Requires manual tuning of $\rho$; aADMM adapts automatically and is more robust
- **Accelerated proximal gradient (FISTA)**: Faster for smooth + simple nonsmooth objectives; aADMM is more flexible for complex composite structures
- **Primal-dual hybrid gradient (PDHG)**: Similar flexibility; aADMM provides an adaptive step size based on primal-dual ratios

## Algorithm / Method

### Adaptive ADMM (aADMM)

**Input**: Initial guess $x^0$, $z^0$, $u^0$, initial penalty $\rho_0 > 0$, tolerance $\epsilon > 0$

1. **Initialize**: Set $k = 0$

2. **while** not converged **do**:
   
3. **Update primal variable $x$**:
   $$
   x^{k+1} = \arg\min_x \left\{ f(x) + \frac{\rho_k}{2} \|x - z^k + u^k\|^2 \right\}
   $$

4. **Update auxiliary variable $z$**:
   $$
   z^{k+1} = \arg\min_z \left\{ g(z) + \frac{\rho_k}{2} \|x^{k+1} - z + u^k\|^2 \right\}
   $$

5. **Update dual variable $u$**:
   $$
   u^{k+1} = u^k + (x^{k+1} - z^{k+1})
   $$

6. **Compute residuals**:
   $$
   r^{k+1} = x^{k+1} - z^{k+1}
   $$
   $$
   s^{k+1} = \rho_k (z^{k+1} - z^k)
   $$

7. **Adapt penalty parameter**:
   $$
   \rho_{k+1} = \min\left\{\rho_{\max}, \max\left\{\rho_{\min}, \frac{\|s^{k+1}\|}{\|r^{k+1}\|}\right\}\right\}
   $$
   
   Bound $\rho_{k+1}$ to $[\rho_{\min}, \rho_{\max}]$ for stability.

8. **Check convergence**:
   $$
   \|r^{k+1}\| \leq \epsilon \quad \text{and} \quad \|s^{k+1}\| \leq \epsilon
   $$

9. **Increment**: $k \leftarrow k + 1$

10. **end while**

**Output**: Approximately optimal $x^* \approx x^{k+1}$, $z^* \approx z^{k+1}$

### Convergence Guarantees

- **Global convergence**: The adaptive ADMM converges to a global optimum for any convex $f$, $g$ (under standard ADMM assumptions)
- **Rate**: $O(1/k)$ ergodic convergence in the objective (same as standard ADMM)
- **Linear convergence**: When $f$ is strongly convex and $g$ has a full-domain proximal operator, the adaptive method achieves linear convergence

## Implementation Details

**Key parameters:**
- $\rho_0$: Initial penalty (default 1.0; aADMM is not very sensitive to this)
- $\rho_{\min}$, $\rho_{\max}$: Bounds to prevent extreme values (e.g., $10^{-4}$ and $10^4$)
- $\epsilon$: Stopping tolerance

**Numerical considerations:**
- The adaptive update rule can cause $\rho$ to oscillate; clipping to $[\rho_{\min}, \rho_{\max}]$ prevents instability
- For poorly scaled problems, the adaptive method is significantly more robust than fixed-$\rho$ ADMM
- The ratio $\|s\|/\|r\|$ can become numerically unstable when both residuals are very small; use a safeguard (e.g., only adapt when $\|r\| > 10^{-8}$)

## Python Implementation

```python
import numpy as np
from typing import Callable, Optional, Tuple


class AdaptiveADMM:
    """
    Adaptive Alternating Direction Method of Multipliers (aADMM).
    
    Implements the adaptive penalty parameter update of
    Bartz, Campoy & Phan (2022), JOTA.
    
    Solves:  min_x  f(x) + g(x)
    
    using the reformulation:
        min_{x, z}  f(x) + g(z)
        s.t.        x - z = 0
    """
    
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        prox_g: Callable[[np.ndarray, float], np.ndarray],
        n: int,
        rho_0: float = 1.0,
        rho_min: float = 1e-4,
        rho_max: float = 1e4,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        f : Callable
            Smooth convex function f(x)
        grad_f : Callable
            Gradient of f
        prox_g : Callable[[np.ndarray, float], np.ndarray]
            Proximal operator of g: prox_g(v, t) = argmin_x g(x) + (1/(2t))||x-v||^2
        n : int
            Dimension of the problem
        rho_0 : float
            Initial penalty parameter
        rho_min : float
            Minimum penalty parameter
        rho_max : float
            Maximum penalty parameter
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Whether to print progress
        """
        self.f = f
        self.grad_f = grad_f
        self.prox_g = prox_g
        self.n = n
        self.rho = rho_0
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # History
        self.history = {
            'obj': [], 'primal_res': [], 'dual_res': [], 'rho': []
        }
    
    def solve(
        self,
        x0: Optional[np.ndarray] = None,
        z0: Optional[np.ndarray] = None,
        u0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve the composite optimization problem using adaptive ADMM.
        
        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial primal variable
        z0 : np.ndarray, optional
            Initial auxiliary variable
        u0 : np.ndarray, optional
            Initial dual variable
            
        Returns
        -------
        x_opt : np.ndarray
            Optimal solution
        info : dict
            Convergence information
        """
        # Initialize
        x = np.zeros(self.n) if x0 is None else x0.copy()
        z = np.zeros(self.n) if z0 is None else z0.copy()
        u = np.zeros(self.n) if u0 is None else u0.copy()
        rho = self.rho
        
        for k in range(self.max_iter):
            # --- x-update: minimize f(x) + (rho/2) * ||x - z + u||^2 ---
            # Using gradient descent on the augmented objective
            
            def augmented_f(x_candidate):
                return (
                    self.f(x_candidate)
                    + 0.5 * rho * np.sum((x_candidate - z + u) ** 2)
                )
            
            def grad_augmented_f(x_candidate):
                return (
                    self.grad_f(x_candidate)
                    + rho * (x_candidate - z + u)
                )
            
            # One step of gradient descent (in practice, solve more accurately)
            # For the demo, we use a few gradient steps
            x_new = x.copy()
            for _ in range(20):
                g = grad_augmented_f(x_new)
                # Backtracking line search
                step = 1.0
                for _ in range(20):
                    x_probe = x_new - step * g
                    if augmented_f(x_probe) <= augmented_f(x_new) \
                       - 0.5 * step * np.sum(g ** 2):
                        break
                    step *= 0.5
                x_new = x_new - step * g
            
            # --- z-update: proximal operator of g ---
            v = x_new + u
            z_new = self.prox_g(v, 1.0 / rho)
            
            # --- u-update ---
            u_new = u + (x_new - z_new)
            
            # --- Compute residuals ---
            primal_res = np.linalg.norm(x_new - z_new)
            dual_res = rho * np.linalg.norm(z_new - z)
            
            # --- Adapt rho ---
            if primal_res > 1e-8 and dual_res > 1e-8:
                rho_new = dual_res / primal_res
                rho_new = np.clip(rho_new, self.rho_min, self.rho_max)
            else:
                rho_new = rho
            
            # --- Record history ---
            obj_val = self.f(x_new) + self._evaluate_g(x_new)
            self.history['obj'].append(obj_val)
            self.history['primal_res'].append(primal_res)
            self.history['dual_res'].append(dual_res)
            self.history['rho'].append(rho)
            
            # --- Print progress ---
            if self.verbose and (k + 1) % 50 == 0:
                print(f"  Iter {k+1}: obj={obj_val:.6f}, "
                      f"r={primal_res:.2e}, s={dual_res:.2e}, "
                      f"rho={rho:.4f}")
            
            # --- Update for next iteration ---
            x, z, u = x_new, z_new, u_new
            rho = rho_new
            
            # --- Convergence check ---
            if primal_res < self.tol and dual_res < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {k+1}: "
                          f"obj={obj_val:.6f}")
                break
        
        info = {
            'iterations': k + 1,
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'final_rho': rho,
            'objective': obj_val,
            'history': self.history
        }
        
        return x, info
    
    def _evaluate_g(self, x: np.ndarray) -> float:
        """
        Evaluate g(x). This is used for tracking the objective.
        In practice, this may not be needed.
        """
        # Default: assume g is the indicator of nonnegative orthant
        # or L1 norm. Override for specific problems.
        return 0.0


# ============================================================
# Example: LASSO Regression via Adaptive ADMM
# ============================================================
def demo_adaptive_admm():
    """
    Demonstrate adaptive ADMM on the LASSO problem:
        min_x  (1/2n) * ||A x - b||^2 + lambda * ||x||_1
    
    Comparing adaptive ADMM with fixed-parameter ADMM.
    """
    np.random.seed(42)
    
    # Generate synthetic data
    n, p = 100, 50
    true_x = np.zeros(p)
    true_x[:10] = np.random.randn(10) * 2.0
    
    A = np.random.randn(n, p)
    b = A @ true_x + 0.1 * np.random.randn(n)
    
    lam = 0.1 * np.max(np.abs(A.T @ b)) / n
    
    print("=" * 60)
    print("ADAPTIVE ADMM FOR LASSO REGRESSION")
    print("=" * 60)
    print(f"n = {n}, p = {p}, lambda = {lam:.4f}")
    print(f"True nonzeros: {np.sum(true_x != 0)}")
    
    # --- Define functions for LASSO ---
    def f(x):
        return 0.5 / n * np.sum((A @ x - b) ** 2)
    
    def grad_f(x):
        return (1.0 / n) * A.T @ (A @ x - b)
    
    def prox_l1(v, t):
        """Proximal operator of t * lambda * ||x||_1."""
        return np.sign(v) * np.maximum(np.abs(v) - t * lam, 0)
    
    def evaluate_g(x):
        return lam * np.sum(np.abs(x))
    
    # --- Adaptive ADMM ---
    print("\n--- Adaptive ADMM ---")
    aadmm = AdaptiveADMM(
        f=f,
        grad_f=grad_f,
        prox_g=prox_l1,
        n=p,
        rho_0=1.0,
        max_iter=500,
        tol=1e-6,
        verbose=True
    )
    
    x_adapt, info_adapt = aadmm.solve()
    
    # --- Fixed ADMM with different rho values ---
    print("\n--- Fixed ADMM comparison ---")
    rho_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = []
    for rho_fixed in rho_values:
        admm_fixed = AdaptiveADMM(
            f=f,
            grad_f=grad_f,
            prox_g=prox_l1,
            n=p,
            rho_0=rho_fixed,
            rho_min=rho_fixed,
            rho_max=rho_fixed,  # no adaptation
            max_iter=500,
            tol=1e-6,
            verbose=False
        )
        
        x_fixed, info_fixed = admm_fixed.solve()
        
        obj_fixed = f(x_fixed) + evaluate_g(x_fixed)
        nz_fixed = np.sum(np.abs(x_fixed) > 1e-4)
        
        results.append({
            'rho': rho_fixed,
            'iter': info_fixed['iterations'],
            'obj': obj_fixed,
            'nz': nz_fixed
        })
        
        print(f"  rho={rho_fixed:<8.4f}  iters={info_fixed['iterations']:4d}  "
              f"obj={obj_fixed:.4f}  nonzeros={nz_fixed}")
    
    # --- Summary comparison ---
    obj_adapt = f(x_adapt) + evaluate_g(x_adapt)
    nz_adapt = np.sum(np.abs(x_adapt) > 1e-4)
    
    print(f"\n--- Final Comparison ---")
    print(f"{'Method':<20} {'Iters':>6} {'Objective':>12} {'Nonzeros':>10}")
    print("-" * 50)
    for r in results:
        print(f"{'Fixed rho='+str(r['rho']):<20} {r['iter']:>6d} "
              f"{r['obj']:>12.4f} {r['nz']:>10d}")
    print(f"{'Adaptive ADMM':<20} {info_adapt['iterations']:>6d} "
          f"{obj_adapt:>12.4f} {nz_adapt:>10d}")
    
    # --- Convergence plot data ---
    print(f"\n--- Adaptive rho trace (first 100 iterations) ---")
    rho_history = info_adapt['history']['rho'][:100]
    print(f"  Initial rho: {rho_history[0]:.4f}")
    print(f"  Final rho:   {rho_history[-1]:.4f}")
    print(f"  Min rho:     {min(rho_history):.4f}")
    print(f"  Max rho:     {max(rho_history):.4f}")
    
    # Quality of solution
    x_adapt_nz = np.abs(x_adapt) > 1e-4
    true_nz = np.abs(true_x) > 1e-4
    support_accuracy = np.mean(x_adapt_nz == true_nz)
    print(f"\n  Support recovery accuracy: {support_accuracy:.2%}")
    
    # Mean squared error of coefficients
    mse = np.mean((x_adapt - true_x) ** 2)
    print(f"  Coefficient MSE:           {mse:.6f}")
    
    return {
        "x_adapt": x_adapt,
        "x_true": true_x,
        "info": info_adapt,
        "fixed_results": results
    }


# ============================================================
# Additional Example: Sparse Logistic Regression
# ============================================================
def sparse_logistic_demo():
    """
    Demonstrate adaptive ADMM on sparse logistic regression.
    """
    np.random.seed(123)
    
    n, p = 200, 50
    true_beta = np.zeros(p)
    true_beta[:5] = np.random.randn(5) * 1.5
    
    X = np.random.randn(n, p)
    logits = X @ true_beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n) < probs).astype(float)
    
    lam = 0.01
    
    def f(beta):
        z = X @ beta
        return np.mean(np.log(1.0 + np.exp(-(2*y-1) * z)))
    
    def grad_f(beta):
        z = X @ beta
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return (1.0 / n) * X.T @ (sigmoid - y)
    
    def prox_l1(v, t):
        return np.sign(v) * np.maximum(np.abs(v) - t * lam, 0)
    
    print("\n" + "=" * 60)
    print("ADAPTIVE ADMM FOR SPARSE LOGISTIC REGRESSION")
    print("=" * 60)
    
    aadmm = AdaptiveADMM(
        f=f,
        grad_f=grad_f,
        prox_g=prox_l1,
        n=p,
        rho_0=1.0,
        max_iter=300,
        tol=1e-6,
        verbose=True
    )
    
    beta_opt, info = aadmm.solve()
    
    nz = np.sum(np.abs(beta_opt) > 1e-4)
    print(f"\nNonzero coefficients: {nz} / {p}")
    print(f"True nonzero: 5 / {p}")
    print(f"Objective: {info['objective']:.6f}")
    
    return beta_opt, info


if __name__ == "__main__":
    results = demo_adaptive_admm()
    # Uncomment for the second demo:
    # beta, info = sparse_logistic_demo()
```

## References

Bartz, S., Campoy, R., & Phan, H. M. (2022). An adaptive alternating direction method of multipliers. *Journal of Optimization Theory and Applications*, 195(3), 1019-1055. https://doi.org/10.1007/s10957-022-02098-9

Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1-122. https://doi.org/10.1561/2200000016

Douglas, J., & Rachford, H. H. (1956). On the numerical solution of heat conduction problems in two and three variables. *Transactions of the American Mathematical Society*, 82(2), 421-439. https://doi.org/10.1090/S0002-9947-1956-0084194-4

Eckstein, J., & Bertsekas, D. P. (1992). On the Douglas-Rachford splitting method and the proximal point algorithm for maximal monotone operators. *Mathematical Programming*, 55(1), 293-318. https://doi.org/10.1007/BF01581204

He, B., & Yuan, X. (2015). On the O(1/n) convergence rate of the Douglas-Rachford alternating direction method. *SIAM Journal on Numerical Analysis*, 50(2), 700-709. https://doi.org/10.1137/110836936
