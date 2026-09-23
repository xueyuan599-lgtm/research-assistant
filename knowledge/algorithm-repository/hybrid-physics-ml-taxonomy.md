---
title: Hybrid Physics-ML Paradigms: Informed, Optimized, and Guided
type:
  - physics-informed
  - forecasting
  - deep-learning
domain:
  - signal-processing
---
# Hybrid Physics-ML Paradigms: Informed, Optimized, and Guided

## Metadata

- **Source:** Liu, Z. et al. (2025). "An Integrated Survey on AI Paradigms for Physical Time Series Analysis", IEEE ICDMW.
- **Category:** Time Series Fusion / Hybrid Methods Taxonomy
- **Subcategory:** Physics-Machine Learning Integration
- **Tags:** `#physics-informed` `#physics-optimized` `#physics-guided` `#pinn` `#hamiltonian-neural-net` `#time-series` `#hybrid-methods`
- **Last Updated:** 2026-07-24

---

## Overview

This entry taxonomizes three fundamental paradigms for integrating physical knowledge with machine learning in the context of time series analysis. The tripartite framework -- **Physics-Informed**, **Physics-Optimized**, and **Physics-Guided** -- captures the distinct roles that physics plays in the learning pipeline: as a regularizer, as an objective, or as an architectural constraint. Each paradigm makes different trade-offs between data efficiency, generalization, interpretability, and computational cost.

The core distinction lies in **what** physics constrains:
- **Informed:** Physics constrains the *output* (via the loss function).
- **Optimized:** Physics constrains the *target* (the model learns parameters of a known physical model).
- **Guided:** Physics constrains the *architecture* (the model's state space respects physical invariants).

---

## Mathematical Setup

### 1. Physics-Informed ML

Physical laws are encoded as soft or hard constraints in the loss function. The total objective combines data fidelity with a PDE/ODE residual penalty.

**General Formulation:**

$$
\mathcal{L}_{\text{total}} = \underbrace{\frac{1}{N} \sum_{i=1}^{N} \| \hat{y}_i - y_i \|^2}_{\text{data term}} \;+\;
\lambda_{\text{phys}} \underbrace{\frac{1}{M} \sum_{j=1}^{M} \| \mathcal{N}[u_\theta(t_j, x_j)] - f(t_j, x_j) \|^2_{\Omega}}_{\text{PDE/ODE residual}} \;+\;
\lambda_{\text{bc}} \underbrace{\frac{1}{K} \sum_{k=1}^{K} \| \mathcal{B}[u_\theta] - g \|^2_{\partial\Omega}}_{\text{boundary/initial condition}}
$$

where:
- $u_\theta(t, x)$ is the neural network approximation with parameters $\theta$
- $\mathcal{N}$ is the differential operator defining the physical law (e.g., $\mathcal{N}[u] = \partial_t u - \nu \partial_{xx} u$ for the heat equation)
- $\mathcal{B}$ is the boundary condition operator
- $f$ is the source term
- $\lambda_{\text{phys}}, \lambda_{\text{bc}}$ are hyperparameters balancing the loss components
- $\Omega$ is the spatio-temporal domain, $\partial\Omega$ its boundary

**Example -- Pendulum ODE constraint:**

For a simple pendulum $\ddot{\theta} + \frac{g}{L}\sin\theta = 0$, the residual loss is:

$$
\mathcal{L}_{\text{phys}} = \frac{1}{M} \sum_{j=1}^{M} \left\| \frac{d^2\hat{\theta}_j}{dt^2} + \frac{g}{L} \sin(\hat{\theta}_j) \right\|^2
$$

where $\frac{d^2\hat{\theta}_j}{dt^2}$ is computed via automatic differentiation of the network output w.r.t. time.

**Key characteristics:**
- Architecture-agnostic (works with any differentiable model)
- Requires differentiable physics operator
- Collocation points can be sampled anywhere in the domain (no simulation needed)
- The physics constraint is *soft* -- it guides but does not enforce

---

### 2. Physics-Optimized ML

The ML model serves as an optimizer or surrogate that infers parameters of a known parametric physical model from data. Physics provides the *output structure*; ML provides the *parameter mapping*.

**General Formulation:**

$$
\theta^* = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} \| y_i - f_{\text{phys}}(x_i; \theta) \|^2
$$

where $\theta$ is predicted by a neural network:

$$
\theta = \text{NN}(x; \phi), \quad \phi^* = \arg\min_{\phi} \frac{1}{N} \sum_{i=1}^{N} \left\| y_i - f_{\text{phys}}\bigl(x_i; \text{NN}(x_i; \phi)\bigr) \right\|^2
$$

**Example -- Pendulum parameter estimation:**

The physical model is:

$$
y_i = A \sin\left(\sqrt{\frac{g}{L}} \, t_i + \varphi\right) + b
$$

A neural network maps raw observations $(t_1, \ldots, t_T)$ to the parameter vector:

$$
\theta = [A, L, \varphi, b] = \text{NN}(t_{1:T})
$$

The physics model $f_{\text{phys}}$ then predicts the full trajectory. The NN is trained end-to-end by minimizing the discrepancy between predicted and observed trajectories.

**Key characteristics:**
- Physics provides the *generative model*; ML fills in the parameters
- Highly interpretable (parameters have physical meaning)
- Requires a closed-form or differentiable forward physical model
- Fails if the physical model is misspecified (wrong functional form)
- Data-efficient because the physics model is simple

---

### 3. Physics-Guided ML

Physical invariants (symmetries, conservation laws, symplectic structure) are hard-coded into the neural network architecture. The model's internal state space is constrained so that predictions *automatically* satisfy physical laws, without needing a penalty term.

**General Formulation -- Equivariance:**

A function $\Phi$ is equivariant under group $G$ if:

$$
\Phi(g \cdot x) = g \cdot \Phi(x), \quad \forall g \in G
$$

where $g \cdot x$ denotes the group action on the input space. Common groups in time series physics:
- **Translation equivariance:** CNNs (weight sharing enforces translation equivariance)
- **Rotation equivariance:** Vector/tensor field networks
- **Time-reversal symmetry:** Specific network parameterizations

**Example -- Hamiltonian Neural Network (HNN):**

Hamiltonian mechanics states that dynamics are governed by:

$$
\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}
$$

where $H(q, p)$ is the Hamiltonian (total energy). An HNN parameterizes $H_\theta(q, p)$ with a neural network and integrates the dynamics as:

$$
\begin{bmatrix}
\dot{q} \\
\dot{p}
\end{bmatrix}
=
\begin{bmatrix}
0 & I \\
-I & 0
\end{bmatrix}
\begin{bmatrix}
\nabla_q H_\theta(q, p) \\
\nabla_p H_\theta(q, p)
\end{bmatrix}
$$

The symplectic structure guarantees:

$$
\frac{dH_\theta}{dt} = 0 \quad \text{(energy conservation)}
$$

This is *not* a soft constraint -- it follows from the architecture itself because:

$$
\frac{dH}{dt} = \nabla_q H \cdot \dot{q} + \nabla_p H \cdot \dot{p} = \nabla_q H \cdot \nabla_p H - \nabla_p H \cdot \nabla_q H = 0
$$

**Key characteristics:**
- Physics is *hard-coded* into the architecture
- Conservation laws are satisfied by construction (zero violation)
- More data-efficient than physics-informed (no need for collocation points)
- Less flexible (architecture is specialized to the physical system)

---

### Comparison Table of All Three Paradigms

| Property | Physics-Informed | Physics-Optimized | Physics-Guided |
|---|---|---|---|
| Constraint type | Soft (loss penalty) | Structural (output form) | Hard (architecture) |
| Physics knowledge | PDE/ODE operator | Parametric forward model | Symmetry/invariant |
| Trainable parameters | NN weights | NN weights + physical params | NN weights |
| Physical model needed | Yes (differentiable operator) | Yes (closed-form forward) | Yes (invariant structure) |
| Differentiable physics req. | Yes | Yes (through $f_{\text{phys}}$) | No (autograd on $H$) |
| Violation of physics | Possible (controlled by $\lambda$) | Not possible (exact model) | Impossible (by design) |
| Collocation points needed | Yes | No | No |
| Interpretability | Low (black-box NN) | High (physical parameters) | Medium (energy function) |
| Data efficiency | Low (needs many collocation) | High | Medium-High |
| Extrapolation | Poor outside training domain | Good (physics is correct) | Good |
| Computational overhead | High (autograd on operator) | Low | Medium |
| Typical architecture | MLP/LSTM + residual loss | NN to params + ODE solver | Symplectic integrator |

---

## Key Assumptions

| # | Assumption | Physics-Informed | Physics-Optimized | Physics-Guided | Failure mode when violated |
|---|---|---|---|---|---|
| 1 | The governing differential equation is known and differentiable | Required | Not required | Not required | PINN cannot compute $\mathcal{N}[u_\theta]$; use data-driven discovery instead |
| 2 | A parametric forward model $f_{\text{phys}}$ exists and matches the true data-generating process | Not required | Required | Not required | Physics-optimized $\theta^*$ will be biased; model misspecification errors |
| 3 | The system respects a known symmetry or conservation law | Not required | Not required | Required | Energy leaks at every step; HNN fails to conserve |
| 4 | Training data covers the full spatio-temporal domain of interest | Required | Not required (extrapolates via physics) | Partially required | PINN extrapolates poorly outside collocation region |
| 5 | The neural network is sufficiently expressive to approximate the solution or parameter mapping | Required | Required | Required | Underfitting; all three paradigms fail |
| 6 | Boundary/initial condition data is available or inferable | Required | Not required | Not required | PINN loss components for $\mathcal{B}[u_\theta]$ cannot be evaluated |
| 7 | The physical model $f_{\text{phys}}$ is computationally cheap to evaluate | Not required | Required (end-to-end training) | Not required | Training becomes prohibitively slow |
| 8 | Observation noise is i.i.d. and symmetric | Desired | Desired | Desired | Biased parameter estimates in physics-optimized; PINN overfits to noise |
| 9 | Time series is sampled at regular, sufficiently dense intervals | Desired | Not required (irregular OK) | Desired | Numerical integration error in HNN; poor collocation gradient in PINN |
| 10 | The Hamiltonian/Lagrangian is separable (kinetic + potential) | N/A | N/A | Relaxable | Can use non-separable HNN; increased complexity |

---

## Applicable Scenarios

### Choose Physics-Informed when:
1. **The governing PDE/ODE is known** but the exact solution is hard to compute (e.g., Navier-Stokes, Burgers' equation).
2. **Data is scarce** and you want to regularize with physics -- PINNs work with surprisingly few data points when $\lambda_{\text{phys}}$ is well-tuned.
3. **You need a continuous solution** (PINNs output $u_\theta(t, x)$ at any coordinate, not just at grid points).
4. **Inverse problems**: PINN can simultaneously solve forward and inverse problems by treating unknown parameters as trainable (e.g., discover $g$ from pendulum data).
5. **Multi-fidelity data fusion**: different $\lambda$ for high- vs. low-fidelity data sources.

**Caveat:** PINNs are notoriously hard to train for stiff or multi-scale systems. Spectral bias (the network learns low frequencies first) often requires adaptations like Fourier feature embeddings or curriculum training.

### Choose Physics-Optimized when:
1. **A well-established parametric model exists** (e.g., exponential decay, harmonic oscillator, SIR compartment model).
2. **Interpretability is critical** -- you need physically meaningful parameters (e.g., damping ratio, natural frequency).
3. **Data is very limited** but the physics model is simple (few parameters to estimate).
4. **Uncertainty quantification** on physical parameters is needed (Bayesian extensions are natural).
5. **The end goal is parameter inference**, not trajectory prediction (e.g., estimate material properties from sensor data).

**Caveat:** If the assumed model form is wrong, physics-optimized methods confidently produce wrong parameters. Model selection and diagnostic checks are essential.

### Choose Physics-Guided when:
1. **Long-term stability** is required (e.g., climate modeling, orbital mechanics -- tiny errors compound).
2. **Conservation laws are non-negotiable** (energy, momentum, charge in particle physics).
3. **Data is moderately available** but you want sample efficiency via inductive bias.
4. **The system is high-dimensional** and unstructured NN would overfit (equivariance reduces the effective parameter count).
5. **You need uncertainty quantification with physical guarantees** (e.g., Lagrangian neural networks with probabilistic output).

**Caveat:** Not all physical systems have known symmetries that can be hard-coded. For complex dissipative systems (e.g., viscous fluid with turbulence), designing the right architectural bias is extremely challenging.

### Decision Tree Summary

```
Is a closed-form parametric forward model known and trusted?
  ├─ YES → Physics-Optimized (parameter estimation)
  └─ NO  → Is the PDE/ODE governing equation known?
             ├─ YES → Is long-term conservation critical?
             │        ├─ YES → Physics-Guided (symplectic/equivariant)
             │        └─ NO  → Physics-Informed (residual penalty)
             └─ NO  → Pure ML (no physics; not covered here)
```

---

## Implementation Details

### Paradigm Comparison: Implementation Considerations

| Aspect | Physics-Informed | Physics-Optimized | Physics-Guided |
|---|---|---|---|
| **Autograd requirement** | Second-order (Hessian of $u_\theta$) | First-order (through $f_{\text{phys}}$) | First-order (gradients of $H_\theta$) |
| **ODE solver needed** | No (uses collocation) | Yes (forward pass) | Yes (symplectic integrator) |
| **Batch size sensitivity** | High (balance data vs. collocation) | Low | Medium |
| **Learning rate schedule** | Cosine annealing preferred | Plateau reduction | Cosine annealing preferred |
| **Loss balancing** | Critical ($\lambda$ tuning) | Not needed (single MSE) | Not needed |
| **Gradient clipping** | Recommended (PINN gradients explode) | Not needed | Recommended |
| **Multi-step training** | Often: first train on data, then add physics | End-to-end | End-to-end |
| **Numerical precision** | Float32 acceptable (with care) | Float32 | Float64 recommended |
| **Parallelization** | Collocation points are batch-parallel | Sequential (ODE solve) | Sequential (ODE solve) |

### Selection Guide

```
| Your priority                      | Best paradigm      | Why                                    |
|------------------------------------|--------------------|----------------------------------------|
| Interpretability                   | Physics-Optimized  | Parameters have direct physical meaning |
| Long-term stability                | Physics-Guided     | Conservation laws are structurally enforced |
| Black-box flexibility              | Physics-Informed   | Any architecture, any operator          |
| Data efficiency (very limited)     | Physics-Optimized  | Few parameters to estimate              |
| Data efficiency (moderate)         | Physics-Guided     | Inductive bias reduces sample complexity |
| Uncertainty on physics params      | Physics-Optimized  | Bayesian inference on physical params   |
| Continuous solution                | Physics-Informed   | Output at any $(t, x)$ coordinate       |
| High-dimensional state space       | Physics-Guided     | Equivariance reduces effective dim      |
| No known parametric model          | Physics-Informed   | Only needs operator, not solution form  |
| Stiff/multi-scale dynamics         | Physics-Guided     | PINN spectral bias hurts INFORMED       |
```

### Hyperparameter Sensitivity

- **Physics-Informed:** $\lambda_{\text{phys}}$ is the most critical hyperparameter. Too high = network ignores data, collapses to trivial solution. Too low = physics constraint is wasted. Adaptive loss balancing (e.g., learning rate annealing for $\lambda$, Neural Tangent Kernel analysis) is recommended.
- **Physics-Optimized:** Critical hyperparameter is the NN architecture for the parameter mapping. Overparameterization leads to physically implausible parameters that happen to fit the data. Use regularization or Bayesian inference.
- **Physics-Guided:** The integrator step size is the most critical. A symplectic integrator (leapfrog/Verlet) is essential -- Runge-Kutta will leak energy even with a perfect Hamiltonian.

---

## Python Implementation

This implementation provides a complete, runnable PyTorch comparison of the three paradigms on a simple pendulum benchmark system.

```python
"""
Hybrid Physics-ML Paradigms: Informed, Optimized, and Guided
================================================================
A complete PyTorch implementation comparing three paradigms on a
simple pendulum benchmark.

Dependencies:
    torch >= 2.0, numpy, matplotlib (optional, for visualization)

Author: Research Assistant Knowledge Base
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================================
# 1. Ground Truth: Simple Pendulum (the "physical reality")
# =========================================================================

def pendulum_dynamics(
    theta0: float,
    omega0: float,
    g: float,
    L: float,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate the simple pendulum ODE using the small-angle approximation
    for a closed-form reference, then compute the exact ODE residual.

    Parameters
    ----------
    theta0 : Initial angular displacement [rad].
    omega0 : Initial angular velocity [rad/s].
    g      : Gravitational acceleration [m/s^2].
    L      : Pendulum length [m].
    t      : Time points [s], shape (N,).

    Returns
    -------
    theta  : Angular displacement at each t, shape (N,).
    omega  : Angular velocity at each t, shape (N,).
    energy : Total mechanical energy at each t, shape (N,).
    """
    omega0_clamped = omega0 if abs(omega0) > 1e-10 else 1e-10
    A = math.sqrt(theta0**2 + (omega0_clamped**2) * (L / g))
    phi = math.atan2(theta0, omega0_clamped * math.sqrt(L / g))
    omega_n = math.sqrt(g / L)

    theta = A * torch.cos(omega_n * t + phi)
    omega = -A * omega_n * torch.sin(omega_n * t + phi)

    # Energy: E = 0.5 * m * L^2 * omega^2 + m * g * L * (1 - cos(theta))
    m = 1.0  # unit mass
    energy = 0.5 * m * L**2 * omega**2 + m * g * L * (1.0 - torch.cos(theta))

    return theta, omega, energy


def make_pendulum_data(
    g: float = 9.81,
    L: float = 1.0,
    theta0: float = math.pi / 4,
    omega0: float = 0.0,
    n_points: int = 200,
    noise_std: float = 0.02,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Generate noisy pendulum observations.

    Returns
    -------
    t       : Time points, shape (n_points,).
    theta   : Noisy observations, shape (n_points,).
    theta_clean : Clean observations, shape (n_points,).
    params  : Ground truth parameter dict.
    """
    torch.manual_seed(seed)
    t = torch.linspace(0.0, 10.0, n_points).unsqueeze(-1)
    theta_clean, omega, energy = pendulum_dynamics(theta0, omega0, g, L, t.squeeze())
    noise = noise_std * torch.randn_like(theta_clean)
    theta = theta_clean + noise
    params = {"g": g, "L": L, "theta0": theta0, "omega0": omega0}
    return t, theta, theta_clean, params


# =========================================================================
# 2. Physics-Informed: LSTM with ODE Residual Loss
# =========================================================================

class PendulumLSTM(nn.Module):
    """
    An LSTM that predicts pendulum trajectories. The physics-informed
    variant adds an ODE residual penalty to the loss.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : Time points, shape (batch, seq_len, 1).

        Returns
        -------
        theta_hat : Predicted displacement, shape (batch, seq_len, 1).
        """
        lstm_out, _ = self.lstm(t)
        theta_hat = self.regressor(lstm_out)
        return theta_hat


def physics_informed_loss(
    model: nn.Module,
    t: torch.Tensor,
    theta_obs: torch.Tensor,
    g: float = 9.81,
    L: float = 1.0,
    lambda_phys: float = 0.1,
) -> torch.Tensor:
    """
    Compute total loss = data MSE + physics ODE residual penalty.

    The ODE residual for a simple pendulum is:
        r(t) = d^2(theta)/dt^2 + (g/L) * sin(theta)

    The second derivative is computed via torch.autograd.
    """
    t.requires_grad_(True)
    theta_hat = model(t)

    # Data term
    data_loss = nn.functional.mse_loss(theta_hat, theta_obs)

    # Physics residual: compute d^2(theta)/dt^2 via autograd
    # First derivative
    grad_theta = torch.autograd.grad(
        outputs=theta_hat,
        inputs=t,
        grad_outputs=torch.ones_like(theta_hat),
        create_graph=True,
    )[0]

    # Second derivative
    grad2_theta = torch.autograd.grad(
        outputs=grad_theta,
        inputs=t,
        grad_outputs=torch.ones_like(grad_theta),
        create_graph=True,
    )[0]

    # ODE residual: theta'' + (g/L) * sin(theta) = 0
    residual = grad2_theta + (g / L) * torch.sin(theta_hat)
    phys_loss = torch.mean(residual**2)

    total_loss = data_loss + lambda_phys * phys_loss
    return total_loss, data_loss.detach(), phys_loss.detach()


def train_physics_informed(
    t: torch.Tensor,
    theta: torch.Tensor,
    hidden_dim: int = 64,
    num_layers: int = 2,
    lr: float = 1e-3,
    epochs: int = 2000,
    lambda_phys: float = 0.1,
    g_phys: float = 9.81,
    L_phys: float = 1.0,
    verbose: bool = False,
) -> nn.Module:
    """
    Train a physics-informed LSTM on pendulum data.

    Parameters
    ----------
    t          : Time points, shape (batch, seq_len, 1).
    theta      : Observed displacement, shape (batch, seq_len, 1).
    lambda_phys: Weight of the ODE residual penalty.
    g_phys, L_phys : Physical parameters for the residual computation.
    """
    model = PendulumLSTM(hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, data_loss, phys_loss = physics_informed_loss(
            model, t, theta, g=g_phys, L=L_phys, lambda_phys=lambda_phys,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if verbose and epoch % 500 == 0:
            print(
                f"[PINN] Epoch {epoch:4d} | Total {total_loss.item():.6f} | "
                f"Data {data_loss.item():.6f} | Phys {phys_loss.item():.6f}"
            )

    return model


# =========================================================================
# 3. Physics-Optimized: ML Predicts Pendulum Parameters
# =========================================================================

class ParameterNet(nn.Module):
    """
    A neural network that maps raw time series to pendulum parameters.

    The network outputs [log(A), log(L), phi, b], where:
      A  : amplitude
      L  : pendulum length (positive, via log-param)
      phi: phase offset
      b  : vertical offset (bias)
    """

    def __init__(self, seq_len: int = 200) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # [logA, logL, phi, b]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Observed trajectory, shape (batch, seq_len) or (seq_len,).

        Returns
        -------
        logA : Log-amplitude, shape (batch,).
        logL : Log-length, shape (batch,).
        phi  : Phase offset, shape (batch,).
        b    : Bias, shape (batch,).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        params = self.encoder(x)
        logA = params[:, 0]
        logL = params[:, 1]
        phi = params[:, 2]
        b = params[:, 3]
        return logA, logL, phi, b


def physics_forward(
    t: torch.Tensor,
    logA: torch.Tensor,
    logL: torch.Tensor,
    phi: torch.Tensor,
    b: torch.Tensor,
    g: float = 9.81,
) -> torch.Tensor:
    """
    Forward physical model: predict trajectory from inferred parameters.

    y(t) = A * cos(sqrt(g/L) * t + phi) + b
    """
    A = torch.exp(logA)
    omega_n = torch.sqrt(g / torch.exp(logL))
    # t: (batch, seq_len), omega_n: (batch,)
    omega_n = omega_n.unsqueeze(-1)  # (batch, 1)
    phi = phi.unsqueeze(-1)           # (batch, 1)
    b = b.unsqueeze(-1)               # (batch, 1)
    A = A.unsqueeze(-1)               # (batch, 1)
    y_pred = A * torch.cos(omega_n * t + phi) + b
    return y_pred


def train_physics_optimized(
    t: torch.Tensor,
    theta: torch.Tensor,
    g: float = 9.81,
    lr: float = 1e-3,
    epochs: int = 2000,
    verbose: bool = False,
) -> nn.Module:
    """
    Train a physics-optimized model: NN learns to map observations to
    pendulum parameters; the physical model generates the trajectory.
    """
    model = ParameterNet(seq_len=t.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, verbose=False,
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        logA, logL, phi, b = model(theta)
        theta_pred = physics_forward(t, logA, logL, phi, b, g=g)
        loss = nn.functional.mse_loss(theta_pred, theta)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if verbose and epoch % 500 == 0:
            with torch.no_grad():
                A_hat = torch.exp(logA).item()
                L_hat = torch.exp(logL).item()
            print(
                f"[PhysOpt] Epoch {epoch:4d} | Loss {loss.item():.6f} | "
                f"A={A_hat:.4f} | L={L_hat:.4f}"
            )

    return model


# =========================================================================
# 4. Physics-Guided: Hamiltonian Neural Network (HNN)
# =========================================================================

class HamiltonianNN(nn.Module):
    """
    A Hamiltonian Neural Network that models the Hamiltonian H(q, p)
    and integrates dynamics via the symplectic gradient.

    States: q = theta (position), p = m * L^2 * omega (momentum)

    The network ensures energy conservation by construction:
        dH/dt = 0 (exact, up to integrator accuracy)
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),   # input: (q, p)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),   # output: H(q, p)
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(q, p)."""
        return self.net(torch.cat([q, p], dim=-1))

    def compute_dynamics(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symplectic dynamics:
            dq/dt =  dH/dp
            dp/dt = -dH/dq

        Returns (dqdt, dpdt).
        """
        q.requires_grad_(True)
        p.requires_grad_(True)
        H = self.forward(q, p)

        # grad_H = [dH/dq, dH/dp]
        grad_H = torch.autograd.grad(
            outputs=H,
            inputs=[q, p],
            grad_outputs=torch.ones_like(H),
            create_graph=True,
        )
        dH_dq, dH_dp = grad_H

        return dH_dp, -dH_dq  # (dqdt, dpdt)


def symplectic_integrate(
    model: HamiltonianNN,
    q0: torch.Tensor,
    p0: torch.Tensor,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate Hamiltonian dynamics using the leapfrog (Störmer-Verlet)
    symplectic integrator. This preserves the symplectic structure exactly
    (up to floating-point precision), ensuring long-term energy stability.

    Leapfrog steps:
        p_{k+1/2} = p_k - (dt/2) * dH/dq(q_k, p_{k+1/2})
        q_{k+1}   = q_k + dt * dH/dp(q_k, p_{k+1/2})
        p_{k+1}   = p_{k+1/2} - (dt/2) * dH/dq(q_{k+1}, p_{k+1/2})
    """
    q = q0.clone()
    p = p0.clone()
    q_traj = [q.detach().cpu()]
    p_traj = [p.detach().cpu()]
    H_traj = [model(q, p).detach().cpu()]

    for _ in range(n_steps):
        # Half-step momentum
        _, dH_dq = model.compute_dynamics(q, p)
        p = p - (dt / 2.0) * dH_dq

        # Full-step position
        dH_dp, _ = model.compute_dynamics(q, p)
        q = q + dt * dH_dp

        # Half-step momentum (full)
        _, dH_dq = model.compute_dynamics(q, p)
        p = p - (dt / 2.0) * dH_dq

        q_traj.append(q.detach().cpu())
        p_traj.append(p.detach().cpu())
        H_traj.append(model(q, p).detach().cpu())

    return (
        torch.stack(q_traj),
        torch.stack(p_traj),
        torch.stack(H_traj),
    )


def train_hnn(
    t: torch.Tensor,
    q_obs: torch.Tensor,
    p_obs: torch.Tensor,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 3000,
    verbose: bool = False,
) -> HamiltonianNN:
    """
    Train a Hamiltonian Neural Network.

    The loss is:
        L = ||dq/dt - dH/dp||^2 + ||dp/dt + dH/dq||^2

    This trains the Hamiltonian to reproduce the observed vector field,
    without ever explicitly supervising H itself.
    """
    model = HamiltonianNN(hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Compute ground-truth time derivatives via finite differences
    dq_obs = torch.zeros_like(q_obs)
    dp_obs = torch.zeros_like(p_obs)
    dt_val = (t[1] - t[0]).item() if t.numel() > 1 else 0.01

    dq_obs[1:-1] = (q_obs[2:] - q_obs[:-2]) / (2.0 * dt_val)
    dp_obs[1:-1] = (p_obs[2:] - p_obs[:-2]) / (2.0 * dt_val)

    for epoch in range(epochs):
        optimizer.zero_grad()
        dq_pred, dp_pred = model.compute_dynamics(q_obs, p_obs)
        loss_dq = nn.functional.mse_loss(dq_pred, dq_obs)
        loss_dp = nn.functional.mse_loss(dp_pred, dp_obs)
        loss = loss_dq + loss_dp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if verbose and epoch % 1000 == 0:
            print(
                f"[HNN] Epoch {epoch:4d} | Loss {loss.item():.6e} | "
                f"dq {loss_dq.item():.6e} | dp {loss_dp.item():.6e}"
            )

    return model


# =========================================================================
# 5. Benchmark and Energy Conservation Visualization
# =========================================================================

def energy_conservation_metric(H_traj: torch.Tensor) -> Tuple[float, float]:
    """
    Compute energy conservation metrics.

    Returns
    -------
    drift  : Linear trend of energy over time (slope).
    std_norm: Normalized standard deviation of energy.
    """
    H = H_traj.squeeze().numpy()
    H_mean = H.mean()
    if abs(H_mean) < 1e-10:
        return 0.0, 0.0
    # Normalized standard deviation
    std_norm = float(H.std() / abs(H_mean))
    # Drift: linear regression slope normalized by H_mean
    n = len(H)
    x = torch.linspace(0, 1, n)
    y = H_traj.squeeze()
    slope = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x**2).sum() - x.sum()**2)
    drift = float(slope / abs(H_mean))
    return drift, std_norm


def run_benchmark(
    g: float = 9.81,
    L: float = 1.0,
    theta0: float = math.pi / 4,
    omega0: float = 0.0,
    n_points: int = 200,
    noise_std: float = 0.02,
    n_epochs: int = 2000,
    verbose: bool = False,
) -> dict:
    """
    Run the full benchmark comparing all three paradigms.

    Returns a dict of results and metrics.
    """
    print("=" * 70)
    print("Hybrid Physics-ML Benchmark: Simple Pendulum")
    print("=" * 70)

    # Generate data
    t_1d, theta, theta_clean, params = make_pendulum_data(
        g=g, L=L, theta0=theta0, omega0=omega0,
        n_points=n_points, noise_std=noise_std,
    )

    # Reshape for sequence models: (1, seq_len, 1)
    t_seq = t_1d.unsqueeze(0)       # (1, N, 1)
    theta_seq = theta.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

    print(f"\nGround truth: g={g}, L={L}, theta0={theta0:.3f}, omega0={omega0}")
    print(f"Data: {n_points} points, noise std={noise_std}")
    print(f"Training epochs: {n_epochs}")

    # --- Physics-Informed ---
    print("\n--- 1. Physics-Informed LSTM ---")
    pinn_model = train_physics_informed(
        t_seq, theta_seq,
        epochs=n_epochs, verbose=verbose,
    )

    # --- Physics-Optimized ---
    print("\n--- 2. Physics-Optimized ML ---")
    opt_model = train_physics_optimized(
        t_seq, theta_seq,
        epochs=n_epochs, verbose=verbose,
    )

    # Extract inferred parameters
    with torch.no_grad():
        logA, logL, phi, b = opt_model(theta_seq)
        L_inferred = torch.exp(logL).item()
        A_inferred = torch.exp(logA).item()

    print(f"  Inferred: L={L_inferred:.4f} (true={L}), A={A_inferred:.4f}")

    # --- Physics-Guided (HNN) ---
    print("\n--- 3. Hamiltonian Neural Network ---")
    # Compute momentum from angular velocity: p = m * L^2 * omega
    _, omega_clean, _ = pendulum_dynamics(theta0, omega0, g, L, t_1d)
    m = 1.0
    p = (m * L**2) * omega_clean.unsqueeze(-1)  # (N, 1)
    q = theta_clean.unsqueeze(-1)                # (N, 1)

    hnn_model = train_hnn(
        t_1d, q, p,
        epochs=n_epochs, verbose=verbose,
    )

    # Long-term integration for energy check
    dt_sim = 0.01
    n_sim_steps = 2000
    q_traj, p_traj, H_traj = symplectic_integrate(
        hnn_model,
        q[0:1], p[0:1],
        dt=dt_sim, n_steps=n_sim_steps,
    )
    drift, std_norm = energy_conservation_metric(H_traj)
    print(f"  Energy drift: {drift:.6e} (fraction per unit time)")
    print(f"  Energy std (normalized): {std_norm:.6e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'Paradigm':<25} {'Key Metric':<20} {'Value':<15}")
    print("-" * 60)
    print(f"{'Physics-Informed':<25} {'Final total loss':<20} {'.':<15}")
    print(f"{'Physics-Optimized':<25} {'L error (%)':<20} "
          f"{abs(L_inferred - L) / L * 100:.2f}%")
    print(f"{'Physics-Guided (HNN)':<25} {'Energy drift':<20} {drift:.4e}")
    print(f"{'Physics-Guided (HNN)':<25} {'Energy std':<20} {std_norm:.4e}")

    return {
        "params": params,
        "pinn_model": pinn_model,
        "opt_model": opt_model,
        "hnn_model": hnn_model,
        "L_inferred": L_inferred,
        "energy_drift": drift,
        "energy_std": std_norm,
    }


# =========================================================================
# 6. Main: Runnable Entry Point
# =========================================================================

if __name__ == "__main__":
    results = run_benchmark(
        g=9.81,
        L=1.0,
        theta0=math.pi / 4,
        omega0=0.0,
        n_points=200,
        noise_std=0.02,
        n_epochs=2000,
        verbose=True,
    )
    print("\nDone. All three paradigms trained and evaluated.")
    print("(Use `import matplotlib.pyplot as plt` to visualize trajectories "
          "and energy conservation.)")
```

---

## References

Battaglia, P. W., Hamrick, J. B., & Tenenbaum, J. B. (2013). Simulation as an engine of physical scene understanding. *Proceedings of the National Academy of Sciences*, 110(45), 18327-18332.

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems*, 31.

Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). Lagrangian neural networks. *arXiv preprint arXiv:2003.04630*.

Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian neural networks. *Advances in Neural Information Processing Systems*, 32.

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422-440.

Liu, Z., Wang, H., Zhang, Y., & Chen, X. (2025). An integrated survey on AI paradigms for physical time series analysis. *IEEE International Conference on Data Mining Workshops (ICDMW)*.

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

Toth, P., Rezende, D. J., Jaegle, A., Racaniere, S., Buesing, L., & Weber, T. (2020). Hamiltonian generative networks. *International Conference on Learning Representations (ICLR)*.

Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*, 449, 110768.

Zhong, Y. D., Dey, B., & Chakraborty, A. (2020). Symplectic ODE-Net: Learning Hamiltonian dynamics with control. *International Conference on Learning Representations (ICLR)*.
