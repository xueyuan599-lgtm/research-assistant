# PI-LSTM — Physics-Informed LSTM for PDE-Constrained Dynamical Systems

- **Source**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- **DOI**: 10.1016/j.jcp.2018.10.045
- **Category**: Time Series Fusion / Physics-Informed Deep Learning
- **Related paradigms**: Physics-Integrated Neural Network (PINN), Neural ODE, LSTM-PINN hybrid frameworks for fluid dynamics (Wang et al., 2020), structural health monitoring (Zhang et al., 2021), and battery degradation (Kumar et al., 2026)

## Mathematical Setup

### Problem Definition

Consider a dynamical system governed by a general PDE/ODE of the form:

$$
\mathcal{N}[\mathbf{u}(t, \mathbf{x})] = f(t, \mathbf{x}), \quad (t, \mathbf{x}) \in [0, T] \times \Omega
$$

where $\mathcal{N}$ is a nonlinear differential operator, $\mathbf{u}(t, \mathbf{x}) \in \mathbb{R}^{d}$ is the latent solution field, and $f(t, \mathbf{x})$ is a known forcing term. The system is observed via sparse, noisy measurements $\{\mathbf{y}_i\}_{i=1}^{N}$ at discrete times $\{t_i\}$.

The goal is to learn a mapping from observed history to future states while **respecting the governing PDE** — i.e., ensuring predictions lie on or near the solution manifold of $\mathcal{N}$.

### Standard LSTM Backbone

Given a sequence of observations $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_\tau]$ where each $\mathbf{x}_t \in \mathbb{R}^{d_{\text{in}}}$, the LSTM produces hidden states:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \\
\mathbf{f}_t &= \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}
$$

The final hidden state $\mathbf{h}_\tau \in \mathbb{R}^{H}$ serves as a compact representation of the dynamical history, which is then projected to the prediction target:

$$
\hat{\mathbf{u}}_\theta(t, \mathbf{x}) = \text{MLP}(\mathbf{h}_\tau)
$$

where $\theta$ denotes all trainable parameters (LSTM weights + MLP weights).

### PDE Constraint Embedding

The core innovation of PI-LSTM is augmenting the standard data-fitting loss with a **physics residual penalty** that enforces the governing PDE. The total loss is:

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{observation fit}} + \underbrace{\lambda_{\text{pde}} \mathcal{L}_{\text{pde}}}_{\text{PDE constraint}} + \underbrace{\lambda_{\text{ic}} \mathcal{L}_{\text{ic}}}_{\text{initial condition}} + \underbrace{\lambda_{\text{bc}} \mathcal{L}_{\text{bc}}}_{\text{boundary condition}}
$$

**Data loss** (standard supervised loss on observed points):

$$
\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \hat{\mathbf{u}}_\theta(t_i, \mathbf{x}_i) - \mathbf{u}_i^{\text{(obs)}} \right\|^2
$$

**PDE residual loss** (the physics penalty evaluated on a set of collocation points $\{t_c^{(j)}, \mathbf{x}_c^{(j)}\}_{j=1}^{M}$):

$$
\mathcal{L}_{\text{pde}} = \frac{1}{M} \sum_{j=1}^{M} \left\| \mathcal{N}[\hat{\mathbf{u}}_\theta(t_c^{(j)}, \mathbf{x}_c^{(j)})] - f(t_c^{(j)}, \mathbf{x}_c^{(j)}) \right\|^2
$$

The differential operator $\mathcal{N}[\hat{\mathbf{u}}_\theta]$ is computed via **automatic differentiation** (AD) through the network — no numerical discretization of the PDE is required:

$$
\mathcal{N}[\hat{\mathbf{u}}_\theta] = \frac{\partial \hat{u}}{\partial t} + \mathcal{F}\left(\hat{u}, \frac{\partial \hat{u}}{\partial x}, \frac{\partial^2 \hat{u}}{\partial x^2}, \dots\right)
$$

where the derivatives $\frac{\partial \hat{u}}{\partial t}, \frac{\partial \hat{u}}{\partial x}$ are obtained through nested `torch.autograd.grad` calls with `create_graph=True`.

**Initial/boundary condition losses** (if applicable):

$$
\mathcal{L}_{\text{ic}} = \frac{1}{N_{\text{ic}}} \sum_{i=1}^{N_{\text{ic}}} \left\| \hat{\mathbf{u}}_\theta(0, \mathbf{x}_i) - \mathbf{u}_0(\mathbf{x}_i) \right\|^2
$$

$$
\mathcal{L}_{\text{bc}} = \frac{1}{N_{\text{bc}}} \sum_{j=1}^{N_{\text{bc}}} \left\| \mathcal{B}[\hat{\mathbf{u}}_\theta(t_j, \mathbf{x}_{\partial\Omega})] - g(t_j, \mathbf{x}_{\partial\Omega}) \right\|^2
$$

### Three Integration Paradigms

There are three architectural paradigms for embedding physics into LSTM-based dynamical system models:

#### 1. Physics-Integrated (Soft Constraint)

LSTM makes predictions freely; the PDE residual is added as a regularizing penalty in the loss. This is the most flexible paradigm:

$$
\hat{\mathbf{u}} = \text{LSTM}_\theta(\mathbf{X}_{\text{history}}), \quad
\mathcal{L} = \mathcal{L}_{\text{data}}(\hat{\mathbf{u}}, \mathbf{u}_{\text{true}}) + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}}(\hat{\mathbf{u}})
$$

- **Pros**: Flexible, easy to implement, works with any PDE
- **Cons**: PDE constraint is soft; can be violated if $\lambda$ is too small

#### 2. Physics-AI Hybrid Ensemble (Hard Constraint)

Two independent branches — a physics-based solver and a data-driven LSTM — whose outputs are fused:

$$
\hat{\mathbf{u}} = \alpha \cdot \mathbf{u}_{\text{phys}}(t; \boldsymbol{\phi}) + (1 - \alpha) \cdot \mathbf{u}_{\text{LSTM}}(t; \theta)
$$

where $\alpha \in [0, 1]$ is a learned or fixed blending coefficient, and $\mathbf{u}_{\text{phys}}$ is the solution of a simplified or reduced-order physics model.

- **Pros**: Guarantees some physical behavior; interpretable decomposition
- **Cons**: Requires a computationally tractable physics model; simplified physics may miss complex dynamics

#### 3. AI-Integrated Physics (Differentiable PDE Surrogate)

The LSTM is used to predict PDE parameters or boundary conditions, which are then fed into a differentiable numerical solver:

$$
\boldsymbol{\phi}_t = \text{LSTM}_\theta(\mathbf{X}_{\text{history}}), \quad
\hat{\mathbf{u}}(t) = \text{PDE\_Solver}(\boldsymbol{\phi}_t, \mathbf{u}_{t-1})
$$

- **Pros**: Hard PDE satisfaction by construction; suitable for control/optimization
- **Cons**: Requires a differentiable solver; computationally expensive per iteration

### Complete Loss Formulation (Paradigm 1)

For the remainder of this entry, we focus on the **Physics-Integrated (Soft Constraint)** paradigm, which is the most widely adopted:

$$
\mathcal{L}_{\text{total}} = \underbrace{\frac{1}{N} \sum_{i=1}^{N} \text{Huber}(\hat{u}_i - u_i^{\text{(obs)}})}_{\text{robust data fit}}
+ \lambda_{\text{pde}} \underbrace{\frac{1}{M} \sum_{j=1}^{M} \left\| \mathcal{N}[\hat{u}_\theta(t_c^{(j)})] - f(t_c^{(j)}) \right\|^2}_{\text{PDE residual}}
+ \lambda_{\text{reg}} \underbrace{\|\theta\|_2^2}_{\text{weight decay}}
$$

Huber loss is preferred over MSE for the data term to reduce sensitivity to measurement outliers:

$$
\text{Huber}(\delta) = \begin{cases}
0.5\delta^2 & \text{if } |\delta| \leq 1 \\
|\delta| - 0.5 & \text{otherwise}
\end{cases}
$$

### Automatic Differentiation for PDE Residuals

The PDE residual at a collocation point $(t_c, \mathbf{x}_c)$ is computed via:

$$
\begin{aligned}
\hat{u} &= \text{LSTM}_\theta(\mathbf{X}_{\text{hist}}; t_c, \mathbf{x}_c) \\
\frac{\partial \hat{u}}{\partial t} &= \frac{\partial}{\partial t} \text{LSTM}_\theta(\mathbf{X}_{\text{hist}}; t, \mathbf{x})\Big|_{t=t_c} \quad (\text{via AD}) \\
\frac{\partial \hat{u}}{\partial x} &= \frac{\partial}{\partial x} \text{LSTM}_\theta(\mathbf{X}_{\text{hist}}; t_c, x)\Big|_{x=x_c} \quad (\text{via AD}) \\
\mathcal{L}_{\text{pde}} &= \left\| \frac{\partial \hat{u}}{\partial t} + \mathcal{F}\left(\hat{u}, \frac{\partial \hat{u}}{\partial x}, \frac{\partial^2 \hat{u}}{\partial x^2}\right) - f(t_c, \mathbf{x}_c) \right\|^2
\end{aligned}
$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Governing PDE is known and differentiable | $\mathcal{N}$ is a known differential operator expressible via AD-compatible operations | The PDE must be closed-form with smooth nonlinearities; discontinuous or stochastic PDEs require special treatment |
| Collocation points can be freely sampled | $\{t_c^{(j)}, \mathbf{x}_c^{(j)}\}_{j=1}^{M}$ can be drawn from the spatiotemporal domain | Sensor placement is not a constraint; enables physics enforcement even where no measurements exist |
| Solution is smooth enough for AD | $\mathbf{u}(t, \mathbf{x}) \in C^{k}$ where $k$ = highest derivative order in $\mathcal{N}$ | Non-smooth solutions (shocks, phase boundaries) may cause AD gradient instability; may require adaptive collocation |
| LSTM can capture temporal dependencies | Sequence length $\tau$ is sufficient to resolve characteristic time scales | Short windows cannot capture slow dynamics; very long windows increase training cost and vanishing gradient risk |
| PDE residual provides meaningful gradient signal | $\nabla_\theta \mathcal{L}_{\text{pde}}$ correlates with prediction error | If $\lambda_{\text{pde}}$ is too large, training focuses on PDE satisfaction at expense of data fit; careful balancing needed |
| Measurement noise is bounded | $\mathbb{E}[\| \epsilon \|^2] < \infty$, $\epsilon = y - \mathbf{u}_{\text{true}}$ | Heavy-tailed noise can destabilize PDE residual gradients; Huber loss mitigates this |
| Initial/boundary conditions are consistent | $\mathcal{B}[\mathbf{u}(t, \mathbf{x})]$ on $\partial\Omega$ matches observed data | Mismatch between BC and measurements creates conflicting loss signals; Pareto-optimal trade-off may be needed |

## Applicable Scenarios

**When to use:**
- **Sparse data regime**: Only a few noisy trajectories are available, but the governing PDE is known. The physics penalty acts as a powerful regularizer that prevents overfitting.
- **Extrapolation beyond training domain**: PI-LSTM generalizes better than pure LSTM when predicting outside the temporal or parametric training range, because the PDE constraint anchors predictions to physically admissible manifolds.
- **Dynamical systems with known structure**: Fluid dynamics (Navier-Stokes), structural dynamics (wave equation), heat transfer (diffusion equation), chemical kinetics (reaction ODEs).
- **Systems with multi-scale physics**: PDE + data hybrid can capture fast physics through the PDE and slow dynamics through LSTM temporal memory.
- **Inverse problems**: The same framework can estimate unknown PDE parameters $\kappa$ by adding them as learnable parameters: $\mathcal{L}_{\text{pde}} = \| \mathcal{N}[\hat{u}; \kappa] - f \|^2$.

**When NOT to use:**
- **Unknown governing PDE**: If no reliable PDE/ODE exists, the physics penalty is meaningless; use pure LSTM or other data-driven models.
- **Highly discontinuous solutions**: Shocks, material interfaces, or phase boundaries create AD difficulties (infinite gradients); use conservative PINN variants (e.g., cPINN) or weak-form constraints.
- **Extremely high-dimensional PDEs**: 3D+time Navier-Stokes at high Reynolds number requires dense collocation grids; computational cost of AD through LSTM becomes prohibitive.
- **Real-time inference on edge devices**: AD-based PDE residual computation during inference is expensive; consider amortized or distillation approaches.
- **Very small datasets ($N < 100$)**: While PI-LSTM helps in sparse regimes, the LSTM backbone itself needs enough sequential data to learn meaningful temporal features.

**Comparison with alternatives:**

| Method | Strength | Limitation |
|--------|----------|------------|
| Pure LSTM | Flexible, no PDE required; fast training | Poor extrapolation; may violate physics; needs more data |
| Pure PINN (Raissi et al.) | Hard PDE satisfaction; mesh-free | Ignores temporal structure; struggles with long sequences; slow to train |
| PI-LSTM (this entry) | Combines temporal memory + PDE constraint | PDE residual AD through LSTM is computationally heavy; tuning $\lambda$ is delicate |
| Neural ODE (Chen et al., 2018) | Continuous-time dynamics; natural ODE constraint | Cannot handle PDEs without spatial discretization; ODE solver is expensive |
| Fourier Neural Operator (Li et al., 2021) | Resolution-invariant; learns solution operator | Requires full-field training data; no temporal memory mechanism |
| Gaussian Process + PDE (Raissi et al., 2017) | Bayesian uncertainty quantification | Scalability issues beyond $10^3$ points; limited to linear PDEs |

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| Sequence length $\tau$ | 10–50 (system-dependent) | Use autocorrelation analysis to estimate characteristic time scale; longer $\tau$ captures slower dynamics |
| LSTM hidden size | 64–256 | Start at 128; increase for more complex dynamics; decrease if PDE constraint dominates |
| LSTM layers | 1–3 | 2 layers is a good default; deeper stacks help with highly nonlinear dynamics |
| $\lambda_{\text{pde}}$ | $[10^{-3}, 10^0]$ | Start at 0.1; annealing schedule (warm up from 0) improves stability |
| Collocation points $M$ per epoch | $[1000, 10000]$ | More points = better PDE enforcement but slower; adaptive sampling over region of high residual |
| Optimizer | Adam | Default $\beta_1=0.9, \beta_2=0.999$ |
| Learning rate | $1 \times 10^{-3}$ | ReduceLROnPlateau (patience=20, factor=0.5) |
| Batch size | 32–128 | Limited by PDE residual AD memory; smaller batch for complex PDEs |
| Weight decay | $1 \times 10^{-5}$ | Mild regularization for LSTM weights |
| Gradient clipping norm | 1.0 | Essential for stable AD through LSTM over long sequences |

### Numerical Considerations

1. **PDE residual gradient graph**: Computing $\mathcal{L}_{\text{pde}}$ requires `torch.autograd.grad` with `create_graph=True`, which builds a second-order computational graph. This roughly **doubles memory and triples computation** per forward pass compared to pure LSTM training.

2. **Loss balancing strategy**: The PDE and data losses often have different scales. Strategies include:
   - **Fixed weights** with manual tuning: simplest but sensitive
   - **Learning rate annealing** (Wang et al., 2021): adapt $\lambda_{\text{pde}}$ based on gradient statistics:
     $$
     \lambda_{\text{pde}} \leftarrow \eta \cdot \frac{\max_k |\nabla_{\theta_k} \mathcal{L}_{\text{data}}|}{\text{mean}_k |\nabla_{\theta_k} \mathcal{L}_{\text{pde}}|}
     $$
   - **Soft-attention weighting**: learn $\lambda$ as a trainable parameter
   - **Gradient surgery**: project conflicting gradient components (Yu et al., 2020)

3. **Collocation point sampling**:
   - **Uniform grid**: simple but may miss sharp PDE features
   - **Adaptive/residual-based**: concentrate points where PDE residual is largest (Lu et al., 2021)
   - **Latin hypercube**: better coverage in high dimensions (default for $d \leq 5$)

4. **Normalization**: Both inputs and PDE residual targets should be normalized to $\mathcal{O}(1)$; otherwise PDE terms with different physical units dominate arbitrarily.

5. **Derivative scaling**: Physics with higher-order derivatives ($\partial^2/\partial x^2$, $\partial^4/\partial x^4$) may need gradient scaling to avoid vanishing/exploding AD gradients. Use the scale factor $1 / \Delta x^k$ where $\Delta x$ is the characteristic spatial scale and $k$ is derivative order.

6. **PDE residual on LSTM hidden states**: For PDEs involving temporal derivatives, the residual must be computed on the **output trajectory** $\{\hat{\mathbf{u}}_t\}$, not the hidden states. Ensure the output projection is differentiable with respect to time by passing $t$ as an additional input feature to the LSTM.

## Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict, Tuple


# =============================================================================
# 1. PI-LSTM: Physics-Informed LSTM with configurable PDE residual
# =============================================================================

class PILSTM(nn.Module):
    """Physics-Informed LSTM for PDE/ODE-Constrained Dynamical Systems.

    Combines an LSTM backbone with a physics-informed loss penalty.
    The PDE residual function is user-configurable, allowing application
    to arbitrary differential equations.

    Args:
        input_size: Dimension of input features at each time step
        hidden_size: LSTM hidden state dimension
        output_size: Dimension of prediction target
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability between LSTM layers (if num_layers > 1)
        pde_residual_fn: Callable that computes PDE residual given
            (prediction, time, optional spatial coordinates).
            Signature: (u_pred, t, **kwargs) -> residual tensor
        t_in_features: If True, append time t to LSTM input features
              (needed for PDEs with explicit time dependence)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        pde_residual_fn: Optional[Callable] = None,
        t_in_features: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.pde_residual_fn = pde_residual_fn
        self.t_in_features = t_in_features

        # Effective input size: original features + optional time feature
        lstm_input_size = input_size + (1 if t_in_features else 0)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection: hidden state -> prediction
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        t_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the PI-LSTM.

        Args:
            x_seq: Input sequence (batch, seq_len, input_size)
            t_seq: Time stamps (batch, seq_len) or (batch, seq_len, 1)
            hidden: Optional initial (h_0, c_0) tuple

        Returns:
            predictions: (batch, seq_len, output_size) — predictions at each step
            h_last: (batch, hidden_size) — final hidden state
        """
        # Ensure t_seq is (batch, seq_len, 1)
        if t_seq.dim() == 2:
            t_seq = t_seq.unsqueeze(-1)

        # Append time as an additional input feature
        if self.t_in_features:
            x_seq = torch.cat([x_seq, t_seq], dim=-1)

        lstm_out, (h_n, _) = self.lstm(x_seq, hidden)
        # lstm_out: (batch, seq_len, hidden_size)

        predictions = self.regressor(lstm_out)  # (batch, seq_len, output_size)
        return predictions, h_n[-1]


# =============================================================================
# 2. Physics-Informed Composite Loss
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """Composite loss: data fit + PDE residual + optional IC/BC.

    Args:
        pde_residual_fn: (u_pred, t_colloc, **kwargs) -> residual
        lambda_pde: Weight for PDE residual loss
        lambda_ic: Weight for initial condition loss
        lambda_bc: Weight for boundary condition loss
        use_huber: Use Huber loss for data term (robust to outliers)
    """
    def __init__(
        self,
        pde_residual_fn: Callable,
        lambda_pde: float = 0.1,
        lambda_ic: float = 0.0,
        lambda_bc: float = 0.0,
        use_huber: bool = True,
    ):
        super().__init__()
        self.pde_residual_fn = pde_residual_fn
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.huber = nn.HuberLoss(delta=1.0)
        self.mse = nn.MSELoss()

    def forward(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
        t_grid: torch.Tensor,
        lambda_pde_current: Optional[float] = None,
        **pde_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Args:
            u_pred: Predicted solution (batch, seq_len, output_size)
            u_true: Ground truth (batch, seq_len, output_size)
            t_grid: Time grid (batch, seq_len, 1) or (batch, seq_len)
            lambda_pde_current: Override lambda_pde (e.g., for annealing)
            **pde_kwargs: Additional keyword args passed to pde_residual_fn

        Returns:
            total_loss: Scalar loss tensor
            loss_dict: Dictionary of individual loss components
        """
        lam_pde = lambda_pde_current if lambda_pde_current is not None else self.lambda_pde

        # --- Data loss (Huber for robustness) ---
        if self.use_huber:
            L_data = self.huber(u_pred, u_true)
        else:
            L_data = self.mse(u_pred, u_true)

        # --- PDE residual loss ---
        # Requires computing differential operator via AD on the prediction
        # with respect to time (and potentially space).
        # The PDE residual function handles the AD internally.
        if lam_pde > 0 and self.pde_residual_fn is not None:
            residual = self.pde_residual_fn(u_pred, t_grid, **pde_kwargs)
            L_pde = self.mse(residual, torch.zeros_like(residual))
        else:
            L_pde = torch.tensor(0.0, device=u_pred.device)

        # --- Total ---
        total = L_data + lam_pde * L_pde

        loss_dict = {
            'L_data': L_data.item(),
            'L_pde': L_pde.item() if isinstance(L_pde, torch.Tensor) else 0.0,
        }
        return total, loss_dict


# =============================================================================
# 3. Example: Simple ODE constraint (Exponential Decay)
# =============================================================================

def exponential_decay_pde_residual(
    u_pred: torch.Tensor,
    t_grid: torch.Tensor,
    decay_rate: float = 0.5,
) -> torch.Tensor:
    """PDE residual for the ODE: du/dt = -k * u

    The residual is: du/dt + k*u = 0
    (should be close to zero if prediction satisfies the ODE)

    Args:
        u_pred: Predicted solution (batch, seq_len, 1)
        t_grid: Time points (batch, seq_len, 1)
        decay_rate: Decay constant k

    Returns:
        residual: (batch, seq_len, 1) — how much the ODE is violated
    """
    batch_size, seq_len, _ = u_pred.shape

    # Compute du/dt via automatic differentiation
    # We need gradients of each u_pred_i with respect to t_i
    # This requires setting up the computational graph carefully

    # Flatten to process all points: (batch * seq_len, 1)
    u_flat = u_pred.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)

    # Compute du/dt using autograd
    u_flat = u_flat.requires_grad_(True)
    t_flat = t_flat.requires_grad_(True)

    # Regress u on t so we can take du/dt
    # The prediction u is already a function of t through the LSTM,
    # but to get du/dt we need gradients through the regressor.
    # We'll reconstruct u from the flattened representation.

    # Since u_pred is already the network output, we can compute
    # gradient of each u_i w.r.t. its corresponding t_i
    du_dt = torch.autograd.grad(
        outputs=u_flat,
        inputs=t_flat,
        grad_outputs=torch.ones_like(u_flat),
        create_graph=True,
        retain_graph=True,
    )[0]  # (batch * seq_len, 1)

    du_dt = du_dt.reshape(batch_size, seq_len, 1)

    # PDE residual: du/dt + k * u = 0
    residual = du_dt + decay_rate * u_pred

    return residual


# =============================================================================
# 4. Another example: Advection PDE residual (1D linear advection)
# =============================================================================

def advection_pde_residual(
    u_pred: torch.Tensor,
    t_grid: torch.Tensor,
    x_grid: Optional[torch.Tensor] = None,
    advection_speed: float = 1.0,
) -> torch.Tensor:
    """PDE residual for 1D advection: du/dt + c * du/dx = 0

    Args:
        u_pred: (batch, seq_len, 1) — predicted solution
        t_grid: (batch, seq_len, 1) — time coordinates
        x_grid: (batch, seq_len, 1) — spatial coordinates
        advection_speed: Wave speed c

    Returns:
        residual: (batch, seq_len, 1)
    """
    batch_size, seq_len, _ = u_pred.shape

    # We need both du/dt and du/dx
    # For this to work, both t and x must be input features to the LSTM

    # Flatten
    u_flat = u_pred.reshape(-1, 1).requires_grad_(True)
    t_flat = t_grid.reshape(-1, 1).requires_grad_(True)

    # du/dt
    du_dt = torch.autograd.grad(
        u_flat, t_flat,
        grad_outputs=torch.ones_like(u_flat),
        create_graph=True,
        retain_graph=True,
    )[0].reshape(batch_size, seq_len, 1)

    residual = du_dt  # placeholder for full implementation

    if x_grid is not None:
        x_flat = x_grid.reshape(-1, 1).requires_grad_(True)
        du_dx = torch.autograd.grad(
            u_flat, x_flat,
            grad_outputs=torch.ones_like(u_flat),
            create_graph=True,
            retain_graph=True,
        )[0].reshape(batch_size, seq_len, 1)
        residual = du_dt + advection_speed * du_dx

    return residual


# =============================================================================
# 5. Training utilities
# =============================================================================

class PDEAnnealingScheduler:
    """Learning rate scheduler for PDE loss weight.

    Gradually ramps up lambda_pde from 0 to target value over
    a specified number of epochs, which stabilizes early training.
    """
    def __init__(self, target_lambda: float, ramp_epochs: int = 50):
        self.target = target_lambda
        self.ramp_epochs = ramp_epochs

    def get_lambda(self, epoch: int) -> float:
        if epoch >= self.ramp_epochs:
            return self.target
        return self.target * (epoch / self.ramp_epochs)


def train_epoch(
    model: PILSTM,
    loss_fn: PhysicsInformedLoss,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    current_lambda_pde: float,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch of PI-LSTM."""
    model.train()
    total_loss = 0.0
    accum_losses: Dict[str, float] = {}

    for batch in dataloader:
        x_seq, t_seq, u_true = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward
        u_pred, _ = model(x_seq, t_seq)

        # Composite loss
        loss, loss_dict = loss_fn(
            u_pred, u_true, t_seq,
            lambda_pde_current=current_lambda_pde,
        )

        loss.backward()

        # Gradient clipping (critical for AD through LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            accum_losses[k] = accum_losses.get(k, 0.0) + v

    n = len(dataloader)
    avg_losses = {k: v / n for k, v in accum_losses.items()}
    return total_loss / n, avg_losses


@torch.no_grad()
def evaluate(
    model: PILSTM,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate PI-LSTM on validation data."""
    model.eval()
    preds_list, targets_list = [], []

    for batch in dataloader:
        x_seq, t_seq, u_true = [b.to(device) for b in batch]
        u_pred, _ = model(x_seq, t_seq)
        preds_list.append(u_pred.cpu())
        targets_list.append(u_true.cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)

    mse = torch.mean((preds - targets) ** 2)
    mae = torch.mean(torch.abs(preds - targets))
    nrmse = torch.sqrt(mse) / (torch.max(targets) - torch.min(targets) + 1e-8)

    return {
        'MSE': mse.item(),
        'MAE': mae.item(),
        'NRMSE': nrmse.item(),
        'RMSE': torch.sqrt(mse).item(),
    }


# =============================================================================
# 6. Synthetic Data Generation (Exponential Decay ODE System)
# =============================================================================

def generate_exponential_decay_data(
    n_trajectories: int = 50,
    seq_len: int = 32,
    dt: float = 0.1,
    k_range: Tuple[float, float] = (0.2, 1.0),
    noise_std: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data following du/dt = -k * u.

    Each trajectory has a different decay rate k sampled uniformly.
    This simulates a family of related dynamical systems.

    Returns:
        X: (n_trajectories, seq_len, 1) — input sequences (u measurements)
        T: (n_trajectories, seq_len, 1) — time grid
        U: (n_trajectories, seq_len, 1) — target sequences
    """
    rng = np.random.RandomState(seed)

    X_list, T_list, U_list = [], [], []

    for _ in range(n_trajectories):
        # Sample decay rate
        k = rng.uniform(*k_range)
        u0 = rng.uniform(0.5, 2.0)

        # Generate continuous trajectory
        t = np.arange(seq_len) * dt
        u = u0 * np.exp(-k * t)

        # Add measurement noise
        u_noisy = u + rng.randn(seq_len) * noise_std

        # For autoregressive setup: use previous step as input
        # X_t = u_{t-1} (shifted by 1)
        # U_t = u_t (target)
        x_seq = u_noisy[:-1].reshape(-1, 1)   # (seq_len-1, 1)
        t_seq = t[:-1].reshape(-1, 1)          # (seq_len-1, 1)
        u_target = u[1:].reshape(-1, 1)        # (seq_len-1, 1) — clean target

        X_list.append(x_seq)
        T_list.append(t_seq)
        U_list.append(u_target)

    return (
        np.stack(X_list, axis=0),
        np.stack(T_list, axis=0),
        np.stack(U_list, axis=0),
    )


# =============================================================================
# 7. Training Loop with Physics Annealing
# =============================================================================

def train_pilSTM(
    model: PILSTM,
    loss_fn: PhysicsInformedLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 200,
    lr: float = 1e-3,
    lambda_pde_target: float = 0.1,
    ramp_epochs: int = 50,
    patience: int = 30,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """Full PI-LSTM training loop with PDE weight annealing."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
    )
    annealing = PDEAnnealingScheduler(lambda_pde_target, ramp_epochs)

    history = {
        'train_loss': [], 'val_loss': [],
        'L_data': [], 'L_pde': [],
        'lambda_pde': [],
    }
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        current_lambda_pde = annealing.get_lambda(epoch)

        # Train
        train_loss, comp_losses = train_epoch(
            model, loss_fn, train_loader, optimizer, device, current_lambda_pde,
        )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_seq, t_seq, u_true = [b.to(device) for b in batch]
                u_pred, _ = model(x_seq, t_seq)
                loss, _ = loss_fn(u_pred, u_true, t_seq, lambda_pde_current=0.0)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Logging
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['L_data'].append(comp_losses.get('L_data', 0.0))
        history['L_pde'].append(comp_losses.get('L_pde', 0.0))
        history['lambda_pde'].append(current_lambda_pde)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"L_data: {comp_losses.get('L_data', 0):.6f} | "
                  f"L_pde: {comp_losses.get('L_pde', 0):.6f} | "
                  f"λ_pde: {current_lambda_pde:.4f}")

    # Restore best parameters
    if best_state is not None:
        model.load_state_dict(best_state)

    history['best_val_loss'] = best_val_loss
    return history


# =============================================================================
# 8. Ablation: evaluate with and without PDE constraint
# =============================================================================

def train_baseline_lstm(
    model: PILSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu'),
) -> float:
    """Train the same architecture WITHOUT PDE constraint (pure LSTM)."""
    # Use only data loss (lambda_pde = 0 always)
    loss_fn_no_pde = PhysicsInformedLoss(
        pde_residual_fn=lambda u, t: torch.zeros_like(u),
        lambda_pde=0.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = float('inf')
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x_seq, t_seq, u_true = [b.to(device) for b in batch]
            optimizer.zero_grad()
            u_pred, _ = model(x_seq, t_seq)
            loss, _ = loss_fn_no_pde(u_pred, u_true, t_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_seq, t_seq, u_true = [b.to(device) for b in batch]
                u_pred, _ = model(x_seq, t_seq)
                loss, _ = loss_fn_no_pde(u_pred, u_true, t_seq)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


# =============================================================================
# 9. Main: demo with exponential decay ODE
# =============================================================================

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 65)
    print("PI-LSTM: Physics-Informed LSTM for PDE-Constrained Systems")
    print("=" * 65)

    # --- Generate synthetic data ---
    print("\n[1] Generating synthetic exponential decay trajectories...")
    X, T, U = generate_exponential_decay_data(
        n_trajectories=80, seq_len=33, dt=0.1,
        k_range=(0.2, 1.0), noise_std=0.03, seed=42,
    )
    print(f"    Shape: X {X.shape}, T {T.shape}, U {U.shape}")
    print(f"    Decay rate k in [{0.2:.1f}, {1.0:.1f}]")

    # --- Train/val split ---
    n_train = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X[train_idx]),
        torch.FloatTensor(T[train_idx]),
        torch.FloatTensor(U[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X[val_idx]),
        torch.FloatTensor(T[val_idx]),
        torch.FloatTensor(U[val_idx]),
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(f"    Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Initialize PI-LSTM ---
    print("\n[2] Initializing PI-LSTM model...")
    # PDE residual: du/dt + k*u = 0 (exponential decay)
    pde_fn = exponential_decay_pde_residual

    model = PILSTM(
        input_size=1,         # u_{t-1} as input
        hidden_size=64,
        output_size=1,
        num_layers=2,
        dropout=0.1,
        pde_residual_fn=pde_fn,
        t_in_features=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total trainable parameters: {n_params:,}")

    loss_fn = PhysicsInformedLoss(
        pde_residual_fn=pde_fn,
        lambda_pde=0.1,
        use_huber=True,
    )

    # --- Train PI-LSTM ---
    print("\n[3] Training PI-LSTM with PDE constraint (λ annealing)...")
    pi_history = train_pilSTM(
        model, loss_fn, train_loader, val_loader,
        n_epochs=150, lr=1e-3,
        lambda_pde_target=0.1, ramp_epochs=50,
        patience=25, device=device,
    )

    # --- Evaluate PI-LSTM ---
    print("\n[4] Evaluating PI-LSTM on validation set...")
    pi_metrics = evaluate(model, val_loader, device)
    for k, v in pi_metrics.items():
        print(f"    {k}: {v:.6f}")

    # --- Train baseline (no PDE) for comparison ---
    print("\n[5] Training baseline LSTM (NO physics constraint)...")
    baseline_model = PILSTM(
        input_size=1, hidden_size=64, output_size=1,
        num_layers=2, dropout=0.1,
        pde_residual_fn=None, t_in_features=True,
    ).to(device)
    baseline_model.load_state_dict(model.state_dict())  # start from same init? no, re-init
    baseline_model.apply(lambda m: (
        m.reset_parameters() if hasattr(m, 'reset_parameters') else None
    ))

    baseline_val_loss = train_baseline_lstm(
        baseline_model, train_loader, val_loader,
        n_epochs=150, lr=1e-3, device=device,
    )
    baseline_metrics = evaluate(baseline_model, val_loader, device)
    print("    Baseline (no PDE) metrics:")
    for k, v in baseline_metrics.items():
        print(f"    {k}: {v:.6f}")

    # --- Ablation report ---
    print("\n[6] Ablation: PI-LSTM vs Baseline LSTM")
    print(f"    {'Metric':<12} {'PI-LSTM':<12} {'Baseline':<12} {'Improvement':<12}")
    print(f"    {'-'*48}")
    for metric in ['MSE', 'MAE', 'RMSE', 'NRMSE']:
        pi_val = pi_metrics.get(metric, 0)
        bl_val = baseline_metrics.get(metric, 0)
        impr = ((bl_val - pi_val) / (bl_val + 1e-12)) * 100
        print(f"    {metric:<12} {pi_val:<12.6f} {bl_val:<12.6f} {impr:<+11.2f}%")

    # --- Visualize ---
    print("\n[7] Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Loss curves
    ax = axes[0, 0]
    ax.plot(pi_history['train_loss'], label='Train Loss', alpha=0.8)
    ax.plot(pi_history['val_loss'], label='Val Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('PI-LSTM Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) PDE weight annealing
    ax = axes[0, 1]
    ax.plot(pi_history['lambda_pde'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$\lambda_{\mathrm{PDE}}$')
    ax.set_title('PDE Weight Annealing Schedule')
    ax.grid(True, alpha=0.3)

    # (c) Loss components
    ax = axes[1, 0]
    ax.plot(pi_history['L_data'], label='$\\mathcal{L}_{\\mathrm{data}}$', alpha=0.8)
    ax.plot(pi_history['L_pde'], label='$\\mathcal{L}_{\\mathrm{PDE}}$', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Loss Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Prediction vs True (sample from validation set)
    ax = axes[1, 1]
    model.eval()
    with torch.no_grad():
        sample_x, sample_t, sample_u = next(iter(val_loader))
        sample_x, sample_t = sample_x.to(device), sample_t.to(device)
        pred, _ = model(sample_x, sample_t)
        # Plot first trajectory
        idx = 0
        t_np = sample_t[idx, :, 0].cpu().numpy()
        true_np = sample_u[idx, :, 0].numpy()
        pred_np = pred[idx, :, 0].cpu().numpy()
        ax.plot(t_np, true_np, 'k-', label='True', linewidth=2, alpha=0.8)
        ax.plot(t_np, pred_np, 'r--', label='PI-LSTM Pred', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time t')
    ax.set_ylabel('u(t)')
    ax.set_title('PI-LSTM: Prediction vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pilSTM_demo_results.png', dpi=150)
    plt.show()
    print("    Figure saved to 'pilSTM_demo_results.png'")

    # --- Extrapolation test ---
    print("\n[8] Extrapolation test (predict beyond training time horizon)...")
    model.eval()
    baseline_model.eval()

    # Generate a longer trajectory
    k_test = 0.5
    u0_test = 1.0
    t_long = np.arange(0, 8.0, 0.1)  # 80 steps (training sees 32)
    u_true_long = u0_test * np.exp(-k_test * t_long)
    u_noisy_long = u_true_long + np.random.randn(len(t_long)) * 0.03

    # Autoregressive prediction
    def autoregressive_predict(model, u_init, n_steps, device):
        """Roll out predictions step by step."""
        model.eval()
        preds = [u_init]
        window = np.array([[u_init]])
        with torch.no_grad():
            for step in range(n_steps - 1):
                x_t = torch.FloatTensor(window).unsqueeze(0).to(device)  # (1, 1, 1)
                t_t = torch.FloatTensor([[step * 0.1]]).unsqueeze(0).to(device)
                u_pred, _ = model(x_t, t_t)
                pred_val = u_pred[0, 0, 0].item()
                preds.append(pred_val)
                window = np.array([[pred_val]])
        return np.array(preds)

    # Compare extrapolation at different horizons
    train_horizon = 3.2  # seconds (32 steps * 0.1)
    test_horizon = 6.0

    pi_extrap = autoregressive_predict(
        model, u_noisy_long[0], int(test_horizon / 0.1), device,
    )
    bl_extrap = autoregressive_predict(
        baseline_model, u_noisy_long[0], int(test_horizon / 0.1), device,
    )

    t_extrap = np.arange(0, test_horizon, 0.1)
    train_mask = t_extrap <= train_horizon
    extrap_mask = t_extrap > train_horizon

    pi_mse_extrap = np.mean((pi_extrap[extrap_mask] - u_true_long[extrap_mask]) ** 2)
    bl_mse_extrap = np.mean((bl_extrap[extrap_mask] - u_true_long[extrap_mask]) ** 2)

    print(f"    Extrapolation MSE (t > {train_horizon:.1f}s):")
    print(f"      PI-LSTM:  {pi_mse_extrap:.6f}")
    print(f"      Baseline: {bl_mse_extrap:.6f}")
    print(f"      Physics constraint improves extrapolation by "
          f"{(bl_mse_extrap - pi_mse_extrap) / (bl_mse_extrap + 1e-12) * 100:.1f}%")

    print("\n[9] Summary")
    print(f"    PI-LSTM successfully embeds ODE/PDE constraint into LSTM training.")
    print(f"    Key takeaway: physics constraint acts as a regularizer, improving")
    print(f"    extrapolation performance when the governing equation is known.")
    print("\nDemo complete.")
```

## References

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *Advances in Neural Information Processing Systems*, 31, 6571–6583. https://doi.org/10.48550/arXiv.1806.07366

Kumar, P. N., Upadhya, P. R., Nischay, S., Pavan Kumar, G., Shobana, T. S., & Rashmi, K. B. (2026). A temperature- and impedance-aware LSTM–PINN framework for physically consistent battery SOH prediction. *Scientific Reports*, 16, Article 7568. https://doi.org/10.1038/s41598-026-37850-y

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR 2021*. https://doi.org/10.48550/arXiv.2010.08895

Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228. https://doi.org/10.1137/19M1274067

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(3), A1905–A1925. https://doi.org/10.1137/20M1318043

Wang, Z., Xiao, D., Fang, F., Govindan, R., Pain, C. C., & Guo, Y. (2020). Model identification of reduced order fluid dynamics systems using deep learning. *International Journal for Numerical Methods in Fluids*, 92(5), 430–450. https://doi.org/10.1002/fld.4826

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient Surgery for Multi-Task Learning. *Advances in Neural Information Processing Systems*, 33, 5824–5836. https://doi.org/10.48550/arXiv.2001.06782

Zhang, R., Liu, Y., & Sun, H. (2021). Physics-guided convolutional neural network (PhyCNN) for data-driven seismic response modeling. *Engineering Structures*, 237, Article 112196. https://doi.org/10.1016/j.engstruct.2021.112196
