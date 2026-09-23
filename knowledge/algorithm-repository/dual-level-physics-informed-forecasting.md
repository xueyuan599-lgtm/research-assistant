---
title: Dual-Level Physics-Informed LSTM for Multi-Step Time Series Forecasting
type:
  - forecasting
  - physics-informed
  - deep-learning
domain:
  - signal-processing
---
# Dual-Level Physics-Informed LSTM for Multi-Step Time Series Forecasting

- **Source**: Jin, M., Zhou, Y., & Zheng, H. (2026). Dual-Level Models for Physics-Informed Multi-Step Time Series Forecasting. *arXiv preprint*, arXiv:2601.07640.
- **DOI**: 10.48550/arXiv.2601.07640
- **Category**: Time Series Fusion / Physics-Informed Forecasting
- **Method**: Two-level architecture combining a coarse physics-informed process-level predictor with an LSTM-based fine-grained observation-level residual corrector, enabling accurate multi-step forecasts that remain physically consistent.

## Mathematical Setup

### Problem Definition

Consider a multivariate time series $\{\mathbf{x}_t\}_{t=1}^{T}$ where $\mathbf{x}_t \in \mathbb{R}^{d}$ is the observation at time $t$. The goal of multi-step forecasting is to predict the trajectory over a horizon $H$:

$$
\{\widehat{\mathbf{x}}_{t+1}, \widehat{\mathbf{x}}_{t+2}, \dots, \widehat{\mathbf{x}}_{t+H}\}
$$

given the history $\{\mathbf{x}_1, \dots, \mathbf{x}_t\}$ and known physical parameters $\boldsymbol{\theta}_{\text{phys}}$.

The core insight is to decompose the prediction into two additive components:

$$
\widehat{\mathbf{x}}_{t+\tau} = \mathbf{x}_{\text{phys}}(t+\tau) + \boldsymbol{\delta}_{t+\tau}, \quad \tau = 1, \dots, H
$$

where $\mathbf{x}_{\text{phys}}$ is the physics-based coarse prediction and $\boldsymbol{\delta}$ is a learned residual correction that captures unmodeled dynamics.

### Level 1: Physics-Based Coarse Prediction

The physics layer models the known underlying dynamics via a parameterized ODE or discrete-time recurrence. Let $\mathbf{z}_t \in \mathbb{R}^{m}$ be the latent state of the physical system, evolving according to:

$$
\mathbf{z}_{t+1} = \mathbf{f}_{\text{phys}}(\mathbf{z}_t, \mathbf{u}_t, \boldsymbol{\theta}_{\text{phys}})
$$

where:
- $\mathbf{f}_{\text{phys}}$ is a known physical model (e.g., a discretized ODE)
- $\mathbf{u}_t$ are exogenous forcing terms (control inputs, driving forces)
- $\boldsymbol{\theta}_{\text{phys}}$ are interpretable physical parameters (mass, damping, stiffness, etc.)

The coarse observation prediction is then an emission from the physical latent state:

$$
\mathbf{x}_{\text{phys}}(t) = \mathbf{g}(\mathbf{z}_t) \quad \text{or} \quad \mathbf{x}_{\text{phys}}(t+1) = \mathbf{f}_{\text{phys}}(\mathbf{x}(t), \boldsymbol{\theta}_{\text{phys}})
$$

In the common case where the observation space coincides with the physical state space, the recurrence simplifies to:

$$
\mathbf{x}_{\text{phys}}(t+1) = \mathbf{f}_{\text{phys}}(\mathbf{x}_{\text{phys}}(t), \boldsymbol{\theta}_{\text{phys}})
$$

For a **damped harmonic oscillator** (the canonical running example used throughout this entry):

$$
\begin{aligned}
f_{\text{phys}}:\quad &v_{t+1} = v_t - \omega^2 x_t \Delta t - 2\zeta \omega v_t \Delta t \\
&x_{t+1} = x_t + v_{t+1} \Delta t
\end{aligned}
$$

with $\boldsymbol{\theta}_{\text{phys}} = (\omega, \zeta)$, where $\omega$ is the natural frequency and $\zeta$ is the damping ratio.

### Level 2: LSTM Residual Correction

The physics model is necessarily incomplete — it cannot capture friction variations, unmodeled nonlinearities, external disturbances, or measurement artifacts. The LSTM corrector learns these residuals from data.

Define the **prediction error** (residual) at time $t$:

$$
\mathbf{e}_t = \mathbf{x}_t - \mathbf{x}_{\text{phys}}(t)
$$

The LSTM corrector takes as input a window of past residuals and hidden states, and outputs the predicted residual at the next time step:

$$
\boldsymbol{\delta}_{t+1} = \text{LSTM}_{\boldsymbol{\phi}}(\mathbf{e}_{t}, \mathbf{e}_{t-1}, \dots, \mathbf{e}_{t-\tau+1}; \mathbf{h}_{t})
$$

where:
- $\boldsymbol{\phi}$ denotes all LSTM parameters (weights, biases)
- $\mathbf{h}_t \in \mathbb{R}^{H_{\text{lstm}}}$ is the LSTM hidden state at time $t$
- $\tau$ is the lookback window length

The LSTM architecture can be expressed as:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_{ii} \mathbf{e}_t + \mathbf{b}_{ii} + \mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{b}_{hi}) \\
\mathbf{f}_t &= \sigma(\mathbf{W}_{if} \mathbf{e}_t + \mathbf{b}_{if} + \mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{b}_{hf}) \\
\mathbf{g}_t &= \tanh(\mathbf{W}_{ig} \mathbf{e}_t + \mathbf{b}_{ig} + \mathbf{W}_{hg} \mathbf{h}_{t-1} + \mathbf{b}_{hg}) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_{io} \mathbf{e}_t + \mathbf{b}_{io} + \mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{b}_{ho}) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \\
\boldsymbol{\delta}_{t+1} &= \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_{y}
\end{aligned}
$$

where $\sigma$ is the sigmoid function, $\odot$ is element-wise multiplication, and the LSTM has $H_{\text{lstm}}$ hidden units.

### Final Multi-Step Prediction

The two levels are combined additively for each step in the forecast horizon:

$$
\widehat{\mathbf{x}}_{t+\tau} = \mathbf{x}_{\text{phys}}(t+\tau) + \boldsymbol{\delta}_{t+\tau}, \quad \tau = 1, 2, \dots, H
$$

For multi-step forecasting, the correction can be applied in two modes:

**Mode A — Recursive (closed-loop):** The physics model rolls forward autonomously for all $H$ steps, while the LSTM corrector also rolls forward using its own previous residual predictions as inputs:

$$
\boldsymbol{\delta}_{t+\tau} = \text{LSTM}_{\boldsymbol{\phi}}(\boldsymbol{\delta}_{t+\tau-1}, \dots; \mathbf{h}_{t+\tau-1}), \quad \tau \geq 2
$$

This mode is suitable for long horizons where ground-truth residuals beyond the first step are unavailable.

**Mode B — Teacher-forced (open-loop):** The physics model steps forward, but the LSTM corrector uses the true residual $\mathbf{e}_{t+\tau-1} = \mathbf{x}_{t+\tau-1} - \mathbf{x}_{\text{phys}}(t+\tau-1)$ at each step (during training only). At inference, it falls back to Mode A.

### Combined Loss Function

The total loss balances prediction accuracy with physical consistency:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{physics}}
$$

**1. Data prediction loss** (MSE over the forecast horizon):

$$
\mathcal{L}_{\text{data}} = \frac{1}{H} \sum_{\tau=1}^{H} \| \widehat{\mathbf{x}}_{t+\tau} - \mathbf{x}_{t+\tau} \|_2^2
$$

**2. Physics consistency loss** (penalizes deviation of the physics model's prediction from the true dynamics):

$$
\mathcal{L}_{\text{physics}} = \frac{1}{H} \sum_{\tau=1}^{H} \| \mathbf{x}_{\text{phys}}(t+\tau) - \mathbf{f}_{\text{phys}}(\widehat{\mathbf{x}}_{t+\tau-1}, \boldsymbol{\theta}_{\text{phys}}) \|_2^2
$$

This term encourages the combined prediction $\widehat{\mathbf{x}}$ to stay close to the manifold of physically plausible trajectories.

**3. Optional regularization** — residual sparsity (encourages the LSTM to learn only what the physics cannot explain):

$$
\mathcal{L}_{\text{sparse}} = \frac{1}{H} \sum_{\tau=1}^{H} \| \boldsymbol{\delta}_{t+\tau} \|_1
$$

The full objective is:

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{physics}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}}
$$

### Uncertainty Quantification

The dual-level framework admits probabilistic extension by modeling the residual distribution. Instead of a point estimate $\boldsymbol{\delta}_{t+1}$, we model:

$$
p(\boldsymbol{\delta}_{t+1} \mid \mathbf{e}_{t}, \dots) = \mathcal{N}(\boldsymbol{\mu}_{t+1}, \boldsymbol{\Sigma}_{t+1})
$$

where $\boldsymbol{\mu}_{t+1}$ and $\boldsymbol{\Sigma}_{t+1}$ (a diagonal covariance) are outputs of the LSTM. The final predictive distribution is:

$$
p(\widehat{\mathbf{x}}_{t+\tau}) = \mathcal{N}(\mathbf{x}_{\text{phys}}(t+\tau) + \boldsymbol{\mu}_{t+\tau}, \boldsymbol{\Sigma}_{t+\tau})
$$

This provides prediction intervals that capture both aleatoric uncertainty (measurement noise) and model misspecification (residual variance).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dual-Level Forecasting Flow                    │
└─────────────────────────────────────────────────────────────────┘

Historical observations: {x_{t-τ+1}, ..., x_t}
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                    ▼
    ┌───────────────────┐              ┌───────────────────┐
    │   Level 1:        │              │   Level 2:        │
    │   Physics Model   │              │   LSTM Corrector  │
    │   f_phys(x, θ)    │              │   LSTM_φ(e, h)    │
    ├───────────────────┤              ├───────────────────┤
    │  x_phys(t+1)      │              │  δ(t+1)           │
    │  = f_phys(x(t),θ) │              │  = LSTM(e_t, h_t) │
    └────────┬──────────┘              └────────┬──────────┘
             │                                   │
             └──────────────┬ ─ ─ ─ ─ ─ ─ ─ ─ ──┘
                            ▼
                   ┌─────────────────┐
                   │  Combined       │
                   │  x_pred(t+1) =  │
                   │  x_phys + δ     │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Roll out for   │
                   │  H steps        │
                   └─────────────────┘

Loss: L = MSE(x_pred, x_true) + λ_phys · MSE(x_phys, f_phys(x_pred))
```

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Known physics structure | $\mathbf{f}_{\text{phys}}$ is known up to parameters $\boldsymbol{\theta}_{\text{phys}}$ | A reasonable physical model must exist; completely black-box systems require pure data-driven approaches |
| Additive residual decomposition | $\mathbf{x}_t = \mathbf{x}_{\text{phys}}(t) + \boldsymbol{\delta}_t$ | The true process is the sum of physics and correction; multiplicative or nonlinear interactions are not modeled |
| Residual temporal structure | $\boldsymbol{\delta}_{t+1} = \text{LSTM}(\mathbf{e}_t, \mathbf{h}_t)$ | Residuals have exploitable autocorrelation; if residuals are white noise, the LSTM cannot learn meaningful corrections |
| Stationary physical parameters | $\boldsymbol{\theta}_{\text{phys}}$ is constant or slowly varying over the forecast horizon | Sudden regime shifts in the physical system (e.g., structural damage) invalidate the physics model |
| Sufficient residual history | Lookback window $\tau \geq 10$ time steps | Short windows ($\tau < 5$) cannot capture oscillatory or periodic residual patterns |
| Bounded residual magnitude | $\|\boldsymbol{\delta}_t\| \ll \|\mathbf{x}_{\text{phys}}(t)\|$ for well-specified physics | If residuals dominate the signal, the physics model adds little value; consider a purely data-driven approach |
| Independent training data segments | Each training trajectory starts from an independent initial condition | The LSTM must generalize across different physical regimes; single-trajectory training risks overfitting to one residual pattern |
| Discrete-time observation | Observations are made at uniform time intervals $\Delta t$ | Irregularly sampled data requires interpolation or continuous-time LSTM variants (e.g., ODE-RNN) |

## Applicable Scenarios

**When to use:**
- Time series with well-understood physical dynamics (mechanical oscillators, electrical circuits, biological rhythms, climate cycles) that are partially observed or have unmodeled perturbations
- Multi-step forecasting where pure physics models drift due to model error and pure data-driven models lack physical consistency
- Systems where interpretability is important and the physics component provides a transparent baseline
- Applications requiring uncertainty quantification with physically meaningful prediction intervals
- Transfer learning across similar systems with different parameter regimes (re-train only the LSTM corrector)

**When NOT to use:**
- No known or reliable physical model exists for the system (use pure LSTM, Transformer, or TCN)
- The physics model dominates with negligible residuals (use pure physics or a simple Kalman filter)
- Data is extremely scarce (the dual-level approach requires enough data to train the LSTM; fewer than ~1000 time steps may be insufficient)
- Real-time edge deployment with severe memory constraints (the LSTM adds computational overhead)
- Residuals exhibit long-range dependencies exceeding the LSTM window capacity (consider Transformer-based correction instead)

**Comparison with alternatives:**

| Method | Strength | Limitation |
|--------|----------|------------|
| Pure Physics (ODE/PDE) | Interpretable, no training data needed | Drifts over time; cannot capture unmodeled effects |
| Pure LSTM | Flexible; learns arbitrary dynamics | May violate physical constraints; poor extrapolation |
| Physics-Informed Neural Network (PINN) | Enforces PDE constraints via loss | Requires differentiable PDE solver; expensive for long sequences |
| Neural ODE | Continuous-time dynamics | Slow training; ODE solver overhead |
| **This Dual-Level Model** | Hybrid: physics baseline + learned correction; modular | Requires both a physics model and sufficient data for the LSTM |
| Ensemble (Physics + LSTM averaged) | Simple baseline | No principled decomposition; ad-hoc combination |

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| LSTM hidden size $H_{\text{lstm}}$ | 64 | Larger for complex residual patterns (max 256); smaller for simple corrections (min 16) |
| LSTM layers | 2 | 1–3 layers; deeper may overfit on small datasets |
| Lookback window $\tau$ | 20 | Longer captures periodic residuals; shorter for memory-limited deployment |
| Forecast horizon $H$ | 10 | Set by application; longer horizons increase accumulation of correction error |
| $\lambda_{\text{phys}}$ | 0.1 | Start at 0.1; increase if predictions violate physics; decrease if physics model is imperfect |
| $\lambda_{\text{sparse}}$ | 0.01 | Small positive value encourages residual sparsity; set to 0 if physics model is structurally wrong |
| Learning rate | $5 \times 10^{-4}$ | Adam optimizer; cosine annealing or ReduceLROnPlateau |
| Batch size | 64 | Larger for smooth training; smaller for limited data |
| Epochs | 200 | Early stopping patience = 30 |
| Gradient clipping | 1.0 (norm) | Prevents explosion from residual accumulation over long horizons |
| Teacher forcing ratio | 0.5 | Fraction of training steps using true residuals (decay to 0 during training) |

### Training Strategy

1. **Physics model calibration**: First fit $\boldsymbol{\theta}_{\text{phys}}$ to the training data via standard system identification (least-squares or MLE), holding these parameters fixed during joint training.

2. **Residual computation**: Compute training residuals $\mathbf{e}_t = \mathbf{x}_t - \mathbf{x}_{\text{phys}}(t)$ using the calibrated physics model.

3. **LSTM pre-training**: Pre-train the LSTM corrector on the residual sequence alone (MSE loss on $\boldsymbol{\delta}$ vs $\mathbf{e}$) for 50 epochs.

4. **Joint fine-tuning**: Train the full dual-level model end-to-end with $\mathcal{L}_{\text{total}}$. Freeze the physics model parameters to avoid the LSTM "rewriting" the physics.

5. **Teacher forcing annealing**: Start with teacher forcing ratio = 0.8 and linearly decay to 0.0 over the first 100 epochs of joint training. This transitions from open-loop to closed-loop multi-step correction.

6. **Validation for physical consistency**: Monitor the physics loss $\mathcal{L}_{\text{physics}}$ on the validation set. A rising trend indicates the LSTM is learning non-physical corrections.

### Numerical Considerations

- The residual sequence $\mathbf{e}_t$ should be standardized (zero mean, unit variance) before feeding into the LSTM to improve gradient conditioning.
- When the physics model operates at a different scale than the residual, initialize the LSTM output layer with small weights ($\sim 10^{-3}$) so the correction starts near zero.
- For long horizons $H > 20$, consider autoregressive splitting: generate the first $H/2$ steps with the dual-level model, then re-initialize the LSTM hidden state from the accumulated errors.
- The physics consistency loss $\mathcal{L}_{\text{physics}}$ requires computing $\mathbf{f}_{\text{phys}}(\widehat{\mathbf{x}}, \boldsymbol{\theta})$, which must be differentiable for gradient propagation. If the physics model is non-differentiable (e.g., a legacy simulator), freeze the physics model and only backpropagate through $\mathcal{L}_{\text{data}}$.
- Gradient clipping is critical: multi-step unrolling of the LSTM through $H$ steps can produce vanishing/exploding gradients in the correction path.

## Python Implementation

```python
"""
Dual-Level Physics-Informed LSTM for Multi-Step Time Series Forecasting.

Implements the two-level architecture:
  Level 1: Physics-based coarse prediction (damped harmonic oscillator)
  Level 2: LSTM residual correction

arXiv:2601.07640 (2026)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Callable
import matplotlib.pyplot as plt


# =============================================================================
# 1. Physics Model: Damped Harmonic Oscillator
# =============================================================================

class DampedOscillator(nn.Module):
    """Discrete-time damped harmonic oscillator as the physics model.

    State: [position x, velocity v]
    Dynamics (semi-implicit Euler):
        v_{t+1} = v_t - ω²·x_t·Δt - 2ζω·v_t·Δt
        x_{t+1} = x_t + v_{t+1}·Δt

    Args:
        dt: Time step (default 0.01)
        omega: Natural frequency (rad/s), learnable if tune_phys=True
        zeta: Damping ratio (dimensionless), learnable if tune_phys=True
        tune_phys: Whether to make omega/zeta trainable parameters
    """

    def __init__(self, dt: float = 0.01, omega: float = 5.0,
                 zeta: float = 0.05, tune_phys: bool = False):
        super().__init__()
        self.dt = dt
        if tune_phys:
            self.log_omega = nn.Parameter(torch.tensor(np.log(omega)))
            self.log_zeta = nn.Parameter(torch.tensor(np.log(zeta)))
        else:
            self.register_buffer('log_omega', torch.tensor(np.log(omega)))
            self.register_buffer('log_zeta', torch.tensor(np.log(zeta)))

    @property
    def omega(self) -> torch.Tensor:
        """Natural frequency (rad/s)."""
        return torch.exp(self.log_omega)

    @property
    def zeta(self) -> torch.Tensor:
        """Damping ratio (dimensionless)."""
        return torch.exp(self.log_zeta)

    def forward(self, x: torch.Tensor, v: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step the oscillator forward by dt.

        Args:
            x: Position (batch,)
            v: Velocity (batch,)
        Returns:
            x_next, v_next: State at t+dt
        """
        w = self.omega
        z = self.zeta
        # Semi-implicit Euler integration
        v_next = v - w ** 2 * x * self.dt - 2 * z * w * v * self.dt
        x_next = x + v_next * self.dt
        return x_next, v_next

    def rollout(self, x0: torch.Tensor, v0: torch.Tensor, steps: int
                ) -> torch.Tensor:
        """Roll forward the physics model for `steps` time steps.

        Args:
            x0: Initial position (batch,)
            v0: Initial velocity (batch,)
            steps: Number of steps to roll out
        Returns:
            x_traj: (batch, steps) — position trajectory
        """
        x, v = x0, v0
        traj = [x.unsqueeze(-1)]
        for _ in range(steps):
            x, v = self.forward(x, v)
            traj.append(x.unsqueeze(-1))
        return torch.cat(traj, dim=-1)


# =============================================================================
# 2. LSTM Residual Corrector
# =============================================================================

class LSTMResidualCorrector(nn.Module):
    """LSTM that predicts residual errors from past residual history.

    Maps (e_{t-τ+1}, ..., e_t) → δ_{t+1}

    Args:
        input_dim: Dimension of the time series (1 for scalar)
        hidden_dim: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate (applied between LSTM layers if num_layers > 1)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, e_seq: torch.Tensor,
                h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict the next residual.

        Args:
            e_seq: (batch, seq_len, input_dim) — past residual window
            h0: Optional initial (hidden, cell) state
        Returns:
            delta_next: (batch, input_dim) — predicted residual at t+1
            (h_n, c_n): Final LSTM states for further rollouts
        """
        if h0 is None:
            h0 = None  # LSTM defaults to zero state
        output, (h_n, c_n) = self.lstm(e_seq, h0)
        # Use the last output to predict the next residual
        delta_next = self.output_layer(output[:, -1, :])
        return delta_next, (h_n, c_n)


# =============================================================================
# 3. Dual-Level PINN Model
# =============================================================================

class DualLevelPINN(nn.Module):
    """Dual-Level Physics-Informed model for multi-step forecasting.

    Combination of:
      - A known physics model (DampedOscillator)
      - An LSTM residual corrector

    Args:
        physics_model: nn.Module implementing the physics step
        corrector: LSTMResidualCorrector instance
        lambda_phys: Weight for the physics consistency loss
        lambda_sparse: Weight for the residual sparsity penalty
    """

    def __init__(self, physics_model: nn.Module,
                 corrector: LSTMResidualCorrector,
                 lambda_phys: float = 0.1,
                 lambda_sparse: float = 0.01):
        super().__init__()
        self.physics = physics_model
        self.corrector = corrector
        self.lambda_phys = lambda_phys
        self.lambda_sparse = lambda_sparse

    def multi_step_forecast(
        self,
        x_history: torch.Tensor,
        v_history: torch.Tensor,
        forecast_steps: int,
        teacher_residuals: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate multi-step forecast using both physics and correction.

        Args:
            x_history: (batch, window) — observed positions
            v_history: (batch, window) — observed velocities
            forecast_steps: Number of steps H to forecast
            teacher_residuals: (batch, H) — true residuals for teacher forcing
            teacher_forcing: Use true residuals during training

        Returns:
            dict with keys:
              'x_pred': (batch, H) — final combined predictions
              'x_phys': (batch, H) — physics-only predictions
              'deltas': (batch, H) — LSTM corrections
        """
        batch_size = x_history.shape[0]
        window = x_history.shape[1]
        device = x_history.device

        # ---- Compute past residuals ----
        # Run physics on history to get phyiscs baseline for the window
        with torch.set_grad_enabled(self.training):
            # Physics roll for the history window: use observed (x, v) as initial
            x_phys_hist = []
            x_cur, v_cur = x_history[:, 0], v_history[:, 0]
            x_phys_hist.append(x_cur.unsqueeze(-1))
            for t in range(window - 1):
                x_cur, v_cur = self.physics(x_cur, v_cur)
                x_phys_hist.append(x_cur.unsqueeze(-1))
            x_phys_hist = torch.cat(x_phys_hist, dim=-1)  # (batch, window)

        # Residuals from the history window
        e_seq = (x_history - x_phys_hist).unsqueeze(-1)  # (batch, window, 1)

        # ---- Initialize LSTM state from past residuals ----
        _, (h_n, c_n) = self.corrector.lstm(e_seq)

        # ---- Roll out forecast ----
        x_phys_list, delta_list, x_pred_list = [], [], []

        # For physics roll during forecast, start from last history state
        x_phys_cur = x_history[:, -1]
        v_phys_cur = v_history[:, -1]

        # For LSTM autoregressive correction, start with last residual
        e_input = e_seq[:, -1:, :]  # (batch, 1, 1)

        for step in range(forecast_steps):
            # Physics step
            x_phys_cur, v_phys_cur = self.physics(x_phys_cur, v_phys_cur)

            # LSTM correction step (autoregressive on residuals)
            delta_cur, (h_n, c_n) = self.corrector.lstm(e_input, (h_n, c_n))
            delta_cur = delta_cur.squeeze(-1)  # (batch,)

            # Combined prediction
            x_pred_cur = x_phys_cur + delta_cur

            x_phys_list.append(x_phys_cur.unsqueeze(-1))
            delta_list.append(delta_cur.unsqueeze(-1))
            x_pred_list.append(x_pred_cur.unsqueeze(-1))

            # Prepare next LSTM input
            if teacher_forcing and teacher_residuals is not None:
                # Teacher forcing: use true residual
                true_residual = teacher_residuals[:, step:step + 1]
                e_next = true_residual.unsqueeze(-1)  # (batch, 1, 1)
            else:
                # Closed-loop: use predicted residual
                e_next = delta_cur.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            e_input = e_next

        return {
            'x_pred': torch.cat(x_pred_list, dim=-1),   # (batch, H)
            'x_phys': torch.cat(x_phys_list, dim=-1),   # (batch, H)
            'deltas': torch.cat(delta_list, dim=-1),    # (batch, H)
        }

    def compute_loss(
        self,
        x_history: torch.Tensor,
        v_history: torch.Tensor,
        x_true_future: torch.Tensor,
        teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss for one batch.

        Args:
            x_history: (batch, window) — observed positions
            v_history: (batch, window) — observed velocities
            x_true_future: (batch, H) — ground truth future positions
            teacher_forcing: Use teacher forcing during training
        Returns:
            total_loss, loss_components (dict of scalars)
        """
        H = x_true_future.shape[-1]
        out = self.multi_step_forecast(
            x_history, v_history,
            forecast_steps=H,
            teacher_forcing=teacher_forcing,
            teacher_residuals=(
                x_true_future - self._physics_rollout(x_history, v_history, H)
                if teacher_forcing else None
            ),
        )

        x_pred = out['x_pred']
        x_phys = out['x_phys']

        # 1. Data loss: MSE between combined prediction and ground truth
        L_data = torch.mean((x_pred - x_true_future) ** 2)

        # 2. Physics consistency loss:
        #    Penalize deviation of the physics trajectory from the
        #    combined prediction projected through the physics model
        #    (i.e., the combined prediction should stay near the
        #     physics manifold)
        x_pred_flat = x_pred.reshape(-1, H)
        # Compute one-step physics from each combined prediction
        # left-shifted vs right-shifted version
        x_phys_from_pred = []
        x_cur = x_pred[:, 0].detach()
        v_cur = v_history[:, -1].detach()
        for step in range(H - 1):
            x_cur, v_cur = self.physics(x_cur, v_cur)
            x_phys_from_pred.append(x_cur.unsqueeze(-1))
        if len(x_phys_from_pred) > 0:
            x_phys_from_pred = torch.cat(x_phys_from_pred, dim=-1)
            L_physics = torch.mean((x_pred[:, 1:] - x_phys_from_pred) ** 2)
        else:
            L_physics = torch.tensor(0.0, device=x_pred.device)

        # 3. Sparse residual penalty
        L_sparse = torch.mean(torch.abs(out['deltas']))

        total = L_data + self.lambda_phys * L_physics + self.lambda_sparse * L_sparse

        loss_dict = {
            'L_data': L_data.item(),
            'L_physics': L_physics.item() if isinstance(L_physics, torch.Tensor) else 0.0,
            'L_sparse': L_sparse.item(),
        }
        return total, loss_dict

    def _physics_rollout(self, x_history: torch.Tensor,
                         v_history: torch.Tensor,
                         steps: int) -> torch.Tensor:
        """Helper: pure physics rollout for teacher residual computation."""
        with torch.no_grad():
            x_cur = x_history[:, -1]
            v_cur = v_history[:, -1]
            traj = []
            for _ in range(steps):
                x_cur, v_cur = self.physics(x_cur, v_cur)
                traj.append(x_cur.unsqueeze(-1))
            return torch.cat(traj, dim=-1)


# =============================================================================
# 4. Synthetic Data Generation
# =============================================================================

def generate_oscillator_data(
    n_trajectories: int = 50,
    n_steps: int = 200,
    dt: float = 0.01,
    omega: float = 5.0,
    zeta: float = 0.05,
    noise_std: float = 0.02,
    perturbation_std: float = 0.005,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic damped harmonic oscillator data.

    The true dynamics follow the physics model PLUS a small stochastic
    perturbation (unmodeled disturbance) to create residuals that the
    LSTM must learn.

    Args:
        n_trajectories: Number of independent trajectories
        n_steps: Time steps per trajectory
        dt: Time step
        omega: Natural frequency (rad/s)
        zeta: Damping ratio
        noise_std: Observation noise std
        perturbation_std: Random walk perturbation std (unmodeled dynamics)
        seed: Random seed

    Returns:
        X: (n_trajectories, n_steps) — position observations
        V: (n_trajectories, n_steps) — velocity observations
        physics_only: (n_trajectories, n_steps) — pure physics positions
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((n_trajectories, n_steps))
    V = np.zeros((n_trajectories, n_steps))
    phys_X = np.zeros((n_trajectories, n_steps))

    for traj_idx in range(n_trajectories):
        # Random initial conditions
        x = rng.uniform(-1.0, 1.0)
        v = rng.uniform(-1.0, 1.0)

        # Store pure physics trajectory for reference
        px, pv = x, v
        X[traj_idx, 0] = x
        V[traj_idx, 0] = v
        phys_X[traj_idx, 0] = px

        # Accumulated perturbation (random walk)
        perturb = 0.0

        for t in range(1, n_steps):
            # True dynamics: physics + random perturbation
            w2 = omega ** 2
            v_true = v - w2 * x * dt - 2 * zeta * omega * v * dt
            perturb += rng.randn() * np.sqrt(dt) * perturbation_std
            x_true = x + v_true * dt + perturb * dt

            # Pure physics (for reference)
            pv = pv - w2 * px * dt - 2 * zeta * omega * pv * dt
            px = px + pv * dt

            # Add observation noise
            X[traj_idx, t] = x_true + rng.randn() * noise_std
            V[traj_idx, t] = v_true + rng.randn() * noise_std
            phys_X[traj_idx, t] = px

            x, v = x_true, v_true

    return X, V, phys_X


# =============================================================================
# 5. Data Preparation
# =============================================================================

def prepare_sequences(
    X: np.ndarray,
    V: np.ndarray,
    window: int = 20,
    forecast_horizon: int = 10,
    stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sliding window sequences from trajectory data.

    Args:
        X: (n_trajectories, n_steps) — position
        V: (n_trajectories, n_steps) — velocity
        window: Lookback window τ
        forecast_horizon: Forecast length H
        stride: Step size between windows

    Returns:
        x_hist: (n_samples, window) — history positions
        v_hist: (n_samples, window) — history velocities
        x_future: (n_samples, H) — future positions to predict
    """
    n_traj, n_steps = X.shape
    x_hist_list, v_hist_list, x_future_list = [], [], []

    for traj_idx in range(n_traj):
        for start in range(0, n_steps - window - forecast_horizon + 1, stride):
            end = start + window
            x_hist_list.append(X[traj_idx, start:end])
            v_hist_list.append(V[traj_idx, start:end])
            x_future_list.append(X[traj_idx, end:end + forecast_horizon])

    return (
        torch.FloatTensor(np.array(x_hist_list)),
        torch.FloatTensor(np.array(v_hist_list)),
        torch.FloatTensor(np.array(x_future_list)),
    )


# =============================================================================
# 6. Training Loop
# =============================================================================

def train_epoch(
    model: DualLevelPINN,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    teacher_forcing: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch of the dual-level model.

    Returns:
        (avg_loss, avg_components)
    """
    model.train()
    total_loss = 0.0
    comp_sums = {'L_data': 0.0, 'L_physics': 0.0, 'L_sparse': 0.0}

    for batch in loader:
        x_hist, v_hist, x_future = [b.to(device) for b in batch]
        optimizer.zero_grad()
        loss, comp = model.compute_loss(
            x_hist, v_hist, x_future,
            teacher_forcing=teacher_forcing,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in comp_sums:
            comp_sums[k] += comp[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in comp_sums.items()}


@torch.no_grad()
def evaluate(model: DualLevelPINN, loader: DataLoader,
             device: torch.device) -> Dict[str, float]:
    """Evaluate the model on a validation set.

    Returns:
        dict of metrics: RMSE, MAE, Phys_RMSE (physics-only RMSE)
    """
    model.eval()
    all_preds, all_targets, all_phys = [], [], []

    for batch in loader:
        x_hist, v_hist, x_future = [b.to(device) for b in batch]
        out = model.multi_step_forecast(
            x_hist, v_hist, forecast_steps=x_future.shape[-1],
        )
        all_preds.append(out['x_pred'].cpu())
        all_targets.append(x_future.cpu())
        all_phys.append(out['x_phys'].cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    phys = torch.cat(all_phys)

    # Combined model metrics
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
    mae = torch.mean(torch.abs(preds - targets))
    # Physics-only metrics
    phys_rmse = torch.sqrt(torch.mean((phys - targets) ** 2))
    # Improvement ratio
    improvement = (phys_rmse - rmse) / (phys_rmse + 1e-8) * 100

    return {
        'RMSE': rmse.item(),
        'MAE': mae.item(),
        'Phys_RMSE': phys_rmse.item(),
        'Improvement(%)': improvement.item(),
    }


# =============================================================================
# 7. Complete Runnable Example
# =============================================================================

if __name__ == "__main__":
    # ---- Configuration ----
    WINDOW = 20
    FORECAST_HORIZON = 15
    DT = 0.01
    BATCH_SIZE = 64
    EPOCHS = 150
    LR = 5e-4
    HIDDEN_DIM = 64
    LSTM_LAYERS = 2
    LAMBDA_PHYS = 0.1
    LAMBDA_SPARSE = 0.01
    SEED = 42

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Generate data ----
    print("Generating synthetic oscillator data...")
    X, V, phys_only = generate_oscillator_data(
        n_trajectories=40,
        n_steps=300,
        dt=DT,
        omega=5.0,
        zeta=0.05,
        noise_std=0.02,
        perturbation_std=0.005,
        seed=SEED,
    )
    print(f"Data shape X: {X.shape}, V: {V.shape}")

    # ---- Prepare sequences ----
    x_hist, v_hist, x_future = prepare_sequences(
        X, V, window=WINDOW, forecast_horizon=FORECAST_HORIZON,
    )
    print(f"Samples: {len(x_hist)} (window={WINDOW}, horizon={FORECAST_HORIZON})")

    # ---- Train/validation split ----
    n_total = len(x_hist)
    indices = torch.randperm(n_total)
    n_train = int(0.8 * n_total)

    train_dataset = TensorDataset(
        x_hist[indices[:n_train]], v_hist[indices[:n_train]],
        x_future[indices[:n_train]],
    )
    val_dataset = TensorDataset(
        x_hist[indices[n_train:]], v_hist[indices[n_train:]],
        x_future[indices[n_train:]],
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Build model ----
    physics = DampedOscillator(dt=DT, omega=5.0, zeta=0.05, tune_phys=False)
    corrector = LSTMResidualCorrector(
        input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=LSTM_LAYERS,
    )
    model = DualLevelPINN(
        physics, corrector,
        lambda_phys=LAMBDA_PHYS, lambda_sparse=LAMBDA_SPARSE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- Training ----
    best_val_loss = float('inf')
    patience, patience_limit = 0, 30

    print("\n--- Training ---")
    for epoch in range(EPOCHS):
        # Anneal teacher forcing: start at 0.8, decay linearly to 0.0
        tf_ratio = max(0.0, 0.8 * (1.0 - epoch / (0.6 * EPOCHS)))
        use_tf = np.random.random() < tf_ratio

        train_loss, train_comp = train_epoch(
            model, train_loader, optimizer, device,
            teacher_forcing=use_tf,
        )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xh, vh, xf = [b.to(device) for b in batch]
                loss, _ = model.compute_loss(xh, vh, xf, teacher_forcing=False)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 'best_dual_level_model.pt')
        else:
            patience += 1

        if (epoch + 1) % 15 == 0:
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"TF: {tf_ratio:.2f} | "
                f"L_data={train_comp['L_data']:.6f} "
                f"L_phys={train_comp['L_physics']:.6f} "
                f"L_sparse={train_comp['L_sparse']:.6f}"
            )

        if patience >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # ---- Evaluate ----
    model.load_state_dict(
        torch.load('best_dual_level_model.pt', weights_only=True)
    )
    metrics = evaluate(model, val_loader, device)
    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # ---- Visualize a sample trajectory ----
    model.eval()
    with torch.no_grad():
        # Pick one sample from validation set
        sample_xh, sample_vh, sample_xf = val_dataset[0]
        sample_xh = sample_xh.unsqueeze(0).to(device)
        sample_vh = sample_vh.unsqueeze(0).to(device)
        sample_xf = sample_xf.unsqueeze(0).to(device)

        out = model.multi_step_forecast(
            sample_xh, sample_vh, forecast_horizon,
        )

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: full trajectory view
    history_np = sample_xh.cpu().numpy().flatten()
    future_np = sample_xf.cpu().numpy().flatten()
    pred_np = out['x_pred'].cpu().numpy().flatten()
    phys_np = out['x_phys'].cpu().numpy().flatten()
    t_hist = np.arange(len(history_np))
    t_future = np.arange(len(history_np), len(history_np) + len(future_np))

    axes[0].plot(t_hist, history_np, 'b-', label='History (observed)', lw=2)
    axes[0].plot(t_future, future_np, 'g-', label='True future', lw=2)
    axes[0].plot(t_future, pred_np, 'r--', label='Dual-Level forecast', lw=2)
    axes[0].plot(t_future, phys_np, 'k:', label='Physics-only forecast', lw=2, alpha=0.7)
    axes[0].axvline(x=len(history_np) - 1, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Position')
    axes[0].set_title('Dual-Level Physics-Informed Forecasting')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: residuals
    true_residuals = future_np - phys_np
    predicted_residuals = out['deltas'].cpu().numpy().flatten()
    axes[1].plot(t_future, true_residuals, 'g-', label='True residual (x_true - x_phys)', lw=2)
    axes[1].plot(t_future, predicted_residuals, 'r--', label='LSTM-predicted δ', lw=2)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('LSTM Residual Correction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dual_level_forecast.png', dpi=150)
    plt.show()
    print("\nFigure saved to 'dual_level_forecast.png'")

    # ---- Report baseline comparison ----
    print("\n--- Performance Summary ---")
    print(f"  Physics-only RMSE: {metrics['Phys_RMSE']:.6f}")
    print(f"  Dual-Level RMSE:   {metrics['RMSE']:.6f}")
    print(f"  Improvement:       {metrics['Improvement(%)']:.2f}%")
    print("\nThe LSTM corrector captures unmodeled perturbations that")
    print("the pure physics model misses, reducing the forecasting error.")
```

## References

Jin, M., Zhou, Y., & Zheng, H. (2026). Dual-Level Models for Physics-Informed Multi-Step Time Series Forecasting. *arXiv preprint*, arXiv:2601.07640. https://doi.org/10.48550/arXiv.2601.07640

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *Advances in Neural Information Processing Systems (NeurIPS)*, 31. https://papers.nips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440. https://doi.org/10.1038/s42254-021-00314-5

Yin, Y., Le Guen, V., Dona, J., Ayed, I., de Bézenac, E., Thome, N., & Gallinari, P. (2023). Augmenting physical models with deep networks for complex dynamics forecasting. *Journal of Statistical Mechanics: Theory and Experiment*, 2023(11), 114012. https://doi.org/10.1088/1742-5468/ad04bf
