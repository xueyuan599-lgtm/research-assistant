# Temperature- and Impedance-Aware LSTM-PINN for Physically Consistent Battery SOH Prediction

- **Source**: Kumar, P. N., Upadhya, P. R., Nischay, S., Pavan Kumar, G., Shobana, T. S., & Rashmi, K. B. (2026). A temperature- and impedance-aware LSTM–PINN framework for physically consistent battery SOH prediction. *Scientific Reports*, 16, Article 7568.
- **DOI**: 10.1038/s41598-026-37850-y
- **GitHub**: [praju120056/PINN-based-Hybrid-LSTM-Architecture-for-Battery-Degradation](https://github.com/praju120056/PINN-based-Hybrid-LSTM-Architecture-for-Battery-Degradation)
- **Category**: Time Series Fusion / Physics-Informed Deep Learning
- **Method**: LSTM backbone + Physics-Informed Neural Network (PINN) with Arrhenius temperature acceleration, impedance-growth modeling, and monotonic degradation constraint

## Mathematical Setup

### Problem Definition

Given a sequence of battery cycling data up to cycle $t$, predict the State of Health (SOH) at cycle $t+1$:

$$
\text{SOH}_t = \frac{Q_t}{Q_{\text{nominal}}} \times 100\%
$$

where $Q_t$ is the current discharge capacity and $Q_{\text{nominal}}$ is the nominal capacity.

**Input features** at each cycle $t$:
- $T_t$: Surface temperature (℃)
- $R_e^{(t)}$: Electrolyte resistance (Ω)
- $R_{ct}^{(t)}$: Charge transfer resistance (Ω)
- Previous SOH values: $\{\text{SOH}_{t-1}, \text{SOH}_{t-2}, \dots, \text{SOH}_{t-\tau}\}$

### LSTM Backbone

The framework first extracts temporal features using an LSTM encoder. Given input sequence $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_\tau]$ where $\mathbf{x}_t = [T_t, R_e^{(t)}, R_{ct}^{(t)}, \text{SOH}_{t-1}]$, the LSTM produces hidden states:

$$
\mathbf{h}_t = \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1}, \mathbf{c}_{t-1}), \quad t = 1, \dots, \tau
$$

The final hidden state $\mathbf{h}_\tau \in \mathbb{R}^{H}$ is used as a compact representation of the degradation history.

### Physics-Informed Degradation ODE

Battery capacity fade follows a **stretched-exponential degradation path** modeled by the first-order ODE:

$$
\frac{d\,\text{SOH}}{dt}_{\text{physics}} = -\alpha_{\text{eff}}^{\beta} \left(1 - \alpha_{\text{eff}} \cdot t\right)^{\beta - 1}
$$

where:
- $\alpha_{\text{eff}}$: Temperature- and impedance-adjusted effective degradation rate coefficient
- $\beta$: Aging-shape parameter controlling trajectory curvature ($\beta > 0$)

The closed-form solution of this ODE is:

$$
\text{SOH}_{\text{physics}}(t) = 1 - (\alpha_{\text{eff}} \cdot t)^{\beta}
$$

### Arrhenius Temperature Acceleration

The degradation rate follows an Arrhenius-type temperature dependence:

$$
\alpha(T) = \alpha_0 \cdot \exp\left(-\frac{E_a}{R \cdot T}\right)
$$

where:
- $\alpha_0$: Pre-exponential factor (baseline degradation rate)
- $E_a$: Activation energy ≈ 39.98 kJ/mol (consistent with SEI layer formation)
- $R$: Universal gas constant (8.314 J/(mol·K))
- $T$: Absolute temperature (K)

This captures the physical principle that elevated temperatures accelerate SEI growth and lithium inventory loss.

### DeepHTPM Impedance Subnetwork

Rather than using computationally costly electrochemical PDE solvers, the framework employs a **neural impedance estimation branch (DeepHTPM)** that dynamically regulates the degradation rate based on real-time impedance measurements:

$$
\alpha_{\text{eff}} = \alpha(T) \cdot f_{\theta}(R_e, R_{ct})
$$

where $f_{\theta}(\cdot)$ is a small feedforward network that learns the impedance-modulation factor from electrolyte and charge-transfer resistances.

### Multi-Component Loss Function

The total loss balances three objectives:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{pred}} \mathcal{L}_{\text{prediction}} + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{grad}} \mathcal{L}_{\text{gradient}}
$$

**1. Prediction loss** (Huber loss for robustness):

$$
\mathcal{L}_{\text{prediction}} = \frac{1}{N}\sum_{i=1}^{N} \text{Huber}\left(\widehat{\text{SOH}}_i - \text{SOH}_i^{\text{(true)}}\right)
$$

$$
\text{Huber}(\delta) = \begin{cases}
0.5\delta^2 & \text{if } |\delta| \leq 1 \\
|\delta| - 0.5 & \text{otherwise}
\end{cases}
$$

**2. Physics residual loss** (SmoothL1 between data-driven and physics-based gradients):

$$
\mathcal{L}_{\text{pde}} = \frac{1}{N}\sum_{i=1}^{N} \text{SmoothL1}\left(\frac{d\,\widehat{\text{SOH}}_i}{dt} - \frac{d\,\text{SOH}_i}{dt}_{\text{physics}}\right)
$$

**3. Monotonic gradient penalty** (ReLU-based enforcement of $d\text{SOH}/dt \leq 0$):

$$
\mathcal{L}_{\text{gradient}} = \frac{1}{N}\sum_{i=1}^{N} \text{ReLU}\left(\frac{d\,\widehat{\text{SOH}}_i}{dt}\right)
$$

This penalizes any predicted increase in SOH, ensuring physically consistent (non-increasing) degradation.

The loss weights are typically set to $\lambda_{\text{pred}} = 1.0$, $\lambda_{\text{pde}} = 0.1$, $\lambda_{\text{grad}} = 0.05$.

### Architecture Overview

```
Input Sequence (τ=20 cycles)
    │
    ▼
┌──────────────────────┐
│   LSTM Encoder       │──→ h_τ (feature vector)
│   (2 layers, 128 h)  │
└──────────────────────┘
    │
    ├────────────────────────────────────────┐
    ▼                                        ▼
┌──────────────────────┐         ┌──────────────────────────┐
│ Data-Driven Branch   │         │ Physics-Informed Branch  │
│ FC layers → SOH_pred │         │ Arrhenius(T) · f_θ(Re,Rct)
└──────────────────────┘         │ → α_eff → ODE → SOH_phys│
    │                            └──────────────────────────┘
    ▼                                        ▼
┌──────────────────────────────────────────────────────────┐
│                Fusion & Loss Computation                  │
│  L_total = λ_pred·L_pred + λ_pde·L_pde + λ_grad·L_grad   │
└──────────────────────────────────────────────────────────┘
```

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Arrhenius temperature dependence | $\alpha(T) = \alpha_0 \exp(-E_a/(RT))$ | Degradation accelerates with temperature; $E_a$ must be calibrated for each battery chemistry |
| Monotonic degradation | $\frac{d\text{SOH}}{dt} \leq 0$ | SOH never increases; forces physical consistency but may conflict with measurement noise rebound |
| Stretched-exponential capacity fade | $\text{SOH}(t) = 1 - (\alpha_{\text{eff}} t)^\beta$ | Capacity fade follows a power-law trajectory; valid for lithium-ion batteries under cyclic aging |
| Impedance and degradation coupling | $\alpha_{\text{eff}} = \alpha(T) \cdot f_\theta(R_e,R_{ct})$ | Impedance growth (Re, Rct) directly modulates degradation rate; requires EIS data availability |
| SEI formation as dominant mechanism | $E_a \approx 39.98\ \text{kJ/mol}$ | Consistent with solid-electrolyte interphase kinetics; other degradation modes (Li plating, particle cracking) may have different $E_a$ |
| Sufficient cycling history | $\tau = 20$ cycles window | Short window may miss long-term degradation trends; longer window reduces training samples |
| Uniform operating conditions | Constant C-rate within each cycle | Does not model dynamic load profiles; real-world driving patterns may violate this |

## Applicable Scenarios

**When to use:**
- Battery SOH estimation with available temperature and impedance spectroscopy (EIS) data
- Applications requiring physically consistent predictions (no SOH recovery artifacts)
- Long-horizon degradation forecasting where pure data-driven models exhibit drift
- Transfer learning across battery cells with different degradation paths

**When NOT to use:**
- No impedance data available (Re, Rct measurements needed for the DeepHTPM branch)
- Extremely short cycling history (< 20 cycles)
- Battery chemistries where degradation is not monotonically decreasing (e.g., some LFP cells show capacity recovery)
- Real-time embedded deployment (LSTM + PINN forward pass may be computationally heavy)

**Comparison with alternatives:**
| Method | Strength | Limitation |
|--------|----------|------------|
| Pure LSTM | Flexible, no physics assumptions | May produce non-physical SOH recovery; poor extrapolation |
| Pure PINN (Raissi et al.) | Physical consistency | Requires PDE solver; slow; poor temporal dependency capture |
| Empirical models (e.g., exponential fit) | Simple, interpretable | Cannot capture complex degradation patterns |
| This LSTM-PINN | Both temporal learning and physical consistency | Requires impedance data; tuned $E_a$ per chemistry |

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| Sequence length $\tau$ | 20 | Longer → more context; shorter → more samples |
| LSTM hidden size | 128 | Increase for more complex degradation patterns |
| LSTM layers | 2 | 1–3 layers; deeper may overfit |
| $\lambda_{\text{pred}}$ | 1.0 | Anchor loss; always highest weight |
| $\lambda_{\text{pde}}$ | 0.1 | Start at 0.1; increase if physics violation persists |
| $\lambda_{\text{grad}}$ | 0.05 | Start small; increase if SOH recovery observed |
| $E_a$ (kJ/mol) | 39.98 | Calibrate per battery chemistry via Arrhenius plot |
| Learning rate | $1 \times 10^{-3}$ | Adam optimizer; ReduceLROnPlateau scheduler |
| Batch size | 32 | Smaller for limited cycling data |
| Epochs | 200 | Early stopping patience = 30 |

### Training Strategy

1. **Pre-train LSTM backbone** with only $\mathcal{L}_{\text{prediction}}$ for 50 epochs
2. **Joint fine-tune** with all three losses for remaining epochs
3. **Physics weight ramp-up**: Gradually increase $\lambda_{\text{pde}}$ from 0 to target over first 20 joint epochs
4. **Gradient penalty for monotonicity**: The $\mathcal{L}_{\text{gradient}}$ term uses `torch.autograd.grad` with `create_graph=True` to compute $\frac{d\widehat{\text{SOH}}}{dt}$ for backpropagation through the ODE residual

### Numerical Considerations

- The ODE residual requires computing $\frac{d\widehat{\text{SOH}}}{dt}$ via automatic differentiation: `torch.autograd.grad(soh_pred, t, grad_outputs=torch.ones_like(soh_pred), create_graph=True)`
- The stretched-exponential term $(\alpha_{\text{eff}} \cdot t)^\beta$ can overflow for large $t$; clip $\alpha_{\text{eff}} \cdot t$ to $[0, 1)$ to maintain numerical stability
- Huber loss is preferred over MSE for $\mathcal{L}_{\text{prediction}}$ to reduce sensitivity to outlier cycles
- The gradient penalty $\frac{d\widehat{\text{SOH}}}{dt}$ must be detached from the physics loss graph to avoid double counting gradients

## Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# =============================================================================
# 1. LSTM Feature Extractor
# =============================================================================

class LSTMEncoder(nn.Module):
    """LSTM encoder for sequential battery cycling data.

    Args:
        input_size: Number of input features (T, Re, Rct, SOH_prev) = 4
        hidden_size: LSTM hidden state dimension
        num_layers: Number of stacked LSTM layers
    """
    def __init__(self, input_size=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            h_last: (batch_size, hidden_size) — last hidden state
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # last layer's hidden state


# =============================================================================
# 2. DeepHTPM Impedance Subnetwork
# =============================================================================

class DeepHTPM(nn.Module):
    """Impedance modulation subnetwork.

    Learns the impedance-based degradation rate scaling factor
    from electrolyte resistance (Re) and charge transfer resistance (Rct).

    Args:
        input_dim: 2 (Re, Rct)
        hidden_dim: 32
    """
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensure positive scaling factor
        )

    def forward(self, re, rct):
        """
        Args:
            re:  (batch_size,) or (batch_size, 1) — electrolyte resistance
            rct: (batch_size,) or (batch_size, 1) — charge transfer resistance
        Returns:
            scaling: (batch_size, 1) — positive impedance modulation factor
        """
        if re.dim() == 1:
            re = re.unsqueeze(-1)
        if rct.dim() == 1:
            rct = rct.unsqueeze(-1)
        z = torch.cat([re, rct], dim=-1)
        return self.net(z)


# =============================================================================
# 3. Physics-Informed Loss Components
# =============================================================================

class PhysicsLoss(nn.Module):
    """Physics-informed loss functions for battery SOH prediction.

    Includes:
    - dSOH/dt residual from stretch-exponential ODE
    - Monotonicity penalty (ReLU on positive gradients)
    - Data prediction loss (Huber)
    """
    def __init__(self, Ea=39.98e3, R_gas=8.314, beta=1.5,
                 lambda_pred=1.0, lambda_pde=0.1, lambda_grad=0.05):
        super().__init__()
        self.Ea = Ea
        self.R_gas = R_gas
        self.beta = beta
        self.lambda_pred = lambda_pred
        self.lambda_pde = lambda_pde
        self.lambda_grad = lambda_grad
        self.huber = nn.HuberLoss(delta=1.0)
        self.smooth_l1 = nn.SmoothL1Loss()

    def arrhenius_alpha(self, temperature_celsius, alpha_0=0.01):
        """Compute Arrhenius temperature-dependent degradation rate.

        Args:
            temperature_celsius: Surface temperature in Celsius
            alpha_0: Pre-exponential factor (baseline rate)
        Returns:
            alpha_T: Temperature-adjusted degradation rate
        """
        T_kelvin = temperature_celsius + 273.15  # convert to Kelvin
        return alpha_0 * torch.exp(-self.Ea / (self.R_gas * T_kelvin))

    def ode_residual(self, soh_pred, t_normalized, alpha_eff):
        """Compute the stretch-exponential ODE residual.

        dSOH/dt = -α_eff^β * (1 - α_eff * t)^(β - 1)

        Args:
            soh_pred: Predicted SOH (detached for gradient computation)
            t_normalized: Normalized cycle time [0, 1]
            alpha_eff: Effective degradation rate (temperature + impedance adjusted)
        Returns:
            residual: Difference between data-driven and physics-based dSOH/dt
        """
        # Data-driven gradient via autograd
        soh_pred = soh_pred.requires_grad_(True)
        t_in = t_normalized.requires_grad_(True)

        dsoh_dt_data = torch.autograd.grad(
            soh_pred, t_in,
            grad_outputs=torch.ones_like(soh_pred),
            create_graph=True,
        )[0]

        # Physics-based gradient from stretch-exponential ODE
        alpha_clamped = torch.clamp(alpha_eff * t_normalized, 0, 1 - 1e-6)
        dsoh_dt_physics = -(alpha_eff ** self.beta) * (1 - alpha_clamped) ** (self.beta - 1)

        return dsoh_dt_data - dsoh_dt_physics

    def forward(self, soh_pred, soh_true, t_normalized, temperature, re, rct):
        """
        Args:
            soh_pred: (batch_size,) — predicted SOH
            soh_true: (batch_size,) — ground truth SOH
            t_normalized: (batch_size,) — normalized cycle time [0, 1]
            temperature: (batch_size,) — surface temperature (Celsius)
            re: (batch_size,) — electrolyte resistance
            rct: (batch_size,) — charge transfer resistance
        Returns:
            total_loss, loss_dict: Total and individual loss components
        """
        # 1. Prediction loss
        L_pred = self.huber(soh_pred, soh_true)

        # 2. Physics residual (ODE)
        alpha_T = self.arrhenius_alpha(temperature)
        # Clamp impedance scaling to prevent extreme values
        scaling = torch.clamp(self.lambda_pde_factor if hasattr(self, 'lambda_pde_factor') else 1.0, 0.1, 10.0)
        alpha_eff = alpha_T  # In full implementation: alpha_T * DeepHTPM(re, rct).squeeze()
        # For simplicity here, use alpha_T directly
        residual = self.ode_residual(soh_pred, t_normalized, alpha_eff)
        L_pde = self.smooth_l1(residual, torch.zeros_like(residual))

        # 3. Monotonic gradient penalty
        dsoh_dt = torch.autograd.grad(
            soh_pred, t_normalized,
            grad_outputs=torch.ones_like(soh_pred),
            create_graph=True,
        )[0]
        L_grad = torch.mean(torch.relu(dsoh_dt))

        total = self.lambda_pred * L_pred + self.lambda_pde * L_pde + self.lambda_grad * L_grad
        return total, {'L_pred': L_pred.item(), 'L_pde': L_pde.item(), 'L_grad': L_grad.item()}


# =============================================================================
# 4. Full LSTM-PINN Model
# =============================================================================

class LSTM_PINN_BatterySOH(nn.Module):
    """Complete LSTM-PINN framework for battery SOH prediction.

    Combines LSTM temporal encoder with physics-informed degradation modeling.
    """
    def __init__(self, input_size=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.deephtpm = DeepHTPM()
        self.physics_loss = PhysicsLoss()

    def forward(self, seq, temperature, re, rct):
        """
        Args:
            seq: (batch, seq_len, 4) — [T, Re, Rct, SOH_prev]
            temperature: (batch,) — current surface temperature
            re: (batch,) — current electrolyte resistance
            rct: (batch,) — current charge transfer resistance
        Returns:
            soh_pred: (batch,) — predicted SOH
        """
        h = self.encoder(seq)          # (batch, hidden_size)
        soh_pred = self.regressor(h)   # (batch, 1)
        return soh_pred.squeeze(-1)


# =============================================================================
# 5. Training Loop
# =============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch of the LSTM-PINN model."""
    model.train()
    total_loss = 0.0
    loss_components = {'L_pred': 0.0, 'L_pde': 0.0, 'L_grad': 0.0}

    for batch in dataloader:
        seq, temp, re, rct, soh_true, t_norm = [b.to(device) for b in batch]

        optimizer.zero_grad()
        soh_pred = model(seq, temp, re, rct)
        loss, losses = model.physics_loss(soh_pred, soh_true, t_norm, temp, re, rct)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += losses[k]

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model and compute metrics."""
    model.eval()
    preds, targets = [], []
    for batch in dataloader:
        seq, temp, re, rct, soh_true, t_norm = [b.to(device) for b in batch]
        soh_pred = model(seq, temp, re, rct)
        preds.append(soh_pred.cpu())
        targets.append(soh_true.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
    mae = torch.mean(torch.abs(preds - targets))
    mape = torch.mean(torch.abs((preds - targets) / (targets + 1e-6))) * 100
    ss_res = torch.sum((preds - targets) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-6)

    return {
        'RMSE': rmse.item() * 100,   # percentage
        'MAE': mae.item() * 100,     # percentage
        'MAPE': mape.item(),         # percentage
        'R²': r2.item(),
    }


# =============================================================================
# 6. Synthetic Data Generation (NASA-like)
# =============================================================================

def generate_synthetic_battery_data(n_cells=10, n_cycles=200, seed=42):
    """Generate synthetic battery degradation data mimicking NASA dataset.

    Each cell follows: SOH(t) = 1 - (α_eff * t)^β + noise
    with temperature and impedance variations.
    """
    rng = np.random.RandomState(seed)
    data = []

    for cell_id in range(n_cells):
        # Cell-specific parameters
        alpha_0 = rng.uniform(0.005, 0.015)
        beta = rng.uniform(1.2, 2.0)
        temp_base = rng.uniform(20, 40)  # Celsius

        for cycle in range(n_cycles):
            t = cycle / n_cycles  # normalized time
            # Temperature variations
            temp = temp_base + rng.randn() * 2
            # Arrhenius
            T_k = temp + 273.15
            Ea = 39.98e3
            R_gas = 8.314
            alpha_T = alpha_0 * np.exp(-Ea / (R_gas * T_k))
            # Impedance growth with cycling
            re = 0.01 + 0.005 * t + rng.randn() * 0.001
            rct = 0.02 + 0.02 * t + rng.randn() * 0.002
            # Impedance modulation
            z_scale = 1.0 + (re - 0.01) * 5 + (rct - 0.02) * 2
            alpha_eff = alpha_T * max(z_scale, 0.5)
            # SOH with noise
            soh = max(0.6, 1.0 - (alpha_eff * t) ** beta + rng.randn() * 0.005)

            data.append({
                'cell_id': cell_id,
                'cycle': cycle,
                't_normalized': t,
                'temperature': temp,
                're': re,
                'rct': rct,
                'soh': soh,
            })

    return data


# =============================================================================
# 7. Prepare Sequences
# =============================================================================

def prepare_sequences(data, seq_len=20):
    """Convert raw battery data into sliding window sequences.

    Args:
        data: List of dicts with keys (cell_id, cycle, t_normalized,
              temperature, re, rct, soh)
        seq_len: Input sequence length (τ = 20 in paper)
    Returns:
        Tensors: X_seq, temperature, re, rct, soh_target, t_norm
    """
    X_seq, X_temp, X_re, X_rct, y_soh, y_t = [], [], [], [], [], []

    # Group by cell
    from collections import defaultdict
    cell_data = defaultdict(list)
    for row in data:
        cell_data[row['cell_id']].append(row)
    for cell_id, rows in cell_data.items():
        rows = sorted(rows, key=lambda r: r['cycle'])
        for i in range(seq_len, len(rows)):
            seq = []
            for j in range(i - seq_len, i):
                r = rows[j]
                seq.append([r['temperature'], r['re'], r['rct'], r['soh']])
            target = rows[i]
            X_seq.append(seq)
            X_temp.append(target['temperature'])
            X_re.append(target['re'])
            X_rct.append(target['rct'])
            y_soh.append(target['soh'])
            y_t.append(target['t_normalized'])

    return (torch.FloatTensor(np.array(X_seq)),
            torch.FloatTensor(np.array(X_temp)),
            torch.FloatTensor(np.array(X_re)),
            torch.FloatTensor(np.array(X_rct)),
            torch.FloatTensor(np.array(y_soh)),
            torch.FloatTensor(np.array(y_t)))


# =============================================================================
# 8. Complete Training Example
# =============================================================================

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Generate synthetic data ---
    print("Generating synthetic battery degradation data...")
    raw_data = generate_synthetic_battery_data(n_cells=8, n_cycles=200)
    X_seq, X_temp, X_re, X_rct, y_soh, y_t = prepare_sequences(raw_data, seq_len=20)
    print(f"Total samples: {len(X_seq)}")

    # --- Train/val split ---
    n_train = int(0.8 * len(X_seq))
    indices = torch.randperm(len(X_seq))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = TensorDataset(
        X_seq[train_idx], X_temp[train_idx], X_re[train_idx],
        X_rct[train_idx], y_soh[train_idx], y_t[train_idx],
    )
    val_dataset = TensorDataset(
        X_seq[val_idx], X_temp[val_idx], X_re[val_idx],
        X_rct[val_idx], y_soh[val_idx], y_t[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- Initialize model ---
    model = LSTM_PINN_BatterySOH(
        input_size=4, hidden_size=128, num_layers=2,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # --- Training loop ---
    n_epochs = 100
    best_val_loss = float('inf')
    patience_counter = 0

    print("\n--- Training ---")
    for epoch in range(n_epochs):
        train_loss, train_components = train_epoch(model, train_loader, optimizer, device)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq, temp, re, rct, soh_true, t_norm = [b.to(device) for b in batch]
                soh_pred = model(seq, temp, re, rct)
                loss, _ = model.physics_loss(soh_pred, soh_true, t_norm, temp, re, rct)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_battery_soh_model.pt')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"L_pred: {train_components['L_pred']:.4f} | "
                  f"L_pde: {train_components['L_pde']:.4f} | "
                  f"L_grad: {train_components['L_grad']:.4f}")

        if patience_counter >= 30:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # --- Load best model and evaluate ---
    model.load_state_dict(torch.load('best_battery_soh_model.pt', weights_only=True))
    metrics = evaluate(model, val_loader, device)

    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Visualize predictions ---
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            seq, temp, re, rct, soh_true, t_norm = [b.to(device) for b in batch]
            soh_pred = model(seq, temp, re, rct)
            preds_list.append(soh_pred.cpu().numpy())
            targets_list.append(soh_true.cpu().numpy())

    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    plt.figure(figsize=(10, 6))
    plt.scatter(targets * 100, preds * 100, alpha=0.5, s=20)
    plt.plot([60, 100], [60, 100], 'r--', lw=2, label='Ideal')
    plt.xlabel('True SOH (%)')
    plt.ylabel('Predicted SOH (%)')
    plt.title('LSTM-PINN Battery SOH Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('battery_soh_prediction.png', dpi=150)
    plt.show()

    print("\nPrediction plot saved to 'battery_soh_prediction.png'")
    print(f"\nPaper reference results on NASA dataset:")
    print(f"  RMSE: 1.01% | MAE: 0.60–0.62% | MAPE: 0.83% | R²: 0.9863")
```

## References

Kumar, P. N., Upadhya, P. R., Nischay, S., Pavan Kumar, G., Shobana, T. S., & Rashmi, K. B. (2026). A temperature- and impedance-aware LSTM–PINN framework for physically consistent battery SOH prediction. *Scientific Reports*, 16, Article 7568. https://doi.org/10.1038/s41598-026-37850-y

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, S., ... & Braatz, R. D. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), 383–391. https://doi.org/10.1038/s41560-019-0356-8

Zhang, Y., Tang, Q., Zhang, Y., Wang, J., Stimming, U., & Lee, A. A. (2020). Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning. *Nature Communications*, 11, Article 1706. https://doi.org/10.1038/s41467-020-15235-7
