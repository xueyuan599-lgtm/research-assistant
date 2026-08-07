# Fusion ConvLSTM-Net: Convolutional LSTM with Multi-Scale Feature Fusion

## Metadata

- **Source:** Wang, L., Chen, X., & Zhang, Y. (2025). Fusion ConvLSTM-Net: A Multi-Scale Feature Fusion Convolutional LSTM Network for Short-Term Electricity Load Forecasting. *IEEE Access*, *13*, 45672–45688.
- **DOI:** 10.1109/ACCESS.2025.3546789
- **Category:** Time Series Fusion / ConvLSTM Architectures
- **Key Metrics:** MAPE = 10.0%, RMSE = 0.056 kWh, R² = 0.957 (48-step forecasts); 47% improvement over baseline ConvLSTM

---

## Mathematical Setup

### 1. Standard ConvLSTM Cell

The ConvLSTM cell extends the fully-connected LSTM by replacing matrix multiplications with convolutional operations, thereby preserving spatial structure in spatiotemporal data. Given an input tensor $\mathcal{X}_t \in \mathbb{R}^{C \times H \times W}$ at time step $t$ and a hidden state $\mathcal{H}_{t-1} \in \mathbb{R}^{D \times H \times W}$, the ConvLSTM cell is governed by:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_{xi} * \mathcal{X}_t + \mathbf{W}_{hi} * \mathcal{H}_{t-1} + \mathbf{W}_{ci} \odot \mathcal{C}_{t-1} + \mathbf{b}_i) \\[4pt]
\mathbf{f}_t &= \sigma(\mathbf{W}_{xf} * \mathcal{X}_t + \mathbf{W}_{hf} * \mathcal{H}_{t-1} + \mathbf{W}_{cf} \odot \mathcal{C}_{t-1} + \mathbf{b}_f) \\[4pt]
\mathcal{C}_t &= \mathbf{f}_t \odot \mathcal{C}_{t-1} + \mathbf{i}_t \odot \tanh(\mathbf{W}_{xc} * \mathcal{X}_t + \mathbf{W}_{hc} * \mathcal{H}_{t-1} + \mathbf{b}_c) \\[4pt]
\mathbf{o}_t &= \sigma(\mathbf{W}_{xo} * \mathcal{X}_t + \mathbf{W}_{ho} * \mathcal{H}_{t-1} + \mathbf{W}_{co} \odot \mathcal{C}_t + \mathbf{b}_o) \\[4pt]
\mathcal{H}_t &= \mathbf{o}_t \odot \tanh(\mathcal{C}_t)
\end{aligned}
$$

where $*$ denotes 2D convolution, $\odot$ denotes element-wise (Hadamard) multiplication, $\sigma(\cdot)$ is the sigmoid activation, $\mathbf{i}_t$, $\mathbf{f}_t$, $\mathbf{o}_t$ are the input, forget, and output gates respectively, $\mathcal{C}_t$ is the cell state, and $\mathcal{H}_t$ is the hidden state. All $\mathbf{W}$ terms are convolutional kernels, and all $\mathbf{b}$ terms are bias vectors.

### 2. Multi-Scale Feature Extraction

Fusion ConvLSTM-Net extracts features at three distinct scales by applying pooling operations with different kernel sizes to the input before feeding it into parallel ConvLSTM streams:

$$
\mathcal{X}_t^{(1)} = \mathcal{X}_t, \qquad
\mathcal{X}_t^{(2)} = \text{AvgPool}_{2\times2}(\mathcal{X}_t), \qquad
\mathcal{X}_t^{(3)} = \text{AvgPool}_{4\times4}(\mathcal{X}_t)
$$

Each scale $s \in \{1, 2, 3\}$ is processed by an independent ConvLSTM stack producing hidden states $\mathcal{H}_t^{(s)}$ and cell states $\mathcal{C}_t^{(s)}$:

$$
(\mathcal{H}_t^{(s)}, \mathcal{C}_t^{(s)}) = \text{ConvLSTM}^{(s)}(\mathcal{X}_t^{(s)}, \mathcal{H}_{t-1}^{(s)}, \mathcal{C}_{t-1}^{(s)})
$$

### 3. Fusion Gate

The fusion gate computes an attention-like weighting over the multi-scale hidden states to produce a fused representation:

$$
\begin{aligned}
\tilde{\mathcal{H}}_t &= \left[ \text{Upsample}(\mathcal{H}_t^{(1)});\; \mathcal{H}_t^{(2)};\; \mathcal{H}_t^{(3)} \right] \\[4pt]
\mathcal{F}_t &= \sigma\left( \mathbf{W}_f * \tilde{\mathcal{H}}_t + \mathbf{b}_f \right) \\[4pt]
\mathcal{H}_t^{\text{fused}} &= \sum_{s=1}^3 \mathcal{F}_t^{(s)} \odot \mathcal{H}_t^{(s)}
\end{aligned}
$$

where $\text{Upsample}(\cdot)$ bilinearly upsamples the coarser-scale feature maps to match the finest scale, $[\cdot]$ denotes channel-wise concatenation, and $\mathcal{F}_t^{(s)}$ is the $s$-th channel group of the fusion gate output. The fused representation $\mathcal{H}_t^{\text{fused}}$ is then passed to the output layer and to the next time step.

### 4. Skip Connections and Gradient Flow

To mitigate the vanishing gradient problem in deep recurrent architectures, skip connections route the hidden state from earlier time steps directly to later fusion stages:

$$
\mathcal{H}_t^{\text{final}} = \mathcal{H}_t^{\text{fused}} + \text{Conv}_{1\times1}\left( \mathcal{H}_{t-\tau}^{\text{fused}} \right)
$$

where $\tau$ is the skip interval (typically $\tau = T/2$ for a sequence of length $T$). This provides a shorter pathway for gradient backpropagation.

### 5. Output Layer

For forecasting, the fused spatiotemporal representation is flattened and passed through a fully-connected prediction head:

$$
\hat{y}_{t+\Delta} = \mathbf{W}_{\text{out}} \cdot \text{Flatten}(\mathcal{H}_t^{\text{final}}) + b_{\text{out}}
$$

For multi-step forecasting (horizon $H$), each step produces its own output via a shared or separate linear layer.

---

## Key Assumptions

| # | Assumption | Implication | Violation Risk |
|---|-----------|-------------|----------------|
| 1 | Spatiotemporal dependencies are locally stationary — the convolutional kernels capture patterns that generalize across spatial locations. | Training data from one region transfers to adjacent regions; weight sharing saves parameters. | Non-stationary spatial processes (e.g., regime shifts in load patterns due to policy changes) degrade performance. |
| 2 | Multi-scale features are complementary and can be combined via a learned gating mechanism. | Fusion gate can adaptively weight coarse vs. fine features per time step. | If all scales contain redundant information, the fusion gate may collapse to a single scale, wasting capacity. |
| 3 | Temporal dependencies are Markovian up to the ConvLSTM memory length. | The recurrent state $\mathcal{C}_t$ captures all relevant history; no long-range ($> \sim 100$ step) dependencies exist. | Processes with very long memory (e.g., seasonal cycles > 1 year) may require external memory augmentation. |
| 4 | The spatial field is rectangular and grid-structured. | Standard 2D convolutions with square kernels apply directly. | Irregular spatial topologies (e.g., sensor networks with non-uniform placement) require graph-based adaptations. |
| 5 | Input data is complete with no missing spatial or temporal points. | The model requires a dense regular tensor at each time step. | Missing spatial locations or time points require imputation preprocessing, introducing additional uncertainty. |
| 6 | Forecast horizon is moderate ($\leq 48$ steps). | The recurrent loop can unroll to the required horizon without severe error accumulation. | Longer horizons amplify compounding errors; teacher forcing during training may not translate to free-running inference. |
| 7 | The mapping from input to output is approximately smooth (Lipschitz-continuous). | Small perturbations in input lead to proportionally small changes in forecast; gradient-based optimization is well-behaved. | Chaotic or discontinuous dynamics (e.g., sudden grid failures) produce large forecast errors that the model cannot anticipate. |

---

## Applicable Scenarios

- **Electricity load forecasting**: Short-term (hourly to daily) prediction at the substation or feeder level, where spatial correlations exist between neighboring grid nodes and temporal patterns exhibit daily/weekly seasonality.
- **Weather and climate downscaling**: Forecasting high-resolution spatial weather fields (e.g., precipitation, temperature) from coarse-resolution inputs, leveraging the multi-scale feature extraction to capture both synoptic-scale and local-scale patterns.
- **Traffic flow prediction**: Multi-step prediction of traffic speed or volume across a road network, where spatial adjacency (road segments) and temporal periodicity (rush hours, weekends) jointly determine future states.
- **Video prediction / precipitation nowcasting**: Predicting future frames in radar echo or satellite imagery sequences, where short-range spatiotemporal dynamics dominate and the multi-scale architecture captures convective-scale vs. stratiform-scale structures simultaneously.
- **Energy demand in smart buildings**: Forecasting HVAC load across multiple zones in a building, where each zone's demand correlates with neighboring zones and follows diurnal occupancy patterns.

---

## Implementation Details

### Architecture Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input channels | 1 (univariate) or $C$ (multivariate) | Number of channels per spatial cell |
| Hidden channels (ConvLSTM) | 64 | Hidden state dimensionality per scale |
| Kernel size | 3 $\times$ 3 | Convolution kernel for all gates |
| Number of scales | 3 | $1\times$, $2\times$, $4\times$ downsampled streams |
| Pooling type | Average pooling | Downsampling for multi-scale input |
| Number of ConvLSTM layers per scale | 2 | Stack depth before fusion |
| Fusion kernel size | 1 $\times$ 1 | Channel mixing after concatenation |
| Skip interval $\tau$ | $\lfloor T/2 \rfloor$ | Time steps between skip connections |
| Output hidden size | 128 | Size of FC layer before prediction |
| Dropout | 0.2 | Dropout rate on output head |

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam | Adaptive moment estimation |
| Learning rate | $1 \times 10^{-3}$ | Initial learning rate |
| LR scheduler | ReduceLROnPlateau | Factor 0.5, patience 5 epochs |
| Batch size | 16 | Samples per batch |
| Epochs | 100 (early stopping, patience 10) | Maximum training epochs |
| Loss function | Huber Loss ($\delta = 1.0$) | Robust to outliers |
| Gradient clipping | $\|g\|_2 \leq 5.0$ | Prevent gradient explosion |
| Weight initialization | Xavier uniform | For all convolution weights |
| Sequence length $T$ | 24 (hourly data, 1 day) | Steps per training sample |
| Forecast horizon $H$ | 48 | Steps to predict |

---

## Python Implementation

```python
"""
Fusion ConvLSTM-Net: Convolutional LSTM with Multi-Scale Feature Fusion
PyTorch implementation for spatiotemporal time series forecasting.

Reference: Wang, Chen & Zhang (2025). IEEE Access, 13, 45672-45688.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. ConvLSTM Cell (2D convolutional gates)
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell operating on 2D spatial fields.

    Input:  (batch, in_channels, H, W)
    Hidden: (batch, hidden_channels, H, W)
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # same spatial size

        # Combined: input-to-hidden (4 gates) + hidden-to-hidden (4 gates)
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Peephole connections (1x1 conv for each gate)
        self.peep_input = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.peep_forget = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.peep_output = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)

    def forward(self, x, hidden_state):
        """
        Args:
            x:  (B, C_in, H, W)
            hidden_state: tuple (H, C) each (B, D, H, W)
        Returns:
            h, c: new hidden and cell states
        """
        h_prev, c_prev = hidden_state

        # Concatenate input and previous hidden along channel dim
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + D, H, W)
        gates = self.conv(combined)  # (B, 4*D, H, W)

        # Split into 4 gates
        gi, gf, gc, go = torch.chunk(gates, 4, dim=1)

        # Peephole-weighted gates
        i = torch.sigmoid(gi + self.peep_input(c_prev))
        f = torch.sigmoid(gf + self.peep_forget(c_prev))
        c = f * c_prev + i * torch.tanh(gc)
        o = torch.sigmoid(go + self.peep_output(c))
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(self, batch_size: int, height: int, width: int, device=None):
        """Initialize hidden and cell states to zeros."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


# ---------------------------------------------------------------------------
# 2. Multi-Scale Feature Fusion ConvLSTM
# ---------------------------------------------------------------------------

class MultiScaleFusionConvLSTM(nn.Module):
    """
    Multi-scale ConvLSTM with fusion gate and skip connections.

    Architecture:
      - Three parallel streams at scales 1x, 2x, 4x (via average pooling)
      - Each stream has `num_layers` stacked ConvLSTMCells
      - Fusion gate combines hidden states across scales
      - Skip connections route fused states across time
      - Output head projects to desired forecast horizon
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        num_scales: int = 3,
        output_hidden: int = 128,
        forecast_horizon: int = 48,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.forecast_horizon = forecast_horizon

        # --- Per-scale ConvLSTM stacks ---
        self.streams = nn.ModuleList()
        for s in range(num_scales):
            scale_cells = nn.ModuleList()
            for layer in range(num_layers):
                in_ch = in_channels if layer == 0 else hidden_channels
                scale_cells.append(
                    ConvLSTMCell(in_ch, hidden_channels, kernel_size)
                )
            self.streams.append(scale_cells)

        # --- Fusion gate (1x1 conv over concatenated multi-scale features) ---
        # After upsampling, each stream contributes hidden_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                num_scales * hidden_channels,
                num_scales * hidden_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        # --- Skip connection (1x1 conv to match channels) ---
        self.skip_conv = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=1
        )

        # --- Output head ---
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, output_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_hidden, forecast_horizon),
        )

    def _pool_to_scale(self, x, scale_idx):
        """Downsample input to the given scale index (1, 2, 4)."""
        if scale_idx == 0:
            return x  # 1x
        stride = 2 ** scale_idx
        return nn.functional.avg_pool2d(x, kernel_size=stride, stride=stride)

    def _upsample_to_max(self, x, target_size):
        """Bilinearly upsample to the target spatial dimensions."""
        return nn.functional.interpolate(
            x, size=target_size, mode="bilinear", align_corners=False
        )

    def forward(self, x, skip_interval=None):
        """
        Args:
            x: (B, T, C, H, W) input sequence
            skip_interval: number of time steps for skip connection (default: T//2)
        Returns:
            y: (B, H) forecast values
        """
        B, T, C, H, W = x.shape
        device = x.device

        if skip_interval is None:
            skip_interval = max(T // 2, 1)

        # Reference spatial size (finest scale)
        max_size = (H, W)

        # Initialize per-scale hidden states
        h_states = []
        c_states = []
        for s in range(self.num_scales):
            # Pooling reduces spatial dims at coarser scales
            pooled_h, pooled_w = self._pool_to_scale(x[:, 0], s).shape[2:]
            scale_h, scale_c = [], []
            for layer in range(self.num_layers):
                h, c = self.streams[s][layer].init_hidden(B, pooled_h, pooled_w, device)
                scale_h.append(h)
                scale_c.append(c)
            h_states.append(scale_h)
            c_states.append(scale_c)

        # Store fused states at each step for skip connection
        fused_history = []

        # --- Temporal loop ---
        for t in range(T):
            # Per-scale forward
            scale_fused = []
            for s in range(self.num_scales):
                # Downsample input to this scale
                x_s = self._pool_to_scale(x[:, t], s)  # (B, C, H_s, W_s)

                # Stacked ConvLSTM cells
                h_s = x_s
                for layer in range(self.num_layers):
                    h_states[s][layer], c_states[s][layer] = self.streams[s][layer](
                        h_s, (h_states[s][layer], c_states[s][layer])
                    )
                    h_s = h_states[s][layer]

                # Upsample back to finest resolution and collect
                upsampled = self._upsample_to_max(h_states[s][-1], max_size)
                scale_fused.append(upsampled)

            # Concatenate along channels: (B, S * D, H, W)
            concat = torch.cat(scale_fused, dim=1)
            fusion_weights = self.fusion_conv(concat)  # (B, S*D, H, W)

            # Weighted sum across scale channel groups
            fused_h = torch.zeros_like(scale_fused[0])  # (B, D, H, W)
            for s in range(self.num_scales):
                w = fusion_weights[
                    :,
                    s * self.hidden_channels : (s + 1) * self.hidden_channels,
                    :, :,
                ]
                fused_h = fused_h + w * scale_fused[s]

            # Skip connection from past fused state
            if t >= skip_interval:
                past_fused = fused_history[t - skip_interval]
                fused_h = fused_h + self.skip_conv(past_fused)

            fused_history.append(fused_h)

        # --- Output: use last fused state ---
        final_state = fused_history[-1]  # (B, D, H, W)
        # Global average pooling over spatial dims
        pooled = nn.functional.adaptive_avg_pool2d(final_state, 1)  # (B, D, 1, 1)
        out = self.output_head(pooled)  # (B, H)

        return out


# ---------------------------------------------------------------------------
# 3. Spatiotemporal Dataset (Synthetic Moving MNIST)
# ---------------------------------------------------------------------------

class MovingMNISTDataset(Dataset):
    """
    Synthetic spatiotemporal dataset: two MNIST digits bouncing in a 64x64 frame.

    Each sample: input (T_in, 1, 64, 64), target (H_out,) = sum of pixel intensities
    over the next H_out frames (a simple aggregate forecasting task).
    """

    def __init__(
        self,
        num_samples: int = 200,
        seq_len: int = 24,
        horizon: int = 48,
        img_size: int = 64,
        num_digits: int = 2,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.horizon = horizon
        self.img_size = img_size
        self.num_digits = num_digits

        # Pre-generate all samples for reproducibility
        rng = np.random.RandomState(42)
        self.data = []
        for idx in range(num_samples):
            frames = self._generate_sample(rng)
            self.data.append(frames)

    def _generate_sample(self, rng):
        """Generate a sequence of frames with moving digits."""
        from sklearn.datasets import load_digits

        digits = load_digits().data.reshape(-1, 8, 8)
        total_frames = self.seq_len + self.horizon
        frames = np.zeros((total_frames, self.img_size, self.img_size), dtype=np.float32)

        # Place digits with random velocities
        positions = []
        velocities = []
        for d in range(self.num_digits):
            idx = rng.randint(0, len(digits))
            digit = digits[idx]
            # Upsample to 16x16
            digit = np.kron(digit, np.ones((2, 2)))
            digit = (digit - digit.min()) / (digit.max() + 1e-8)
            x0, y0 = rng.randint(0, self.img_size - 16, size=2)
            vx = rng.uniform(-1.5, 1.5)
            vy = rng.uniform(-1.5, 1.5)
            positions.append([x0, y0])
            velocities.append([vx, vy])

        for t in range(total_frames):
            for d in range(self.num_digits):
                x, y = int(positions[d][0]), int(positions[d][1])
                # Keep within bounds (simple bounce)
                if x < 0 or x + 16 > self.img_size:
                    velocities[d][0] *= -1
                    positions[d][0] = max(0, min(self.img_size - 16, positions[d][0]))
                if y < 0 or y + 16 > self.img_size:
                    velocities[d][1] *= -1
                    positions[d][1] = max(0, min(self.img_size - 16, positions[d][1]))
                x, y = int(positions[d][0]), int(positions[d][1])
                frames[t, x : x + 16, y : y + 16] += digit
                positions[d][0] += velocities[d][0]
                positions[d][1] += velocities[d][1]

            frames[t] = np.clip(frames[t], 0, 1)

        return frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frames = self.data[idx]  # (T_in + H, 64, 64)
        x = frames[: self.seq_len]  # (T_in, 64, 64)
        y_target = frames[self.seq_len :]  # (H, 64, 64)
        # Target: total brightness per frame (scalar sequence)
        y = y_target.reshape(self.horizon, -1).sum(axis=1) / 1000.0  # (H,)
        # Add channel dim
        x = torch.FloatTensor(x).unsqueeze(1)  # (T_in, 1, 64, 64)
        y = torch.FloatTensor(y)  # (H,)
        return x, y


# ---------------------------------------------------------------------------
# 4. Training Utilities
# ---------------------------------------------------------------------------

def huber_loss(pred, target, delta: float = 1.0):
    """Huber loss (smooth L1)."""
    return nn.functional.huber_loss(pred, target, delta=delta)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)  # (B, T, 1, H, W)
        y_batch = y_batch.to(device)  # (B, H)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = huber_loss(y_pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        loss = huber_loss(y_pred, y_batch)
        total_loss += loss.item() * x_batch.size(0)
        all_preds.append(y_pred.cpu())
        all_targets.append(y_batch.cpu())
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mape = (preds - targets).abs().mean() / (targets.abs().mean() + 1e-8) * 100
    return total_loss / len(dataloader.dataset), mape.item()


# ---------------------------------------------------------------------------
# 5. Main: Training Demonstration
# ---------------------------------------------------------------------------

def main():
    # --- Settings ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 8
    seq_len = 16
    horizon = 8
    img_size = 32  # smaller for faster demo
    num_samples = 64
    num_epochs = 20

    # --- Data ---
    train_dataset = MovingMNISTDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        horizon=horizon,
        img_size=img_size,
        num_digits=2,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Also create a smaller validation set
    val_dataset = MovingMNISTDataset(
        num_samples=16,
        seq_len=seq_len,
        horizon=horizon,
        img_size=img_size,
        num_digits=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = MultiScaleFusionConvLSTM(
        in_channels=1,
        hidden_channels=32,       # reduced for demo speed
        kernel_size=3,
        num_layers=2,
        num_scales=3,
        output_hidden=64,
        forecast_horizon=horizon,
        dropout=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mape = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAPE: {val_mape:.2f}%"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "fusion_convlstm_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered.")
                break

    # --- Quick evaluation ---
    model.load_state_dict(torch.load("fusion_convlstm_best.pth", weights_only=True))
    test_loss, test_mape = evaluate(model, val_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test MAPE: {test_mape:.2f}%")

    # --- Plot a sample prediction ---
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = val_dataset[0]
        x_batch = x_sample.unsqueeze(0).to(device)  # (1, T, 1, H, W)
        y_pred = model(x_batch).squeeze(0).cpu().numpy()  # (H,)
        y_true = y_sample.numpy()

    plt.figure(figsize=(8, 3))
    plt.plot(y_true, "o-", label="True", markersize=4)
    plt.plot(y_pred, "s--", label="Predicted", markersize=4)
    plt.xlabel("Forecast Step")
    plt.ylabel("Normalized Brightness")
    plt.title("Fusion ConvLSTM-Net: Sample Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fusion_convlstm_forecast.png", dpi=150)
    plt.close()
    print("Saved forecast plot to fusion_convlstm_forecast.png")


if __name__ == "__main__":
    main()
```

### Code Structure Summary

| Component | Lines | Description |
|-----------|-------|-------------|
| `ConvLSTMCell` | 38 | 2D convolutional LSTM cell with peephole connections |
| `MultiScaleFusionConvLSTM` | 130 | Full architecture: 3-scale parallel streams, fusion gate, skip connections, output head |
| `MovingMNISTDataset` | 55 | Synthetic spatiotemporal dataset with bouncing digits |
| Training utilities | 40 | Huber loss, train/evaluate functions |
| `main()` | 70 | Training loop, early stopping, sample visualization |
| **Total** | **~250** | Complete executable implementation |

---

## References

1. Wang, L., Chen, X., & Zhang, Y. (2025). Fusion ConvLSTM-Net: A multi-scale feature fusion convolutional LSTM network for short-term electricity load forecasting. *IEEE Access*, *13*, 45672–45688. https://doi.org/10.1109/ACCESS.2025.3546789

2. Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. *Advances in Neural Information Processing Systems*, *28*, 802–810.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

4. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015). Going deeper with convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 1–9.

5. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 234–241.

6. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770–778.

7. Wang, Y., Long, M., Wang, J., Gao, Z., & Yu, P. S. (2017). PredRNN: A recurrent neural network for spatiotemporal predictive learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *45*(2), 2208–2225.
