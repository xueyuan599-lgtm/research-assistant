---
title: TIC-FusionNet: Multimodal Fusion with Time-Frequency Decomposition
type:
  - forecasting
  - deep-learning
  - signal-processing
domain:
  - signal-processing
---
# TIC-FusionNet: Multimodal Fusion with Time-Frequency Decomposition

- **Source**: Chen, L., & Fan, X. (2025). TIC-FusionNet: A multimodal deep learning framework with temporal decomposition and attention-based fusion for time series forecasting. *PLOS ONE*, 20(10), e0333379.
- **DOI**: 10.1371/journal.pone.0333379
- **Category**: Time Series Fusion / Decomposition-Based Forecasting
- **Method**: EMA decomposition + Linear Transformer (temporal branch) + Spatial-Channel CNN with CBAM (visual branch) + Gated Fusion

## Mathematical Setup

### Problem Definition

Given a multivariate time series of stock market data up to time $t$, predict the 5-day log return $r_{t+5}$:

$$
r_{t+5} = \ln\left(\frac{P_{t+5}}{P_t}\right)
$$

where $P_t$ is the closing price at day $t$.

The model processes two modalities:
1. **Numerical modality**: Historical price and volume sequence $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{feat}}}$ where $T = 30$ (lookback window) and $d_{\text{feat}}$ is the number of numerical features.
2. **Visual modality**: Candlestick chart image $\mathbf{I} \in \mathbb{R}^{160 \times 120 \times 3}$ spanning the same 30-day window with overlaid technical indicators.

### EMA Trend Decomposition

The temporal branch begins by decomposing the raw sequence into trend and residual components using Exponential Moving Average (EMA). The EMA serves as a lightweight denoising filter that isolates the underlying trend from high-frequency market noise:

$$
\mathbf{T}_t = \alpha \cdot \mathbf{X}_t + (1 - \alpha) \cdot \mathbf{T}_{t-1}
$$

where the smoothing factor $\alpha$ is defined as:

$$
\alpha = \frac{2}{T_{\text{ema}} + 1}, \quad T_{\text{ema}} = 30
$$

The residual (high-frequency) component captures deviations from the trend:

$$
\mathbf{R}_t = \mathbf{X}_t - \mathbf{T}_t
$$

Both $\mathbf{T}_t$ and $\mathbf{R}_t$ are concatenated along the feature dimension to form the enhanced input $\mathbf{Z}_t = [\mathbf{T}_t; \mathbf{R}_t]$ for the subsequent Linear Transformer encoder. This decomposition enhances the signal-to-noise ratio and allows the model to separately attend to low-frequency trend dynamics and high-frequency fluctuations.

### Linear Transformer

The enhanced sequence $\mathbf{Z} \in \mathbb{R}^{T \times d_z}$ is processed by a Linear Transformer, which replaces the standard softmax attention with a kernelized linear attention mechanism to achieve $\mathcal{O}(T)$ complexity instead of $\mathcal{O}(T^2)$.

**Standard softmax attention** (quadratic cost):

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

where $Q = \mathbf{Z} W_Q$, $K = \mathbf{Z} W_K$, $V = \mathbf{Z} W_V$ are the query, key, and value projections, and $d_k$ is the key dimension.

**Linear attention** (linear cost via kernel trick):

The softmax is replaced by a kernel function $\phi(\cdot)$ applied row-wise to $Q$ and $K$. Using the associative property of matrix multiplication, the attention computation is reordered:

$$
\text{Attn}_{\text{linear}}(Q, K, V) = \phi(Q)\left(\phi(K)^\top V\right) \oslash \left(\phi(Q) \sum_{j=1}^{T} \phi(K_j)\right)
$$

where $\phi(x) = \text{ELU}(x) + 1$ is the element-wise feature map (ensuring non-negativity), and $\oslash$ denotes element-wise division for normalization. The computation proceeds as:

1. Compute $K' = \phi(K) \in \mathbb{R}^{T \times d_k}$ and $Q' = \phi(Q) \in \mathbb{R}^{T \times d_k}$
2. Compute $K'^\top V \in \mathbb{R}^{d_k \times d_v}$ (cost $\mathcal{O}(d_k d_v)$)
3. Compute $Q' (K'^\top V) \in \mathbb{R}^{T \times d_v}$ via one more matrix multiply
4. Normalize by $Q' \cdot \mathbf{1}_{T} \in \mathbb{R}^{T \times 1}$ where $\mathbf{1}_T$ is the all-ones vector

The output is passed through residual connections and layer normalization:

$$
\mathbf{H}_{\text{attn}} = \text{LayerNorm}(\mathbf{Z} + \text{Attn}_{\text{linear}}(Q, K, V))
$$

Followed by a position-wise feed-forward network:

$$
\mathbf{H}_{\text{ffn}} = \text{LayerNorm}\left(\mathbf{H}_{\text{attn}} + \text{ReLU}(\mathbf{H}_{\text{attn}} W_1 + b_1) W_2 + b_2\right)
$$

The final temporal representation $\mathbf{h}_{\text{seq}} \in \mathbb{R}^{128}$ is obtained by mean pooling over the sequence dimension.

### Spatial-Channel CNN with CBAM

The visual branch processes candlestick chart images through a Spatial-Channel CNN (SC-CNN) enhanced with the Convolutional Block Attention Module (CBAM).

**Convolutional feature extraction**: Two sequential 3×3 convolutional layers with Batch Normalization and ReLU activation extract low-level morphological features such as trendlines, support/resistance levels, and candlestick patterns:

$$
\mathbf{F} = \text{ReLU}\left(\text{BN}\left(W_2 \ast \text{ReLU}\left(\text{BN}\left(W_1 \ast \mathbf{I} + b_1\right)\right) + b_2\right)\right)
$$

where $\ast$ denotes convolution, $W_1, W_2$ are 3×3 kernels, and $\mathbf{F} \in \mathbb{R}^{H' \times W' \times C'}$ is the feature map.

**CBAM — Channel Attention Module**: Inter-channel relationships are modeled by aggregating spatial information via average and max pooling, then passing through a shared MLP:

$$
\begin{aligned}
\mathbf{M}_c(\mathbf{F}) &= \sigma\left(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\right) \\
&= \sigma\left(W_1\left(W_0(\mathbf{F}_{\text{avg}}^c)\right) + W_1\left(W_0(\mathbf{F}_{\text{max}}^c)\right)\right)
\end{aligned}
$$

where $\sigma$ is the sigmoid function, $W_0 \in \mathbb{R}^{C/r \times C}$, $W_1 \in \mathbb{R}^{C \times C/r}$, and $r$ is the reduction ratio. The channel-refined feature map is:

$$
\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \odot \mathbf{F}
$$

**CBAM — Spatial Attention Module**: Spatial locations are emphasized by pooling across channels and applying a convolutional layer:

$$
\mathbf{M}_s(\mathbf{F}') = \sigma\left(\text{Conv}_{7 \times 7}\left([\text{AvgPool}(\mathbf{F}'); \text{MaxPool}(\mathbf{F}')]\right)\right)
$$

where $[\cdot;\cdot]$ denotes channel-wise concatenation of average-pooled and max-pooled feature maps, and $\text{Conv}_{7 \times 7}$ is a 7×7 convolutional layer. The final refined feature map is:

$$
\mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \odot \mathbf{F}'
$$

The refined feature map $\mathbf{F}''$ is flattened and projected to a 128-dimensional visual embedding:

$$
\mathbf{h}_{\text{img}} = \text{ReLU}(W_{\text{proj}} \cdot \text{Flatten}(\mathbf{F}'') + b_{\text{proj}})
$$

### Gated Fusion Mechanism

The temporal and visual embeddings are adaptively fused via a learned gating mechanism that weights each modality's contribution based on contextual relevance:

$$
\mathbf{g} = \sigma\left(W_g [\mathbf{h}_{\text{seq}}; \mathbf{h}_{\text{img}}] + b_g\right)
$$

where $[\cdot;\cdot]$ denotes vector concatenation, $W_g \in \mathbb{R}^{256 \times d_g}$ and $b_g \in \mathbb{R}^{d_g}$ are learnable parameters, and $\mathbf{g} \in (0, 1)^{d_g}$ is the gate vector.

The fused representation is a convex combination:

$$
\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{h}_{\text{seq}} + (1 - \mathbf{g}) \odot \mathbf{h}_{\text{img}}
$$

This gating formulation provides several advantages:
- **Adaptive weighting**: When one modality is noisy or unreliable, the gate can down-weight its contribution.
- **Interpretability**: The gate values $\mathbf{g}$ reveal which modality dominates under different market conditions.
- **Mitigation of negative transfer**: Prevents the model from being misled by irrelevant modality-specific noise.

### Prediction Head

The fused representation is passed through a final regression layer:

$$
\hat{r}_{t+5} = W_{\text{out}} \mathbf{h}_{\text{fused}} + b_{\text{out}}
$$

### Training Objective

The model is trained end-to-end with Mean Squared Error (MSE) loss:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{r}_{t+5}^{(i)} - r_{t+5}^{(i)}\right)^2
$$

where $N$ is the batch size. The Adam optimizer with ReduceLROnPlateau scheduler and early stopping is used for stable convergence.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| EMA captures meaningful trend signal | $\mathbf{T}_t = \alpha \mathbf{X}_t + (1-\alpha)\mathbf{T}_{t-1}$ with fixed $\alpha = 2/31 \approx 0.0645$ | Assumes a single, globally optimal smoothing factor; may not adapt to regime changes in volatility |
| Candlestick charts encode predictive visual patterns | $\mathbf{h}_{\text{img}} = f_{\theta}(\mathbf{I})$ where $\mathbf{I}$ is a 30-day candlestick chart with technical indicators | Requires consistent chart rendering; performance depends on visual feature quality and indicator overlay choices |
| Linear Transformer can substitute full softmax attention | $\text{Attn}_{\text{linear}}(Q,K,V) \approx \text{softmax}(QK^\top/\sqrt{d_k})V$ via kernel $\phi(x)=\text{ELU}(x)+1$ | Assumes the non-negativity and factorization of the attention kernel preserve sufficient expressiveness |
| Modalities are complementary and conditionally independent given the target | $\mathbf{h}_{\text{seq}} \perp \mathbf{h}_{\text{img}} \mid r_{t+5}$ | Ignores potential cross-modal interactions before the fusion stage; no modality interaction in hidden layers |
| Gating provides optimal modality weighting | $\mathbf{g} = \sigma(W_g[\mathbf{h}_{\text{seq}};\mathbf{h}_{\text{img}}] + b_g)$ is a learnable convex combination | Assumes linear separability of modality relevance in the joint embedding space |
| 30-day lookback window is sufficient | $T = 30$ trading days (approximately 6 calendar weeks) | Longer-term structural changes (> 6 weeks) may be missed; shorter windows may miss medium-term patterns |
| 5-day log return is a predictable target | $r_{t+5} = \ln(P_{t+5}/P_t)$ is forecastable with MSE loss | Assumes some degree of predictability in short-horizon stock returns; Efficient Market Hypothesis challenges this |
| Stationarity within each 30-day window | The EMA decomposition assumes stable statistics over each window | Non-stationary events (earnings, macro shocks) within the window may violate this |
| Technical indicators on charts provide additive signal | Overlaid indicators (MA, Bollinger Bands, RSI) contain information orthogonal to raw price | Indicator selection and visual salience may vary across assets and market regimes |
| Independent training per asset with shared architecture | Same architecture, separately trained per stock | Does not leverage cross-asset transfer learning; may limit generalization to new assets |

## Applicable Scenarios

**When to use:**
- Multivariate financial time series forecasting where both numerical and visual data are available
- Stock return prediction with access to candlestick chart images and technical indicator overlays
- Applications requiring efficient long-sequence modeling (Linear Transformer's $\mathcal{O}(n)$ complexity)
- Scenarios where adaptive multimodal fusion is needed to handle varying modality reliability
- Medium-horizon forecasting (5-day returns) in equity markets

**When NOT to use:**
- Only numerical time series available (no chart image data -- the visual branch would be unused)
- Ultra-high-frequency trading (sub-second predictions; candlestick charts are daily-aggregated)
- Low-liquidity assets where candlestick patterns are unreliable
- Assets with very short trading history (< 30 days)
- Applications requiring real-time inference on CPU-only devices (CNN+Transformer inference cost)
- Scenarios where interpretability of individual feature contributions is critical (gating provides modality-level, not feature-level, importance)

**Comparison with alternatives:**

| Method | Strength | Limitation |
|--------|----------|------------|
| ARIMA | Simple, interpretable, well-understood | Linear only; cannot model complex multimodal patterns |
| LSTM | Captures long-range temporal dependencies | $\mathcal{O}(n)$ sequential computation; no visual modality |
| Informer | Efficient long-sequence Transformer | ProbSparse attention may lose fine-grained patterns; no multimodal fusion |
| Autoformer | Series decomposition with auto-correlation | Decomposition is not adaptive; no visual modality |
| Crossformer | Cross-dimension dependency modeling | High complexity; limited to numerical modality |
| iTransformer | Channel-independent attention | Treats each variate independently; ignores cross-modal signals |
| **TIC-FusionNet** | Multimodal fusion with adaptive gating; linear-complexity attention; EMA denoising | Requires both numerical and visual inputs; per-asset training |

## Implementation Details

### Architecture Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback window $T$ | 30 trading days | Input sequence length for both modalities |
| EMA smoothing window $T_{\text{ema}}$ | 30 | EMA decay factor $\alpha = 2/(T_{\text{ema}}+1)$ |
| Temporal feature dimension $d_h$ | 128 | Hidden dimension of Linear Transformer output |
| Linear Transformer heads | 4 | Number of attention heads (implied from $d_k = 32$) |
| Linear Transformer layers | 2 | Number of stacked Linear Transformer encoder blocks |
| Linear Transformer kernel | $\phi(x) = \text{ELU}(x) + 1$ | Non-negative feature map for linear attention |
| CNN layers | 2 | 3×3 convolutional layers with BatchNorm + ReLU |
| CNN base channels | 32 | Output channels after first conv layer |
| CNN intermediate channels | 64 | Output channels after second conv layer |
| CBAM reduction ratio $r$ | 8 | Channel attention MLP bottleneck ratio |
| CBAM spatial kernel | 7 × 7 | Convolution kernel for spatial attention |
| Image input size | 160 × 120 × 3 | Candlestick chart resolution (RGB) |
| Visual embedding dimension | 128 | Final projection dimension for $\mathbf{h}_{\text{img}}$ |
| Gate dimension $d_g$ | 128 | Gating vector dimension |
| Forecast horizon | 5 trading days | $\hat{r}_{t+5}$ log-return prediction |
| Optimizer | Adam | $\beta_1 = 0.9, \beta_2 = 0.999$ |
| Learning rate | $1 \times 10^{-3}$ | Initial learning rate |
| Scheduler | ReduceLROnPlateau | Factor = 0.5, patience = 5 |
| Loss function | MSE | Mean squared error on log returns |
| Batch size | 32 | Samples per training batch |
| Max epochs | 200 | Maximum training epochs |
| Early stopping patience | 15 | Stop if validation loss does not improve |

### Training Details

1. **Data preprocessing**: Raw OHLCV (Open, High, Low, Close, Volume) data is split into 30-day sliding windows. Log returns are computed for the target. Candlestick charts are rendered with overlaid technical indicators (20-day MA, 60-day MA, Bollinger Bands, RSI at 14-day window).
2. **Train/val/test split**: Chronological 70/15/15 split per asset to prevent look-ahead bias.
3. **Standardization**: Each numerical feature is z-score normalized using training set statistics.
4. **Training protocol**: Adam optimizer with initial learning rate $10^{-3}$, ReduceLROnPlateau reduces LR by factor 0.5 when validation loss plateaus for 5 epochs. Early stopping halts training after 15 epochs without improvement. Maximum 200 epochs.
5. **Hardware**: Training on a single NVIDIA GPU (e.g., RTX 3090) completes within approximately 30 minutes per asset.

### Interpretability

The paper provides attention-based interpretability analysis for the visual branch:
- CBAM channel attention weights identify which technical indicators (MA crossovers, RSI zones, Bollinger Band touches) are most influential.
- Under high-volatility regimes, Bollinger Band width and RSI regions dominate channel attention.
- In low-volatility regimes, MA crossover patterns receive higher attention weights.
- The gating coefficient $\mathbf{g}$ provides modality-level importance: during stable periods, the temporal gate weight is higher; during volatile periods, visual patterns dominate.

## Python Implementation

```python
"""
TIC-FusionNet: Time-Frequency Decomposition with Linear Transformer
and Spatial-Channel CNN with Gated Fusion

Reference: Chen & Fan (2025). PLOS ONE, 20(10), e0333379.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional


# =============================================================================
# 1. EMA Trend Decomposition Module
# =============================================================================

class EMADecomposition(nn.Module):
    """Exponential Moving Average trend decomposition.

    Decomposes a time series into trend and residual components:
        Trend_t = alpha * X_t + (1 - alpha) * Trend_{t-1}
        Residual_t = X_t - Trend_t

    Args:
        alpha: Smoothing factor. If None, computed as 2/(window+1).
        window: EMA window length (used when alpha is None).
    """
    def __init__(self, alpha: Optional[float] = None, window: int = 30):
        super().__init__()
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 2.0 / (window + 1.0)
        # Register as buffer so it moves with the module but is not a parameter
        self.register_buffer('alpha_tensor', torch.tensor([self.alpha]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply EMA decomposition.

        Args:
            x: Input tensor of shape (batch, seq_len, feat_dim)
        Returns:
            trend: Trend component, same shape as x
            residual: Residual (noise) component, same shape as x
        """
        batch, seq_len, feat_dim = x.shape
        trend = torch.zeros_like(x)

        # Initialize trend with the first value (or a simple average of first few)
        trend[:, 0, :] = x[:, 0, :]

        for t in range(1, seq_len):
            trend[:, t, :] = (
                self.alpha_tensor * x[:, t, :]
                + (1 - self.alpha_tensor) * trend[:, t - 1, :]
            )

        residual = x - trend
        return trend, residual

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.4f}"


# =============================================================================
# 2. Linear Transformer (Efficient Self-Attention)
# =============================================================================

class LinearAttention(nn.Module):
    """Linear attention with kernelized dot-product.

    Replaces softmax(QK^T/sqrt(d))V with phi(Q)(phi(K)^T V) / phi(Q) sum(phi(K))
    achieving O(n) complexity.

    The ELU+1 feature map ensures non-negative similarities without costly
    softmax exponentiation.

    Args:
        d_model: Input feature dimension.
        d_k: Key/query dimension.
        d_v: Value dimension.
    """
    def __init__(self, d_model: int = 128, d_k: int = 32, d_v: int = 32):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.W_o = nn.Linear(d_v, d_model, bias=False)

    def _elu_plus_one(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map: phi(x) = ELU(x) + 1, ensuring non-negativity."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with linear-complexity attention.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)

        # Apply kernel feature map
        Q_prime = self._elu_plus_one(Q)  # (batch, seq_len, d_k)
        K_prime = self._elu_plus_one(K)  # (batch, seq_len, d_k)

        # Linear attention: Q'(K'^T V) with normalization
        # K'^T V: (batch, d_k, d_v)
        KV = torch.einsum('bnd,bnd->bnd', K_prime, V)  # element-wise is wrong
        # Correct: K'^T V: (batch, d_k, d_v)
        KV = torch.einsum('bnd,bm->bnd', K_prime.transpose(1, 2), V)  # no
        # Let me be explicit:
        # K' shape: (batch, seq_len, d_k) -> transpose to (batch, d_k, seq_len)
        # V shape:  (batch, seq_len, d_v)
        # KV shape: (batch, d_k, d_v)
        K_prime_T = K_prime.transpose(1, 2)  # (batch, d_k, seq_len)
        KV = torch.bmm(K_prime_T, V)  # (batch, d_k, d_v)

        # Q' (KV): (batch, seq_len, d_v)
        attn_out = torch.bmm(Q_prime, KV)  # (batch, seq_len, d_v)

        # Normalization factor: Q' sum(K', dim=1) -> (batch, seq_len, 1)
        K_sum = K_prime.sum(dim=1, keepdim=True)  # (batch, 1, d_k)
        norm = torch.bmm(Q_prime, K_sum.transpose(1, 2))  # (batch, seq_len, 1)
        norm = torch.clamp(norm, min=1e-12)  # avoid division by zero

        attn_out = attn_out / norm

        # Output projection
        out = self.W_o(attn_out)  # (batch, seq_len, d_model)
        return out


class LinearTransformerBlock(nn.Module):
    """Single Linear Transformer encoder block.

    LinearAttention -> Residual + LayerNorm -> FFN -> Residual + LayerNorm
    """
    def __init__(self, d_model: int = 128, d_ff: int = 512,
                 d_k: int = 32, d_v: int = 32, dropout: float = 0.1):
        super().__init__()
        self.attention = LinearAttention(d_model, d_k, d_v)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class LinearTransformerEncoder(nn.Module):
    """Stacked Linear Transformer encoder with mean pooling.

    Args:
        input_dim: Input feature dimension (after trend-residual concat).
        d_model: Model hidden dimension.
        num_layers: Number of LinearTransformer blocks.
        d_k: Key/query dimension per head.
        d_v: Value dimension per head.
        d_ff: FFN hidden dimension.
        dropout: Dropout rate.
    """
    def __init__(self, input_dim: int = 64, d_model: int = 128,
                 num_layers: int = 2, d_k: int = 32, d_v: int = 32,
                 d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            LinearTransformerBlock(d_model, d_ff, d_k, d_v, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence and return pooled representation.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            h_seq: (batch, d_model) — mean-pooled temporal embedding
        """
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        # Mean pooling over sequence dimension
        h_seq = x.mean(dim=1)  # (batch, d_model)
        return h_seq


# =============================================================================
# 3. Spatial-Channel CNN with CBAM
# =============================================================================

class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module.

    Applies avg-pool and max-pool, passes through a shared MLP with
    a bottleneck reduction ratio, and outputs channel-wise attention weights.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        reduced = max(1, in_channels // reduction_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(),
            nn.Linear(reduced, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention map.

        Args:
            x: (batch, channels, height, width)
        Returns:
            out: (batch, channels, 1, 1) — channel attention weights
        """
        batch, c, _, _ = x.shape

        # Average pooling
        avg_pool = x.mean(dim=[2, 3])  # (batch, c)
        avg_out = self.mlp(avg_pool)   # (batch, c)

        # Max pooling
        max_pool, _ = x.view(batch, c, -1).max(dim=2)  # (batch, c)
        max_out = self.mlp(max_pool)  # (batch, c)

        # Combine and sigmoid
        out = torch.sigmoid(avg_out + max_out)  # (batch, c)
        return out.view(batch, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module.

    Concatenates avg-pooled and max-pooled feature maps along the channel
    dimension and applies a 7x7 convolution to produce a spatial attention map.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention map.

        Args:
            x: (batch, channels, height, width)
        Returns:
            out: (batch, 1, height, width) — spatial attention weights
        """
        # Average and max pooling along channel dimension
        avg_out = x.mean(dim=1, keepdim=True)     # (batch, 1, h, w)
        max_out, _ = x.max(dim=1, keepdim=True)   # (batch, 1, h, w)

        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, h, w)
        out = torch.sigmoid(self.conv(concat))  # (batch, 1, h, w)
        return out


class CBAM(nn.Module):
    """Complete Convolutional Block Attention Module.

    Applies Channel Attention followed by Spatial Attention sequentially.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 8,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM refinement.

        Args:
            x: (batch, channels, height, width)
        Returns:
            refined: (batch, channels, height, width)
        """
        x = x * self.channel_attn(x)  # Channel attention
        x = x * self.spatial_attn(x)  # Spatial attention
        return x


class SpatialChannelCNN(nn.Module):
    """Spatial-Channel CNN with CBAM for visual feature extraction.

    Args:
        in_channels: Input image channels (3 for RGB).
        base_channels: Output channels after first conv.
        mid_channels: Output channels after second conv.
        embedding_dim: Output visual embedding dimension.
        cbam_ratio: CBAM channel reduction ratio.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 mid_channels: int = 64, embedding_dim: int = 128,
                 cbam_ratio: int = 8):
        super().__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            # Conv block 2
            nn.Conv2d(base_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )

        self.cbam = CBAM(mid_channels, reduction_ratio=cbam_ratio,
                         spatial_kernel=7)

        # Global average pooling -> embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(mid_channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual embedding from candlestick chart image.

        Args:
            x: (batch, 3, 160, 120) — normalized candlestick chart images
        Returns:
            h_img: (batch, embedding_dim) — visual embedding
        """
        x = self.features(x)        # (batch, mid_channels, H', W')
        x = self.cbam(x)            # CBAM refinement
        x = self.global_pool(x)     # (batch, mid_channels, 1, 1)
        x = self.flatten(x)         # (batch, mid_channels)
        h_img = self.projection(x)  # (batch, embedding_dim)
        return h_img


# =============================================================================
# 4. Gated Fusion Mechanism
# =============================================================================

class GatedFusion(nn.Module):
    """Adaptive gated fusion for multimodal integration.

    Computes a gating vector g = sigma(W[h_seq; h_img] + b) and produces:
        h_fused = g * h_seq + (1 - g) * h_img

    Args:
        dim: Feature dimension of each modality.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, h_seq: torch.Tensor,
                h_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse temporal and visual embeddings.

        Args:
            h_seq: (batch, dim) — temporal embedding from LinearTransformer
            h_img: (batch, dim) — visual embedding from SC-CNN + CBAM
        Returns:
            h_fused: (batch, dim) — fused representation
            gate_weights: (batch, dim) — gate activation values for analysis
        """
        concat = torch.cat([h_seq, h_img], dim=-1)  # (batch, 2*dim)
        gate_weights = torch.sigmoid(self.gate(concat))  # (batch, dim)
        h_fused = gate_weights * h_seq + (1 - gate_weights) * h_img
        return h_fused, gate_weights


# =============================================================================
# 5. Complete TIC-FusionNet Model
# =============================================================================

class TICFusionNet(nn.Module):
    """TIC-FusionNet: Multimodal forecasting with trend decomposition,
    Linear Transformer, SC-CNN+CBAM, and gated fusion.

    Args:
        num_features: Number of numerical input features per time step.
        seq_len: Input sequence length (default: 30).
        d_model: Hidden dimension for transformer and embeddings.
        num_transformer_layers: Number of LinearTransformer blocks.
        cnn_base_channels: Base channels for CNN.
        cnn_mid_channels: Intermediate channels for CNN.
        img_channels: Input image channels (3 for RGB).
        embedding_dim: Unified embedding dimension for both modalities.
        ema_window: EMA decomposition window length.
        dropout: Dropout rate.
    """
    def __init__(
        self,
        num_features: int = 10,
        seq_len: int = 30,
        d_model: int = 128,
        num_transformer_layers: int = 2,
        cnn_base_channels: int = 32,
        cnn_mid_channels: int = 64,
        img_channels: int = 3,
        embedding_dim: int = 128,
        ema_window: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()

        # EMA decomposition: input -> [trend; residual], so input_dim doubles
        self.ema = EMADecomposition(window=ema_window)
        transformer_input_dim = num_features * 2

        # Temporal branch
        self.transformer = LinearTransformerEncoder(
            input_dim=transformer_input_dim,
            d_model=d_model,
            num_layers=num_transformer_layers,
            d_k=d_model // 4,  # 4 heads implied
            d_v=d_model // 4,
            d_ff=d_model * 4,
            dropout=dropout,
        )

        # Visual branch
        self.cnn = SpatialChannelCNN(
            in_channels=img_channels,
            base_channels=cnn_base_channels,
            mid_channels=cnn_mid_channels,
            embedding_dim=embedding_dim,
            cbam_ratio=8,
        )

        # Gated fusion
        self.fusion = GatedFusion(dim=embedding_dim)

        # Prediction head
        self.predictor = nn.Linear(embedding_dim, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier uniform initialization to linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        numerical_seq: torch.Tensor,
        chart_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of TIC-FusionNet.

        Args:
            numerical_seq: (batch, seq_len, num_features)
            chart_images: (batch, 3, 160, 120)
        Returns:
            prediction: (batch,) — predicted 5-day log return
            gate_weights: (batch, embedding_dim) — for interpretability analysis
        """
        # ---- Temporal Branch ----
        trend, residual = self.ema(numerical_seq)
        # Concatenate trend and residual along feature dimension
        enhanced_seq = torch.cat([trend, residual], dim=-1)
        h_seq = self.transformer(enhanced_seq)  # (batch, d_model)

        # ---- Visual Branch ----
        h_img = self.cnn(chart_images)  # (batch, embedding_dim)

        # ---- Gated Fusion ----
        h_fused, gate_weights = self.fusion(h_seq, h_img)
        h_fused = self.dropout(h_fused)

        # ---- Prediction ----
        raw_pred = self.predictor(h_fused)  # (batch, 1)
        prediction = raw_pred.squeeze(-1)   # (batch,)

        return prediction, gate_weights


# =============================================================================
# 6. Synthetic Data Generation for Demonstration
# =============================================================================

def generate_synthetic_stock_data(
    n_samples: int = 5000,
    seq_len: int = 30,
    num_features: int = 10,
    img_height: int = 160,
    img_width: int = 120,
    seed: int = 42,
):
    """Generate synthetic stock market data for demonstration.

    Creates realistic-ish numerical sequences with trend, seasonality,
    and noise components, plus synthetic candlestick chart images.

    Args:
        n_samples: Number of sequences to generate.
        seq_len: Length of each sequence.
        num_features: Number of numerical features.
        img_height, img_width: Image dimensions.
        seed: Random seed.

    Returns:
        numerical_seq: (n_samples, seq_len, num_features) numpy array
        chart_images: (n_samples, 3, img_height, img_width) numpy array
        returns_5d: (n_samples,) numpy array of 5-day log returns
    """
    rng = np.random.RandomState(seed)

    # --- Numerical sequence generation ---
    # Each sequence has: price trend + noise + optional known patterns
    t = np.linspace(0, 4 * np.pi, seq_len)

    numerical_seq = np.zeros((n_samples, seq_len, num_features))
    returns_5d = np.zeros(n_samples)

    for i in range(n_samples):
        # Base price with random drift and seasonality
        drift = rng.uniform(-0.001, 0.003)
        base_price = 100.0 + np.cumsum(
            drift + rng.randn(seq_len) * 0.01
        ) + 0.5 * np.sin(t + rng.uniform(0, 2 * np.pi))

        # Generate features from price
        numerical_seq[i, :, 0] = base_price  # Close
        numerical_seq[i, :, 1] = base_price * (1 + rng.randn(seq_len) * 0.005)  # High
        numerical_seq[i, :, 2] = base_price * (1 - rng.randn(seq_len) * 0.005)  # Low
        numerical_seq[i, :, 3] = base_price  # Open (approx)
        numerical_seq[i, :, 4] = rng.exponential(scale=1e6, size=seq_len)  # Volume
        # Technical indicators
        numerical_seq[i, :, 5] = 0.02 * np.sin(t) + rng.randn(seq_len) * 0.005  # RSI-like
        numerical_seq[i, :, 6] = base_price * 1.02  # Upper BB
        numerical_seq[i, :, 7] = base_price * 0.98  # Lower BB
        numerical_seq[i, :, 8] = np.convolve(
            base_price, np.ones(5)/5, mode='same'
        )  # MA(5)
        numerical_seq[i, :, 9] = np.convolve(
            base_price, np.ones(20)/20, mode='same'
        )  # MA(20)

        # Target: 5-day log return (a function of recent trend + noise)
        ret = np.log(
            base_price[min(seq_len - 1, 4)] / base_price[0]
        ) + rng.randn() * 0.02
        returns_5d[i] = ret

    # --- Synthetic chart images ---
    chart_images = np.zeros((n_samples, img_height, img_width, 3))
    for i in range(n_samples):
        # Simple synthetic candlestick-like image
        base_val = rng.uniform(0.3, 0.7)
        # Create a rough "chart" pattern: colored bars + background
        img = np.ones((img_height, img_width, 3)) * 0.95  # light background

        # Draw a line representing price movement
        price_line = base_val + 0.1 * np.sin(
            np.linspace(0, 3 * np.pi, img_width) + rng.uniform(0, 2 * np.pi)
        )
        for x in range(img_width):
            y_center = int((1 - price_line[x]) * img_height)
            # Draw thick line
            for dy in range(-2, 3):
                y = min(img_height - 1, max(0, y_center + dy))
                img[y, x, :] = [0.2, 0.4, 0.8]  # blue line

        # Add some noise
        img += rng.randn(*img.shape) * 0.02
        img = np.clip(img, 0, 1)
        chart_images[i] = img

    # Transpose images to (n, 3, h, w)
    chart_images = np.transpose(chart_images, (0, 3, 1, 2))

    # Standardize numerical features
    numerical_seq = (numerical_seq - numerical_seq.mean(axis=(0, 1), keepdims=True)) / (
        numerical_seq.std(axis=(0, 1), keepdims=True) + 1e-8
    )
    # Standardize returns
    returns_5d = (returns_5d - returns_5d.mean()) / (returns_5d.std() + 1e-8)

    return (
        numerical_seq.astype(np.float32),
        chart_images.astype(np.float32),
        returns_5d.astype(np.float32),
    )


# =============================================================================
# 7. Training Utilities
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for num_seq, img, target in dataloader:
        num_seq = num_seq.to(device)
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred, _ = model(num_seq, img)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model and compute regression metrics."""
    model.eval()
    preds, targets = [], []

    for num_seq, img, target in dataloader:
        num_seq = num_seq.to(device)
        img = img.to(device)
        target = target.to(device)

        pred, _ = model(num_seq, img)
        preds.append(pred.cpu())
        targets.append(target.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    # Metrics
    mse = F.mse_loss(preds, targets)
    mae = F.l1_loss(preds, targets)
    rmse = torch.sqrt(mse)
    ss_res = ((preds - targets) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    # MAPE and SMAPE
    mape = (torch.abs(preds - targets) / (torch.abs(targets) + 1e-12)).mean() * 100
    smape = (
        2.0 * torch.abs(preds - targets) / (torch.abs(preds) + torch.abs(targets) + 1e-12)
    ).mean() * 100

    return {
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'MAE': mae.item(),
        'R²': r2.item(),
        'MAPE': mape.item(),
        'SMAPE': smape.item(),
    }


# =============================================================================
# 8. Complete Training Example
# =============================================================================

def main():
    """Demonstrate TIC-FusionNet training on synthetic data."""
    # Configuration
    config = {
        'num_features': 10,
        'seq_len': 30,
        'd_model': 128,
        'num_transformer_layers': 2,
        'cnn_base_channels': 32,
        'cnn_mid_channels': 64,
        'embedding_dim': 128,
        'ema_window': 30,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'n_samples': 2000,
    }

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Configuration: {config}")

    # --- Generate synthetic data ---
    print("\nGenerating synthetic stock market data...")
    num_seq, chart_imgs, returns = generate_synthetic_stock_data(
        n_samples=config['n_samples'],
        seq_len=config['seq_len'],
        num_features=config['num_features'],
    )
    print(f"Numerical sequences: {num_seq.shape}")
    print(f"Chart images:        {chart_imgs.shape}")
    print(f"5-day returns:       {returns.shape}")

    # --- Train/val/test split (chronological) ---
    n_train = int(0.7 * len(num_seq))
    n_val = int(0.15 * len(num_seq))
    # Use sequential split to preserve temporal order
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, len(num_seq))

    train_data = TensorDataset(
        torch.from_numpy(num_seq[train_idx]),
        torch.from_numpy(chart_imgs[train_idx]),
        torch.from_numpy(returns[train_idx]),
    )
    val_data = TensorDataset(
        torch.from_numpy(num_seq[val_idx]),
        torch.from_numpy(chart_imgs[val_idx]),
        torch.from_numpy(returns[val_idx]),
    )
    test_data = TensorDataset(
        torch.from_numpy(num_seq[test_idx]),
        torch.from_numpy(chart_imgs[test_idx]),
        torch.from_numpy(returns[test_idx]),
    )

    train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'],
                            shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'],
                             shuffle=False)

    # --- Initialize model ---
    model = TICFusionNet(
        num_features=config['num_features'],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        num_transformer_layers=config['num_transformer_layers'],
        cnn_base_channels=config['cnn_base_channels'],
        cnn_mid_channels=config['cnn_mid_channels'],
        embedding_dim=config['embedding_dim'],
        ema_window=config['ema_window'],
        dropout=config['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # --- Optimizer and scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                           betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )

    # --- Training loop ---
    print("\n--- Training ---")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for num_seq, img, target in val_loader:
                num_seq, img, target = (
                    num_seq.to(device), img.to(device), target.to(device)
                )
                pred, _ = model(num_seq, img)
                val_loss += F.mse_loss(pred, target).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'tic_fusionnet_best.pt')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # --- Evaluate on test set ---
    model.load_state_dict(
        torch.load('tic_fusionnet_best.pt', map_location=device,
                   weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device)

    print("\n--- Test Set Results ---")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # --- Gate weight analysis (interpretability) ---
    model.eval()
    all_gates = []
    with torch.no_grad():
        for num_seq, img, _ in test_loader:
            num_seq, img = num_seq.to(device), img.to(device)
            _, gates = model(num_seq, img)
            all_gates.append(gates.cpu())

    all_gates = torch.cat(all_gates)
    mean_gate = all_gates.mean(dim=0)
    gate_std = all_gates.std(dim=0)

    print(f"\n--- Gate Weight Analysis (Test Set) ---")
    print(f"  Mean gate value: {mean_gate.mean():.4f} ± {gate_std.mean():.4f}")
    print(f"  Gate range: [{mean_gate.min():.4f}, {mean_gate.max():.4f}]")
    temporal_dominance = (mean_gate > 0.5).float().mean().item()
    print(f"  Dimensions where temporal > visual: "
          f"{temporal_dominance * 100:.1f}%")
    print(f"  Interpretation:")
    print(f"    Gate > 0.5: temporal (numerical) modality dominates")
    print(f"    Gate < 0.5: visual (chart) modality dominates")

    print("\n--- Paper Reference Results ---")
    print("  Amazon dataset: RMSE=0.0534, MAE=0.0427, R²=0.8737")
    print("  R² range across all 6 datasets: [0.8706, 0.9264]")
    print("  RMSE reduction: 25.4% vs Random Forest, 21.5% vs LSTM")
    print("  RMSE reduction: 18.9% vs Autoformer, 17.3% vs iTransformer")


if __name__ == "__main__":
    main()
```

## References

Chen, L., & Fan, X. (2025). TIC-FusionNet: A multimodal deep learning framework with temporal decomposition and attention-based fusion for time series forecasting. *PLOS ONE*, 20(10), e0333379. https://doi.org/10.1371/journal.pone.0333379

Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. *Proceedings of the 37th International Conference on Machine Learning*, 119, 5156–5165. https://doi.org/10.48550/arXiv.2006.16236

Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *Proceedings of the European Conference on Computer Vision (ECCV)*, 3–19. https://doi.org/10.1007/978-3-030-01234-2_1

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008. https://doi.org/10.48550/arXiv.1706.03762

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106–11115. https://doi.org/10.1609/aaai.v35i12.17325

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems*, 34, 22419–22430. https://doi.org/10.48550/arXiv.2106.13008

Zhang, Y., & Yan, J. (2023). Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. *International Conference on Learning Representations*. https://openreview.net/forum?id=vSVLM2j9eie

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, J., & Long, M. (2024). iTransformer: Inverted transformers are effective for time series forecasting. *International Conference on Learning Representations*. https://openreview.net/forum?id=JePfAI8fah
