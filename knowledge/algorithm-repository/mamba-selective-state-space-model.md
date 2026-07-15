# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Source**: Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*. https://arxiv.org/abs/2312.00752

**Category**: Machine Learning / Sequence Modeling / State Space Models

## Mathematical Setup

### State Space Model Foundation

A continuous-time linear time-invariant (LTI) state space model (SSM) maps a 1-dimensional input signal $u(t) \in \mathbb{R}$ to an output $y(t) \in \mathbb{R}$ through a hidden state $h(t) \in \mathbb{R}^N$:

$$
\begin{aligned}
h'(t) &= \mathbf{A} h(t) + \mathbf{B} u(t) \\
y(t) &= \mathbf{C} h(t) + \mathbf{D} u(t)
\end{aligned}
$$

where $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix, $\mathbf{B} \in \mathbb{R}^{N \times 1}$ and $\mathbf{C} \in \mathbb{R}^{1 \times N}$ are projection parameters, and $\mathbf{D}$ is a skip connection.

### Discretization via Zero-Order Hold

For discrete sequence processing, the continuous parameters are discretized with a timescale $\Delta$:

$$
\begin{aligned}
\overline{\mathbf{A}} &= \exp(\Delta \mathbf{A}) \\
\overline{\mathbf{B}} &= (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B} \approx \Delta \mathbf{B} \quad \text{(first-order Taylor)}
\end{aligned}
$$

The recurrent inference is then:

$$
\begin{aligned}
h_t &= \overline{\mathbf{A}} h_{t-1} + \overline{\mathbf{B}} u_t \\
y_t &= \mathbf{C} h_t
\end{aligned}
$$

### Selective State Spaces (S6) -- Key Innovation

Prior SSMs (S4, S4D) used **input-independent** parameters. Mamba makes $\Delta$, $\mathbf{B}$, and $\mathbf{C}$ functions of the input $x \in \mathbb{R}^{B \times L \times D}$ at each timestep:

$$
\begin{aligned}
\mathbf{B} &= \text{Linear}_B(x_t) \in \mathbb{R}^{N} \\
\mathbf{C} &= \text{Linear}_C(x_t) \in \mathbb{R}^{N} \\
\Delta_t &= \text{softplus}(\text{Linear}_\Delta(x_t) + \text{bias}) \in \mathbb{R}
\end{aligned}
$$

This enables **content-aware filtering**: the model learns what to remember and what to forget based on the input.

### Mamba Block Architecture

The simplified SSM-only block (expansion factor $E=2$) does not require separate attention or MLP sub-layers:

$$
\begin{aligned}
z &= \text{SiLU}(\text{Linear}_z(x)) \quad &\text{(gating branch)} \\
x' &= \text{SiLU}(\text{Linear}_x(x)) \quad &\text{(SSM input branch)} \\
y &= \text{SSM}(x') \odot z \quad &\text{(element-wise gating)} \\
\text{output} &= \text{Linear}_{\text{out}}(y)
\end{aligned}
$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| **Linear recurrence** | $h_t = \overline{\mathbf{A}} h_{t-1} + \overline{\mathbf{B}}u_t$ with fixed hidden dimension $N$ | Training in O(L) time, inference in O(1) per step; no quadratic attention |
| **Diagonal state matrix** | $\mathbf{A} = \text{diag}(a_1, \ldots, a_N)$ with $a_i < 0$ (negative real) | Stabilizes the recurrence; enables efficient parallel scan via associative property |
| **Input-dependent parameters** | $\Delta_t = f_\Delta(x_t)$, $\mathbf{B} = f_B(x_t)$, $\mathbf{C} = f_C(x_t)$ | Selective copying and induction-head capabilities absent in LTI SSMs |
| **Sufficient state size** | Hidden state dimension $N = 16$ (per channel) | Balances expressivity with computational cost |

## Applicable Scenarios

**When to use:**
- Long-context sequence modeling (DNA, audio, text with context >8K tokens)
- Real-time autoregressive generation (linear decoding time)
- Resource-constrained deployment (no KV cache needed)
- Modalities requiring selective recall (selective copy, induction heads)

**When NOT to use:**
- Short, simple sequences where Transformers are already efficient (<1K tokens)
- Tasks requiring exact cross-attention over long documents (e.g., retrieval)
- When extensive pretrained Transformer infrastructure is preferred for ecosystem compatibility

**Comparison:** Transformers scale quadratically with sequence length but offer global attention. Mamba scales linearly but uses a compressed recurrent state; it excels at very long sequences but may lose fine-grained recall of distant tokens.

## Algorithm / Method Details

### Training (Parallel Scan)

The selective SSM requires a hardware-efficient parallel scan algorithm:

1. **Discretize**: For each timestep $t$, compute $\overline{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$ and $\overline{\mathbf{B}}_t = \Delta_t \mathbf{B}_t$.
2. **Associative scan**: Compute the prefix sums $h_t = \overline{\mathbf{A}}_t h_{t-1} + \overline{\mathbf{B}}_t u_t$ using a parallel associative scan (O(log L) steps on GPU).
3. **Output projection**: Compute $y_t = \mathbf{C}_t h_t$ and apply the gating mechanism.

### Inference (Recurrent)

At inference time, the selective SSM operates as a standard RNN:

1. Maintain hidden state $h \in \mathbb{R}^{B \times D \times N}$.
2. At each step, read $x_t$, compute $\Delta_t$, $\mathbf{B}_t$, $\mathbf{C}_t$.
3. Update $h \leftarrow \overline{\mathbf{A}}_t h + \overline{\mathbf{B}}_t x_t$.
4. Output $y_t = \mathbf{C}_t h$.

### Theoretical Guarantees

- **Stability**: With $\text{Re}(\text{eig}(\mathbf{A})) < 0$, the continuous system is stable; discretization preserves stability for small $\Delta$.
- **Universal approximation**: SSMs with state dimension $N$ can approximate any linear dynamical system of order $N$.
- **Complexity**: Training O($L \cdot D \cdot N$) time, O($B \cdot L \cdot D$) memory for activations.

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| State dimension $N$ | 16 | Higher for more memory, lower for speed |
| Expansion factor $E$ | 2 | Controls hidden dimension: $d_{\text{hidden}} = E \cdot d_{\text{model}}$ |
| $\Delta$ initialization | 0.001-0.1 | Controls "selectivity bandwidth"; tune per modality |
| Activation | SiLU (Swish) | Default; GELU is a viable alternative |
| Normalization | LayerNorm (optional) | Use for deep stacks (>24 layers) |

### Numerical Considerations

- Use `float32` for the SSM recurrence to avoid numerical instability with small $\Delta$.
- The parallel scan can accumulate numerical errors over very long sequences (>100K tokens); double precision recommended for validation.
- Clamp $\Delta$ to a minimum value (e.g., 0.001) to prevent near-zero discretization steps.

## Python Implementation

```python
"""
Minimal implementation of Mamba's core SSM layer (S6).
Based on: Gu & Dao (2023) https://arxiv.org/abs/2312.00752
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class SelectiveSSM(nn.Module):
    """
    Mamba's selective state space model (S6) core layer.
    Processes a single channel/head independently with its own state.
    """
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
    ):
        """
        Args:
            d_model: Model dimension (input/output features)
            d_state: SSM state dimension (hidden size)
            dt_min: Minimum initial step size delta
            dt_max: Maximum initial step size delta
            dt_init: Initialization strategy for delta
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Learnable state transition matrix (diagonal)
        # Parameterized in log space for numerical stability
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        # Delta projection (input -> step size)
        self.dt_proj = nn.Linear(d_model, 1)
        # Init dt to be in [dt_min, dt_max]
        if dt_init == "random":
            dt = torch.empty(d_model, 1).uniform_(dt_min, dt_max)
        else:
            dt = torch.ones(d_model, 1) * dt_min
        self.dt = nn.Parameter(torch.log(dt))

        # Input-dependent B and C projections
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Compute input-dependent parameters
        delta = F.softplus(self.dt.expand(batch, seq_len, -1))  # (B, L, 1)
        B = self.B_proj(x)  # (B, L, d_state)
        C = self.C_proj(x)  # (B, L, d_state)

        # Discretize A and B
        A = -torch.exp(self.A_log)  # (d_model, d_state) -- ensure negative real
        # Discretization via ZOH approximation
        # A_bar = exp(ΔA), B_bar ≈ ΔB
        A_discrete = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_model, d_state)

        # Scan computation -- we use a simple sequential scan for clarity
        # In practice, use the associative parallel scan with cumprod for efficiency
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)  # (B, d_model, d_state)
        outputs = []

        for t in range(seq_len):
            # Equivalent to: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            # Here A_bar_t = exp(Δt * A) with shape (d_model, d_state)
            A_bar_t = A_discrete[:, t]  # (B, d_model, d_state)
            B_bar_t = delta[:, t, :, None] * B[:, t, :, None]  # (B, d_model, 1) * (B, 1, d_state) -> use broadcast
            x_t = x[:, t, :]  # (B, d_model)

            # h: (B, d_model, d_state) -> update: h = A_bar ⊙ h + B_bar ⊙ u
            # A_bar * h: element-wise (since A is diagonal in state dim)
            h = A_bar_t * h  # (B, d_model, d_state)
            h = h + B_bar_t * x_t.unsqueeze(-1)  # (B, d_model, d_state)

            # Output: y_t = C_t @ h_t + D * x_t  (inner product over state)
            y_t = (C[:, t, :].unsqueeze(1) * h).sum(dim=-1)  # (B, d_model)
            y_t = y_t + self.D * x_t  # skip connection
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_model)


class MambaBlock(nn.Module):
    """
    Full Mamba block: gated MLP + selective SSM with residual connection.
    """
    def __init__(self, d_model: int = 256, expansion_factor: int = 2, d_state: int = 16):
        super().__init__()
        d_expand = d_model * expansion_factor

        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_expand * 2, bias=False)

        self.ssm = SelectiveSSM(d_model=d_expand, d_state=d_state)

        self.proj_out = nn.Linear(d_expand, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        # Split into two branches: SSM pathway and gating pathway
        x, gate = x.chunk(2, dim=-1)

        # Apply SSM and activation
        x = self.ssm(x)
        x = x * F.silu(gate)  # element-wise gating

        x = self.proj_out(x)
        return x + residual


# ============================================================
# Complete usage example with synthetic data
# ============================================================
def test_mamba():
    """Test Mamba block on a synthetic sequence modeling task."""
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 2
    seq_len = 64
    d_model = 128
    d_state = 16

    # Create model
    model = MambaBlock(d_model=d_model, d_state=d_state)
    print(f"MambaBlock parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Synthetic data: (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    y = model(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean:  {y.mean().item():.4f}")
    print(f"Output std:   {y.std().item():.4f}")
    print("Forward pass successful!")

    # Backward pass (check gradients flow)
    loss = y.sum()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"Gradient norm after backward: {grad_norm.item():.4f}")
    print("Backward pass successful!")


def compare_with_transformer():
    """
    Demonstrate linear vs quadratic scaling by measuring
    forward time of Mamba vs simple attention at different lengths.
    """
    import time

    d_model = 256
    seq_lengths = [128, 512, 2048]

    # Simplified "transformer" for comparison: single head attention
    class SimpleAttention(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.Wq = nn.Linear(d_model, d_model)
            self.Wk = nn.Linear(d_model, d_model)
            self.Wv = nn.Linear(d_model, d_model)

        def forward(self, x):
            q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
            attn = (q @ k.transpose(-2, -1)) / (d_model ** 0.5)
            attn = F.softmax(attn, dim=-1)
            return attn @ v

    mamba = MambaBlock(d_model=d_model)
    attn = SimpleAttention(d_model=d_model)

    print("\n===== Scaling Comparison =====")
    print(f"{'Seq Len':<12} {'Mamba (ms)':<16} {'Attention (ms)':<16} {'Ratio':<12}")
    print("-" * 56)

    for L in seq_lengths:
        x = torch.randn(1, L, d_model)

        start = time.perf_counter()
        with torch.no_grad():
            _ = mamba(x)
        mamba_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        with torch.no_grad():
            _ = attn(x)
        attn_time = (time.perf_counter() - start) * 1000

        ratio = attn_time / max(mamba_time, 1e-8)
        print(f"{L:<12} {mamba_time:<16.2f} {attn_time:<16.2f} {ratio:<12.2f}")


if __name__ == "__main__":
    test_mamba()
    compare_with_transformer()
```

## References

Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*. https://arxiv.org/abs/2312.00752

Gu, A., Goel, K., & Re, C. (2022). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2111.00396

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems (NeurIPS), 35*. https://arxiv.org/abs/2205.14135

Smith, J. T. H., Warrington, A., & Linderman, S. (2023). Simplified state space layers for sequence modeling. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2208.04933
