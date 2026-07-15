# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Source**: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems (NeurIPS), 35*, 16344-16359. https://arxiv.org/abs/2205.14135

**Category**: Machine Learning / Efficient Deep Learning / Attention Mechanisms

## Mathematical Setup

### Standard Attention

Given query $\mathbf{Q} \in \mathbb{R}^{L \times d}$, key $\mathbf{K} \in \mathbb{R}^{L \times d}$, and value $\mathbf{V} \in \mathbb{R}^{L \times d}$ matrices where $L$ is sequence length and $d$ is head dimension, attention computes:

$$
\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{L \times L}, \quad
\mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{L \times L}, \quad
\mathbf{O} = \mathbf{P} \mathbf{V} \in \mathbb{R}^{L \times d}
$$

The standard implementation materializes the full $L \times L$ attention matrix $\mathbf{P}$ in GPU high-bandwidth memory (HBM), leading to O($L^2$) memory and a memory-bound computation bottleneck.

### IO-Awareness: GPU Memory Hierarchy

Modern GPUs have a hierarchical memory structure:

| Level | Size | Bandwidth | Example (A100-80GB) |
|-------|------|-----------|--------------------|
| HBM (global) | 40-80 GB | 2 TB/s | Main GPU memory |
| SRAM (shared) | 192 KB per SM | ~19 TB/s | On-chip per streaming multiprocessor |

Standard attention reads/writes the full softmax matrix from HBM, bottlenecked by the lower-bandwidth HBM. FlashAttention exploits the **much faster** on-chip SRAM through tiling.

### Tiled Softmax Decomposition

The core mathematical insight is that softmax can be computed incrementally. For two sub-blocks of row vectors $\mathbf{m}^{(1)}, \mathbf{m}^{(2)} \in \mathbb{R}^{B_r}$ of the pre-softmax logits $\mathbf{s}$:

$$
m^{(1)} = \max(\mathbf{m}^{(1)}), \quad m^{(2)} = \max(\mathbf{m}^{(2)})
$$
$$
m = \max(m^{(1)}, m^{(2)})
$$
$$
\ell = \ell^{(1)} e^{m^{(1)} - m} + \ell^{(2)} e^{m^{(2)} - m}
$$
$$
\text{softmax}(\mathbf{s})_j = \frac{e^{s_j - m}}{\ell}
$$

where $\ell^{(1)} = \sum e^{\mathbf{m}^{(1)} - m^{(1)}}$, $\ell^{(2)} = \sum e^{\mathbf{m}^{(2)} - m^{(2)}}$.

This **online softmax** (also known as safe softmax with incremental normalization) allows computing attention without materializing the full $L \times L$ matrix.

### Block-Sparse Extension

Block-Sparse FlashAttention applies the same tiling strategy but skips blocks where the attention score is below a threshold, achieving 2-4x further speedup. Formally, for a block mask $\mathbf{M} \in \{0, 1\}^{L/M_c \times L/M_r}$:

$$
\mathbf{O}_i = \sum_{\{j: \mathbf{M}_{i,j} = 1\}} \text{softmax}(\mathbf{Q}_i \mathbf{K}_j^\top) \mathbf{V}_j
$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| **Exact softmax via online reduction** | Softmax of a concatenated vector equals incrementally normalized sub-blocks | Tiling does not alter the result -- mathematically exact |
| **SRAM capacity** | Block sizes fit in on-chip memory: $B_r \cdot d + B_c \cdot d + B_r \cdot B_c < M_{\text{SRAM}}$ | Limits parallelism per SM; must choose block sizes adaptively |
| **Recomputation is cheaper** | Storing only statistics $m, \ell$ and recomputing intermediate values in backward pass | 1.33x more FLOPs but 5-20x less HBM access; net speedup |
| **CUDA thread block model** | Blocks mapped to thread blocks; no synchronization across blocks | Scales to arbitrary sequence length and batch size |

## Applicable Scenarios

**When to use:**
- Any Transformer training with sequence length > 512
- Long-document models (1K-64K tokens)
- High-throughput inference (batch of 1 or small batch)
- Situations where GPU memory is the bottleneck (common in practice)

**When NOT to use:**
- Very short sequences (<128) where overhead of tiling dominates
- When using CPU or non-CUDA accelerators (though now ported to ROCm, PyTorch native)
- When approximate attention with linear complexity is acceptable and sequence length exceeds GPU capacity even for FlashAttention

**Comparison:** Compared to standard attention, FlashAttention is exact (identical output) but 2-4x faster and uses 5-20x less memory. Compared to sparse/linear attention, FlashAttention has no approximation error but still has O($L^2$) compute; it simply makes the O($L^2$) computation memory-efficient.

## Algorithm / Method Details

### Forward Pass (Tiled Computation)

```
Input: Q, K, V in HBM, on-chip SRAM of size M
Block sizes: B_r = floor(M / (4d)), B_c = floor(M / (4d))

Initialize output O = zeros(L, d) in HBM
Initialize statistics: m = zeros(L), l = zeros(L) in HBM

Divide Q into blocks Q_1, ..., Q_T_r of size B_r
Divide K, V into blocks K_1, ..., K_T_c of size B_c

For each Q_i block (loaded into SRAM):
  For each K_j, V_j block (loaded into SRAM):
    S_ij = Q_i @ K_j^T          # (B_r, B_c) in SRAM
    m_ij = rowmax(S_ij)          # (B_r,) in SRAM
    P_ij = exp(S_ij - m_ij)      # (B_r, B_c) in SRAM
    l_ij = rowsum(P_ij)          # (B_r,) in SRAM
    O_i = O_i * diag(l_i^{old} / (l_i^{new}))^{-1} + (P_ij @ V_j) * diag(1/l_i^{new})
    m_i = max(m_i^{old}, m_ij)
    l_i = l_i^{old} * exp(m_i^{old} - m_i) + l_ij * exp(m_ij - m_i)

Write O_i, m_i, l_i to HBM
```

### Backward Pass (with Recomputation)

Instead of storing the full attention matrix $\mathbf{P}$ for gradient computation, FlashAttention stores only the statistics $m$ and $l$ from the forward pass, then **recomputes** the attention matrix on-chip during backward:

```
For each Q_i block:
  For each K_j, V_j block:
    Reload Q_i, K_j from HBM
    Recompute S_ij = Q_i @ K_j^T on-chip
    Recompute P_ij from S_ij using stored m_i, l_i
    Compute dQ_i += (dO_i @ V_j^T * P_ij) @ K_j + (P_ij * (dO_i @ V_j^T)) @ K_j
    Compute dK_j += (dO_i^T @ P_ij)^T @ Q_i + (P_ij^T * (Q_i @ dO_i^T)) @ Q_i
    Compute dV_j += P_ij^T @ dO_i
```

### Complexity Analysis

- **FLOPs**: Same as standard attention: O($L^2 d$) for both forward and backward.
- **HBM access**: O($L^2 d / M_{\text{SRAM}}$) vs O($L^2$) for standard attention -- the key improvement.
- **Memory**: O($L d + L$) for outputs and softmax statistics, vs O($L^2$) for standard attention.

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| Block size $B_r$ | floor($M_{\text{SRAM}} / (4d)$) | Maximize while fitting Q, K, V, S, P blocks |
| Block size $B_c$ | Same as $B_r$ | Symmetric block partition for load balancing |
| Head dimension $d$ | 64-128 | FlashAttention optimized for standard $d$ values |
| Thread block config | 128-256 threads | Standard CUDA thread block size |

### Framework/Library Support

- **PyTorch**: `torch.nn.functional.scaled_dot_product_attention()` (since 2.0) uses FlashAttention automatically when available.
- **FlashAttention library**: `pip install flash-attn` (https://github.com/HazyResearch/flash-attention).
- **Hugging Face Transformers**: Integrated internally; no user action needed.

## Python Implementation

```python
"""
Reference implementation of FlashAttention-style tiled attention.
Based on: Dao et al. (2022) https://arxiv.org/abs/2205.14135

This implementation demonstrates the IO-aware tiling algorithm in pure PyTorch.
For actual speed, use the CUDA kernel from github.com/HazyResearch/flash-attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def standard_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Standard attention (baseline for verification).
    
    Args:
        q: (batch, heads, L, d)
        k: (batch, heads, L, d)
        v: (batch, heads, L, d)
        mask: Optional attention mask (broadcastable to (..., L, L))
        scale: Scaling factor (default: 1/sqrt(d))
    
    Returns:
        output: (batch, heads, L, d)
        stats: dict with softmax statistics (m, l vectors for verification)
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
    
    # S = Q @ K^T
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        s = s + mask
    
    # P = softmax(S)
    m = s.max(dim=-1, keepdim=True).values
    p = torch.exp(s - m)
    l = p.sum(dim=-1, keepdim=True)
    p = p / l
    
    # O = P @ V
    o = torch.matmul(p, v)
    
    return o, {"m": m.squeeze(-1), "l": l.squeeze(-1), "p": p}


def flashattention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_size: int = 64,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    FlashAttention forward pass with tiled softmax.
    
    Implements the online softmax tiling algorithm.
    
    Args:
        q: (batch, heads, L, d)
        k: (batch, heads, L, d)
        v: (batch, heads, L, d)
        block_size: Tile size for K/V blocks
        scale: Scaling factor (default: 1/sqrt(d))
    
    Returns:
        output: (batch, heads, L, d)
        (m, l): Softmax statistics for backward pass
    """
    batch, heads, L, d = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    
    # Divide into blocks
    n_blocks = (L + block_size - 1) // block_size
    
    # Initialize output and statistics
    O = torch.zeros_like(q)
    m = torch.full((batch, heads, L), -float('inf'), device=q.device)
    l = torch.zeros(batch, heads, L, device=q.device)
    
    for i in range(n_blocks):
        qi_start = i * block_size
        qi_end = min((i + 1) * block_size, L)
        Qi = q[:, :, qi_start:qi_end, :]  # (B, H, B_r, d)
        
        # Initialize running statistics for this query block
        O_i = torch.zeros_like(Qi)
        m_i = torch.full((batch, heads, Qi.size(2)), -float('inf'), device=q.device)
        l_i = torch.zeros(batch, heads, Qi.size(2), device=q.device)
        
        for j in range(n_blocks):
            kj_start = j * block_size
            kj_end = min((j + 1) * block_size, L)
            Kj = k[:, :, kj_start:kj_end, :]
            Vj = v[:, :, kj_start:kj_end, :]
            
            # Compute attention scores on-chip
            S_ij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale  # (B, H, B_r, B_c)
            
            # Online softmax update
            m_ij = S_ij.max(dim=-1).values  # (B, H, B_r)
            P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))  # (B, H, B_r, B_c)
            l_ij = P_ij.sum(dim=-1)  # (B, H, B_r)
            
            # Merge: update running max and sum
            m_new = torch.max(m_i, m_ij)
            
            # Correction factor for l
            l_i = l_i * torch.exp(m_i - m_new) + l_ij * torch.exp(m_ij - m_new)
            
            # Rescale output with corrected statistics
            O_i = O_i * torch.exp(m_i - m_new).unsqueeze(-1) + \
                  torch.matmul(P_ij, Vj) * torch.exp(m_ij - m_new).unsqueeze(-1)
            
            m_i = m_new
        
        # Normalize
        O_i = O_i / l_i.unsqueeze(-1)
        
        # Write to global memory
        O[:, :, qi_start:qi_end, :] = O_i
        m[:, :, qi_start:qi_end] = m_i
        l[:, :, qi_start:qi_end] = l_i
    
    return O, (m, l)


class FlashAttentionWrapper(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention using FlashAttention.
    Works with any sequence length; uses tiling internally.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self-attention with FlashAttention.
        
        Args:
            x: (batch, L, embed_dim)
        
        Returns:
            out: (batch, L, embed_dim)
        """
        batch, L, _ = x.shape
        
        # Project and reshape
        q = self.q_proj(x).reshape(batch, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, L, self.num_heads, self.head_dim)
        
        # (batch, heads, L, d)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        
        # FlashAttention forward
        attn_out, _ = flashattention_forward(
            q, k, v, block_size=self.block_size, scale=self.scale
        )
        
        # Reshape back
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch, L, self.embed_dim)
        attn_out = self.dropout(attn_out)
        out = self.out_proj(attn_out)
        
        return out


# ============================================================
# Verification and benchmarking
# ============================================================
@torch.no_grad()
def verify_correctness():
    """
    Verify that FlashAttention produces the same output as standard attention.
    """
    print("=" * 60)
    print("Verifying FlashAttention correctness vs standard attention")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    batch, heads, L, d = 2, 4, 64, 64
    
    q = torch.randn(batch, heads, L, d)
    k = torch.randn(batch, heads, L, d)
    v = torch.randn(batch, heads, L, d)
    
    # Standard
    out_std, _ = standard_attention(q, k, v)
    
    # Flash
    out_flash, _ = flashattention_forward(q, k, v, block_size=32)
    
    # Compare
    diff = (out_std - out_flash).abs().max().item()
    print(f"Max absolute difference: {diff:.8f}")
    assert diff < 1e-5, f"Difference too large: {diff}"
    print("PASS: FlashAttention produces identical output to standard attention")
    
    return diff


@torch.no_grad()
def benchmark_attention():
    """
    Benchmark FlashAttention vs standard attention at different sequence lengths.
    """
    print("\n" + "=" * 60)
    print("Benchmark: FlashAttention vs Standard Attention")
    print("=" * 60)
    
    import time
    
    batch = 1
    heads = 8
    d = 64
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print(f"{'Seq Len':<12} {'Standard (ms)':<16} {'Flash (ms)':<16} {'Ratio':<12}")
    print("-" * 56)
    
    for L in seq_lengths:
        q = torch.randn(batch, heads, L, d)
        k = torch.randn(batch, heads, L, d)
        v = torch.randn(batch, heads, L, d)
        
        # Standard
        start = time.perf_counter()
        for _ in range(5):
            _ = standard_attention(q, k, v)
        std_time = (time.perf_counter() - start) / 5 * 1000
        
        # Flash (tiled)
        start = time.perf_counter()
        for _ in range(5):
            _ = flashattention_forward(q, k, v, block_size=64)
        flash_time = (time.perf_counter() - start) / 5 * 1000
        
        ratio = std_time / max(flash_time, 1e-8)
        print(f"{L:<12} {std_time:<16.2f} {flash_time:<16.2f} {ratio:<12.2f}")
    
    # Memory comparison
    print("\nMemory comparison (estimated from output):")
    print(f"{'Seq Len':<12} {'Standard (GB)':<16} {'Flash (GB)':<16}")
    print("-" * 44)
    for L in [1024, 4096, 16384]:
        # Standard stores P (L x L) matrix
        std_mem = L * L * 4 / (1024**3)  # float32
        # Flash stores only O (L x d)
        flash_mem = L * d * 4 / (1024**3)
        print(f"{L:<12} {std_mem:<16.6f} {flash_mem:<16.6f}")


def test_pytorch_builtin():
    """
    Show how to use PyTorch's built-in scaled_dot_product_attention
    (which uses FlashAttention automatically on compatible GPUs).
    """
    print("\n" + "=" * 60)
    print("PyTorch 2.0+ built-in FlashAttention")
    print("=" * 60)
    
    # Since 2.0, torch.nn.functional.scaled_dot_product_attention
    # automatically selects FlashAttention backend when input meets conditions
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    v = torch.randn(2, 8, 128, 64)
    
    # This uses FlashAttention on CUDA if available
    out = F.scaled_dot_product_attention(q, k, v)
    print(f"F.scaled_dot_product_attention output shape: {out.shape}")
    print("Note: PyTorch selects FlashAttention backend automatically.")
    print("Conditions: CUDA device, head_dim in {64, 128}, dtype in {fp16, bf16}")
    
    return out


if __name__ == "__main__":
    # Verify correctness
    verify_correctness()
    
    # Benchmark
    benchmark_attention()
    
    # PyTorch integration
    test_pytorch_builtin()
    
    # Full model test
    print("\n" + "=" * 60)
    print("End-to-end model test")
    print("=" * 60)
    model = FlashAttentionWrapper(embed_dim=512, num_heads=8)
    x = torch.randn(2, 256, 512)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print("Model forward pass successful!")
```

## References

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems (NeurIPS), 35*, 16344-16359. https://arxiv.org/abs/2205.14135

Dao, T. (2024). FlashAttention-2: Faster attention with better parallelism and work partitioning. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2307.08691

Shah, J., Bhargava, A., et al. (2024). FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. *arXiv preprint arXiv:2407.08608*. https://arxiv.org/abs/2407.08608

Rabe, M. N., & Staats, C. (2021). Self-attention does not need O(n^2) memory. *arXiv preprint arXiv:2112.05682*. https://arxiv.org/abs/2112.05682

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS), 30*. https://arxiv.org/abs/1706.03762
