# Enformer: Gene Expression Prediction from DNA Sequence via Long-Range Interactions

**Source**: Avsec, Z., Agarwal, V., Visentin, D., Ledsam, J. R., Grabska-Barwinska, A., Taylor, K. R., Assael, Y., Jumper, J., Kohli, P., & Kelley, D. R. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*, 18(10), 1196--1203. https://doi.org/10.1038/s41592-021-01252-x

**Category**: Bioinformatics / Regulatory Genomics

## Biological / Computational Problem

Understanding how DNA sequence encodes gene expression is a fundamental problem in genomics. Gene expression is regulated by cis-regulatory elements (promoters, enhancers, silencers) that can be located hundreds of kilobases away from the transcription start site (TSS). The key challenge is to predict gene expression levels **directly from DNA sequence** while capturing these long-range regulatory interactions.

- **Input data**: DNA sequence (200 kb window centered on TSS), one-hot encoded
- **Output**: Predicted genomic track values across 5,313 human and mouse tracks (CAGE expression, chromatin accessibility, transcription factor binding, histone marks)

## Mathematical / Computational Model

### Architecture Overview

Enformer uses a **hybrid CNN + Transformer architecture** - a portmanteau of "enhancer" and "transformer":

```
Input (200 kb sequence, one-hot, 196,608 bp)
  └─> Convolutional stem (7 conv blocks, pooling to 128 bp bins)
       └─> 11 Transformer encoder layers (self-attention + feed-forward)
            └─> Cropping (remove 64 bins from each side)
                 └─> Output heads (5,313 tracks at 128 bp resolution)
```

### Convolutional Stem

The first layers are convolutional and pooling blocks that reduce the resolution:

$$\mathbf{h}^{(0)} = \text{Pool}(\text{ReLU}(\text{BatchNorm}(\text{Conv1D}(\mathbf{X})))))$$

where $\mathbf{X} \in \mathbb{R}^{L \times 4}$ is the one-hot encoded DNA sequence (A, C, G, T). After 7 convolutional blocks with progressive pooling (factor 2 each), the length reduces from 196,608 bp to 1,536 bins at 128 bp resolution.

### Transformer (Self-Attention) Blocks

Each transformer block applies **multi-head self-attention** followed by a feed-forward network:

$$\mathbf{h}^{(\ell+1)} = \text{LayerNorm}(\mathbf{h}^{(\ell)} + \text{MHA}(\mathbf{h}^{(\ell)}))$$

$$\mathbf{h}^{(\ell+1)} = \text{LayerNorm}(\mathbf{h}^{(\ell+1)} + \text{FFW}(\mathbf{h}^{(\ell+1)}))$$

The **self-attention** mechanism allows each position to attend to all other positions, enabling the model to capture long-range enhancer-promoter interactions:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, $V$ are learned linear projections of the input. With 8 attention heads and 1,536 positions, the attention matrix is $1,536 \times 1,536$, enabling modeling of interactions up to $\sim$200 kb away.

### Multi-Task Output

The final layer produces $T$ genomic tracks:

$$\mathbf{y}_t = \text{Linear}(\mathbf{h}^{(L)}_{\text{cropped}}) \quad \text{for } t = 1, \dots, T$$

with $T = 5,313$ tracks including:
- CAGE expression (across human + mouse tissues)
- DNase-seq / ATAC-seq accessibility
- ChIP-seq for transcription factors and histone modifications

### Loss Function

The model is trained with a **Poisson-like loss** (negative log-likelihood under a Poisson observation model):

$$\mathcal{L} = \sum_{t=1}^{T} \sum_{i=1}^{N} \left( \hat{y}_{t,i} - y_{t,i} \log \hat{y}_{t,i} \right)$$

where $\hat{y}$ are predictions and $y$ are observed values (quantile-transformed).

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Sequence is sufficient | $p(\text{expression} \mid \text{DNA})$ only | Cannot capture epigenetic state, cell-type-specific chromatin conformation |
| 200 kb context window is sufficient | Input is $\pm$100 kb from TSS | May miss very distal enhancers ($>$200 kb away) |
| Additive contributions across bins | Predictions from non-overlapping 128 bp bins | Cannot model sub-128 bp regulatory grammar |
| One-hot encoding captures all sequence features | $\mathbf{X} \in \{0,1\}^{L \times 4}$ | Does not explicitly model DNA shape, methylation, or structural features |

## Applicable Scenarios

**When to use**:
- Predicting the effect of non-coding genetic variants on gene expression
- Identifying candidate causal variants from GWAS/eQTL fine-mapping
- Understanding regulatory grammar and enhancer-promoter interactions
- Scoring the regulatory impact of sequence perturbations in silico

**When NOT to use**:
- Genome-scale predictions (inference on $\sim$30,000 TSS regions is computationally intensive)
- Situations requiring single-cell resolution (predicts bulk expression)
- Variants in regions with strong allele-specific effects from chromatin state (not captured)

**Comparison**: Outperforms Basenji2 (previous state-of-the-art) from 0.81 to 0.85 Pearson correlation for CAGE. Borzoi (2025, *Nature Genetics*) later extended this to 524 kb context at 32 bp resolution with 0.87 correlation. Sei (Chen et al., 2022, *Nature Genetics*) offers complementary interpretability-focused architecture.

## Implementation Details

- **Key parameters**: 11 transformer layers (1,536 dim, 8 heads), 7 conv blocks (288--1,536 channels), dropout 0.3
- **Computational requirements**: Trained on 8 TPUv3 cores for 7 days; inference $\sim$1 second per TSS region on GPU
- **Preprocessing**:
  - Reference genome: hg38 (human), mm10 (mouse)
  - Sequence: one-hot encode with reverse complement augmentation
  - Targets: quantile normalization across tracks, capped at 95th percentile

## Python Implementation

```python
"""
Minimal implementation of Enformer-style gene expression prediction from DNA sequence.

This provides a simplified version of the CNN + Transformer architecture,
demonstrating the core computational approach on synthetic genomic data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


def one_hot_encode(sequence: str, max_length: int = 196_608) -> torch.Tensor:
    """
    One-hot encode a DNA sequence.
    
    Parameters
    ----------
    sequence : str
        DNA sequence of A, C, G, T characters.
    max_length : int
        Pad/truncate to this length.
    
    Returns
    -------
    Tensor (4 x max_length)
        One-hot encoded sequence.
    """
    seq = sequence.upper()[:max_length]
    seq = seq.ljust(max_length, "N")[:max_length]
    
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    tokens = torch.tensor([mapping.get(c, 4) for c in seq], dtype=torch.long)  # (L,)
    
    one_hot = F.one_hot(tokens, num_classes=5).float()  # (L, 5)
    one_hot = one_hot[:, :4]  # Drop N channel, keep A,C,G,T
    return one_hot.T  # (4, L)


class ConvBlock(nn.Module):
    """Convolutional block with batch norm, ReLU, and pooling."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, 
                 dilation: int = 1, pool_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding="same", dilation=dilation
        )
        self.bn = nn.Batchorm1d(out_channels)
        self.pool = nn.AvgPool1d(pool_factor) if pool_factor > 1 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with relative positional encoding.
    
    Simplified version using absolute positional encoding.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-layer normalization."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SimplifiedEnformer(nn.Module):
    """
    Simplified Enformer model for gene expression prediction from DNA sequence.
    
    Architecture: Conv stem -> Transformer -> Output heads.
    
    Parameters
    ----------
    n_output_tracks : int
        Number of genomic tracks to predict.
    d_model : int
        Dimension of transformer representations.
    n_heads : int
        Number of attention heads.
    n_transformer_layers : int
        Number of transformer blocks.
    """
    
    def __init__(
        self,
        n_output_tracks: int = 5,
        d_model: int = 256,
        n_heads: int = 4,
        n_transformer_layers: int = 4,
    ):
        super().__init__()
        
        # Convolutional stem: downsample from ~200k bp to ~1.5k bins
        # Using smaller scale for the simplified version
        
        # Input: (B, 4, L) one-hot sequence
        
        # We remove the first conv (vanilla conv) and start from the next
        # For simplicity, we downsample by factor ~128
        self.conv_stem = nn.Sequential(
            ConvBlock(4, 64, kernel_size=15, dilation=1, pool_factor=2),
            ConvBlock(64, 128, kernel_size=11, dilation=1, pool_factor=2),
            ConvBlock(128, 128, kernel_size=7, dilation=1, pool_factor=2),
            ConvBlock(128, d_model, kernel_size=7, dilation=1, pool_factor=2),
            ConvBlock(d_model, d_model, kernel_size=5, dilation=1, pool_factor=2),
            ConvBlock(d_model, d_model, kernel_size=3, dilation=1, pool_factor=2),
            ConvBlock(d_model, d_model, kernel_size=3, dilation=1, pool_factor=2),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )  # total pooling: 2^7 = 128x
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_transformer_layers)
        ])
        
        # Final output head per track
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, n_output_tracks),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, 4, L)
            One-hot encoded DNA sequences.
            
        Returns
        -------
        Tensor (B, n_bins, n_output_tracks)
            Predictions for each genomic bin and track.
        """
        # Convolutional stem
        h = self.conv_stem(x)  # (B, d_model, n_bins)
        h = h.permute(0, 2, 1)  # (B, n_bins, d_model)
        
        # Transformer layers
        for block in self.transformer_blocks:
            h = block(h)  # (B, n_bins, d_model)
        
        # Output: predict for all bins
        output = self.output_head(h)  # (B, n_bins, n_output_tracks)
        
        return output


def simulate_genomic_data(
    n_samples: int = 100,
    seq_length: int = 8192,
    n_output_tracks: int = 5,
    n_active_regions: int = 3,
    seed: int = 42,
):
    """
    Simulate synthetic genomic sequence-expression data.
    
    A random subset of positions are "enhancers" that contribute to 
    expression of nearby "genes" at target positions.
    
    Parameters
    ----------
    n_samples : int
        Number of sequence samples.
    seq_length : int
        Length of each sequence.
    n_output_tracks : int
        Number of output tracks (simulating different tissues/assays).
    n_active_regions : int
        Number of regulatory elements per sequence.
    
    Returns
    -------
    sequences, targets
    """
    rng = np.random.default_rng(seed)
    
    # Generate random DNA sequences
    sequences = []
    targets = np.zeros((n_samples, seq_length // 128, n_output_tracks))
    
    for i in range(n_samples):
        seq = "".join(rng.choice(["A", "C", "G", "T"], size=seq_length))
        sequences.append(seq)
        
        # Place some "enhancer" motifs that boost expression at target bins
        for _ in range(n_active_regions):
            enh_pos = rng.integers(0, seq_length - 10)
            target_bin = rng.integers(0, seq_length // 128)
            
            # Boost expression in specific tracks
            for t in range(n_output_tracks):
                if rng.random() < 0.4:
                    targets[i, target_bin, t] += rng.exponential(2.0)
        
        # Add noise
        targets[i] += rng.exponential(0.3, size=targets[i].shape)
    
    return sequences, torch.tensor(targets, dtype=torch.float32)


class EnformerTrainer:
    """
    Trainer wrapper for SimplifiedEnformer with data preprocessing.
    """
    
    def __init__(self, model: SimplifiedEnformer, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
        
    def train_step(self, sequences: list, targets: torch.Tensor) -> float:
        """
        Single training step.
        
        Parameters
        ----------
        sequences : list of str
            DNA sequences.
        targets : Tensor (B, n_bins, n_tracks)
            Observed expression values.
        
        Returns
        -------
        Loss value.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # One-hot encode
        encoded = torch.stack([
            one_hot_encode(s).T for s in sequences
        ])  # (B, L, 4)
        encoded = encoded.permute(0, 2, 1)  # (B, 4, L)
        
        # Forward
        predictions = self.model(encoded)  # (B, n_bins, n_tracks)
        
        # Ensure same length (crop/align)
        min_len = min(predictions.shape[1], targets.shape[1])
        pred_crop = predictions[:, :min_len, :]
        targ_crop = targets[:, :min_len, :]
        
        # Loss
        loss = self.loss_fn(pred_crop, targ_crop + 1e-6)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, sequences: list, targets: torch.Tensor) -> dict:
        """
        Evaluate model performance.
        
        Returns
        -------
        dict of metrics.
        """
        self.model.eval()
        
        with torch.no_grad():
            encoded = torch.stack([
                one_hot_encode(s).T for s in sequences
            ]).permute(0, 2, 1)
            
            predictions = self.model(encoded)
            
            min_len = min(predictions.shape[1], targets.shape[1])
            pred_crop = predictions[:, :min_len, :]
            targ_crop = targets[:, :min_len, :]
            
            # Pearson correlation per track
            corrs = []
            for t in range(targets.shape[-1]):
                pred_flat = pred_crop[:, :, t].reshape(-1)
                targ_flat = targ_crop[:, :, t].reshape(-1)
                
                if pred_flat.std() > 0 and targ_flat.std() > 0:
                    corr = np.corrcoef(pred_flat.numpy(), targ_flat.numpy())[0, 1]
                else:
                    corr = 0.0
                corrs.append(corr)
            
            # Poisson loss
            loss = self.loss_fn(pred_crop, targ_crop + 1e-6).item()
        
        return {
            "loss": loss,
            "mean_corr": np.mean(corrs),
            "per_track_corr": corrs,
        }


# ============================================================================
# Complete usage example
# ============================================================================

def main():
    """
    Demonstrate Enformer-style sequence-to-expression prediction.
    """
    print("=" * 60)
    print("Enformer-style Gene Expression Prediction")
    print("=" * 60)
    
    # --- 1. Simulate genomic data ---
    print("\n[1] Generating synthetic genomic data...")
    seq_length = 8192  # Smaller scale for demonstration
    n_tracks = 5
    
    train_seqs, train_targets = simulate_genomic_data(
        n_samples=80, seq_length=seq_length, n_output_tracks=n_tracks
    )
    test_seqs, test_targets = simulate_genomic_data(
        n_samples=20, seq_length=seq_length, n_output_tracks=n_tracks, seed=99
    )
    
    print(f"    Training samples: {len(train_seqs)}")
    print(f"    Test samples: {len(test_seqs)}")
    print(f"    Sequence length: {seq_length} bp")
    print(f"    Output bins: {train_targets.shape[1]}")
    print(f"    Output tracks: {n_tracks}")
    
    # --- 2. Create model ---
    print("\n[2] Initializing Enformer model...")
    model = SimplifiedEnformer(
        n_output_tracks=n_tracks,
        d_model=128,
        n_heads=4,
        n_transformer_layers=3,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {n_params:,}")
    
    # --- 3. Train for a few epochs ---
    print("\n[3] Training (20 epochs, mini-batch)...")
    trainer = EnformerTrainer(model, lr=5e-4)
    
    batch_size = 16
    n_epochs = 20
    n_batches = len(train_seqs) // batch_size
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            batch_seqs = train_seqs[start:end]
            batch_targets = train_targets[start:end]
            
            loss = trainer.train_step(batch_seqs, batch_targets)
            epoch_loss += loss
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"    Epoch {epoch+1}/{n_epochs}, loss: {avg_loss:.4f}")
    
    # --- 4. Evaluate on test set ---
    print("\n[4] Evaluation on test set...")
    metrics = trainer.evaluate(test_seqs, test_targets)
    
    print(f"    Test Poisson loss: {metrics['loss']:.4f}")
    print(f"    Mean Pearson corr: {metrics['mean_corr']:.4f}")
    print(f"    Per-track correlations:")
    for t in range(n_tracks):
        print(f"      Track {t+1}: r = {metrics['per_track_corr'][t]:.4f}")
    
    # --- 5. Demonstrate variant effect prediction ---
    print("\n[5] Variant effect prediction example...")
    trainer.model.eval()
    ref_seq = test_seqs[0]
    
    # Create a "mutant" with a single nucleotide change
    pos = 100  # arbitrary position
    orig_base = ref_seq[pos]
    alt_base = {"A": "C", "C": "G", "G": "T", "T": "A"}[orig_base]
    mut_seq = ref_seq[:pos] + alt_base + ref_seq[pos+1:]
    
    with torch.no_grad():
        ref_enc = one_hot_encode(ref_seq).T.unsqueeze(0).permute(0, 2, 1)
        mut_enc = one_hot_encode(mut_seq).T.unsqueeze(0).permute(0, 2, 1)
        
        ref_pred = trainer.model(ref_enc).squeeze()
        mut_pred = trainer.model(mut_enc).squeeze()
    
    effect = (mut_pred - ref_pred).abs().max().item()
    print(f"    Reference base at position {pos}: {orig_base}")
    print(f"    Altered base: {alt_base}")
    print(f"    Max absolute expression change: {effect:.4f}")
    
    print("\n" + "=" * 60)
    print("Enformer demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## References

Avsec, Z., Agarwal, V., Visentin, D., et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*, 18(10), 1196--1203. https://doi.org/10.1038/s41592-021-01252-x

Linder, J., Srivastava, D., et al. (2025). Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation. *Nature Genetics*, 57, 164--174. https://doi.org/10.1038/s41588-024-02053-6

Chen, K. M., Wong, A. K., Troyanskaya, O. G., & Zhou, J. (2022). A sequence-based global map of regulatory activity for deciphering human genetics. *Nature Genetics*, 54, 1058--1069. https://doi.org/10.1038/s41588-022-01102-2

Kelley, D. R., Reshef, Y. A., Bileschi, M., Belanger, D., McLean, C. Y., & Snoek, J. (2018). Sequential regulatory activity prediction across chromosomes with convolutional neural networks. *Genome Research*, 28(5), 739--750. https://doi.org/10.1101/gr.227819.117

Avsec, Z., Weilert, M., Shrikumar, A., et al. (2021). Base-resolution models of transcription-factor binding reveal soft motif syntax. *Nature Genetics*, 53, 354--366. https://doi.org/10.1038/s41588-021-00782-6
