# Masked Autoencoder (MAE): Scalable Self-Supervised Learning for Vision

**Source**: He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16000-16009. https://arxiv.org/abs/2111.06377

**Category**: Machine Learning / Self-Supervised Learning / Vision Transformers

## Mathematical Setup

### Problem Definition

Given an input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, we partition it into non-overlapping patches $\{\mathbf{x}^{(i)}\}_{i=1}^N$ where $N = \frac{HW}{P^2}$ for patch size $P \times P$. A random subset $\mathcal{V} \subset \{1, \ldots, N\}$ (size $N - M$) is **visible**, and the complement $\mathcal{M}$ (size $M$) is **masked**.

The encoder maps only visible patches to latent representations:

$$
\mathbf{z}^{(i)} = f_\theta(\mathbf{x}^{(i)}) \quad \forall i \in \mathcal{V}
$$

The decoder reconstructs the original pixels of masked patches from the encoded visible representations plus learnable mask tokens:

$$
\hat{\mathbf{x}}^{(i)} = g_\phi(\mathbf{z}^{(i)}, \mathbf{e}_{\text{mask}}^{(i)}) \quad \forall i \in \mathcal{M}
$$

### Reconstruction Loss

The training minimizes the mean squared error (MSE) in pixel space, computed only on masked patches:

$$
\mathcal{L}_{\text{MAE}}(\theta, \phi) = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \hat{\mathbf{x}}^{(i)} - \mathbf{x}^{(i)} \right\|_2^2
$$

In practice, normalized pixel values are used: each masked patch is normalized to zero mean and unit variance before computing the loss.

### Architecture Design

**Encoder** (ViT): Processes only visible patches (25%):
- Input: Linear projection of visible patches + positional embeddings
- stack of Transformer blocks with standard self-attention
- Output: Latent representations of visible patches

**Decoder** (Lightweight Transformer): Reconstructs the full image:
- Input: Full set of N tokens (encoded visible + learnable mask tokens with positional embeddings)
- Stack of Transformer blocks (fewer, narrower than encoder)
- Output: Linear projection to pixel values for each patch

**Key asymmetry**: The encoder processes only 25% of patches, making it 4x more computationally efficient per forward pass. The decoder is lightweight (8-12% of encoder FLOPs).

### Masking Strategy

Random masking with a **high masking ratio** (75%) is critical. This creates a challenging reconstruction task that forces the model to learn **semantic relationships** rather than simply interpolating nearby pixels.

The probability of masking is uniform across patches:

$$
P(i \in \mathcal{M}) = r = 0.75 \quad \forall i
$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| **Semantic reconstruction** | Reconstructing 75% of patches from 25% requires understanding object parts | Purely low-level texture/color copying is insufficient; model must learn high-level semantics |
| **Asymmetric encoder-decoder** | Encoder processes only visible patches; decoder processes full set | Encoder is 4x more efficient than standard ViT; total computation is comparable to standard ViT despite the decoder |
| **Shallow decoder suffices** | $g_\phi$ can be a lightweight Transformer (1-8 layers, 512 dim) | Pre-training memory is dominated by encoder activations; decoder overhead is minimal |
| **Mask token agnosticism** | All mask tokens share the same learned embedding | Positional embeddings provide location information; mask tokens carry no patch-specific content |

## Applicable Scenarios

**When to use:**
- Pre-training Vision Transformers for image classification, detection, segmentation
- Label-efficient learning (use with 1-10% of labels for fine-tuning)
- Learning general-purpose visual representations transferable across tasks
- Any scenario where you can use a ViT backbone and want the best self-supervised pretraining

**When NOT to use:**
- With convolutional backbones (MAE is specifically designed for patch-based architectures)
- When you need representations at test time for multiple patch resolutions (MAE uses fixed patch size)
- When you have full supervision with sufficient labeled data (supervised pretraining may be simpler)

**Comparison:** MAE outperforms contrastive methods (SimCLR, MoCo v3) and previous masked-image-modeling approaches (BEiT) on downstream transfer. ViT-Huge/MAE achieves 87.8% ImageNet top-1 with fine-tuning only on ImageNet-1K, comparable to the best results using external data.

## Algorithm / Method Details

### Pre-training Algorithm

1. **Divide** input image into $N$ non-overlapping patches.
2. **Sample** a random binary mask with masking ratio $r$ (75%).
3. **Encode** only visible patches through the ViT encoder.
4. **Pad** the encoded sequence with learnable mask tokens at masked positions.
5. **Add** positional embeddings to all $N$ tokens.
6. **Decode** the full sequence through the lightweight decoder.
7. **Compute** MSE loss on masked patches only.

### Fine-tuning Protocol

1. Remove the decoder (discard after pre-training).
2. Add a task-specific head (e.g., linear classifier).
3. Optionally, fine-tune all encoder parameters.
4. Optionally, use intermediate layers for linear probing (without fine-tuning).

### Complexity Analysis

- **Pre-training FLOPs**: Approximately 3.6x less than standard ViT training per iteration (encoder sees only 25% patches, decoder is lightweight).
- **Memory**: 2-3x less than full-image ViT for the encoder; total comparable to standard ViT due to decoder overhead.
- **Convergence**: Converges in 400-1600 epochs (MAE benefits from longer training).

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| Masking ratio | 75% | Optimal range: 60-80%. Too low: task too easy. Too high: no context left |
| Patch size | 16x16 | Standard ViT setting |
| Encoder architecture | ViT-B/L/H | Larger models benefit more from MAE pretraining |
| Decoder depth | 8 blocks | 1-12 blocks; deeper helps but diminishing returns |
| Decoder width | 512 (for ViT-B) | 256-1024; must be narrower than encoder |
| Decoder MLP ratio | 4 | Same as encoder MLP ratio |
| Pre-training epochs | 400-1600 | Longer training consistently improves linear probing |
| Optimizer | AdamW | LR = 1.5e-4 (batch 4096), weight decay = 0.05 |
| Learning rate schedule | Cosine decay | Warmup 40 epochs |

### Numerical Considerations

- Use **normalized pixel targets** per patch (subtract mean, divide by std) -- this stabilizes training and improves representation quality by 0.5-1%.
- The decoder is discarded after pre-training for most downstream tasks -- only the encoder weights are kept.
- For linear probing (evaluation), use BatchNorm or layer normalization on pooled features; linear probing on MAE features is sensitive to normalization.

## Python Implementation

```python
"""
Minimal implementation of Masked Autoencoder (MAE).
Based on: He et al. (2022) https://arxiv.org/abs/2111.06377
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, D)"""
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ============================================================
# MAE Components
# ============================================================

class MAEEncoderBlock(nn.Module):
    """Pre-norm Transformer block for the encoder."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoderBlock(nn.Module):
    """Pre-norm Transformer block for the decoder (same structure)."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone.
    
    Features:
    - Asymmetric encoder-decoder architecture
    - High masking ratio (75%)
    - MSE loss on normalized pixel targets
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        masking_ratio: float = 0.75,
    ):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable positional embeddings for encoder
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, encoder_dim) * 0.02)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            MAEEncoderBlock(encoder_dim, encoder_heads, mlp_ratio)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # Decoder embedding
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        # Decoder positional embeddings
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_dim) * 0.02)

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            MAEDecoderBlock(decoder_dim, decoder_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Prediction head: maps decoder output to pixel values
        self.pred_head = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following MAE/DeiT strategy."""
        # Positional embeddings: initialize with sine-cosine and trunc_normal
        pos_embed = self.get_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        dec_pos_embed = self.get_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5)
        )
        self.decoder_pos_embed.data.copy_(dec_pos_embed.unsqueeze(0))

        # Timm trunc_normal_
        for p in self.parameters():
            if p.ndim >= 2 and p is not self.pos_embed and p is not self.decoder_pos_embed:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def get_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
        """Create 2D sine-cosine positional embedding."""
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

        pos_dim = embed_dim // 2
        omega = torch.exp(torch.arange(pos_dim, dtype=torch.float32) * (-math.log(10000.0) / pos_dim))
        out_h = grid_h.flatten()[:, None] * omega[None, :]
        out_w = grid_w.flatten()[:, None] * omega[None, :]
        pos_embed = torch.cat([torch.sin(out_h), torch.cos(out_w)], dim=1)
        return pos_embed

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches.

        Args:
            x: (B, N, D) input patches
            mask_ratio: fraction of patches to mask

        Returns:
            x_masked: (B, N_visible, D) visible patches
            mask: (B, N) binary mask (0=visible, 1=masked)
            ids_restore: (B, N) indices to restore original order
        """
        B, N, D = x.shape
        n_masked = int(N * mask_ratio)

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep visible patches, mask the rest
        ids_keep = ids_shuffle[:, :N - n_masked]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Mask: 0 = visible, 1 = masked
        mask = torch.ones([B, N], device=x.device)
        mask[:, :N - n_masked] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode visible patches.

        Args:
            x: (B, C, H, W) input image

        Returns:
            latent: (B, N_visible, D_encoder) encoded visible patches
            mask: (B, N) binary mask
            ids_restore: (B, N) restore indices
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed

        # Masking
        x, mask, ids_restore = self.random_masking(x, self.masking_ratio)

        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Decode and reconstruct.

        Args:
            x: (B, N_visible, D_encoder) encoded visible patches
            ids_restore: (B, N) restore indices

        Returns:
            pred: (B, N, patch_size^2 * C) pixel predictions for all patches
        """
        # Project encoder output to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible, D_decoder)

        # Add mask tokens
        B, N_visible, D_decoder = x.shape
        N = ids_restore.shape[1]
        n_masked = N - N_visible

        mask_tokens = self.mask_token.repeat(B, n_masked, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, D_decoder)

        # Unshuffle
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D_decoder)
        )

        # Add positional embeddings
        x_full = x_full + self.decoder_pos_embed

        # Decoder
        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)

        # Predict pixels
        pred = self.pred_head(x_full)  # (B, N, p^2 * C)

        return pred

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss on masked patches with normalized targets.

        Args:
            x: (B, C, H, W) input image
            pred: (B, N, p^2 * C) predictions
            mask: (B, N) binary mask (0=visible, 1=masked)

        Returns:
            loss: scalar MSE on masked patches
        """
        # Target: reshape image patches
        B, C, H, W = x.shape
        p = self.patch_size
        N = (H // p) * (W // p)
        # (B, C, H, W) -> (B, N, p^2 * C)
        target = x.reshape(B, C, H // p, p, W // p, p)
        target = target.permute(0, 2, 4, 3, 5, 1).reshape(B, N, -1)

        # Normalize each patch to mean=0, var=1
        target_mean = target.mean(dim=-1, keepdim=True)
        target_var = target.var(dim=-1, keepdim=True, unbiased=False)
        target_normalized = (target - target_mean) / (target_var.sqrt() + 1e-6)

        # MSE loss on masked patches only
        loss = (pred - target_normalized) ** 2
        loss = loss.mean(dim=-1)  # per-patch loss
        loss = (loss * mask).sum() / mask.sum()  # average over masked patches

        return loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full MAE pre-training forward pass.

        Args:
            x: (B, C, H, W) input image

        Returns:
            loss: reconstruction loss
            pred: predictions
            mask: mask for visualization
        """
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask


# ============================================================
# Usage example
# ============================================================

def test_mae():
    """Test MAE on synthetic image data."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create a small MAE for testing
    model = MaskedAutoencoderViT(
        img_size=32,        # Small images for testing
        patch_size=4,       # 8x8 = 64 patches
        in_chans=3,
        encoder_dim=192,
        encoder_depth=6,
        encoder_heads=6,
        decoder_dim=128,
        decoder_depth=4,
        decoder_heads=4,
        masking_ratio=0.75,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    encoder_params = sum(p.numel() for p in model.encoder_blocks.parameters())
    decoder_params = sum(p.numel() for p in model.decoder_blocks.parameters())
    print(f"Encoder params: {encoder_params:,}")
    print(f"Decoder params: {decoder_params:,}")

    # Synthetic image: create a "checkerboard" pattern
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32, device=device)

    # Forward pass
    loss, pred, mask = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked fraction: {mask.mean().item():.3f} (expected ~0.75)")

    # Visualize reconstruction
    with torch.no_grad():
        # Example: show first image
        img = x[0].cpu()
        pred_img = pred[0].cpu()
        mask_2d = mask[0].reshape(8, 8).cpu()

        # Reconstruct image from predictions
        p = model.patch_size
        C = 3
        N = 64
        pred_patches = pred_img.reshape(N, p, p, C)
        pred_patches = pred_patches.permute(0, 3, 1, 2)  # (N, C, p, p)
        pred_image = pred_patches.reshape(8, 8, C, p, p)
        pred_image = pred_image.permute(2, 0, 3, 1, 4).reshape(C, 8 * p, 8 * p)

        print(f"\nReconstruction shape: {pred_image.shape}")
        print(f"Masked pixels (should show which regions are reconstructed):")
        print(f"  {mask_2d.numpy().tolist()}")

    # Backward
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"\nTotal gradient norm: {grad_norm:.4f}")
    print("MAE forward/backward successful!")


def linear_probing_demo():
    """
    Demonstrate linear probing: freeze encoder, train a linear classifier.
    Uses synthetic 2D "blob" classification data.
    """
    print("\n" + "=" * 50)
    print("Linear probing demo with MAE features")
    print("=" * 50)

    # Generate synthetic 2D data (simplified: we use this to show the probing logic)
    # In practice, MAE features would be used on real images
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Linear probing accuracy: {acc:.4f}")
    print("(This is a proxy demo; real MAE probing uses frozen ViT features.)")


if __name__ == "__main__":
    test_mae()
    linear_probing_demo()
```

## References

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16000-16009. https://arxiv.org/abs/2111.06377

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2010.11929

Chen, X., Xie, S., & He, K. (2021). An empirical study of training self-supervised vision transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 9640-9649. https://arxiv.org/abs/2104.02057

Bao, H., Dong, L., Piao, S., & Wei, F. (2022). BEiT: BERT pre-training of image transformers. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2106.08254

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 9650-9660. https://arxiv.org/abs/2104.14294
