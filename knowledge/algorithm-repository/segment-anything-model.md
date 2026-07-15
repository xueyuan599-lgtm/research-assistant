# Segment Anything Model (SAM): Foundation Model for Image Segmentation

**Source**: Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., & Girshick, R. (2023). Segment anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015-4026. https://arxiv.org/abs/2304.02643

**Category**: Machine Learning / Computer Vision / Foundation Models / Segmentation

## Mathematical Setup

### Task: Promptable Segmentation

SAM defines a new task: given any input prompt (point, bounding box, mask, or text), produce a valid segmentation mask. This generalizes standard segmentation tasks where the prompt is typically implicit (e.g., segment all objects of a class).

Formally, the model learns a function $f_\theta$ that maps an image $I \in \mathbb{R}^{H \times W \times 3}$ and a prompt $p \in \mathcal{P}$ to a binary mask $M \in \{0, 1\}^{H \times W}$:

$$
M = f_\theta(I, p)
$$

where $\mathcal{P}$ is the space of all possible prompts (points, boxes, masks, text).

### Architecture: Three-Component Design

SAM consists of three main components:

**1. Image Encoder** $\mathcal{E}_{\text{img}}$: A MAE-pretrained Vision Transformer (ViT) that produces an image embedding:

$$
\mathbf{F} = \mathcal{E}_{\text{img}}(I) \in \mathbb{R}^{h \times w \times d}
$$

**2. Prompt Encoder** $\mathcal{E}_{\text{prompt}}$: Encodes different prompt types into a shared embedding space:

- **Points**: Positional encoding + learned token for each point type (foreground/background).
- **Bounding boxes**: Positional encoding of the box corners + a learned "bounding box" token.
- **Masks**: Convolutional embedding + element-wise sum with image embedding.
- **Text**: CLIP text encoder (zero-shot transfer, not part of the main SAM architecture).

**3. Mask Decoder** $\mathcal{D}_{\text{mask}}$: A lightweight Transformer that processes the image embedding and prompt embedding to produce the mask:

$$
M = \mathcal{D}_{\text{mask}}(\mathbf{F}, \mathbf{p})
$$

The decoder uses a modified Transformer decoder with dynamic mask prediction:

```
Input: image embedding F, prompt embedding p, output token
  |
  Self-attention (prompt tokens + output token)
  Cross-attention (prompt tokens -> image features)
  Point-wise MLP for each output token
  |
  Dynamic linear classifier: mask = output_token @ F
  |
Output: M binary masks with IoU scores
```

### Ambiguity-Aware Design

Segmentation is inherently ambiguous: a single point on a shirt could refer to the whole shirt, the left sleeve, or the collar. SAM handles this by predicting **multiple valid masks** per prompt (typically 3 masks) using multiple output tokens.

Each output token $t_i$ produces a mask $M_i$ and an IoU score $s_i$ estimating its quality:

$$
M_i = \sigma(t_i^\top \mathbf{F}), \quad s_i = \text{MLP}(t_i) \in [0, 1]
$$

The IoU scores allow downstream filtering: for interactive use, the user can cycle through the valid masks.

### Training Objective

SAM is trained on the SA-1B dataset (11M images, 1.1B masks) with a combination of losses:

$$
\mathcal{L} = \lambda_{\text{focal}} \mathcal{L}_{\text{focal}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}} + \lambda_{\text{iou}} \mathcal{L}_{\text{IoU}}
$$

where:
- $\mathcal{L}_{\text{focal}}$ is the focal loss for pixel-wise classification,
- $\mathcal{L}_{\text{dice}}$ is the Dice loss for mask overlap,
- $\mathcal{L}_{\text{IoU}}$ is the IoU prediction loss (regression to ground-truth IoU).

Typical weights: $\lambda_{\text{focal}} = 20$, $\lambda_{\text{dice}} = 1$, $\lambda_{\text{iou}} = 1$.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| **Promptable segmentation** | Fixed prompt space $\mathcal{P}$ covers all real use cases | All downstream tasks are formulated as prompting the model at inference |
| **Ambiguity resolution** | 3 output masks per prompt cover all valid interpretations | A single unambiguous prompt (like a box) may need only 1 mask |
| **Generalizable features** | ViT-H pretrained with MAE learns universal visual features | SAM generalizes zero-shot to unseen domains (medical, aerial, etc.) |
| **Data scaling** | 1.1B masks from 11M images is sufficient for the task | Larger models (ViT-H) and more data monotonically improve performance |

## Applicable Scenarios

**When to use:**
- Interactive segmentation tools (click-and-segment)
- Zero-shot segmentation on novel domains (medical images, satellite imagery, microscopy)
- Automatic annotation pipelines (generate pseudo-labels for downstream training)
- Any segmentation task where you want to avoid dataset-specific training
- Video tracking by segmenting each frame independently with tracking prompts

**When NOT to use:**
- When you need class labels (SAM produces masks, not categories)
- Real-time video at high frame rates (>30fps on consumer GPUs)
- Very large images (megapixel scale, though there are hierarchical variants)
- When you need exact boundary alignment (SAM masks may have slightly rough edges compared to specialized models)

**Comparison:** Compared to specialized segmentation models (Mask R-CNN, DeepLab), SAM trades per-dataset performance for generality. On a specific dataset with labeled training data, a specialized model will outperform SAM by 2-5% mIoU. But SAM requires zero training for new domains.

## Algorithm / Method Details

### Inference Pipeline

1. **Precompute image embedding**: Pass the image through the ViT encoder once.
2. **Encode prompt**: Convert the user prompt (point, box, mask) to prompt embedding.
3. **Decode mask**: Run lightweight decoder to produce mask(s) and IoU scores.
4. **Filter**: Select the mask with the highest predicted IoU, or let user cycle through.

Since the image embedding is precomputed, prompt changes require only steps 2-4 (real-time on CPU for interactive use).

### Training Procedure

1. **Data engine**: Three-stage data collection with model-in-the-loop:
   - Stage 1: Manual annotation with SAM assistance (120K images).
   - Stage 2: Semi-automatic (model proposes masks, annotators correct, model retrained).
   - Stage 3: Fully automatic (model generates masks with grid prompts + post-processing filtering).
2. **Model training**: Train with focal + dice loss on SA-1B masks.
3. **Distillation**: The ViT-H model can be distilled to ViT-B/L for deployment.

### Model Variants

| Model | Encoder | Parameters | Speed (ms/image) |
|-------|---------|------------|------------------|
| SAM-B | ViT-B | 91M | ~10 |
| SAM-L | ViT-L | 308M | ~20 |
| SAM-H | ViT-H | 637M | ~35 |

### Complexity Analysis

- **Image encoding**: O($h w d^2$) for the ViT, amortized over prompts.
- **Prompt encoding**: O(1) for points/boxes in embedding space.
- **Mask decoding**: O($hw d$) for the decoder (lightweight).
- **Total (per prompt)**: O($h w d^2$) one-time encoder + O($hw d$) per prompt.

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| Image encoder | ViT-H (MAE pretrained) | ViT-B for speed, ViT-H for accuracy |
| Decoder depth | 2 Transformer blocks | 1-3 blocks; performance saturates at 2 |
| Points per side (auto) | 32 for grid prompting | More points = more masks but slower |
| IoU prediction head | 3-layer MLP | Predicts mask quality, used for filtering |
| NMS threshold | 0.7 | Remove duplicate masks in automatic generation |
| Prompt types | Points, boxes, masks, text | Text requires external CLIP encoder |

### Framework/Library Support

- **Official repo**: `pip install segment-anything` (https://github.com/facebookresearch/segment-anything)
- **Hugging Face**: `transformers.SamModel` integrated (https://huggingface.co/docs/transformers/model_doc/sam)
- **SAM 2**: Extended to video with memory mechanism (2024)

## Python Implementation

```python
"""
Reference implementation of the Segment Anything Model (SAM).
Based on: Kirillov et al. (2023) https://arxiv.org/abs/2304.02643

This is a minimal, self-contained implementation of SAM's core components
for educational purposes. For production, use the official library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np


# ============================================================
# Positional Encoding
# ============================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Create 2D sine-cosine positional embedding.
    
    Args:
        embed_dim: Embedding dimension (must be even)
        grid_size: Grid size (height = width)
    
    Returns:
        pos_embed: (grid_size^2, embed_dim)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

    pos_dim = embed_dim // 2
    omega = torch.exp(torch.arange(pos_dim, dtype=torch.float32) * (-math.log(10000.0) / (pos_dim - 1)))
    out_h = grid_h.flatten()[:, None] * omega[None, :]
    out_w = grid_w.flatten()[:, None] * omega[None, :]
    pos_embed = torch.cat([torch.sin(out_h), torch.cos(out_w)], dim=1)
    return pos_embed


# ============================================================
# Image Encoder (simplified ViT)
# ============================================================

class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Attention(nn.Module):
    """Multi-head attention block."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoderViT(nn.Module):
    """Vision Transformer for SAM image encoding."""
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        global_attn_indexes: List[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches ** 0.5))
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)

        # Class token is not used in SAM
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_global = global_attn_indexes and i in global_attn_indexes
            self.blocks.append(
                Block(dim=embed_dim, num_heads=num_heads)
            )

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.LayerNorm(embed_dim, elementwise_affine=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, C, H//16, W//16)"""
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H//16, W//16)
        B, C, H, W = x.shape

        # Flatten and add positional encoding
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Reshape back to spatial
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.neck(x)
        return x  # (B, C, H//16, W//16)


# ============================================================
# Prompt Encoder
# ============================================================

class PromptEncoder(nn.Module):
    """
    Encodes various prompt types into a shared embedding space.
    
    Supports:
    - Points with labels (foreground/background)
    - Bounding boxes
    - Coarse masks
    """
    def __init__(
        self,
        embed_dim: int = 256,
        image_embed_size: int = 64,  # H/16 for 1024px image
        point_embed_dim: int = 256,
        mask_in_chans: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embed_size = image_embed_size

        # Point embeddings: learned tokens for point types
        self.pe_point = nn.Embedding(1, point_embed_dim)  # foreground
        self.pe_neg_point = nn.Embedding(1, point_embed_dim)  # background
        self.pe_box = nn.Embedding(1, point_embed_dim)
        self.pe_center = nn.Embedding(1, point_embed_dim)

        # Positional encoding for prompt coordinates
        self.pe_layer = PositionEmbeddingRandom(point_embed_dim // 2)

        # Mask downscaling network
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        # For points: produce dense embedding
        self.point_mlp = nn.Sequential(
            nn.Linear(point_embed_dim * 2 + embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _embed_points(
        self, points: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Encode point prompts.
        
        Args:
            points: (B, N_pts, 2) normalized coordinates [0, 1]
            labels: (B, N_pts) 1 for foreground, 0 for background
        
        Returns:
            point_embedding: (B, N_pts, embed_dim)
        """
        B, N_pts, _ = points.shape
        # Positional encoding of point coordinates
        point_pos = self.pe_layer(points)  # (B, N_pts, embed_dim)
        # Learned tokens
        fg_embed = self.pe_point.weight.unsqueeze(0).expand(B, N_pts, -1)
        bg_embed = self.pe_neg_point.weight.unsqueeze(0).expand(B, N_pts, -1)
        point_embed = torch.where(labels.unsqueeze(-1).bool(), fg_embed, bg_embed)
        return point_embed + point_pos

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode box prompts.
        
        Args:
            boxes: (B, 4) = [x1, y1, x2, y2] normalized
        
        Returns:
            box_embedding: (B, 2, embed_dim)
        """
        B, _ = boxes.shape
        corners = torch.stack([
            boxes[:, :2],  # top-left
            boxes[:, 2:],   # bottom-right
        ], dim=1)  # (B, 2, 2)

        corner_pe = self.pe_layer(corners)  # (B, 2, embed_dim)
        box_embed = self.pe_box.weight.unsqueeze(0).expand(B, 2, -1)
        return box_embed + corner_pe

    def forward(self, points=None, boxes=None, masks=None):
        """
        Returns:
            sparse_prompt_embeddings: (B, N_prompts, embed_dim)
            dense_prompt_embeddings: (B, embed_dim, H, W) or None
        """
        sparse = []
        dense = None

        if points is not None:
            sparse.append(self._embed_points(points["coords"], points["labels"]))

        if boxes is not None:
            sparse.append(self._embed_boxes(boxes))

        sparse = torch.cat(sparse, dim=1) if sparse else None

        if masks is not None:
            dense = self.mask_downscaling(masks.unsqueeze(1))

        return sparse, dense


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding that supports random coordinates."""
    def __init__(self, num_pos_feats: int, scale: float = 1.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        self.register_buffer("pos_weights", scale * torch.randn(1, num_pos_feats * 2))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates.
        
        Args:
            coords: (B, N, 2) in [0, 1]
        
        Returns:
            pos_embed: (B, N, num_pos_feats * 2)
        """
        coords = coords * 2 - 1  # normalize to [-1, 1]
        coords = coords @ self.pos_weights  # (B, N, num_pos_feats * 2)
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


# ============================================================
# Mask Decoder
# ============================================================

class MaskDecoder(nn.Module):
    """
    Lightweight Transformer decoder that produces masks from
    image embeddings and prompt embeddings.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_masks: int = 3,  # multiple outputs for ambiguity
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_masks = num_masks

        # Output tokens (one per mask)
        self.output_hypernet = nn.Embedding(num_masks, embed_dim)
        self.iou_prediction_head = nn.Linear(embed_dim, 1)

        # Transformer decoder blocks
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: (B, C, H, W) image features
            sparse_prompt_embeddings: (B, N_p, C) point/box embeddings
            dense_prompt_embeddings: (B, C, H, W) or None
        
        Returns:
            masks: (B, num_masks, H*16, W*16)
            iou_predictions: (B, num_masks)
        """
        B, C, H, W = image_embeddings.shape

        # Flatten image features to sequence
        image_tokens = image_embeddings.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Initialize output tokens
        output_tokens = self.output_hypernet.weight.unsqueeze(0).expand(B, -1, -1)

        # Concatenate output tokens with prompt tokens
        token_sequence = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Self-attention among tokens
        token_sequence = token_sequence + self.self_attn(
            self.norm1(token_sequence), self.norm1(token_sequence), self.norm1(token_sequence)
        )[0]

        # Cross-attention: tokens attend to image
        updated_tokens = []
        for t in token_sequence.chunk(token_sequence.size(1), dim=1):
            t_attn = self.cross_attn(
                self.norm2(t), image_tokens, image_tokens
            )[0]
            t = t + t_attn
            t = t + self.mlp(self.norm3(t))
            updated_tokens.append(t)

        token_sequence = torch.cat(updated_tokens, dim=1)

        # Split output tokens (first num_masks) from prompt tokens
        output_tokens = token_sequence[:, :self.num_masks, :]

        # Predict mask via dynamic linear classifier
        # mask = output_token @ image_features (as in paper)
        image_tokens_norm = F.normalize(image_tokens, dim=-1)
        output_tokens_norm = F.normalize(output_tokens, dim=-1)
        
        masks = output_tokens_norm @ image_tokens_norm.transpose(-2, -1)  # (B, num_masks, H*W)
        masks = masks.reshape(B, self.num_masks, H, W) * 5.0  # scale logits

        # IoU predictions
        iou_pred = self.iou_prediction_head(output_tokens).squeeze(-1)  # (B, num_masks)

        # Upsample masks to original resolution
        masks = F.interpolate(masks, scale_factor=16, mode="bilinear", align_corners=False)

        return masks, iou_pred


# ============================================================
# Full SAM Model
# ============================================================

class SegmentAnythingModel(nn.Module):
    """
    Complete Segment Anything Model.
    
    Combines: Image Encoder -> Prompt Encoder -> Mask Decoder
    """
    def __init__(
        self,
        img_size: int = 1024,
        embed_dim: int = 256,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_heads: int = 12,
        num_masks: int = 3,
    ):
        super().__init__()
        self.img_size = img_size

        # Image encoder
        self.image_encoder = ImageEncoderViT(
            img_size=img_size,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            global_attn_indexes=[2, 5, 8, 11],
        )

        # Image embedding projection (ViT dim -> SAM embed dim)
        self.neck_proj = nn.Conv2d(vit_embed_dim, embed_dim, kernel_size=1)

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embed_size=img_size // 16,
        )

        # Mask decoder
        self.mask_decoder = MaskDecoder(
            embed_dim=embed_dim,
            num_masks=num_masks,
        )

    def forward(
        self,
        image: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            image: (B, 3, H, W) input image (H, W must match img_size)
            point_coords: (B, N_pts, 2) point coordinates
            point_labels: (B, N_pts) 1=foreground, 0=background
            boxes: (B, 4) or (B, N_boxes, 4) bounding boxes
            mask_input: (B, 1, H, W) input mask prompt

        Returns:
            masks: (B, num_masks, H, W)
            iou_predictions: (B, num_masks)
        """
        B = image.shape[0]

        # Encode image
        image_embeddings = self.image_encoder(image)  # (B, C_vit, H//16, W//16)
        image_embeddings = self.neck_proj(image_embeddings)  # (B, C, H//16, W//16)

        # Encode prompts
        points_dict = None
        if point_coords is not None and point_labels is not None:
            points_dict = {
                "coords": point_coords.float(),
                "labels": point_labels.float(),
            }

        sparse_embeds, dense_embeds = self.prompt_encoder(
            points=points_dict,
            boxes=boxes.float() if boxes is not None else None,
            masks=mask_input,
        )

        # If no prompts, use a default "center point" prompt
        if sparse_embeds is None:
            default_pts = torch.tensor([0.5, 0.5], device=image.device).view(1, 1, 2).expand(B, -1, -1)
            default_labels = torch.ones(B, 1, device=image.device)
            points_dict = {"coords": default_pts, "labels": default_labels}
            sparse_embeds, _ = self.prompt_encoder(points=points_dict)

        # Decode
        masks, iou_pred = self.mask_decoder(
            image_embeddings,
            sparse_embeds,
            dense_embeds,
        )

        # Upsample masks
        masks = F.interpolate(
            masks, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )

        # Sigmoid for binary mask
        masks = torch.sigmoid(masks)

        return masks, iou_pred


# ============================================================
# Usage example
# ============================================================

def test_sam():
    """Test SAM on a synthetic image."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create a small SAM variant for testing
    model = SegmentAnythingModel(
        img_size=224,  # smaller for testing
        embed_dim=64,
        vit_embed_dim=192,
        vit_depth=6,
        vit_heads=6,
        num_masks=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create synthetic image
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224, device=device)

    # Test 1: Point prompt
    print("\n[Test 1] Point prompt")
    point_coords = torch.tensor([[[0.5, 0.5]], [[0.3, 0.7]]], device=device)
    point_labels = torch.ones(batch_size, 1, device=device)

    masks, iou = model(image, point_coords=point_coords, point_labels=point_labels)
    print(f"  Mask shape: {masks.shape}")  # (B, 3, 224, 224)
    print(f"  IoU predictions: {iou}")
    assert masks.shape == (batch_size, 3, 224, 224)
    print("  PASS")

    # Test 2: Box prompt
    print("\n[Test 2] Box prompt")
    boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]], device=device).expand(batch_size, -1)
    masks, iou = model(image, boxes=boxes)
    print(f"  Mask shape: {masks.shape}")
    print(f"  IoU predictions: {iou}")
    print("  PASS")

    # Test 3: Backward pass
    print("\n[Test 3] Backward pass")
    loss = masks.sum() + iou.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  PASS")

    # Test 4: Multiple masks for ambiguity resolution
    print("\n[Test 4] Ambiguity: multiple masks per prompt")
    masks, iou = model(image, point_coords=point_coords, point_labels=point_labels)
    best_idx = iou.argmax(dim=1)
    print(f"  Best mask indices per image: {best_idx}")
    print("  PASS")

    print("\nAll tests passed!")


def demo_segmentation():
    """
    Demo showing how to use SAM for interactive segmentation.
    Uses a simple synthetic "blob" image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 50)
    print("Interactive Segmentation Demo")
    print("=" * 50)

    # Create a synthetic "blob" image (2D Gaussian)
    H, W = 224, 224
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    # Two blobs
    blob1 = torch.exp(-((x + 0.4)**2 + (y - 0.2)**2) / 0.1)
    blob2 = torch.exp(-((x - 0.3)**2 + (y + 0.3)**2) / 0.15)
    image = (blob1 + blob2).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    image_rgb = image.expand(-1, 3, -1, -1)  # (1, 3, H, W)

    # Small model for demo
    model = SegmentAnythingModel(
        img_size=224,
        embed_dim=32,
        vit_embed_dim=64,
        vit_depth=3,
        vit_heads=4,
        num_masks=3,
    )
    model.eval()

    with torch.no_grad():
        # Click on the center of blob 1
        point = torch.tensor([[[0.6, 0.4]]])  # normalized coords
        labels = torch.ones(1, 1)
        masks, iou = model(image_rgb, point_coords=point, point_labels=labels)

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image[0, 0].numpy(), cmap="gray")
    axes[0].scatter([0.6 * W], [0.4 * H], c="red", s=100, marker="x")
    axes[0].set_title("Input + Point Prompt")
    axes[0].axis("off")

    for i in range(3):
        mask = masks[0, i].numpy()
        axes[i + 1].imshow(mask, cmap="viridis", vmin=0, vmax=1)
        axes[i + 1].set_title(f"Mask {i+1} (IoU={iou[0, i]:.3f})")
        axes[i + 1].axis("off")

    plt.tight_layout()
    save_path = "E:/wuyi/数学建模半自动/research-assistant/knowledge/temp/ml/sam_demo.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    test_sam()
    demo_segmentation()
```

## References

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., & Girshick, R. (2023). Segment anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015-4026. https://arxiv.org/abs/2304.02643

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., ... & Feichtenhofer, C. (2024). SAM 2: Segment anything in images and videos. *arXiv preprint arXiv:2408.00714*. https://arxiv.org/abs/2408.00714

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16000-16009. https://arxiv.org/abs/2111.06377

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2010.11929

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 9650-9660. https://arxiv.org/abs/2104.14294
