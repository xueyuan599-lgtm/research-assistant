# Geneformer: Transfer Learning for Single-Cell Network Biology

**Source**: Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., Mantineo, H., Brydon, E. M., Zeng, Z., Liu, X. S., & Ellinor, P. T. (2023). Transfer learning enables predictions in network biology. *Nature*, 618, 616--624. https://doi.org/10.1038/s41586-023-06139-9

**Category**: Bioinformatics / Single-Cell Genomics

## Biological / Computational Problem

Single-cell RNA-seq data suffer from limited sample sizes, especially for rare diseases or cell types. Most deep learning models in single-cell biology require large training sets for each new task. The key challenge is to develop a **pretrained foundation model** that can transfer knowledge across diverse single-cell datasets and enable accurate predictions with limited fine-tuning data.

- **Input data**: Single-cell gene expression matrices (cells x ~20,000 genes). Genes are rank-ordered by expression within each cell.
- **Output**: Fine-tuned predictions for diverse downstream tasks (cell-type annotation, gene network inference, perturbation response, disease gene discovery)

## Mathematical / Computational Model

### Core Architecture

Geneformer is a **6-layer transformer encoder** (modified BERT-style) applied to single-cell transcriptomes:

```
Input: Rank-ordered gene tokens [2543, 74, 18901, ...]
  └─> Embedding: Gene token embedding + Position embedding
       └─> 6x Transformer encoder blocks
            └─> Output: Gene-level contextualized embeddings
                 └─> Cell-level summary (mean pooling)
                       └─> Downstream task heads
```

### Rank-Value Encoding

Geneformer uses a unique encoding strategy: instead of raw expression values, each gene is represented by its **expression rank** within the cell:

$$\text{rank}(g, c) = \text{position of gene } g \text{ when genes in cell } c \text{ are sorted by expression (descending)}$$

Key properties:
- Top 2,048 expressed genes are kept per cell (out of ~20,000)
- Rank encoding is robust to batch effects (relative expression matters, not absolute values)
- The rank order provides information about gene importance within each cell

### Masked Language Modeling Pretraining

Geneformer is pretrained using a **masked gene expression objective** (analogous to BERT's masked language modeling):

$$\mathcal{L}_{\text{pretrain}} = -\mathbb{E}_{c \sim \mathcal{D}} \sum_{g \in \mathcal{M}_c} \log p(\text{gene}_g \mid \mathbf{r}_{c \setminus \mathcal{M}_c})$$

where:
- $\mathcal{D}$ is the training corpus of $\sim$30 million cells
- $\mathcal{M}_c$ is the set of masked genes (15% of the 2,048 input genes per cell)
- $\mathbf{r}_{c \setminus \mathcal{M}_c}$ is the rank encoding of unmasked genes
- The model predicts which gene was masked given the remaining context

### Attention Mechanism for Gene Networks

The self-attention weights between gene tokens can be interpreted as **gene-gene interaction strengths**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For a given cell type, the attention matrix across layers encodes regulatory relationships. The **context score** for a gene pair $(g_i, g_j)$ is:

$$\text{CS}(g_i, g_j) = \frac{1}{L} \sum_{\ell=1}^{L} \frac{1}{H} \sum_{h=1}^{H} A_{i,j}^{(\ell, h)}$$

where $A_{i,j}^{(\ell, h)}$ is the attention weight from gene $i$ to gene $j$ in layer $\ell$, head $h$.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Rank encoding captures biology | $\text{rank}(g,c)$ is sufficient; absolute values discarded | Loses information about fold-change magnitude and dropout patterns |
| Top 2,048 genes are sufficient | $\approx$10% of transcriptome per cell | Rare genes or lowly expressed transcription factors may be excluded |
| Transformer captures gene networks | Attention weights $\propto$ regulatory interaction strength | Correlation bias; may capture co-expression rather than causal regulation |
| Pretraining on healthy tissue generalizes | $p(\text{disease gene} \mid \text{healthy})$ transferable | May fail for cell states absent from training data |

## Applicable Scenarios

**When to use**:
- Limited training data (fine-tune with as few as 10--100 labeled cells)
- Discovering novel therapeutic targets from small patient cohorts
- Predicting perturbation effects (TF dosage sensitivity, CRISPR perturbations)
- Cell-type annotation with transfer across tissues and conditions

**When NOT to use**:
- Quantitative expression prediction (rank encoding discards magnitude)
- Cells with very low sequencing depth ($<$500 genes detected, rank encoding becomes noisy)
- Situations where batch effects dominate biological signal (pretraining may encode technical artifacts)
- High-stakes clinical predictions without validation (attention-based gene network interpretation requires caution)

**Comparison**: Geneformer excels in low-data regimes compared to scVI, scANVI, and conventional ML. A 2025 *Nature Methods* benchmark found that foundation models (including Geneformer and scGPT) did not consistently outperform simple linear baselines for perturbation prediction, though Geneformer remains strong for cell-type annotation and network inference.

## Implementation Details

- **Key parameters**: 6 transformer layers, 256 embedding dim, 4 attention heads, gelu activation, 0.15 masking rate
- **Computational requirements**: Pretraining on 30M cells requires 4--8 GPUs for 2--3 weeks; fine-tuning is fast (minutes to hours on a single GPU)
- **Preprocessing**:
  - Standard scRNA-seq filtering (minimum genes per cell, minimum cells per gene)
  - Normalize to counts per million (CPM), then rank genes by CPM within each cell
  - Select top 2,048 ranked genes; pad shorter cells with zero token
  - Vocabulary: 60,774 unique gene tokens (Ensembl gene IDs)

## Python Implementation

```python
"""
Minimal implementation of Geneformer-style single-cell foundation model.

This provides a simplified transformer-based model with rank-value gene encoding
and masked gene prediction pretraining, demonstrated on synthetic scRNA-seq data.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


def simulate_scrna_data(
    n_cells: int = 500,
    n_genes: int = 5000,
    n_cell_types: int = 4,
    n_per_type: int = None,
    n_input_genes: int = 2048,
    seed: int = 42,
):
    """
    Simulate single-cell RNA-seq data with cell-type-specific expression programs.
    
    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes in vocabulary.
    n_cell_types : int
        Number of distinct cell types.
    n_input_genes : int
        Number of top-ranked genes to keep per cell.
    
    Returns
    -------
    rank_matrix : Tensor (n_cells x n_input_genes)
        Rank-encoded gene tokens.
    cell_type_labels : Tensor (n_cells,)
        Cell type annotations.
    gene_ids : list
        Gene IDs making up the vocabulary.
    """
    rng = np.random.default_rng(seed)
    
    # Define cell-type-specific gene programs
    # Each cell type has ~50 marker genes with high expression
    n_markers = 50
    type_programs = {}
    
    for ct in range(n_cell_types):
        markers = rng.choice(n_genes, size=n_markers, replace=False)
        type_programs[ct] = markers
    
    # Generate expression data
    count_matrix = np.zeros((n_cells, n_genes), dtype=np.float32)
    cell_type_labels = np.repeat(range(n_cell_types), n_cells // n_cell_types)[:n_cells]
    
    for i in range(n_cells):
        ct = cell_type_labels[i]
        
        # Base expression level (log-normal)
        mu = rng.normal(-1, 0.5, size=n_genes)
        
        # Boost marker genes
        mu[type_programs[ct]] += rng.exponential(2.0, size=n_markers)
        
        # Add some noise for other cell types' markers
        for other_ct in range(n_cell_types):
            if other_ct != ct:
                noise_boost = rng.exponential(0.3, size=n_markers)
                mu[type_programs[other_ct]] += noise_boost
        
        # Convert to counts (Poisson)
        lam = np.exp(mu) * 1000 / np.exp(mu).sum() * 10000  # ~10K library size
        count_matrix[i] = rng.poisson(lam)
    
    # Rank transform: for each cell, rank genes by expression
    rank_matrix = np.zeros((n_cells, n_input_genes), dtype=np.int64)
    for i in range(n_cells):
        # Get gene indices sorted by descending expression
        sorted_idx = np.argsort(count_matrix[i])[::-1]
        top_genes = sorted_idx[:n_input_genes]
        rank_matrix[i] = top_genes
    
    # Gene vocabulary (0 = padding, 1..n_genes = actual genes)
    gene_ids = [f"GENE_{i}" for i in range(n_genes)]
    
    return (
        torch.tensor(rank_matrix, dtype=torch.long),
        torch.tensor(cell_type_labels, dtype=torch.long),
        gene_ids,
    )


class GeneEmbedding(nn.Module):
    """
    Gene token embedding with position encoding.
    
    Maps gene token IDs to dense vectors.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        max_position: int = 2048,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size + 1, d_model, padding_idx=padding_idx
        )
        self.position_embedding = nn.Embedding(max_position, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, L)
            Gene token IDs (ranked by expression).
        
        Returns
        -------
        Tensor (B, L, d_model)
            Token + position embeddings.
        """
        B, L = x.shape
        token_emb = self.token_embedding(x)
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)
        
        return token_emb + pos_emb


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with pre-norm and gelu activation.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SimplifiedGeneformer(nn.Module):
    """
    Simplified Geneformer model for single-cell transcriptomics.
    
    A BERT-style transformer with gene token input.
    
    Parameters
    ----------
    vocab_size : int
        Number of unique genes.
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    d_ff : int
        Feed-forward hidden dimension.
    max_seq_len : int
        Maximum number of input genes.
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        self.embedding = GeneEmbedding(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size + 1)  # +1 for padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, L)
            Gene token IDs.
        
        Returns
        -------
        Tensor (B, L, vocab_size+1)
            Logits for masked gene prediction.
        """
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = self.output_proj(h)
        return logits
    
    def get_cell_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract cell-level embedding by mean pooling.
        
        Parameters
        ----------
        x : Tensor (B, L)
            Gene token IDs.
        
        Returns
        -------
        Tensor (B, d_model)
            Cell-level embedding.
        """
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        # Mean pool over gene dimension (excluding padding)
        mask = (x != 0).float().unsqueeze(-1)
        cell_emb = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return cell_emb
    
    def get_gene_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for gene network inference.
        
        Parameters
        ----------
        x : Tensor (B, L)
            Gene token IDs.
        
        Returns
        -------
        Tensor (n_layers, n_heads, B, L, L)
            Attention weights.
        """
        h = self.embedding(x)
        attention_weights = []
        
        for layer in self.layers:
            norm_h = layer.norm1(h)
            with torch.no_grad():
                attn_out, attn_w = layer.attention(
                    norm_h, norm_h, norm_h, average_attn_weights=False
                )
            attention_weights.append(attn_w.detach().cpu())
            h = h + attn_out
            h = h + layer.ffn(layer.norm2(h))
        
        return torch.stack(attention_weights, dim=0)


class Pretrainer:
    """
    Masked gene prediction pretraining for Geneformer.
    """
    
    def __init__(self, model: SimplifiedGeneformer, lr: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
    def _mask_input(self, x: torch.Tensor, mask_prob: float = 0.15) -> tuple:
        """
        Apply masking to input gene tokens.
        
        Parameters
        ----------
        x : Tensor (B, L)
            Input gene tokens.
        mask_prob : float
            Probability of masking each token.
        
        Returns
        -------
        masked_x : Tensor (B, L)
            Input with some tokens masked (replaced with 0).
        mask : Tensor (B, L)
            Boolean mask indicating masked positions.
        labels : Tensor (B, L)
            Original token IDs at masked positions (-100 elsewhere).
        """
        B, L = x.shape
        labels = x.clone()
        
        # 80% MASK (replace with 0/padding), 10% random, 10% unchanged
        prob = torch.rand(B, L, device=x.device)
        mask = prob < mask_prob
        
        # Random genes
        random_mask = (prob >= mask_prob) & (prob < mask_prob * 1.1)
        random_genes = torch.randint_like(x, 1, self.model.embedding.token_embedding.num_embeddings)
        x[random_mask] = random_genes[random_mask]
        
        # Set masked positions to 0 (padding/mask token)
        x[mask] = 0
        
        # Labels: -100 for non-masked (ignored in loss)
        labels[~mask] = -100
        
        return x, mask, labels
    
    def pretrain_step(self, x: torch.Tensor) -> float:
        """
        Single pretraining step with masked gene prediction.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x_masked, _, labels = self._mask_input(x.clone())
        logits = self.model(x_masked)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()


class FineTuner:
    """
    Fine-tune Geneformer for cell-type classification.
    """
    
    def __init__(self, model: SimplifiedGeneformer, n_classes: int, 
                 hidden_dim: int = 64, lr: float = 1e-4):
        self.model = model
        # Freeze pretrained layers, train only classifier head
        for param in model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(model.embedding.token_embedding.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )
        self.optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        self.classifier.train()
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            cell_emb = self.model.get_cell_embedding(x)
        
        logits = self.classifier(cell_emb)
        loss = F.cross_entropy(logits, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.classifier.eval()
        
        cell_emb = self.model.get_cell_embedding(x)
        logits = self.classifier(cell_emb)
        return logits.argmax(dim=-1)


# ============================================================================
# Complete usage example
# ============================================================================

def main():
    """
    Run a complete Geneformer-style analysis pipeline.
    """
    print("=" * 60)
    print("Geneformer: Single-Cell Foundation Model Demo")
    print("=" * 60)
    
    # --- 1. Simulate single-cell data ---
    print("\n[1] Simulating single-cell transcriptomics data...")
    n_genes_vocab = 3000  # Vocabulary size (subset for demo)
    n_input_genes = 512   # Input genes per cell
    n_cell_types = 4
    
    token_matrix, cell_labels, gene_ids = simulate_scrna_data(
        n_cells=400,
        n_genes=n_genes_vocab,
        n_cell_types=n_cell_types,
        n_input_genes=n_input_genes,
    )
    print(f"    Cells: {token_matrix.shape[0]}")
    print(f"    Input genes per cell: {token_matrix.shape[1]}")
    print(f"    Gene vocabulary: {n_genes_vocab}")
    print(f"    Cell types: {n_cell_types}")
    
    # --- 2. Split into pretrain / fine-tune sets ---
    print("\n[2] Splitting data...")
    train_mask = torch.rand(400) < 0.5
    cell_types_for_label = cell_labels.clone()
    
    pretrain_data = token_matrix[train_mask]
    finetune_data = token_matrix[~train_mask]
    finetune_labels = cell_labels[~train_mask]
    
    print(f"    Pretrain set: {pretrain_data.shape[0]} cells")
    print(f"    Fine-tune set: {finetune_data.shape[0]} cells")
    
    # --- 3. Create and pretrain Geneformer ---
    print("\n[3] Initializing Geneformer model...")
    model = SimplifiedGeneformer(
        vocab_size=n_genes_vocab,
        d_model=64,
        n_heads=2,
        n_layers=3,
        d_ff=256,
        max_seq_len=n_input_genes,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {n_params:,}")
    
    print("\n[4] Pretraining with masked gene prediction...")
    pretrainer = Pretrainer(model, lr=1e-3)
    
    n_epochs = 30
    for epoch in range(n_epochs):
        loss = pretrainer.pretrain_step(pretrain_data)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, loss: {loss:.4f}")
    
    # --- 5. Fine-tune for cell-type classification ---
    print("\n[5] Fine-tuning for cell-type classification...")
    finetuner = FineTuner(
        model, 
        n_classes=n_cell_types,
        hidden_dim=32,
    )
    
    n_ft_epochs = 20
    for epoch in range(n_ft_epochs):
        loss = finetuner.train_step(
            finetune_data[:100],  # Use only 100 cells for fine-tuning (few-shot)
            finetune_labels[:100],
        )
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{n_ft_epochs}, loss: {loss:.4f}")
    
    # --- 6. Evaluate ---
    print("\n[6] Evaluation...")
    predictions = finetuner.predict(finetune_data)
    accuracy = (predictions == finetune_labels).float().mean().item()
    print(f"    Fine-tuned test accuracy: {accuracy:.3f}")
    
    # --- 7. Analyze gene-gene attention ---
    print("\n[7] Gene network attention analysis...")
    with torch.no_grad():
        attn_weights = model.get_gene_attention(finetune_data[:10])
    
    print(f"    Attention tensor shape: {attn_weights.shape}")
    print(f"    (layers x heads x batch x seq x seq)")
    
    # Average across layers and heads
    mean_attn = attn_weights.mean(dim=(0, 1)).mean(dim=0)  # (L, L)
    top_pairs = mean_attn.fill_diagonal_(0).flatten().topk(5)
    print(f"    Top 5 gene-gene attention scores:")
    for idx, val in enumerate(top_pairs.values):
        gene_i = int(top_pairs.indices[idx] // mean_attn.shape[1])
        gene_j = int(top_pairs.indices[idx] % mean_attn.shape[1])
        if gene_i < len(gene_ids) and gene_j < len(gene_ids):
            print(f"      {gene_ids[gene_i]} <-> {gene_ids[gene_j]}: {val:.4f}")
    
    print("\n" + "=" * 60)
    print("Geneformer demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## References

Theodoris, C. V., Xiao, L., Chopra, A., et al. (2023). Transfer learning enables predictions in network biology. *Nature*, 618, 616--624. https://doi.org/10.1038/s41586-023-06139-9

Cui, H., Wang, C., Maan, H., et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21, 1470--1480. https://doi.org/10.1038/s41592-024-02201-0

Hao, Y., Hao, S., Andersen-Nissen, E., et al. (2021). Integrated analysis of multimodal single-cell data. *Cell*, 184(13), 3573--3587. https://doi.org/10.1016/j.cell.2021.04.048

Zappia, L., & Theis, F. J. (2021). Over 1000 tools reveal trends in the single-cell RNA-seq analysis landscape. *Genome Biology*, 22, 301. https://doi.org/10.1186/s13059-021-02519-4

Lotfollahi, M., Naghipourfar, M., Luecken, M. D., et al. (2022). Mapping single-cell data to reference atlases by transfer learning. *Nature Biotechnology*, 40, 121--130. https://doi.org/10.1038/s41587-021-01001-7
