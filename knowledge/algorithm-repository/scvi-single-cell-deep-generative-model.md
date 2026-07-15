# scVI and scANVI: Deep Generative Models for Single-Cell Transcriptomics

**Source**: Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15, 1053--1058. https://doi.org/10.1038/s41592-018-0229-2

**Updated Reference for benchmarking**: Luecken, M. D., Buttner, M., Chaichoompu, K., Danese, A., Interlandi, M., Mueller, M. F., Strobl, D. C., Zappia, L., Dugas, M., Colome-Tatche, M., & Theis, F. J. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nature Methods*, 19, 41--50. https://doi.org/10.1038/s41592-021-01336-8

**Category**: Bioinformatics / Single-Cell Genomics

## Biological / Computational Problem

Single-cell RNA-seq (scRNA-seq) data are characterized by high sparsity (dropout), technical batch effects, and varying sequencing depth across cells. The key computational challenge is to learn a **denoised, batch-corrected, low-dimensional representation** of each cell that captures biological variation while removing technical noise.

- **Input data**: Raw count matrix (cells x genes) with optional batch labels and cell-type annotations
- **Output**: Low-dimensional latent representation (cells x latent_dim), denoised expression values, batch-corrected expression matrix

## Mathematical / Computational Model

### scVI: Variational Autoencoder for scRNA-seq

scVI uses a **hierarchical variational autoencoder (VAE)** tailored for the statistical properties of scRNA-seq data.

### Generative Process

For cell $n$ with total observed UMI count $N_n$ and batch assignment $s_n$, the generative process is:

1. Sample latent variable $z_n \sim \mathcal{N}(0, \mathbf{I})$ (low-dimensional cell state)
2. Sample library size scaling $l_n \sim \text{LogNormal}(l_\mu^\top z_n, l_\sigma^\top z_n)$
3. For each gene $g$, the observed expression $x_{ng}$ is:

$$x_{ng} \sim \text{NegativeBinomial}(\mu_{ng}, \theta_g)$$

where:
- $\mu_{ng} = l_n \cdot f_g(z_n, s_n)$ is the mean expression
- $f_g(z_n, s_n) = \text{softmax}(\mathbf{W}_g \cdot h(z_n, s_n) + \mathbf{b}_g)$ is a neural network decoder
- $\theta_g$ is the gene-specific dispersion (inverse overdispersion)
- $l_n$ is the cell-specific scale factor (library size)

The Negative Binomial (NB) distribution captures the overdispersed count nature of scRNA-seq:

$$p(x_{ng} \mid \mu_{ng}, \theta_g) = \frac{\Gamma(x_{ng} + \theta_g)}{\Gamma(\theta_g)\Gamma(x_{ng}+1)} \left(\frac{\mu_{ng}}{\mu_{ng}+\theta_g}\right)^{x_{ng}} \left(\frac{\theta_g}{\mu_{ng}+\theta_g}\right)^{\theta_g}$$

### Variational Inference

The posterior $p(z_n, l_n \mid x_n)$ is approximated by a variational distribution:

$$q(z_n, l_n \mid x_n) = q(z_n \mid x_n) \cdot q(l_n \mid x_n)$$

where both are parameterized by neural networks (encoder):

$$q(z_n \mid x_n) = \mathcal{N}(\mu_z(x_n), \sigma_z^2(x_n) \mathbf{I})$$
$$q(l_n \mid x_n) = \text{LogNormal}(\mu_l(x_n), \sigma_l^2(x_n))$$

The model is trained by maximizing the ELBO:

$$\mathcal{L} = \mathbb{E}_{q}[\log p(x_n \mid z_n, l_n, s_n)] - \text{KL}[q(z_n \mid x_n) \| p(z_n)] - \text{KL}[q(l_n \mid x_n) \| p(l_n \mid z_n)]$$

### scANVI: Semi-Supervised Extension

scANVI extends scVI to incorporate **partial cell-type annotations** using a semi-supervised variational approach. It introduces a categorical latent variable $y_n$ representing cell type:

$$p(y_n) = \text{Categorical}(\pi)$$

The generative process adds a cell-type decoder:

1. Sample $y_n \sim \text{Categorical}(\pi)$ (cell type)
2. Sample $z_n \sim \mathcal{N}(\mu_y(y_n), \sigma_y^2(y_n))$ (type-conditional latent)
3. Sample $x_n \sim \text{NB}(l_n \cdot f_g(z_n, y_n, s_n), \theta_g)$

The semi-supervised objective combines a supervised term for labeled cells and an unsupervised term for all cells:

$$\mathcal{L}_{\text{scANVI}} = \sum_{n \in \text{labeled}} \log p(y_n \mid x_n) + \sum_{n} \mathcal{L}_{\text{ELBO}}(x_n)$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| NB distribution for counts | $x_{ng} \sim \text{NB}(\mu_{ng}, \theta_g)$ | Captures overdispersion; zero-inflated NB not generally needed |
| Batch effects are additive in latent space | $f(z_n, s_n)$ includes batch $s_n$ as input | Linear batch correction may miss complex batch-biology interactions |
| Gaussian prior on latent space | $z_n \sim \mathcal{N}(0, \mathbf{I})$ | Standard VAE assumption; may oversimplify complex topologies |
| Independent genes given $z_n$ | $p(x_n \mid z_n) = \prod_g p(x_{ng} \mid z_n)$ | Does not explicitly model gene-gene correlations beyond latent space |

## Applicable Scenarios

**When to use**:
- Batch correction and integration of multiple scRNA-seq datasets
- Dimensionality reduction for visualization and downstream analysis
- Denoising expression data (recovering dropouts)
- Data imputation across modalities (with extensions)
- Cell-type annotation (scANVI with reference data)

**When NOT to use**:
- Spatial transcriptomics without spatial-aware extensions (use Cell2location, SpaGCN instead)
- Very small datasets (<100 cells, VAE may overfit)
- Strong technical artifacts not captured by batch labels
- Gene-level differential expression testing (scVI provides corrected expression, but dedicated tools like DESeq2 may be more appropriate)

**Comparison**: In the Luecken et al. (2022) benchmark, scVI ranked among the top 4 methods for scRNA-seq integration (alongside scANVI, Scanorama, scGen), excelling particularly at complex integration tasks with strong batch effects.

## Implementation Details

- **Key parameters**: `n_latent` (10--30, default 10), `n_layers` (1--2), `n_hidden` (128--256), `dropout_rate` (0.1--0.3)
- **Computational requirements**: GPU recommended for >50K cells; training time ~5--30 min on GPU for 100K cells
- **Preprocessing**:
  - Filter low-quality cells (min genes, max mitochondrial)
  - Select highly variable genes (1,000--5,000)
  - Raw counts required (not log-normalized)
  - Batch labels as categorical covariates

## Python Implementation

```python
"""
Minimal implementation of scVI and scANVI-style deep generative models
for single-cell RNA-seq analysis.

This provides simplified VAE-based models with Negative Binomial likelihood
and batch correction capabilities, demonstrated on synthetic scRNA-seq data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


def simulate_batched_scrna(
    n_cells: int = 600,
    n_genes: int = 200,
    n_batches: int = 3,
    n_cell_types: int = 4,
    n_markers_per_type: int = 15,
    seed: int = 42,
):
    """
    Simulate scRNA-seq data with multiple batches and cell types.
    
    Each batch has slightly different technical characteristics
    (different capture efficiency, mean library size).
    
    Parameters
    ----------
    n_cells : int
    n_genes : int
    n_batches : int
    n_cell_types : int
    n_markers_per_type : int
    
    Returns
    -------
    dict with count matrix, batch labels, cell type labels.
    """
    rng = np.random.default_rng(seed)
    
    cells_per_batch = n_cells // n_batches
    
    count_matrix = np.zeros((n_cells, n_genes), dtype=np.float32)
    batch_labels = np.repeat(range(n_batches), cells_per_batch)
    cell_type_labels = np.zeros(n_cells, dtype=np.int64)
    
    # Cell-type-specific marker genes
    markers = {}
    for ct in range(n_cell_types):
        markers[ct] = rng.choice(n_genes, size=n_markers_per_type, replace=False)
    
    for b in range(n_batches):
        start = b * cells_per_batch
        end = start + cells_per_batch
        n_batch_cells = end - start
        
        # Batch-specific library size and efficiency
        batch_scale = rng.uniform(0.7, 1.3)  # technical batch effect
        batch_efficiency = rng.uniform(0.8, 1.2)
        
        # Assign cell types within batch
        types_in_batch = rng.choice(n_cell_types, size=n_batch_cells)
        cell_type_labels[start:end] = types_in_batch
        
        for i in range(start, end):
            ct = types_in_batch[i - start]
            
            # Base expression
            mu = rng.normal(-1, 0.5, size=n_genes)
            
            # Marker gene expression
            mu[markers[ct]] += rng.exponential(3.0, size=n_markers_per_type)
            
            # Batch effect on expression (multiplicative)
            batch_effect = np.where(
                rng.random(n_genes) < 0.05,  # 5% of genes have batch effects
                rng.normal(1, 0.3, n_genes),  # batch-specific shift
                1.0
            )
            mu = mu * batch_effect * batch_efficiency
            
            # Convert to counts
            lam = np.exp(mu) / np.exp(mu).sum() * 10000 * batch_scale
            count_matrix[i] = rng.poisson(lam)
    
    return {
        "counts": torch.tensor(count_matrix, dtype=torch.float32),
        "batch_labels": torch.tensor(batch_labels, dtype=torch.long),
        "cell_types": torch.tensor(cell_type_labels, dtype=torch.long),
        "gene_names": [f"GENE_{i}" for i in range(n_genes)],
        "markers": markers,
    }


class NegativeBinomialLoss(nn.Module):
    """
    Negative binomial log-likelihood for scRNA-seq count data.
    
    Parameterizes as mean (mu) and dispersion (theta).
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(
        self, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, G)
            Observed counts.
        mu : Tensor (B, G)
            Predicted mean.
        theta : Tensor (G,) or (B, G)
            Gene-level dispersion (inverse of overdispersion).
        
        Returns
        -------
        Tensor (,)
            Negative log-likelihood (scalar).
        """
        # Negative Binomial NB(mu, theta):
        # P(X=x) = C(x+theta-1, x) * (mu/(mu+theta))^x * (theta/(mu+theta))^theta
        mu = mu + self.eps
        theta = theta + self.eps
        
        # Log probability
        log_mu_theta = torch.log(mu + theta)
        
        # Log gamma terms
        log_gamma = (
            torch.lgamma(x + theta) 
            - torch.lgamma(theta) 
            - torch.lgamma(x + 1)
        )
        
        log_prob = (
            log_gamma
            + x * (torch.log(mu) - log_mu_theta)
            + theta * (torch.log(theta) - log_mu_theta)
        )
        
        return -log_prob.sum(dim=-1).mean()


class Encoder(nn.Module):
    """
    Variational encoder: maps counts to latent parameters.
    """
    
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_batch: int = 1,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input: log-transformed counts + batch one-hot
        layers = []
        in_dim = n_input + n_batch
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = n_hidden
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads
        self.z_mean = nn.Linear(n_hidden, n_latent)
        self.z_logvar = nn.Linear(n_hidden, n_latent)
        self.l_mean = nn.Linear(n_hidden, 1)  # library size log-mean
        self.l_logvar = nn.Linear(n_hidden, 1)  # library size log-var
    
    def forward(
        self, x: torch.Tensor, batch_oh: torch.Tensor
    ) -> tuple:
        """
        Parameters
        ----------
        x : Tensor (B, G)
            Raw counts.
        batch_oh : Tensor (B, n_batch)
            One-hot batch encoding.
        
        Returns
        -------
        z_mu, z_logvar, l_mu, l_logvar
        """
        # Log transform input
        x_log = torch.log1p(x)
        
        # Concatenate with batch
        h = torch.cat([x_log, batch_oh], dim=-1)
        h = self.encoder(h)
        
        z_mu = self.z_mean(h)
        z_logvar = self.z_logvar(h)
        l_mu = self.l_mean(h).squeeze(-1)
        l_logvar = self.l_logvar(h).squeeze(-1).clamp(-5, 2)
        
        return z_mu, z_logvar, l_mu, l_logvar


class Decoder(nn.Module):
    """
    Decoder: reconstructs expression from latent + batch.
    """
    
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_batch: int = 1,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_input
        
        # Build decoder network
        layers = []
        in_dim = n_latent + n_batch
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = n_hidden
        
        self.decoder = nn.Sequential(*layers)
        
        # Gene-level output (softmax-normalized)
        self.gene_out = nn.Linear(n_hidden, n_input)
        
    def forward(
        self, z: torch.Tensor, batch_oh: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor (B, n_latent)
        batch_oh : Tensor (B, n_batch)
        
        Returns
        -------
        Tensor (B, G)
            Gene proportion logits.
        """
        h = torch.cat([z, batch_oh], dim=-1)
        h = self.decoder(h)
        return self.gene_out(h)


class scVI(nn.Module):
    """
    Simplified scVI model: VAE with Negative Binomial likelihood.
    
    Parameters
    ----------
    n_genes : int
    n_batch : int
        Number of batches.
    n_latent : int
        Latent dimensionality.
    n_hidden : int
        Hidden layer dimension.
    n_layers : int
        Number of hidden layers.
    """
    
    def __init__(
        self,
        n_genes: int,
        n_batch: int = 1,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_latent = n_latent
        
        self.encoder = Encoder(
            n_genes, n_hidden, n_latent, n_batch, n_layers, dropout
        )
        self.decoder = Decoder(
            n_genes, n_hidden, n_latent, n_batch, n_layers, dropout
        )
        
        # Gene dispersion parameters (log-transformed, shared across cells)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        
        self.nll_loss = NegativeBinomialLoss()
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> dict:
        """
        Forward pass with full loss computation.
        
        Parameters
        ----------
        x : Tensor (B, G)
            Raw count data.
        batch : Tensor (B,)
            Batch indices.
        
        Returns
        -------
        dict with loss, latent, predictions.
        """
        B = x.shape[0]
        n_batch = self.encoder.encoder[0].in_features - self.n_genes
        
        # One-hot batch encoding
        batch_oh = F.one_hot(batch, num_classes=n_batch).float()
        
        # Encode
        z_mu, z_logvar, l_mu, l_logvar = self.encoder(x, batch_oh)
        
        # Sample latent (reparameterization trick)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + eps * std
        
        # Sample library size
        l_std = torch.exp(0.5 * l_logvar)
        l_eps = torch.randn_like(l_std)
        l = l_mu + l_eps * l_std  # log library size
        l_scaled = torch.exp(l)  # linear scale
        
        # Decode
        gene_logits = self.decoder(z, batch_oh)
        
        # Compute mean expression (softmax + scale by library size)
        mu = F.softmax(gene_logits, dim=-1) * l_scaled.unsqueeze(-1)  # (B, G)
        
        # Dispersion
        theta = torch.exp(self.log_theta)  # (G,)
        
        # Reconstruction loss
        recon_loss = self.nll_loss(x, mu, theta)
        
        # KL divergence on z
        kl_z = -0.5 * torch.mean(
            1 + z_logvar - z_mu ** 2 - torch.exp(z_logvar)
        )
        
        # KL divergence on l (approximate)
        kl_l = -0.5 * torch.mean(1 + l_logvar - l_mu ** 2 - torch.exp(l_logvar))
        
        total_loss = recon_loss + kl_z + kl_l
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss.item(),
            "kl_z": kl_z.item(),
            "kl_l": kl_l.item(),
            "z": z,
            "z_mu": z_mu,
            "mu": mu,
        }
    
    @torch.no_grad()
    def get_latent(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract latent representation (for downstream analysis).
        
        Parameters
        ----------
        x : Tensor (B, G)
        batch : Tensor (B,)
        
        Returns
        -------
        Tensor (B, n_latent)
        """
        n_batch = self.encoder.encoder[0].in_features - self.n_genes
        batch_oh = F.one_hot(batch, num_classes=n_batch).float()
        z_mu, _, _, _ = self.encoder(x, batch_oh)
        return z_mu
    
    @torch.no_grad()
    def get_denoised(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Get denoised (batch-corrected) expression.
        
        Parameters
        ----------
        x : Tensor (B, G)
        batch : Tensor (B,)
        
        Returns
        -------
        Tensor (B, G)
        """
        n_batch = self.encoder.encoder[0].in_features - self.n_genes
        batch_oh = F.one_hot(batch, num_classes=n_batch).float()
        
        z_mu, _, l_mu, _ = self.encoder(x, batch_oh)
        gene_logits = self.decoder(z_mu, batch_oh)
        l_scaled = torch.exp(l_mu)
        mu = F.softmax(gene_logits, dim=-1) * l_scaled.unsqueeze(-1)
        
        return mu


class scANVI(nn.Module):
    """
    Simplified scANVI: semi-supervised extension of scVI.
    
    Adds a classifier on top of scVI for cell-type annotation.
    """
    
    def __init__(
        self,
        scvi_model: scVI,
        n_cell_types: int,
        n_hidden: int = 64,
    ):
        super().__init__()
        self.scvi = scvi_model
        self.n_cell_types = n_cell_types
        
        # Cell-type classifier
        self.classifier = nn.Sequential(
            nn.Linear(scvi_model.n_latent, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, n_cell_types),
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Parameters
        ----------
        x : Tensor (B, G)
        batch : Tensor (B,)
        labels : Tensor (B,) or None
            Cell-type labels (for supervised loss).
        
        Returns
        -------
        dict
        """
        # Get scVI latent and VAE loss
        scvi_out = self.scvi(x, batch)
        z = scvi_out["z"]
        
        # Classifier predictions
        logits = self.classifier(z)
        probs = F.softmax(logits, dim=-1)
        
        result = {
            **scvi_out,
            "class_logits": logits,
            "class_probs": probs,
        }
        
        # Supervised loss (if labels provided)
        if labels is not None:
            supervised_loss = self.ce_loss(logits, labels)
            result["supervised_loss"] = supervised_loss.item()
            
            # Combined loss: ELBO + classification loss
            alpha = 0.5  # weighting between VAE and classifier
            result["loss"] = scvi_out["loss"] + alpha * supervised_loss
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            result["accuracy"] = (preds == labels).float().mean().item()
        
        return result
    
    @torch.no_grad()
    def predict_cell_types(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict cell types for new data.
        
        Parameters
        ----------
        x : Tensor (B, G)
        batch : Tensor (B,)
        
        Returns
        -------
        Tensor (B,) predicted cell-type indices.
        """
        z = self.scvi.get_latent(x, batch)
        logits = self.classifier(z)
        return logits.argmax(dim=-1)


class SCModelTrainer:
    """
    Trainer for scVI / scANVI models.
    """
    
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_step(self, x: torch.Tensor, batch: torch.Tensor,
                   labels: torch.Tensor = None) -> float:
        """
        Single training step.
        
        Parameters
        ----------
        x : Tensor (B, G)
        batch : Tensor (B,)
        labels : Tensor (B,) or None
        
        Returns
        -------
        float loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        if isinstance(self.model, scANVI):
            out = self.model(x, batch, labels)
        else:
            out = self.model(x, batch)
        
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()


# ============================================================================
# Complete usage example
# ============================================================================

def compute_batch_mixing(
    latent: torch.Tensor, batch_labels: torch.Tensor
) -> float:
    """
    Compute batch mixing metric (mean silhouette score across batches).
    
    Higher = better mixing (batches are well-integrated).
    """
    from sklearn.metrics import silhouette_score
    
    n_batches = batch_labels.max().item() + 1
    scores = []
    
    for b in range(n_batches):
        mask = (batch_labels == b)
        if mask.sum() > 1:
            # Silhouette for this batch vs others
            batch_mask = (batch_labels == b).numpy()
            s = silhouette_score(
                latent.numpy(), 
                batch_mask.astype(int),
                metric="euclidean"
            )
            scores.append(s)
    
    # Lower silhouette = better mixing (we negate for interpretability)
    return -np.mean(scores) if scores else 0.0


def compute_biological_conservation(
    latent: torch.Tensor, cell_types: torch.Tensor
) -> float:
    """
    Compute biological conservation (silhouette by cell type).
    
    Higher = better separation of cell types.
    """
    from sklearn.metrics import silhouette_score
    
    types_np = cell_types.numpy()
    latent_np = latent.numpy()
    
    if len(np.unique(types_np)) > 1:
        s = silhouette_score(latent_np, types_np, metric="euclidean")
        return s
    return 0.0


def main():
    """
    Run a complete scVI/scANVI-style analysis on simulated scRNA-seq data.
    """
    print("=" * 60)
    print("scVI / scANVI: Deep Generative Models for scRNA-seq")
    print("=" * 60)
    
    # --- 1. Simulate batched scRNA-seq data ---
    print("\n[1] Simulating multi-batch scRNA-seq data...")
    data = simulate_batched_scrna(
        n_cells=500,
        n_genes=150,
        n_batches=3,
        n_cell_types=4,
        n_markers_per_type=12,
    )
    
    counts = data["counts"]
    batch_labels = data["batch_labels"]
    cell_types = data["cell_types"]
    
    print(f"    Cells: {counts.shape[0]}")
    print(f"    Genes: {counts.shape[1]}")
    print(f"    Batches: {batch_labels.max().item() + 1}")
    print(f"    Cell types: {cell_types.max().item() + 1}")
    print(f"    Batch distribution: {torch.bincount(batch_labels).tolist()}")
    
    # --- 2. Create and train scVI model ---
    print("\n[2] Creating scVI model...")
    scvi_model = scVI(
        n_genes=counts.shape[1],
        n_batch=batch_labels.max().item() + 1,
        n_latent=10,
        n_hidden=64,
        n_layers=2,
        dropout=0.1,
    )
    
    n_params = sum(p.numel() for p in scvi_model.parameters())
    print(f"    scVI parameters: {n_params:,}")
    
    print("\n[3] Training scVI (unsupervised)...")
    trainer_scvi = SCModelTrainer(scvi_model, lr=1e-3)
    
    n_epochs = 50
    for epoch in range(n_epochs):
        loss = trainer_scvi.train_step(counts, batch_labels)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, loss: {loss:.2f}")
    
    # --- 4. Evaluate scVI latent space ---
    print("\n[4] Evaluating scVI latent space...")
    with torch.no_grad():
        z_scvi = scvi_model.get_latent(counts, batch_labels)
    
    mixing = compute_batch_mixing(z_scvi, batch_labels)
    bio_cons = compute_biological_conservation(z_scvi, cell_types)
    print(f"    Batch mixing score: {mixing:.3f} (higher = better integrated)")
    print(f"    Biological conservation: {bio_cons:.3f} (higher = better separated)")
    
    # --- 5. Create and train scANVI ---
    print("\n[5] Creating scANVI model (semi-supervised)...")
    # Use 30% of cells with labels
    n_labeled = int(counts.shape[0] * 0.3)
    label_mask = torch.zeros(counts.shape[0], dtype=torch.bool)
    label_mask[:n_labeled] = True
    
    scanvi_model = scANVI(scvi_model, n_cell_types=4, n_hidden=32)
    trainer_scanvi = SCModelTrainer(scanvi_model, lr=5e-4)
    
    print("    Training scANVI with 30% labeled cells...")
    for epoch in range(30):
        # Use batch where we have labels
        labeled_idx = torch.where(label_mask)[0]
        x_labeled = counts[labeled_idx]
        b_labeled = batch_labels[labeled_idx]
        y_labeled = cell_types[labeled_idx]
        
        # Train with labeled + unlabeled
        out = scanvi_model(x_labeled, b_labeled, y_labeled)
        loss = out["loss"]
        
        # Update on the full batch (unsupervised only)
        trainer_scanvi.optimizer.zero_grad()
        loss.backward()
        # Also do unsupervised on unlabeled data
        unlabeled_idx = torch.where(~label_mask)[0]
        if len(unlabeled_idx) > 0:
            x_unlabeled = counts[unlabeled_idx]
            b_unlabeled = batch_labels[unlabeled_idx]
            out2 = scanvi_model.scvi(x_unlabeled, b_unlabeled)
            unsupervised_loss = out2["loss"]
            unsupervised_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(scanvi_model.parameters(), 1.0)
        trainer_scanvi.optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            acc = out.get("accuracy", 0)
            print(f"    Epoch {epoch+1}/30, total loss: {loss.item():.2f}, "
                  f"labeled acc: {acc:.3f}")
    
    # --- 6. Evaluate scANVI cell-type prediction ---
    print("\n[6] Evaluating scANVI cell-type prediction on all cells...")
    with torch.no_grad():
        predictions = scanvi_model.predict_cell_types(counts, batch_labels)
    
    accuracy = (predictions == cell_types).float().mean().item()
    print(f"    Cell-type classification accuracy: {accuracy:.3f}")
    
    # Per-type accuracy
    for ct in range(4):
        mask = (cell_types == ct)
        if mask.sum() > 0:
            ct_acc = (predictions[mask] == cell_types[mask]).float().mean().item()
            print(f"    Type {ct}: accuracy = {ct_acc:.3f} "
                  f"(n = {mask.sum().item()})")
    
    # --- 7. Compare denoised vs. raw ---
    print("\n[7] Denoising effect on marker genes...")
    denoised = scvi_model.get_denoised(counts, batch_labels)
    
    # For a specific marker gene, show raw vs. denoised
    ct0_markers = data["markers"][0]
    ct0_cells = (cell_types == 0)
    
    for g in ct0_markers[:3]:
        raw_mean = counts[ct0_cells, g].mean().item()
        denoised_mean = denoised[ct0_cells, g].mean().item()
        print(f"    Gene {g}: raw mean = {raw_mean:.1f}, "
              f"denoised mean = {denoised_mean:.1f}")
    
    print("\n" + "=" * 60)
    print("scVI/scANVI demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## References

Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15, 1053--1058. https://doi.org/10.1038/s41592-018-0229-2

Luecken, M. D., Buttner, M., Chaichoompu, K., et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nature Methods*, 19, 41--50. https://doi.org/10.1038/s41592-021-01336-8

Xu, C., Lopez, R., Mehlman, E., Regier, J., Jordan, M. I., & Yosef, N. (2021). Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models. *Molecular Systems Biology*, 17(1), e9620. https://doi.org/10.15252/msb.20209620

Gayoso, A., Lopez, R., Xing, G., et al. (2022). A Python library for probabilistic analysis of single-cell omics data. *Nature Biotechnology*, 40, 163--166. https://doi.org/10.1038/s41587-021-01206-w

Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *arXiv*, 1312.6114. https://doi.org/10.48550/arXiv.1312.6114
