# Cell2location: Fine-Grained Cell Type Mapping in Spatial Transcriptomics

**Source**: Kleshchevnikov, V., Shmatko, A., Dann, E., Aivazidis, A., King, H. W., Li, T., Elmentaite, R., Lomakin, A., Kedlian, V., Gayoso, A., Jain, M. S., Park, J. S., Ramona, L., Tuck, E., Arutyunyan, A., Vento-Tormo, R., Gerstung, M., James, L., Stegle, O., & Bayraktar, O. A. (2022). Cell2location maps fine-grained cell types in spatial transcriptomics. *Nature Biotechnology*, 40(5), 661--671. https://doi.org/10.1038/s41587-021-01139-4

**Category**: Bioinformatics / Spatial Transcriptomics

## Biological / Computational Problem

Spatial transcriptomics technologies (e.g., Visium, Slide-seq, MERFISH) measure gene expression across tissue sections but at spatial spots that typically contain multiple cells. The key question is: **what cell types are present at each spatial location, and in what proportions?**

- **Input data**: Spatial gene expression counts (spots x genes) from a spatial transcriptomics experiment, plus a single-cell RNA-seq reference dataset (cells x genes) with cell-type annotations.
- **Output**: Estimated cell-type proportions for each spatial spot, along with posterior uncertainty estimates.

Cell2location differs from previous deconvolution methods (e.g., SPOTlight, CIBERSORTx) by using a **hierarchical Bayesian model** that explicitly accounts for technical variation, per-platform sensitivity differences, and shared information across neighboring spots.

## Mathematical / Computational Model

### Negative Binomial Regression Model

Cell2location models the spatial gene expression count matrix $y_{s,g}$ for spot $s$ and gene $g$ as:

$$y_{s,g} \sim \text{NB}(\mu_{s,g}, \theta_g)$$

where $\theta_g$ is the gene-level dispersion parameter and the mean $\mu_{s,g}$ is:

$$\mu_{s,g} = \underbrace{D_s}_{\text{spot scale}} \times \underbrace{\sum_{c=1}^{C} \underbrace{z_{s,c}}_{\text{cell abundance}} \times \underbrace{w_{c,g}}_{\text{cell-type profile}}}_{\text{expected expression}}$$

### Hierarchical Structure

1. **Cell-type expression profiles** $w_{c,g}$: Estimated from scRNA-seq reference data. $w_{c,g}$ represents the expected RNA count for gene $g$ in cell type $c$.

2. **Cell abundance** $z_{s,c}$: The number of cells of type $c$ in spot $s$, modeled as:
   $$z_{s,c} \sim \text{Gamma}(\alpha_c, \beta_c)$$
   where $\alpha_c$ and $\beta_c$ are learned hyperparameters.

3. **Spot scale factor** $D_s \sim \text{LogNormal}(m_D, s_D)$: Accounts for technical variation in total RNA capture across spots.

4. **Reference expression** $w_{c,g}$ also has a hierarchical prior sharing information between cell types:
   $$w_{c,g} \sim \text{Gamma}(\mu_{g}^0 / \kappa_g, 1/\kappa_g)$$

### Variational Inference

The model is fit using **mean-field variational inference** (Adam optimizer, Pyro/NumPyro backend), making it scalable to large spatial datasets.

### Algorithm Summary

1. **Prepare reference signatures**: Extract cell-type-specific gene expression from scRNA-seq reference (mean expression per cell type for each gene).
2. **Build model**: Set up the hierarchical Bayesian model with variables for spot factors, cell abundance, and expression profiles.
3. **Variational inference**: Optimize the evidence lower bound (ELBO) using stochastic gradient descent.
4. **Posterior estimates**: Extract posterior means of $z_{s,c}$ as estimated cell counts per type per spot.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Expression profiles are additive | $\mu_{s,g} = D_s \sum_c z_{s,c} w_{c,g}$ | Cannot model cell-cell interactions affecting expression |
| Reference scRNA-seq captures in vivo expression | $w_{c,g}$ estimated from scRNA-seq data | Sensitivity differences between platforms must be accounted for |
| Gamma prior on cell abundances | $z_{s,c} \sim \text{Gamma}(\alpha_c, \beta_c)$ | Regularizes estimates; works well when most spots contain mixtures |
| Gene-level dispersion | $\theta_g$ shared across all spots | Assumes constant overdispersion per gene |

## Applicable Scenarios

**When to use**:
- Spatial transcriptomics data with spot-level resolution (Visium, Slide-seq, ST)
- Available scRNA-seq reference with matching tissue and cell-type annotations
- Need for uncertainty quantification in deconvolution (Bayesian framework)
- Fine-grained cell-type mapping (can distinguish closely related subtypes)

**When NOT to use**:
- Single-cell resolution spatial data (e.g., MERFISH, Xenium) -- use single-cell methods directly
- No matching scRNA-seq reference available
- Subcellular resolution data
- Very large spatial datasets with millions of spots (runtime scales linearly but may be slow without GPU)

**Comparison**: Outperforms SPOTlight, CIBERSORTx, and Stereoscope on benchmark datasets (Luecken et al., 2021) particularly for rare cell types and similar subtypes.

## Implementation Details

- **Key parameters**: `detection_alpha` (20--200, higher = more confident about shared cell types); `N_cells_per_location` (prior on total cell count per spot, typically 5--30)
- **Computational requirements**: GPU recommended for datasets >100K spots; 16+ GB RAM; runtime ~10--60 min per dataset
- **Preprocessing**: 
  - scRNA-seq: standard filtering, normalization (scran/library-size), log-transformation
  - Spatial data: raw counts required (no normalization)
  - Use 2,000--10,000 highly variable genes for efficiency

## Python Implementation

```python
"""
Minimal implementation of Cell2location-style spatial deconvolution.

This provides a simplified version of the core Bayesian model using
Pyro (PyTorch-based probabilistic programming) for estimating cell-type
proportions from spatial transcriptomics data given a single-cell reference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Gamma, Poisson, LogNormal
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)


def simulate_data(
    n_spots: int = 300,
    n_genes: int = 500,
    n_cell_types: int = 6,
    n_ref_cells: int = 1000,
    avg_cells_per_spot: int = 8,
    seed: int = 42,
):
    """
    Simulate spatial transcriptomics data with ground truth cell-type proportions.
    
    Parameters
    ----------
    n_spots : int
        Number of spatial spots.
    n_genes : int
        Number of genes.
    n_cell_types : int
        Number of cell types.
    n_ref_cells : int
        Number of cells in the scRNA-seq reference.
    avg_cells_per_spot : int
        Average number of cells per spot.
    
    Returns
    -------
    dict containing spatial counts, reference counts, reference labels, 
    and ground truth proportions.
    """
    rng = np.random.default_rng(seed)
    
    # Cell-type-specific gene expression profiles (log-scale)
    # Each cell type has ~15 marker genes with high expression
    true_profile = np.zeros((n_cell_types, n_genes))
    for c in range(n_cell_types):
        # Marker genes: high expression
        markers = rng.choice(n_genes, size=15, replace=False)
        true_profile[c, markers] = rng.exponential(5.0, size=15)
        # Background expression
        background = rng.exponential(0.5, size=n_genes)
        background[markers] = 0.0
        true_profile[c] += background
    
    # Scale to count-like values
    prof_scale = np.exp(true_profile) * 10
    prof_scale = (prof_scale.T / prof_scale.sum(axis=1)).T * 100
    
    # scRNA-seq reference: sample cells from the profile (NB-distributed noise)
    ref_counts = np.zeros((n_ref_cells, n_genes), dtype=np.int32)
    ref_labels = np.repeat(np.arange(n_cell_types), n_ref_cells // n_cell_types)[:n_ref_cells]
    for i in range(n_ref_cells):
        c = ref_labels[i]
        mu = prof_scale[c]
        # Negative binomial sampling
        ref_counts[i] = rng.negative_binomial(
            n=mu / (mu + 0.5 + 1e-10),  # r parameter
            p=1.0 / (1.0 + 0.5),  # p parameter
            size=n_genes
        ).astype(np.int32)
    
    # Spatial data: mixture of cell types per spot
    spot_counts = np.zeros((n_spots, n_genes), dtype=np.int32)
    true_proportions = np.zeros((n_spots, n_cell_types))
    
    for s in range(n_spots):
        # Random cell-type composition
        if rng.random() < 0.3:
            # Some spots are relatively pure
            pure_type = rng.integers(0, n_cell_types)
            props = rng.dirichlet(np.ones(n_cell_types) * 0.1)
            props = props / props.sum()
            props[pure_type] += 0.5
            props = props / props.sum()
        else:
            props = rng.dirichlet(np.ones(n_cell_types) * 0.5)
        
        true_proportions[s] = props
        
        # Expected expression
        n_cells = rng.poisson(avg_cells_per_spot)
        mu = props @ prof_scale * n_cells
        
        # Negative binomial sampling
        spot_counts[s] = rng.negative_binomial(
            n=mu / (mu + 0.3 + 1e-10),
            p=1.0 / (1.0 + 0.3),
            size=n_genes
        ).astype(np.int32)
    
    # Gene names
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    cell_type_names = [f"Type_{c}" for c in range(n_cell_types)]
    
    return {
        "spot_counts": torch.tensor(spot_counts, dtype=torch.float32),
        "ref_counts": torch.tensor(ref_counts, dtype=torch.float32),
        "ref_labels": torch.tensor(ref_labels, dtype=torch.long),
        "true_proportions": torch.tensor(true_proportions, dtype=torch.float32),
        "gene_names": gene_names,
        "cell_type_names": cell_type_names,
        "true_profile": torch.tensor(prof_scale, dtype=torch.float32),
    }


def prepare_reference_signatures(
    ref_counts: torch.Tensor,
    ref_labels: torch.Tensor,
    n_cell_types: int,
    pseudocount: float = 1.0,
):
    """
    Compute cell-type-specific gene expression signatures from scRNA-seq reference.
    
    Parameters
    ----------
    ref_counts : Tensor (n_cells x n_genes)
        Raw count matrix from scRNA-seq reference.
    ref_labels : Tensor (n_cells,)
        Cell-type annotations.
    n_cell_types : int
        Number of cell types.
    pseudocount : float
        Additive smoothing pseudocount.
    
    Returns
    -------
    signatures : Tensor (n_cell_types x n_genes)
        Mean expression per gene per cell type.
    """
    signatures = []
    for c in range(n_cell_types):
        mask = ref_labels == c
        type_counts = ref_counts[mask]  # (n_type_cells x n_genes)
        # Normalize to library size per cell, then average
        lib_sizes = type_counts.sum(dim=1, keepdim=True)
        norm_counts = type_counts / lib_sizes * 10000  # CPM-like
        mu = norm_counts.mean(dim=0)
        signatures.append(mu)
    return torch.stack(signatures, dim=0)  # (n_cell_types x n_genes)


class Cell2locationModel:
    """
    Simplified Cell2location Bayesian deconvolution model.
    
    Uses variational inference to estimate cell-type abundances
    at each spatial spot given reference signatures.
    """
    
    def __init__(
        self,
        n_cell_types: int,
        n_genes: int,
        n_spots: int,
        detection_alpha: float = 50.0,
        n_cells_prior: float = 8.0,
    ):
        self.n_cell_types = n_cell_types
        self.n_genes = n_genes
        self.n_spots = n_spots
        self.detection_alpha = detection_alpha
        self.n_cells_prior = n_cells_prior
        
    def model(self, spot_counts, signatures):
        """
        Generative model for spatial gene expression.
        
        Parameters
        ----------
        spot_counts : Tensor (n_spots x n_genes)
            Observed spatial counts.
        signatures : Tensor (n_cell_types x n_genes)
            Reference expression profiles.
        """
        n_spots, n_genes = spot_counts.shape
        n_cell_types = signatures.shape[0]
        
        # Global scale factor for each spot (technical variation)
        D_s = pyro.sample("D_s", dist.LogNormal(0.0, 1.0).expand([n_spots]).to_event(1))
        
        # Cell abundances per spot per type (Gamma prior)
        with pyro.plate("cell_types_plate", n_cell_types):
            alpha_c = torch.full((1,), self.detection_alpha)
            beta_c = torch.full((1,), self.detection_alpha / self.n_cells_prior)
            z_sc = pyro.sample(
                "z_sc", 
                dist.Gamma(alpha_c.expand(n_spots), beta_c.expand(n_spots)).to_event(1)
            )
        
        # Expected expression: spot_scale * sum_celltypes(cell_abundance * signature)
        # z_sc: (n_spots x n_cell_types), signatures: (n_cell_types x n_genes)
        expected_mu = D_s.unsqueeze(-1) * (z_sc @ signatures)
        
        # Gene-level dispersion
        theta_g = pyro.sample("theta_g", dist.Gamma(3.0, 1.0).expand([n_genes]).to_event(1))
        
        # Observation likelihood: Negative Binomial (via Gamma-Poisson mixture)
        with pyro.plate("spots_plate", n_spots):
            with pyro.plate("genes_plate", n_genes):
                rate = expected_mu + 1e-6
                # Gamma-Poisson mixture -> Negative Binomial
                # lambda ~ Gamma(rate, theta), y ~ Poisson(lambda)
                # This is parameterized as NB(mean=rate, dispersion=1/theta)
                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(
                        total_count=theta_g,  # r
                        probs=theta_g / (theta_g + rate + 1e-6)  # r/(r+mu)
                    ),
                    obs=spot_counts,
                )
    
    def guide(self, spot_counts, signatures):
        """
        Variational distribution (mean-field) for the model.
        """
        n_spots, n_genes = spot_counts.shape
        n_cell_types = signatures.shape[0]
        
        # D_s: LogNormal variational params
        D_loc = pyro.param("D_loc", torch.zeros(n_spots))
        D_scale = pyro.param("D_scale", torch.ones(n_spots) * 0.1,
                             constraint=dist.constraints.positive)
        pyro.sample("D_s", dist.LogNormal(D_loc, D_scale).to_event(1))
        
        # z_sc: Gamma variational params (using LogNormal for simplicity)
        z_loc = pyro.param("z_loc", torch.zeros(n_spots, n_cell_types))
        z_scale = pyro.param("z_scale", torch.ones(n_spots, n_cell_types) * 0.3,
                             constraint=dist.constraints.positive)
        pyro.sample("z_sc", dist.LogNormal(z_loc, z_scale).to_event(1))
        
        # theta_g: Gamma variational params
        theta_a = pyro.param("theta_a", 
                              torch.ones(n_genes) * 2.0,
                              constraint=dist.constraints.positive)
        theta_b = pyro.param("theta_b",
                              torch.ones(n_genes),
                              constraint=dist.constraints.positive)
        pyro.sample("theta_g", dist.Gamma(theta_a, theta_b).to_event(1))
    
    def fit(
        self,
        spot_counts: torch.Tensor,
        signatures: torch.Tensor,
        n_steps: int = 2000,
        lr: float = 0.01,
        verbose: bool = True,
    ):
        """
        Run variational inference to estimate cell-type abundances.
        
        Parameters
        ----------
        spot_counts : Tensor (n_spots x n_genes)
        signatures : Tensor (n_cell_types x n_genes)
        n_steps : int
        lr : float
        verbose : bool
        
        Returns
        -------
        self
        """
        pyro.clear_param_store()
        
        optim = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        
        losses = []
        for step in range(n_steps):
            loss = svi.step(spot_counts, signatures)
            losses.append(loss)
            if verbose and (step + 1) % 500 == 0:
                print(f"Step {step+1}/{n_steps}, ELBO loss: {loss:.1f}")
        
        self.losses = losses
        self._extract_results()
        return self
    
    def _extract_results(self):
        """
        Extract posterior estimates from variational parameters.
        """
        # Cell abundance posterior (LogNormal mean)
        z_loc = pyro.param("z_loc").detach()
        z_scale = pyro.param("z_scale").detach()
        self.z_est = torch.exp(z_loc + 0.5 * z_scale ** 2)  # LogNormal mean
        
        # Normalize to proportions
        self.proportions_est = self.z_est / (self.z_est.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Spot scale factor
        self.D_est = torch.exp(
            pyro.param("D_loc").detach() + 0.5 * pyro.param("D_scale").detach() ** 2
        )


def compute_error(estimated: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute root mean squared error between estimated and true proportions.
    
    Parameters
    ----------
    estimated : Tensor (n_spots x n_cell_types)
    true : Tensor (n_spots x n_cell_types)
    
    Returns
    -------
    RMSE value.
    """
    return torch.sqrt(torch.mean((estimated - true) ** 2)).item()


def select_hvgs(counts: torch.Tensor, n_top: int = 500) -> torch.Tensor:
    """
    Select highly variable genes based on variance-to-mean ratio.
    
    Parameters
    ----------
    counts : Tensor (n_cells x n_genes)
    n_top : int
    
    Returns
    -------
    Boolean mask for selected genes.
    """
    mu = counts.mean(dim=0)
    var = counts.var(dim=0, unbiased=False)
    cv2 = var / (mu + 1e-10)
    # Select top N by CV^2
    _, indices = torch.topk(cv2, min(n_top, counts.shape[1]))
    mask = torch.zeros(counts.shape[1], dtype=torch.bool)
    mask[indices] = True
    return mask


# ============================================================================
# Complete usage example
# ============================================================================

def main():
    """
    Run a complete Cell2location-style deconvolution analysis on simulated data.
    """
    print("=" * 60)
    print("Cell2location-style Spatial Deconvolution")
    print("=" * 60)
    
    # --- 1. Simulate spatial transcriptomics data ---
    print("\n[1] Simulating spatial transcriptomics data...")
    data = simulate_data(
        n_spots=200,
        n_genes=500,
        n_cell_types=5,
        n_ref_cells=1000,
        avg_cells_per_spot=8,
    )
    print(f"    Spatial spots: {data['spot_counts'].shape[0]}")
    print(f"    Reference cells: {data['ref_counts'].shape[0]}")
    print(f"    Cell types: {data['cell_type_names']}")
    
    # --- 2. Select highly variable genes ---
    print("\n[2] Selecting highly variable genes...")
    hvg_mask = select_hvgs(data["ref_counts"], n_top=300)
    spot_counts_hvg = data["spot_counts"][:, hvg_mask]
    ref_counts_hvg = data["ref_counts"][:, hvg_mask]
    print(f"    Selected {hvg_mask.sum().item()} HVGs")
    
    # --- 3. Prepare reference signatures ---
    print("\n[3] Computing reference signatures from scRNA-seq...")
    signatures = prepare_reference_signatures(
        ref_counts_hvg,
        data["ref_labels"],
        n_cell_types=5,
    )
    print(f"    Signatures shape: {signatures.shape}")
    
    # --- 4. Run Cell2location deconvolution ---
    print("\n[4] Running Cell2location variational inference...")
    model = Cell2locationModel(
        n_cell_types=5,
        n_genes=spot_counts_hvg.shape[1],
        n_spots=spot_counts_hvg.shape[0],
        detection_alpha=50.0,
        n_cells_prior=8.0,
    )
    model.fit(
        spot_counts_hvg,
        signatures,
        n_steps=1500,
        lr=0.01,
        verbose=True,
    )
    
    # --- 5. Evaluate results ---
    print("\n[5] Evaluating deconvolution accuracy...")
    est_props = model.proportions_est
    true_props = data["true_proportions"]
    
    rmse = compute_error(est_props, true_props)
    print(f"\n    Overall RMSE: {rmse:.4f}")
    
    # Per-cell-type RMSE
    print("\n    Per-cell-type RMSE:")
    for c in range(5):
        ct_rmse = torch.sqrt(
            torch.mean((est_props[:, c] - true_props[:, c]) ** 2)
        ).item()
        print(f"      {data['cell_type_names'][c]}: {ct_rmse:.4f}")
    
    # Correlation between estimated and true proportions
    print("\n    Per-cell-type correlation:")
    for c in range(5):
        corr = np.corrcoef(
            est_props[:, c].numpy(), true_props[:, c].numpy()
        )[0, 1]
        print(f"      {data['cell_type_names'][c]}: {corr:.3f}")
    
    # --- 6. Summary statistics ---
    print("\n[6] Summary Statistics")
    print(f"    Mean estimated proportion per type:")
    for c in range(5):
        print(f"      {data['cell_type_names'][c]}: "
              f"est={est_props[:, c].mean():.3f}, "
              f"true={true_props[:, c].mean():.3f}")
    
    print("\n" + "=" * 60)
    print("Deconvolution complete.")
    print("=" * 60)
    
    return {
        "model": model,
        "estimates": est_props,
        "ground_truth": true_props,
        "rmse": rmse,
    }


if __name__ == "__main__":
    result = main()
```

## References

Kleshchevnikov, V., Shmatko, A., Dann, E., et al. (2022). Cell2location maps fine-grained cell types in spatial transcriptomics. *Nature Biotechnology*, 40(5), 661--671. https://doi.org/10.1038/s41587-021-01139-4

Luecken, M. D., Buttner, M., Chaichoompu, K., et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nature Methods*, 19, 41--50. https://doi.org/10.1038/s41592-021-01336-8

Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15, 1053--1058. https://doi.org/10.1038/s41592-018-0229-2

Danaher, P., Kim, Y., Nelson, B., et al. (2022). Advances in mixed cell deconvolution enable identification of cell types in spatial transcriptomic data. *Nature Communications*, 13, 385. https://doi.org/10.1038/s41467-022-28020-9

Argelaguet, R., Arnol, D., Bredikhin, D., et al. (2020). MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. *Genome Biology*, 21, 111. https://doi.org/10.1186/s13059-020-02015-1
