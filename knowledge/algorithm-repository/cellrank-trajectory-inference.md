# CellRank: Directed Single-Cell Fate Mapping with RNA Velocity

**Source**: Lange, M., Bergen, V., Klein, M., Setty, M., Reuter, B., Bakhti, M., Lickert, H., Ansari, M., Schniering, J., Schiller, H. B., Pe'er, D., & Theis, F. J. (2022). CellRank for directed single-cell fate mapping. *Nature Methods*, 19, 159--170. https://doi.org/10.1038/s41592-021-01346-6

**Category**: Bioinformatics / Single-Cell Genomics

## Biological / Computational Problem

Single-cell RNA-seq captures a snapshot of gene expression, obscuring the dynamic processes of cell differentiation, development, and disease progression. Key questions include: which cells are transitioning between states, what are the terminal fates, and what genes drive these transitions? RNA velocity (the ratio of unspliced to spliced mRNA) provides directional information, but deriving probabilistic fate predictions from noisy velocity estimates remains challenging.

- **Input data**: Single-cell gene expression matrix (cells x genes) + RNA velocity vectors (cells x genes, estimated by scVelo/velocyto)
- **Output**: Cell fate probabilities (cells x terminal states), initial/terminal state identification, gene expression trends along trajectories

## Mathematical / Computational Model

### Markov Chain Construction

CellRank models cell-state transitions as a **Markov chain**. Each cell $i$ is a state, and the transition probability from cell $i$ to cell $j$ is:

$$T_{ij} = \frac{1}{Z_i} \exp\left(\alpha \cdot \cos(\mathbf{v}_i, \mathbf{x}_j - \mathbf{x}_i)\right) \cdot K_{\sigma}(\mathbf{x}_i, \mathbf{x}_j)$$

where:
- $\mathbf{v}_i$ is the RNA velocity vector for cell $i$ (direction of change)
- $\mathbf{x}_i$ is the gene expression vector of cell $i$ (low-dimensional embedding, e.g., UMAP)
- $\cos(\mathbf{v}_i, \mathbf{x}_j - \mathbf{x}_i)$ measures alignment between velocity direction and the vector pointing to cell $j$
- $K_{\sigma}(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$ is a Gaussian kernel that downweights distant cells
- $\alpha$ controls the weight of velocity information; $Z_i$ is a normalization constant
- $\sigma$ is a local kernel width (set to the distance to the $k$-th nearest neighbor)

### Coarse-Graining via GPCCA

CellRank uses **Generalized Perron Cluster Cluster Analysis (GPCCA)** to coarse-grain the Markov chain into macrostates:

1. Compute the transition matrix $T \in \mathbb{R}^{N \times N}$
2. Solve the eigenvalue problem to identify the dominant slow relaxation modes:
   $$T \psi = \lambda \psi$$
3. For a chosen number of metastable states $m$, compute the Schur vectors (real-valued basis)
4. Assign each cell to a macrostate via fuzzy membership: $\text{membership}_{i,k} \in [0, 1]$

The GPCCA decomposition identifies:
- **Initial states**: Populations with high probability of transitioning out
- **Terminal states**: Absorbing or nearly absorbing populations (self-transition probability near 1)
- **Intermediate states**: Transient populations along differentiation paths

### Fate Probabilities

For each cell $i$ and each terminal state $k$, the fate probability is:

$$f_{ik} = \sum_{j} \Phi_{ij} \cdot \delta_{j \in \mathcal{T}_k}$$

where $\Phi$ is the matrix of absorption probabilities and $\mathcal{T}_k$ is the set of cells assigned to terminal state $k$.

The absorption probability from cell $i$ to terminal set $j$ satisfies:

$$\Phi_{ij} = T_{ij} + \sum_{\ell \notin \mathcal{T}} T_{i\ell} \Phi_{\ell j}$$

which can be solved efficiently as a linear system.

### Uncertainty Propagation

CellRank propagates uncertainty from velocity estimates through the Markov chain:

$$\sigma^2(f_{ik}) = \sum_j \left(\frac{\partial f_{ik}}{\partial \mathbf{v}_j}\right)^2 \sigma^2(\mathbf{v}_j)$$

where $\sigma^2(\mathbf{v}_j)$ is the estimated variance of the velocity vector for cell $j$.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Markovian dynamics | $p(\text{state}_{t+1} \mid \text{state}_t, \dots) = p(\text{state}_{t+1} \mid \text{state}_t)$ | No long-range memory in differentiation |
| Velocity is proportional to expression change | $\mathbf{v}_i \propto \frac{d\mathbf{x}_i}{dt}$ | Requires steady-state splicing model for estimation |
| Gaussian local approximation | $K_{\sigma}(\mathbf{x}_i, \mathbf{x}_j)$ weights transitions | Choice of $\sigma$ affects trajectory granularity |
| Terminal states are absorbing | $T_{jj} \approx 1$ for terminal cells | May not capture plastic/regenerative states |

## Applicable Scenarios

**When to use**:
- Single-cell data with RNA velocity estimates available
- Developmental biology, reprogramming, regeneration, disease progression
- Identifying novel transient cell states and branch points
- Systems with clear directionality (differentiation, activation)

**When NOT to use**:
- No RNA velocity information available (static data only)
- Systems without clear directionality (steady-state tissues, cycling cells)
- Very small datasets (<200 cells)
- Strong technical batch effects masking biological transitions

**Comparison**: Previous methods (Monocle, Slingshot, PAGA) require prior knowledge of root or terminal states; CellRank automatically identifies these from velocity. CellRank 2 extends the framework to incorporate additional data modalities.

## Implementation Details

- **Key parameters**: `n_neighbors` (30--100, neighborhood size for kernel), `n_states` (automatic detection by GPCCA or user-specified)
- **Computational requirements**: 8--32 GB RAM for datasets up to 100K cells; runtime ~10--30 min
- **Preprocessing**:
  - Standard scRNA-seq pipeline: filtering, normalization, HVG selection, PCA (50 components)
  - Compute RNA velocity with scVelo (dynamical model) or velocyto
  - Low-dimensional embedding (UMAP/PHATE) for visualization

## Python Implementation

```python
"""
Minimal implementation of CellRank-style trajectory inference with RNA velocity.

This provides a simplified version of the Markov chain + GPCCA framework
for computing directed cell fate probabilities, demonstrated on synthetic data.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


def simulate_lineage_data(
    n_cells: int = 500,
    n_genes: int = 100,
    n_terminal_states: int = 2,
    n_branch_genes: int = 10,
    seed: int = 42,
):
    """
    Simulate single-cell data along a bifurcating developmental trajectory
    with RNA velocity information.
    
    The simulation generates cells along a tree-like manifold with two
    terminal branches and computes approximate velocity vectors.
    
    Parameters
    ----------
    n_cells : int
    n_genes : int
    n_terminal_states : int
    n_branch_genes : int
    
    Returns
    -------
    dict with expression, velocity, pseudotime, and terminal labels.
    """
    rng = np.random.default_rng(seed)
    
    # Assign cells to trajectory positions
    # Root -> branch point -> terminal states
    
    n_root = int(n_cells * 0.15)
    n_mid = int(n_cells * 0.25)
    n_terminal = n_cells - n_root - n_mid
    
    cells_per_terminal = n_terminal // n_terminal_states
    n_terminal = cells_per_terminal * n_terminal_states
    
    # Pseudotime assignment
    pseudotime = np.zeros(n_cells)
    pseudotime[:n_root] = rng.uniform(0, 0.2, size=n_root)
    pseudotime[n_root:n_root + n_mid] = rng.uniform(0.15, 0.55, size=n_mid)
    
    # Terminal branch assignment
    branch_labels = np.zeros(n_cells, dtype=np.int64)
    for b in range(n_terminal_states):
        start = n_root + n_mid + b * cells_per_terminal
        end = start + cells_per_terminal
        pseudotime[start:end] = rng.uniform(0.45, 1.0, size=cells_per_terminal)
        branch_labels[start:end] = b + 1
    
    # Generate expression matrix along the tree
    # Key branch-specific genes drive differentiation
    branch_genes = {}
    for b in range(n_terminal_states):
        genes = rng.choice(n_genes, size=n_branch_genes, replace=False)
        branch_genes[b] = genes
    
    expression = np.zeros((n_cells, n_genes))
    velocity = np.zeros((n_cells, n_genes))
    
    for i in range(n_cells):
        # Baseline expression
        base = rng.normal(0, 0.5, size=n_genes)
        
        # Pseudotime-dependent expression (global program)
        pt = pseudotime[i]
        base += pt * rng.normal(0.3, 0.1, size=n_genes)
        
        # Branch-specific expression
        if branch_labels[i] > 0:
            b = branch_labels[i] - 1
            for g in branch_genes[b]:
                base[g] += pt * rng.exponential(3.0)  # Upregulate branch markers
        
        expression[i] = base
        
        # Velocity: derivative w.r.t. pseudotime (approximated)
        velocity[i] = rng.normal(0.3, 0.1, size=n_genes)
        if branch_labels[i] > 0:
            b = branch_labels[i] - 1
            for g in branch_genes[b]:
                velocity[i, g] += rng.exponential(1.0)
    
    # Add noise
    expression += rng.normal(0, 0.1, size=expression.shape)
    
    # PCA reduction for visualization
    pca = PCA(n_components=30)
    expr_pca = pca.fit_transform(expression)
    
    # Use PCA coordinates for the Markov chain
    # Keep top PCs and add velocity projected to PCA space
    velo_pca = pca.transform(velocity)
    
    return {
        "expression_pca": torch.tensor(expr_pca[:, :10], dtype=torch.float32),
        "velocity_pca": torch.tensor(velo_pca[:, :10], dtype=torch.float32),
        "pseudotime": torch.tensor(pseudotime, dtype=torch.float32),
        "branch_labels": torch.tensor(branch_labels, dtype=torch.long),
        "n_cells": n_cells,
        "n_terminal": n_terminal_states,
    }


class CellRankModel:
    """
    Simplified CellRank: Markov chain with RNA velocity for fate mapping.
    
    Key steps:
    1. Build transition matrix using velocity-aligned cell-cell similarities
    2. Identify terminal states via GPCCA (simplified spectral decomposition)
    3. Compute fate probabilities via absorption probabilities
    """
    
    def __init__(
        self,
        n_neighbors: int = 30,
        velocity_scale: float = 1.0,
        kernel_scale: float = 1.0,
        n_components: int = 10,
    ):
        self.n_neighbors = n_neighbors
        self.velocity_scale = velocity_scale
        self.kernel_scale = kernel_scale
        self.n_components = n_components
        
        self.transition_matrix = None
        self.terminal_states = None
        self.fate_probs = None
        self.memberships = None
        
    def fit(
        self,
        data_pca: torch.Tensor,
        velocity_pca: torch.Tensor,
    ):
        """
        Fit CellRank model to compute fate probabilities.
        
        Parameters
        ----------
        data_pca : Tensor (n_cells x n_pcs)
            Low-dimensional embedding of expression data.
        velocity_pca : Tensor (n_cells x n_pcs)
            RNA velocity vectors in PCA space.
        """
        n_cells = data_pca.shape[0]
        
        # Step 1: Build cell-cell similarity graph (kNN)
        data_np = data_pca.numpy()
        nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, n_cells),
            metric="euclidean"
        )
        nn_model.fit(data_np)
        distances, indices = nn_model.kneighbors(data_np)
        
        # Remove self-neighbor (first column)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        
        # Step 2: Compute velocity-aligned transition probabilities
        # Transition from cell i to cell j depends on alignment of velocity
        # with the direction from i to j
        
        T = torch.zeros((n_cells, n_cells))
        
        for i in range(n_cells):
            neighbors = indices[i]
            n_neigh = len(neighbors)
            
            # Direction vectors from i to neighbors
            dir_vectors = data_pca[neighbors] - data_pca[i]  # (n_neigh, n_pcs)
            
            # Velocity vector for cell i
            v_i = velocity_pca[i]  # (n_pcs,)
            
            # Cosine similarity between velocity and direction
            v_norm = torch.norm(v_i)
            dir_norms = torch.norm(dir_vectors, dim=1)
            
            cos_sim = torch.zeros(n_neigh)
            mask = (v_norm > 0) & (dir_norms > 0)
            if mask.any():
                cos_sim[mask] = (dir_vectors[mask] @ v_i) / (
                    dir_norms[mask] * v_norm + 1e-10
                )
            
            # Clamp to [-1, 1]
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            
            # Gaussian kernel weight
            kernel_width = self.kernel_scale * distances[i].mean()
            if kernel_width < 1e-6:
                kernel_width = 1.0
            gauss_weights = torch.exp(
                -torch.tensor(distances[i], dtype=torch.float32) ** 2
                / (2 * kernel_width ** 2 + 1e-10)
            )
            
            # Combined weight: velocity alignment * distance
            weights = torch.exp(self.velocity_scale * cos_sim) * gauss_weights
            
            # Remove negative contributions (velocity opposes direction)
            weights[cos_sim < -0.3] = 0.0
            
            # Normalize
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
            T[i, neighbors] = weights
        
        self.transition_matrix = T
        
        # Step 3: Identify terminal states using simplified spectral analysis
        self._find_terminal_states()
        
        # Step 4: Compute absorption probabilities (fate probabilities)
        self._compute_fate_probabilities()
        
        return self
    
    def _find_terminal_states(self):
        """
        Identify terminal and initial states using the transition matrix.
        
        Terminal states: cells with high self-loop probability (low outgoing flux).
        Initial states: cells with high outgoing flux.
        
        Uses a simplified spectral approach based on the graph Laplacian
        to identify metastable groups.
        """
        T = self.transition_matrix
        
        # Compute stationary distribution
        # (power iteration)
        pi = torch.ones(T.shape[0]) / T.shape[0]
        for _ in range(100):
            pi_new = pi @ T
            if torch.norm(pi_new - pi) < 1e-6:
                break
            pi = pi_new
        
        # Reversibility check / self-transition probability
        self_loop = T.diag()
        
        # Compute "terminal score": high if cell is absorbing
        outgoing = T.sum(dim=1)
        terminal_score = 1 - outgoing  # high = likely terminal
        
        # Compute "initial score": low if cell is terminal
        # Use stationary distribution * self-loop probability
        # Cells that accumulate probability (sinks) are terminal
        sink_score = pi * (1 - self_loop)
        
        # Simple thresholding for terminal state identification
        n_terminal = 3  # estimate initial+terminal states
        _, terminal_idx = torch.topk(terminal_score, n_terminal)
        
        # Remove very low terminal scores
        terminal_scores = terminal_score[terminal_idx]
        valid = terminal_scores > terminal_scores.median()
        terminal_idx = terminal_idx[valid]
        
        if len(terminal_idx) < 2:
            # Fallback: use local maxima of density
            knn = NearestNeighbors(n_neighbors=10)
            knn.fit(T.numpy())
            densities = 1.0 / knn.kneighbors(T.numpy())[0].mean(axis=1)
            _, terminal_idx = torch.topk(
                torch.tensor(densities, dtype=torch.float32), 
                min(3, T.shape[0] // 20)
            )
        
        self.terminal_states = terminal_idx
        
    def _compute_fate_probabilities(self):
        """
        Compute fate probabilities by solving the absorption probability system.
        
        For each non-terminal cell i and terminal set k:
        f_{i,k} = sum over direct transitions + indirect paths
        """
        n_cells = self.transition_matrix.shape[0]
        n_terminal = len(self.terminal_states)
        
        T = self.transition_matrix
        term_set = set(self.terminal_states.tolist())
        
        # Mark non-terminal cells
        non_terminal = torch.tensor([
            i for i in range(n_cells) if i not in term_set
        ], dtype=torch.long)
        
        if len(non_terminal) == 0:
            self.fate_probs = torch.eye(n_cells)[:, :n_terminal]
            return
        
        # Solve (I - Q) * F = R where:
        # Q = transitions among non-terminal states
        # R = transitions from non-terminal to terminal states
        
        # Q matrix: (n_non_terminal x n_non_terminal)
        Q = T[non_terminal][:, non_terminal]
        
        # R matrix: (n_non_terminal x n_terminal)
        R = T[non_terminal][:, self.terminal_states]
        
        # (I - Q) * F = R  =>  F = (I - Q)^{-1} * R
        I = torch.eye(len(non_terminal))
        
        try:
            F = torch.linalg.solve(I - Q, R)
        except torch.linalg.LinAlgError:
            # Fallback: use iterative method (Neumann series)
            F = R.clone()
            Q_power = Q.clone()
            for _ in range(50):
                Q_power = Q_power @ Q
                term = Q_power @ R
                F = F + term
                if term.abs().max() < 1e-6:
                    break
        
        # Construct full fate probability matrix
        fate_probs = torch.zeros((n_cells, n_terminal))
        fate_probs[self.terminal_states] = torch.eye(n_terminal)
        fate_probs[non_terminal] = F
        
        # Normalize rows
        fate_probs = fate_probs / (fate_probs.sum(dim=1, keepdim=True) + 1e-10)
        
        self.fate_probs = fate_probs
    
    def predict_terminal_state(self, confidence_threshold: float = 0.8) -> torch.Tensor:
        """
        Predict terminal state for each cell based on fate probabilities.
        
        Parameters
        ----------
        confidence_threshold : float
            Minimum probability to assign a terminal state.
        
        Returns
        -------
        Tensor (n_cells,)
            Predicted terminal state (-1 if below threshold).
        """
        max_probs, predictions = self.fate_probs.max(dim=1)
        predictions[max_probs < confidence_threshold] = -1
        return predictions
    
    def compute_lineage_driver_genes(
        self, expression: torch.Tensor, lineage: int, n_top: int = 10
    ) -> list:
        """
        Identify genes whose expression correlates with a lineage's fate probability.
        
        Parameters
        ----------
        expression : Tensor (n_cells x n_genes)
        lineage : int
        n_top : int
        
        Returns
        -------
        list of (gene_idx, correlation) tuples
        """
        lineage_prob = self.fate_probs[:, lineage]
        corrs = []
        
        n_genes = expression.shape[1]
        for g in range(n_genes):
            corr = np.corrcoef(
                lineage_prob.numpy(), expression[:, g].numpy()
            )[0, 1]
            corrs.append((g, abs(corr) if not np.isnan(corr) else 0.0))
        
        corrs.sort(key=lambda x: x[1], reverse=True)
        return corrs[:n_top]


# ============================================================================
# Complete usage example
# ============================================================================

def main():
    """
    Run a complete CellRank-style trajectory inference analysis.
    """
    print("=" * 60)
    print("CellRank: Directed Single-Cell Fate Mapping")
    print("=" * 60)
    
    # --- 1. Simulate lineage data ---
    print("\n[1] Simulating bifurcating developmental trajectory...")
    data = simulate_lineage_data(
        n_cells=400,
        n_genes=50,
        n_terminal_states=2,
        n_branch_genes=8,
    )
    
    n_cells = data["n_cells"]
    print(f"    Cells: {n_cells}")
    print(f"    PCA dims: {data['expression_pca'].shape[1]}")
    print(f"    Terminal states: {data['n_terminal']}")
    print(f"    Branch distribution: "
          f"{torch.bincount(data['branch_labels']).tolist()}")
    
    # --- 2. Build and fit CellRank model ---
    print("\n[2] Building CellRank Markov chain model...")
    model = CellRankModel(
        n_neighbors=20,
        velocity_scale=1.5,
        kernel_scale=1.0,
    )
    model.fit(data["expression_pca"], data["velocity_pca"])
    
    print(f"    Identified terminal states: {model.terminal_states.tolist()}")
    print(f"    Transition matrix shape: {model.transition_matrix.shape}")
    
    # --- 3. Examine fate probabilities ---
    print("\n[3] Fate probability summary...")
    fate_probs = model.fate_probs
    
    for t in range(fate_probs.shape[1]):
        print(f"    Terminal state {t}: "
              f"mean prob = {fate_probs[:, t].mean():.3f}, "
              f"cells with prob > 0.8 = {(fate_probs[:, t] > 0.8).sum().item()}")
    
    # --- 4. Compare with ground truth ---
    print("\n[4] Comparison with ground truth branch labels...")
    predictions = model.predict_terminal_state(confidence_threshold=0.6)
    
    # Align predictions with ground truth via best matching
    true_labels = data["branch_labels"]
    
    # For each terminal state, compute agreement with each ground truth branch
    from scipy.optimize import linear_sum_assignment
    
    n_term = fate_probs.shape[1]
    cost_matrix = np.zeros((n_term, data["n_terminal"] + 1))  # +1 for unassigned
    
    for t in range(n_term):
        prob = fate_probs[:, t].numpy()
        for b in range(data["n_terminal"] + 1):
            mask = (true_labels.numpy() == b)
            if mask.sum() > 0:
                cost_matrix[t, b] = prob[mask].mean()
    
    best_alignment = cost_matrix.argmax(axis=1)
    print(f"    Terminal state alignment: {best_alignment.tolist()}")
    
    # Accuracy
    aligned_probs = fate_probs.numpy()
    aligned_preds = aligned_probs.argmax(axis=1)
    
    # Map using best alignment
    mapping = {t: best_alignment[t].item() for t in range(n_term)}
    mapped_preds = np.array([mapping.get(p, -1) for p in aligned_preds])
    
    accuracy = (mapped_preds == true_labels.numpy()).mean()
    print(f"    Fate prediction accuracy: {accuracy:.3f}")
    
    # --- 5. Pseudo-temporal ordering ---
    print("\n[5] Pseudo-temporal ordering analysis...")
    # Cells with low fate probability to any terminal state are "early"
    uncertainty = 1 - fate_probs.max(dim=1).values
    early_cells = uncertainty > 0.4
    
    # Cells committed to a terminal state
    committed = uncertainty < 0.2
    
    print(f"    Early/undecided cells: {early_cells.sum().item()}")
    print(f"    Committed cells: {committed.sum().item()}")
    
    # --- 6. Transition entropy analysis ---
    print("\n[6] Transition entropy (differentiation plasticity)...")
    entropy = -(fate_probs * torch.log(fate_probs + 1e-10)).sum(dim=1)
    
    print(f"    Mean entropy: {entropy.mean():.3f}")
    print(f"    Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
    
    early_entropy = entropy[early_cells].mean().item()
    committed_entropy = entropy[committed].mean().item()
    print(f"    Early cell entropy: {early_entropy:.3f}")
    print(f"    Committed cell entropy: {committed_entropy:.3f}")
    
    print("\n" + "=" * 60)
    print("CellRank demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## References

Lange, M., Bergen, V., Klein, M., et al. (2022). CellRank for directed single-cell fate mapping. *Nature Methods*, 19, 159--170. https://doi.org/10.1038/s41592-021-01346-6

Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020). Generalizing RNA velocity to transient cell states through dynamical modeling. *Nature Biotechnology*, 38, 1408--1414. https://doi.org/10.1038/s41587-020-0591-3

La Manno, G., Soldatov, R., Zeisel, A., et al. (2018). RNA velocity of single cells. *Nature*, 560, 494--498. https://doi.org/10.1038/s41586-018-0414-6

Setty, M., Kiseliovas, V., Levine, J., Gayoso, A., Mazutis, L., & Pe'er, D. (2019). Characterization of cell fate probabilities in single-cell data with Palantir. *Nature Biotechnology*, 37, 451--460. https://doi.org/10.1038/s41587-019-0068-4

Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*, 19, 15. https://doi.org/10.1186/s13059-017-1382-0
