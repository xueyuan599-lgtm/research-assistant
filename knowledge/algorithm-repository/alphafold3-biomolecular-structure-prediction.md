# AlphaFold 3: Unified Biomolecular Structure Prediction via Diffusion

**Source**: Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., Bodenstein, S. W., Evans, D. A., Hung, C. C., O'Neill, M., Reiman, D., Tunyasuvunakool, K., Wu, Z., Zemgulyte, A., Arvaniti, E., ... Jumper, J. M. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630(8016), 493--500. https://doi.org/10.1038/s41586-024-07487-w

**Category**: Bioinformatics / Structural Biology

## Biological / Computational Problem

Proteins function through interactions with diverse molecular partners: other proteins, nucleic acids (DNA/RNA), small-molecule ligands, ions, and post-translational modifications. Predicting the 3D structure of these biomolecular complexes from sequence alone is a grand challenge in computational biology. AlphaFold 2 (2021) solved single-protein structure prediction; the next frontier was predicting **multi-component biomolecular complexes** with atomic accuracy.

- **Input data**: Amino acid sequences (protein chains), nucleic acid sequences (if present), ligand SMILES strings (if present), optionally known residue modifications
- **Output**: 3D coordinates for all atoms in the biomolecular complex, per-residue confidence metrics (pLDDT, pTM, ipTM)

## Mathematical / Computational Model

### Architecture Overview

AlphaFold 3 introduces three major architectural changes from AlphaFold 2:

```
Input sequences + templates
  |
  ├─ Pairformer (replaces Evoformer)
  |   Simplified MSA processing; only pairwise representation propagated
  |   Output: single representation (s_i) + pair representation (z_ij)
  |
  ├─ Diffusion Module (replaces Structure Module)
  |   Denoises random atomic coordinates into predicted positions
  |   Operates on all-atom representations (protein + nucleic acid + ligand)
  |
  └─ Confidence Head
      Predicts pLDDT, pTM, ipTM metrics
```

### Pairformer Module

The Pairformer reduces the complexity of the Evoformer from AlphaFold 2. Key changes:

1. **No recycling of pair representations from structure predictions** (AF2 recycled both MSA and pair representations)
2. **Only the pair representation** $z_{ij}$ is retained for downstream processing (AF2 kept both MSA and pair representations)
3. **Shared weights** across a reduced number of blocks (4 Pairformer blocks vs. 48 Evoformer blocks in AF2)

The Pairformer processes the MSA representation $m_{si}$ and pair representation $z_{ij}$:

$$m_{si}^{(t+1)} = m_{si}^{(t)} + \text{MSAUpdate}(m_{s}^{(t)}, z^{(t)})_{si}$$
$$z_{ij}^{(t+1)} = z_{ij}^{(t)} + \text{PairUpdate}(m^{(t+1)}, z^{(t)})_{ij}$$

### Diffusion Module

The most transformative change is replacing the geometric structure module with a **diffusion-based generative model**:

**Forward process (training only)**: Gradually add noise to true atomic coordinates over $T$ timesteps:

$$q(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}) = \mathcal{N}(\mathbf{x}^{(t)}; \sqrt{1 - \beta_t} \mathbf{x}^{(t-1)}, \beta_t \mathbf{I})$$

**Reverse process (sampling)**: Starting from random noise, iteratively denoise to recover the structure:

$$p_\theta(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}, \text{features}) = \mathcal{N}(\mathbf{x}^{(t-1)}; \mu_\theta(\mathbf{x}^{(t)}, t, \text{features}), \sigma_t^2 \mathbf{I})$$

The denoising network $\mu_\theta$ is conditioned on the Pairformer outputs, enabling the diffusion process to generate structures consistent with input sequences.

**Token-level representation**: Unlike AF2 which used per-residue frames (N, CA, C atoms), AF3-based diffusion operates on a **token-level representation** where:
- Each protein residue = 1 token
- Each nucleic acid residue = 1 token  
- Each ligand heavy atom = 1 token
- Total tokens typically range from 1,000 to 4,000

**Cross-distillation**: Training data is augmented with AF2-Multimer predictions to reduce hallucination in disordered regions and improve generalization.

### Confidence Metrics

The model predicts several quality scores:
- **pLDDT**: Per-residue predicted Local Distance Difference Test (confidence in local structure)
- **pTM**: Predicted Template Modeling score (global fold confidence)
- **ipTM**: Interface pTM (confidence in predicted interactions)

### Loss Function

The training loss combines diffusion denoising and confidence prediction:

$$\mathcal{L} = \mathbb{E}_{t, \epsilon}[\|\epsilon - \epsilon_\theta(\mathbf{x}^{(t)}, t, \text{features})\|^2] + \lambda \cdot \mathcal{L}_{\text{confidence}}$$

where $\epsilon$ is the added noise, $\epsilon_\theta$ is the predicted noise (standard diffusion loss), and $\mathcal{L}_{\text{confidence}}$ is a cross-entropy loss for discretized pLDDT/pTM values.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Static single conformation | Output is a single structure | Does not capture conformational ensembles or dynamics |
| Sufficient coevolutionary signal | MSA depth adequate | Fails for orphan proteins, fast-evolving systems |
| Ligand represented as SMILES/graph | Ligand tokens learnable | Generalizes to novel ligands but requires full-atom representation |
| Diffusion is equivariant-free | Noise added in global frame | Requires cross-distillation; may produce stereochemical errors |

## Applicable Scenarios

**When to use**:
- Predicting protein-ligand complex structures (drug discovery)
- Antibody-antigen complex modeling
- Protein-nucleic acid complex prediction
- Multi-chain protein complex assembly
- Understanding mutation effects on complex formation

**When NOT to use**:
- Predicting protein dynamics or multiple conformations
- Fold-switching proteins (known failure case)
- Very large complexes (>10 chains, >4,000 tokens)
- Systems requiring accurate stereochemistry (4.4% chirality errors reported)
- Cryo-EM or experimental structure refinement

**Comparison**: AF3 achieves 76.4% success on PoseBusters protein-ligand benchmark (vs. 52.3% for Vina), 63% success on antibody-antigen (vs. ~30% for AF-Multimer), and outperforms all automated methods on the CASP15 RNA structure challenge.

## Implementation Details

- **Key parameters**: 4 Pairformer blocks, diffusion with 200 denoising steps, 1,536-dimensional pair representation
- **Computational requirements**: GPU inference ~10--60 seconds per complex; training on 128 TPUv5e chips
- **Preprocessing**:
  - MSA construction via MMseqs2 against BIG-FAST database (2.5B sequences)
  - Template search against PDB (70% sequence identity filtered)
  - Ligand parameterization via RDKit or custom graph featurization
  - Input tokens: protein residues, RNA/DNA bases, ligand atoms (max ~4,000 tokens)

## Python Implementation

```python
"""
Minimal implementation of AlphaFold 3-style diffusion-based structure prediction.

This provides a simplified diffusion model for protein structure prediction,
demonstrating the core diffusion mechanism with a toy representation.
For real usage, see the official AlphaFold 3 repository.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


def simulate_protein_features(
    n_residues: int = 50,
    n_templates: int = 5,
    seed: int = 42,
):
    """
    Generate synthetic protein features mimicking what the Pairformer outputs.
    
    Parameters
    ----------
    n_residues : int
        Number of residues.
    n_templates : int
        Number of template structures.
    
    Returns
    -------
    single_repr : Tensor (n_residues, d_single)
        Per-residue features.
    pair_repr : Tensor (n_residues, n_residues, d_pair)
        Pairwise features.
    true_coords : Tensor (n_residues * 4, 3)  [N, CA, C, O per residue]
        Ground truth coordinates (synthetic).
    """
    rng = np.random.default_rng(seed)
    d_single = 64
    d_pair = 32
    
    # Random single representation
    single_repr = torch.tensor(rng.normal(0, 0.5, size=(n_residues, d_single)),
                                dtype=torch.float32)
    
    # Random pair representation (symmetric)
    pair_repr = torch.tensor(rng.normal(0, 0.3, size=(n_residues, n_residues, d_pair)),
                              dtype=torch.float32)
    pair_repr = (pair_repr + pair_repr.transpose(0, 1)) / 2.0
    
    # Generate a synthetic "true structure" (C-alpha trace)
    ca_coords = np.zeros((n_residues, 3))
    bond_length = 3.8
    
    angles = rng.uniform(-1.5, 1.5, size=n_residues - 1)
    dihedrals = rng.uniform(-np.pi, np.pi, size=n_residues - 2)
    
    for i in range(1, n_residues):
        if i == 1:
            ca_coords[i] = ca_coords[i-1] + np.array([bond_length, 0, 0])
        elif i == 2:
            theta = angles[i-1]
            ca_coords[i] = ca_coords[i-1] + np.array([
                bond_length * np.cos(theta), bond_length * np.sin(theta), 0
            ])
        else:
            theta = angles[i-1]
            phi = dihedrals[i-2]
            
            prev_vec = ca_coords[i-1] - ca_coords[i-2]
            prev_prev_vec = ca_coords[i-2] - ca_coords[i-3]
            
            n_vec = np.cross(prev_vec, prev_prev_vec)
            n_norm = np.linalg.norm(n_vec)
            if n_norm > 1e-8:
                n_vec = n_vec / n_norm
            else:
                n_vec = np.array([0, 0, 1])
            
            b_vec = np.cross(n_vec, prev_vec) / (np.linalg.norm(prev_vec) + 1e-8)
            
            prev_norm = prev_vec / np.linalg.norm(prev_vec)
            direction = (np.cos(theta) * prev_norm +
                         np.sin(theta) * np.cos(phi) * b_vec +
                         np.sin(theta) * np.sin(phi) * n_vec)
            direction = direction / np.linalg.norm(direction)
            ca_coords[i] = ca_coords[i-1] + direction * bond_length
    
    # Add N, C, O backbone atoms (offset from CA)
    true_coords = np.zeros((n_residues * 4, 3))
    for i in range(n_residues):
        true_coords[i * 4] = ca_coords[i]  # CA
        if i > 0:
            # N: between previous C and current CA
            prev_ca = ca_coords[i-1] if i > 0 else ca_coords[i] - np.array([3.8, 0, 0])
            true_coords[i * 4] = (prev_ca + ca_coords[i]) / 2 + rng.normal(0, 0.2, 3)
        # Shift CA, N, C, O positions slightly
        for j in range(4):
            true_coords[i * 4 + j] += rng.normal(0, 0.1, 3)
    
    return {
        "single_repr": single_repr,
        "pair_repr": pair_repr,
        "true_coords": torch.tensor(true_coords, dtype=torch.float32),
        "n_residues": n_residues,
        "n_atoms": n_residues * 4,
    }


class DiffusionSchedule:
    """
    Cosine noise schedule for the diffusion process.
    """
    
    def __init__(self, n_timesteps: int = 200, s: float = 0.008):
        self.n_timesteps = n_timesteps
        self.s = s
        
        # Precompute beta schedule (cosine)
        t = torch.linspace(0, n_timesteps, n_timesteps + 1)
        f_t = torch.cos((t / n_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        
        self.betas = torch.clamp(
            1 - alpha_bar[1:] / alpha_bar[:-1], max=0.999
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = alpha_bar[1:]
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        
    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple:
        """
        Forward diffusion: add noise at timestep t.
        
        Parameters
        ----------
        x_0 : Tensor (B, N, 3)
            Clean coordinates.
        t : Tensor (B,)
            Timestep indices.
        
        Returns
        -------
        x_t : Tensor (B, N, 3)
            Noised coordinates.
        noise : Tensor (B, N, 3)
            Added noise.
        """
        noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_1m_ab * noise
        return x_t, noise


class DenoisingNetwork(nn.Module):
    """
    Simplified denoising network for diffusion-based structure prediction.
    
    Takes noised coordinates and conditions (from Pairformer) and predicts
    the noise to remove (epsilon-prediction).
    """
    
    def __init__(
        self,
        d_single: int = 64,
        d_pair: int = 32,
        d_time: int = 32,
        n_blocks: int = 4,
        n_atoms_per_res: int = 4,
    ):
        super().__init__()
        self.n_atoms_per_res = n_atoms_per_res
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time),
        )
        
        # Coordinate embedding
        self.coord_proj = nn.Linear(3, d_single)
        
        # Invariant point attention blocks (simplified)
        self.blocks = nn.ModuleList([
            InvariantBlock(d_single, d_pair, d_time)
            for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_single, d_single),
            nn.SiLU(),
            nn.Linear(d_single, 3),  # Predict delta coordinates
        )
        
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        single_repr: torch.Tensor,
        pair_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise to remove from noised coordinates.
        
        Parameters
        ----------
        x_t : Tensor (B, N_atoms, 3)
            Noised coordinates.
        t : Tensor (B,)
            Timestep.
        single_repr : Tensor (B, n_res, d_single)
        pair_repr : Tensor (B, n_res, n_res, d_pair)
        
        Returns
        -------
        pred_noise : Tensor (B, N_atoms, 3)
        """
        B, N_atoms, _ = x_t.shape
        n_res = single_repr.shape[1]
        
        # Time embedding
        t_emb = self.time_embed(t.float().unsqueeze(-1))  # (B, d_time)
        
        # Coordinate embedding
        coord_feat = self.coord_proj(x_t)  # (B, N_atoms, d_single)
        
        # Expand single_repr to atom level
        # Each residue contributes 4 atoms (N, CA, C, O)
        single_atom = single_repr.repeat_interleave(self.n_atoms_per_res, dim=1)
        
        # Combine with coordinate features
        h = single_atom + coord_feat
        
        # Expand pair representation to atom-level (approximation: same pair for all atoms)
        pair_expanded = pair_repr.repeat_interleave(
            self.n_atoms_per_res, dim=1
        ).repeat_interleave(self.n_atoms_per_res, dim=2)
        
        # Process through invariant blocks
        for block in self.blocks:
            h = block(h, pair_expanded, t_emb)
        
        # Predict per-atom delta coordinates
        pred_delta = self.output_proj(h)  # (B, N_atoms, 3)
        
        return pred_delta


class InvariantBlock(nn.Module):
    """
    Simplified invariant point attention block.
    
    Operates on per-atom features with pairwise bias from pair representation.
    """
    
    def __init__(self, d_single: int, d_pair: int, d_time: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_single)
        self.self_attn = nn.MultiheadAttention(
            d_single, num_heads=4, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_single)
        self.ffn = nn.Sequential(
            nn.Linear(d_single + d_time, d_single * 4),
            nn.SiLU(),
            nn.Linear(d_single * 4, d_single),
        )
        
        # Pair bias projection
        self.pair_bias = nn.Linear(d_pair, 4)  # one bias per head
        
    def forward(
        self,
        h: torch.Tensor,
        pair_repr: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : Tensor (B, N_atoms, d_single)
        pair_repr : Tensor (B, N_atoms, N_atoms, d_pair)
        t_emb : Tensor (B, d_time)
        """
        # Self-attention with pair bias
        h_norm = self.norm1(h)
        
        # Compute attention bias from pair representation
        # pair_bias: (B, N, N, heads) -> (B, heads, N, N)
        pair_bias = self.pair_bias(pair_repr).permute(0, 3, 1, 2)
        
        # Multi-head attention (applying bias via attention mask)
        attn_out, _ = self.self_attn(
            h_norm, h_norm, h_norm,
            attn_mask=None,
            need_weights=False,
        )
        
        # Apply pair bias manually (simplified: just add after attention)
        bias = pair_bias.mean(dim=1)  # average over heads -> (B, N, N)
        bias_weighted = (bias @ h_norm) / math.sqrt(h_norm.shape[-1])
        h = h + attn_out + bias_weighted
        
        # FFN with time embedding
        h_norm2 = self.norm2(h)
        t_expand = t_emb.unsqueeze(1).expand(-1, h.shape[1], -1)
        ffn_in = torch.cat([h_norm2, t_expand], dim=-1)
        h = h + self.ffn(ffn_in)
        
        return h


class SimplifiedAlphaFold3(nn.Module):
    """
    Simplified AlphaFold 3: diffusion-based structure prediction.
    
    Combines a Pairformer-like conditioning network with a diffusion
    denoising module.
    """
    
    def __init__(
        self,
        d_single: int = 64,
        d_pair: int = 32,
        d_time: int = 32,
        n_diffusion_steps: int = 200,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.n_diffusion_steps = n_diffusion_steps
        self.noise_schedule = DiffusionSchedule(n_diffusion_steps)
        
        # Conditioning network (simplified Pairformer)
        self.cond_network = nn.Sequential(
            nn.Linear(d_single, d_single),
            nn.ReLU(),
            nn.Linear(d_single, d_single),
        )
        
        # Denoising network
        self.denoiser = DenoisingNetwork(
            d_single=d_single,
            d_pair=d_pair,
            d_time=d_time,
            n_blocks=n_blocks,
        )
        
    def forward(
        self,
        single_repr: torch.Tensor,
        pair_repr: torch.Tensor,
    ) -> dict:
        """
        Training step: add noise, predict it back.
        
        Parameters
        ----------
        single_repr : Tensor (B, n_res, d_single)
        pair_repr : Tensor (B, n_res, n_res, d_pair)
        
        Returns
        -------
        dict with loss and predictions.
        """
        # Condition the representation
        cond = self.cond_network(single_repr)
        
        # The method would normally take true coordinates as input.
        # This is called during training with ground truth structures.
        # During inference, we start from noise and denoise iteratively.
        return {
            "cond": cond,
        }
    
    @torch.no_grad()
    def sample(
        self,
        single_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        n_atoms: int = None,
        n_steps: int = None,
    ) -> torch.Tensor:
        """
        Sample structure by iteratively denoising from random noise.
        
        Parameters
        ----------
        single_repr : Tensor (B, n_res, d_single)
        pair_repr : Tensor (B, n_res, n_res, d_pair)
        n_atoms : int
            Number of atoms.
        n_steps : int
            Number of denoising steps (default: n_diffusion_steps).
        
        Returns
        -------
        Tensor (B, N_atoms, 3)
            Predicted coordinates.
        """
        if n_steps is None:
            n_steps = self.n_diffusion_steps
        
        B = single_repr.shape[0]
        if n_atoms is None:
            n_atoms = single_repr.shape[1] * 4
        
        # Start from random noise
        x = torch.randn(B, n_atoms, 3, device=single_repr.device)
        
        # Annealed Langevin / DDIM sampling
        timesteps = torch.linspace(
            self.n_diffusion_steps - 1, 1, n_steps, dtype=torch.long
        )
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=single_repr.device)
            
            # Predict noise
            pred = self.denoiser(x, t, single_repr, pair_repr)
            
            # Update (simplified DDPM step)
            alpha = self.noise_schedule.alphas[t_val]
            alpha_bar = self.noise_schedule.alpha_bars[t_val]
            beta = self.noise_schedule.betas[t_val]
            
            if t_val > 1:
                noise = torch.randn_like(x)
            else:
                noise = 0.0
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * pred
            ) + torch.sqrt(beta) * noise
        
        return x


class DiffusionTrainer:
    """
    Trainer for the AlphaFold 3-style diffusion model.
    """
    
    def __init__(self, model: SimplifiedAlphaFold3, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=1e-5)
        
    def train_step(
        self,
        true_coords: torch.Tensor,
        single_repr: torch.Tensor,
        pair_repr: torch.Tensor,
    ) -> float:
        """
        Single training step: add noise to coordinates, predict noise.
        
        Parameters
        ----------
        true_coords : Tensor (B, N_atoms, 3)
        single_repr : Tensor (B, n_res, d_single)
        pair_repr : Tensor (B, n_res, n_res, d_pair)
        
        Returns
        -------
        float loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        B = true_coords.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.model.n_diffusion_steps, (B,))
        
        # Forward diffusion
        x_t, noise = self.model.noise_schedule.q_sample(true_coords, t)
        
        # Predict noise
        pred_noise = self.model.denoiser(x_t, t, single_repr, pair_repr)
        
        # Simple MSE loss on noise prediction
        loss = F.mse_loss(pred_noise, noise)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()


# ============================================================================
# Complete usage example
# ============================================================================

def main():
    """
    Run a complete AlphaFold 3-style diffusion demonstration.
    """
    print("=" * 60)
    print("AlphaFold 3: Diffusion-Based Structure Prediction")
    print("=" * 60)
    
    # --- 1. Simulate protein data ---
    print("\n[1] Generating synthetic protein data...")
    data = simulate_protein_features(n_residues=40)
    
    print(f"    Residues: {data['n_residues']}")
    print(f"    Atoms: {data['n_atoms']}")
    print(f"    Single representation: {data['single_repr'].shape}")
    print(f"    Pair representation: {data['pair_repr'].shape}")
    
    # Add batch dimension
    single_repr = data["single_repr"].unsqueeze(0)  # (1, N, d)
    pair_repr = data["pair_repr"].unsqueeze(0)  # (1, N, N, d)
    true_coords = data["true_coords"].unsqueeze(0)  # (1, N_atoms, 3)
    
    # --- 2. Create AlphaFold 3-style model ---
    print("\n[2] Initializing diffusion model...")
    model = SimplifiedAlphaFold3(
        d_single=64,
        d_pair=32,
        d_time=16,
        n_diffusion_steps=100,
        n_blocks=3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {n_params:,}")
    
    # --- 3. Train diffusion model ---
    print("\n[3] Training diffusion model...")
    trainer = DiffusionTrainer(model, lr=5e-4)
    
    n_epochs = 50
    for epoch in range(n_epochs):
        loss = trainer.train_step(true_coords, single_repr, pair_repr)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, loss: {loss:.5f}")
    
    # --- 4. Demonstrate diffusion forward process ---
    print("\n[4] Forward diffusion process (adding noise)...")
    noise_schedule = DiffusionSchedule(n_timesteps=50)
    
    noise_levels = [0, 10, 25, 40, 49]
    for nl in noise_levels:
        t = torch.tensor([nl])
        x_t, _ = noise_schedule.q_sample(true_coords, t)
        rmsd = torch.sqrt(torch.mean((x_t - true_coords) ** 2)).item()
        print(f"    t={nl}: RMSD from native = {rmsd:.2f} A")
    
    # --- 5. Generate structure from random noise ---
    print("\n[5] Generating structure from random noise (sampling)...")
    gen_coords = model.sample(
        single_repr, pair_repr,
        n_atoms=data["n_atoms"],
        n_steps=50,
    )
    
    # Compare to ground truth
    rmsd = torch.sqrt(
        torch.mean((gen_coords - true_coords) ** 2)
    ).item()
    print(f"    Generated vs. true RMSD: {rmsd:.2f} A")
    
    # --- 6. Self-distillation analysis ---
    print("\n[6] Self-distillation analysis...")
    # Generate slightly different structures
    coords_list = []
    for _ in range(5):
        c = model.sample(
            single_repr, pair_repr,
            n_atoms=data["n_atoms"],
            n_steps=30,
        )
        coords_list.append(c)
    
    # Pairwise RMSD between generated structures
    rmsds = []
    for i in range(5):
        for j in range(i + 1, 5):
            r = torch.sqrt(
                torch.mean((coords_list[i] - coords_list[j]) ** 2)
            ).item()
            rmsds.append(r)
    
    print(f"    Mean pairwise RMSD between samples: {np.mean(rmsds):.2f} A")
    print(f"    Std pairwise RMSD between samples: {np.std(rmsds):.2f} A")
    
    print("\n" + "=" * 60)
    print("AlphaFold 3 demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## References

Abramson, J., Adler, J., Dunger, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630(8016), 493--500. https://doi.org/10.1038/s41586-024-07487-w

Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583--589. https://doi.org/10.1038/s41586-021-03819-2

Watson, J. L., Juergens, D., Bennett, N. R., et al. (2023). De novo design of protein structure and function with RFdiffusion. *Nature*, 620, 1089--1100. https://doi.org/10.1038/s41586-023-06415-8

Krishna, R., Wang, J., Ahern, W., et al. (2024). Generalized biomolecular modeling and design with RoseTTAFold All-Atom. *Science*, 384(6693), eadl2528. https://doi.org/10.1126/science.adl2528

Chakravarty, D., Schafer, J. W., Chen, E. A., et al. (2024). AlphaFold predictions of fold-switched conformations are driven by structure memorization. *Nature Communications*, 15, 7296. https://doi.org/10.1038/s41467-024-51801-z
