# Fourier Neural Operator (FNO) for Learning PDE Solutions

**Source**: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2010.08895

**Category**: Machine Learning / Scientific Machine Learning / Neural Operators

## Mathematical Setup

### Problem: Learning Solution Operators for PDEs

Consider a parametric family of partial differential equations (PDEs). Let $\mathcal{A}$ be the space of input functions (e.g., initial conditions, forcing terms) and $\mathcal{U}$ be the space of output functions (e.g., solutions). The goal is to learn the **solution operator** $\mathcal{G}^\dagger: \mathcal{A} \to \mathcal{U}$ mapping from any input function $a \in \mathcal{A}$ to the corresponding solution $u \in \mathcal{U}$.

Given observations $\{(a_j, u_j)\}_{j=1}^N$ where $u_j = \mathcal{G}^\dagger(a_j)$, we seek a parametric approximation $\mathcal{G}_\theta \approx \mathcal{G}^\dagger$.

### Operator Learning Framework

Neural operators approximate the solution operator through an iterative architecture consisting of:

1. **Lifting**: $v_0(x) = P(a(x))$ where $P: \mathbb{R}^{d_a} \to \mathbb{R}^{d_v}$ is a pointwise linear map.
2. **Iterative updates**: $v_{t+1}(x) = \sigma(W v_t(x) + (\mathcal{K} v_t)(x))$ for $t = 0, \ldots, T-1$.
3. **Projection**: $u(x) = Q(v_T(x))$ where $Q: \mathbb{R}^{d_v} \to \mathbb{R}^{d_u}$ is a pointwise linear map.

Here $\sigma$ is a nonlinear activation, $W$ is a pointwise linear transform, and $\mathcal{K}$ is a **kernel integral operator**.

### Fourier Kernel Integral Operator -- Key Innovation

The standard kernel integral operator is:

$$
(\mathcal{K} v_t)(x) = \int_D \kappa(x, y) v_t(y) \, dy
$$

This has O($N^2$) cost for $N$ discretization points. The FNO replaces this with a **convolution operator parameterized in Fourier space**:

$$
(\mathcal{K} v_t)(x) = \mathcal{F}^{-1}\left( R_\phi \cdot (\mathcal{F} v_t) \right)(x)
$$

where:
- $\mathcal{F}$ is the Fourier transform,
- $R_\phi$ is a learnable complex-valued weight matrix in Fourier space,
- $\mathcal{F}^{-1}$ is the inverse Fourier transform.

The **Fourier layer** is:

$$
v_{t+1}(x) = \sigma\left( W v_t(x) + \mathcal{F}^{-1}\left( R_\phi \cdot \mathcal{F}(v_t) \right)(x) \right)
$$

### Discretization

For discrete data on an $n \times n$ grid, the 2D FFT is O($n^2 \log n$) and we keep only the $k_{\max}$ lowest-frequency modes (typically $k_{\max} = 8{\!-\!}16$):

$$
(\mathcal{K} v_t)(x) = \text{IFFT}_{k_{\max}}\left( R_\phi \cdot \text{FFT}(v_t) \right)(x)
$$

where $R_\phi \in \mathbb{C}^{k_{\max} \times d_v \times d_v}$ is a complex tensor parameterizing the kernel in Fourier space.

### Training Objective

Minimize the mean squared error between predicted and true solutions:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{j=1}^N \| \mathcal{G}_\theta(a_j) - u_j \|_2^2
$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| **Translation equivariance** | The kernel depends only on $x - y$, i.e., $\kappa(x, y) = \kappa(x - y)$ | The Fourier convolution becomes a multiplication; FNO is a global convolution operator |
| **Smoothness of solutions** | The solution function has a compact representation in low Fourier frequencies | Low-frequency truncation ($k_{\max}$) preserves most information; FNO works best for smooth solutions |
| **Grid-independence** | The same neural operator works on different discretizations | Once trained, FNO can be evaluated on any resolution without retraining |
| **Periodic boundary conditions** | $\mathcal{F}$ assumes periodic data | Non-periodic boundaries require padding or mirroring; accuracy degrades near boundaries |

## Applicable Scenarios

**When to use:**
- Solving parametric PDEs where you need fast inference (millions of evaluations)
- Training on low-resolution data and evaluating at high resolution (super-resolution capability)
- Steady-state or time-dependent PDEs with smooth solutions (Navier-Stokes, Darcy flow, Burgers)
- Any-to-any mapping between function spaces (not just PDEs -- can be used for weather, climate, etc.)

**When NOT to use:**
- Solutions with sharp discontinuities (shocks, interfaces) -- these require high-frequency modes that are truncated
- Problems with strong spatial inhomogeneity -- translational equivariance breaks down
- Small datasets where the model cannot learn the spectral representation
- When exact conservation laws are critical (FNO may not conserve mass/energy exactly)

**Comparison:** FNO is orders of magnitude faster than classical PDE solvers at inference time. Compared to other neural operator methods (DeepONet, GNN-based), FNO has better resolution transfer and is generally more parameter-efficient for problems with smooth solutions.

## Algorithm / Method Details

### Architecture

```
Input: a(x), coordinate grid positions
  |
  v0 = P(a(x))                  # Lifting: pointwise linear (dx -> dv)
  |
  for t = 1..T:
    ft = FFT(v_{t-1})           # Fast Fourier Transform
    ft_filtered = ft[:, :k_max] # Truncate high frequencies
    ft_weighted = R_t @ ft      # Multiply by learned weights
    vt_ft = IFFT(ft_weighted)   # Inverse FFT
    vt_local = W_t * v_{t-1}    # Local linear transform
    vt = sigma(vt_ft + vt_local) # Nonlinear activation
  |
  u(x) = Q(v_T)                 # Projection: pointwise linear (dv -> du)
  |
Output: u(x) approximate solution
```

### Training Procedure

1. Sample input functions $a_j$ from the distribution (e.g., random Gaussian fields for permeability).
2. Solve PDE numerically (FEM, FDM) to obtain ground truth solutions $u_j$.
3. Pair $(a_j, u_j)$ as training data.
4. Train FNO with AdamW optimizer, learning rate 1e-3, cosine decay.
5. Evaluate relative L2 error on test set.

### Complexity Analysis

- **Time**: O($n \log n$) per Fourier layer (FFT), O($n d_v^2 k_{\max}$) for spectral multiplication.
- **Memory**: O($n d_v + d_v^2 k_{\max}$) per Fourier layer.
- **Inference**: 3-5 orders of magnitude faster than traditional PDE solvers.

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| Number of Fourier layers $T$ | 4 | More layers increase receptive field; 4-5 is standard |
| Hidden dimension $d_v$ | 32-64 | Larger for more complex problems |
| Modes $k_{\max}$ | 12 per dimension | Determines frequency cutoff; larger for high-frequency solutions |
| Activation | GELU or ReLU | GELU slightly better for smooth solutions |
| Padding | 2-4 points per dimension | Helps with non-periodic boundary conditions |

### Numerical Considerations

- The FFT treats the grid as periodic. For non-periodic conditions, pad the input by 2-4 points per dimension and crop after FNO.
- Use `torch.fft.rfft` for real-valued data to save memory and compute.
- The spectral weight $R_\phi$ is complex-valued. In PyTorch, use complex dtypes (`torch.complex64`).
- Normalize inputs and outputs to zero mean and unit variance for stable training.

## Python Implementation

```python
"""
Minimal implementation of the Fourier Neural Operator (FNO).
Based on: Li et al. (2021) https://arxiv.org/abs/2010.08895
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple


# ============================================================
# Fourier Layer
# ============================================================

class SpectralConv2d(nn.Module):
    """
    2D Fourier convolution layer.

    Performs: v -> IFFT(R * FFT(v)) where R is learned in Fourier space.
    Only the lowest k_max modes are kept for efficiency.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int = 12):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to keep in each dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of modes in x and y

        # Complex weights for the Fourier modes
        # shape: (out_channels, in_channels, modes, modes)
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(out_channels, in_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(out_channels, in_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, height, width)

        Returns:
            y: (batch, out_channels, height, width)
        """
        batch, _, h, w = x.shape

        # FFT: (batch, in_channels, h, w)
        x_ft = torch.fft.rfft2(x)

        # We only need the lowest modes (handling conjugate symmetry)
        # The output is complex: (batch, in_channels, h, w//2 + 1)
        _, _, h_ft, w_ft = x_ft.shape

        # Initialize output Fourier coefficients
        out_ft = torch.zeros(batch, self.out_channels, h, w // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Multiply by learned weights for the first M modes
        # Top-left corner: first modes x first modes
        m = min(self.modes, h_ft)
        n = min(self.modes, w_ft)
        out_ft[:, :, :m, :n] = torch.einsum(
            "oixy,bixy->boxy",
            self.weights1[:, :, :m, :n],
            x_ft[:, :, :m, :n],
        )

        # Handle the conjugate-symmetric part: bottom-right would be
        # the complex conjugate of top-right, so we skip to avoid redundancy
        # But for completeness, we multiply the conjugate part
        m2 = min(self.modes, h_ft)
        n2 = min(self.modes, w_ft - n)
        if n2 > 0:
            out_ft[:, :, -m2:, -n2:] = torch.einsum(
                "oixy,bixy->boxy",
                self.weights2[:, :, :m2, :n2],
                x_ft[:, :, -m2:, -n2:],
            )

        # Inverse FFT
        y = torch.fft.irfft2(out_ft, s=(h, w))
        return y


class FourierLayer(nn.Module):
    """
    Single Fourier layer: v -> sigma(Wv + K(v))

    K(v): Fourier convolution
    W(v): pointwise linear transform
    """
    def __init__(self, channels: int, modes: int = 12, activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes=modes)
        self.w = nn.Conv2d(channels, channels, 1)  # pointwise linear
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral(x)
        x2 = self.w(x)
        return self.activation(x1 + x2)


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.

    Maps input function a(x, y) to output function u(x, y)
    via lifting -> Fourier layers -> projection.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
        modes: int = 12,
        lifting_dim: int = 64,
        activation: str = "gelu",
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Width of the Fourier layers
            n_layers: Number of Fourier layers
            modes: Number of Fourier modes to keep
            lifting_dim: Width of the lifting layer
            activation: Activation function
        """
        super().__init__()

        # Lifting: pointwise linear from in_channels to lifting_dim
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, lifting_dim, 1),
            nn.GELU(),
            nn.Conv2d(lifting_dim, hidden_channels, 1),
        )

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(hidden_channels, modes, activation)
            for _ in range(n_layers)
        ])

        # Projection: pointwise linear from hidden_channels to out_channels
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, lifting_dim, 1),
            nn.GELU(),
            nn.Conv2d(lifting_dim, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, height, width) input function on grid

        Returns:
            u: (batch, out_channels, height, width) predicted solution
        """
        x = self.lifting(x)
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.projection(x)
        return x


# ============================================================
# Data Generation: 2D Darcy Flow
# ============================================================

def darcy_flow_fem(
    permeability: np.ndarray,
    forcing: float = 1.0,
    resolution: int = 64,
) -> np.ndarray:
    """
    Solve Darcy flow equation using finite differences:
        -div(k(x) * grad(u)) = f    on [0,1]^2
        u = 0                       on boundary

    Args:
        permeability: (H, W) permeability field k(x)
        forcing: Source term f (constant)
        resolution: Grid resolution

    Returns:
        u: (H, W) pressure field
    """
    h = 1.0 / (resolution + 1)
    N = resolution
    k = permeability

    # Build the system matrix (5-point stencil for variable coefficient)
    # -div(k * grad(u)) at (i,j):
    #   = -[k_{i+1/2,j}(u_{i+1,j} - u_{i,j}) - k_{i-1/2,j}(u_{i,j} - u_{i-1,j})] / h^2
    #     -[k_{i,j+1/2}(u_{i,j+1} - u_{i,j}) - k_{i,j-1/2}(u_{i,j} - u_{i,j-1})] / h^2

    # Harmonic mean for interface permeabilities
    k_xp = np.zeros((N, N))  # k_{i+1/2, j}
    k_xm = np.zeros((N, N))  # k_{i-1/2, j}
    k_yp = np.zeros((N, N))  # k_{i, j+1/2}
    k_ym = np.zeros((N, N))  # k_{i, j-1/2}

    k_xp[:, :-1] = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:] + 1e-10)
    k_xp[:, -1] = 2.0 * k[:, -1] * 1.0 / (k[:, -1] + 1.0)
    k_xm[:, 1:] = k_xp[:, :-1]
    k_xm[:, 0] = 2.0 * k[:, 0] * 1.0 / (k[:, 0] + 1.0)

    k_yp[:-1, :] = 2.0 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :] + 1e-10)
    k_yp[-1, :] = 2.0 * k[-1, :] * 1.0 / (k[-1, :] + 1.0)
    k_ym[1:, :] = k_yp[:-1, :]
    k_ym[0, :] = 2.0 * k[0, :] * 1.0 / (k[0, :] + 1.0)

    # Build sparse matrix
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve

    n_dofs = N * N
    A = lil_matrix((n_dofs, n_dofs))
    F = np.full(n_dofs, forcing * h * h)

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            diag = 0.0
            # West neighbor
            if j > 0:
                val = k_xm[i, j] / h**2
                A[idx(i, j), idx(i, j - 1)] = -val
                diag += val
            # East neighbor
            if j < N - 1:
                val = k_xp[i, j] / h**2
                A[idx(i, j), idx(i, j + 1)] = -val
                diag += val
            # South neighbor
            if i > 0:
                val = k_ym[i, j] / h**2
                A[idx(i, j), idx(i - 1, j)] = -val
                diag += val
            # North neighbor
            if i < N - 1:
                val = k_yp[i, j] / h**2
                A[idx(i, j), idx(i + 1, j)] = -val
                diag += val
            A[idx(i, j), idx(i, j)] = diag

    A = A.tocsr()
    u_flat = spsolve(A, F)
    return u_flat.reshape(N, N)


def generate_darcy_dataset(
    n_samples: int = 1000,
    resolution: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Darcy flow dataset with random permeability fields.

    Permeability is modeled as exp(Gaussian random field).

    Args:
        n_samples: Number of samples
        resolution: Grid resolution

    Returns:
        permeability: (n_samples, H, W) permeability fields (inputs)
        pressure: (n_samples, H, W) pressure fields (outputs)
    """
    np.random.seed(seed)

    # Gaussian random field generation using Fourier method
    def grf(n: int, resolution: int, length_scale: float = 0.1) -> np.ndarray:
        """Generate n Gaussian random fields on [0,1]^2."""
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y, indexing="ij")

        fields = []
        for _ in range(n):
            # Fourier sampling
            kx = np.fft.fftfreq(resolution, d=1.0 / resolution)
            ky = np.fft.fftfreq(resolution, d=1.0 / resolution)
            KX, KY = np.meshgrid(kx, ky, indexing="ij")

            # Power spectrum: (1 + (kx^2 + ky^2) * L^2)^(-alpha)
            alpha = 2.0
            power = (1 + (KX**2 + KY**2) * length_scale**2) ** (-alpha)

            # Random phases
            random_phase = np.exp(2j * np.pi * np.random.rand(resolution, resolution))
            field_ft = np.sqrt(power) * random_phase
            field = np.fft.ifft2(field_ft).real
            field = field / np.std(field)
            fields.append(field)

        return np.array(fields)

    # Generate permeability: log-normal random field
    log_k = grf(n_samples, resolution)
    k = np.exp(log_k)

    # Solve Darcy for each sample
    u = np.zeros((n_samples, resolution, resolution))
    for i in range(n_samples):
        if i % 200 == 0:
            print(f"  Solving sample {i+1}/{n_samples}")
        u[i] = darcy_flow_fem(k[i], forcing=1.0, resolution=resolution)

    return k, u


# ============================================================
# Training and evaluation
# ============================================================

def train_fno():
    """Train FNO on Darcy flow data."""
    print("=" * 60)
    print("FNO: Darcy Flow Training Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Generate small dataset for demo
    print("\nGenerating Darcy flow data...")
    n_train = 200
    n_test = 50
    resolution = 32

    k_all, u_all = generate_darcy_dataset(
        n_samples=n_train + n_test,
        resolution=resolution,
        seed=42,
    )

    # Train/test split
    k_train, u_train = k_all[:n_train], u_all[:n_train]
    k_test, u_test = k_all[n_train:], u_all[n_train:]

    print(f"Train: {k_train.shape}, Test: {k_test.shape}")

    # Convert to tensors and normalize
    k_train_t = torch.tensor(k_train[:, None, :, :], dtype=torch.float32)  # (N, 1, H, W)
    u_train_t = torch.tensor(u_train[:, None, :, :], dtype=torch.float32)
    k_test_t = torch.tensor(k_test[:, None, :, :], dtype=torch.float32)
    u_test_t = torch.tensor(u_test[:, None, :, :], dtype=torch.float32)

    # Log-normalize permeability (see paper)
    k_train_t = torch.log(k_train_t + 1e-10)
    k_test_t = torch.log(k_test_t + 1e-10)

    # Normalize targets
    u_mean = u_train_t.mean()
    u_std = u_train_t.std()
    u_train_t = (u_train_t - u_mean) / u_std
    u_test_t = (u_test_t - u_mean) / u_std

    # Create model
    model = FNO2d(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        modes=12,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    n_epochs = 100
    batch_size = 32
    n_batches = len(k_train_t) // batch_size

    print("\nTraining...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        perm = torch.randperm(len(k_train_t))
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            k_batch = k_train_t[idx].to(device)
            u_batch = u_train_t[idx].to(device)

            optimizer.zero_grad()
            pred = model(k_batch)
            loss = criterion(pred, u_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_pred = model(k_test_t.to(device))
                test_loss = criterion(test_pred, u_test_t.to(device))
            print(f"  Epoch {epoch+1}/{n_epochs}, "
                  f"Train Loss: {epoch_loss / n_batches:.6f}, "
                  f"Test Loss: {test_loss.item():.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_all = model(k_test_t.to(device))

    # Relative L2 error
    u_test_np = u_test_t.numpy()
    pred_np = pred_all.cpu().numpy()
    rel_l2 = np.linalg.norm(pred_np - u_test_np) / np.linalg.norm(u_test_np)
    print(f"\nFinal relative L2 error: {rel_l2:.4f}")

    # Resolution transfer test
    print("\n[Resolution Transfer]")
    print("Evaluating on different resolutions (should work without retraining)...")
    for res in [16, 32, 64]:
        k_res, u_res = generate_darcy_dataset(
            n_samples=10, resolution=res, seed=100
        )
        k_res_t = torch.tensor(np.log(k_res + 1e-10)[:, None, :, :], dtype=torch.float32).to(device)
        u_res_t = torch.tensor(u_res[:, None, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_res = model(k_res_t)
        rel_l2_res = np.linalg.norm(
            pred_res.cpu().numpy() - u_res_t.cpu().numpy()
        ) / np.linalg.norm(u_res_t.cpu().numpy())
        print(f"  Resolution {res}x{res}: Relative L2 = {rel_l2_res:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    train_fno()
```

## References

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2010.08895

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Neural operator: Graph kernel network for partial differential equations. *arXiv preprint arXiv:2003.03485*. https://arxiv.org/abs/2003.03485

Lu, L., Jin, P., & Karniadakis, G. E. (2021). DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229. https://arxiv.org/abs/1910.03193

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. *Journal of Machine Learning Research*, 24(89), 1-97. https://jmlr.org/papers/v24/21-1524.html

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
