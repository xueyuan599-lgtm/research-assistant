# Flow Matching for Generative Modeling

**Source**: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2210.02747

**Category**: Machine Learning / Generative Models / Continuous Normalizing Flows

## Mathematical Setup

### Background: Continuous Normalizing Flows (CNFs)

A continuous normalizing flow models a probability distribution $p_1(x)$ by pushing a simple prior $p_0(x)$ through a time-dependent vector field $v_t(x)$ defined by an ordinary differential equation (ODE):

$$
\frac{d}{dt} \phi_t(x) = v_t(\phi_t(x)), \quad \phi_0(x) = x
$$

where $\phi_t$ is a time-dependent diffeomorphism (the flow) that maps points at time $t=0$ to time $t=1$. The probability density evolves according to the **continuity equation** (also known as the push-forward equation):

$$
p_t = [\phi_t]_* p_0 \quad \text{or equivalently} \quad \frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t v_t)
$$

Traditional CNF training requires simulating the ODE and computing expensive log-determinants via the instantaneous change of variable formula (Chen et al., 2018), making it impractical for high-dimensional data.

### Flow Matching Objective

Flow matching simplifies CNF training by directly regressing the vector field $v_t(x; \theta)$ toward a **target vector field** $u_t(x)$ that generates a prescribed probability path $p_t(x)$ between $p_0$ and $p_1$:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], x \sim p_t(x)} \| v_t(x; \theta) - u_t(x) \|^2
$$

However, this objective is intractable because we do not have access to $u_t$ directly.

### Conditional Flow Matching (CFM) -- Key Innovation

The key insight is that one can construct an **unconditional** probability path $p_t$ via a **conditional** probability path $p_t(x \mid z)$ marginalized over a latent variable $z = (x_0, x_1)$:

$$
p_t(x) = \int p_t(x \mid z) q(z) dz
$$

where $q(z)$ is a coupling between the prior and data distributions. Using this, the **conditional flow matching** objective is:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], z \sim q(z), x_t \sim p_t(x \mid z)} \| v_t(x_t; \theta) - u_t(x_t \mid z) \|^2
$$

This objective has the **same gradient** as $\mathcal{L}_{\text{FM}}$ but is tractable because $u_t(x \mid z)$ can be defined analytically.

### Gaussian Probability Paths

The simplest and most effective choice is a **Gaussian conditional path**:

$$
p_t(x \mid z) = \mathcal{N}(x \mid \mu_t(z), \sigma_t(z)^2 I)
$$

where $\mu_t(z)$ is the conditional mean and $\sigma_t(z)$ is the conditional standard deviation. The corresponding **conditional vector field** has a closed form:

$$
u_t(x \mid z) = \frac{\sigma'_t(z)}{\sigma_t(z)} (x - \mu_t(z)) + \mu'_t(z)
$$

### Optimal Transport Conditional Paths

Lipman et al. (2023) propose using **optimal transport (OT) displacement interpolation**:

$$
\mu_t(z) = t x_1 + (1 - t) x_0, \quad \sigma_t(z) = 0 \quad \text{(deterministic path)}
$$

with the conditional vector field $u_t(x \mid z) = x_1 - x_0$ (constant velocity). This yields **straight trajectories**, which minimize transport cost and enable fast ODE solving with few steps.

### Relation to Diffusion Models

Flow matching generalizes diffusion models. The variance-preserving diffusion path is a specific instance of the Gaussian conditional path with:
- $\mu_t(z) = \sqrt{1 - \sigma(t)^2} x_1 + \sigma(t) \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$
- $\sigma_t(z) = \sigma(t)$

This subsumes score matching, with the score function being $\nabla \log p_t(x) = (u_t(x) - v_t(x)) / \sigma_t$ (depending on convention).

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| **Gaussian conditional paths** | $p_t(x \mid z) = \mathcal{N}(x \mid \mu_t(z), \sigma_t(z)^2 I)$ | Closed-form vector field enables simulation-free training |
| **Linear interpolant** | $\mu_t = t x_1 + (1 - t) x_0$ | Straight paths reduce ODE steps during sampling |
| **Independent coupling** | $q(z) = p_0(x_0) p_1(x_1)$ with $x_0 \perp x_1$ | Simplifies sampling but may not be optimal transport |
| **Sufficient model capacity** | Neural network $v_t(x; \theta)$ approximates $u_t(x \mid z)$ | No theoretical guarantee in finite sample regime |

## Applicable Scenarios

**When to use:**
- High-quality generative modeling (images, audio, video, molecular conformations)
- Fast sampling with few function evaluations (1-10 ODE steps)
- Likelihood evaluation (via the instantaneous change-of-variable formula)
- Scenarios where diffusion models work but you need faster sampling
- Data on manifolds (with appropriate conditional path design)

**When NOT to use:**
- When the coupling between prior and data is highly complex and independent coupling is insufficient
- When only samples are needed and likelihood is not important (GANs may be cheaper)
- When discrete data (text tokens) requires discrete flow matching extensions

**Comparison:** Training is simulation-free (like score matching) but produces straighter trajectories, enabling 10x fewer sampling steps than DDPM while maintaining comparable quality.

## Algorithm / Method Details

### Training Procedure

1. **Sample**: Draw $t \sim \mathcal{U}[0, 1]$, $x_0 \sim p_0$ (prior, e.g., $\mathcal{N}(0, I)$), $x_1$ from data.
2. **Interpolate**: Compute $x_t = t x_1 + (1 - t) x_0$.
3. **Predict velocity**: Compute $v_t(x_t; \theta)$.
4. **Loss**: $\mathcal{L} = \| v_t(x_t; \theta) - (x_1 - x_0) \|^2$.

### Sampling (ODE Solve)

1. Start with $x_0 \sim p_0$ at $t = 0$.
2. Solve ODE $dx/dt = v_t(x; \theta)$ from $t=0$ to $t=1$ using any ODE solver (e.g., Euler, RK4, Dormand-Prince).
3. $x_1$ is the generated sample.

### Complexity Analysis

- **Training**: O($T \cdot D$) per sample where $T$ is the number of timesteps evaluated (typically 1, since CFM samples a single random $t$ per minibatch).
- **Sampling**: O($L \cdot D$) with $L$ ODE steps; straight paths allow $L \approx 1{\!-\!}10$ vs 50-1000 for DDPM.

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| ODE solver steps | 1-10 (Euler or RK4) | More steps improve quality but slow sampling |
| Noise schedule | Linear $\sigma_t = \sigma_{\min} + t(\sigma_{\max} - \sigma_{\min})$ | Sets path curvature; smaller $\sigma$ gives straighter paths |
| Prior distribution | $\mathcal{N}(0, I)$ | Standard normal works for most modalities |
| Time conditioning | Sinusoidal positional encoding | Common for diffusion/flow matching models |
| Model architecture | U-Net or DiT | Same as diffusion models; DiT preferred for image generation |

### Numerical Considerations

- Clamp $t$ to avoid $t=0$ instability (e.g., sample $t \sim \mathcal{U}[\epsilon, 1- \epsilon]$ with $\epsilon = 10^{-5}$).
- Use ODE solvers with adaptive step size (dopri5) for accurate likelihood evaluation.
- For the deterministic OT path, ensure the network output is scaled appropriately (the target velocity has the same scale as the data).

## Python Implementation

```python
"""
Minimal implementation of Flow Matching for generative modeling.
Based on: Lipman et al. (2023) https://arxiv.org/abs/2210.02747
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional


# ============================================================
# Conditional Flow Matching Components
# ============================================================

class TimeMLP(nn.Module):
    """Simple MLP with time embedding for flow matching vector field."""
    def __init__(
        self,
        dim: int = 128,
        hidden_dims: list = None,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]
        if out_dim is None:
            out_dim = dim

        # Sinusoidal time embedding
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        layers = []
        prev_dim = dim + 128  # input + time embedding
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.SiLU(),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        h = torch.cat([x, t_embed], dim=-1)
        return self.net(h)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for time encoding."""
    def __init__(self, embed_dim: int = 256, scale: float = 30.0):
        super().__init__()
        # Randomly sample frequencies
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_proj = t.unsqueeze(-1) * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class FlowMatchingModel(nn.Module):
    """
    Vector field model v_t(x; theta) for flow matching.

    Can use any architecture (U-Net, ViT, MLP). Here we use a simple MLP
    for illustration on low-dimensional data.
    """
    def __init__(self, data_dim: int = 2, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        self.net = TimeMLP(
            dim=data_dim,
            hidden_dims=hidden_dims,
            out_dim=data_dim,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the velocity vector field at (x, t)."""
        return self.net(x, t)


class FlowMatching:
    """
    Flow matching training and sampling utilities.

    Implements the optimal transport (OT) conditional path:
        mu_t = t*x1 + (1-t)*x0
        sigma_t = sigma_min + t*(sigma_max - sigma_min)
    """
    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 0.001,
        sigma_max: float = 0.001,  # small sigma -> nearly deterministic OT path
    ):
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sample_ot_path(
        self, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the OT probability path.

        Args:
            x1: Data samples (batch, dim)
            x0: Prior samples (batch, dim)
            t: Time (batch,) in [0, 1]

        Returns:
            x_t: Interpolated point
            target: Target velocity (x1 - x0)
        """
        t = t.unsqueeze(-1)  # (batch, 1) for broadcasting
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.sigma_min + t * (self.sigma_max - self.sigma_min)
        noise = torch.randn_like(x1)
        x_t = mu_t + sigma_t * noise
        target = x1 - x0  # constant velocity for OT path
        return x_t, target

    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional flow matching loss.

        Args:
            x1: Data batch (batch, dim)

        Returns:
            loss: MSE loss
        """
        batch = x1.shape[0]
        # Sample time uniformly
        t = torch.rand(batch, device=x1.device)
        # Sample prior
        x0 = torch.randn_like(x1)

        # Sample from the OT path
        x_t, target = self.sample_ot_path(x1, x0, t)

        # Predict velocity
        v_t = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_t, target)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        data_dim: int,
        num_steps: int = 10,
        method: str = "euler",
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate samples by solving the ODE.

        Args:
            num_samples: Number of samples
            data_dim: Data dimensionality
            num_steps: Number of ODE solver steps
            method: 'euler' or 'rk4'
            device: Device for computation

        Returns:
            samples: Generated samples (num_samples, data_dim)
        """
        # Start from prior
        x = torch.randn(num_samples, data_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)

            if method == "euler":
                v = self.model(x, t)
                x = x + v * dt
            elif method == "rk4":
                # Runge-Kutta 4th order
                k1 = self.model(x, t)
                k2 = self.model(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = self.model(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = self.model(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown ODE method: {method}")

        return x


# ============================================================
# Complete usage example with synthetic data
# ============================================================
def train_flow_matching_2d():
    """
    Demonstrate flow matching on a 2D dataset (mixture of Gaussians).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create synthetic 2D dataset: mixture of 8 Gaussians in a ring
    def create_mixture_dataset(n_samples: int = 10000) -> torch.Tensor:
        n_components = 8
        angles = torch.linspace(0, 2 * torch.pi, n_components + 1)[:n_components]
        centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 3
        component_idx = torch.randint(0, n_components, (n_samples,))
        noise = torch.randn(n_samples, 2) * 0.3
        x1 = centers[component_idx] + noise
        return x1

    # Data
    data = create_mixture_dataset(20000).to(device)
    print(f"Dataset shape: {data.shape}")

    # Model
    model = FlowMatchingModel(data_dim=2).to(device)
    fm = FlowMatching(model, sigma_min=1e-4, sigma_max=1e-4)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training
    n_epochs = 200
    batch_size = 512
    n_batches = len(data) // batch_size

    print("Training flow matching model...")
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        perm = torch.randperm(len(data))
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch = data[idx]

            optimizer.zero_grad()
            loss = fm.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    # Generate samples
    print("\nGenerating samples...")
    samples = fm.sample(num_samples=2000, data_dim=2, num_steps=10, device=device)
    samples = samples.cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Real data
    axes[0].scatter(data.cpu()[:2000, 0], data.cpu()[:2000, 1],
                    alpha=0.5, s=5, c="blue")
    axes[0].set_title("Real Data (Mixture of 8 Gaussians)")
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].set_aspect("equal")

    # Generated data
    axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5, c="red")
    axes[1].set_title(f"Flow Matching Samples (10 ODE steps)")
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    save_path = "E:/wuyi/数学建模半自动/research-assistant/knowledge/temp/ml/flow_matching_demo.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Figure saved to: {save_path}")

    # Also sample with fewer steps for comparison
    for n_steps in [1, 5, 20]:
        samples_few = fm.sample(
            num_samples=2000, data_dim=2, num_steps=n_steps, device=device
        )
        l2_dist = torch.mean(
            (torch.tensor(samples_few) - torch.tensor(samples)) ** 2
        ).item()
        print(f"  {n_steps} steps: L2 diff from 10-step reference = {l2_dist:.4f}")

    return losses


def test_conditional_flow_matching_loss():
    """Unit test for CFM loss computation."""
    torch.manual_seed(42)
    model = FlowMatchingModel(data_dim=10)
    fm = FlowMatching(model)

    x1 = torch.randn(32, 10)
    loss = fm.compute_loss(x1)
    print(f"CFM loss shape: scalar, value = {loss.item():.6f}")

    # Backward should work
    loss.backward()
    assert model.net[0].weight.grad is not None, "Gradients not flowing!"
    print("Backward pass successful.")

    return loss


if __name__ == "__main__":
    test_conditional_flow_matching_loss()
    train_flow_matching_2d()
```

## References

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2210.02747

Liu, X., Gong, C., & Liu, Q. (2023). Flow straight and fast: Learning to generate and transfer data with rectified flow. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2209.03003

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems (NeurIPS), 31*. https://arxiv.org/abs/1806.07366

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2011.13456

Albergo, M. S., Boffi, N. M., & Vanden-Eijnden, E. (2023). Stochastic interpolants: A unifying framework for flows and diffusions. *arXiv preprint arXiv:2303.08797*. https://arxiv.org/abs/2303.08797
