# False Discovery Rate Control for Structured Multiple Testing: Asymmetric Rules and Conformal Q-values

**Source**: Zhao, Z., & Sun, W. (2025). False discovery rate control for structured multiple testing: Asymmetric rules and conformal q-values. *Journal of the American Statistical Association*, 120(550), 805--817. https://doi.org/10.1080/01621459.2024.2359739

**Category**: Statistics / Multiple Testing / Conformal Inference

## Mathematical Setup

Multiple testing with **structured hypotheses** (e.g., spatial, temporal, or network-structured) requires FDR-controlling procedures that can incorporate structural information. Traditional methods like the Benjamini-Hochberg (BH) procedure are symmetric (permutation-invariant) and cannot exploit structure.

Zhao & Sun (2025) introduce the **Pseudo Local Index of Significance (PLIS)** procedure that:
1. Accommodates **asymmetric decision rules** (score functions that vary across hypotheses)
2. Requires only **pairwise exchangeability** (weaker than joint exchangeability)
3. Provides **conformal q-values** for interpretable FDR control

Consider $m$ hypothesis tests with observations $Z_i$ for $i = 1, \ldots, m$. Under the null $H_{0i}$, $Z_i \sim P_{0i}$. Define a conformal score function $s_i(\cdot)$ for each hypothesis. The key insight is to construct a **mirror process**:

$$
M_i(t) = \mathbb{1}\{s_i(Z_i) > t\} - \mathbb{1}\{s_i(Z_i') > t\}
$$

where $Z_i'$ is a knockoff-like copy satisfying pairwise exchangeability with $Z_i$ under the null. The PLIS statistic is:

$$
\text{PLIS}_i = \sup\{t \geq 0 : M_i(t) > 0\}
$$

or equivalently, the conformal q-value:

$$
\hat{q}_i = \min_{t \geq 0} \frac{\sum_{j=1}^m \mathbb{1}\{s_j(Z_j') \geq t\}}{\max\left(1, \sum_{j=1}^m \mathbb{1}\{s_j(Z_j) \geq t\}\right)}
$$

The PLIS procedure rejects hypotheses with $\text{PLIS}_i \geq \tau$ where $\tau$ is chosen to control FDR:

$$
\tau = \max\left\{t : \frac{\sum_{i=1}^m \mathbb{1}\{s_i(Z_i') \geq t\}}{\max(1, \sum_{i=1}^m \mathbb{1}\{s_i(Z_i) \geq t\})} \leq q\right\}
$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Pairwise exchangeability | Under $H_{0i}$, $(Z_i, Z_i') \overset{d}{=} (Z_i', Z_i)$ | The mirror process is mean-zero under the null |
| Independence of null pairs | $(Z_i, Z_i') \perp (Z_j, Z_j')$ for $i \neq j$ under nulls | Standard multiple testing independence |
| Score asymmetry allowed | $s_i(\cdot)$ can depend on $i$ arbitrarily | Structural information can be encoded in scores |
| Consistency of non-null scores | $s_i(Z_i) > s_i(Z_i')$ in probability for non-null $i$ | Signals are detectable |

The conformal q-values satisfy the finite-sample FDR control property:

$$
\text{FDR} = \mathbb{E}\left[\frac{|\mathcal{H}_0 \cap \mathcal{R}|}{|\mathcal{R}| \vee 1}\right] \leq q
$$

under pairwise exchangeability, where $\mathcal{R}$ is the rejection set and $\mathcal{H}_0$ is the set of true null hypotheses.

## Applicable Scenarios

- **When to use**: Large-scale multiple testing with spatial/temporal structure (neuroimaging, genomics, astronomical surveys), when hypotheses have known grouping or ordering, when domain-specific score functions can improve power, when structural dependencies should be exploited rather than treated as nuisance.
- **When NOT to use**: When hypotheses are exchangeable (no structural information), BH or standard conformal methods suffice; when pairwise exchangeability is violated; when $m$ is very small ($m < 20$).
- **Comparison with classical alternatives**: BH procedure is symmetric and cannot incorporate structure. Storey's q-value requires estimating the null proportion. AdaPT and related methods require iterative fitting. PLIS provides finite-sample FDR control with asymmetric rules in a single pass.

## Method Details

1. **Score construction**: For each hypothesis $i$, construct a score function $s_i(\cdot)$ that captures both the evidence against $H_{0i}$ and any structural information (e.g., spatial smoothness penalty, group membership).
2. **Mirror copy generation**: For each $i$, generate $Z_i'$ that is pairwise exchangeable with $Z_i$ under the null. This can be a theoretical null distribution, a permutation copy, or a model-X knockoff.
3. **Mirror process computation**: Compute $M_i(t)$ or the conformal q-value for each hypothesis.
4. **Threshold selection**: Find the largest threshold $\tau$ such that the estimated FDP $\leq q$.
5. **Rejection**: Reject all hypotheses with PLIS $\geq \tau$ (or conformal q-value $\leq q$).

**Theoretical guarantees**:
- **Finite-sample FDR control**: For any $m$ and any dependence structure among non-null hypotheses, $\text{FDR} \leq q$
- **Asymptotic consistency**: Under mild conditions, $\text{FDP}(\tau) \to \text{FDR}(\tau)$ as $m \to \infty$
- **Adaptivity**: The procedure automatically adapts to signal strength and sparsity
- **Power improvement**: Asymmetric rules incorporating domain structure can significantly improve power over symmetric rules

## Implementation Details

- **Key parameters**: Target FDR level $q$, score function family $\{s_i\}_{i=1}^m$, mirror copy generation method
- **Computational considerations**: $O(m)$ for score evaluation, $O(m \log m)$ for threshold search; highly scalable
- **Software availability**: Code provided by the authors at https://github.com/ZhaoZinan/PLIS

## Python Implementation

```python
import numpy as np
from typing import Callable, Optional, Tuple
from scipy.stats import norm


class PLISProcedure:
    """Pseudo Local Index of Significance (PLIS) for structured multiple testing.
    
    Implements the conformal q-value procedure of Zhao & Sun (2025, JASA)
    for FDR control with asymmetric decision rules that can incorporate
    structural information.
    
    Parameters
    ----------
    q : float, default=0.1
        Target FDR level.
    score_function : Callable, default=None
        Function s_i(z) -> conformal score for hypothesis i.
        If None, uses the identity (z itself as score).
        Should accept (z, i) where z is the test statistic and i is
        the hypothesis index.
    mirror_method : str, default='permutation'
        Method for generating mirror copies: 'permutation', 'gaussian',
        or 'theoretical'.
    structural_weights : ndarray, default=None
        Prior structural weights for each hypothesis (higher = more
        likely to be a discovery). Used to weight scores.
    random_state : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        q: float = 0.1,
        score_function: Optional[Callable] = None,
        mirror_method: str = 'gaussian',
        structural_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ):
        self.q = q
        self.score_function = score_function or (lambda z, i: z)
        self.mirror_method = mirror_method
        self.structural_weights = structural_weights
        self.random_state = random_state
        self._fitted = False
    
    def _generate_mirror_copies(
        self, z: np.ndarray
    ) -> np.ndarray:
        """Generate mirror copies Z_i' satisfying pairwise exchangeability.
        
        Under the null, Z_i' should be exchangeable with Z_i.
        For theoretical null N(0,1), Z_i' is an independent draw from N(0,1).
        """
        n = len(z)
        
        if self.mirror_method == 'gaussian':
            # For theoretical null N(0, sigma^2)
            sigma_hat = np.std(z)  # conservative: use all data
            z_prime = sigma_hat * np.random.randn(n)
            
        elif self.mirror_method == 'permutation':
            # Permutation-based: shuffle labels
            idx = np.random.permutation(n)
            z_prime = z[idx]
            
        else:  # 'theoretical'
            # Known null distribution (e.g., standard normal for z-scores)
            z_prime = np.random.randn(n)
        
        return z_prime
    
    def _compute_scores(
        self, z: np.ndarray, z_prime: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute conformal scores for observed and mirror copies.
        
        Applies structural weighting if provided.
        """
        n = len(z)
        scores = np.array([self.score_function(z[i], i) for i in range(n)])
        scores_prime = np.array([
            self.score_function(z_prime[i], i) for i in range(n)
        ])
        
        # Apply structural weights (higher weight = higher score = more likely to reject)
        if self.structural_weights is not None:
            scores = scores * self.structural_weights
            scores_prime = scores_prime * self.structural_weights
        
        return scores, scores_prime
    
    def _conformal_qvalues(
        self, scores: np.ndarray, scores_prime: np.ndarray
    ) -> np.ndarray:
        """Compute conformal q-values.
        
        The conformal q-value for hypothesis i is the minimum FDR level
        at which it would be rejected.
        """
        n = len(scores)
        q_vals = np.ones(n)
        
        # Sort scores
        sorted_idx = np.argsort(-scores)
        
        for i in range(n):
            t = scores[sorted_idx[i]]
            
            # Count exceedances
            R_t = max(1, np.sum(scores >= t))
            V_t = np.sum(scores_prime >= t)
            
            # Estimated FDP at threshold t
            fdp_hat = V_t / R_t
            
            # q-value is the minimum FDP over thresholds where this point is rejected
            q_vals[sorted_idx[i]] = fdp_hat
        
        # Ensure monotonicity (like in Storey's q-value)
        q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
        
        return q_vals
    
    def fit(
        self, z: np.ndarray, **kwargs
    ) -> 'PLISProcedure':
        """Fit the PLIS procedure.
        
        Parameters
        ----------
        z : ndarray of shape (m,)
            Test statistics for m hypotheses.
        **kwargs : dict
            Additional keyword arguments for score functions or
            mirror generation.
            
        Returns
        -------
        self : PLISProcedure
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        m = len(z)
        
        # Generate mirror copies
        z_prime = self._generate_mirror_copies(z)
        
        # Compute scores
        scores, scores_prime = self._compute_scores(z, z_prime)
        
        # Compute conformal q-values
        self.q_values_ = self._conformal_qvalues(scores, scores_prime)
        
        # Determine rejection set
        self.rejected_ = np.where(self.q_values_ <= self.q)[0]
        self.n_rejected_ = len(self.rejected_)
        
        # Estimate FDP
        threshold = scores[self.rejected_].min() if len(self.rejected_) > 0 else np.inf
        self.threshold_ = threshold
        R = max(1, self.n_rejected_)
        V_hat = np.sum(scores_prime >= threshold)
        self.fdp_estimate_ = V_hat / R
        
        self.scores_ = scores
        self.scores_prime_ = scores_prime
        self._fitted = True
        
        return self
    
    def summary(self) -> None:
        """Print summary of PLIS results."""
        if not self._fitted:
            raise RuntimeError("Must call .fit() before .summary()")
        
        print("=== PLIS: Conformal Q-values for FDR Control ===\n")
        print(f"Target FDR: q = {self.q}")
        print(f"Number of hypotheses: m = {len(self.q_values_)}")
        print(f"Rejected: {self.n_rejected_}")
        print(f"Estimated FDP: {self.fdp_estimate_:.4f}")
        
        if self.n_rejected_ > 0:
            print(f"\nTop rejected hypotheses:")
            top_k = min(10, self.n_rejected_)
            top_idx = self.rejected_[np.argsort(
                -self.scores_[self.rejected_]
            )[:top_k]]
            for idx in top_idx:
                print(f"  H{idx}: score = {self.scores_[idx]:.4f}, "
                      f"q-value = {self.q_values_[idx]:.4f}")
    
    def get_qvalues(self) -> np.ndarray:
        """Return conformal q-values for all hypotheses."""
        if not self._fitted:
            raise RuntimeError("Must call .fit() first.")
        return self.q_values_
    
    def get_rejected(self) -> np.ndarray:
        """Return indices of rejected hypotheses."""
        if not self._fitted:
            raise RuntimeError("Must call .fit() first.")
        return self.rejected_


# ============================================================
# Example: Structured multiple testing with spatial signals
# ============================================================

def spatial_score(z: float, i: int, locations: np.ndarray,
                  neighbors: np.ndarray, z_all: np.ndarray) -> float:
    """Spatially-aware score function.
    
    Boosts scores for hypotheses whose neighbors also show signal.
    This is an asymmetric rule: different hypotheses get different
    score transformations based on their spatial context.
    
    Parameters
    ----------
    z : float
        Test statistic for hypothesis i.
    i : int
        Hypothesis index.
    locations : ndarray of shape (m, d)
        Spatial locations of each hypothesis.
    neighbors : ndarray of shape (m, k)
        Indices of k nearest neighbors for each hypothesis.
    z_all : ndarray of shape (m,)
        All test statistics.
    
    Returns
    -------
    score : float
        Spatially-weighted conformal score.
    """
    # Base score: absolute z-statistic
    base_score = abs(z)
    
    # Spatial context: average statistic among neighbors
    neighbor_scores = np.mean([abs(z_all[j]) for j in neighbors[i]])
    
    # Combine: higher weight when both own signal and neighbors have signal
    score = base_score + 0.3 * neighbor_scores
    
    return score


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=== Structured Multiple Testing: PLIS with Spatial Scores ===\n")
    
    # Simulate 2D spatial hypothesis testing
    m = 400  # 20x20 grid
    grid_size = int(np.sqrt(m))
    
    # Create spatial locations
    x_coords, y_coords = np.meshgrid(
        np.arange(grid_size), np.arange(grid_size)
    )
    locations = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # Generate sparse spatial signal
    n_signals = 12  # number of true signals
    signal_centers = np.random.choice(m, n_signals, replace=False)
    signal_strength = np.random.uniform(2.5, 4.0, n_signals)
    
    # Generate z-statistics
    z = np.random.randn(m)
    for idx, strength in zip(signal_centers, signal_strength):
        z[idx] += strength
    
    # Ground truth
    true_null = np.ones(m, dtype=bool)
    true_null[signal_centers] = False
    pi0 = true_null.mean()
    
    print(f"Spatial grid: {grid_size}x{grid_size}")
    print(f"Total hypotheses: m = {m}")
    print(f"True signals: {n_signals}")
    print(f"Null proportion pi0 = {pi0:.2f}")
    
    # ---- BH procedure (symmetric) ----
    from scipy.stats import norm as norm_dist
    
    p_values = 2 * (1 - norm_dist.cdf(np.abs(z)))
    sorted_p = np.sort(p_values)
    bh_threshold = np.max(np.where(
        sorted_p <= np.arange(1, m + 1) / m * 0.1
    )[0])
    bh_rejected = np.where(
        p_values <= (bh_threshold + 1) / m * 0.1
    )[0] if bh_threshold >= 0 else np.array([])
    
    bh_fdp = np.sum(true_null[bh_rejected]) / max(1, len(bh_rejected))
    bh_power = np.sum(~true_null[bh_rejected]) / n_signals
    
    print(f"\n--- BH Procedure (symmetric, q = 0.1) ---")
    print(f"Rejected: {len(bh_rejected)}")
    print(f"FDP: {bh_fdp:.3f}")
    print(f"Power: {bh_power:.3f}")
    
    # ---- PLIS with spatial score (asymmetric) ----
    # Build nearest neighbor graph
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=9, metric='euclidean')
    nn.fit(locations)
    _, neighbor_idx = nn.kneighbors(locations)
    
    # Custom score function using spatial context
    def make_spatial_score(loc, nbr_idx, z_all):
        def score_fn(z_i, i):
            return spatial_score(z_i, i, loc, nbr_idx, z_all)
        return score_fn
    
    spatial_score_fn = make_spatial_score(locations, neighbor_idx, z)
    
    plis = PLISProcedure(
        q=0.1,
        score_function=spatial_score_fn,
        mirror_method='gaussian',
        random_state=42,
    )
    plis.fit(z)
    plis.summary()
    
    plis_rejected = plis.get_rejected()
    plis_fdp = np.sum(true_null[plis_rejected]) / max(1, len(plis_rejected))
    plis_power = np.sum(~true_null[plis_rejected]) / n_signals
    
    print(f"\n  FDP: {plis_fdp:.3f}")
    print(f"  Power: {plis_power:.3f}")
    
    # ---- PLIS with uniform scores (symmetric) ----
    plis_uniform = PLISProcedure(
        q=0.1,
        score_function=lambda z_i, i: abs(z_i),  # symmetric
        mirror_method='gaussian',
        random_state=42,
    )
    plis_uniform.fit(z)
    uniform_rejected = plis_uniform.get_rejected()
    uniform_fdp = np.sum(true_null[uniform_rejected]) / max(1, len(uniform_rejected))
    uniform_power = np.sum(~true_null[uniform_rejected]) / n_signals
    
    # ---- Comparison summary ----
    print(f"\n--- Comparison ---")
    print(f"{'Method':<20} {'Rejected':<10} {'FDP':<10} {'Power':<10}")
    print("-" * 50)
    print(f"{'BH (symmetric)':<20} {len(bh_rejected):<10} {bh_fdp:<10.3f} {bh_power:<10.3f}")
    print(f"{'PLIS (symmetric)':<20} {len(uniform_rejected):<10} {uniform_fdp:<10.3f} {uniform_power:<10.3f}")
    print(f"{'PLIS (spatial)':<20} {len(plis_rejected):<10} {plis_fdp:<10.3f} {plis_power:<10.3f}")
    
    # ---- Visualize as heatmap ----
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        titles = ['True Signals', 'BH Rejections', 'PLIS (Spatial) Rejections']
        data = [
            ~true_null,
            np.isin(np.arange(m), bh_rejected),
            np.isin(np.arange(m), plis_rejected),
        ]
        
        for ax, title, dat in zip(axes, titles, data):
            im = ax.imshow(
                dat.reshape(grid_size, grid_size),
                cmap='Reds', aspect='equal'
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('plis_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("\n(matplotlib not available for visualization)")
```

## References

Zhao, Z., & Sun, W. (2025). False discovery rate control for structured multiple testing: Asymmetric rules and conformal q-values. *Journal of the American Statistical Association*, 120(550), 805--817. https://doi.org/10.1080/01621459.2024.2359739

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 57(1), 289--300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Storey, J. D. (2002). A direct approach to false discovery rates. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 64(3), 479--498. https://doi.org/10.1111/1467-9868.00346

Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 80(3), 551--577. https://doi.org/10.1111/rssb.12265

Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *The Annals of Statistics*, 43(5), 2055--2085. https://doi.org/10.1214/15-AOS1337
