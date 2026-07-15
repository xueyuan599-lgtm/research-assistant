# Causal Inference under Network Interference

**Source**: Bargagli-Stoffi, F. J., Tortu, C., & Forastiere, L. (2025). Heterogeneous treatment and spillover effects under clustered network interference. *The Annals of Applied Statistics*, 19(1), 28–55. https://doi.org/10.1214/24-AOAS1913

**Source**: Leung, M. (2022). Causal inference under approximate neighborhood interference. *Econometrica*, 90(1), 267–301. https://doi.org/10.3982/ECTA17946

**Source**: Forastiere, L., Airoldi, E. M., & Mealli, F. (2021). Identification and estimation of treatment and interference effects in observational studies on networks. *Journal of the American Statistical Association*, 116(534), 901–918. https://doi.org/10.1080/01621459.2020.1768100

**Category**: Causal Inference / Network Interference / Spillover Effects

## Mathematical Setup

Standard causal inference assumes the **Stable Unit Treatment Value Assumption (SUTVA)**, which requires that one unit's treatment does not affect another unit's outcome. In network settings, this assumption is violated because treatment can **spill over** through social ties, peer effects, or market interactions.

### Potential Outcomes under Interference

Consider a population of $n$ units connected by a network $G = (V, E)$. Each unit $i$ has:
- Treatment $D_i \in \{0, 1\}$
- Outcome $Y_i$
- Covariates $X_i$

Under network interference, the potential outcome for unit $i$ depends on the entire treatment vector $\mathbf{D} = (D_1, \dots, D_n)$:

$$Y_i(\mathbf{D}) \neq Y_i(D_i) \quad \text{in general}$$

This is a high-dimensional object (2^n potential outcomes per unit), so structural assumptions are needed.

### Exposure Mapping and Neighborhood Interference

The key simplification is the **exposure mapping** assumption: a unit's outcome depends only on its own treatment and the treatments of units within a certain neighborhood. For a neighborhood defined by network distance $r$:

$$\mathcal{N}_i(r) = \{j: \text{dist}(i, j) \leq r\}$$

The **neighborhood treatment vector** is $\mathbf{D}_{\mathcal{N}_i}$.

### Target Estimands

The literature defines several causal estimands of interest:

1. **Direct (treatment) effect** for unit $i$:

   $$\delta(d) = \mathbb{E}[Y_i(D_i = d, \mathbf{D}_{\mathcal{N}_i} = \mathbf{0})]$$

2. **Spillover (peer) effect**:

   $$\zeta(s) = \mathbb{E}[Y_i(D_i = 0, \text{proportion treated in }\mathcal{N}_i = s)]$$

3. **Total effect**:

   $$\tau(d, s) = \mathbb{E}[Y_i(D_i = d, \text{proportion treated in }\mathcal{N}_i = s)]$$

### Network Causal Trees (NCT) — Bargagli-Stoffi et al., 2025

NCT adapts the causal tree algorithm of Athey and Imbens (2016) to handle interference within clusters. The algorithm:

1. **Defines clusters**: Partitions network units into clusters (by community detection or design)
2. **Estimates cluster-level treatment assignment probabilities**: Accounts for interference patterns
3. **Splits on covariates**: Forms leaves based on covariates, maximizing heterogeneity in both direct and spillover effects
4. **Uses Horvitz-Thompson estimation** within each leaf: Weights by inverse probability of treatment vector

The heterogeneity criterion is:

$$\min_{\text{split}} \left( \text{MSE}_{\text{direct}} + \text{MSE}_{\text{spillover}} + \lambda \cdot \text{complexity} \right)$$

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Exposure mapping | $Y_i(\mathbf{D}) = Y_i(\mathbf{D}_{\mathcal{N}_i})$ | Only neighbors matter |
| Partial interference (clusters) | No interference across clusters | Independent clusters |
| Unconfounded network | $\mathbf{D}_{\mathcal{N}_i} \perp Y_i(D_i, \mathbf{D}_{\mathcal{N}_i}) \mid X_i, \text{network}$ | No network confounders |
| No spillover from | $\mathbb{E}[Y_i(D_i, \mathbf{D}_{\mathcal{N}_i})]$ is continuous in | Continuity assumption for |
| beyond neighborhood | $\mathbf{D}_{\mathcal{N}_i}$ for units beyond radius $r$ | bounding interference |

## Applicable Scenarios

**When to use:**
- Social network experiments (friends, followers)
- Peer effects in education, health, or development
- Market interventions with competition effects
- Platform experiments (recommendation systems, advertising)
- Vaccination programs (herd immunity)

**When NOT to use:**
- When complete network is unobserved and exposure mapping is unknown
- When interference is global and cannot be localized
- When cluster sizes are very small or very imbalanced
- When the network structure itself is endogenous to treatment

**Comparison:**
- The Local Approach (Leung, 2022) uses k-NN on network configurations and does not require clustering
- GNN-based methods (HINet) learn the exposure mapping from data but are less interpretable
- NCT is most natural when clusters are given (e.g., villages, classrooms)

## Method Details

### Step-by-Step Procedure (NCT-style)

1. **Define neighborhoods/clusters**: For each unit, identify the set of potential interferers (e.g., one-hop neighbors, or cluster members in a cluster-randomized design).

2. **Define exposure mapping**: Choose how to summarize neighbors' treatments. Common choices:
   - Proportion of treated neighbors (continuous)
   - Any treated neighbor vs. none (binary)
   - Number of treated neighbors (count)

3. **Estimate propensity**: For each unit, estimate the joint probability of its own treatment and its neighbors' treatments:

   $$\hat{\pi}_i(d, \mathbf{d}_{\mathcal{N}_i}) = \hat{\mathbb{P}}(D_i = d, \mathbf{D}_{\mathcal{N}_i} = \mathbf{d}_{\mathcal{N}_i} \mid X_i, \text{network})$$

4. **Build causal tree**: Recursively partition the covariate space. At each split, for each leaf:
   - Estimate the direct effect and spillover effect using Horvitz-Thompson or Hajek estimators
   - Compute the MSE for both estimands
   - Choose the split that maximizes heterogeneity

5. **Inference**: Use cluster-robust standard errors (if clusters are independent) or bootstrap procedures.

### Asymptotic Properties
- **Consistency**: Under appropriate neighborhood sparsity and overlap conditions, NCT estimates converge to the true leaf-specific effects
- **Rate**: $O_p(n^{-1/2} \log n)$ under bounded degree and correctly specified exposure mapping
- **Normality**: Leaf-specific estimates are asymptotically normal with enough units per leaf

## Implementation Details

**Key hyperparameters:**
- Neighborhood radius $r$ (or cluster definition)
- Exposure mapping specification (proportion vs. indicator)
- Minimum leaf size (number of clusters/units per leaf)
- Number of trees (for causal forest vs. single tree)

**Available software:**
- R: `interference` package, `grf` (with cluster-robust options)
- Python: `networkx` for network construction, `causalml` for uplift modeling

## Python Implementation

```python
"""
Causal Inference under Network Interference

Network Causal Tree (NCT) for estimating heterogeneous direct and
spillover effects under clustered network interference.

References:
    Bargagli-Stoffi et al. (2025). Heterogeneous treatment and
        spillover effects under clustered network interference.
        Annals of Applied Statistics, 19(1), 28-55.
    Leung (2022). Causal inference under approximate neighborhood
        interference. Econometrica, 90(1), 267-301.
"""

import numpy as np
from collections import defaultdict


class NetworkCausalTree:
    """Network Causal Tree for heterogeneous treatment and spillover effects.

    Estimates direct and spillover CATE under clustered network interference
    using a tree-based partitioning of the covariate space.

    Parameters
    ----------
    min_leaf_size : int, default=20
        Minimum number of units in each leaf.
    max_depth : int, default=4
        Maximum tree depth.
    random_state : int, default=42
    """
    def __init__(self, min_leaf_size=20, max_depth=4, random_state=42):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.random_state = random_state

    def _estimate_exposure(self, adj_matrix, D):
        """Compute exposure mapping: proportion of treated neighbors.

        Parameters
        ----------
        adj_matrix : ndarray, shape (n, n)
            Binary adjacency matrix.
        D : ndarray, shape (n,)
            Treatment vector.

        Returns
        -------
        exposure : ndarray, shape (n,)
            Proportion of treated neighbors (NaN if no neighbors).
        deg : ndarray, shape (n,)
            Degree (number of neighbors).
        """
        n = len(D)
        deg = adj_matrix.sum(axis=1)
        n_treated_neighbors = adj_matrix @ D
        exposure = np.where(deg > 0, n_treated_neighbors / deg, 0.0)
        return exposure, deg

    def _ht_estimator(self, Y, D, exposure, leaf_idx):
        """Horvitz-Thompson-like estimator for direct and spillover effects
        within a leaf.

        Direct effect: E[Y | D=1, low exposure] - E[Y | D=0, low exposure]
        Spillover effect: E[Y | D=0, high exposure] - E[Y | D=0, low exposure]
        """
        leaf_units = np.where(leaf_idx)[0]
        if len(leaf_units) < self.min_leaf_size:
            return None, None

        Y_leaf = Y[leaf_units]
        D_leaf = D[leaf_units]
        exp_leaf = exposure[leaf_units]

        # Define "low" and "high" exposure
        exp_median = np.median(exp_leaf)
        low_exp = exp_leaf <= exp_median
        high_exp = exp_leaf > exp_median

        # Direct effect: treated vs control at low exposure
        treated_low = (D_leaf == 1) & low_exp
        control_low = (D_leaf == 0) & low_exp

        if np.sum(treated_low) < 5 or np.sum(control_low) < 5:
            direct_effect = None
        else:
            direct_effect = Y_leaf[treated_low].mean() - Y_leaf[control_low].mean()

        # Spillover effect: high vs low exposure among controls
        control_high = (D_leaf == 0) & high_exp

        if np.sum(control_high) < 5 or np.sum(control_low) < 5:
            spillover_effect = None
        else:
            spillover_effect = Y_leaf[control_high].mean() - Y_leaf[control_low].mean()

        return direct_effect, spillover_effect

    def fit(self, X, adj_matrix, D, Y):
        """Fit a network causal tree.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Covariates.
        adj_matrix : array-like, shape (n, n)
            Binary symmetric adjacency matrix.
        D : array-like, shape (n,)
            Binary treatment.
        Y : array-like, shape (n,)
            Outcome.

        Returns
        -------
        self : NetworkCausalTree
        """
        X = np.asarray(X)
        adj_matrix = np.asarray(adj_matrix)
        D = np.asarray(D).ravel()
        Y = np.asarray(Y).ravel()
        n = len(Y)

        self.exposure_, self.deg_ = self._estimate_exposure(adj_matrix, D)

        # Build the tree recursively
        self.tree_ = self._build_tree(X, Y, D, self.exposure_,
                                       np.ones(n, dtype=bool), depth=0)

        # Predict for training data
        self.leaf_predictions_ = self._predict_leaves(X)

        return self

    def _build_tree(self, X, Y, D, exposure, idx, depth):
        """Recursively build the causal tree."""
        n_current = np.sum(idx)

        if depth >= self.max_depth or n_current <= 2 * self.min_leaf_size:
            direct, spillover = self._ht_estimator(Y, D, exposure, idx)
            return {
                "is_leaf": True,
                "idx": idx,
                "direct_effect": direct,
                "spillover_effect": spillover,
                "n": n_current,
                "depth": depth,
            }

        # Find the best split
        best_split = None
        best_score = -np.inf

        n_features = X.shape[1]
        n_trials = int(np.ceil(np.sqrt(n_features)))
        feature_candidates = np.random.choice(
            n_features, size=n_trials, replace=False)

        for col in feature_candidates:
            x_col = X[idx, col]
            thresholds = np.percentile(x_col, np.linspace(15, 85, 10))

            for thresh in thresholds:
                left_idx = idx.copy()
                left_mask = X[:, col] <= thresh
                left_idx = idx & left_mask

                n_left = np.sum(left_idx)
                n_right = n_current - n_left

                if n_left < self.min_leaf_size or n_right < self.min_leaf_size:
                    continue

                # Evaluate direct + spillover heterogeneity
                direct_l, spill_l = self._ht_estimator(
                    Y, D, exposure, left_idx) or (0, 0)
                direct_r, spill_r = self._ht_estimator(
                    Y, D, exposure, idx & ~left_mask) or (0, 0)

                if direct_l is None or direct_r is None:
                    continue

                # Score: variance of leaf-specific effects
                direct_var = n_left * direct_l**2 + n_right * direct_r**2
                spill_var = n_left * (spill_l or 0)**2 + n_right * (spill_r or 0)**2

                # We want high variance (more heterogeneity) across leaves
                score = direct_var + 0.5 * spill_var

                if score > best_score:
                    best_score = score
                    best_split = (col, thresh, left_idx, idx & ~left_mask)

        if best_split is None:
            direct, spillover = self._ht_estimator(Y, D, exposure, idx)
            return {
                "is_leaf": True,
                "idx": idx,
                "direct_effect": direct,
                "spillover_effect": spillover,
                "n": n_current,
                "depth": depth,
            }

        col, thresh, left_idx, right_idx = best_split

        return {
            "is_leaf": False,
            "col": col,
            "threshold": thresh,
            "n": n_current,
            "depth": depth,
            "left": self._build_tree(X, Y, D, exposure,
                                      left_idx, depth + 1),
            "right": self._build_tree(X, Y, D, exposure,
                                       right_idx, depth + 1),
        }

    def _predict_leaves(self, X):
        """Assign each training unit to a leaf and record leaf effects."""
        n = X.shape[0]
        predictions = np.zeros((n, 2))

        def _traverse(node, indices):
            if node["is_leaf"]:
                for i in np.where(indices)[0]:
                    predictions[i, 0] = node["direct_effect"] or 0.0
                    predictions[i, 1] = node["spillover_effect"] or 0.0
                return
            col = node["col"]
            thresh = node["threshold"]
            left_mask = X[:, col] <= thresh
            left_idx = indices & left_mask
            right_idx = indices & ~left_mask
            _traverse(node["left"], left_idx)
            _traverse(node["right"], right_idx)

        _traverse(self.tree_, np.ones(n, dtype=bool))
        return predictions

    def predict(self, X):
        """Assign new points to leaves and return leaf effects.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        direct_effects : ndarray, shape (n_samples,)
        spillover_effects : ndarray, shape (n_samples,)
        """
        X = np.asarray(X)
        n = X.shape[0]
        results = np.zeros((n, 2))

        for i in range(n):
            node = self.tree_
            while not node["is_leaf"]:
                if X[i, node["col"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            results[i, 0] = node["direct_effect"] or 0.0
            results[i, 1] = node["spillover_effect"] or 0.0

        return results[:, 0], results[:, 1]


def simulate_network_data(n_units=500, n_clusters=25, p=5, seed=42):
    """Simulate data with clustered network interference.

    Units are organized into clusters. Within each cluster,
    units interact (partial interference across clusters).

    True DGP:
        Y_i = 1.5 * D_i + 0.8 * exposure_i - 1.0 * D_i * exposure_i
            + X_i @ beta + noise
    """
    rng = np.random.RandomState(seed)

    # Generate cluster structure
    cluster_sizes = np.random.multinomial(
        n_units, np.ones(n_clusters) / n_clusters)
    actual_n = cluster_sizes.sum()
    cluster_id = np.repeat(np.arange(n_clusters), cluster_sizes)[:actual_n]
    n = len(cluster_id)

    # Covariates
    X = rng.randn(n, p)

    # Build adjacency matrix (within-cluster edges only, Erdos-Renyi within)
    adj = np.zeros((n, n))
    for c in range(n_clusters):
        c_idx = np.where(cluster_id == c)[0]
        n_c = len(c_idx)
        # Within-cluster edge probability 0.3
        for i in range(n_c):
            for j in range(i + 1, n_c):
                if rng.random() < 0.3:
                    adj[c_idx[i], c_idx[j]] = 1
                    adj[c_idx[j], c_idx[i]] = 1

    # Treatment: cluster-level randomization + individual noise
    cluster_treatment_prob = 0.5
    D = np.zeros(n)
    for c in range(n_clusters):
        c_idx = np.where(cluster_id == c)[0]
        # Cluster-level assignment
        treat_cluster = rng.binomial(1, cluster_treatment_prob)
        # Individual: mostly follows cluster, with some noise
        for i in c_idx:
            D[i] = treat_cluster
        # Add individual noise to some units
        flip_idx = c_idx[rng.binomial(1, 0.05, len(c_idx)) == 1]
        D[flip_idx] = 1 - D[flip_idx]

    # Exposure: proportion of treated neighbors
    deg = adj.sum(axis=1)
    n_treated_neighbors = adj @ D
    exposure = np.where(deg > 0, n_treated_neighbors / deg, 0.0)

    # Outcome with direct + spillover effects
    beta = rng.randn(p) * 0.3
    Y = (1.5 * D + 0.8 * exposure - 0.5 * D * exposure
         + X @ beta + 0.5 * rng.randn(n))

    return X, adj, D, Y, exposure, cluster_id


if __name__ == "__main__":
    print("=" * 65)
    print("Causal Inference under Network Interference")
    print("=" * 65)

    X, adj, D, Y, exposure, cluster_id = simulate_network_data(
        n_units=600, n_clusters=30, p=5, seed=42)

    print(f"Data: n = {len(D)}, features = {X.shape[1]}, "
          f"clusters = {len(np.unique(cluster_id))}")
    print(f"Treatment rate: {D.mean():.2%}")
    print(f"Mean exposure (prop. treated neighbors): {exposure.mean():.3f}")

    # Fit Network Causal Tree
    nct = NetworkCausalTree(min_leaf_size=30, max_depth=3, random_state=42)
    nct.fit(X, adj, D, Y)

    # Assess heterogeneity found
    direct_pred, spill_pred = nct.leaf_predictions_[:, 0], nct.leaf_predictions_[:, 1]

    print(f"\n--- Discovered Heterogeneity ---")
    print(f"Direct effects: mean = {direct_pred.mean():.3f}, "
          f"std = {direct_pred.std():.3f}, "
          f"range = [{direct_pred.min():.3f}, {direct_pred.max():.3f}]")
    print(f"Spillover effects: mean = {spill_pred.mean():.3f}, "
          f"std = {spill_pred.std():.3f}, "
          f"range = [{spill_pred.min():.3f}, {spill_pred.max():.3f}]")

    # Naive estimate ignoring interference
    from sklearn.linear_model import LinearRegression
    naive = LinearRegression()
    X_design = np.column_stack([D, X])
    naive.fit(X_design, Y)
    print(f"\n--- Comparison ---")
    print(f"Naive OLS (D coeff, ignores spillovers): {naive.coef_[0]:.4f}")
    print(f"True avg direct effect (simulated): ~1.5")
    print(f"True avg spillover effect (simulated): ~0.8")

    # Evaluate by exposure level
    low_exp = exposure <= np.median(exposure)
    high_exp = exposure > np.median(exposure)

    print(f"\n--- Effects by Exposure Level ---")
    print(f"Low exposure (≤ median):")
    print(f"  Treated mean: {Y[(D==1) & low_exp].mean():.3f}")
    print(f"  Control mean: {Y[(D==0) & low_exp].mean():.3f}")
    print(f"  Simple diff: {Y[(D==1) & low_exp].mean() - Y[(D==0) & low_exp].mean():.3f}")
    print(f"High exposure (> median):")
    print(f"  Treated mean: {Y[(D==1) & high_exp].mean():.3f}")
    print(f"  Control mean: {Y[(D==0) & high_exp].mean():.3f}")
    print(f"  Simple diff: {Y[(D==1) & high_exp].mean() - Y[(D==0) & high_exp].mean():.3f}")

    print(f"\nSpillover effect (control, high vs low exposure):")
    print(f"  {Y[(D==0) & high_exp].mean() - Y[(D==0) & low_exp].mean():.3f}")
```

## References

Bargagli-Stoffi, F. J., Tortu, C., & Forastiere, L. (2025). Heterogeneous treatment and spillover effects under clustered network interference. *The Annals of Applied Statistics*, 19(1), 28–55. https://doi.org/10.1214/24-AOAS1913

Leung, M. (2022). Causal inference under approximate neighborhood interference. *Econometrica*, 90(1), 267–301. https://doi.org/10.3982/ECTA17946

Forastiere, L., Airoldi, E. M., & Mealli, F. (2021). Identification and estimation of treatment and interference effects in observational studies on networks. *Journal of the American Statistical Association*, 116(534), 901–918. https://doi.org/10.1080/01621459.2020.1768100

Sobel, M. E. (2006). What do randomized studies of housing mobility demonstrate? Causal inference in the face of interference. *Journal of the American Statistical Association*, 101(476), 1398–1407. https://doi.org/10.1198/016214506000000736

Ogburn, E. L., & VanderWeele, T. J. (2014). Causal diagrams for interference. *Statistical Science*, 29(4), 559–578. https://doi.org/10.1214/14-STS501
```

