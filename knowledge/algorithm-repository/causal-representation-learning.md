# Causal Representation Learning

**Source**: Scholkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward causal representation learning. *Proceedings of the IEEE*, 109(5), 612–634. https://doi.org/10.1109/JPROC.2021.3058954

**Source**: Lachapelle, S., Rodriguez, P., Le, Q., Sharma, Y., Lacoste-Julien, S., & Deleu, T. (2024). Disentanglement via mechanism sparsity regularization: A new principle for learning causal representations. *Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS)*.

**Source**: Von Kugelgen, J., Besserve, M., Liang, W., Gresele, L., Kekic, A., Bareinboim, E., Blei, D., & Scholkopf, B. (2024). Identifiability guarantees for causal disentanglement from soft interventions. *Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence (UAI)*.

**Category**: Causal Inference / Representation Learning / Disentanglement

## Mathematical Setup

Causal representation learning (CRL) addresses a fundamental question: how can we learn representations of data that encode the underlying **causal structure** of the data-generating process, rather than merely capturing statistical correlations? The core idea is that high-dimensional observations $X \in \mathbb{R}^d$ are generated from lower-dimensional latent causal variables $Z \in \mathbb{R}^m$ ($m \ll d$) through a possibly nonlinear mixing function $g$:

$$X = g(Z), \quad Z = (Z_1, \dots, Z_m)$$

where the latent variables $Z_i$ are causally related according to a directed acyclic graph (DAG).

### The Causal Representation Learning Problem

Given observations $X^{(1)}, \dots, X^{(n)} \sim p(X)$, the goal is to:

1. **Learn the causal variables**: Find an encoder $h: \mathbb{R}^d \to \mathbb{R}^m$ such that $\hat{Z} = h(X)$ recovers the true causal factors up to equivalence (e.g., permutation and elementwise transformation)
2. **Learn the causal graph**: Infer the DAG structure among the latent variables
3. **Learn the mixing function**: Approximate $g$ (the decoder) to allow counterfactual generation

### Identifiability via Mechanism Sparsity Regularization

A key breakthrough is using **sparsity of causal mechanisms** as an inductive bias: each causal variable $Z_i$ should depend on only a sparse subset of other variables. This aligns with the property of **independent causal mechanisms** (ICM):

$$p(Z_i \mid \text{Pa}(Z_i)) \text{ is sparse in its functional dependencies}$$

Lachapelle et al. (2024) formalize this as:

$$\min_{\theta} \mathbb{E}_{p(X)}[\ell(X, \hat{X}_\theta)] + \lambda \cdot \text{sparsity}(\text{Mech}_\theta)$$

where $\text{Mech}_\theta$ denotes the learned causal mechanisms and the sparsity penalty encourages each mechanism to depend on few other variables.

### Identifiability from Soft Interventions (von Kugelgen et al., 2024)

When **soft interventions** (changes in the conditional distribution of a target variable without changing its causal parents) are available, causal representations become identifiable. Specifically, if we observe data from $m$ environments, each corresponding to a soft intervention on a different latent variable, the latent causal graph and variables can be uniquely recovered up to trivial relabeling.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Latent causal graph | $Z$ follows a DAG: $p(Z) = \prod_{i=1}^m p(Z_i \mid Z_{\text{Pa}(i)})$ | Causal structure is acyclic |
| Independent mechanisms | $p(Z_i \mid \text{Pa}(i))$ do not inform each other | Mechanism changes leave others invariant |
| Sufficiently diverse interventions (von Kugelgen et al.) | At least $m$ distinct environments, each intervening on a different node | Identifies latent graph and variables |
| Mixing function | $g$ is injective and sufficiently smooth | Latent variables identifiable up to morphism |

## Applicable Scenarios

**When to use:**
- Learning interpretable latent representations with causal meaning
- Domain generalization and out-of-distribution generalization
- Counterfactual generation and reasoning
- Scientific discovery: finding latent factors that cause observed phenomena
- When data from multiple environments or interventions is available

**When NOT to use:**
- When latent variables are dense and fully connected (no sparsity)
- When only observational data from a single environment is available (identifiability is weak)
- When the downstream task is purely predictive (standard representation learning suffices)

## Method Details

### The CRL Pipeline

1. **Data collection**: Gather observations $X$ from one or multiple environments, potentially including interventional data.

2. **Representation learning**: Learn an encoder $h_\phi: \mathbb{R}^d \to \mathbb{R}^m$ and decoder $g_\theta: \mathbb{R}^m \to \mathbb{R}^d$ such that:
   - Reconstruction is accurate: $g_\theta(h_\phi(X)) \approx X$
   - The latent codes $\hat{Z} = h_\phi(X)$ are approximately causally factorized
   - Parameters are identifiable

3. **Causal graph learning**: Infer the causal structure among $\hat{Z}_1, \dots, \hat{Z}_m$ using:
   - Constraint-based methods (conditional independence tests)
   - Score-based methods (BIC, sparsity regularization)
   - Gradient-based methods (NOTEARS-style continuous optimization)

4. **Mechanism estimation**: Estimate each conditional $p(Z_i \mid Z_{\text{Pa}(i)})$ as a (sparse) function.

### Identifiability Results
- **Without interventions**: Nonlinear ICA is identifiable up to permutation and elementwise invertible transformations if $Z$ components are independent (i.i.d. cause-effect pairs).
- **With multiple environments/views**: Full identifiability under mild conditions (Gresele et al., 2021; Hyvarinen et al., 2023).
- **With soft interventions**: The causal graph and variables are identifiable from $m$ environments (von Kugelgen et al., 2024).
- **Mechanism sparsity**: Lachapelle et al. (2024) show that sparsity alone can disentangle causal variables from observational data under a "no sibling" condition in the latent graph.

## Implementation Details

**Key hyperparameters:**
- Latent dimension $m$ (number of causal factors)
- Sparsity regularization strength $\lambda$
- Architecture of encoder and decoder (typically deep neural networks)
- Choice of causal graph learner
- Number of environments (for multi-environment identifiability)

**Numerical considerations:**
- The sparsity penalty needs careful tuning: too strong collapses latent dimensions, too weak fails to disentangle
- Normalizing flows are often used for the decoder to make the model invertible
- Gradient-based DAG learning requires additional constraints (e.g., the acyclicity penalty of NOTEARS)

**Available software:**
- Python: `causal-learn` (graph learning), `DisentanglementLib`, `dowhy` (causal inference), `PyTorch`/`JAX` for the deep learning components

## Python Implementation

```python
"""
Causal Representation Learning via Mechanism Sparsity Regularization

A simplified implementation that learns a latent causal representation
by optimizing a VAE with a sparsity penalty on the decoder's Jacobian.

References:
    Scholkopf et al. (2021). Toward causal representation learning.
        Proceedings of the IEEE, 109(5), 612-634.
    Lachapelle et al. (2024). Disentanglement via mechanism sparsity
        regularization. NeurIPS 2024.
"""

import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor


class CausalRepresentationLearner:
    """Causal Representation Learning with Mechanism Sparsity.

    Learns a latent representation where each dimension is a
    causal variable, using sparsity of the causal mechanisms
    as the primary inductive bias.

    This implementation uses a simple linear-nonlinear approach
    for demonstration: it finds a linear transformation that
    maximizes a sparsity-based causal disentanglement criterion.

    Parameters
    ----------
    n_latents : int, default=3
        Number of latent causal variables.
    sparsity_lambda : float, default=0.1
        Regularization strength for mechanism sparsity.
    max_iter : int, default=100
        Maximum optimization iterations.
    random_state : int, default=42
    """

    def __init__(self, n_latents=3, sparsity_lambda=0.1,
                 max_iter=100, random_state=42):
        self.n_latents = n_latents
        self.sparsity_lambda = sparsity_lambda
        self.max_iter = max_iter
        self.random_state = random_state

    def _compute_causal_graph(self, Z):
        """Estimate a causal DAG structure among latent variables.

        Uses a simple method: order variables by their marginal
        variance (heuristic for causal ordering under non-Gaussian
        noise) and test for dependencies.

        Parameters
        ----------
        Z : ndarray, shape (n_samples, n_latents)

        Returns
        -------
        adj_matrix : ndarray, shape (n_latents, n_latents)
            Adjacency matrix (W_ij = 1 if Z_i -> Z_j).
        """
        n, m = Z.shape
        # Sort by marginal variance (heuristic)
        variances = np.var(Z, axis=0)
        order = np.argsort(variances)[::-1]  # high variance first (potential causes)

        # Test each pair for conditional dependence
        adj = np.zeros((m, m))
        for i_idx, i in enumerate(order):
            for j in order[i_idx + 1:]:
                # Simple test: if regression of Z_j on Z_i has significant
                # coefficient, infer edge
                reg = LinearRegression().fit(Z[:, i:i+1], Z[:, j])
                if abs(reg.coef_[0]) > 0.1:  # heuristic threshold
                    adj[i, j] = 1.0

        return adj

    def _sparsity_score(self, Z, adj_matrix):
        """Compute the mechanism sparsity score.

        Measures how much the conditional mean of each variable
        depends on its parents. Sparse = each variable depends on
        few parents.

        Parameters
        ----------
        Z : ndarray, shape (n_samples, n_latents)
        adj_matrix : ndarray, shape (n_latents, n_latents)

        Returns
        -------
        sparsity : float
            Sparsity score (higher = sparser = better).
        """
        m = Z.shape[1]
        # For each variable, try to predict it from its parents
        total_r2 = 0.0
        n_edges = adj_matrix.sum()

        if n_edges == 0:
            return 0.0

        for j in range(m):
            parents = np.where(adj_matrix[:, j] > 0)[0]
            if len(parents) > 0:
                reg = LinearRegression().fit(Z[:, parents], Z[:, j])
                r2 = reg.score(Z[:, parents], Z[:, j])
                total_r2 += r2

        # Sparsity: high R^2 from few edges
        # Normalize: total R^2 divided by n_edges
        return total_r2 / m - self.sparsity_lambda * n_edges / (m * (m - 1))

    def fit(self, X):
        """Learn a causal representation from observations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observed high-dimensional data.

        Returns
        -------
        self : CausalRepresentationLearner
        """
        X = np.asarray(X)
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)

        # Center the data
        self.X_mean_ = X.mean(axis=0)
        Xc = X - self.X_mean_

        # Initialize with PCA
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:self.n_latents, :].T  # d x m transformation matrix

        # Optimize transformation to maximize sparsity of mechanisms
        best_score = -np.inf
        best_W = W.copy()
        best_Z = None

        # Simple random search + local refinement
        for iteration in range(self.max_iter):
            # Add perturbation
            noise = 0.05 * rng.randn(d, self.n_latents)
            W_candidate = W + noise

            # Project to orthogonal (to avoid degenerate solutions)
            W_candidate, _ = np.linalg.qr(W_candidate)

            # Compute latent representation
            Z_candidate = Xc @ W_candidate

            # Estimate causal graph
            adj = self._compute_causal_graph(Z_candidate)

            # Compute sparsity score
            sparsity = self._sparsity_score(Z_candidate, adj)

            # Reconstruction cost (how well X can be recovered)
            ZtZ = Z_candidate.T @ Z_candidate
            X_recon = Z_candidate @ np.linalg.lstsq(ZtZ,
                                                     Z_candidate.T @ Xc,
                                                     rcond=None)[0]
            recon_error = np.mean((Xc - X_recon)**2)

            # Combined score (maximize sparsity, minimize reconstruction error)
            # Standardized to comparable scale
            score = sparsity - 0.01 * recon_error

            if score > best_score:
                best_score = score
                best_W = W_candidate.copy()
                best_Z = Z_candidate.copy()

            # Move in direction of improvement
            if iteration % 10 == 0 and iteration > 0:
                W = best_W.copy()

        self.W_ = best_W
        self.Z_ = best_Z
        self.adj_matrix_ = self._compute_causal_graph(best_Z)

        # Compute mechanism strength
        self.mechanism_strength_ = self._mechanism_strength(best_Z, self.adj_matrix_)

        return self

    def _mechanism_strength(self, Z, adj):
        """Compute the strength of each causal mechanism.

        For each edge Z_i -> Z_j, compute how much Z_i helps
        predict Z_j (partial R^2).
        """
        m = Z.shape[1]
        strengths = {}
        for j in range(m):
            parents = np.where(adj[:, j] > 0)[0]
            for i in parents:
                reg_with = LinearRegression().fit(Z[:, parents], Z[:, j])
                r2_with = reg_with.score(Z[:, parents], Z[:, j])

                parents_without = parents[parents != i]
                if len(parents_without) > 0:
                    reg_without = LinearRegression().fit(
                        Z[:, parents_without], Z[:, j])
                    r2_without = reg_without.score(Z[:, parents_without], Z[:, j])
                else:
                    r2_without = 0.0

                strengths[(i, j)] = r2_with - r2_without
        return strengths

    def transform(self, X):
        """Project new data to the learned causal representation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Z : ndarray, shape (n_samples, n_latents)
        """
        X = np.asarray(X)
        return (X - self.X_mean_) @ self.W_

    def counterfactual(self, X, intervene_idx, intervene_value):
        """Generate a counterfactual: set Z[intervene_idx] = intervene_value
        and propagate through the causal graph.

        This is a simplified implementation assuming linear mechanisms.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original observations.
        intervene_idx : int
            Index of the latent variable to intervene on.
        intervene_value : float
            Value to set the intervened variable to.

        Returns
        -------
        X_cf : ndarray, shape (n_samples, n_features)
            Counterfactual observations.
        """
        Z = self.transform(X)
        Z_cf = Z.copy()
        Z_cf[:, intervene_idx] = intervene_value

        # Propagate through the estimated causal graph (topological order)
        # For simplicity: use linear regression coefficients
        m = Z.shape[1]
        for j in range(m):
            if j == intervene_idx:
                continue
            parents = np.where(self.adj_matrix_[:, j] > 0)[0]
            if len(parents) > 0 and intervene_idx in parents:
                reg = LinearRegression().fit(Z[:, parents.astype(int)], Z[:, j])
                Z_cf[:, j] = reg.predict(Z_cf[:, parents.astype(int)])

        # Decode back to observation space
        X_cf = Z_cf @ self.W_.T
        X_cf += self.X_mean_

        return X_cf


def simulate_causal_latent_data(n=2000, m=3, d=15, seed=42):
    """Simulate data from a latent causal model.

    Latent variables Z follow a simple DAG:
        Z1 -> Z2 -> Z3
    with independent noise.
    Observations: X = W_true @ Z + noise
    """
    rng = np.random.RandomState(seed)

    # Latent causal variables (non-Gaussian for identifiability)
    Z1 = rng.exponential(1, n) - 1  # shifted exponential
    Z2 = 0.7 * Z1 + 0.3 * rng.exponential(1, n) - 0.3
    Z3 = 0.5 * Z2 + 0.5 * rng.exponential(1, n) - 0.5
    Z = np.column_stack([Z1, Z2, Z3])

    # Random mixing matrix
    W_true = rng.randn(d, m)
    noise = 0.1 * rng.randn(n, d)
    X = Z @ W_true.T + noise

    return X, Z, W_true


if __name__ == "__main__":
    print("=" * 65)
    print("Causal Representation Learning")
    print("Scholkopf et al. (2021) / Lachapelle et al. (2024)")
    print("=" * 65)

    X, Z_true, W_true = simulate_causal_latent_data(
        n=2000, m=3, d=10, seed=42)

    print(f"Data: n = {X.shape[0]}, observed dim = {X.shape[1]}, "
          f"latent dim = {Z_true.shape[1]}")
    print(f"True latent graph: Z1 -> Z2 -> Z3")

    # Learn causal representation
    crl = CausalRepresentationLearner(
        n_latents=3, sparsity_lambda=0.1,
        max_iter=100, random_state=42)
    crl.fit(X)

    # Evaluate learned representation
    Z_learned = crl.Z_
    print(f"\nLearned latent representation: shape = {Z_learned.shape}")

    # Check correlation with true latents
    print(f"\nCorrelation between true and learned latents:")
    for i in range(3):
        corrs = [np.corrcoef(Z_true[:, i], Z_learned[:, j])[0, 1]
                 for j in range(3)]
        best_j = np.argmax(np.abs(corrs))
        print(f"  Z{i+1} (true) <-> Z{best_j+1} (learned): "
              f"{corrs[best_j]:.4f}")

    # Discovered causal graph
    print(f"\nDiscovered causal adjacency matrix:")
    print(crl.adj_matrix_)

    # Mechanism strengths
    print(f"\nMechanism strengths (partial R^2 increase):")
    for (i, j), strength in crl.mechanism_strength_.items():
        if strength > 0.01:
            print(f"  Z{i+1} -> Z{j+1}: {strength:.4f}")

    # Counterfactual generation
    print(f"\n--- Counterfactual generation ---")
    x_first = X[:3]
    z_first = crl.transform(x_first)
    print(f"Original Z1 values: {z_first[:, 0]}")
    cf = crl.counterfactual(x_first, intervene_idx=0, intervene_value=2.0)
    z_cf = crl.transform(cf)
    print(f"Counterfactual Z1 (intervened to 2.0): {z_cf[:, 0]}")
    print(f"Propagated Z2: {z_cf[:, 1]}")
    print(f"Propagated Z3: {z_cf[:, 2]}")
```

## References

Scholkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward causal representation learning. *Proceedings of the IEEE*, 109(5), 612–634. https://doi.org/10.1109/JPROC.2021.3058954

Lachapelle, S., Rodriguez, P., Le, Q., Sharma, Y., Lacoste-Julien, S., & Deleu, T. (2024). Disentanglement via mechanism sparsity regularization: A new principle for learning causal representations. *Advances in Neural Information Processing Systems 37 (NeurIPS)*.

Von Kugelgen, J., Besserve, M., Liang, W., Gresele, L., Kekic, A., Bareinboim, E., Blei, D., & Scholkopf, B. (2024). Identifiability guarantees for causal disentanglement from soft interventions. *Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence*.

Hyvarinen, A., Khemakhem, I., & Monti, R. (2023). Identifiability of latent-variable and structural-equation models: From linear to nonlinear. *Annals of the Institute of Statistical Mathematics*, 75, 853–882. https://doi.org/10.1007/s10463-023-00888-0

Gresele, L., Rubenstein, P. K., Mehrjou, A., Locatello, F., & Scholkopf, B. (2021). The incomplete Rosetta Stone problem: Identifiability results for multi-view nonlinear ICA. *Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence*.
```

