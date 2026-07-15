# Causal Forest and Generalized Random Forest

**Source**: Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics*, 47(2), 1148–1178. https://doi.org/10.1214/18-AOS1709

**Source**: Friedberg, R., Tibshirani, J., Athey, S., & Wager, S. (2021). Local linear forests. *Journal of Computational and Graphical Statistics*, 30(2), 503–517. https://doi.org/10.1080/10618600.2020.1831930

**Category**: Causal Inference / Heterogeneous Treatment Effects

## Mathematical Setup

Causal forests extend the random forest algorithm to estimate heterogeneous treatment effects (CATE). The key idea is to construct a forest that solves a local moment condition rather than predicting outcomes directly. This is formalized through the generalized random forest (GRF) framework.

### Potential Outcomes Framework

For each unit $i = 1, \dots, n$, we observe $(X_i, Y_i, D_i)$ where:
- $X_i \in \mathbb{R}^p$ are covariates
- $D_i \in \{0, 1\}$ is the treatment assignment
- $Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)$ is the observed outcome

The target estimand is the Conditional Average Treatment Effect (CATE):

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

### Moment Condition Formulation

GRF estimates $\tau(x)$ as the solution to a local moment equation at each test point $x$:

$$\sum_{i=1}^n \alpha_i(x) \psi(O_i; \theta(x)) = 0$$

where $\alpha_i(x)$ are forest-based weights (how often unit $i$ falls in the same leaf as $x$), and $\psi$ is a moment function. For a causal forest with unconfoundedness and a partially linear specification:

$$\psi(O_i; \tau) = \bigg((Y_i - \hat{m}^{(-i)}(X_i)) - \tau (D_i - \hat{\pi}^{(-i)}(X_i))\bigg) (D_i - \hat{\pi}^{(-i)}(X_i))$$

where $\hat{m}(x) = \mathbb{E}[Y \mid X=x]$ and $\hat{\pi}(x) = \mathbb{E}[D \mid X=x]$ are estimated via separate forests (local centering).

### Honest Splitting

The causal forest uses **honest splitting**: the data is split into two subsets; one is used to determine the tree structure (splits), and the other to estimate leaf parameters. This reduces overfitting bias at the cost of some efficiency.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Unconfoundedness | $Y(0), Y(1) \perp D \mid X$ | No unmeasured confounders conditional on $X$ |
| Overlap | $0 < \pi(x) = \mathbb{P}(D=1 \mid X=x) < 1$ | Every unit has a positive probability of receiving either treatment |
| Continuity | $\tau(x)$ is Lipschitz continuous in $x$ | Smooth treatment effect surface |
| Random Forest regularity | Standard RF conditions (subsampling, tree depth) | Consistency and asymptotic normality |

## Applicable Scenarios

**When to use:**
- Estimating heterogeneous treatment effects (CATE)
- Binary treatment with observational data
- High-dimensional covariate space with limited sample size
- Settings where model misspecification is a concern (nonparametric flexibility)
- Policy targeting: identifying subgroups with the largest treatment effects

**When NOT to use:**
- Very low sample size (n < 200)
- Severe confounding with weak instruments/propensity overlap
- When interpretability of the treatment effect model is paramount
- Settings with many discrete treatments (use multi-arm causal forest instead)

**Comparison:**
- Causal forest outperforms causal BART for moderate-to-large samples with many irrelevant covariates
- Local linear forest is preferred when CATE is smooth but nuisance functions are complex

## Method Details

### Step-by-Step Procedure (Causal Forest)

1. **Propensity and outcome estimation**: Estimate $\hat{\pi}(x)$ and $\hat{m}(x)$ via separate regression forests (or other methods).

2. **Construct orthogonalized outcomes and treatments**: Compute residuals $\tilde{Y}_i = Y_i - \hat{m}(X_i)$, $\tilde{D}_i = D_i - \hat{\pi}(X_i)$.

3. **Build honest causal forest**: Split sample into two halves (structure vs. estimation). Grow trees by maximizing heterogeneity in $\tau$:
   - At each candidate split, compute leaf estimates of $\tau$ on the estimation sample
   - Choose split that maximizes $\sum_{leaf} n_{leaf} \cdot \hat{\tau}_{leaf}^2$ (variance of CATE)

4. **Predict CATE**: For a new point $x$, average $\tau$ estimates across trees, weighted by forest weights $\alpha_i(x)$.

5. **Inference**: Construct confidence intervals using the infinitesimal jackknife variance estimator.

### Asymptotic Properties
- **Consistency**: $\hat{\tau}(x) \xrightarrow{p} \tau(x)$
- **Asymptotic normality**: $\sqrt{n_k} (\hat{\tau}(x) - \tau(x)) \xrightarrow{d} \mathcal{N}(0, V(x))$ (where $n_k$ is the effective leaf sample size)
- **Rate of convergence**: $O_p(n^{-\frac{1}{d+2}} \log n)$ for a $d$-dimensional smooth CATE

## Implementation Details

**Key hyperparameters:**
- `num.trees`: Number of trees (default 2000)
- `min.node.size`: Minimum node size (determines depth)
- `sample.fraction`: Fraction of data used per tree
- `honesty`: Whether to use honest splitting
- `honesty.fraction`: Fraction for splitting vs. estimation in honesty

**Available software:**
- R: `grf` package (CRAN)
- Python: `grf` Python wrapper, `econml` (causal forest via HeterogeneousTreatmentEffects)
- Stata: `grf` Stata integration

## Python Implementation

```python
"""
Causal Forest via Generalized Random Forest (GRF) Framework

References:
    Athey, Tibshirani & Wager (2019). Generalized random forests.
    Friedberg et al. (2021). Local linear forests.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy import special, linalg


class CausalForest:
    """Causal Forest for Heterogeneous Treatment Effect Estimation.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    min_leaf_size : int, default=10
        Minimum number of samples in each leaf.
    sample_fraction : float, default=0.5
        Fraction of data used per tree (subsampling).
    n_folds : int, default=5
        Number of folds for cross-fitting nuisance functions.
    random_state : int, default=42
        Random seed.

    Attributes
    ----------
    tau_hat_ : ndarray, shape (n_samples,)
        In-sample CATE estimates.
    """
    def __init__(self, n_trees=500, min_leaf_size=10,
                 sample_fraction=0.5, n_folds=5, random_state=42):
        self.n_trees = n_trees
        self.min_leaf_size = min_leaf_size
        self.sample_fraction = sample_fraction
        self.n_folds = n_folds
        self.random_state = random_state
        self.trees = []
        self.x_train = None
        self.d_resid = None
        self.y_resid = None

    def _residualize(self, X, D, Y):
        """Compute orthogonalized residuals via cross-fitting.

        Residualizes D and Y w.r.t. X using random forests.
        Returns residuals D_resid and Y_resid.
        """
        n = len(Y)
        Y_resid = np.empty(n)
        D_resid = np.empty(n)

        cv = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)

        for train_idx, test_idx in cv.split(X):
            X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y[train_idx]
            X_te, D_te, Y_te = X[test_idx], D[test_idx], Y[test_idx]

            rf_y = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=10,
                random_state=self.random_state).fit(X_tr, Y_tr)
            rf_d = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=10,
                random_state=self.random_state).fit(X_tr, D_tr)

            Y_hat = rf_y.predict(X_te)
            D_hat = rf_d.predict(X_te)

            Y_resid[test_idx] = Y_te - Y_hat
            D_resid[test_idx] = D_te - D_hat

        return Y_resid, D_resid

    def _split_node(self, X, Y_resid, D_resid, depth=0, max_depth=6):
        """Recursively split a node to maximize CATE heterogeneity."""
        n = len(Y_resid)
        if (n <= self.min_leaf_size or depth >= max_depth):
            gamma = np.sum(D_resid * Y_resid) / max(np.sum(D_resid**2), 1e-8)
            return {"is_leaf": True, "gamma": gamma, "n": n,
                    "depth": depth, "x_mean": X.mean(axis=0) if n > 0 else None}

        best_score = -np.inf
        best_col = None
        best_thresh = None

        n_features = X.shape[1]
        feat_subset = np.random.choice(n_features,
                                       size=max(1, int(np.sqrt(n_features))),
                                       replace=False)

        for col in feat_subset:
            x_col = X[:, col]
            thresholds = np.percentile(x_col, np.linspace(10, 90, 20))
            for thresh in thresholds:
                left_idx = x_col <= thresh
                right_idx = x_col > thresh

                if np.sum(left_idx) < self.min_leaf_size or \
                   np.sum(right_idx) < self.min_leaf_size:
                    continue

                # Compute CATE in left and right child
                gl = np.sum(D_resid[left_idx] * Y_resid[left_idx]) / \
                     max(np.sum(D_resid[left_idx]**2), 1e-8)
                gr = np.sum(D_resid[right_idx] * Y_resid[right_idx]) / \
                     max(np.sum(D_resid[right_idx]**2), 1e-8)

                # Score: weighted variance of leaf CATE estimates
                score = (np.sum(left_idx) * gl**2 +
                         np.sum(right_idx) * gr**2)

                if score > best_score:
                    best_score = score
                    best_col = col
                    best_thresh = thresh

        if best_col is None:
            gamma = np.sum(D_resid * Y_resid) / max(np.sum(D_resid**2), 1e-8)
            return {"is_leaf": True, "gamma": gamma, "n": n,
                    "depth": depth, "x_mean": X.mean(axis=0) if n > 0 else None}

        left_idx = X[:, best_col] <= best_thresh
        right_idx = X[:, best_col] > best_thresh

        node = {
            "is_leaf": False,
            "col": best_col,
            "threshold": best_thresh,
            "n": n,
            "depth": depth,
            "left": self._split_node(X[left_idx], Y_resid[left_idx],
                                     D_resid[left_idx],
                                     depth + 1, max_depth),
            "right": self._split_node(X[right_idx], Y_resid[right_idx],
                                      D_resid[right_idx],
                                      depth + 1, max_depth)
        }
        return node

    def _predict_tree(self, node, x):
        """Predict CATE from a single tree."""
        while not node["is_leaf"]:
            if x[node["col"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["gamma"]

    def fit(self, X, D, Y):
        """Fit the causal forest.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Covariates.
        D : array-like, shape (n_samples,)
            Binary treatment.
        Y : array-like, shape (n_samples,)
            Outcome.

        Returns
        -------
        self : CausalForest
        """
        X = np.asarray(X)
        D = np.asarray(D).ravel()
        Y = np.asarray(Y).ravel()
        n = len(Y)

        self.x_train = X

        # Step 1: Residualization (orthogonalization)
        self.y_resid, self.d_resid = self._residualize(X, D, Y)

        rng = np.random.RandomState(self.random_state)

        # Step 2: Build ensemble of causal trees
        self.trees = []
        for t in range(self.n_trees):
            # Bootstrap sample
            n_sample = max(int(n * self.sample_fraction), 100)
            idx = rng.choice(n, size=n_sample, replace=False)

            X_s = X[idx]
            Y_resid_s = self.y_resid[idx]
            D_resid_s = self.d_resid[idx]

            tree = self._split_node(X_s, Y_resid_s, D_resid_s, max_depth=6)
            self.trees.append(tree)

        # In-sample CATE predictions
        self.tau_hat_ = np.array([self.predict(X_i.reshape(1, -1))[0]
                                   for X_i in X])

        return self

    def predict(self, X):
        """Predict CATE for new samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        tau_hat : ndarray, shape (n_samples,)
            Estimated CATE values.
        """
        X = np.asarray(X)
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for t, tree in enumerate(self.trees):
            for i in range(X.shape[0]):
                predictions[i, t] = self._predict_tree(tree, X[i])
        return predictions.mean(axis=1)

    def variable_importance(self):
        """Compute variable importance (proportion of times each
        feature is used for splitting).

        Returns
        -------
        importances : ndarray, shape (n_features,)
        """
        importances = []

        def _count_usage(node, counts, depth=0):
            if node["is_leaf"]:
                return
            counts[node["col"]] += 1
            _count_usage(node["left"], counts, depth + 1)
            _count_usage(node["right"], counts, depth + 1)

        n_features = self.x_train.shape[1]
        counts = np.zeros((len(self.trees), n_features))
        for t, tree in enumerate(self.trees):
            _count_usage(tree, counts[t])

        return counts.mean(axis=0) / counts.sum(axis=1, keepdims=True).mean(axis=0)


def simulate_cate_data(n=1000, p=10, seed=42):
    """Simulate data with heterogeneous treatment effects.

    True CATE: tau(x) = 1 + 2 * x1 + sin(x2)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)

    # True CATE: depends nonlinearly on x[0] and x[1]
    tau = 1.0 + 2.0 * X[:, 0] + np.sin(X[:, 1])

    # Propensity score (confounded)
    pi = 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
    D = rng.binomial(1, pi)

    # Outcome with heterogeneous treatment effect
    # Y = tau(X) * D + g(X) + noise
    g = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 - X[:, 2]
    Y = tau * D + g + 0.5 * rng.randn(n)

    return X, D, Y, tau


if __name__ == "__main__":
    print("=" * 60)
    print("Causal Forest — Heterogeneous Treatment Effects")
    print("=" * 60)

    X, D, Y, true_tau = simulate_cate_data(n=1500, p=5, seed=42)
    print(f"Sample size: n = {len(Y)}, p = {X.shape[1]}")

    cf = CausalForest(n_trees=100, min_leaf_size=15,
                      sample_fraction=0.5, random_state=42)
    cf.fit(X, D, Y)

    # Evaluate CATE estimation accuracy
    mse = np.mean((cf.tau_hat_ - true_tau)**2)
    corr = np.corrcoef(cf.tau_hat_, true_tau)[0, 1]
    print(f"\nCATE estimation performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  Correlation with true CATE: {corr:.4f}")

    # Average treatment effect
    ate_hat = np.mean(cf.tau_hat_)
    ate_true = np.mean(true_tau)
    print(f"\nATE (mean of CATE):")
    print(f"  Estimated: {ate_hat:.4f}")
    print(f"  True:      {ate_true:.4f}")

    # Variable importance
    imp = cf.variable_importance()
    print(f"\nTop 3 most important features:")
    for idx in np.argsort(imp)[-3:][::-1]:
        print(f"  X{idx}: importance = {imp[idx]:.3f}")

    # Heterogeneity detection test
    tau_std = np.std(cf.tau_hat_)
    is_heterogeneous = tau_std > 0.2
    print(f"\nHeterogeneity detection:")
    print(f"  Std of estimated CATE: {tau_std:.3f}")
    print(f"  Detected heterogeneity: {is_heterogeneous} (threshold: 0.2)")

    print("\n--- Sensitivity: varying n_trees ---")
    for ntrees in [50, 100, 200]:
        cf_s = CausalForest(n_trees=ntrees, min_leaf_size=15,
                            random_state=42)
        cf_s.fit(X, D, Y)
        mse_s = np.mean((cf_s.tau_hat_ - true_tau)**2)
        print(f"  n_trees={ntrees}: MSE={mse_s:.4f}, "
              f"corr={np.corrcoef(cf_s.tau_hat_, true_tau)[0,1]:.4f}")
```

## References

Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics*, 47(2), 1148–1178. https://doi.org/10.1214/18-AOS1709

Friedberg, R., Tibshirani, J., Athey, S., & Wager, S. (2021). Local linear forests. *Journal of Computational and Graphical Statistics*, 30(2), 503–517. https://doi.org/10.1080/10618600.2020.1831930

Dandl, S., Haslinger, C., Hothorn, T., Seibold, H., Sverdrup, E., Wager, S., & Zeileis, A. (2024). What makes forest-based heterogeneous treatment effect estimators work? *The Annals of Applied Statistics*, 18(1), 506–528. https://doi.org/10.1214/23-AOAS1799

Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360. https://doi.org/10.1073/pnas.1510489113

Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242. https://doi.org/10.1080/01621459.2017.1319839
```

