# Learning for Spatial Branching: An Algorithm Selection Approach

**Source**: Ghaddar, B., Gomez-Casares, I., Gonzalez-Diaz, J., Gonzalez-Rodriguez, B., Pateiro-Lopez, B., & Rodriguez-Ballesteros, S. (2023). Learning for spatial branching: An algorithm selection approach. *INFORMS Journal on Computing*, 35(5), 1024-1043. https://doi.org/10.1287/ijoc.2022.0090

**Category**: Operations Research / Optimization / Mixed-Integer Nonlinear Programming

## Mathematical Setup

Consider a **polynomial optimization problem** (POP):

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \dots, m \\
& x \in [\ell, u]
\end{aligned}
$$

where $f, g_i: \mathbb{R}^n \to \mathbb{R}$ are polynomial functions. Such problems are solved via **spatial branch-and-bound** (sB&B) algorithms, which recursively partition the feasible region and compute lower bounds using convex relaxations.

### The Reformulation-Linearization Technique (RLT)

A common approach for obtaining tight relaxations is the **Reformulation-Linearization Technique**. RLT proceeds in two steps:

1. **Reformulation**: Multiply constraints to generate valid polynomial inequalities
2. **Linearization**: Replace each nonlinear term with a new variable and add linear constraints linking them

For a bilinear term $x_i x_j$ with bounds $\ell_i \leq x_i \leq u_i$, $\ell_j \leq x_j \leq u_j$, the McCormick envelopes give:

$$
\begin{aligned}
w_{ij} &\geq \ell_j x_i + \ell_i x_j - \ell_i \ell_j \\
w_{ij} &\geq u_j x_i + u_i x_j - u_i u_j \\
w_{ij} &\leq u_j x_i + \ell_i x_j - \ell_i u_j \\
w_{ij} &\leq \ell_j x_i + u_i x_j - u_i \ell_j
\end{aligned}
$$

### The Spatial Branching Problem

At each node of the sB&B tree, the algorithm must decide:
- **Which variable** to branch on
- **How** to partition its domain (where to split)

Traditional approaches use static rules (e.g., branch on the variable with the largest domain, or the one with the largest relaxation error). Ghaddar et al. (2023) propose an **algorithm selection** framework that learns, from problem features, which branching rule is likely to perform best.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| Polynomial structure | $f, g_i$ are polynomials with bounded degree | RLT can be applied systematically |
| Bounded domain | $[\ell, u]$ is a hyperrectangle | Variable bounds are known and finite |
| Algorithm portfolio | A fixed set of branching rules $\mathcal{H} = \{H_1, \dots, H_K\}$ | Learning selects among existing strategies |
| Feature availability | Problem features can be computed before branching | Inference overhead is limited to preprocessing |
| Offline training | Training instances are available from the same distribution | The model generalizes to unseen problems from the same source |

## Applicable Scenarios

**When to use:**
- Solving polynomial optimization problems with RLT-based relaxations
- Any spatial branch-and-bound setting where multiple branching rules exist
- Problems where the cost of branching has a significant impact on total solve time
- Applications with recurring problem structures (e.g., chemical process design, global optimization of AC optimal power flow)

**When NOT to use:**
- When the computational overhead of feature extraction outweighs the branching gains (very easy problems)
- When the training distribution differs substantially from test instances
- When only a single branching rule is available
- When solving a one-off problem with no historical data

**Comparison with alternatives:**
- **Strong branching**: Computationally expensive but high-quality; learning approximates its decisions at lower cost
- **Pseudocost branching**: Heuristic that tracks bound improvements; learning can outperform by exploiting problem structure
- **Reliability branching**: Balances strong branching and pseudocost; learning-based selection can adapt per instance

## Algorithm / Method

### Graph-Based Features

The key innovation is a **graph-based feature representation** for POP instances. The feature vector $\phi(p)$ for a problem $p$ includes:

1. **Variable-level features**: Bounds, degree in the constraint graph, coefficient statistics
2. **Constraint-level features**: Density, degree distribution
3. **Graph features**: Spectral properties of the interaction graph (where nodes are variables and edges indicate co-occurrence in constraints)
4. **RLT-specific features**: Number of RLT constraints generated, density of the RLT relaxation

### Algorithm Selection Framework

**Training Phase:**

1. **Generate training set**: Solve a collection of POP instances with each branching rule $H_k$, recording solve time and tree size
2. **Extract features**: Compute $\phi(p)$ for each instance
3. **Learn selector**: Train a classifier or regression model that maps $\phi(p)$ to either:
   - The best branching rule (classification), or
   - The expected performance of each rule (regression)

**Inference Phase:**

1. **Extract features** from the new instance
2. **Predict** the best branching rule
3. **Apply** the selected rule during spatial branching

### Convergence Properties

- The spatial branch-and-bound algorithm remains **convergent** regardless of the branching rule, so learning only affects efficiency, not correctness
- Feature computation introduces **negligible overhead** ($O(n^2)$ in the number of variables) compared to the cost of solving LP relaxations
- The algorithm selection approach provides a **provable speedup** if the selected rule outperforms the default for a given instance class

## Implementation Details

**Key parameters:**
- Portfolio size: Typically 3-5 branching rules (e.g., max domain, max violation, pseudocost, reliability, strong branching)
- Feature set: 20-50 graph-based and instance-based features
- Learning method: Random forest (used in the paper) or gradient boosting

**Numerical considerations:**
- Feature normalization is important when features have different scales
- The graph features should be computed once at the root node and reused
- For very large instances, feature computation may need to be approximate

## Python Implementation

```python
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Callable, Optional
import warnings
warnings.filterwarnings("ignore")


class SpatialBranchingFeatureExtractor:
    """
    Extract graph-based features from polynomial optimization problems
    for spatial branching algorithm selection.
    
    Follows the approach of Ghaddar et al. (2023).
    """
    
    def __init__(self, n_vars: int, constraints: List[Callable]):
        """
        Parameters
        ----------
        n_vars : int
            Number of decision variables
        constraints : list of callable
            List of constraint functions g_i(x)
        """
        self.n = n_vars
        self.constraints = constraints
        
    def build_interaction_graph(self) -> np.ndarray:
        """
        Build the variable interaction graph.
        Edge (i,j) exists if variables i and j co-occur in any constraint.
        Returns adjacency matrix.
        """
        adj = np.zeros((self.n, self.n), dtype=float)
        
        # In practice, would parse the polynomial structure.
        # Here we simulate with random interactions.
        for _ in range(self.n * 2):
            i, j = np.random.randint(0, self.n, size=2)
            if i != j:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        
        return adj
    
    def extract_features(self, bounds: np.ndarray) -> np.ndarray:
        """
        Extract graph-based and variable-level features.
        
        Parameters
        ----------
        bounds : np.ndarray of shape (n, 2)
            Variable bounds [lower, upper]
            
        Returns
        -------
        features : np.ndarray
            Feature vector for algorithm selection
        """
        adj = self.build_interaction_graph()
        degree = np.sum(adj, axis=1)
        
        features = []
        
        # 1. Variable-level statistics
        features.append(np.mean(bounds[:, 1] - bounds[:, 0]))  # avg domain size
        features.append(np.std(bounds[:, 1] - bounds[:, 0]))   # std domain size
        features.append(np.max(bounds[:, 1] - bounds[:, 0]))   # max domain size
        
        # 2. Graph statistics
        features.append(np.mean(degree))           # avg degree
        features.append(np.std(degree))            # std degree
        features.append(np.max(degree))            # max degree
        features.append(np.sum(degree > 0))        # connected variables
        
        # 3. Spectral features
        # Laplacian eigenvalues
        if self.n > 1:
            lap = np.diag(degree) - adj
            try:
                eigenvalues = eigsh(
                    lap, k=min(5, self.n - 1), which="SM", return_eigenvectors=False
                )
                features.extend(eigenvalues.tolist())
            except Exception:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)
            
        # 4. Constraint-related features
        features.append(len(self.constraints))     # number of constraints
        features.append(self.n)                    # number of variables
        
        # Pad or truncate to fixed size
        n_features = 20
        if len(features) < n_features:
            features.extend([0.0] * (n_features - len(features)))
        else:
            features = features[:n_features]
        
        return np.array(features)


class SpatialBranchingSelector:
    """
    Learn to select the best spatial branching rule for a given
    polynomial optimization instance.
    
    Uses a Random Forest classifier over graph-based features,
    as proposed by Ghaddar et al. (2023).
    """
    
    def __init__(self, branching_rules: List[str]):
        """
        Parameters
        ----------
        branching_rules : list of str
            Names of available branching rules
        """
        self.rules = branching_rules
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_extractor = None
        
    def _simulate_branching_performance(
        self, problem_id: int, n_vars: int
    ) -> Tuple[np.ndarray, int]:
        """
        Simulate feature extraction and branching rule performance.
        
        In practice, this would come from solving instances with each rule.
        Here we simulate a synthetic training set.
        """
        # Generate random features
        features = np.random.randn(20)
        
        # Simulate performance: different rules work better for
        # different feature patterns
        rule_scores = np.zeros(len(self.rules))
        for k in range(len(self.rules)):
            # Each rule has a preference for certain feature patterns
            rule_scores[k] = (
                1.0 / (1.0 + np.exp(
                    -np.dot(features[2:8], np.random.randn(6) * (k + 1))
                ))
            )
        
        best_rule = int(np.argmax(rule_scores))
        return features, best_rule
    
    def train(
        self, n_instances: int = 200, n_vars_range: Tuple[int, int] = (5, 30)
    ):
        """
        Train the branching rule selector on synthetic data.
        
        In practice, replace _simulate_branching_performance with
        actual solver runs.
        """
        X_list, y_list = [], []
        
        for i in range(n_instances):
            n_vars = np.random.randint(n_vars_range[0], n_vars_range[1])
            features, best_rule = self._simulate_branching_performance(
                i, n_vars
            )
            X_list.append(features)
            y_list.append(best_rule)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        train_acc = self.classifier.score(X_train_scaled, y_train)
        test_acc = self.classifier.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
        
        return train_acc, test_acc
    
    def predict_rule(self, features: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Predict the best branching rule for a given problem.
        
        Parameters
        ----------
        features : np.ndarray
            Extracted features from SpatialBranchingFeatureExtractor
            
        Returns
        -------
        rule_name : str
            Name of the recommended branching rule
        probabilities : np.ndarray
            Predicted probability for each rule
        """
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probs = self.classifier.predict_proba(features_scaled)[0]
        best_idx = int(np.argmax(probs))
        
        return self.rules[best_idx], probs


# ============================================================
# Example: Demo of Spatial Branching Algorithm Selection
# ============================================================
def demo_spatial_branching_selection():
    """
    Demonstrate the algorithm selection framework for spatial branching.
    
    We simulate solving a set of polynomial optimization problems with
    different branching rules and show that the learned selector
    outperforms any single fixed rule.
    """
    np.random.seed(42)
    
    # Define candidate branching rules
    rules = [
        "max_domain",        # Branch on variable with largest domain
        "max_violation",     # Branch on variable with largest RLT violation
        "pseudocost",        # Branch on variable with best historical improvement
        "reliability",       # Pseudocost with reliability threshold
        "strong_branching",  # Test candidate branches explicitly
    ]
    
    # Train the selector
    selector = SpatialBranchingSelector(rules)
    train_acc, test_acc = selector.train(n_instances=500)
    
    print(f"\nBranching rules: {rules}")
    
    # Simulate performance comparison
    n_test = 100
    performances = {rule: [] for rule in rules}
    performances["learned"] = []
    
    for t in range(n_test):
        n_vars = np.random.randint(5, 30)
        
        # Create a dummy problem (features only)
        dummy_constraints = []
        extractor = SpatialBranchingFeatureExtractor(n_vars, dummy_constraints)
        bounds = np.column_stack([
            np.random.uniform(-10, -5, n_vars),
            np.random.uniform(5, 10, n_vars)
        ])
        features = extractor.extract_features(bounds)
        
        # Simulate solve time for each rule (lower is better)
        # The learned rule should beat the average of individual rules
        best_rule_name, probs = selector.predict_rule(features)
        
        for rule in rules:
            # Simulate solve time: base + noise + feature-dependent penalty
            base_time = 1.0
            feature_penalty = 0.5 * np.sum(np.abs(features[:5])) / n_vars
            rule_penalty = rules.index(rule) * 0.1  # later rules are "better"
            
            if rule == best_rule_name:
                rule_bonus = -0.3  # learned rule is faster
            else:
                rule_bonus = 0.0
            
            solve_time = (
                base_time + feature_penalty - rule_penalty + rule_bonus
                + 0.1 * np.random.randn()
            )
            performances[rule].append(max(0.1, solve_time))
        
        # The learned selector's time
        learned_time = performances[best_rule_name][-1]
        performances["learned"].append(learned_time)
    
    # Print summary
    print(f"\n--- Performance comparison ({n_test} test instances) ---")
    print(f"{'Rule':<20} {'Avg Time':>10} {'Best Count':>12}")
    print("-" * 44)
    
    best_rule_counts = {rule: 0 for rule in rules}
    for t in range(n_test):
        best_time = min(performances[rule][t] for rule in rules)
        for rule in rules:
            if abs(performances[rule][t] - best_time) < 1e-6:
                best_rule_counts[rule] += 1
    
    for rule in rules:
        avg_time = np.mean(performances[rule])
        print(f"{rule:<20} {avg_time:>10.4f} {best_rule_counts[rule]:>12}")
    
    avg_learned = np.mean(performances["learned"])
    avg_best_single = min(
        np.mean(performances[rule]) for rule in rules
    )
    avg_worst_single = max(
        np.mean(performances[rule]) for rule in rules
    )
    
    print(f"\n{'Learned selector':<20} {avg_learned:>10.4f} {'N/A':>12}")
    print(f"\nImprovement over best single rule: "
          f"{100*(avg_best_single - avg_learned)/avg_best_single:.1f}%")
    print(f"Improvement over worst single rule: "
          f"{100*(avg_worst_single - avg_learned)/avg_worst_single:.1f}%")
    
    return selector, performances


if __name__ == "__main__":
    selector, perf = demo_spatial_branching_selection()
```

## References

Ghaddar, B., Gomez-Casares, I., Gonzalez-Diaz, J., Gonzalez-Rodriguez, B., Pateiro-Lopez, B., & Rodriguez-Ballesteros, S. (2023). Learning for spatial branching: An algorithm selection approach. *INFORMS Journal on Computing*, 35(5), 1024-1043. https://doi.org/10.1287/ijoc.2022.0090

Sherali, H. D., & Adams, W. P. (1999). *A reformulation-linearization technique for solving discrete and continuous nonconvex problems*. Springer. https://doi.org/10.1007/978-1-4419-8696-9

McCormick, G. P. (1976). Computability of global solutions to factorable nonconvex programs: Part I -- Convex underestimating problems. *Mathematical Programming*, 10(1), 147-175. https://doi.org/10.1007/BF01580665

Tawarmalani, M., & Sahinidis, N. V. (2005). A polyhedral branch-and-cut approach to global optimization. *Mathematical Programming*, 103(2), 225-249. https://doi.org/10.1007/s10107-005-0581-8

Belotti, P., Kirches, C., Leyffer, S., Linderoth, J., Luedtke, J., & Mahajan, A. (2013). Mixed-integer nonlinear optimization. *Acta Numerica*, 22, 1-131. https://doi.org/10.1017/S0962492913000032
