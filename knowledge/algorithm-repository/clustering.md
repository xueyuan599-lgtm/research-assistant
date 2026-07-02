# K-Means / DBSCAN — 聚类

- **来源**: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297. Ester, M. et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96*, 226–231.
- **方法类别**: 机器学习 / 无监督学习

## 数学设定

### K-Means

#### 目标函数

给定数据集 $\{x_1, x_2, \dots, x_N\}$，$x_i \in \mathbb{R}^d$，K-Means 将数据划分为 $K$ 个互不相交的簇 $C_1, C_2, \dots, C_K$，最小化簇内平方和（within-cluster sum of squares, WCSS，又称 inertia）：

$$
\min_{\{C_k\}, \{\mu_k\}} \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|_2^2
$$

其中 $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$ 是簇 $k$ 的质心（centroid）。

等价地，可写作：

$$
\min_{\{C_k\}} \sum_{k=1}^{K} \frac{1}{2|C_k|} \sum_{i,j \in C_k} \|x_i - x_j\|_2^2
$$

#### Lloyd 算法

交替优化（坐标下降法的特例），保证单调收敛到局部最优（有限步内）：

**Assignment step（E-step）**：将每个样本分配到最近的质心

$$
C_k^{(t)} = \left\{ x_i : \|x_i - \mu_k^{(t)}\|_2^2 \leq \|x_i - \mu_j^{(t)}\|_2^2, \; \forall j \right\}
$$

**Update step（M-step）**：重新计算每个簇的质心

$$
\mu_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{i \in C_k^{(t)}} x_i
$$

#### 初始化方法

- **Forgy（随机选择）**：从数据中随机选取 $K$ 个样本作为初始质心
- **K-Means++**（Arthur & Vassilvitskii, 2007）：逐个选择质心，每个新质心以概率正比于 $D(x)^2$ 被选中，其中 $D(x)$ 是 $x$ 到已选最近质心的距离

  算法：
  1. 随机选第一个质心 $\mu_1$
  2. 对每个样本 $x_i$，计算 $D(x_i) = \min_{j} \|x_i - \mu_j\|_2$
  3. 以概率 $\frac{D(x_i)^2}{\sum_j D(x_j)^2}$ 选下一个质心
  4. 重复 2–3 直到选出 $K$ 个质心

  K-Means++ 可将最优解的下界保证从 $O(\log K)$ 因子提升至 $\Omega(\log K)$-competitive。

#### Kernel K-Means

通过映射函数 $\phi: \mathbb{R}^d \to \mathcal{H}$ 将数据映射到高维特征空间，利用核技巧 $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ 隐式计算：

$$
\min_{\{C_k\}} \sum_{k=1}^{K} \sum_{i \in C_k} \|\phi(x_i) - \mu_k^\phi\|_2^2, \quad \mu_k^\phi = \frac{1}{|C_k|} \sum_{i \in C_k} \phi(x_i)
$$

等价地，利用核矩阵 $K_{ij} = K(x_i, x_j)$，无需显式计算 $\phi(\cdot)$。

---

### DBSCAN（Density-Based Spatial Clustering of Applications with Noise）

#### 基本概念

给定数据集 $D$，参数 $(\varepsilon, \text{MinPts})$：

- **$\varepsilon$-邻域**：$N_\varepsilon(p) = \{ q \in D \mid \text{dist}(p, q) \leq \varepsilon \}$
- **核心点（Core point）**：$|N_\varepsilon(p)| \geq \text{MinPts}$
- **边界点（Border point）**：$|N_\varepsilon(p)| < \text{MinPts}$ 但 $\exists q \in D$ 使得 $p \in N_\varepsilon(q)$ 且 $q$ 为核心点
- **噪声点（Noise point）**：既非核心点也非边界点

#### 密度可达性与密度连接性

- **直接密度可达（Directly density-reachable）**：$p$ 是核心点且 $q \in N_\varepsilon(p)$
- **密度可达（Density-reachable）**：存在链 $p_1, p_2, \dots, p_n$ 使得 $p_1 = p$, $p_n = q$，且 $p_{i+1}$ 从 $p_i$ 直接密度可达（非对称关系）
- **密度连接（Density-connected）**：存在 $o \in D$ 使得 $p$ 和 $q$ 均从 $o$ 密度可达（对称关系）

#### 簇形成

簇是满足以下条件的最大非空子集 $C \subseteq D$：

1. **极大性**：$\forall p, q \in D$，若 $p \in C$ 且 $q$ 从 $p$ 密度可达，则 $q \in C$
2. **连接性**：$\forall p, q \in C$，$p$ 和 $q$ 是密度连接的

等价于簇是密度可达关系的传递闭包。

#### 算法步骤

1. 遍历所有点，标记核心点（$|N_\varepsilon(p)| \geq \text{MinPts}$）
2. 对每个未分配的核心点，创建一个新簇，将其所有密度可达点加入该簇（BFS/DFS 扩散）
3. 所有未分配的非核心点标记为噪声

**计算瓶颈**：区域查询（Region Query）。通过空间索引（KD-Tree, Ball Tree, R-Tree）可将平均复杂度从 $O(N^2)$ 降至 $O(N \log N)$。

---

### 聚类评估指标

#### 内部指标（无需真实标签）

- **Silhouette Coefficient（轮廓系数）**（Rousseeuw, 1987）：

  $$
  s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}, \quad s(i) \in [-1, 1]
  $$

  其中 $a(i)$ 是 $i$ 到同簇所有其他点的平均距离（簇内凝聚度），$b(i) = \min_{k \neq C(i)} \frac{1}{|C_k|} \sum_{j \in C_k} \|x_i - x_j\|_2$ 是到最近邻簇的平均距离（簇间分离度）。

  全局轮廓系数：$\text{SC} = \frac{1}{N} \sum_{i=1}^{N} s(i)$

- **Davies-Bouldin Index（DBI）**：

  $$
  \text{DBI} = \frac{1}{K} \sum_{k=1}^{K} \max_{j \neq k} \frac{\bar{d}_k + \bar{d}_j}{\|\mu_k - \mu_j\|_2}
  $$

  其中 $\bar{d}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \|x_i - \mu_k\|_2$。DBI 越小越好。

- **Calinski-Harabasz Index（CH / Variance Ratio Criterion）**：

  $$
  \text{CH} = \frac{\text{tr}(B_K)}{\text{tr}(W_K)} \cdot \frac{N - K}{K - 1}
  $$

  其中 $B_K = \sum_{k=1}^{K} |C_k| (\mu_k - \mu)(\mu_k - \mu)^\top$ 为簇间散度矩阵，$W_K = \sum_{k=1}^{K} \sum_{i \in C_k} (x_i - \mu_k)(x_i - \mu_k)^\top$ 为簇内散度矩阵，$\mu$ 为全局均值。CH 越大越好。

- **Dunn Index**：

  $$
  \text{Dunn} = \frac{\min_{i < j} \delta(C_i, C_j)}{\max_{1 \leq k \leq K} \Delta(C_k)}
  $$

  其中 $\delta(C_i, C_j) = \min_{x \in C_i, y \in C_j} \|x - y\|_2$（最小簇间距离），$\Delta(C_k) = \max_{x, y \in C_k} \|x - y\|_2$（最大簇内直径）。Dunn 越大越好。

#### 外部指标（需真实标签）

- **Adjusted Rand Index（ARI）**：在随机划分的期望值上校正的 Rand Index

  $$
  \text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]} \in [-1, 1]
  $$

  其中 RI = (一致对数量) / (总对数量)。

- **Normalized Mutual Information（NMI）**：

  $$
  \text{NMI} = \frac{2 \cdot I(Y; \hat{Y})}{H(Y) + H(\hat{Y})}
  $$

  其中 $I$ 为互信息，$H$ 为熵。NMI $\in [0, 1]$。

---

## 关键假设

### K-Means
- **簇为球形（spherical）**：各向同性的协方差结构，K-Means 倾向产生 Voronoi 划分
- **簇大小近似相等**：大簇会"吸引"小簇的边界点（因为质心受大簇主导）
- **簇方差近似相等**：方差大的簇会吞噬方差小的簇的边界区域
- **欧氏距离有意义**：特征应在相同尺度上，维度间可比较
- **$K$ 已知**：必须预先指定簇数
- **无异常值**：离群点会显著扭曲质心位置

### DBSCAN
- **簇是高密度区域被低密度区域分隔**：密度的相对差异定义了簇边界
- **$\varepsilon$ 和 MinPts 能恰当描述数据密度**：对全局参数选择敏感
- **距离度量合适**：需反映数据的相似性结构
- **对异常值不敏感**：自然区分噪声点

---

## 适用场景

### K-Means
- **大规模数据集**：$O(N \cdot K \cdot d \cdot T)$ 复杂度，线性于样本量，支持 Mini-Batch 进一步加速
- **簇近似球形的数据**：如文档聚类（TF-IDF 经预处理后）、颜色量化、图像分割
- **快速原型验证**：计算轻量，易于理解和部署
- **作为其他算法的初始化**（如 Gaussian Mixture Models 的 EM 初始化）
- **使用 K-Means++ 初始化 + 多次重启时具有可靠的实际表现**

### DBSCAN
- **任意形状的簇**：可发现 S 形、环形、新月形等非凸簇
- **含噪声/异常值的真实数据**：自动标记噪声点
- **无需预知簇数**：完全数据驱动
- **空间/地理数据**：自然距离度量（经纬度、坐标）
- **异常检测**：噪声点本身即为异常候选

### 不适用

| 方法 | 场景 | 原因 |
|------|------|------|
| K-Means | **非球形簇**（如环形、螺旋形） | 质心模型无法描述 |
| K-Means | **密度/方差差异大的簇** | 质心偏向大簇/高方差簇 |
| K-Means | **高维数据**（$d > 50$） | 维数灾难使欧氏距离区分度下降 |
| K-Means | **类别不平衡严重** | 小簇可能被合并或为空 |
| K-Means | **含大量噪声/离群点** | 每个点都被强制分配，质心被污染 |
| K-Means | **期望概率性隶属度** | K-Means 给出硬分配 |
| DBSCAN | **密度高度变化的簇** | 单一 $\varepsilon$ 无法同时适应 |
| DBSCAN | **高维数据**（$d > 20$） | 维数灾难使密度定义失效 |
| DBSCAN | **簇间密度差异极大** | 需手动调参且对 $\varepsilon$ 极为敏感 |
| DBSCAN | **期望硬聚类数可控制** | 簇数由参数隐式决定 |
| DBSCAN | **大数据集**（$N > 10^6$） | 即使空间索引也不如 K-Means 可扩展 |

---

## 实现要点

### K-Means 关键注意事项

| 要点 | 说明 |
|------|------|
| **K-Means++ 初始化** | 必要！随机初始化可能导致极差局部最优，K-Means++ + 多次重启（`n_init=10`）是最佳实践 |
| **Elbow Method** | 绘制 WCSS ~ K 曲线，选取"肘部点"（elbow），拐点处为最优 $K$ |
| **Gap Statistic**（Tibshirani et al., 2001） | 比较真实数据与 null reference 分布的 WCSS 差异，选取使 Gap 最大的 $K$ |
| **Silhouette Analysis** | 同时计算轮廓系数辅助选 $K$，比 Elbow 更稳定 |
| **多次重启（n_init）** | 默认 10 次随机初始化，选取 inertia 最小的结果 |
| **收敛容差（tol）** | 通常 $10^{-4}$，质心变化低于 tol 时停止迭代 |
| **Mini-Batch K-Means** | 每次随机样本子集更新质心，适合 $N > 10^5$，速度提升数个数量级 |
| **特征缩放** | 标准化（Z-score）或 Min-Max 归一化，否则量纲大的特征主导距离计算 |
| **空簇处理** | 若某簇无点分配，通常重新随机初始化该质心或取最远点 |

### DBSCAN 关键注意事项

| 要点 | 说明 |
|------|------|
| **$\varepsilon$ 选择** | 使用 k-distance plot：对所有点计算到第 $k = \text{MinPts}$ 近邻的距离，排序后绘制，选取"elbow"处对应的距离作为 $\varepsilon$ |
| **MinPts 选择** | 经验法则：$\text{MinPts} \approx 2 \times \text{dimensions}$，一般 MinPts $\geq 3$，常用 5–20 |
| **距离度量** | 欧氏距离最常用；文本数据可用余弦距离；高维数据 $\ell_2$ 区分度差，可尝试 $\ell_1$ 或余弦 |
| **空间加速** | 使用 KD-Tree（低维）或 Ball Tree（中维），sklearn 默认自动选择 |
| **OPTICS**（Ankerst et al., 1999） | 层次化 DBSCAN，无需单一 $\varepsilon$，生成 reachability plot，从中提取任意密度的簇 |
| **HDBSCAN**（Campello et al., 2013） | DBSCAN 的现代变体，基于 Mutual Reachability Distance 构建层次树，对变密度集群鲁棒 |
| **边界点归属** | DBSCAN 将边界点分配给任一可达核心簇（非确定性的第一个访问顺序），HDBSCAN 基于概率软化 |
| **特征缩放** | 绝对必要！不同尺度的特征会完全破坏 $\varepsilon$ 的含义 |
| **距离矩阵** | 可预计算距离矩阵（`metric='precomputed'`）用于自定义距离 |

---

## 代码

### 完整 Python 实现（基于 NumPy / SciPy）

```python
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
from collections import deque


# ==============================================================================
# K-Means（从头实现）
# ==============================================================================

class KMeans:
    """K-Means 聚类 — Lloyd 算法 + K-Means++ 初始化

    Parameters
    ----------
    n_clusters : int
        簇的数量 K
    max_iter : int
        最大迭代次数
    n_init : int
        随机初始化次数（取 inertia 最小的结果）
    tol : float
        质心变化小于 tol 时视为收敛
    random_state : int or None
        随机种子
    """
    def __init__(self, n_clusters=8, max_iter=300, n_init=10,
                 tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _kmeans_plusplus(self, X):
        """K-Means++ 初始化（Arthur & Vassilvitskii, 2007）"""
        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # 随机选择第一个质心
        centers = [X[rng.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            # 计算每个点到最近已选质心的距离平方
            dists = cdist(X, np.array(centers), metric='sqeuclidean')
            min_dists = np.min(dists, axis=1)  # D(x)^2
            # 按概率加权采样
            probs = min_dists / min_dists.sum()
            idx = rng.choice(n_samples, p=probs)
            centers.append(X[idx])

        return np.array(centers)

    def _assign_labels(self, X):
        """分配步骤：每个点归入最近质心"""
        dists = cdist(X, self.cluster_centers_, metric='sqeuclidean')
        labels = np.argmin(dists, axis=1)
        inertia = np.sum(np.min(dists, axis=1))
        return labels, inertia

    def _update_centers(self, X, labels):
        """更新步骤：重新计算每个簇的质心"""
        new_centers = np.zeros_like(self.cluster_centers_)
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = X[mask].mean(axis=0)
            else:
                # 空簇：保留原质心或重新初始化
                new_centers[k] = self.cluster_centers_[k]
        return new_centers

    def _fit_once(self, X):
        """单次 K-Means 拟合"""
        self.cluster_centers_ = self._kmeans_plusplus(X)

        for i in range(self.max_iter):
            labels, inertia = self._assign_labels(X)
            new_centers = self._update_centers(X, labels)

            shift = np.linalg.norm(new_centers - self.cluster_centers_, axis=1).max()
            self.cluster_centers_ = new_centers

            if shift < self.tol:
                break

        labels, inertia = self._assign_labels(X)
        return labels, inertia, i + 1

    def fit(self, X):
        """拟合 K-Means 模型（多次初始化取最优）

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            训练数据

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        best_n_iter = 0

        for init in range(self.n_init):
            labels, inertia, n_iter = self._fit_once(X)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = self.cluster_centers_.copy()
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        """预测新样本的簇归属

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        dists = cdist(X, self.cluster_centers_, metric='sqeuclidean')
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        """拟合并预测"""
        self.fit(X)
        return self.labels_

    def silhouette_score(self, X):
        """计算全局轮廓系数（Rousseeuw, 1987）"""
        X = np.asarray(X, dtype=np.float64)
        labels = self.labels_
        if labels is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        n = len(X)
        if n < 2:
            return 0.0

        # 成对距离矩阵
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        scores = np.zeros(n)

        for i in range(n):
            same_cluster = labels == labels[i]
            same_cluster[i] = False  # 排除自身

            if same_cluster.sum() == 0:
                # 单点簇：轮廓系数定义为 0
                scores[i] = 0.0
                continue

            # 簇内平均距离 a(i)
            a_i = np.mean(dist_matrix[i, same_cluster])

            # 最近邻簇的平均距离 b(i)
            other_clusters = np.unique(labels[labels != labels[i]])
            if len(other_clusters) == 0:
                scores[i] = 0.0
                continue

            b_i = np.min([
                np.mean(dist_matrix[i, labels == c]) for c in other_clusters
            ])

            scores[i] = (b_i - a_i) / max(a_i, b_i)

        return np.mean(scores)

    def elbow_method(self, X, K_range=range(1, 11), ax=None):
        """Elbow Method：计算不同 K 的 WCSS

        Parameters
        ----------
        X : ndarray
        K_range : iterable of int
        ax : matplotlib Axes or None

        Returns
        -------
        inertias : list
        """
        inertias = []
        for K in K_range:
            km = KMeans(n_clusters=K, random_state=self.random_state)
            km.fit(X)
            inertias.append(km.inertia_)

        if ax is not None:
            ax.plot(list(K_range), inertias, 'bo-')
            ax.set_xlabel('K')
            ax.set_ylabel('Inertia (WCSS)')
            ax.set_title('Elbow Method for Optimal K')
            ax.grid(True, alpha=0.3)

        return inertias


# ==============================================================================
# DBSCAN（从头实现）
# ==============================================================================

class DBSCAN:
    """DBSCAN 聚类 — 基于密度的空间聚类（Ester et al., 1996）

    使用 KD-Tree 加速区域查询，支持任意距离度量。

    Parameters
    ----------
    eps : float
        邻域半径
    min_samples : int
        核心点的最小邻域样本数（包含自身）
    metric : str
        距离度量（'euclidean', 'manhattan', 'cosine', 'precomputed'）
    leaf_size : int
        KD-Tree / Ball Tree 的叶节点大小
    """
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', leaf_size=40):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.leaf_size = leaf_size
        self.labels_ = None
        self.components_ = None  # 核心样本索引
        self.core_sample_indices_ = None
        self.noise_ = None

    def _region_query(self, tree, point_idx):
        """查询点 point_idx 的 eps-邻域内所有点索引"""
        # KD-Tree 的 query_ball_point 返回 eps 半径内的所有点索引
        indices = tree.query_ball_point(
            tree.data[point_idx], r=self.eps, p=2 if self.metric == 'euclidean' else 1
        )
        return np.array(indices, dtype=int)

    def _expand_cluster(self, tree, labels, point_idx, cluster_id, core_flags):
        """BFS 扩展簇：从核心点出发，将所有密度可达点加入簇"""
        seeds = deque([point_idx])
        labels[point_idx] = cluster_id

        while seeds:
            current = seeds.popleft()

            if core_flags[current]:
                # 核心点：将其 eps-邻域内所有未处理点入队
                neighbors = self._region_query(tree, current)
                for nb_idx in neighbors:
                    if labels[nb_idx] == -1:  # 未访问
                        labels[nb_idx] = cluster_id
                        seeds.append(nb_idx)

    def fit(self, X):
        """拟合 DBSCAN 模型

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            训练数据。若 metric='precomputed'，则 X 为距离矩阵。

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        if self.metric == 'precomputed':
            # 预计算距离矩阵，不适合 KD-Tree
            n = X.shape[0]
            labels = -np.ones(n, dtype=int)
            # 判断核心点
            core_flags = np.zeros(n, dtype=bool)
            for i in range(n):
                neighbors = np.where(X[i] <= self.eps)[0]
                core_flags[i] = len(neighbors) >= self.min_samples

            cluster_id = 0
            for i in range(n):
                if labels[i] != -1:
                    continue
                if core_flags[i]:
                    # BFS
                    cluster_id += 1
                    labels[i] = cluster_id
                    queue = deque([i])
                    while queue:
                        c = queue.popleft()
                        if not core_flags[c]:
                            continue
                        nbs = np.where(X[c] <= self.eps)[0]
                        for nb in nbs:
                            if labels[nb] == -1:
                                labels[nb] = cluster_id
                                queue.append(nb)
        else:
            # 构建 KD-Tree
            if self.metric in ('euclidean', 'l2'):
                tree = KDTree(X, leafsize=self.leaf_size)
            else:
                # 其他度量 — 回退到 ball tree 或暴力搜索
                from sklearn.neighbors import BallTree
                tree = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)

            n = X.shape[0]
            labels = -np.ones(n, dtype=int)

            # 标记核心点
            core_flags = np.zeros(n, dtype=bool)
            for i in range(n):
                neighbors = tree.query_ball_point(X[i], r=self.eps)
                core_flags[i] = len(neighbors) >= self.min_samples

            # 扩展簇
            cluster_id = 0
            for i in range(n):
                if labels[i] != -1:
                    continue
                if core_flags[i]:
                    cluster_id += 1
                    self._expand_cluster(tree, labels, i, cluster_id, core_flags)
                # 非核心点保持 labels[i] = -1（噪声）

        self.labels_ = labels
        core_mask = core_flags & (labels != -1)
        self.core_sample_indices_ = np.where(core_mask)[0]
        self.components_ = X[self.core_sample_indices_]
        self.noise_ = np.where(labels == -1)[0]

        return self

    def fit_predict(self, X):
        """拟合并返回聚类标签"""
        self.fit(X)
        return self.labels_


# ==============================================================================
# 辅助函数
# ==============================================================================

def k_distance_plot(X, k=5, ax=None):
    """绘制 k-distance 图，用于估计 DBSCAN 的 eps 参数

    对所有点计算到第 k 近邻的距离，排序后绘图。
    选取曲线"elbow"处对应的距离作为 eps。

    Parameters
    ----------
    X : ndarray
    k : int
        近邻数（通常等于 MinPts）
    ax : matplotlib Axes or None
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # 第 k 近邻距离（排除自身，取索引 k）
    k_dist = np.sort(distances[:, k])[::-1]

    if ax is not None:
        ax.plot(np.arange(len(k_dist)), k_dist, 'b-')
        ax.set_xlabel('Points (sorted by distance)')
        ax.set_ylabel(f'{k}-th Nearest Neighbor Distance')
        ax.set_title('k-Distance Plot for DBSCAN eps Selection')
        ax.grid(True, alpha=0.3)


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_blobs, make_moons

    print("=" * 60)
    print("K-Means 示例")
    print("=" * 60)

    # 生成球形簇数据
    X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, n_features=2,
                                   cluster_std=1.0, random_state=42)

    # K-Means 聚类
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels_km = km.fit_predict(X_blobs)
    sil = km.silhouette_score(X_blobs)
    print(f"K-Means inertia: {km.inertia_:.2f}")
    print(f"K-Means silhouette: {sil:.4f}")

    # Elbow Method
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # K-Means 聚类结果
    ax1 = axes[0]
    ax1.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_km, cmap='viridis',
                s=10, alpha=0.7)
    ax1.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    ax1.set_title(f'K-Means (K=4, Silhouette={sil:.3f})')
    ax1.legend()

    # Elbow 图
    ax2 = axes[1]
    km.elbow_method(X_blobs, K_range=range(1, 11), ax=ax2)
    plt.tight_layout()
    plt.savefig('outputs/kmeans_demo.png', dpi=150)
    plt.show()
    print("K-Means demo figure saved.\n")

    print("=" * 60)
    print("DBSCAN 示例")
    print("=" * 60)

    # 生成月牙形数据（非球形簇，K-Means 会失败）
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

    # K-Means 对比（非球形簇）
    km2 = KMeans(n_clusters=2, random_state=42)
    labels_km2 = km2.fit_predict(X_moons)

    # DBSCAN
    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    labels_db = db.fit_predict(X_moons)
    n_clusters_db = len(np.unique(labels_db[labels_db != -1]))
    n_noise = np.sum(labels_db == -1)
    print(f"DBSCAN clusters: {n_clusters_db}, noise points: {n_noise}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 真实标签
    axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons,
                    cmap='coolwarm', s=10, alpha=0.7)
    axes[0].set_title('Ground Truth')

    # K-Means 结果（错误）
    axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_km2,
                    cmap='coolwarm', s=10, alpha=0.7)
    axes[1].scatter(km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1],
                    c='red', marker='x', s=200, linewidths=3)
    axes[1].set_title('K-Means (fail on non-spherical)')

    # DBSCAN 结果（正确）
    axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_db,
                    cmap='coolwarm', s=10, alpha=0.7)
    axes[2].set_title(f'DBSCAN ({n_clusters_db} clusters)')

    for ax in axes:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.tight_layout()
    plt.savefig('outputs/dbscan_vs_kmeans.png', dpi=150)
    plt.show()
    print("DBSCAN demo figure saved.\n")

    # k-distance 图
    fig, ax = plt.subplots(figsize=(6, 4))
    k_distance_plot(X_moons, k=5, ax=ax)
    plt.tight_layout()
    plt.savefig('outputs/k_distance_plot.png', dpi=150)
    plt.show()
    print("k-distance plot saved.")
```

### 基于 scikit-learn 的生产用法

```python
from sklearn.cluster import KMeans, DBSCAN, OPTICS, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
import numpy as np

# ====================
# 1. K-Means 标准用法
# ====================

# 特征缩放（必需）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 最优实践
km = KMeans(
    n_clusters=5,
    init='k-means++',       # K-Means++ 初始化
    n_init=10,              # 多次初始化取最优
    max_iter=300,
    tol=1e-4,
    random_state=42
)
labels = km.fit_predict(X_scaled)

# 评估
sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil:.4f}")
print(f"Inertia (WCSS): {km.inertia_:.2f}")
print(f"Centroids:\n{km.cluster_centers_}")

# 自动选 K：Elbow + Silhouette
from sklearn.cluster import KMeans as SKMeans

def select_k(X, K_range=range(2, 11)):
    results = []
    for k in K_range:
        km = SKMeans(n_clusters=k, init='k-means++', n_init=10,
                     random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        results.append((k, km.inertia_, sil))
    return results

# =============
# 2. DBSCAN
# =============

# 特征缩放（必需！）
X_scaled = StandardScaler().fit_transform(X)

# DBSCAN
db = DBSCAN(
    eps=0.3,
    min_samples=5,
    metric='euclidean',
    leaf_size=40
)
labels = db.fit_predict(X_scaled)
n_clusters = len(set(labels) - {-1})
n_noise = list(labels).count(-1)
print(f"Estimated clusters: {n_clusters}, noise: {n_noise}")

# 核心样本
print(f"Core samples: {len(db.core_sample_indices_)}")

# =============
# 3. OPTICS（层次化 DBSCAN，无需 eps）
# =============

optics = OPTICS(
    min_samples=5,
    xi=0.05,           # 簇边界检测的阈值
    min_cluster_size=0.1
)
labels_optics = optics.fit_predict(X_scaled)

# ===============
# 4. HDBSCAN（现代变体，推荐）
# ===============

# 安装：pip install hdbscan
clusterer = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
labels_hdb = clusterer.fit_predict(X_scaled)

# HDBSCAN 提供概率软分配
probabilities = clusterer.probabilities_

# ===================
# 5. Pipeline 示例
# ===================

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('cluster', KMeans(n_clusters=5, init='k-means++', random_state=42))
])
labels = pipeline.fit_predict(X)
```

### 关键参数速查

| 算法 | 参数 | 默认 | 说明 |
|------|------|------|------|
| K-Means | `n_clusters` | 8 | 簇数（必调） |
| | `init` | 'k-means++' | 初始化方法 |
| | `n_init` | 10 | 独立运行次数 |
| | `max_iter` | 300 | 最大迭代 |
| | `tol` | 1e-4 | 收敛容差 |
| | `algorithm` | 'lloyd' | 算法变体（lloyd / elkan） |
| Mini-Batch K-Means | `batch_size` | 1024 | 每批次样本量 |
| DBSCAN | `eps` | 0.5 | 邻域半径（关键参数） |
| | `min_samples` | 5 | 核心点最小邻居数 |
| | `metric` | 'euclidean' | 距离度量 |
| | `leaf_size` | 40 | 树结构叶节点大小 |
| OPTICS | `min_samples` | 5 | 同 DBSCAN |
| | `xi` | 0.05 | 簇边界检测阈值 |
| | `min_cluster_size` | 0.025 | 最小簇比例/大小 |
| HDBSCAN | `min_cluster_size` | 5 | 最小簇大小 |
| | `min_samples` | None | 同 DBSCAN min_samples |
| | `cluster_selection_epsilon` | 0.0 | 平切提取簇 |

---

## 参考文献

Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, 1027–1035.

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *Advances in Knowledge Discovery and Data Mining (PAKDD)*, 160–172.

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96*, 226–231.

MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.

Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *Journal of the Royal Statistical Society: Series B*, 63(2), 411–423.
