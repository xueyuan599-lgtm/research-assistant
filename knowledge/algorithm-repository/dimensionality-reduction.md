# PCA / t-SNE / UMAP 降维

- **来源**: Pearson, K. (1901). On Lines and Planes of Closest Fit to Systems of Points in Space. *Philosophical Magazine*, 2(11), 559–572.; Hotelling, H. (1933). Analysis of a Complex of Statistical Variables into Principal Components. *Journal of Educational Psychology*, 24(6), 417–441.; Van der Maaten, L. & Hinton, G. (2008). Visualizing Data Using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.; McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint* arXiv:1802.03426.
- **方法类别**: 机器学习 / 降维与可视化

## 数学设定

### PCA（主成分分析）

PCA 寻找一组正交投影方向，使投影后方差最大化。等价于最小化重构误差。

#### 基于 SVD 的定义

设中心化数据矩阵 $X \in \mathbb{R}^{n \times d}$（每列已减均值），其奇异值分解为：

$$
X = U \Sigma V^{\mathsf{T}}
$$

其中 $U \in \mathbb{R}^{n \times r}$ 为左奇异向量，$\Sigma \in \mathbb{R}^{r \times r}$ 为对角奇异值矩阵，$V \in \mathbb{R}^{d \times r}$ 为右奇异向量（即主成分方向），$r = \text{rank}(X)$。

主成分得分（低维表示）：

$$
T = XV = U\Sigma
$$

取前 $k$ 列即 $k$ 维投影。

#### 基于协方差矩阵特征分解

样本协方差矩阵：

$$
C = \frac{1}{n-1} X^{\mathsf{T}}X
$$

特征分解：

$$
C = V \Lambda V^{\mathsf{T}}, \quad \Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d), \; \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d
$$

特征向量 $V$ 即主成分方向（载荷/loadings），特征值 $\lambda_k$ 表示对应方向的方差大小。

#### 解释方差比

第 $k$ 个主成分解释的方差比例：

$$
\frac{\lambda_k}{\sum_{i=1}^{d} \lambda_i}
$$

前 $k$ 个主成分的累计解释方差比：

$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}
$$

#### Biplot

在 PC1–PC2 平面上同时展示样本散点（得分）和原始变量方向（载荷向量）。载荷向量 $v_j = (v_{1j}, v_{2j})$ 的长度反映该变量对 PC 的贡献，夹角反映变量间相关性。

#### Incremental PCA

适用于内存无法容纳完整 $X$ 的场景。每次读入一个 batch $X_b \in \mathbb{R}^{b \times d}$，增量更新均值 $\bar{x}$ 和协方差 $C$：

$$
\bar{x}_{\text{new}} = \frac{n\bar{x} + m\bar{x}_b}{n + m}
$$

$$
C_{\text{new}} = \frac{n}{n+m}C + \frac{m}{n+m}C_b + \frac{nm}{(n+m)^2}(\bar{x} - \bar{x}_b)(\bar{x} - \bar{x}_b)^{\mathsf{T}}
$$

---

### t-SNE（t-distributed Stochastic Neighbor Embedding）

t-SNE 将高维数据点间的相似性转换为概率分布，然后在低维空间中最小化两个分布间的 KL 散度。

#### 高维相似性（高斯核）

给定高维点 $x_i$，其余点 $x_j$ 的条件概率：

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

$\sigma_i$ 由用户指定的 **perplexity** 参数决定。

#### 困惑度（Perplexity）

$$
\text{Perp}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_{j} p_{j|i} \log_2 p_{j|i}
$$

$\sigma_i$ 通过二分搜索确定，使 $\text{Perp}(P_i)$ 等于用户设定的 perplexity 值（典型范围 5–50）。

#### 对称联合分布

为简化梯度，定义对称化的联合分布：

$$
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
$$

#### 低维相似性（Student-t 分布，1 自由度）

低维嵌入点 $y_i, y_j$ 之间的相似性用 t-分布（1 自由度，即 Cauchy 分布）度量：

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

t-分布的长尾特性缓解了"拥挤问题"（高维空间中中等距离的点在低维中被拉开）。

#### KL 散度（损失函数）

$$
\text{KL}(P \| Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

KL 散度不对称：$p_{ij}$ 大而 $q_{ij}$ 小则惩罚大（局部结构保持），$p_{ij}$ 小而 $q_{ij}$ 大则惩罚小（允许全局结构松散）。

#### 梯度下降

梯度具有明确的物理意义（$n \times 2$ 个点间的弹簧力）：

$$
\frac{\partial \text{KL}}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}
$$

优化策略：
- **Momentum**: 加速收敛并使用动量的梯度更新
- **Early exaggeration**: 前若干轮将 $p_{ij}$ 乘以常数（如 4–12），强化聚类

---

### UMAP（Uniform Manifold Approximation and Projection）

UMAP 基于三个拓扑学假设：数据均匀采样于低维流形、流形局部连接、全局最优嵌入。

#### 模糊单纯集构建

1. 对每个点 $x_i$，找到 $k$ 个最近邻
2. 定义局部度量：$d_i = \text{distance to } k\text{-th nearest neighbor}$
3. 构造加权 $k$-近邻图，边权重为：

$$
w_{ij} = \exp\left(-\frac{d(x_i, x_j) - \rho_i}{\sigma_i}\right)
$$

其中 $\rho_i$ 是 $x_i$ 到其最近邻的距离，确保局部连通性。

#### 低维嵌入

低维空间中，吸引力（弹簧力，同类点靠近）与排斥力（静电斥力，异类点分开）平衡：

**吸引力**（在 $k$-近邻边上作用）：

$$
F_{\text{attr}}(y_i, y_j) = -w_{ij} \cdot 2ab\|y_i - y_j\|^{2(b-1)} \cdot (1 + a\|y_i - y_j\|^{2b})^{-1} \cdot (y_i - y_j)
$$

**排斥力**（在所有非近邻对上作用，负采样近似）：

$$
F_{\text{rep}}(y_i, y_j) = (1 - w_{ij}) \cdot 2b(\epsilon + \|y_i - y_j\|^2)^{-1} \cdot (y_i - y_j)
$$

其中 $a, b$ 为拟合 Student-t 核的参数。

#### 交叉熵损失

UMAP 优化的是两个模糊集的交叉熵：

$$
\text{CE}(P, Q) = \sum_{i}\sum_{j} \left[ p_{ij} \log\frac{p_{ij}}{q_{ij}} + (1 - p_{ij})\log\frac{1 - p_{ij}}{1 - q_{ij}} \right]
$$

右项 $(1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}$ 使 UMAP 比 t-SNE 更好地保留全局结构（因为 $p_{ij} \approx 0$ 的点对也会受惩罚）。

#### 与 t-SNE 的关键区别

| 特性 | t-SNE | UMAP |
|------|-------|------|
| 高维图 | 全连接高斯核 | $k$-NN 截断图（稀疏） |
| 低维核 | Student-t（1 df） | 可调参数 $a,b$ |
| 损失 | KL 散度（不对称） | 交叉熵（对称） |
| 初始化 | 随机正态 | 谱嵌入（Laplacian eigenmap） |
| 优化 | 梯度下降（批） | 随机梯度下降（负采样） |
| 速度 | $O(n^2)$ | $O(nkd)$ |
| 全局结构 | 弱 | 较强 |

---

### 比较框架

| 维度 | PCA | t-SNE | UMAP |
|------|-----|-------|------|
| 类型 | 线性 | 非线性 | 非线性 |
| 保留结构 | 全局方差 | 局部邻域 | 局部 + 部分全局 |
| 输出确定性 | 确定 | 随机（多解） | 随机但可设 seed |
| 超参数 | $k$（维数） | perplexity, lr, iter | $k$-NN, min_dist, $k$ |
| 计算复杂度 | $O(n \cdot d^2)$ | $O(n^2)$ | $O(n \cdot k \cdot d)$ |
| 内存复杂度 | $O(d^2)$ | $O(n^2)$ | $O(nk)$ |
| 可解释性 | 高（载荷） | 无 | 无 |
| 适用数据量 | 任意 | $< 10^5$ | $> 10^5$ 也可 |

## 关键假设

### PCA
- **线性**：数据的主要变化沿着正交直线方向
- **均值居中**：$X$ 列中心化是必要条件（否则第一主成分被均值偏移主导）
- **方差 = 信息**：方差最大的方向就是最重要的方向（未利用标注信息）
- **特征同尺度**：变量需在同一量纲下比较（标准化的前提）

### t-SNE
- **局部邻域结构有意义**：高维空间中的近邻关系能反映真实类别结构
- **数据嵌入于低维流形**：高维数据点可以映射到 2D/3D 保持局部相似性
- **Perplexity 合理选择**：对邻域大小的猜测需匹配数据密度

### UMAP
- **数据位于或靠近低维流形**：流形假设是 UMAP 拓扑框架的基础
- **流形上均匀分布**：数据点在流形上近似均匀采样
- **局部连接性**：每个点的 $k$-NN 能捕捉局部流形结构

## 适用场景

### PCA
- **多重共线性诊断**：特征值接近 0 的主成分对应冗余变量组合
- **特征提取**：将 $d$ 维压缩到 $k$ 维去噪特征
- **可视化**：取前 2–3 个主成分做散点图
- **噪声过滤**：丢弃后几个方差最小的主成分（相当于对 $X$ 低秩近似）
- **其他模型预处理**：降维后加快后续 KNN / 回归 / 分类速度

### t-SNE
- **高维聚类可视化**：探索性数据分析中的聚类结构发现
- **单次 2D/3D 视图**：适用于论文中展示样本分布模式
- **数据质量评估**：查看异常点、batch effect、数据划分是否均衡

### UMAP
- **可视化（优于 t-SNE）**：更快的计算速度 + 更好的全局结构保留
- **通用非参降维**：从任意高维数据生成低维表征
- **下游任务预处理**：降维后作为分类/聚类/异常检测的特征输入
- **大规模数据降维**：$n > 10^5$ 时首选

### 不适用

**PCA 不适用：**
- **非线性流形**（如 Swiss Roll、S-curve）：PCA 强行线性投影会丢失结构 — 用 UMAP / t-SNE / Autoencoder
- **需保留原始特征可解释性**：主成分是所有特征的线性组合 — 用 **Sparse PCA**（稀疏载荷）或特征选择方法
- **异常值较多**：PCA 对异常值敏感 — 用 **Robust PCA**

**t-SNE 不适用：**
- **需要保留全局结构**：t-SNE 只保证局部邻域 — 用 UMAP 或 PCA
- **嵌入结果用于下游模型**：t-SNE 嵌入不稳定（不同运行差异大、数据新增需重新计算）— 用 UMAP 或 Parametric t-SNE
- **超大型数据集（$n > 10^5$）**：$O(n^2)$ 复杂度不可承受 — 用 UMAP 或 Fit-SNE（FFT-accelerated）
- **需要可重复的定量分析**：t-SNE 对随机种子、perplexity 高度敏感

**UMAP 不适用：**
- **需精确保持原始邻域关系**：UMAP 的近似图会损失部分精确邻域信息 — 用 PCA 或多维缩放（MDS）
- **对可解释性有要求**：UMAP 嵌入无天然语义 — 用 PCA（载荷可解释）
- **非常小的数据集（$n < 50$）**：UMAP 的 $k$-NN 图不稳定 — 用 PCA 或 t-SNE

## 实现要点

### PCA

- **标准化**：使用相关矩阵 PCA 还是协方差矩阵 PCA — 若变量量纲不同，必须先标准化（`StandardScaler`），相当于做相关矩阵 PCA
- **Scree plot**：绘制特征值下降曲线 + 累计解释方差比，选取"肘部"或累计 > 80% 的前 $k$ 个成分
- **载荷解释**：载荷向量 $v_j$ 的元素符号和大小表示原始变量对 PC 的贡献方向和强度
- **Sparse PCA**：在 PCA 目标上加 $\ell_1$ 惩罚使载荷稀疏，提升可解释性

### t-SNE

| 参数 | 范围 | 说明 |
|------|------|------|
| perplexity | [5, 50] | 控制邻域大小；小 perplexity 强调局部，大 perplexity 考虑全局 |
| early_exaggeration | [4, 12] | 前期放大聚类的吸引力，使簇更分明 |
| learning_rate | [10, 1000] | 建议 $n/12$ 或 $n/10$ 为起点 |
| n_iter | [250, 1000] | 迭代次数，太少未收敛 |

- **Perplexity 调优**：确实没有"正确"值 — 在 5–50 范围内多跑几个值，观察稳定结构
- **多次运行**：t-SNE 每次结果不同；建议多次运行选择最稳定的可视化结果（或报告运行间的变异性）
- **不是聚类方法**：t-SNE 不学习聚类边界；簇的外观完全取决于 perplexity 及随机种子的交互
- **随机种子**：固定 `random_state` 确保可重复

### UMAP

| 参数 | 范围 | 默认 | 说明 |
|------|------|------|------|
| n_neighbors | [2, 200] | 15 | 大值保留全局结构，小值强调局部 |
| min_dist | [0.0, 0.99] | 0.1 | 控制低维点的最小距离，小则簇紧密，大则分散 |
| n_components | [2, ...] | 2 | 嵌入维度 |
| metric | — | 'euclidean' | 距离度量，可自定义 |

- **n_neighbors 权衡**：增大 $n\_neighbors$ 使 UMAP 更关注全局结构（代价是计算更慢、局部细节平滑）
- **min_dist 控制聚类紧凑度**：接近 0 时簇非常紧密，适合有清晰分离的数据；大值更适合连续流形或过度聚类的数据
- **Reproducible embedding**：设置 `random_state` 得到可复现结果

### 通用要点

- **特征缩放前置**：无论使用 PCA / t-SNE / UMAP，都应在降维前对特征做标准化（`StandardScaler`）
- **先 PCA 再 t-SNE/UMAP**：对超高维数据（$d > 1000$），先用 PCA 压缩到 50–100 维再 t-SNE/UMAP，可去除噪声并加速
- **降维不是特征选择**：PCA / t-SNE / UMAP 是特征提取（变换），不是特征子集选择

## 代码

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, load_digits, fetch_openml
from sklearn.decomposition import PCA as SklearnPCA


# =============================================================================
# 从零实现 PCA（基于 SVD）
# =============================================================================

class PCA:
    """主成分分析 — 基于 SVD 的从零实现

    Parameters
    ----------
    n_components : int
        保留的主成分数

    Attributes
    ----------
    components_ : ndarray, shape (n_components, n_features)
        主成分方向（载荷向量），每行是一个 PC
    explained_variance_ratio_ : ndarray, shape (n_components,)
        各主成分解释方差比例
    cumulative_variance_ratio_ : ndarray, shape (n_components,)
        累计解释方差比例
    mean_ : ndarray, shape (n_features,)
        训练集的均值向量
    n_components_ : int
        实际保留的成分数
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """用 SVD 拟合 PCA"""
        n, d = X.shape
        # Step 1: 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: SVD
        # 对高宽矩阵（n >> d）用完整 SVD 浪费，但这里保持教学清晰
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Vt 的行是右奇异向量（即主成分方向）
        # S 是奇异值（降序），方差 = S^2 / (n-1)
        var = S ** 2 / (n - 1)

        # Step 3: 确定保留成分数
        max_components = min(n, d)
        if self.n_components is None:
            self.n_components_ = max_components
        else:
            self.n_components_ = min(self.n_components, max_components)

        # Step 4: 存储结果
        self.components_ = Vt[:self.n_components_]
        self.explained_variance_ratio_ = var[:self.n_components_] / np.sum(var)
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        self.singular_values_ = S[:self.n_components_]
        return self

    def transform(self, X):
        """投影到主成分空间"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """拟合并投影"""
        self.fit(X)
        return self.transform(X)

    def scree_plot(self, figsize=(8, 4)):
        """绘制 Scree plot + 累计解释方差

        Parameters
        ----------
        figsize : tuple
            图像尺寸
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        k = len(self.explained_variance_ratio_)

        # 左图：各成分解释方差条形图
        axes[0].bar(range(1, k + 1), self.explained_variance_ratio_,
                    color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Scree Plot')
        axes[0].set_xticks(range(1, k + 1))

        # 右图：累计解释方差
        axes[1].plot(range(1, k + 1), self.cumulative_variance_ratio_,
                     'o-', color='crimson', markersize=6)
        axes[1].axhline(y=0.8, color='grey', linestyle='--', alpha=0.7,
                        label='80% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance')
        axes[1].set_xticks(range(1, k + 1))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def biplot(self, X, feature_names=None, sample_labels=None,
               scale_loadings=1.5, figsize=(8, 8), colors=None):
        """PC1 vs PC2 双标图

        Parameters
        ----------
        X : ndarray
            原始数据（用于投影）
        feature_names : list of str, optional
            特征名称
        sample_labels : array-like, optional
            样本标签（用于着色）
        scale_loadings : float
            载荷箭头放大倍数
        figsize : tuple
            图像尺寸
        colors : array-like, optional
            样本颜色
        """
        if self.n_components_ < 2:
            raise ValueError("Biplot requires n_components >= 2")

        # 投影到 PC1, PC2
        scores = self.transform(X)

        # 取前两个成分
        pc1, pc2 = scores[:, 0], scores[:, 1]

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制样本
        if sample_labels is not None:
            unique_labels = np.unique(sample_labels)
            for label in unique_labels:
                mask = sample_labels == label
                ax.scatter(pc1[mask], pc2[mask], s=30, alpha=0.7,
                           label=str(label))
            ax.legend(fontsize=9)
        elif colors is not None:
            sc = ax.scatter(pc1, pc2, c=colors, s=30, alpha=0.7, cmap='viridis')
            plt.colorbar(sc, ax=ax)
        else:
            ax.scatter(pc1, pc2, s=30, alpha=0.7, color='steelblue')

        # 绘制载荷向量
        for i in range(self.components_.shape[1]):
            load_i = self.components_[:2, i]  # (PC1, PC2) 上的载荷
            dx = load_i[0] * scale_loadings
            dy = load_i[1] * scale_loadings

            arrow = FancyArrowPatch(
                (0, 0), (dx, dy),
                arrowstyle='-|>', mutation_scale=20,
                color='darkred', alpha=0.8, linewidth=1.5
            )
            ax.add_patch(arrow)

            label = feature_names[i] if feature_names else f'feat{i}'
            ax.text(dx * 1.1, dy * 1.1, label, fontsize=10,
                    color='darkred', ha='center', va='center')

        # 中心十字 + 等轴比例
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')

        # 等轴比例以保证载荷角度真实
        max_val = max(np.max(np.abs(pc1)), np.max(np.abs(pc2)))
        ax.set_xlim(-max_val * 1.3, max_val * 1.3)
        ax.set_ylim(-max_val * 1.3, max_val * 1.3)
        ax.set_aspect('equal')

        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Biplot: PC1 vs PC2')

        return fig


# =============================================================================
# t-SNE 与 UMAP API 使用说明
# =============================================================================
#
# t-SNE 推荐使用 openTSNE（pip install opentsne），而非 sklearn 的版本：
#   from openTSNE import TSNE
#   tsne = TSNE(perplexity=30, random_state=42)
#   Z_tsne = tsne.fit(X_scaled)
#
# sklearn 的内置 t-SNE 也可用，但较慢且不支持 out-of-core：
#   from sklearn.manifold import TSNE
#   Z_tsne = TSNE(perplexity=30, random_state=42).fit_transform(X_scaled)
#
# UMAP 使用 umap-learn 包（pip install umap-learn）：
#   import umap
#   reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
#   Z_umap = reducer.fit_transform(X_scaled)


def compare_embeddings(X, y=None, perplexity=30, n_neighbors=15,
                       min_dist=0.1, random_state=42, figsize=(16, 5)):
    """PCA vs t-SNE vs UMAP 降维对比图

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        输入数据（建议已标准化）
    y : array-like, optional
        标签（用于着色）
    perplexity : int
        t-SNE perplexity
    n_neighbors : int
        UMAP n_neighbors
    min_dist : float
        UMAP min_dist
    random_state : int
        随机种子
    figsize : tuple
        图像尺寸

    Returns
    -------
    dict of embeddings: {'pca': Z_pca, 'tsne': Z_tsne, 'umap': Z_umap}
    """
    from sklearn.manifold import TSNE
    import umap

    # PCA
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)

    # t-SNE
    tsne = TSNE(perplexity=perplexity, random_state=random_state,
                learning_rate='auto', init='pca')
    Z_tsne = tsne.fit_transform(X)

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=random_state
    )
    Z_umap = reducer.fit_transform(X)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    titles = [
        f'PCA (pca)',
        f't-SNE (perp={perplexity})',
        f'UMAP (k={n_neighbors}, d={min_dist})'
    ]
    embeddings = [Z_pca, Z_tsne, Z_umap]
    results = {'pca': Z_pca, 'tsne': Z_tsne, 'umap': Z_umap}

    for ax, Z, title in zip(axes, embeddings, titles):
        if y is not None:
            unique = np.unique(y)
            for label in unique:
                mask = y == label
                ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.7,
                           label=str(label))
            ax.legend(fontsize=7, loc='best', markerscale=2)
        else:
            ax.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7, color='steelblue')
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout()
    return results


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # ---------- 1. 合成数据演示 ----------
    print("=== 1. 合成数据：PCA 从零实现 vs sklearn 对比 ===")
    X_syn, y_syn = make_classification(
        n_samples=500, n_features=30, n_informative=10,
        n_redundant=5, n_clusters_per_class=2, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_syn)

    # 自实现 PCA
    pca_custom = PCA(n_components=10)
    pca_custom.fit(X_scaled)
    print(f"Custom PCA — Explained variance ratio (first 5): "
          f"{pca_custom.explained_variance_ratio_[:5].round(4)}")
    print(f"Custom PCA — Cumulative (first 5): "
          f"{pca_custom.cumulative_variance_ratio_[:5].round(4)}")

    # Scree plot
    pca_custom.scree_plot()
    plt.savefig("outputs/pca_scree_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Scree plot saved -> outputs/pca_scree_plot.png")

    # Biplot
    pca_2 = PCA(n_components=2).fit(X_scaled)
    pca_2.biplot(X_scaled, sample_labels=y_syn)
    plt.savefig("outputs/pca_biplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Biplot saved -> outputs/pca_biplot.png")

    # sklearn 对比
    sklearn_pca = SklearnPCA(n_components=10)
    sklearn_pca.fit(X_scaled)
    comp_match = np.allclose(
        np.abs(pca_custom.components_),
        np.abs(sklearn_pca.components_),
        atol=1e-10
    )
    print(f"Custom vs sklearn components match: {comp_match}")

    # ---------- 2. Digits 数据集降维对比 ----------
    print("\n=== 2. Digits 数据集：PCA vs t-SNE vs UMAP ===")
    digits = load_digits()
    X_digits = StandardScaler().fit_transform(digits.data)
    y_digits = digits.target

    try:
        # 注意：umap 和 opentsne 可能需要安装
        results = compare_embeddings(
            X_digits, y=y_digits, perplexity=30,
            n_neighbors=15, min_dist=0.1
        )
        plt.savefig("outputs/dr_comparison_digits.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("Comparison plot saved -> outputs/dr_comparison_digits.png")
        print(f"Embedding shapes: PCA {results['pca'].shape}, "
              f"t-SNE {results['tsne'].shape}, "
              f"UMAP {results['umap'].shape}")
    except ImportError as e:
        print(f"Skip comparison (install openTSNE & umap-learn for full demo): {e}")

    # ---------- 3. PCA 的 Incremental 模式说明 ----------
    print("\n=== 3. Incremental PCA (API 示意) ===")
    print("""
    # 大数据场景：使用 sklearn 的 IncrementalPCA
    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=10)
    for batch in np.array_split(X_scaled, 10):
        ipca.partial_fit(batch)      # 增量更新
    Z_ipca = ipca.transform(X_scaled)  # 最终投影
    print(f"Incremental PCA done, shape: {Z_ipca.shape}")
    """)
```

### 基于 sklearn / openTSNE / umap-learn 的生产用法

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# ---------- 数据准备 ----------
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

# ---------- PCA ----------
pca = PCA(n_components=0.9)    # 保留 90% 方差
X_pca = pca.fit_transform(X)
print(f"PCA reduced {X.shape[1]} -> {X_pca.shape[1]} dims "
      f"(explained: {pca.explained_variance_ratio_.sum():.2%})")

# ---------- t-SNE（内置 sklearn，小数据可用）----------
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X)

# ---------- UMAP ----------
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

# ---------- 并排可视化 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
titles = ['PCA (first 2 PCs)', 't-SNE (perp=30)', 'UMAP (k=15, d=0.1)']
embeddings = [X_pca[:, :2], X_tsne, X_umap]

for ax, Z, title in zip(axes, embeddings, titles):
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10',
                         s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
plt.colorbar(scatter, ax=axes, shrink=0.6)
plt.tight_layout()
plt.savefig("outputs/dr_production_comparison.png", dpi=150)
plt.show()
```

## 参考文献

- Pearson, K. (1901). On Lines and Planes of Closest Fit to Systems of Points in Space. *Philosophical Magazine*, 2(11), 559–572.
- Hotelling, H. (1933). Analysis of a Complex of Statistical Variables into Principal Components. *Journal of Educational Psychology*, 24(6), 417–441.
- Van der Maaten, L. & Hinton, G. (2008). Visualizing Data Using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
- McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint* arXiv:1802.03426.
- Tipping, M. E. & Bishop, C. M. (1999). Probabilistic Principal Component Analysis. *Journal of the Royal Statistical Society: Series B*, 61(3), 611–622.
