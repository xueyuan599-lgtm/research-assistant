# XGBoost / LightGBM — 梯度提升树

- **来源**: Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- **DOI**: 10.1145/2939672.2939785
- **来源**: Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- **DOI**: 10.5555/3294996.3295074
- **方法类别**: 统计建模 / 集成学习

## 数学设定

### 梯度提升框架（Generalized Gradient Boosting）

梯度提升是一个**前向分阶段加法模型**（Forward Stagewise Additive Modeling）。在第 $m$ 轮迭代中，当前模型 $F_{m-1}(x)$ 通过添加一棵新树 $f_m(x)$ 来更新：

$$
F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x)
$$

其中 $\eta$ 为学习率（shrinkage）。最终的集成模型为 $M$ 棵树的加权和：

$$
\hat{y}_i = F_M(x_i) = \sum_{m=1}^{M} \eta \cdot f_m(x_i)
$$

### XGBoost 目标函数

XGBoost 在传统梯度提升的损失函数基础上加入了**显式正则化项**，构成了带正则的目标函数：

$$
\text{Obj} = \sum_{i=1}^{N} L(y_i, \hat{y}_i) + \sum_{k=1}^{M} \Omega(f_k)
$$

其中正则项 $\Omega(f)$ 对树的复杂度进行惩罚：

$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2 + \alpha \|w\|_1
$$

- $T$：树的叶子节点数量（$\gamma$ 控制叶子数惩罚）
- $w \in \mathbb{R}^T$：叶子权重（分数）向量
- $\lambda$：L2 正则化系数（Ridge）
- $\alpha$：L1 正则化系数（Lasso）

### 二阶近似（Second-Order Approximation）

XGBoost 的核心贡献之一是对损失函数进行**二阶泰勒展开**，比传统一阶 GBDT 收敛更快、精度更高：

$$
\text{Obj}^{(m)} \approx \sum_{i=1}^{N} \left[ L(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(x_i) + \frac{1}{2} h_i f_m(x_i)^2 \right] + \Omega(f_m)
$$

其中一阶和二阶梯度统计量分别为：

$$
g_i = \frac{\partial L(y_i, \hat{y})}{\partial \hat{y}} \bigg|_{\hat{y}=\hat{y}^{(m-1)}}
\qquad
h_i = \frac{\partial^2 L(y_i, \hat{y})}{\partial \hat{y}^2} \bigg|_{\hat{y}=\hat{y}^{(m-1)}}
$$

移除常数项后，第 $m$ 轮的优化目标简化为：

$$
\tilde{\text{Obj}}^{(m)} = \sum_{i=1}^{N} \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m(x_i)^2 \right] + \Omega(f_m)
$$

### 叶子权重求解

将样本划分到 $T$ 个叶子节点上，定义 $I_j = \{i \mid q(x_i) = j\}$ 为落入叶子 $j$ 的样本集。令 $w_j$ 为叶子 $j$ 的分数，目标函数可逐叶展开：

$$
\tilde{\text{Obj}}^{(m)} = \sum_{j=1}^{T} \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda \right) w_j^2 \right] + \gamma T
$$

记 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$，则目标为各叶子的独立二次型之和。对 $w_j$ 求导得**最优叶子权重**：

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

代入最优权重得**最小目标值**（结构分数）：

$$
\tilde{\text{Obj}}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T
$$

### 分裂增益公式（Split Finding）

对于某个候选分裂点，将叶子 $j$ 分裂为左子 $L$ 和右子 $R$，分裂收益为：

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

- 第一项：左子结构分数
- 第二项：右子结构分数
- 第三项：不分裂时的结构分数（即父节点分数）
- $\gamma$：分裂惩罚——只有当 Gain $> 0$ 时分裂才有意义

### 剪枝策略

- **Pre-stopping**：如果最优分裂的 Gain $\leq 0$，则停止生长（叶节点保持为父节点权重）
- **Post-pruning**（深度优先）：先生长到最大深度，然后自底向上剪枝——如果子节点分裂带来的 Gain $\leq \gamma$ 则回退

实际上两者等价：XGBoost 使用 `max_depth` 限制树深，同时用 $\gamma$ 控制分裂阈值。

### 列采样与行采样

- **列采样（Column Subsampling）**：$\text{colsample}_\text{bytree}$ 每棵树选特征子集；$\text{colsample}_\text{bynode}$ 每节点选特征子集
- **行采样（Row Subsampling / Subsample）**：每轮迭代随机采样一定比例的样本

两种采样均能降低树间相关性，提升泛化能力。

### LightGBM 核心创新

**GOSS（Gradient-based One-Side Sampling）**：
- 保留梯度大的样本（信息量大），随机采样梯度小的样本
- 在保持精度的同时大幅降低计算量

**EFB（Exclusive Feature Bundling）**：
- 将互斥特征（几乎不同时取非零值）捆绑为单个特征
- 降低特征维度，加速直方图构建

**Leaf-wise（Best-first）树生长**：
- XGBoost 使用 Level-wise（按层生长），LightGBM 使用 Leaf-wise（每次选增益最大的叶子分裂）
- Leaf-wise 在相同迭代次数下损失更低，但需要限制 `num_leaves` 防止过拟合

**直方图算法（Histogram-based）**：
- 将连续特征离散化为 $k$ 个桶，用直方图统计代替预排序
- 内存占用从 $O(N \times \text{features})$ 降到 $O(k \times \text{features})$

## 关键假设

- **损失函数二阶可导**：XGBoost 的二阶近似要求 $L(y, \hat{y})$ 对 $\hat{y}$ 至少二阶可导。常见损失（MSE、LogLoss、Poisson）均满足
- **加法可加性**：树的预测是加性组合，假设各轮树之间相互独立除学习率外无交互机制
- **特征含预测信号**：每棵树能从残差中学习到有效模式
- **样本独立性**：loss 按样本求和，假设样本间独立
- **特征编码**：数值特征无需标准化，但需要合理处理缺失值。LightGBM 原生支持类别特征（`categorical_feature` 参数），无需 One-Hot 编码；XGBoost 需要预编码
- **不假设线性关系**：树模型自动捕捉非线性和交互效应

## 适用场景

- **结构化/表格数据（最佳模型之一）**：在 Kaggle、KDD Cup 等竞赛中，XGBoost/LightGBM 长期占据表格数据的 SOTA 地位
- **分类与回归**：二分类、多分类、连续值回归
- **排序学习（Learning to Rank）**：XGBoost 原生支持 `rank:ndcg`、`rank:map` 等排序目标
- **不平衡分类**：可通过 `scale_pos_weight`、`max_delta_step` 参数调整
- **缺失值自动处理**：XGBoost 自动学习缺失值的最佳分裂方向（默认路径 + 稀疏感知）
- **特征重要性筛选**：Gain / Cover / Frequency 三种重要性指标
- **大规模数据**：LightGBM 在百万级以上样本中训练速度显著快于 XGBoost

### 不适用

- **超高维稀疏数据**（如文本 BoW/TF-IDF）：线性模型（Logistic Regression 带 L1 正则）通常更优
- **图像/文本/音频数据**：深度学习（CNN、Transformer）远优于树模型
- **实时推理在资源受限设备**：单次推理需遍历多棵树，延迟高于线性模型或小型神经网络
- **需要完整线性可解释性**：虽可提取特征重要性，但不如 Logistic Regression 那样有清晰的系数解读
- **小样本数据（$N < 1000$）**：简单模型（正则化线性回归、朴素贝叶斯）可能更好且不易过拟合
- **在线/流式学习**：XGBoost/LightGBM 是批量训练模型，不支持增量更新

## 实现要点

### 关键超参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| $n\_estimators$ | [100, 10000] | 100 | 迭代轮数，越大越好但需配合 early stopping |
| $learning\_rate\;(\eta)$ | [0.001, 0.3] | 0.3 | 步长收缩，小学习率需要更多树，精度更高 |
| $max\_depth$ | [3, 15] | 6 | 树深度（XGBoost Level-wise），过深易过拟合 |
| $min\_child\_weight$ | [1, 50] | 1 | 叶子节点最小 Hessian 和，越大越保守 |
| $subsample$ | [0.5, 1.0] | 1.0 | 行采样比例 |
| $colsample\_bytree$ | [0.3, 1.0] | 1.0 | 每棵树特征采样比例 |
| $colsample\_bynode$ | [0.3, 1.0] | 1.0 | 每节点特征采样比例 |
| $\gamma$ | [0, 10] | 0 | 分裂最小损失减少量，越大树越保守 |
| $\lambda\;(reg\_lambda)$ | [0, 10] | 1 | L2 正则化权重 |
| $\alpha\;(reg\_alpha)$ | [0, 10] | 0 | L1 正则化权重 |

**LightGBM 特有参数**：

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| $num\_leaves$ | [16, 256] | 31 | 叶子数，控制树复杂度（替代 max\_depth） |
| $min\_data\_in\_leaf$ | [10, 1000] | 20 | 叶子最少样本数，防过拟合 |
| $feature\_fraction$ | [0.3, 1.0] | 1.0 | 每棵树特征采样（类似 colsample\_bytree） |
| $bagging\_fraction$ | [0.5, 1.0] | 1.0 | 行采样比例 |
| $bagging\_freq$ | [1, 10] | 0 | 行采样频率 |
| $lambda\_l1$ / $lambda\_l2$ | [0, 10] | 0 | L1/L2 正则化 |
| $min\_gain\_to\_split$ | [0, 15] | 0 | 分裂最小增益（类似 gamma） |
| $max\_bin$ | [64, 1024] | 255 | 直方图桶数 |

### 调优经验

1. **学习率与迭代轮数的权衡**：$\eta$ 越小，需要的 $n\_estimators$ 越多，精度通常越高但训练更慢。推荐初设 $\eta = 0.1$ + 早停，然后逐步降低
2. **先控过拟合**：$max\_depth$（或 $num\_leaves$）是控制复杂度的首要参数。LightGBM 中 $num\_leaves$ 通常设为 $2^{max\_depth}$ 左右
3. **$\gamma$ 和 $min\_child\_weight$**：两者都起正则化作用，$\gamma$ 惩罚分裂次数，$min\_child\_weight$ 防止小样本叶子
4. **采样参数**：$subsample + colsample\_bytree < 1.0$ 可显著降低过拟合，同时加速训练
5. **早停（Early Stopping）**：设置验证集和 $eval\_metric$，当验证集指标在 $patience$ 轮内无提升时停止训练
6. **类别特征**：LightGBM 原生支持类别特征，指定 `categorical_feature` 即可；XGBoost 需手动 One-Hot 编码或 Label Encoding
7. **缺失值处理**：XGBoost 自动学习缺失方向（默认分到增益最大的子树）；LightGBM 默认忽略缺失值
8. **GPU 加速**：XGBoost 和 LightGBM 均支持 GPU 训练，在大数据上加速明显

### 常用目标函数

| 目标 | XGBoost 参数 | 说明 |
|------|-------------|------|
| 回归（MSE） | `reg:squarederror` | 均方误差 |
| 回归（MAE） | `reg:absoluteerror` | 平均绝对误差 |
| 二分类 | `binary:logistic` | 输出概率 |
| 多分类 | `multi:softmax` / `multi:softprob` | 输出类别 / 概率 |
| 排序 | `rank:ndcg` / `rank:map` | Learning to Rank |
| 生存分析 | `survival:cox` | Cox 比例风险 |

### 特征重要性类型

- **Gain**：该特征在分裂中带来的平均增益（推荐，最可靠）
- **Cover**：该特征覆盖的样本数（受影响的样本量）
- **Frequency**：该特征被用于分裂的次数（可能有偏，偏向高基数特征）

### 单调性约束（Monotone Constraints）

当领域知识要求某个特征与目标呈单调关系时，可设置 `monotone_constraints`：

```python
# XGBoost: 第0个特征单调递增(+1)，第1个特征单调递减(-1)，第2个特征无约束(0)
model = xgb.XGBRegressor(monotone_constraints=[1, -1, 0])
```

## 代码

### 从零实现：XGBoost 风格的梯度提升

```python
import numpy as np


# =====================
# 辅助函数
# =====================

def _sigmoid(x):
    """数值稳定的 Sigmoid 函数"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# =====================
# 回归树（梯度提升的基学习器）
# =====================

class RegressionTree:
    """回归树 —— 使用二阶梯度统计量（g, h）进行分裂决策

    这是 XGBoost 风格的基础树模型，用于梯度提升框架中。
    与传统 CART 不同，它不直接拟合原始标签 y，而是拟合一阶梯度 g 和二阶梯度 h。
    """

    def __init__(self, max_depth=3, min_child_weight=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, colsample=1.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.colsample = colsample
        self.tree_ = None
        self.feature_importances_ = None

    def _gain(self, G_L, H_L, G_R, H_R):
        """计算分裂增益

        Gain = 1/2 [G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - (G_L+G_R)^2/(H_L+H_R+λ)] - γ

        参数
        ----
        G_L, H_L : float — 左子节点梯度和与 Hessian 和
        G_R, H_R : float — 右子节点梯度和与 Hessian 和

        返回
        ----
        gain : float — 如果 <= 0 则不应分裂
        """
        gain = 0.5 * (
            (G_L ** 2) / (H_L + self.reg_lambda)
            + (G_R ** 2) / (H_R + self.reg_lambda)
            - ((G_L + G_R) ** 2) / (H_L + H_R + self.reg_lambda)
        ) - self.gamma
        return gain

    def _leaf_weight(self, G, H):
        """计算最优叶子权重

        w*_j = -G_j / (H_j + λ)
        """
        return -G / (H + self.reg_lambda)

    def _build(self, X, g, h, depth=0):
        """递归构建回归树

        参数
        ----
        X : ndarray, shape=(n_samples, n_features)
        g : ndarray, shape=(n_samples,) — 一阶梯度
        h : ndarray, shape=(n_samples,) — 二阶梯度
        depth : int — 当前深度
        """
        n_samples, n_features = X.shape
        G_total = np.sum(g)
        H_total = np.sum(h)

        # 终止条件
        if (self.max_depth is not None and depth >= self.max_depth) \
                or n_samples < 2:
            val = self._leaf_weight(G_total, H_total)
            return {'weight': val, 'size': n_samples}

        # 如果 Hessian 和小于 min_child_weight，也停止分裂
        if H_total < self.min_child_weight:
            val = self._leaf_weight(G_total, H_total)
            return {'weight': val, 'size': n_samples}

        # 特征子采样
        if self.colsample < 1.0:
            n_sub = max(1, int(n_features * self.colsample))
            feat_idx = np.sort(np.random.choice(n_features, n_sub, replace=False))
        else:
            feat_idx = np.arange(n_features)

        best_gain = -np.inf
        best_feat = None
        best_thr = None
        best_left_mask = None
        best_right_mask = None

        for f in feat_idx:
            # 按特征值排序
            sorted_idx = np.argsort(X[:, f])
            x_sorted = X[sorted_idx, f]
            g_sorted = g[sorted_idx]
            h_sorted = h[sorted_idx]

            # 累计求和（用于快速计算左右子节点的 G 和 H）
            cum_g = np.cumsum(g_sorted)
            cum_h = np.cumsum(h_sorted)

            # 遍历每个候选分裂点
            for i in range(n_samples - 1):
                # 跳过相同特征值（无法分裂）
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                G_L = cum_g[i]
                H_L = cum_h[i]
                G_R = G_total - G_L
                H_R = H_total - H_L

                # min_child_weight 约束
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                gain = self._gain(G_L, H_L, G_R, H_R)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thr = (x_sorted[i] + x_sorted[i + 1]) / 2.0

        # 无有效分裂
        if best_feat is None:
            val = self._leaf_weight(G_total, H_total)
            return {'weight': val, 'size': n_samples}

        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            val = self._leaf_weight(G_total, H_total)
            return {'weight': val, 'size': n_samples}

        return {
            'feature': best_feat,
            'threshold': best_thr,
            'left': self._build(X[left_mask], g[left_mask], h[left_mask], depth + 1),
            'right': self._build(X[right_mask], g[right_mask], h[right_mask], depth + 1),
            'size': n_samples
        }

    def fit(self, X, g, h):
        """拟合回归树

        参数
        ----
        X : ndarray, shape=(n_samples, n_features)
        g : ndarray, shape=(n_samples,) — 一阶梯度
        h : ndarray, shape=(n_samples,) — 二阶梯度
        """
        n_features = X.shape[1]
        self.tree_ = self._build(X, g, h)

        # 提取特征重要性（按被选为分裂特征的次数）
        self.feature_importances_ = np.zeros(n_features)
        self._extract_importance(self.tree_)
        return self

    def _extract_importance(self, node):
        if 'feature' in node:
            self.feature_importances_[node['feature']] += 1.0
            self._extract_importance(node['left'])
            self._extract_importance(node['right'])

    def _predict_row(self, x, node):
        """对单行样本进行预测"""
        if 'feature' not in node:
            return node['weight']
        if x[node['feature']] <= node['threshold']:
            return self._predict_row(x, node['left'])
        return self._predict_row(x, node['right'])

    def predict(self, X):
        """预测叶子权重值

        返回 shape=(n_samples,) 的预测值（即该树的输出 f_m(x)）
        """
        return np.array([self._predict_row(x, self.tree_) for x in X])


# =====================
# XGBoost 风格梯度提升
# =====================

class XGBoost:
    """XGBoost 风格梯度提升树 —— 二阶近似 + 显式正则化

    支持回归（L2 损失）和分类（对数损失）。

    参数
    ----
    n_estimators : int — 提升轮数
    learning_rate : float — 学习率（步长收缩）
    max_depth : int — 每棵树最大深度
    min_child_weight : float — 最小 Hessian 和
    subsample : float — 行采样比例
    colsample_bytree : float — 特征采样比例
    reg_lambda : float — L2 正则化系数
    reg_alpha : float — L1 正则化系数
    gamma : float — 分裂最小增益阈值
    task : str — 'regression' 或 'classification'
    random_state : int — 随机种子
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
                 task='regression', random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.task = task
        self.random_state = random_state
        self.trees_ = []
        self.base_pred_ = None
        self.feature_importances_ = None
        self.train_loss_ = []
        self._is_fitted = False

    def _compute_grad_hess(self, y, current_pred):
        """根据任务类型计算一阶梯度 g 和二阶梯度 h

        回归（L2 损失）:
            L = 0.5 * (y - ŷ)^2
            g = ŷ - y
            h = 1

        分类（对数损失）:
            L = -[y log(p) + (1-y) log(1-p)], p = 1/(1+exp(-ŷ))
            g = p - y
            h = p * (1 - p)
        """
        if self.task == 'regression':
            g = current_pred - y
            h = np.ones_like(y)
        else:
            p = _sigmoid(current_pred)
            # 将 p 裁剪到 (eps, 1-eps) 防止极端值
            eps = 1e-7
            p = np.clip(p, eps, 1.0 - eps)
            g = p - y
            h = p * (1.0 - p)
        return g, h

    def _compute_loss(self, y, current_pred):
        """计算当前损失值（用于监控训练过程）"""
        if self.task == 'regression':
            return 0.5 * np.mean((current_pred - y) ** 2)
        else:
            p = _sigmoid(current_pred)
            eps = 1e-15
            p = np.clip(p, eps, 1.0 - eps)
            return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    def fit(self, X, y):
        """训练梯度提升模型

        参数
        ----
        X : ndarray, shape=(n_samples, n_features)
        y : ndarray, shape=(n_samples,)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.trees_ = []
        self.train_loss_ = []
        self.feature_importances_ = np.zeros(n_features)

        # 初始化预测值
        if self.task == 'regression':
            self.base_pred_ = float(np.mean(y))
        else:
            # 用 log-odds 初始化
            p = np.mean(y)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            self.base_pred_ = float(np.log(p / (1.0 - p)))

        current_pred = np.full(n_samples, self.base_pred_)

        for m in range(self.n_estimators):
            # 1. 计算梯度和 Hessian
            g, h = self._compute_grad_hess(y, current_pred)

            # 2. 行采样（subsample）
            if self.subsample < 1.0:
                n_sample = max(2, int(n_samples * self.subsample))
                idx = np.sort(np.random.choice(n_samples, n_sample, replace=False))
                X_sub, g_sub, h_sub = X[idx], g[idx], h[idx]
            else:
                X_sub, g_sub, h_sub = X, g, h

            # 3. 拟合回归树
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                colsample=self.colsample_bytree
            )
            tree.fit(X_sub, g_sub, h_sub)
            self.trees_.append(tree)

            # 4. 累积特征重要性
            if tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_

            # 5. 更新预测
            update = tree.predict(X)
            current_pred += self.learning_rate * update

            # 6. 记录损失
            loss = self._compute_loss(y, current_pred)
            self.train_loss_.append(loss)

        self._is_fitted = True
        return self

    def _raw_margin(self, X):
        """计算原始边际值（ŷ = base_pred + Σ η·f_m(x)）"""
        pred = np.full(X.shape[0], self.base_pred_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred

    def predict(self, X):
        """预测

        回归：返回预测值
        分类：返回类别标签 (0/1)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        margin = self._raw_margin(X)

        if self.task == 'regression':
            return margin
        else:
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """预测概率（仅分类任务可用）

        返回 shape=(n_samples, 2)，第一列为 P(y=0)，第二列为 P(y=1)
        """
        if self.task != 'classification':
            raise AttributeError("predict_proba is only available for classification task.")

        margin = self._raw_margin(X)
        p1 = _sigmoid(margin)
        return np.column_stack([1.0 - p1, p1])

    def get_feature_importance(self, importance_type='gain'):
        """获取特征重要性

        importance_type : str
            - 'gain' : 基于增益（当前默认使用分裂次数，与真实 XGBoost 的 gain 不同）
            - 'frequency' : 基于分裂次数

        注意：此处实现的是 split-based importance（频率），
        与 xgboost 库中的 'gain' 不同（需要实际增益累加）。
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.feature_importances_.copy()


# =====================
# 使用示例
# =====================

if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score

    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------
    # 1. 回归示例
    # ------------------------------------
    print("=" * 60)
    print("回归任务（Regression）")
    print("=" * 60)

    X_reg, y_reg = make_regression(n_samples=1000, n_features=10,
                                   noise=0.5, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    gbm_reg = XGBoost(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.0,
        task='regression',
        random_state=42
    )
    gbm_reg.fit(Xr_train, yr_train)

    y_pred_reg = gbm_reg.predict(Xr_test)
    rmse = np.sqrt(mean_squared_error(yr_test, y_pred_reg))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"训练损失（前10轮）: {[f'{l:.4f}' for l in gbm_reg.train_loss_[:10]]}")
    print(f"训练损失（最后10轮）: {[f'{l:.4f}' for l in gbm_reg.train_loss_[-10:]]}")

    # ------------------------------------
    # 2. 分类示例
    # ------------------------------------
    print("\n" + "=" * 60)
    print("分类任务（Binary Classification）")
    print("=" * 60)

    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42
    )

    gbm_clf = XGBoost(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.0,
        task='classification',
        random_state=42
    )
    gbm_clf.fit(Xc_train, yc_train)

    y_pred_clf = gbm_clf.predict(Xc_test)
    acc = accuracy_score(yc_test, y_pred_clf)
    proba = gbm_clf.predict_proba(Xc_test)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"样本1预测概率: P(y=0)={proba[0, 0]:.4f}, P(y=1)={proba[0, 1]:.4f}")
    print(f"样本2预测概率: P(y=0)={proba[1, 0]:.4f}, P(y=1)={proba[1, 1]:.4f}")
    print(f"训练损失（前10轮）: {[f'{l:.4f}' for l in gbm_clf.train_loss_[:10]]}")

    # ------------------------------------
    # 3. 特征重要性展示
    # ------------------------------------
    print("\n" + "=" * 60)
    print("特征重要性（基于分裂频率）")
    print("=" * 60)
    fi = gbm_reg.get_feature_importance()
    print(f"特征重要性向量: {fi}")
    print(f"最高重要性特征: argmax = {np.argmax(fi)}, "
          f"value = {fi[np.argmax(fi)]:.1f} / {np.sum(fi):.1f} total splits")
```

### 基于 xgboost / lightgbm 库的生产用法

```python
# =====================
# XGBoost 生产用法
# =====================

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

# 快速 Baseline
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    gamma=0,
    eval_metric='rmse',
    random_state=42
)

# 早停训练
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# 或使用原生 API（更细粒度控制）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.0,
    'gamma': 0,
    'eval_metric': 'rmse',
    'seed': 42
}

evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)

# 特征重要性
importance = model.get_score(importance_type='gain')  # 'gain', 'cover', 'weight'


# 超参数搜索（XGBoost）
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0],
    'gamma': [0, 0.1, 1.0],
}
search = GridSearchCV(
    xgb.XGBClassifier(n_estimators=200, random_state=42),
    param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
)
search.fit(X_train, y_train)
print("Best Params:", search.best_params_)


# =====================
# LightGBM 生产用法
# =====================

import lightgbm as lgb

# LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': -1,  # 无限制（由 num_leaves 控制）
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.0,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0.0,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'seed': 42
}

# 带早停的训练
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# 原生类别特征支持
# 将类别特征列索引传入 categorical_feature 参数
# X_train 中第 3、5 列是类别特征
model_lgb_cat = lgb.train(
    params,
    train_data,
    categorical_feature=[3, 5],  # LightGBM 自动处理
    num_boost_round=100
)

# scikit-learn 接口
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1e-3,
    random_state=42
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50)]
)

print(f"LightGBM Test Accuracy: {accuracy_score(y_test, lgbm_model.predict(X_test)):.4f}")


# =====================
# 实用技巧
# =====================

# 1. 学习率衰减 + 更多树的策略（推荐）
model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,  # 小学习率
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=100
)

# 2. 自定义评估指标（以 top-k 准确率为例）
def top_k_accuracy(y_pred, dtrain):
    y_true = dtrain.get_label()
    # 自定义指标逻辑
    return 'top_k_acc', 0.95  # 返回 (名称, 值)

# 3. 模型持久化
# model.save_model('xgboost_model.json')   # XGBoost JSON 格式
# model.booster_.save_model('model.ubj')   # XGBoost 二进制格式
# lgb_model.save_model('lightgbm_model.txt')  # LightGBM 文本格式
```

## 参考文献

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 3146–3154.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
