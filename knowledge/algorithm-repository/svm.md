# SVM — 支持向量机

- **来源**: Cortes, C. & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297; Vapnik, V. (1998). *Statistical Learning Theory*. Wiley.
- **DOI**: 10.1023/A:1022627411411
- **方法类别**: 机器学习 / 监督学习 / 分类与回归

## 数学设定

SVM 的核心思想是在特征空间中构建一个最大间隔分离超平面，并通过核技巧将线性方法推广到非线性情形。涵盖 SVC (分类) 和 SVR (回归) 两大分支。

---

### 一、最大间隔分类器 (Maximal Margin Classifier)

#### 线性可分情形
设训练数据 $\{(x_i, y_i)\}_{i=1}^N$，其中 $x_i \in \mathbb{R}^p$，$y_i \in \{-1, +1\}$。

**分离超平面**：
$$
w \cdot x + b = 0, \quad w \in \mathbb{R}^p,\; b \in \mathbb{R}
$$

**决策函数**：
$$
f(x) = \operatorname{sign}(w \cdot x + b)
$$

**几何间隔**：样本点到超平面的距离为 $\frac{|w \cdot x + b|}{\|w\|}$。
对于正确分类且满足 $y_i(w \cdot x_i + b) \geq 1$ 的样本，间隔为 $2/\|w\|$。

**优化问题（原问题）**：
$$
\min_{w, b} \; \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w \cdot x_i + b) \geq 1,\; \forall i
$$

最大化间隔等价于最小化 $\|w\|^2$。

---

#### 拉格朗日对偶 (Lagrangian Dual)

构造拉格朗日函数（$\alpha_i \geq 0$ 为 Lagrange 乘子）：
$$
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{N} \alpha_i \bigl[ y_i (w \cdot x_i + b) - 1 \bigr]
$$

KKT 条件给出：
$$
\frac{\partial \mathcal{L}}{\partial w} = 0 \quad\Rightarrow\quad w = \sum_{i=1}^{N} \alpha_i y_i x_i
$$
$$
\frac{\partial \mathcal{L}}{\partial b} = 0 \quad\Rightarrow\quad \sum_{i=1}^{N} \alpha_i y_i = 0
$$

代入后得到 **对偶问题**：
$$
\max_{\alpha} \; \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$
$$
\text{s.t.} \quad \alpha_i \geq 0,\; \sum_{i=1}^{N} \alpha_i y_i = 0
$$

**决策函数的对偶形式**：
$$
f(x) = \operatorname{sign}\!\left( \sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x) + b \right)
$$

**支持向量 (Support Vectors)**：满足 $\alpha_i > 0$ 的样本点，它们恰好位于间隔边界上（$y_i (w \cdot x_i + b) = 1$）。非支持向量的 $\alpha_i = 0$，对模型无贡献。

---

### 二、软间隔 (Soft-Margin SVM / C-SVM)

当数据线性不可分时，引入松弛变量 $\xi_i \geq 0$ 允许部分样本越界：

**原问题**：
$$
\min_{w, b, \xi} \; \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \xi_i
$$
$$
\text{s.t.} \quad y_i (w \cdot x_i + b) \geq 1 - \xi_i,\; \xi_i \geq 0,\; \forall i
$$

其中 $C > 0$ 是正则化参数，控制间隔最大化与误分类惩罚之间的权衡。

**对偶问题**（仅 $\alpha_i$ 的上限变为 $C$）：
$$
\max_{\alpha} \; \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$
$$
\text{s.t.} \quad 0 \leq \alpha_i \leq C,\; \sum_{i=1}^{N} \alpha_i y_i = 0
$$

**合页损失 (Hinge Loss)**：C-SVM 等价于最小化正则化合页损失
$$
\min_{w, b} \; \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \max\bigl(0, 1 - y_i (w \cdot x_i + b)\bigr)
$$

---

### 三、核技巧 (Kernel Trick)

通过非线性映射 $\phi: \mathbb{R}^p \to \mathcal{H}$ 将数据映射到高维（甚至无穷维）特征空间，使得在该空间中数据线性可分。核技巧避免了显式计算 $\phi(x)$，只需定义核函数：

$$
K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)
$$

对偶问题中的内积 $(x_i \cdot x_j)$ 全部替换为 $K(x_i, x_j)$：
$$
\max_{\alpha} \; \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j K(x_i, x_j)
$$

决策函数：
$$
f(x) = \operatorname{sign}\!\left( \sum_{i=1}^{N} \alpha_i y_i K(x_i, x) + b \right)
$$

#### 常用核函数

| 核函数 | 公式 | 关键参数 |
|--------|------|----------|
| 线性核 (Linear) | $K(x, y) = x \cdot y$ | — |
| 多项式核 (Polynomial) | $K(x, y) = (\gamma x \cdot y + r)^d$ | $\gamma$, $r$, $d$ |
| RBF/高斯核 (Gaussian) | $K(x, y) = \exp(-\gamma \|x - y\|^2)$ | $\gamma$ |
| Sigmoid 核 | $K(x, y) = \tanh(\gamma x \cdot y + r)$ | $\gamma$, $r$ |
| Laplacian 核 | $K(x, y) = \exp(-\gamma \|x - y\|_1)$ | $\gamma$ |

#### Mercer 条件 (核有效性)

函数 $K(x, y)$ 是合法核函数的充要条件：对任意有限样本集 $\{x_i\}_{i=1}^N$，核矩阵 $K_{ij} = K(x_i, x_j)$ 是 **半正定矩阵**（所有特征值 $\geq 0$）。

- **线性核**、**RBF 核**：始终满足 Mercer 条件
- **多项式核**：当 $\gamma \geq 0$ 时有效
- **Sigmoid 核**：仅在特定参数组合下满足（非 Mercer 核，但实践中仍可用）

RBF 核是默认首选，因为它是一个 **通用核**（Universal Kernel），具有逼近任意连续函数的能力。

---

### 四、SVR — 支持向量回归

SVR 将 SVM 的思想推广到回归问题，核心是 **$\varepsilon$-不敏感损失函数** ($\varepsilon$-insensitive loss)：

$$
L_\varepsilon\bigl(y, f(x)\bigr) = \max\bigl(0, |y - f(x)| - \varepsilon\bigr)
$$

只惩罚预测值与真实值的偏差超过 $\varepsilon$ 的样本，允许 $\varepsilon$-管内误差不计。

#### 原问题
$$
\min_{w, b, \xi, \xi^*} \; \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} (\xi_i + \xi_i^*)
$$
$$
\text{s.t.} \quad
\begin{cases}
y_i - (w \cdot \phi(x_i) + b) \leq \varepsilon + \xi_i, \\[2pt]
(w \cdot \phi(x_i) + b) - y_i \leq \varepsilon + \xi_i^*, \\[2pt]
\xi_i, \xi_i^* \geq 0
\end{cases}
$$

#### 对偶问题
引入两组 Lagrange 乘子 $\alpha_i, \alpha_i^* \in [0, C]$：

$$
\max_{\alpha, \alpha^*} \; -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*) K(x_i, x_j) - \varepsilon \sum_{i=1}^{N} (\alpha_i + \alpha_i^*) + \sum_{i=1}^{N} y_i (\alpha_i - \alpha_i^*)
$$
$$
\text{s.t.} \quad \sum_{i=1}^{N} (\alpha_i - \alpha_i^*) = 0,\quad 0 \leq \alpha_i, \alpha_i^* \leq C
$$

令 $\beta_i = \alpha_i - \alpha_i^*$，则 $\beta_i \in [-C, C]$，$\sum \beta_i = 0$。

**决策函数**：
$$
f(x) = \sum_{i=1}^{N} (\alpha_i - \alpha_i^*) K(x_i, x) + b = \sum_{i=1}^{N} \beta_i K(x_i, x) + b
$$

支持向量对应于 $\alpha_i > 0$ 或 $\alpha_i^* > 0$（即 $|\beta_i| > 0$）的样本，这些样本位于 $\varepsilon$-管边界上或外部。

---

### 五、$\nu$-SVM

$\nu$-SVM 用参数 $\nu \in (0, 1]$ 替代 $C$，提供更直观的控制：

- **$\nu$-SVC**：$\nu$ 是支持向量比例的下界，同时也是间隔错误样本比例的上界
- **$\nu$-SVR**：$\nu$ 是 $\varepsilon$-管外样本比例的上界

$\nu$ 的范围总是 $[0, 1]$，而 $C$ 的范围是 $(0, \infty)$，因此 $\nu$ 更易调优。

---

## 关键假设

1. **核变换后的可分性**：存在一个特征空间（通过核映射）使得数据在该空间中近似线性可分（软间隔放宽了严格可分的要求）
2. **核相似性假设**：选择的核函数能够编码领域内合理的相似性度量（如 RBF 假设局部相似性）
3. **特征尺度敏感性**：SVM 对特征尺度高度敏感——必须对输入特征进行标准化/归一化（零均值单位方差），否则数值大的特征会主导核函数值
4. **边际主导**：SVM 的泛化能力由间隔大小和支撑向量数目决定，与数据维度无直接依赖关系（这是其在高维空间中有效的理论基础）
5. **SVR 的 $\varepsilon$ 先验**：SVR 假设存在一个合理的误差容忍度 $\varepsilon$，在此范围内的偏差可忽略

---

## 适用场景

### 推荐使用
- **中等规模数据**（$N$ 至多约 10 万）：标准 SVM 训练复杂度 $O(N^2 \sim N^3)$，对大规模数据需使用近似方法
- **高维特征空间**（$p$ 可大于 $N$）：SVM 的泛化误差理论上仅与支持向量数有关，而非 $p$
- **边界清晰的中等复杂度分类**：如文本分类（线性核）、图像识别（RBF 核）、生物信息学
- **非线性决策边界**：RBF 核可捕获复杂非线性模式
- **特征数远大于样本数的场景**：如基因表达数据 ($p \gg N$)
- **需要稳健的回归模型**：SVR 对异常值相对不敏感（$\varepsilon$ 管的存在）

### 不适用
- **超大规模数据**（$N > 10^5$）：标准 SMO 复杂度 $O(N^2)$，需用近似方法（LinearSVC、随机傅里叶特征、Nystrom 近似）
- **缺乏明显间隔的密集高维数据**：所有样本纠缠在一起时，SVM 退化为接近随机
- **概率校准要求高时**：SVM 的 Platt 缩放概率估计不如逻辑回归或 GBDT 可靠
- **可解释性至关重要时**：非线性核 SVM 的决策边界难以解释；线性 SVM 可通过特征权重部分解释
- **特征尺度差异大且无法标准化时**：如某些特征为计数，另一些为比例
- **在线/流式学习场景**：标准 SVM 是批处理模型（增量 SVM 存在但不够成熟）

---

## 实现要点

### 关键超参数

| 参数 | 范围 | 默认值 (sklearn) | 说明 |
|------|------|-------------------|------|
| $C$ (正则化) | $(0, \infty)$，常用 $[10^{-3}, 10^3]$ | 1.0 | 越大越容忍误分类（低偏差高方差），越小间隔越宽（高偏差低方差） |
| $\gamma$ (RBF 宽度) | $(0, \infty)$，常用 $[10^{-3}, 10^3]$ | `'scale'` | $\gamma$ 越大 → 高斯核越窄 → 每个支持向量的影响范围越小 → 模型越复杂 |
| $\mathrm{kernel}$ | `linear / poly / rbf / sigmoid` | `'rbf'` | 决定特征映射类型；无先验时默认 RBF |
| $\varepsilon$ (SVR 管宽) | $[0, \infty)$，常用 $[0.01, 1]$ | 0.1 | $\varepsilon$ 越大 → 管内样本越多 → 支持向量越少 → 回归函数越平滑 |
| $\nu$ ($\nu$-SVM) | $(0, 1]$ | 0.5 | 控制支持向量比例下界（SVC）或管外样本比例上界（SVR） |
| $d$ (多项式次数) | $\mathbb{N}^+$ | 3 | 多项式核的度数，越高模型越灵活但易过拟合 |
| $\mathrm{class\\_weight}$ | `None / 'balanced' / dict` | `None` | 处理类别不平衡：自动按类别频率反比加权 |

### 调优经验

1. **先标准化，再调参**：特征标准化是不可跳过的步骤——对 RBF 核 SVM，未标准化会导致 $\gamma$ 完全失效
2. **RBF 核的网格搜索策略**：$(C, \gamma)$ 应在对数尺度上搜索，常用 $C \in \{2^{-5}, 2^{-3}, \dots, 2^{15}\}$，$\gamma \in \{2^{-15}, 2^{-13}, \dots, 2^{3}\}$
3. **$C$ 与模型复杂度**：$C$ 越大，对误分类的惩罚越重，决策边界越复杂（趋向过拟合）；$C$ 越小，间隔越宽但可能欠拟合
4. **$\gamma$ 的物理意义**：$\gamma \approx 1/(p \cdot \text{Var}(X))$ 是一个合理的初始值；$\gamma$ 接近 $0$ 时所有样本被平等对待（近似线性）；$\gamma$ 极大时每个样本只影响自身（最近邻行为）
5. **先试线性核**：当 $p$ 很大或 $N$ 很大时，线性核往往已有足够好的表现，且训练速度快得多
6. **SVR 的 $\varepsilon$ 选择**：$\varepsilon$ 通常设为目标变量标准差的 10%~20%；过小会导致支持向量过多
7. **$\nu$ vs $C$**：$\nu$ 比 $C$ 更直观（有界 $[0,1]$，含义清晰），但 $\nu$-SVM 的优化问题稍复杂；sklearn 两者均支持

### 核心技术细节

- **特征缩放 — 强制要求**：所有特征必须标准化为均值为 0、方差为 1。使用 `StandardScaler` 进行，否则基于距离的核（RBF、多项式）会失效
- **Platt 缩放 (Platt Scaling)**：通过拟合一个逻辑回归模型 $P(y=1|x) = 1/(1 + \exp(Af(x) + B))$ 将决策值 $f(x)$ 转换为概率输出。交叉验证训练 $(A, B)$ 可防止过拟合
- **SMO (Sequential Minimal Optimization)**：Platt (1998) 提出的高效优化算法，每次选取两个 $\alpha$ 进行解析优化，将原问题分解为一系列二维子问题。复杂度约 $O(N^2)$ 到 $O(N^{2.3})$
- **多分类策略**：
  - One-vs-One (OvO)：libsvm 默认，训练 $K(K-1)/2$ 个二分类器，投票决策
  - One-vs-Rest (OvR)：训练 $K$ 个二分类器，每个区分一类与其余；sklearn `SVC(decision_function_shape='ovr')`
  - OvO 通常对不平衡数据更鲁棒，但 OvR 计算量更小
- **类别不平衡处理**：设置 `class_weight='balanced'` 自动按 $w_i = N / (K \cdot N_i)$ 分配权重，等价于对少数类使用更大的 $C$
- **Nu-SVM 替代 C-SVM**：$\nu$ 参数更直观且有界，当需要精确控制支持向量比例时使用 `NuSVC` / `NuSVR`

---

## 完整 Python 代码

### 从零实现 — SVM 分类 (SVC with SMO)

```python
import numpy as np
from numpy.random import choice


# =====================
# 核函数
# =====================
def linear_kernel(x, y):
    return np.dot(x, y)


def poly_kernel(x, y, gamma=1.0, coef0=1.0, degree=3):
    return (gamma * np.dot(x, y) + coef0) ** degree


def rbf_kernel(x, y, gamma=1.0):
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))


def sigmoid_kernel(x, y, gamma=1.0, coef0=0.0):
    return np.tanh(gamma * np.dot(x, y) + coef0)


_KERNEL_MAP = {
    'linear': linear_kernel,
    'poly': poly_kernel,
    'rbf': rbf_kernel,
    'sigmoid': sigmoid_kernel,
}


# =====================
# SVC — SMO 算法
# =====================
class SVC:
    """Support Vector Classification (from scratch with SMO)"""

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, max_passes=100, random_state=None):
        self.C = C
        self.kernel_name = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state

    def _kernel_fn(self, x, y):
        fn = _KERNEL_MAP[self.kernel_name]
        if self.kernel_name == 'linear':
            return fn(x, y)
        elif self.kernel_name == 'poly':
            return fn(x, y, self.gamma_, self.coef0, self.degree)
        elif self.kernel_name == 'rbf':
            return fn(x, y, self.gamma_)
        elif self.kernel_name == 'sigmoid':
            return fn(x, y, self.gamma_, self.coef0)

    def _kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = self._kernel_fn(X[i], X[j])
                K[i, j] = val
                K[j, i] = val
        return K

    def fit(self, X, y):
        n, d = X.shape

        # 转换标签为 {+1, -1}
        y_ = np.where(y <= 0, -1.0, 1.0)
        y_ = y_.astype(float).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 确定 gamma
        if self.gamma == 'scale':
            self.gamma_ = 1.0 / (d * X.var())
        elif self.gamma == 'auto':
            self.gamma_ = 1.0 / d
        else:
            self.gamma_ = self.gamma

        # 预计算核矩阵
        K = self._kernel_matrix(X)

        # SMO 初始化
        alphas = np.zeros(n)
        b = 0.0
        passes = 0

        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                # 计算误差 E_i = f(x_i) - y_i
                E_i = np.sum(alphas * y_ * K[i]) + b - y_[i]

                # KKT 条件检查:
                #   α_i = 0   => y_i·E_i ≥ 0
                #   0 < α_i < C => y_i·E_i = 0
                #   α_i = C   => y_i·E_i ≤ 0
                if (y_[i] * E_i < -self.tol and alphas[i] < self.C) or \
                   (y_[i] * E_i > self.tol and alphas[i] > 0):

                    # 随机选择 j ≠ i
                    j = choice([k for k in range(n) if k != i])
                    E_j = np.sum(alphas * y_ * K[j]) + b - y_[j]

                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                    # 计算裁剪边界 [L, H]
                    if y_[i] != y_[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])

                    if abs(L - H) < 1e-10:
                        continue

                    # η = K_ii + K_jj - 2K_ij
                    eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                    if eta <= 0:
                        continue

                    # 更新 α_j
                    alphas[j] += y_[j] * (E_i - E_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新 α_i
                    alphas[i] += y_[i] * y_[j] * (alpha_j_old - alphas[j])

                    # 更新偏置 b
                    b1 = b - E_i \
                         - y_[i] * (alphas[i] - alpha_i_old) * K[i, i] \
                         - y_[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j \
                         - y_[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                         - y_[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # 保存模型
        sv_mask = alphas > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.dual_coef_ = alphas[sv_mask] * y_[sv_mask]
        self.intercept_ = b
        self.n_support_ = np.sum(sv_mask)

        # 保留预测所需的数据
        self.X_fit_ = X
        self.y_fit_ = y_
        self.alphas_ = alphas

        return self

    def decision_function(self, X):
        """计算决策值 f(x) = Σα_i y_i K(x_i, x) + b"""
        sv_mask = self.alphas_ > 1e-5
        sv_alphas = self.alphas_[sv_mask]
        sv_y = self.y_fit_[sv_mask]
        sv_X = self.X_fit_[sv_mask]

        y_pred = np.zeros(X.shape[0])
        for t in range(X.shape[0]):
            s = 0.0
            for a, yi, sv_x in zip(sv_alphas, sv_y, sv_X):
                s += a * yi * self._kernel_fn(X[t], sv_x)
            y_pred[t] = s + self.intercept_
        return y_pred

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)

    def score(self, X, y):
        y_ = np.where(y <= 0, -1, 1).ravel()
        pred = self.predict(X)
        return np.mean(pred == y_)


# =====================
# SVR — SMO 算法
# =====================
class SVR:
    """Support Vector Regression (from scratch with SMO)"""

    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, max_passes=100,
                 random_state=None):
        self.C = C
        self.epsilon = epsilon
        self.kernel_name = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state

    def _kernel_fn(self, x, y):
        fn = _KERNEL_MAP[self.kernel_name]
        if self.kernel_name == 'linear':
            return fn(x, y)
        elif self.kernel_name == 'poly':
            return fn(x, y, self.gamma_, self.coef0, self.degree)
        elif self.kernel_name == 'rbf':
            return fn(x, y, self.gamma_)
        elif self.kernel_name == 'sigmoid':
            return fn(x, y, self.gamma_, self.coef0)

    def _kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = self._kernel_fn(X[i], X[j])
                K[i, j] = val
                K[j, i] = val
        return K

    def fit(self, X, y):
        n, d = X.shape
        y = y.astype(float).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.gamma == 'scale':
            self.gamma_ = 1.0 / (d * X.var())
        elif self.gamma == 'auto':
            self.gamma_ = 1.0 / d
        else:
            self.gamma_ = self.gamma

        K = self._kernel_matrix(X)

        # 对偶变量 β_i = α_i - α*_i, β_i ∈ [-C, C], Σβ_i = 0
        beta = np.zeros(n)
        b = 0.0

        # 存储 f(x_k) 的无偏置部分：f_no_b[k] = Σβ_j K(x_k, x_j)
        f_no_b = np.zeros(n)

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                f_i = f_no_b[i] + b
                E_i = f_i - y[i]

                # KKT 违反条件 (ε-SVR):
                #   E_i > ε 且 β_i > -C → 应降低 β_i 以减小 f(x_i)
                #   E_i < -ε 且 β_i < C  → 应升高 β_i 以增大 f(x_i)
                violate = False
                if E_i > self.epsilon and beta[i] > -self.C:
                    violate = True
                if E_i < -self.epsilon and beta[i] < self.C:
                    violate = True

                if not violate:
                    continue

                # 选择 j ≠ i
                j = choice([k for k in range(n) if k != i])
                f_j = f_no_b[j] + b
                E_j = f_j - y[j]

                beta_i_old, beta_j_old = beta[i], beta[j]

                # Σβ = 0 约束 => β_i + β_j 为常数
                const = beta[i] + beta[j]

                eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                if eta <= 0:
                    continue

                # 无约束更新 β_j
                beta_j_new = beta[j] + (E_i - E_j) / eta

                # 裁剪: β_i, β_j 均需 ∈ [-C, C]
                L = max(-self.C, const - self.C)
                H = min(self.C, const + self.C)
                beta_j_new = np.clip(beta_j_new, L, H)

                if abs(beta_j_new - beta[j]) < 1e-5:
                    continue

                beta[j] = beta_j_new
                beta[i] = const - beta[j]

                # 增量更新 f_no_b
                d_i = beta[i] - beta_i_old
                d_j = beta[j] - beta_j_old
                for k in range(n):
                    f_no_b[k] += d_i * K[i, k] + d_j * K[j, k]

                # 更新偏置 b：利用自由支持向量 (0 < |β| < C) 恢复
                b_vals = []
                for k in range(n):
                    if 1e-5 < beta[k] < self.C - 1e-5:
                        # β_k > 0 → 上边界: f(x_k) = y_k + ε
                        b_vals.append(y[k] - f_no_b[k] - self.epsilon)
                    elif -self.C + 1e-5 < beta[k] < -1e-5:
                        # β_k < 0 → 下边界: f(x_k) = y_k - ε
                        b_vals.append(y[k] - f_no_b[k] + self.epsilon)

                if len(b_vals) > 0:
                    b = np.mean(b_vals)
                else:
                    # 所有支持向量均在边界上，使用 i, j 估计
                    bi = y[i] - f_no_b[i] - self.epsilon * np.sign(beta[i]) \
                         if abs(beta[i]) > 1e-5 else None
                    bj = y[j] - f_no_b[j] - self.epsilon * np.sign(beta[j]) \
                         if abs(beta[j]) > 1e-5 else None
                    candidates = [v for v in [bi, bj] if v is not None]
                    if candidates:
                        b = np.mean(candidates)

                num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # 保存模型
        sv_mask = np.abs(beta) > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.dual_coef_ = beta[sv_mask]
        self.intercept_ = b
        self.n_support_ = np.sum(sv_mask)

        self.X_fit_ = X
        self.beta_ = beta

        return self

    def predict(self, X):
        """预测: f(x) = Σβ_i K(x_i, x) + b"""
        sv_mask = np.abs(self.beta_) > 1e-5
        sv_beta = self.beta_[sv_mask]
        sv_X = self.X_fit_[sv_mask]

        y_pred = np.zeros(X.shape[0])
        for t in range(X.shape[0]):
            s = 0.0
            for beta_k, sv_x in zip(sv_beta, sv_X):
                s += beta_k * self._kernel_fn(X[t], sv_x)
            y_pred[t] = s + self.intercept_
        return y_pred

    def score(self, X, y):
        y = y.astype(float).ravel()
        pred = self.predict(X)
        u = ((y - pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v


# =====================
# 使用示例
# =====================
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    # ======== 分类示例 ========
    print("=" * 60)
    print("SVC 分类示例")
    print("=" * 60)

    X_clf, y_clf = make_classification(
        n_samples=300, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42
    )

    # 标准化 (SVM 必需!)
    scaler = StandardScaler()
    Xc_train_s = scaler.fit_transform(Xc_train)
    Xc_test_s = scaler.transform(Xc_test)

    # 训练自定义 SVC
    svc = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
    svc.fit(Xc_train_s, yc_train)
    acc = svc.score(Xc_test_s, yc_test)
    print(f"自定义 SVC 测试准确率: {acc:.4f}")
    print(f"支持向量数: {svc.n_support_} / {len(Xc_train_s)}")

    # ======== 回归示例 ========
    print()
    print("=" * 60)
    print("SVR 回归示例")
    print("=" * 60)

    X_reg, y_reg = make_regression(
        n_samples=200, n_features=5, noise=0.3, random_state=42
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    Xr_train_s = scaler_X.fit_transform(Xr_train)
    Xr_test_s = scaler_X.transform(Xr_test)
    yr_train_s = scaler_y.fit_transform(yr_train.reshape(-1, 1)).ravel()
    yr_test_s = scaler_y.transform(yr_test.reshape(-1, 1)).ravel()

    svr = SVR(C=1.0, epsilon=0.2, kernel='rbf', gamma='scale', random_state=42)
    svr.fit(Xr_train_s, yr_train_s)
    r2 = svr.score(Xr_test_s, yr_test_s)
    print(f"自定义 SVR 测试 R²: {r2:.4f}")
    print(f"支持向量数: {svr.n_support_} / {len(Xr_train_s)}")
```

### 基于 scikit-learn 的生产用法

```python
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ======== SVC 快速 Baseline ========
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                random_state=42))
])
pipe.fit(X_train, y_train)
print(f"SVC accuracy: {pipe.score(X_test, y_test):.4f}")

# ======== 网格搜索 (C, gamma) ========
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
    'svc__kernel': ['rbf'],
}
grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('svc', SVC(random_state=42))]),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳 CV 准确率: {grid.best_score_:.4f}")

# ======== SVR ========
svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1))
])
svr_pipe.fit(X_train, y_train)
print(f"SVR R²: {svr_pipe.score(X_test, y_test):.4f}")

# ======== NuSVC (ν 参数控制) ========
nusvc = NuSVC(nu=0.3, kernel='rbf', gamma='scale', random_state=42)
nusvc.fit(X_train_scaled, y_train)
print(f"NuSVC accuracy: {nusvc.score(X_test_scaled, y_test):.4f}")

# ======== 线性 SVM (大规模数据) ========
# LinearSVC 基于 liblinear，复杂度 O(N)，适合大规模数据
linear_svc = LinearSVC(C=1.0, max_iter=10000, random_state=42)
linear_svc.fit(X_train_scaled, y_train)
print(f"LinearSVC accuracy: {linear_svc.score(X_test_scaled, y_test):.4f}")

# ======== 核函数对比 ========
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    svc = SVC(kernel=k, C=1.0, gamma='scale', random_state=42)
    svc.fit(X_train_scaled, y_train)
    acc = svc.score(X_test_scaled, y_test)
    print(f"kernel={k:10s}  accuracy={acc:.4f}")
```

---

## 参考文献

1. Cortes, C. & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
2. Vapnik, V. (1998). *Statistical Learning Theory*. Wiley.
3. Platt, J. C. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. Microsoft Research Technical Report MSR-TR-98-14.
4. Smola, A. J. & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14(3), 199–222.
5. Schölkopf, B. & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
6. Chang, C.-C. & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. *ACM Transactions on Intelligent Systems and Technology*, 2(3), 1–27.
