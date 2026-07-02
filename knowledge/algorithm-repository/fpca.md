# FPCA 函数型主成分分析 / 函数型回归

- **来源**: Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer.
- **DOI**: 10.1007/b98888
- **方法类别**: 函数型数据分析

## 数学设定

### 函数型数据表示

观测数据 $\{X_i(t)\}_{i=1}^{N}$ 被视作定义在紧集 $\mathcal{T} \subset \mathbb{R}$ 上的光滑函数。实际观测到的往往是离散点 $(t_{ij}, y_{ij})$，其中 $y_{ij} = X_i(t_{ij}) + \varepsilon_{ij}$。

### 基展开

将每个函数在选定的基函数系 $\{\phi_k(t)\}_{k=1}^{K}$ 上展开：

$$
X_i(t) = \sum_{k=1}^{K} c_{ik} \phi_k(t)
$$

常用基函数：

- **B-splines**（非周期数据，最常用）：分段多项式，紧支撑，数值稳定
- **Fourier 基**（周期数据）：$\{1, \sin(k\omega t), \cos(k\omega t)\}$
- **小波基**（局部突变信号）：多尺度分析

### 平滑（Penalized Least Squares）

通过粗糙度惩罚实现平滑：

$$
\hat{X}_i = \arg\min_{f \in \mathcal{F}} \sum_{j=1}^{m_i} \bigl(y_{ij} - f(t_{ij})\bigr)^2 + \lambda \int_{\mathcal{T}} [f''(t)]^2 dt
$$

- 第一项：数据拟合程度
- 第二项：粗糙度惩罚（曲率平方积分）
- $\lambda$：平滑参数（$\lambda \to 0$ 插值，$\lambda \to \infty$ 退化为线性回归）
- $\lambda$ 选择：广义交叉验证（GCV）或 restricted maximum likelihood（REML）

平滑的基-惩罚形式（P-splines, Eilers & Marx 1996）：

$$
\hat{\mathbf{c}}_i = \arg\min_{\mathbf{c}} \|\mathbf{y}_i - \mathbf{\Phi} \mathbf{c}\|^2 + \lambda \mathbf{c}^\top \mathbf{P} \mathbf{c}
$$

其中 $\mathbf{\Phi}_{j,k} = \phi_k(t_j)$ 为设计矩阵，$\mathbf{P} = \mathbf{D}_d^\top \mathbf{D}_d$ 为 $d$ 阶差分惩罚矩阵。

### FPCA（函数型主成分分析）

**均值函数**：

$$
\mu(t) = \frac{1}{N} \sum_{i=1}^{N} X_i(t)
$$

**协方差函数**：

$$
v(s, t) = \frac{1}{N} \sum_{i=1}^{N} \bigl(X_i(s) - \mu(s)\bigr)\bigl(X_i(t) - \mu(t)\bigr), \quad s, t \in \mathcal{T}
$$

**特征分解**（Fredholm 积分方程）：

$$
\int_{\mathcal{T}} v(s, t) \, \xi_k(t) \, dt = \lambda_k \, \xi_k(s), \quad k = 1, 2, \dots
$$

- $\xi_k(t)$：第 $k$ 个特征函数（主成分权重函数），满足正交归一：
  $$
  \int_{\mathcal{T}} \xi_k(t) \xi_l(t) \, dt = \delta_{kl}
  $$
- $\lambda_k$：第 $k$ 个特征值，$\lambda_1 \ge \lambda_2 \ge \cdots \ge 0$

**FPC 得分**（函数在主成分方向上的投影）：

$$
f_{ik} = \int_{\mathcal{T}} \bigl(X_i(t) - \mu(t)\bigr) \, \xi_k(t) \, dt
$$

**Karhunen-Loeve 展开**（函数的重构表示）：

$$
X_i(t) = \mu(t) + \sum_{k=1}^{\infty} f_{ik} \, \xi_k(t)
$$

**方差解释比例**：

$$
\text{PVE}_k = \frac{\lambda_k}{\sum_{j=1}^{\infty} \lambda_j}, \quad \text{累积 PVE} = \frac{\sum_{j=1}^{K} \lambda_j}{\sum_{j=1}^{\infty} \lambda_j}
$$

### 函数型线性回归（scalar-on-function）

模型设定：

$$
Y_i = \alpha + \int_{\mathcal{T}} X_i(t) \, \beta(t) \, dt + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)
$$

其中 $\beta(t)$ 为系数函数（regression coefficient function），是整个模型的核心解释对象。

**FPCA 基展开法**（PCR 风格）：

将 $\beta(t)$ 在 FPCA 特征函数上展开：

$$
\beta(t) = \sum_{k=1}^{K} b_k \, \xi_k(t)
$$

利用 FPC 得分 $f_{ik} = \int (X_i(t) - \mu(t)) \xi_k(t) dt$，模型化为多元回归：

$$
Y_i = \alpha + \sum_{k=1}^{K} f_{ik} b_k + \varepsilon_i
$$

系数估计（带惩罚）：

$$
\hat{\mathbf{b}} = \arg\min_{\mathbf{b}} \sum_{i=1}^{N} \bigl(Y_i - \alpha - \mathbf{f}_i^\top \mathbf{b}\bigr)^2 + \lambda \int_{\mathcal{T}} [\beta''(t)]^2 dt
$$

惩罚项在 FPCA 基下等价于 $\lambda \mathbf{b}^\top \mathbf{\Lambda} \mathbf{b}$，其中 $\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_K)$（近似对角化）。

### 函数型共变模型（function-on-function / concurrent）

模型设定：

$$
Y_i(t) = \alpha(t) + X_i(t) \, \beta(t) + \varepsilon_i(t)
$$

- $\alpha(t)$：截距函数
- $\beta(t)$：系数函数（逐点线性效应）
- 这是函数型响应中最简单的模型，每个时间点 $t$ 上的回归是独立的

## 关键假设

- **光滑性**：函数至少二次可微（粗糙度惩罚的前提）
- **独立同分布**：各曲线 $X_i(t)$ 是独立同分布的随机函数实现
- **协方差算子迹有限**：$\sum_{k=1}^\infty \lambda_k < \infty$（保证 Karhunen-Loeve 收敛）
- **截断充分性**：$K$ 足够大以捕获主要变异模式，但不过大以避免噪声
- **共同域**：所有函数定义在同一紧集 $\mathcal{T}$ 上
- **稠密观测**（经典 FPCA）：每条曲线在足够密的网格上观测；稀疏/不规则观测需使用 PACE 方法
- **均方可积**：$E[\int X(t)^2 dt] < \infty$
- **连续性**（FLM）：系数函数 $\beta(t)$ 满足 $\int \beta(t)^2 dt < \infty$

## 适用场景

- **纵向数据**：生长曲线、认知发展轨迹、疾病进展曲线
- **曲线型观测**：光谱数据、近红外光谱、质谱、药代动力学曲线
- **运动捕捉**：关节角度轨迹、步态分析
- **环境与气象**：气温曲线、降水模式、污染浓度日变化
- **语音与信号**：音高轮廓、脑电图（EEG）/ 脑磁图（MEG）信号
- **经济与金融**：收益曲线、波动率曲面、消费模式

### 不适用

- **纯多元数据**：变量间无内在函数结构（如人口统计属性表），使用标准 PCA 或因子模型
- **极度稀疏观测**：每条曲线只有 2-3 个观测点，无附加信息无法恢复函数形态（需 PACE 法且仍谨慎）
- **无函数结构的交叉截面数据**：单次测量而非曲线，无函数性可挖掘
- **函数性不比多元方法带来额外收益时**：若各时间点独立，标准多元方法更简洁
- **极高频数据未预处理**：原始高频噪声未经平滑直接 FPCA 会导致偏估计

## 实现要点

### 基函数选择
| 数据类型 | 推荐基 | 说明 |
|---------|--------|------|
| 非周期、光滑 | B-splines (3-5 阶) | 最通用，紧支撑 |
| 周期数据 | Fourier | 正弦余弦，天然周期 |
| 局部突变 | Wavelets | 多尺度，稀疏表示 |
| 间断/分段 | B-splines + 节点放置 | 在间断点处加密节点 |

### 平滑参数 $\lambda$ 选择
- **GCV**：最小化 $\operatorname{GCV}(\lambda) = \frac{n^{-1}\|\mathbf{y} - \hat{\mathbf{y}}_\lambda\|^2}{(1 - n^{-1}\operatorname{tr}(\mathbf{S}_\lambda))^2}$
- **REML**：将平滑视为方差分量估计，适用于混合模型视角
- **AIC / BIC**：信息准则，对 $\lambda$ 的敏感度低于 GCV

### FPCA 实现路径

1. **平滑协方差法（推荐）**：
   - 先计算原始协方差面 $v(s, t)$
   - 对协方差面进行平滑（二维平滑）
   - 对平滑后的协方差面做特征分解
   - 更稳定，尤其适合中等噪声

2. **原始协方差法**：
   - 直接对离散化的协方差矩阵做特征分解
   - 计算快但噪声敏感

3. **SVD 捷径**（等间距稠密网格）：
   - 对中心化数据矩阵 $\mathbf{Z}_{N \times P}$ 做 SVD
   - $\mathbf{Z} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$
   - 特征值：$\lambda_k = \sigma_k^2 / N$
   - 特征函数：$\xi_k(t) = \mathbf{v}_k / \sqrt{\Delta t}$
   - FPC 得分：$f_{ik} = \sqrt{\Delta t} \cdot (\mathbf{U} \mathbf{\Sigma})_{ik}$

### 成分数选择
- **累积方差 85-95%** 规则：选取 $K$ 使得 $\sum_{k=1}^K \lambda_k / \sum_{k=1}^\infty \lambda_k \ge 0.85$
- **Scree plot / 肘部法则**：特征值曲线拐点处
- **交叉验证**：预测误差最小时的 $K$
- **AIC / BIC**：基于似然的准则

### 关键预处理
- **配准（Registration/Alignment）**：FPCA 前的必要步骤。如果曲线存在相位变异（如峰值位置漂移），未配准的 FPCA 会混淆幅度变异和相位变异。建议使用 fdasrsf 的 SRSF 弹性配准
- **中心化**：始终减去均值函数
- **标准化**：若各曲线变异幅度差异大，考虑尺度标准化

### PACE 方法（稀疏/不规则数据）
- Yao, Müller, Wang (2005) 提出的 Principal Analysis by Conditional Expectation
- 适用：每条曲线少量稀疏观测，观测时间点不规则且个体间不同
- 思路： pooled 所有观测估计均值函数和协方差面（二维局部加权平滑），再用条件期望估计个体 FPC 得分
- Python 实现：`scikit-fda` 的 `FPCA` 类支持 `method='pace'`

### 注意事项
- FPCA 特征函数符号任意（$\pm \xi_k(t)$ 都是合法解），解释时需确认符号方向
- $\beta(t)$ 的估计对 $K$ 敏感：$K$ 太小丢失信号，$K$ 太大引入噪声
- 函数型回归的惩罚推荐在 FPCA 基下进行（对角近似），计算高效

## 完整 Python 代码

```python
import numpy as np
from scipy import linalg, interpolate
import matplotlib.pyplot as plt


# ============================================================
# 工具：平滑（P-spline / 光滑样条）
# ============================================================

def smooth_data(y, t, lamb=None):
    """平滑函数型数据：argmin Σ(y_j - f(t_j))² + λ∫[f''(t)]²dt

    使用三次光滑样条（等价于惩罚最小二乘）。

    Parameters
    ----------
    y : ndarray, shape (n_curves, n_points) or (n_points,)
        观测函数值。
    t : ndarray, shape (n_points,)
        观测网格。
    lamb : float or None
        平滑参数。None 时由 scipy 自动通过 GCV 选择。

    Returns
    -------
    y_smooth : ndarray, shape (n_curves, n_points) or (n_points,)
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    y_smooth = np.zeros_like(y)
    for i in range(len(y)):
        spl = interpolate.UnivariateSpline(t, y[i], s=lamb)
        y_smooth[i] = spl(t)
    return y_smooth.squeeze()


# ============================================================
# FunctionalData：函数型数据对象
# ============================================================

class FunctionalData:
    """函数型数据对象。

    表示在共同网格上观测的 N 条函数曲线。

    Parameters
    ----------
    grid_points : ndarray, shape (n_points,)
        共同评估网格。
    values : ndarray, shape (n_samples, n_points)
        网格上的函数值。
    """
    def __init__(self, grid_points, values):
        self.grid_points = np.asarray(grid_points, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[np.newaxis, :]
        self.n_samples, self.n_points = self.values.shape

    def mean(self):
        """样本均值函数 μ(t) = (1/N) Σ X_i(t)"""
        return FunctionalData(
            self.grid_points, self.values.mean(axis=0)
        )

    def center(self):
        """返回中心化后的函数型数据"""
        return FunctionalData(
            self.grid_points,
            self.values - self.values.mean(axis=0)
        )

    def plot(self, n_curves=10, ax=None, alpha=0.3, color='steelblue', **kwargs):
        """绘制函数型数据的前 n_curves 条曲线 + 均值函数。"""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        for i in range(min(n_curves, self.n_samples)):
            ax.plot(self.grid_points, self.values[i], alpha=alpha,
                    color=color, **kwargs)
        ax.plot(self.grid_points, self.values.mean(axis=0), 'k-',
                linewidth=2, label='Mean')
        ax.set_xlabel('t')
        ax.set_ylabel('X(t)')
        ax.legend()
        return ax


# ============================================================
# FPCA 函数型主成分分析（from scratch）
# ============================================================

class FPCA:
    """函数型主成分分析。

    基于协方差矩阵特征分解实现。
    假设所有函数在共同稠密网格上观测。

    Parameters
    ----------
    n_components : int or None
        保留的主成分数。None = 保留所有。
    """
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """拟合 FPCA。

        Parameters
        ----------
        X : FunctionalData
            函数型数据。

        Returns
        -------
        self : FPCA
        """
        self.grid_points_ = X.grid_points.copy()
        self.n_samples_ = X.n_samples
        self.mean_ = X.mean()

        # 中心化
        Z = X.center().values                     # (N, P)

        # 网格间距（均匀网格假设）
        self.delta_ = np.mean(np.diff(self.grid_points_))

        # --- 协方差矩阵 ---
        # V(s, t) = (1/N) Σ Z_i(s) Z_i(t),  shape (P, P)
        V = (Z.T @ Z) / self.n_samples_

        # --- 特征分解 ---
        eigenvalues, eigenvectors = linalg.eigh(V)

        # 降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]        # 列向量为特征向量

        # 确定成分数
        K = len(eigenvalues)
        if self.n_components is not None:
            K = min(self.n_components, K)

        self.eigenvalues_ = eigenvalues[:K]
        # 特征函数：ξ_k(t_j) = v_k(t_j) / √Δt
        # 保证 ∫ ξ_k(t)² dt = Σ ξ_k(t_j)² Δt = 1
        self.components_ = eigenvectors[:, :K].T / np.sqrt(self.delta_)
        # components_[k] 为第 k 个特征函数在网格上的值, shape (K, P)

        # 方差解释比例
        total_var = max(np.sum(eigenvalues), 1e-15)
        self.explained_variance_ratio_ = (
            self.eigenvalues_ / total_var
        )
        self.cumulative_variance_ratio_ = np.cumsum(
            self.explained_variance_ratio_
        )

        # FPC 得分：f_{ik} = ∫ Z_i(t) ξ_k(t) dt ≈ √Δt · Σ_j Z_i(t_j) v_k(t_j)
        self.scores_ = (Z @ eigenvectors[:, :K]) * np.sqrt(self.delta_)

        return self

    def transform(self, X_new):
        """将新函数型数据投影到 FPC 基上。

        Parameters
        ----------
        X_new : FunctionalData
            新函数型数据。

        Returns
        -------
        scores : ndarray, shape (n_new, n_components)
        """
        Z = X_new.values - self.mean_.values
        # ∫ Z_i(t) ξ_k(t) dt ≈ Σ_j Z_i(t_j) ξ_k(t_j) Δt
        return Z @ (self.components_.T * self.delta_)

    def cumulative_variance(self):
        """返回累积方差解释比例。"""
        return self.cumulative_variance_ratio_

    def plot_modes(self, n_modes=3, n_std=2, ax=None):
        """绘制变异模态：mean ± n_std × √λ_k × ξ_k(t)

        Parameters
        ----------
        n_modes : int
            绘制的模态数。
        n_std : float
            偏离均值的标准差倍数。
        """
        n_plot = min(n_modes, self.n_components)
        if ax is None:
            _, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 3))
            if n_plot == 1:
                axes = [axes]
        else:
            axes = [ax] * n_plot

        for k in range(n_plot):
            offset = np.sqrt(self.eigenvalues_[k]) * self.components_[k]
            axes[k].plot(self.grid_points_, self.mean_.values, 'k-',
                         label='Mean', linewidth=2)
            axes[k].plot(self.grid_points_,
                         self.mean_.values + n_std * offset,
                         'r--', label=f'+{n_std}√λ_{k+1}')
            axes[k].plot(self.grid_points_,
                         self.mean_.values - n_std * offset,
                         'b--', label=f'-{n_std}√λ_{k+1}')
            axes[k].set_title(
                f'FPC {k+1}  ({self.explained_variance_ratio_[k]:.1%})'
            )
            axes[k].set_xlabel('t')
            axes[k].legend(fontsize=8)

        return ax


# ============================================================
# FLMRegressor 函数型线性模型（scalar-on-function）
# ============================================================

class FLMRegressor:
    """函数型线性回归模型。

    模型：
        Y_i = α + ∫_T X_i(t) β(t) dt + ε_i

    β(t) 在 FPCA 特征基上展开：β(t) = Σ b_k ξ_k(t)
    通过惩罚最小二乘估计。

    Parameters
    ----------
    n_components : int or None
        FPCA 基的截断数。
    lamb : float
        岭惩罚系数。
    """
    def __init__(self, n_components=None, lamb=0.0):
        self.n_components = n_components
        self.lamb = lamb

    def fit(self, X, y):
        """拟合 FLM。

        Parameters
        ----------
        X : FunctionalData
            函数型预测变量。
        y : ndarray, shape (n_samples,)
            标量响应变量。

        Returns
        -------
        self : FLMRegressor
        """
        # Step 1: FPCA 降维
        self.fpca_ = FPCA(n_components=self.n_components)
        self.fpca_.fit(X)
        scores = self.fpca_.scores_               # (N, K)

        # Step 2: 在 FPC 基下估计回归系数
        K = scores.shape[1]
        y_mean = y.mean()
        y_c = y - y_mean                          # 中心化响应

        if self.lamb > 0 and K > 0:
            # 岭回归：b = (S^T S + λ I_K)^{-1} S^T y_c
            I = np.eye(K)
            b = linalg.solve(
                scores.T @ scores + self.lamb * I,
                scores.T @ y_c
            )
        elif K > 0:
            b, *_ = linalg.lstsq(scores, y_c)
        else:
            b = np.array([])

        self.alpha_ = y_mean
        self.coef_ = b                            # β_k 在 FPC 基下的系数

        # 重建系数函数 β(t)
        if len(b) > 0:
            self.beta_ = self.fpca_.components_.T @ b
        else:
            self.beta_ = np.zeros_like(X.grid_points)

        # 拟合值
        self.fitted_values_ = y_mean + (
            scores @ b if len(b) > 0 else np.zeros(len(y))
        )

        return self

    def predict(self, X_new):
        """对新函数型数据预测。

        Parameters
        ----------
        X_new : FunctionalData
            新函数型预测变量。

        Returns
        -------
        y_pred : ndarray, shape (n_new,)
        """
        scores_new = self.fpca_.transform(X_new)
        alpha = self.alpha_
        coef = self.coef_
        return alpha + (
            scores_new @ coef if len(coef) > 0 else np.zeros(len(scores_new))
        )


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # --- 生成合成函数型数据 ---
    t = np.linspace(0, 4 * np.pi, 200)
    n_samples = 100

    # 模拟的三条特征函数（已正交归一化）
    xi_1 = np.sin(t) / np.sqrt(np.trapz(np.sin(t)**2, t))
    xi_2 = np.cos(t) / np.sqrt(np.trapz(np.cos(t)**2, t))
    xi_3 = np.sin(2 * t) / np.sqrt(np.trapz(np.sin(2 * t)**2, t))
    xi_true = np.array([xi_1, xi_2, xi_3])

    # 真实特征值
    lam_true = np.array([4.0, 2.0, 0.8])

    # 生成 FPC 得分
    f_scores = np.random.randn(n_samples, 3) * np.sqrt(lam_true)

    # 均值函数
    mu = 10 + 2 * np.sin(t / 2)

    # 生成曲线：X_i(t) = μ(t) + Σ f_{ik} ξ_k(t)
    X_clean = mu + f_scores @ xi_true            # (N, P)

    # 添加观测噪声
    noise_std = 0.3
    X_noisy = X_clean + noise_std * np.random.randn(n_samples, len(t))

    X = FunctionalData(t, X_noisy)

    # --- 平滑 ---
    X_smooth_vals = smooth_data(X_noisy, t, lamb=1.0)
    X_smooth = FunctionalData(t, X_smooth_vals)

    # --- FPCA ---
    print("=" * 50)
    print("FPCA 结果")
    print("=" * 50)

    fpca = FPCA(n_components=5)
    fpca.fit(X_smooth)

    print(f"估计特征值 (前5): {np.round(fpca.eigenvalues_, 4)}")
    print(f"真实特征值:       {lam_true}")
    print(f"\n累积方差比例 (前5): "
          f"{np.round(fpca.cumulative_variance_ratio_, 4)}")

    # 绘制前 2 个模态
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fpca.plot_modes(n_modes=2, ax=axes)
    plt.tight_layout()
    plt.savefig(r"E:/wuyi/数学建模半自动/research-assistant/outputs/fpca_modes.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- 函数型线性回归 ---
    print("\n" + "=" * 50)
    print("FLM 结果")
    print("=" * 50)

    # 真实的系数函数 β(t)
    beta_true = np.sin(t) + 0.3 * np.sin(3 * t)

    # 响应 Y = α + ∫ X(t)β(t)dt + ε
    integrals = np.trapz(X_clean * beta_true, t, axis=1)
    alpha_true = 2.0
    y = alpha_true + integrals + 0.5 * np.random.randn(n_samples)

    # 拆分训练集 / 测试集
    n_train = 70
    X_train = FunctionalData(t, X_smooth_vals[:n_train])
    X_test = FunctionalData(t, X_smooth_vals[n_train:])
    y_train, y_test = y[:n_train], y[n_train:]

    # 训练 FLM
    flm = FLMRegressor(n_components=5, lamb=0.1)
    flm.fit(X_train, y_train)

    # 预测
    y_pred_train = flm.predict(X_train)
    y_pred_test = flm.predict(X_test)

    def r2(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2) / np.sum(
            (y_true - y_true.mean())**2
        )

    print(f"训练 R² = {r2(y_train, y_pred_train):.4f}")
    print(f"测试 R² = {r2(y_test, y_pred_test):.4f}")

    # 绘制 β(t) 估计
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, beta_true, 'k-', linewidth=2, label=r'True $\beta(t)$')
    ax.plot(t, flm.beta_, 'r--', linewidth=2,
            label=r'Estimated $\beta(t)$')
    ax.set_xlabel('t')
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        r"E:/wuyi/数学建模半自动/research-assistant/outputs/flm_beta.png",
        dpi=150, bbox_inches='tight'
    )
    plt.close()

    print("\n结果已保存至 outputs/fpca_modes.png 和 outputs/flm_beta.png")
```


### 基于 scikit-fda 的生产用法

```python
# scikit-fda 提供了完整的函数型数据分析工具箱
# 安装：pip install scikit-fda

import skfda
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.operators import LinearDifferentialOperator
from skfda.preprocessing.registration import LeastSquaresRegistration
from skfda.ml.regression import LinearRegression

# 1. 创建 FDataGrid
fd = skfda.FDataGrid(X_smooth_vals, t)

# 2. 配准（对齐相位变异）
#    使用 SRSF 弹性对齐（需 fdasrsf 或 skfda 的 registration 模块）
#    registration = LeastSquaresRegistration()
#    fd_reg = registration.fit_transform(fd)

# 3. FPCA
fpca_sk = FPCA(n_components=5)
fpca_sk.fit(fd)
scores = fpca_sk.transform(fd)
print(f"Explained variance: {fpca_sk.explained_variance_ratio_}")

# 4. 函数型线性回归（scalar-on-function）
from skfda.ml.regression import LinearRegression as FLMRegression
flm_sk = FLMRegression()
flm_sk.fit(fd, y)
y_pred_sk = flm_sk.predict(fd)

# 5. 函数型共变模型（function-on-function concurrent）
#    Y(t) = α(t) + X(t)β(t) + ε(t)
#    skfda 的 concurrent 模型通过 FRegression 实现
from skfda.ml.regression import FRegression
# 注意：skfda 的 FRegression 也支持 function-to-function 回归

# 6. 稀疏 FPCA（PACE 方法）
#    method='pace' 支持不规则稀疏数据
# fpca_pace = FPCA(n_components=3, method='PACE')
# fpca_pace.fit(fd_sparse)
```

### 基于 fdasrsf 的生产用法

```python
# fdasrsf 专攻 SRSF 弹性函数型数据分析
# 安装：pip install fdasrsf

from fdasrsf import fdacurve, curve_srvf_align

# 弹性对齐多条曲线
# obj = fdacurve()
# obj.srvf_align()  # 去除相位变异

# 弹性 FPCA
# obj.fdacurve() 中包含弹性的 FPCA
# 返回的特征函数考虑了幅度和相位分离
```

## 参考文献

1. Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer. DOI: 10.1007/b98888
2. Ramsay, J. O., Hooker, G., & Graves, S. (2009). *Functional Data Analysis with R and MATLAB*. Springer.
3. Yao, F., Müller, H.-G., & Wang, J.-L. (2005). Functional data analysis for sparse longitudinal data. *Journal of the American Statistical Association*, 100(470), 577–590. DOI: 10.1198/016214504000001745
4. Ferraty, F., & Vieu, P. (2006). *Nonparametric Functional Data Analysis: Theory and Practice*. Springer.
5. Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121.
6. Wang, J.-L., Chiou, J.-M., & Müller, H.-G. (2016). Functional data analysis. *Annual Review of Statistics and Its Application*, 3, 257–295.
7. Tucker, J. D., Wu, W., & Srivastava, A. (2013). Generative models for functional data using phase and amplitude separation. *Computational Statistics & Data Analysis*, 61, 50–66.
