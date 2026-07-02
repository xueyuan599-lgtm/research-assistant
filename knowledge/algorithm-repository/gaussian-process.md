# Gaussian Process — 高斯过程

- **来源**: Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- **DOI**: 10.7551/mitpress/3206.001.0001
- **方法类别**: 贝叶斯方法 / 非参数模型 / 回归

## 数学设定

### 定义

高斯过程是一个随机过程，任意有限个函数值的联合分布服从高斯分布。记作：

$$
f(x) \sim \mathcal{GP}(m(x),\, k(x, x'))
$$

其中：
- $m(x) = \mathbb{E}[f(x)]$ 为均值函数（中心化后通常取 0）
- $k(x, x') = \text{Cov}(f(x), f(x'))$ 为核函数（协方差函数）

对于一个训练集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n,\; x_i \in \mathbb{R}^d,\; y_i \in \mathbb{R}$，GP 假设：

$$
y_i = f(x_i) + \varepsilon_i,\quad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)
$$

### 核函数（协方差函数）

核函数编码了关于函数 $f$ 的光滑性、周期性和其他结构特性的先验信念。

---

**径向基核（RBF / SE）**

$$
k(r) = \sigma^2 \exp\left(-\frac{r^2}{2\ell^2}\right), \quad r = |x - x'|
$$

无限可微，对应极其光滑的函数。

---

**Matérn 3/2**

$$
k(r) = \sigma^2\left(1 + \frac{\sqrt{3}\,r}{\ell}\right) \exp\left(-\frac{\sqrt{3}\,r}{\ell}\right)
$$

一次可微（实际样本函数连续不可微），过程 $C^1$。

---

**Matérn 5/2**

$$
k(r) = \sigma^2\left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}\,r}{\ell}\right)
$$

两次可微，过程 $C^2$。

---

**有理二次核（Rational Quadratic）**

$$
k(r) = \sigma^2\left(1 + \frac{r^2}{2\alpha\ell^2}\right)^{-\alpha}
$$

可视为不同长度尺度的 RBF 的加权和（尺度混合），$\alpha \to \infty$ 时退化为 RBF。

---

**周期核（Periodic）**

$$
k(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2\left(\frac{\pi|x-x'|}{p}\right)}{\ell^2}\right)
$$

适用于周期模式，$p$ 控制周期，$\ell$ 控制周期内光滑程度。

---

**线性核**

$$
k(x, x') = \sigma_b^2 + \sigma_v^2(x - c)(x' - c)
$$

等价于贝叶斯线性回归。

---

**白噪声核**

$$
k(x, x') = \sigma_n^2 \cdot \delta_{x, x'}
$$

独立噪声项，一般加在其他核上。

---

**核的组合**

封闭性：若 $k_1, k_2$ 是有效核，则：

- $k(x, x') = k_1(x, x') + k_2(x, x')$（求和）
- $k(x, x') = k_1(x, x') \cdot k_2(x, x')$（乘积）
- $k(x, x') = c \cdot k_1(x, x')$（缩放）

都是有效核。这一性质使得 GP 可以灵活建模复杂结构信号（如：RBF + Periodic 建模趋势+季节性）。

### GP 回归（GPR）

**先验**

$$
f \mid X \sim \mathcal{N}(0,\, K(X, X))
$$

其中 $K_{ij} = k(x_i, x_j)$。

**联合分布**

$$
\begin{bmatrix}
y \\ f_*
\end{bmatrix}
\sim \mathcal{N}\left(
0,\,
\begin{bmatrix}
K + \sigma_n^2 I & K_*^\top \\
K_* & K_{**}
\end{bmatrix}
\right)
$$

其中：
- $K = K(X, X) \in \mathbb{R}^{n \times n}$
- $K_* = K(X_*, X) \in \mathbb{R}^{m \times n}$
- $K_{**} = K(X_*, X_*) \in \mathbb{R}^{m \times m}$

**后验预测分布**

$$
f_* \mid X, y, X_* \sim \mathcal{N}(\bar{f}_*,\, \text{Cov}(f_*))
$$

$$
\bar{f}_* = K_* (K + \sigma_n^2 I)^{-1} y
$$

$$
\text{Cov}(f_*) = K_{**} - K_* (K + \sigma_n^2 I)^{-1} K_*^\top
$$

预测均值是训练观测 $y$ 的线性组合（权重由核决定），预测方差反映数据稀疏区域的不确定性。

**对数边际似然（证据）**

$$
\log p(y \mid X) = -\frac{1}{2} y^\top (K + \sigma_n^2 I)^{-1} y
                  - \frac{1}{2} \log |K + \sigma_n^2 I|
                  - \frac{n}{2} \log 2\pi
$$

三项分别解释为：数据拟合项、复杂度惩罚项、归一化常数。

### 超参数学习

通过最大化对数边际似然来学习核超参数 $\theta = \{\ell, \sigma^2, \sigma_n^2, \dots\}$。

$$
\frac{\partial}{\partial \theta_j} \log p(y \mid X, \theta) =
\frac{1}{2} y^\top K^{-1} \frac{\partial K}{\partial \theta_j} K^{-1} y
- \frac{1}{2} \operatorname{tr}\left(K^{-1} \frac{\partial K}{\partial \theta_j}\right)
$$

采用共轭梯度法 / L-BFGS 等优化器进行多起点优化以避免局部最优。

### 计算复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 训练（Cholesky 分解） | $\mathcal{O}(n^3)$ | $K + \sigma_n^2 I = LL^\top$ |
| 存储 | $\mathcal{O}(n^2)$ | 核矩阵 $n \times n$ |
| 单点预测 | $\mathcal{O}(n)$ | $L \backslash (L \backslash y)$ |
| 预测方差（单点） | $\mathcal{O}(n)$ | 需解一次三角系统 |

## 关键假设

- **高斯似然**：观测噪声服从独立同分布高斯分布 $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$
- **函数光滑性由核决定**：核函数正确编码了目标函数的先验光滑程度
- **平稳性（使用平稳核时）**：协方差仅依赖于 $|x - x'|$，与绝对位置无关（RBF、Matérn 等）
- **马尔可夫性（非必要）**：GP 本身非马尔可夫，但特定核（如 Matérn）可以与状态空间模型建立联系
- **联合高斯性**：任意有限点集的函数值服从联合高斯——这是 GP 定义的本质假设
- **无模型偏差**：GP 是一种非参数方法，模型复杂度随数据自适应增长

## 适用场景

- **中小规模数据集**（$n < 10^4$）：GP 在样本量适中时效果好且提供 uncertainty
- **需要不确定性量化**：物理仿真、可靠性分析、主动学习、安全关键系统
- **贝叶斯优化**：GP 是 BO 的标准代理模型（EI、PI、UCB 等采集函数）
- **时空建模**：用核结构分离时间/空间相关性（如 STGP）
- **试验设计**：计算机实验的代理模型（surrogate / emulator）
- **多任务/多输出学习**：多输出 GP 建模任务间相关性
- **符号回归 + 核搜索**：自动组合核结构（Duvenaud, 2014）

### 不适用

- **大规模数据**（$n > 10^4$ 无近似）：需稀疏 GP（SVGP、FITC）或 KISS-GP
- **高维输入**（$d > 20$）：维度诅咒使样本急剧稀疏，核函数难以分辨远近
- **仅需点预测**：GBDT、神经网络更快且扩展性更好
- **流数据/在线学习**：标准 GP 是批量模型（在线近似如 OGP 有限适用）
- **离散输入/分类特征为主**：核函数在离散空间上的定义不自然

## 实现要点

### 数值稳定性
1. **Cholesky 分解**：始终使用 $L = \operatorname{cholesky}(K + \sigma_n^2 I)$ 而非直接求逆
2. **Jitter 项**：在核矩阵对角加小量 $10^{-6}$ 防止条件数过大导致分解失败
3. **避免直接求 $K^{-1}$**：用 $L \backslash (L \backslash y)$ 解线性系统而非显式求逆
4. **对数行列式计算**：$\log|K + \sigma_n^2 I| = 2\sum_i \log L_{ii}$（Cholesky 副产品）

### 核函数选择
| 数据结构 | 推荐核 | 说明 |
|---------|--------|------|
| 光滑连续 | RBF / Matérn 5/2 | 默认首选 Matérn 5/2（比 RBF 更现实） |
| 粗糙连续 | Matérn 3/2 | 样本函数一次可微 |
| 多尺度 | Rational Quadratic | 等价于 RBF 的尺度混合 |
| 周期模式 | Periodic + RBF | 季节项 + 趋势项 |
| 线性趋势 | Linear 或 + Linear | 允许外推线性趋势 |
| 异方差噪声 | 学习 $\sigma_n^2(x)$ | 需要异方差 GP 变体 |

### 超参数优化
- **多起点**：边际似然非凸，需多组初始值（5-20 个随机起点）
- **ARD（自动相关性确定）**：每个维度一个独立长度尺度 $\ell_d$，优化后 $\ell_d$ 大表示该维度不重要
- **标准化输入**：建议将每个特征标准化到零均值单位方差，使长度尺度的优化更稳定
- **梯度辅助**：边际似然梯度可解析计算，用 L-BFGS 高效优化

### 扩展到大规模数据
| 方法 | 复杂度 | 核心思想 |
|------|--------|---------|
| FITC（稀疏伪输入） | $\mathcal{O}(nm^2)$ | $m \ll n$ 个诱导点近似全核矩阵 |
| SVGP（Hensman, 2013） | $\mathcal{O}(nm^2)$ | 变分推断 + 小批量 SGD |
| KISS-GP（Wilson, 2015） | $\mathcal{O}(n)$ | 核插值 + Toeplitz/网格结构 |
| 随机特征展开 | $\mathcal{O}(n m^2)$ | 随机傅里叶特征近似核 |

### 非高斯似然
- **GP 分类**：用 Laplace 近似或期望传播（EP）处理非高斯似然
- **计数数据**：GP 泊松回归（GPPR），使用对数链接函数
- **鲁棒 GP**：Student-t 似然替代高斯似然以处理异常值

### 代码

```python
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Kernels:
    """常用核函数集合"""
    
    @staticmethod
    def RBF(X1, X2, length_scale=1.0, sigma_f=1.0):
        """径向基核 (Squared Exponential)"""
        sqdist = np.sum(X1**2, axis=1, keepdims=True) \
               + np.sum(X2**2, axis=1) \
               - 2 * X1 @ X2.T
        return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)
    
    @staticmethod
    def Matern32(X1, X2, length_scale=1.0, sigma_f=1.0):
        """Matérn 3/2 核"""
        sqdist = np.sum(X1**2, axis=1, keepdims=True) \
               + np.sum(X2**2, axis=1) \
               - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 0))
        return sigma_f**2 * (1 + np.sqrt(3) * r / length_scale) \
                          * np.exp(-np.sqrt(3) * r / length_scale)
    
    @staticmethod
    def Matern52(X1, X2, length_scale=1.0, sigma_f=1.0):
        """Matérn 5/2 核"""
        sqdist = np.sum(X1**2, axis=1, keepdims=True) \
               + np.sum(X2**2, axis=1) \
               - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 0))
        sqrt5_r = np.sqrt(5) * r / length_scale
        return sigma_f**2 * (1 + sqrt5_r + 5 * r**2 / (3 * length_scale**2)) \
                          * np.exp(-sqrt5_r)
    
    @staticmethod
    def RationalQuadratic(X1, X2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
        """有理二次核"""
        sqdist = np.sum(X1**2, axis=1, keepdims=True) \
               + np.sum(X2**2, axis=1) \
               - 2 * X1 @ X2.T
        return sigma_f**2 * (1 + sqdist / (2 * alpha * length_scale**2)) ** (-alpha)
    
    @staticmethod
    def Periodic(X1, X2, length_scale=1.0, sigma_f=1.0, period=1.0):
        """周期核"""
        dist = np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :])
        sin_term = np.sin(np.pi * dist / period) ** 2
        return sigma_f**2 * np.exp(-2 * np.sum(sin_term, axis=-1) / length_scale**2)
    
    @staticmethod
    def WhiteKernel(X1, X2, sigma_n=1.0):
        """白噪声核"""
        if X1.shape[0] != X2.shape[0] or not np.allclose(X1, X2):
            raise ValueError("WhiteKernel is only defined for identical X1 and X2")
        return sigma_n**2 * np.eye(X1.shape[0])


class GaussianProcess:
    """高斯过程回归 — 从零实现
    
    支持多种核函数、超参数优化、后验预测。
    
    Parameters
    ----------
    kernel : str, default='RBF'
        核函数类型：'RBF', 'Matern32', 'Matern52', 'RationalQuadratic'
    length_scale : float, default=1.0
        核长度尺度
    sigma_f : float, default=1.0
        核幅度（信号标准差）
    sigma_n : float, default=1.0
        噪声标准差
    alpha : float, default=1.0
        有理二次核的 alpha 参数
    period : float, default=1.0
        周期核的周期参数
    n_restarts : int, default=5
        优化超参数时随机重启次数
    random_state : int or None, default=None
        随机种子
    """
    
    def __init__(self, kernel='RBF', length_scale=1.0, sigma_f=1.0,
                 sigma_n=1.0, alpha=1.0, period=1.0,
                 n_restarts=5, random_state=None):
        self.kernel_name = kernel
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.alpha_kernel = alpha
        self.period = period
        self.n_restarts = n_restarts
        self.random_state = random_state
        
        self._kernel_map = {
            'RBF': Kernels.RBF,
            'Matern32': Kernels.Matern32,
            'Matern52': Kernels.Matern52,
            'RationalQuadratic': Kernels.RationalQuadratic,
        }
        
        if kernel not in self._kernel_map:
            raise ValueError(f"Unknown kernel: {kernel}. Choose from {list(self._kernel_map.keys())}")
        
    def _kernel(self, X1, X2, length_scale, sigma_f, sigma_n=0):
        """计算核矩阵"""
        k = self._kernel_map[self.kernel_name](
            X1, X2, length_scale=length_scale, sigma_f=sigma_f,
            alpha=self.alpha_kernel
        )
        if sigma_n > 0 and X1.shape[0] == X2.shape[0] and np.array_equal(X1, X2):
            k += sigma_n**2 * np.eye(X1.shape[0])
        return k
    
    def _cholesky(self, K):
        """稳定的 Cholesky 分解（加 jitter）"""
        jitter = 1e-6
        while True:
            try:
                L = cholesky(K, lower=True)
                return L
            except np.linalg.LinAlgError:
                jitter *= 10
                K += jitter * np.eye(K.shape[0])
    
    def fit(self, X, y):
        """训练 GP 模型
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            训练输入
        y : ndarray of shape (n_samples,)
            训练目标
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        
        if X.ndim == 1:
            X = X[:, None]
        
        self.X_train_ = X
        self.y_train_ = y
        
        # 构建核矩阵（含噪声项）
        K = self._kernel(X, X, self.length_scale, self.sigma_f, self.sigma_n)
        
        # Cholesky 分解
        self.L_ = self._cholesky(K)
        
        # alpha = K^{-1} y = L^{-T} L^{-1} y
        self.alpha_ = solve_triangular(
            self.L_.T,
            solve_triangular(self.L_, y, lower=True),
            lower=False
        )
        
        # 存储对数边际似然
        n = X.shape[0]
        log_det = 2 * np.sum(np.log(np.diag(self.L_)))
        self.log_marginal_likelihood_ = (
            -0.5 * y @ self.alpha_ - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        )
        
        return self
    
    def predict(self, X, return_std=True):
        """后验预测
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            测试输入
        return_std : bool, default=True
            是否返回标准差
            
        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
            预测均值
        y_std : ndarray of shape (n_samples,), optional
            预测标准差
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        
        # K_* 和 K_{**}
        K_s = self._kernel(X, self.X_train_, self.length_scale, self.sigma_f)
        K_ss = self._kernel(X, X, self.length_scale, self.sigma_f)
        
        # 预测均值 = K_* @ alpha
        y_mean = K_s @ self.alpha_
        
        if not return_std:
            return y_mean
        
        # 预测方差 = K_{**} - K_* K^{-1} K_*^T
        # v = L^{-1} K_*^T, 即 L v = K_*^T
        v = solve_triangular(self.L_, K_s.T, lower=True)
        y_var = np.diag(K_ss) - np.sum(v**2, axis=0)
        y_var = np.maximum(y_var, 0)  # 防止数值误差导致负方差
        y_std = np.sqrt(y_var)
        
        return y_mean, y_std
    
    def log_marginal_likelihood(self, params, X, y):
        """负对数边际似然（用于优化）"""
        length_scale, sigma_f, sigma_n = params
        
        K = self._kernel(X, X, length_scale, sigma_f, sigma_n)
        n = X.shape[0]
        
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10  # 无效参数
        
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        return 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
    
    def optimize(self, X, y):
        """最大化对数边际似然学习超参数
        
        对 length_scale, sigma_f, sigma_n 进行多起点 L-BFGS 优化。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X[:, None]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        best_params = None
        best_nlml = np.inf
        
        # 初始起点（用户提供）
        initial_guesses = [
            [self.length_scale, self.sigma_f, self.sigma_n]
        ]
        
        # 随机起点
        for _ in range(self.n_restarts):
            guess = [
                10**np.random.uniform(-2, 2),   # length_scale
                10**np.random.uniform(-1, 1),   # sigma_f
                10**np.random.uniform(-2, 0),   # sigma_n
            ]
            initial_guesses.append(guess)
        
        bounds = [
            (1e-3, 1e3),   # length_scale
            (1e-3, 1e3),   # sigma_f
            (1e-6, 10),    # sigma_n
        ]
        
        for theta0 in initial_guesses:
            res = minimize(
                self.log_marginal_likelihood,
                theta0,
                args=(X, y),
                method='L-BFGS-B',
                bounds=bounds
            )
            if res.fun < best_nlml:
                best_nlml = res.fun
                best_params = res.x
        
        if best_params is not None:
            self.length_scale, self.sigma_f, self.sigma_n = best_params
            # 用最优参数重新 fit
            self.fit(X, y)
        
        return self


class GPPlot:
    """GP 1D 回归可视化辅助"""
    
    @staticmethod
    def plot_gp(gp, X_train, y_train, X_test, 
                title="Gaussian Process Regression",
                ax=None, show_legend=True):
        """绘制 GP 后验均值和 95% 置信区间"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        y_mean, y_std = gp.predict(X_test)
        
        # 置信区间
        y_upper = y_mean + 1.96 * y_std
        y_lower = y_mean - 1.96 * y_std
        
        ax.plot(X_test, y_mean, 'b-', label='Predictive mean', linewidth=2)
        ax.fill_between(X_test.ravel(), y_lower, y_upper,
                        color='blue', alpha=0.2, label='95% confidence')
        ax.scatter(X_train, y_train, c='red', s=40, zorder=5,
                   label='Training data')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        if show_legend:
            ax.legend()
        
        return ax
    
    @staticmethod
    def plot_prior_samples(gp_kernel, n_samples=5, x_lim=(-5, 5),
                           title="GP Prior Samples", ax=None):
        """从 GP 先验中采样函数"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        X = np.linspace(x_lim[0], x_lim[1], 200)[:, None]
        K = gp_kernel(X, X)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
        
        for i in range(n_samples):
            f_sample = L @ np.random.randn(len(X))
            ax.plot(X, f_sample, lw=1.5, alpha=0.8, label=f'Sample {i+1}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(title)
        ax.legend()
        return ax


# =====================
# 使用示例：1D 函数拟合
# =====================
if __name__ == "__main__":
    np.random.seed(42)
    
    # 生成训练数据：sin 函数 + 噪声
    X_train = np.random.uniform(-4, 4, 20)
    y_train = np.sin(X_train) + 0.1 * np.random.randn(20)
    
    # 测试点（密集网格）
    X_test = np.linspace(-6, 6, 200)
    
    # --- 示例 1: 手动指定超参数 ---
    print("=" * 60)
    print("Example 1: Manual hyperparameters (RBF)")
    gp_manual = GaussianProcess(kernel='RBF', length_scale=0.8,
                                sigma_f=1.2, sigma_n=0.15, random_state=42)
    gp_manual.fit(X_train, y_train)
    y_mean, y_std = gp_manual.predict(X_test)
    nlml = gp_manual.log_marginal_likelihood(
        [gp_manual.length_scale, gp_manual.sigma_f, gp_manual.sigma_n],
        X_train[:, None], y_train
    )
    print(f"  Length scale: {gp_manual.length_scale:.3f}")
    print(f"  Signal std:   {gp_manual.sigma_f:.3f}")
    print(f"  Noise std:    {gp_manual.sigma_n:.3f}")
    print(f"  NLML:         {nlml:.3f}")
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    GPPlot.plot_gp(gp_manual, X_train, y_train, X_test[:, None],
                   title="GP Regression with RBF Kernel (Manual)", ax=ax1)
    plt.tight_layout()
    plt.show()
    
    # --- 示例 2: 超参数优化 ---
    print("\n" + "=" * 60)
    print("Example 2: Optimized hyperparameters (Matern 5/2)")
    gp_opt = GaussianProcess(kernel='Matern52', n_restarts=10, random_state=42)
    gp_opt.optimize(X_train[:, None], y_train)
    print(f"  Optimized length scale: {gp_opt.length_scale:.3f}")
    print(f"  Optimized signal std:   {gp_opt.sigma_f:.3f}")
    print(f"  Optimized noise std:    {gp_opt.sigma_n:.3f}")
    print(f"  Log marginal likelihood: {gp_opt.log_marginal_likelihood_:.3f}")
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    GPPlot.plot_gp(gp_opt, X_train, y_train, X_test[:, None],
                   title="GP Regression with Matérn 5/2 (Optimized)", ax=ax2)
    plt.tight_layout()
    plt.show()
    
    # --- 示例 3: 不同核的比较 ---
    print("\n" + "=" * 60)
    print("Example 3: Kernel comparison")
    kernels_to_test = ['RBF', 'Matern32', 'Matern52', 'RationalQuadratic']
    
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    for ax, kernel_name in zip(axes3.ravel(), kernels_to_test):
        gp = GaussianProcess(kernel=kernel_name, n_restarts=5, random_state=42)
        gp.optimize(X_train[:, None], y_train)
        GPPlot.plot_gp(gp, X_train, y_train, X_test[:, None],
                       title=f"{kernel_name} (ℓ={gp.length_scale:.2f})",
                       ax=ax, show_legend=False)
    plt.tight_layout()
    plt.show()
    
    # --- 示例 4: 先验采样 ---
    print("\n" + "=" * 60)
    print("Example 4: Prior samples from different kernels")
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 6))
    prior_kernels = [
        (lambda X1, X2: Kernels.RBF(X1, X2, ls, sf), f"RBF (ℓ={ls})")
        for ls, sf in [(0.5, 1.0), (2.0, 1.0), (0.5, 0.5), (1.0, 1.5)]
    ]
    for ax, (kfn, title) in zip(axes4.ravel(), prior_kernels):
        GPPlot.plot_prior_samples(kfn, n_samples=5, title=title, ax=ax)
    plt.tight_layout()
    plt.show()
```

### 基于 scikit-learn / GPyTorch 的生产用法

```python
# =====================
# scikit-learn 用法
# =====================
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, DotProduct
from sklearn.model_selection import cross_val_score

# 构造组合核：RBF + WhiteKernel（含噪声）
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10))

gp_sklearn = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42,
    alpha=0.0,  # 噪声已在 WhiteKernel 中包含
)

gp_sklearn.fit(X_train[:, None], y_train)
y_mean, y_std = gp_sklearn.predict(X_test[:, None], return_std=True)

print(f"Learned kernel: {gp_sklearn.kernel_}")
print(f"Log-marginal likelihood: {gp_sklearn.log_marginal_likelihood(gp_sklearn.kernel_.theta):.3f}")

# =====================
# GPyTorch 用法（GPU/大规模）
# =====================
# import torch
# import gpytorch
#
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel()
#         )
#     def forward(self, x):
#         mean = self.mean_module(x)
#         covar = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)
#
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(torch.Tensor(X_train), torch.Tensor(y_train), likelihood)
# 训练循环：使用 PyTorch 优化器 + marginal likelihood
```

## 参考文献
Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Lecture Notes in Statistics, 118. Springer.

Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *NeurIPS*.

Duvenaud, D. (2014). *Automatic Model Construction with Gaussian Processes*. PhD Thesis, University of Cambridge.

Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian Processes for Big Data. *UAI*.

Wilson, A. G. & Nickisch, H. (2015). Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP). *ICML*.
