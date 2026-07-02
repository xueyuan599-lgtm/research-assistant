# Regression Discontinuity Design — 断点回归设计

- **来源**: Thistlethwaite, D. L., & Campbell, D. T. (1960). Regression-discontinuity analysis: An alternative to the ex post facto experiment. *Journal of Educational Psychology*, 51(6), 309–317.
- **DOI**: 10.1037/h0044319
- **补充**: Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281–355.
- **方法类别**: 因果推断 / 准实验方法

---

## 数学设定

### 基本框架

断点回归设计（RDD）利用**分配规则中的已知断点**来识别因果效应。核心思想：在断点（cutoff）附近，个体是"近乎随机"地进入处理组或控制组。

设 $X_i$ 为**驱动变量（running variable）**，$c$ 为已知断点，$T_i$ 为处理指示变量，$Y_i$ 为结果变量。

### Sharp RDD（精确断点回归）

处理分配规则为确定性跳跃：

$$
T_i = \mathbf{1}\{X_i \geq c\}
$$

因果效应定义为**断点处处理组与控制组的潜在结果期望之差**：

$$
\tau_{\text{SRD}} = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = c]
$$

由可观测数据识别为：

$$
\tau_{\text{SRD}} = \lim_{x \downarrow c} \mathbb{E}[Y_i \mid X_i = x] \;-\; \lim_{x \uparrow c} \mathbb{E}[Y_i \mid X_i = x]
$$

即结果变量在断点处的**双侧极限之差**。

### Fuzzy RDD（模糊断点回归）

处理概率在断点处发生跳跃，但并非确定性的 $0 \to 1$：

$$
\lim_{x \downarrow c} \Pr(T_i = 1 \mid X_i = x) \neq \lim_{x \uparrow c} \Pr(T_i = 1 \mid X_i = x)
$$

因果效应为 Wald 估计量（工具变量法，以 $Z_i = \mathbf{1}\{X_i \geq c\}$ 为工具）：

$$
\tau_{\text{FRD}} = \frac{\displaystyle\lim_{x \downarrow c} \mathbb{E}[Y_i \mid X_i = x] - \lim_{x \uparrow c} \mathbb{E}[Y_i \mid X_i = x]}
{\displaystyle\lim_{x \downarrow c} \mathbb{E}[T_i \mid X_i = x] - \lim_{x \uparrow c} \mathbb{E}[T_i \mid X_i = x]}
$$

等价于两阶段最小二乘（2SLS）：
- **第一阶段**：$T_i = \pi_0 + \pi_1 Z_i + \pi_2 (X_i - c) + \pi_3 Z_i (X_i - c) + \eta_i$
- **第二阶段**：$Y_i = \beta_0 + \tau \hat{T}_i + \beta_1 (X_i - c) + \beta_2 \hat{T}_i (X_i - c) + \varepsilon_i$

---

### 局部多项式回归

Sharp RDD 的估计通过**断点附近的局部加权回归**实现。对给定**带宽（bandwidth）** $h$ 和**核函数（kernel）** $K(\cdot)$，求解加权最小二乘：

设 $u_i = (X_i - c) / h$，权重 $w_i = K(u_i)$。最小化目标函数：

$$
\min_{\beta_0, \ldots, \beta_P, \tau, \gamma_1, \ldots, \gamma_P}
\sum_{i=1}^n K\!\left(\frac{X_i - c}{h}\right)
\Bigg[ Y_i - \beta_0 - \sum_{p=1}^P \beta_p (X_i - c)^p
      - \tau T_i - \sum_{p=1}^P \gamma_p T_i (X_i - c)^p \Bigg]^2
$$

**一阶局部线性回归**（$P = 1$，最常用）：

$$
Y_i = \beta_0 + \beta_1 (X_i - c) + \tau T_i + \gamma_1 T_i (X_i - c) + \varepsilon_i
$$

系数矩阵形式：
$$
\mathbf{X} = [\mathbf{1},\; (X-c),\; T,\; T \cdot (X-c)]
$$
$$
\boldsymbol{\beta} = [\beta_0,\; \beta_1,\; \tau,\; \gamma_1]'
$$

其中 $\tau$ 即为断点处的处理效应估计值 $\hat{\tau}_{\text{SRD}}$。

---

### 核函数加权

核函数控制不同距离观测值的权重，**断点处的观测获得最大权重**。

| 核函数 | 公式 $K(u)$ | 特点 |
|--------|-------------|------|
| 三角核（Triangular） | $(1 - \lvert u \rvert) \cdot \mathbf{1}\{\lvert u \rvert \leq 1\}$ | **边界最优**（最小化边界点AMSE），RDD 首选 |
| 均匀核（Uniform） | $\frac{1}{2} \cdot \mathbf{1}\{\lvert u \rvert \leq 1\}$ | 等权重，等价于未加权 OLS |
| Epanechnikov | $\frac{3}{4}(1 - u^2) \cdot \mathbf{1}\{\lvert u \rvert \leq 1\}$ | 渐近效率最优（内点），但边界表现不如三角核 |

---

### 最优带宽选择

带宽 $h$ 是 RDD 最关键的平滑参数。选择遵循**偏差-方差权衡**：
- $h$ 过小：方差大，估计不稳定
- $h$ 过大：偏差大，包含远离断点的观测

#### Imbens-Kalyanaraman（IK, 2012）最优带宽

IK 提出最小化**渐近均方误差（AMSE）** 的插件法带宽：

$$
h_{\text{IK}} = C_K \cdot \left[
\frac{2 \cdot \sigma^2(c)}{f(c) \cdot \big(m''_+(c)^2 + m''_-(c)^2\big)}
\right]^{1/5} \cdot n^{-1/5}
$$

其中：
- $\sigma^2(c)$ = 断点处条件方差
- $f(c)$ = 驱动变量在断点处的密度
- $m''_+(c), m''_-(c)$ = 回归函数在断点左右两侧的二阶导数
- $C_K$ = 核函数常数（三角核约为 3.47）

#### Calonico-Cattaneo-Farrell（CCF, 2014）稳健带宽

在 IK 基础上引入**偏差矫正（bias correction）** 和**稳健标准误**，是目前推荐的实践标准：
- 先使用较大带宽估计偏差
- 再从偏差矫正后的估计量计算标准误
- 对带宽选择错误具有更强的鲁棒性

#### 交叉验证带宽

在实践中也可使用留一法交叉验证（LOOCV）：
- 对每个候选 $h$，对每个观测 $i$ 在去掉 $i$ 后用带宽 $h$ 估计模型，预测 $Y_i$
- 选择使 MSE 最小的 $h$

---

## 关键假设

### （1）潜在结果的连续性（Continuity of Potential Outcomes）

核心识别假设：**潜在结果的条件期望在断点处连续**。

$$
\lim_{x \downarrow c} \mathbb{E}[Y_i(0) \mid X_i = x] = \mathbb{E}[Y_i(0) \mid X_i = c] = \lim_{x \uparrow c} \mathbb{E}[Y_i(0) \mid X_i = x]
$$
$$
\lim_{x \downarrow c} \mathbb{E}[Y_i(1) \mid X_i = x] = \mathbb{E}[Y_i(1) \mid X_i = c] = \lim_{x \uparrow c} \mathbb{E}[Y_i(1) \mid X_i = x]
$$

直观含义：如果 $X_i$ 在 $c$ 处没有其他干扰变量的同时跳跃，断点处的差值可归因于处理效应。

### （2）驱动变量不可精确操纵（No Precise Manipulation / Local Randomization）

个体不能**精确控制** $X_i$ 是否越过断点 $c$。如果个体能精确操纵 $X_i$，则断点两侧的个体将存在系统性差异，破坏识别。

可检验：McCrary（2008）密度检验 — 检查 $X_i$ 的密度函数在 $c$ 处是否连续。

### （3）排除限制（仅有处理效应在断点处跳跃）

其他影响 $Y$ 的因素在 $c$ 处不得同时发生跳跃（无同时发生的政策变化）。

可检验：协变量平衡检验 — 检查协变量在断点处是否有显著跳跃。

---

## 适用场景

| 场景 | 驱动变量 $X$ | 断点 $c$ | 处理 $T$ |
|------|-------------|----------|---------|
| 奖学金与学业表现 | 考试分数 | 录取分数线 | 获得奖学金 |
| 选举与连任优势 | 得票率 | 50% | 赢得选举 |
| 政策年龄门槛 | 年龄 | 18 / 65 岁 | 获得福利/退休 |
| 药品审批与监管 | 药品指标值 | 审批标准线 | 获批上市 |
| 教育学位效应 | 入学考试成绩 | 录取线 | 被录取 |
| 最低工资影响 | 地区工资水平 | 最低工资线 | 工资上调 |
| 医疗保险效应 | 年龄 | 65 岁 | Medicare 资格 |

### 不适用

- **精确操纵驱动变量**：当个体能精确控制 $X_i$ 时（如通过复读控制考试成绩），连续密度假设被破坏
- **多个政策同时变化**：当 $c$ 处有多个政策同时跳跃时，无法分离出单一政策的因果效应
- **样本量不足**：断点附近观测数过少（< 100）时估计不可靠
- **离散驱动变量**：$X$ 只有少数几个取值时（如仅 3–5 个整数值），极限外推难以识别
- **存在跨断点的选择效应**：个体可以选择是否进入断点邻近区域（如搬到不同政策区域）

---

## 实现要点

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 核函数 $K(\cdot)$ | **三角核（Triangular）** | 边界最优，最小化断点处 AMSE |
| 多项式阶数 $P$ | **1**（局部线性）或 **2**（局部二次） | 更高阶（$P \geq 3$）偏差大且不稳定 |
| 带宽 $h$ | IK / CCF 最优带宽 | 必须伴随敏感性分析 |
| 标准误类型 | **异方差稳健 SE（HC1 / HC2）** | 或聚类稳健 SE（若 $X$ 离散） |

### 诊断检验流程

一个可信的 RDD 研究必须通过以下全套诊断：

```
1. McCrary 密度检验  ────────────────────  检验驱动变量不可操纵
        │
2. 协变量平衡检验  ────────────────────────  检验协变量在断点处无跳跃
        │
3. 安慰剂断点检验  ────────────────────────  在其他位置无显著效应
        │
4. 安慰剂结果检验  ────────────────────────  对不应受影响的 Y 无效应
        │
5. 带宽敏感性分析  ────────────────────────  效应随 h 变化是否稳定
        │
6. 多项式阶数敏感性  ──────────────────────  P=1,2 是否结论一致
        │
7. Donut hole RDD  ───────────────────────  去掉断点附近观测后是否稳健
```

#### 具体方法说明

- **McCrary 密度检验**：构建 $X$ 的直方图，对 $\ln(\text{频数})$ 做局部线性回归，检验断点处是否存在不连续
- **安慰剂断点检验**：选 $c \pm \delta$ 作为假想的断点，重复 RDD 估计，预期无显著效应
- **安慰剂结果检验**：选择理论上不受处理的 $Y$ 变量（如预处理结果），预期无显著效应
- **Donut hole RDD**：排除断点附近小区间内的观测（如 $c \pm \varepsilon$），避免局部操纵嫌疑
- **带宽敏感性**：报告 $0.5h, 0.75h, h, 1.25h, 1.5h, 2h$ 等多组带宽下的结果，制作灵敏度图

---

## Python 实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import statsmodels.api as sm
import warnings
from copy import deepcopy


class RDD:
    """断点回归设计（Sharp RDD）— 局部多项式回归估计

    Parameters
    ----------
    cutoff : float
        驱动变量的断点值
    kernel : str, default='triangular'
        核函数类型: 'triangular', 'uniform', 'epanechnikov'
    poly_order : int, default=1
        局部多项式阶数（推荐 1 或 2）

    Attributes
    ----------
    bandwidth_ : float
        使用的带宽
    coef_ : ndarray
        模型参数估计值
    tau_ : float
        处理效应估计值
    tau_se_ : float
        处理效应的标准误
    tau_ci_ : tuple
        处理效应的 95% 置信区间
    """

    def __init__(self, cutoff=0.0, kernel='triangular', poly_order=1):
        self.cutoff = float(cutoff)
        self.kernel = kernel
        self.poly_order = poly_order
        self.bandwidth_ = None
        self.coef_ = None
        self.tau_ = None
        self.tau_se_ = None
        self.tau_ci_ = None
        self._x = None
        self._y = None
        self._results = None

    # ------------------------------------------------------------------
    # 核函数
    # ------------------------------------------------------------------
    def _kernel_weights(self, x, h):
        """计算核权重 w_i = K((x_i - c) / h)"""
        u = (np.asarray(x, dtype=float) - self.cutoff) / h
        if self.kernel == 'triangular':
            w = np.clip(1.0 - np.abs(u), 0.0, 1.0)
        elif self.kernel == 'uniform':
            w = np.where(np.abs(u) <= 1.0, 0.5, 0.0)
        elif self.kernel == 'epanechnikov':
            w = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        return w

    # ------------------------------------------------------------------
    # 设计矩阵
    # ------------------------------------------------------------------
    def _design_matrix(self, x):
        """构建设计矩阵：
        [1, (X-c), ..., (X-c)^P, T, T·(X-c), ..., T·(X-c)^P]
        """
        x = np.asarray(x, dtype=float)
        x_c = x - self.cutoff
        T = (x >= self.cutoff).astype(float)

        cols = [np.ones_like(x_c)]
        for p in range(1, self.poly_order + 1):
            cols.append(x_c ** p)          # β_p · (X-c)^p
        cols.append(T)                     # τ · T
        for p in range(1, self.poly_order + 1):
            cols.append(T * (x_c ** p))    # γ_p · T·(X-c)^p

        return np.column_stack(cols)

    # ------------------------------------------------------------------
    # 最优带宽选择（IK 2012 简化版）
    # ------------------------------------------------------------------
    def optimal_bandwidth(self, x, y, method='ik'):
        """MSE-最优带宽选择（IK 框架简化实现）

        计算插件公式：
        h_opt = C_K · [2·σ²(c) / (f(c) · (m''_+² + m''_-²))]^{1/5} · n^{-1/5}
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)

        # ---- Step 1: 估计断点处条件方差 σ²(c) ----
        # 使用全局三阶多项式残差
        X_pilot = np.column_stack([
            np.ones_like(x), x, x ** 2, x ** 3,
            (x >= self.cutoff).astype(float)
        ])
        pilot_model = sm.OLS(y, X_pilot).fit()
        resid_var = np.var(pilot_model.resid)

        # ---- Step 2: 估计驱动变量在断点处的密度 f(c) ----
        try:
            kde = gaussian_kde(x)
            f_c = kde.evaluate(self.cutoff)[0]
            if f_c <= 0:
                f_c = 1.0 / (np.max(x) - np.min(x))
        except Exception:
            f_c = 1.0 / (np.max(x) - np.min(x))

        # ---- Step 3: 估计回归函数的二阶导数 m''(c) ----
        left = x < self.cutoff
        right = x >= self.cutoff

        def _second_derivative(x_side, y_side):
            """使用四阶多项式估计在断点处的二阶导数"""
            if len(x_side) < 10:
                return None
            X_poly = np.column_stack([
                np.ones_like(x_side), x_side,
                x_side ** 2, x_side ** 3, x_side ** 4
            ])
            m = sm.OLS(y_side, X_poly).fit()
            c = self.cutoff
            # m''(c) = 2β₂ + 6β₃·c + 12β₄·c²
            return 2.0 * m.params[2] + 6.0 * m.params[3] * c + 12.0 * m.params[4] * c ** 2

        m2_left = _second_derivative(x[left], y[left])
        m2_right = _second_derivative(x[right], y[right])

        if m2_left is None or m2_right is None:
            curvature = 1.0
        else:
            curvature = max(abs(m2_left), abs(m2_right)) + 1e-10

        # ---- Step 4: 计算最优带宽 ----
        # C_K 常数：三角核 + 局部线性 ≈ 3.47  (IK 2012, Table 1)
        C_K = 3.47
        numerator = 2.0 * resid_var
        denominator = f_c * curvature ** 2 + 1e-10
        h_opt = C_K * (numerator / denominator) ** (1.0 / 5.0) * n ** (-1.0 / 5.0)

        # ---- Step 5: 裁剪至合理范围 ----
        x_range = np.max(x) - np.min(x)
        h_min = 0.01 * x_range
        h_max = 0.50 * x_range
        h_opt = np.clip(h_opt, h_min, h_max)

        return h_opt

    # ------------------------------------------------------------------
    # 模型拟合
    # ------------------------------------------------------------------
    def fit(self, x, y, bandwidth=None):
        """拟合 Sharp RDD 模型

        Parameters
        ----------
        x : array-like
            驱动变量
        y : array-like
            结果变量
        bandwidth : float, optional
            带宽，若为 None 则自动选择 MSE-最优带宽

        Returns
        -------
        self : RDD
        """
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

        if bandwidth is None:
            bandwidth = self.optimal_bandwidth(self._x, self._y)
        self.bandwidth_ = bandwidth

        # 计算核权重
        w = self._kernel_weights(self._x, bandwidth)
        valid = w > 0
        n_valid = valid.sum()

        if n_valid < self.poly_order * 2 + 4:  # 至少比参数多
            raise ValueError(
                f"Insufficient observations ({n_valid}) within bandwidth {bandwidth:.4f}. "
                f"Need at least {self.poly_order * 2 + 4}."
            )

        X = self._design_matrix(self._x[valid])
        y_in = self._y[valid]
        w_in = w[valid]

        # WLS 估计（异方差稳健标准误）
        model = sm.WLS(y_in, X, weights=w_in)
        results = model.fit(cov_type='HC1')

        self.coef_ = results.params
        self._results = results

        # 提取处理效应（T 的系数位于 poly_order + 1 列）
        idx_tau = self.poly_order + 1
        self.tau_ = results.params[idx_tau]
        self.tau_se_ = results.bse[idx_tau]
        z = norm.ppf(0.975)
        self.tau_ci_ = (self.tau_ - z * self.tau_se_, self.tau_ + z * self.tau_se_)

        return self

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict(self, x):
        """基于估计模型预测 E[Y|X]"""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x = np.asarray(x, dtype=float)
        X = self._design_matrix(x)
        return X @ self.coef_

    def predict_bands(self, x, alpha=0.05):
        """预测值及其置信区间"""
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x = np.asarray(x, dtype=float)
        X = self._design_matrix(x)
        pred = X @ self.coef_

        # 仅在有数据支撑的范围内计算 CI
        try:
            pred_obj = self._results.get_prediction(X)
            se_mean = pred_obj.se_mean
            z = norm.ppf(1.0 - alpha / 2.0)
            lower = pred - z * se_mean
            upper = pred + z * se_mean
        except Exception:
            lower = pred * np.nan
            upper = pred * np.nan

        return pred, lower, upper

    # ------------------------------------------------------------------
    # 断点可视化
    # ------------------------------------------------------------------
    def plot(self, x=None, y=None, ax=None, n_bins=20, ci=True,
             figsize=(10, 6), scatter_alpha=0.15, show_ci_band=True):
        """绘制断点回归图

        包含：箱线均值散点、局部多项式拟合曲线、95% 置信带、断点标记线、带宽范围

        Parameters
        ----------
        x, y : array-like, optional
            若为 None 则使用 fit() 时存储的数据
        ax : matplotlib Axes, optional
        n_bins : int
            每侧的分箱数
        ci : bool
            是否显示置信区间
        figsize : tuple
            图像尺寸
        scatter_alpha : float
            原始数据散点透明度
        show_ci_band : bool
            是否显示拟合曲线的置信带

        Returns
        -------
        ax : matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        _x = self._x if x is None else np.asarray(x, dtype=float)
        _y = self._y if y is None else np.asarray(y, dtype=float)

        # -------- 1. 原始数据散点（半透明） --------
        ax.scatter(_x, _y, s=12, c='gray', alpha=scatter_alpha,
                   edgecolors='none', zorder=1)

        # -------- 2. 分箱均值 --------
        def _binned_means(x_side, y_side, n):
            if len(x_side) < 4:
                return np.array([]), np.array([]), np.array([])
            # 等分位数分箱
            percentiles = np.linspace(0, 100, min(n + 1, len(x_side)))
            edges = np.percentile(x_side, percentiles)
            edges = np.unique(edges)
            digitized = np.digitize(x_side, edges)
            bx, by, bse = [], [], []
            for i in range(1, len(edges)):
                mask = digitized == i
                cnt = mask.sum()
                if cnt >= 3:
                    bx.append(np.mean(x_side[mask]))
                    by.append(np.mean(y_side[mask]))
                    bse.append(np.std(y_side[mask], ddof=1) / np.sqrt(cnt))
            return np.array(bx), np.array(by), np.array(bse)

        left = _x < self.cutoff
        right = _x >= self.cutoff

        bx_l, by_l, se_l = _binned_means(_x[left], _y[left], n_bins)
        bx_r, by_r, se_r = _binned_means(_x[right], _y[right], n_bins)

        ax.errorbar(bx_l, by_l, yerr=1.96 * se_l, fmt='o', color='#2166ac',
                     capsize=3, markersize=6, markeredgecolor='white',
                     markeredgewidth=0.5, zorder=3, label='Binned mean (L)')
        ax.errorbar(bx_r, by_r, yerr=1.96 * se_r, fmt='s', color='#b2182b',
                     capsize=3, markersize=6, markeredgecolor='white',
                     markeredgewidth=0.5, zorder=3, label='Binned mean (R)')

        # -------- 3. 拟合曲线 --------
        x_grid = np.linspace(_x.min(), _x.max(), 400)
        y_pred, y_lower, y_upper = self.predict_bands(x_grid)

        left_grid = x_grid < self.cutoff
        right_grid = x_grid >= self.cutoff

        ax.plot(x_grid[left_grid], y_pred[left_grid], '-', color='#2166ac',
                linewidth=2.5, zorder=4, label='Local poly. fit (L)')
        ax.plot(x_grid[right_grid], y_pred[right_grid], '-', color='#b2182b',
                linewidth=2.5, zorder=4, label='Local poly. fit (R)')

        if ci and show_ci_band:
            ax.fill_between(x_grid[left_grid], y_lower[left_grid], y_upper[left_grid],
                             alpha=0.15, color='#2166ac', zorder=2)
            ax.fill_between(x_grid[right_grid], y_lower[right_grid], y_upper[right_grid],
                             alpha=0.15, color='#b2182b', zorder=2)

        # -------- 4. 断点标记 --------
        ax.axvline(x=self.cutoff, color='black', linestyle='--', linewidth=1.5,
                   alpha=0.9, zorder=5, label=f'Cutoff = {self.cutoff}')

        # -------- 5. 带宽范围标记 --------
        if self.bandwidth_ is not None:
            h = self.bandwidth_
            ax.axvspan(self.cutoff - h, self.cutoff + h, alpha=0.06,
                       color='gray', zorder=0, label=f'BW = {h:.3f}')

        # -------- 6. 效应标注 --------
        if self.tau_ is not None:
            text = (f'$\\hat{{\\tau}}$ = {self.tau_:.4f}\n'
                    f'SE = {self.tau_se_:.4f}\n'
                    f'95% CI = [{self.tau_ci_[0]:.4f}, {self.tau_ci_[1]:.4f}]')
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                              alpha=0.8, edgecolor='gray'),
                    zorder=6)

        ax.set_xlabel('Running Variable (X)', fontsize=12)
        ax.set_ylabel('Outcome (Y)', fontsize=12)
        ax.set_title('Regression Discontinuity Design', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='lower left', framealpha=0.9,
                  ncol=2, columnspacing=0.8)
        ax.tick_params(labelsize=10)

        return ax

    # ------------------------------------------------------------------
    # 安慰剂断点检验
    # ------------------------------------------------------------------
    def placebo_test(self, x=None, y=None, n_placebos=15, bandwidth=None):
        """安慰剂断点检验：在假断点位置重复估计

        Parameters
        ----------
        x, y : array-like, optional
        n_placebos : int
            假断点数量
        bandwidth : float, optional
            固定带宽（若不指定则对每个假断点重新选择最优带宽）

        Returns
        -------
        results : list of dict
            每个假断点的估计结果
        """
        _x = self._x if x is None else np.asarray(x, dtype=float)
        _y = self._y if y is None else np.asarray(y, dtype=float)

        # 生成假断点（在 X 的分位数上均匀采样，排除真实断点附近）
        p_low, p_high = 10, 90
        q_low = np.percentile(_x, p_low)
        q_high = np.percentile(_x, p_high)
        buffer = 0.05 * (_x.max() - _x.min())

        candidates = np.linspace(q_low, q_high, n_placebos + 2)[1:-1]
        candidates = candidates[np.abs(candidates - self.cutoff) > buffer]

        results = []
        orig_cutoff = self.cutoff

        for c in candidates:
            rdd_tmp = RDD(cutoff=c, kernel=self.kernel, poly_order=self.poly_order)
            try:
                h = bandwidth if bandwidth is not None else rdd_tmp.optimal_bandwidth(_x, _y)
                rdd_tmp.fit(_x, _y, bandwidth=h)
                results.append({
                    'cutoff': c,
                    'tau': rdd_tmp.tau_,
                    'se': rdd_tmp.tau_se_,
                    'ci_lower': rdd_tmp.tau_ci_[0],
                    'ci_upper': rdd_tmp.tau_ci_[1],
                    'bandwidth': h,
                })
            except Exception:
                continue

        self.cutoff = orig_cutoff
        return results

    def plot_placebo_test(self, x=None, y=None, n_placebos=15, bandwidth=None,
                          ax=None, figsize=(9, 5)):
        """绘制安慰剂检验结果"""
        placebos = self.placebo_test(x, y, n_placebos, bandwidth)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        cutoffs = [p['cutoff'] for p in placebos]
        estimates = [p['tau'] for p in placebos]
        lower = [p['ci_lower'] for p in placebos]
        upper = [p['ci_upper'] for p in placebos]

        ax.errorbar(cutoffs, estimates, yerr=[
            [e - l for e, l in zip(estimates, lower)],
            [u - e for u, e in zip(upper, estimates)]
        ], fmt='o', color='steelblue', capsize=3, markersize=6, label='Placebo estimates')

        # 真实断点效应
        ax.axhline(y=self.tau_, color='red', linestyle='--', linewidth=1.5,
                   label=f'True cutoff estimate = {self.tau_:.4f}')
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

        ax.set_xlabel('Placebo cutoffs', fontsize=12)
        ax.set_ylabel('Estimated treatment effect', fontsize=12)
        ax.set_title('Placebo Cutoff Test', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.axvline(x=self.cutoff, color='red', linestyle=':', alpha=0.5, linewidth=1)

        return ax

    # ------------------------------------------------------------------
    # McCrary 密度检验（操纵检验）
    # ------------------------------------------------------------------
    def density_test(self, x=None, bandwidth=None):
        """McCrary (2008) 密度检验 — 检验驱动变量在断点处是否被操纵

        构建直方图后对 ln(频数) 做局部线性回归，检验断点处不连续性。

        Parameters
        ----------
        x : array-like, optional
        bandwidth : float, optional
            密度估计的平滑带宽

        Returns
        -------
        result : dict
            检验统计量、p 值、估计值及其标准误
        """
        _x = self._x if x is None else np.asarray(x, dtype=float)
        n = len(_x)

        # ----- 1. 确定 bin 宽度（Freedman-Diaconis 规则） -----
        iqr = np.percentile(_x, 75) - np.percentile(_x, 25)
        if iqr < 1e-10:
            iqr = np.std(_x)
        bin_width = 2.0 * iqr * n ** (-1.0 / 3.0)

        # 至少 15 个 bins
        n_bins = int(np.ceil((_x.max() - _x.min()) / bin_width))
        n_bins = max(n_bins, 15)

        # ----- 2. 构建直方图 -----
        counts, edges = np.histogram(_x, bins=n_bins)
        midpoints = (edges[:-1] + edges[1:]) / 2.0

        # 仅使用断点附近的 bins
        x_range = np.max(_x) - np.min(_x)
        near = np.abs(midpoints - self.cutoff) < 0.5 * x_range
        midpoints = midpoints[near]
        counts = counts[near]

        if len(midpoints) < 8:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'theta': np.nan,
                'theta_se': np.nan,
                'message': 'Insufficient bins near cutoff for density test.'
            }

        # 处理空 bin（加小常数避免 log(0)）
        log_counts = np.log(counts + 0.5)  # 加入平滑

        # ----- 3. 断点处局部线性回归 -----
        x_c = midpoints - self.cutoff
        T = (midpoints >= self.cutoff).astype(float)

        X_dens = np.column_stack([
            np.ones_like(x_c), x_c, T, T * x_c
        ])

        # 密度估计带宽（Silverman 规则）
        if bandwidth is None:
            h_dens = 0.9 * min(np.std(_x), iqr / 1.34) * n ** (-0.2)
        else:
            h_dens = bandwidth

        # 使用较宽的核进行平滑
        w_dens = self._kernel_weights(midpoints, h_dens * 3)
        valid = w_dens > 0

        if valid.sum() < 8:
            return {
                'statistic': np.nan, 'p_value': np.nan,
                'theta': np.nan, 'theta_se': np.nan,
                'message': 'Insufficient bins with positive weight.'
            }

        try:
            model = sm.WLS(log_counts[valid], X_dens[valid], weights=w_dens[valid])
            results = model.fit(cov_type='HC1')

            theta = results.params[2]        # T 的系数（断点处跳跃）
            theta_se = results.bse[2]
            z_stat = theta / theta_se
            p_val = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

            message = (
                f'McCrary density test | theta = {theta:.4f}, '
                f'SE = {theta_se:.4f}, z = {z_stat:.3f}, p = {p_val:.4f} | '
                + ('FAIL: density discontinuity detected' if p_val < 0.05
                   else 'PASS: no evidence of manipulation')
            )

            return {
                'statistic': z_stat,
                'p_value': p_val,
                'theta': theta,
                'theta_se': theta_se,
                'message': message,
                'n_bins': len(midpoints),
            }
        except Exception as e:
            return {
                'statistic': np.nan, 'p_value': np.nan,
                'theta': np.nan, 'theta_se': np.nan,
                'message': f'Density test failed: {str(e)}'
            }

    # ------------------------------------------------------------------
    # 协变量平衡检验
    # ------------------------------------------------------------------
    def covariate_balance_test(self, covariates, x=None, bandwidth=None):
        """协变量平衡检验：检查协变量在断点处是否有跳跃

        Parameters
        ----------
        covariates : dict of str -> array-like
            协变量字典 {name: values}
        x : array-like, optional
        bandwidth : float, optional

        Returns
        -------
        results : dict
            每个协变量的检验结果
        """
        _x = self._x if x is None else np.asarray(x, dtype=float)
        h = bandwidth if bandwidth is not None else self.bandwidth_

        results = {}
        orig_cutoff = self.cutoff

        for name, cov in covariates.items():
            cov = np.asarray(cov, dtype=float)
            try:
                rdd_cov = RDD(cutoff=self.cutoff, kernel=self.kernel,
                              poly_order=self.poly_order)
                if h is None:
                    h_cov = rdd_cov.optimal_bandwidth(_x, cov)
                else:
                    h_cov = h
                rdd_cov.fit(_x, cov, bandwidth=h_cov)
                results[name] = {
                    'tau': rdd_cov.tau_,
                    'se': rdd_cov.tau_se_,
                    'ci': rdd_cov.tau_ci_,
                    'p_value': 2 * (1 - norm.cdf(abs(rdd_cov.tau_ / rdd_cov.tau_se_))),
                    'bandwidth': h_cov,
                }
            except Exception as e:
                results[name] = {'error': str(e)}

        return results


# ======================================================================
# 使用示例
# ======================================================================
if __name__ == "__main__":
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    # ----------------------------------------------------------------
    # 1. 模拟数据：Sharp RDD
    # ----------------------------------------------------------------
    np.random.seed(2024)
    n = 1000
    cutoff = 0.0

    x = np.random.uniform(-1.5, 1.5, n)
    T = (x >= cutoff).astype(float)

    # 真实 DGP：Y = 0.3 + 0.5·(X-c) + 0.8·T + 0.3·T·(X-c) + ε
    y = (0.3
         + 0.5 * (x - cutoff)
         + 0.8 * T
         + 0.3 * T * (x - cutoff)
         + np.random.normal(0, 0.25, n))

    print("=" * 60)
    print("RDD — Regression Discontinuity Design")
    print("=" * 60)
    print(f"Simulated data: n = {n}, cutoff = {cutoff}")
    print(f"True treatment effect: 0.8000")
    print()

    # ----------------------------------------------------------------
    # 2. 模型估计
    # ----------------------------------------------------------------
    rdd = RDD(cutoff=cutoff, kernel='triangular', poly_order=1)

    # 最优带宽
    h_opt = rdd.optimal_bandwidth(x, y)
    print(f"[Bandwidth] MSE-optimal h = {h_opt:.4f}")

    # 拟合
    rdd.fit(x, y, bandwidth=h_opt)
    print(f"\n[Estimation] Local linear RDD with {rdd.kernel} kernel")
    print(f"  Bandwidth        = {rdd.bandwidth_:.4f}")
    print(f"  Observations in BW = {(np.abs(x - cutoff) <= rdd.bandwidth_).sum()}")
    print(f"  Treatment effect τ = {rdd.tau_:.4f}")
    print(f"  SE                = {rdd.tau_se_:.4f}")
    print(f"  95% CI            = [{rdd.tau_ci_[0]:.4f}, {rdd.tau_ci_[1]:.4f}]")
    print(f"  Bias              = {rdd.tau_ - 0.8:.4f}")

    # ----------------------------------------------------------------
    # 3. 密度检验（McCrary 检验）
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Diagnostic] McCrary Density Test")
    dens = rdd.density_test(x)
    print(f"  {dens['message']}")

    # ----------------------------------------------------------------
    # 4. 安慰剂断点检验
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Diagnostic] Placebo Cutoff Test")
    placebos = rdd.placebo_test(x, y, n_placebos=10)
    n_sig = sum(1 for p in placebos
                if p['ci_lower'] > 0 or p['ci_upper'] < 0)
    print(f"  {n_sig}/{len(placebos)} placebo cutoffs significant at 5% level")
    if placebos:
        print(f"  Placebo effects range: "
              f"[{min(p['tau'] for p in placebos):.4f}, "
              f"{max(p['tau'] for p in placebos):.4f}]")

    # ----------------------------------------------------------------
    # 5. 协变量平衡检验
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Diagnostic] Covariate Balance Test")
    # 模拟协变量（不应在断点处跳跃）
    cov1 = 0.5 + 0.2 * x + np.random.normal(0, 0.15, n)   # 连续协变量
    cov2 = np.random.normal(0, 1, n)                       # 随机协变量
    balance = rdd.covariate_balance_test(
        {'covariate_A': cov1, 'covariate_B': cov2}
    )
    for name, res in balance.items():
        if 'error' in res:
            print(f"  {name}: ERROR - {res['error']}")
        else:
            flag = '*** JUMP ***' if res['p_value'] < 0.05 else 'OK'
            print(f"  {name}: τ = {res['tau']:+.4f}, "
                  f"p = {res['p_value']:.4f} [{flag}]")

    # ----------------------------------------------------------------
    # 6. 可视化
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Visualization] Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 主 RDD 图
    rdd.plot(x, y, ax=axes[0], n_bins=25, scatter_alpha=0.12)

    # 安慰剂检验图
    rdd.plot_placebo_test(x, y, n_placebos=15, ax=axes[1])

    plt.tight_layout()
    plt.savefig('outputs/rdd_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/rdd_analysis.png")

    # ----------------------------------------------------------------
    # 7. 带宽敏感性分析
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Sensitivity] Bandwidth Sensitivity Analysis")
    bandwidths = np.linspace(0.3 * h_opt, 2.5 * h_opt, 10)
    sens_results = []
    for h in bandwidths:
        rdd_sens = RDD(cutoff=cutoff, kernel='triangular', poly_order=1)
        try:
            rdd_sens.fit(x, y, bandwidth=h)
            sens_results.append({
                'h': h,
                'tau': rdd_sens.tau_,
                'se': rdd_sens.tau_se_,
                'ci_lower': rdd_sens.tau_ci_[0],
                'ci_upper': rdd_sens.tau_ci_[1],
            })
        except Exception:
            continue

    sens_h = np.array([r['h'] for r in sens_results])
    sens_tau = np.array([r['tau'] for r in sens_results])
    sens_lower = np.array([r['ci_lower'] for r in sens_results])
    sens_upper = np.array([r['ci_upper'] for r in sens_results])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(sens_h, sens_tau,
                yerr=[sens_tau - sens_lower, sens_upper - sens_tau],
                fmt='o-', color='darkblue', capsize=3, markersize=6)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1,
               label='True effect = 0.8')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axvline(x=h_opt, color='green', linestyle=':', linewidth=1,
               label=f'IK-optimal h = {h_opt:.3f}')
    ax.set_xlabel('Bandwidth (h)', fontsize=12)
    ax.set_ylabel('Estimated treatment effect', fontsize=12)
    ax.set_title('Bandwidth Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/rdd_bandwidth_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/rdd_bandwidth_sensitivity.png")

    # ----------------------------------------------------------------
    # 8. 汇总报告
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Method         : Regression Discontinuity Design (Sharp)")
    print(f"Kernel         : {rdd.kernel}")
    print(f"Polynomial     : Order {rdd.poly_order}")
    print(f"Optimal BW     : {h_opt:.4f}")
    print(f"Treatment Eff. : {rdd.tau_:.4f} (SE = {rdd.tau_se_:.4f})")
    print(f"95% CI         : [{rdd.tau_ci_[0]:.4f}, {rdd.tau_ci_[1]:.4f}]")
    print(f"McCrary Test   : p = {dens['p_value']:.4f}")
    print(f"Plots saved to : outputs/rdd_analysis.png, "
          f"outputs/rdd_bandwidth_sensitivity.png")
    print("=" * 60)
```

---

## 参考文献

1. **Thistlethwaite, D. L., & Campbell, D. T. (1960).** Regression-discontinuity analysis: An alternative to the ex post facto experiment. *Journal of Educational Psychology*, 51(6), 309–317. DOI: 10.1037/h0044319

2. **Lee, D. S., & Lemieux, T. (2010).** Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281–355. DOI: 10.1257/jel.48.2.281

3. **Imbens, G., & Kalyanaraman, K. (2012).** Optimal bandwidth choice for the regression discontinuity estimator. *Review of Economic Studies*, 79(3), 933–959. DOI: 10.1093/restud/rdr043

4. **Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014).** Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326. DOI: 10.3982/ECTA11757

5. **McCrary, J. (2008).** Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714. DOI: 10.1016/j.jeconom.2007.05.005

6. **Hahn, J., Todd, P., & Van der Klaauw, W. (2001).** Identification and estimation of treatment effects with a regression-discontinuity design. *Econometrica*, 69(1), 201–209. DOI: 10.1111/1468-0262.00183

7. **Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2019).** *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press. DOI: 10.1017/9781108684606
