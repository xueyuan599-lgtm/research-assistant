# Prophet — 时间序列预测

- **来源**: Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37–45.
- **DOI**: 10.1080/00031305.2017.1380080
- **方法类别**: 时间序列 / 贝叶斯结构时间序列 / 商业预测

## 数学设定

### 模型框架

Prophet 将时间序列分解为三个核心分量的加法（或乘法）组合：

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t
$$

其中 $g(t)$ 为趋势项，$s(t)$ 为季节项，$h(t)$ 为节假日效应项，$\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ 为不可约噪声。

---

### 趋势项 $g(t)$

Prophet 提供两种趋势形式，可根据业务场景灵活选择。

#### 1. 分段线性逻辑斯蒂增长（饱和增长）

适用于有增长上限的场景（如用户规模、市场份额）：

$$
g(t) = \frac{C(t)}{1 + \exp\big(-(k + \boldsymbol{a}(t)^\top \boldsymbol{\delta})(t - (m + \boldsymbol{a}(t)^\top \boldsymbol{\gamma}))\big)}
$$

其中：
- $C(t)$ 为时变的承载能力（carrying capacity，即饱和上限）
- $k$ 为基准增长率
- $\boldsymbol{\delta} = (\delta_1, \dots, \delta_J)^\top$ 为 $J$ 个变点处的增长率调整量
- $m$ 为基准偏移参数
- $\boldsymbol{a}(t) \in \{0,1\}^J$ 是指示向量，$a_j(t) = 1$ 当且仅当 $t \geq s_j$（$s_j$ 为第 $j$ 个变点位置）
- $\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_J)^\top$ 为保证函数连续的偏移调整量，满足：
  $$
  \gamma_j = \frac{s_j - m - \sum_{l < j} \gamma_l}{1 + \sum_{l < j} \delta_l} \cdot \delta_j
  $$

#### 2. 分段线性趋势（无饱和限）

适用于无自然上限的场景（如气温、销量绝对量）：

$$
g(t) = \big(k + \boldsymbol{a}(t)^\top \boldsymbol{\delta}\big) \cdot t + \big(m + \boldsymbol{a}(t)^\top \boldsymbol{\gamma}\big)
$$

连续性条件给出 $\gamma_j = -s_j \delta_j$，使得在变点 $s_j$ 处左右极限相等：
$$
\lim_{t \to s_j^-} g(t) = \lim_{t \to s_j^+} g(t)
$$

---

### 变点检测（Changepoints）

Prophet 不主动搜索变点位置，而是**在时间轴上均匀预设 $J$ 个候选变点**，然后通过稀疏先验自动收缩不必要的调整量：

$$
\delta_j \sim \text{Laplace}(0, \tau), \quad j = 1, \dots, J
$$

超参数 $\tau$（`changepoint_prior_scale`）控制趋势的灵活性：
- $\tau$ 越大，允许趋势变化越剧烈（可能过拟合）
- $\tau$ 越小，趋势越平滑（可能欠拟合）

变点范围参数 `changepoint_range` 控制在哪段时间内放置变点（默认只在前 80% 的时间段内放置，给预测段留出余量）。

---

### 季节项 $s(t)$

Prophet 使用 **傅里叶级数** 拟合周期性模式：

$$
s(t) = \sum_{n=1}^{N} \left[ a_n \cos\!\left(\frac{2\pi n t}{P}\right) + b_n \sin\!\left(\frac{2\pi n t}{P}\right) \right]
$$

其中 $P$ 为周期长度：
- 年周期：$P = 365.25$（默认 $N = 10$，共 20 个参数）
- 周周期：$P = 7$（默认 $N = 3$，共 6 个参数）
- 日周期：$P = 1$（默认 $N = 6$，共 12 个参数，仅对亚日数据启用）

傅里叶系数 $(a_n, b_n)$ 共同组成参数向量 $\boldsymbol{\beta}$，服从正态先验：
$$
\boldsymbol{\beta} \sim \mathcal{N}(0, \sigma_s^2)
$$

$\sigma_s$（`seasonality_prior_scale`）控制季节分量的强度。

#### 乘法季节项

当季节波动幅度随趋势增长时，可以将季节项改为乘法形式：

$$
y(t) = g(t) \cdot \big(1 + s(t)\big) + h(t) + \varepsilon_t
$$

等价于在 log 空间做加法（适用于对数正态型数据）。

---

### 节假日效应 $h(t)$

节假日效应使用 **虚拟变量回归** 建模。对每个节假日 $i$，定义其影响窗口 $D_i$（如圣诞节前后各 3 天），则：

$$
h(t) = \sum_{i=1}^{L} \kappa_i \cdot \mathbb{I}[t \in D_i] = \boldsymbol{Z}(t)^\top \boldsymbol{\kappa}
$$

其中 $\boldsymbol{Z}(t)$ 为指示向量，$\boldsymbol{\kappa}$ 服从正态先验：
$$
\kappa_i \sim \mathcal{N}(0, \nu^2)
$$

$\nu$（`holidays_prior_scale`）控制节假日效应的强度。不同节假日之间独立。

---

### 完整模型与估计

将以上三项合并，Prophet 本质上是一个**贝叶斯广义线性模型**（Bayesian GLM）：

$$
y(t) \sim \mathcal{N}\big(g(t) + s(t) + h(t),\; \sigma^2\big)
$$

三个正则化参数构成分层先验：
| 参数 | 先验 | 控制对象 |
|------|------|----------|
| $\tau$ | — | 趋势灵活性（变点调整幅度） |
| $\sigma_s$ | — | 季节性强度 |
| $\nu$ | — | 节假日效应强度 |

估计方法有两种：
1. **MAP 估计**（默认）：使用 L-BFGS-B 最大化对数后验，快速得到点估计
2. **完整 MCMC**：通过 Stan 对全后验分布采样，获得更准确的置信区间（尤其适用于短序列或强先验场景）

---

## 关键假设

1. **加法/乘法可分解性**：趋势、季节、节假日三项独立叠加，交互作用由模式选择（additive/multiplicative）近似捕捉
2. **变点稀疏性**：$\delta_j \sim \text{Laplace}(0, \tau)$ 假设大多数候选变点不发生实质性的斜率变化
3. **节假日独立性**：各节假日效应 $\kappa_i$ 独立同分布，互不影响
4. **傅里叶光滑性**：季节模式可用有限阶傅里叶级数充分逼近（低阶截断意味着频率受限）
5. **误差同方差**（基本模型）：$\varepsilon_t$ 为独立同分布高斯白噪声（可选 MCMC 获得异方差区间）
6. **时间均匀性**：观测时刻等距（或近似等距），不等距数据需预处理为重采样网格
7. **缺失数据机制**：Prophet 对缺失值鲁棒（跳过缺失点即可，无需插补）

---

## 适用场景

### 适用
- **商业指标预测**：日活跃用户、GMV、订单量等具有周/年周期和节假日效应的序列
- **运营监控**：网站流量、服务器负载、客服工单量的趋势+季节分解
- **具有强周期性的中长期预测**：12–24 个月的日/周数据预测
- **缺失值多、异常值多的数据**：Prophet 对数据质量不敏感，无需预处理
- **需要可解释性**：趋势、周季节、年季节、节假日分开输出，业务归因直观
- **快速基线模型**：几行代码完成建模，适合作为复杂模型的对照
- **变化点明显的序列**：如疫情前后的用户行为突变

### 不适用
- **高频（亚日）数据无聚合**：未经聚合的小时/分钟级数据噪声过大，需先聚合到日
- **纯自回归过程**：Prophet 无 ARMA 结构，不适合纯自回归型预测（如白噪声驱动的振动信号）
- **长周期（>1 年）预测无明确季节模式**：Prophet 外推能力有限，长期预测依赖季节模式的延续性
- **复杂多元依赖**：如需建模多个序列间的交叉依赖（VAR、状态空间模型更合适）
- **非均匀采样/严重缺失块**：大段连续缺失会破坏趋势估计
- **需要严格统计推断**：Prophet 的置信区间假设高斯误差，对非高斯厚尾数据可能不准确
- **超短期预测（< 7 个时间点）**：数据量不足以估计季节成分

---

## 实现要点

### 关键超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `changepoint_prior_scale` | 0.05 | $[10^{-3}, 0.5]$ | 趋势灵活性，越大变点调整越剧烈 |
| `seasonality_prior_scale` | 10.0 | $[0.1, 100]$ | 季节分量强度，越大季节波动越强 |
| `holidays_prior_scale` | 10.0 | $[0.1, 100]$ | 节假日效应强度 |
| `n_changepoints` | 25 | $[0, 50]$ | 候选变点数量 |
| `changepoint_range` | 0.8 | $[0.5, 0.95]$ | 变点放置的时间范围比例 |
| `seasonality_mode` | additive | additive / multiplicative | 季节项与趋势的组合方式 |
| `yearly_seasonality` | auto | `True` / `False` / int | 年季节傅里叶阶数（auto 自动判断） |
| `weekly_seasonality` | auto | `True` / `False` / int | 周季节傅里叶阶数 |
| `daily_seasonality` | auto | `True` / `False` / int | 日季节傅里叶阶数 |
| `interval_width` | 0.80 | $(0, 1)$ | 预测区间置信水平 |

### 调优经验

1. **先调 `changepoint_prior_scale`**：CV 确定最合适的趋势灵活度，通常 0.01–0.5
2. **再定季节模式**：如果季节振幅随趋势增长，切换到 multiplicative
3. **后调 `seasonality_prior_scale`**：减少到 1–5 可防止季节过拟合
4. **变点数量**：短序列（<1 年）可减少到 5–10，长序列（>2 年）默认 25 即可
5. **节假日列表**：务必包含所有已知节假日，包含前后窗口（`lower_window` / `upper_window`）
6. **特殊事件**：用 `add_regressor()` 加入促销、天气、竞品事件等外部变量

### 交叉验证

Prophet 提供专用的时间序列交叉验证：

```python
from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(
    model,
    initial='730 days',   # 初始训练集长度
    period='180 days',    # 每次向前滑动步长
    horizon='365 days',   # 预测视野
    parallel='processes'  # 并行
)
df_metrics = performance_metrics(df_cv)
```

### 不确定性量化

- **MAP 模式**：仅考虑观测噪声 $\sigma$，区间偏窄
- **MCMC 模式**：完整后验采样，包含参数不确定性，区间更宽更可靠
- **不确定性分解**：趋势不确定性（变点） + 季节不确定性 + 观测噪声

---

## 完整 Python 代码

### 从零实现：ProphetModel

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


class ProphetModel:
    """
    Simplified Prophet: piecewise linear trend + Fourier seasonality.

    Fit via MAP estimation (L-BFGS-B). Supports weekly and yearly
    Fourier seasonality with automatic changepoint detection.
    """

    def __init__(self, n_changepoints=25, changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0, weekly_order=3,
                 yearly_order=10, changepoint_range=0.8):
        self.n_changepoints = n_changepoints
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.weekly_order = weekly_order
        self.yearly_order = yearly_order
        self.changepoint_range = changepoint_range
        self.params_ = None
        self.changepoints_ = None
        self.fitted_ = None
        self.t_ = None
        self.y_ = None

    # ──────────────────────────────────────────────
    #  Fourier basis
    # ──────────────────────────────────────────────
    def _make_fourier_features(self, t, period, n):
        """Build Fourier series design matrix: (len(t), 2*n)."""
        features = np.zeros((len(t), 2 * n))
        for i in range(1, n + 1):
            features[:, 2 * (i - 1)] = np.sin(2 * np.pi * i * t / period)
            features[:, 2 * (i - 1) + 1] = np.cos(2 * np.pi * i * t / period)
        return features

    # ──────────────────────────────────────────────
    #  Changepoint detection
    # ──────────────────────────────────────────────
    def add_changepoints(self, t, y):
        """
        Automatically detect changepoint locations using rolling slope
        analysis.  Stores result in self.changepoints_.
        """
        n = len(t)
        if n < 30:
            self.changepoints_ = np.array([])
            self.n_changepoints = 0
            return self

        window = max(10, n // 30)
        slopes = np.full(n, np.nan)

        for i in range(window, n - window):
            x_seg = t[i - window:i + window + 1]
            y_seg = y[i - window:i + window + 1]
            slopes[i] = np.polyfit(x_seg - x_seg[0], y_seg, 1)[0]

        #  Detect abrupt changes in slope
        slope_diff = np.abs(np.diff(slopes))
        slope_diff[np.isnan(slope_diff)] = 0.0
        threshold = np.percentile(slope_diff, 95)

        candidates = np.where(slope_diff > threshold)[0]
        if len(candidates) == 0:
            candidates = np.array([np.argmax(slope_diff)])

        #  Keep at most n_changepoints
        if len(candidates) > self.n_changepoints:
            top = np.argsort(slope_diff[candidates])[-self.n_changepoints:]
            cp_idx = np.sort(candidates[top])
        else:
            cp_idx = np.sort(candidates)

        self.changepoints_ = t[cp_idx]
        self.n_changepoints = len(self.changepoints_)
        return self

    # ──────────────────────────────────────────────
    #  Parameter initialisation
    # ──────────────────────────────────────────────
    def _init_params(self, t, y):
        """Place changepoints uniformly and build initial parameter vector."""
        n = len(t)

        #  Default changepoint placement (uniform in first changepoint_range)
        if self.changepoints_ is None:
            cp_count = int(self.changepoint_range * n) - 1
            cp_count = min(cp_count, self.n_changepoints)
            cp_idx = np.linspace(0, n - 1, cp_count).astype(int)
            self.changepoints_ = t[cp_idx]
            self.n_changepoints = len(self.changepoints_)

        #  Parameter layout:
        #  [k, m, delta_1..delta_J, beta_w1..beta_w(2*WO), beta_y1..beta_y(2*YO), log_sigma]
        n_param = (2 + self.n_changepoints
                   + 2 * self.weekly_order
                   + 2 * self.yearly_order
                   + 1)
        params = np.zeros(n_param)

        #  k : rough slope
        params[0] = (y[-1] - y[0]) / (t[-1] - t[0]) if t[-1] != t[0] else 0.0
        #  m : rough intercept
        params[1] = y[0] - params[0] * t[0]
        #  delta : all zero (no adjustments initially)
        #  beta : small random perturbations (break symmetry)
        n_beta = 2 * (self.weekly_order + self.yearly_order)
        offset = 2 + self.n_changepoints
        params[offset:offset + n_beta] = np.random.randn(n_beta) * 0.05
        #  log_sigma
        params[-1] = np.log(np.std(y) * 0.2 + 1e-6)

        return params

    # ──────────────────────────────────────────────
    #  Core prediction (internal)
    # ──────────────────────────────────────────────
    def _predict(self, params, t):
        """Return y_pred = trend + seasonality given parameter vector."""
        n = len(t)
        k = params[0]
        m = params[1]
        delta = params[2:2 + self.n_changepoints]

        # --- Trend ---
        A = np.zeros((n, self.n_changepoints))
        for j, s in enumerate(self.changepoints_):
            A[:, j] = (t >= s).astype(float)

        gamma = -self.changepoints_ * delta          # continuity correction
        trend = (k + A @ delta) * t + (m + A @ gamma)

        # --- Seasonality ---
        f_w = self._make_fourier_features(t, 7.0, self.weekly_order)
        f_y = self._make_fourier_features(t, 365.25, self.yearly_order)

        offset = 2 + self.n_changepoints
        nw = 2 * self.weekly_order
        ny = 2 * self.yearly_order
        beta_w = params[offset:offset + nw]
        beta_y = params[offset + nw:offset + nw + ny]

        seasonality = f_w @ beta_w + f_y @ beta_y
        return trend + seasonality

    # ──────────────────────────────────────────────
    #  Negative log posterior (MAP objective)
    # ──────────────────────────────────────────────
    def _nlpost(self, params, t, y):
        """Negative log posterior = NLL + Laplace prior + Normal prior."""
        y_pred = self._predict(params, t)
        n = len(y)
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)

        #  Gaussian likelihood
        nll = n * log_sigma + 0.5 * np.sum((y - y_pred) ** 2) / (sigma ** 2)

        #  Laplace prior on changepoint adjustments
        delta = params[2:2 + self.n_changepoints]
        nll += np.sum(np.abs(delta)) / self.changepoint_prior_scale

        #  Normal prior on seasonality coefficients
        offset = 2 + self.n_changepoints
        n_beta = 2 * (self.weekly_order + self.yearly_order)
        beta = params[offset:offset + n_beta]
        nll += 0.5 * np.sum(beta ** 2) / (self.seasonality_prior_scale ** 2)

        return nll

    # ──────────────────────────────────────────────
    #  Fit
    # ──────────────────────────────────────────────
    def fit(self, t, y):
        """Estimate parameters via L-BFGS-B (MAP)."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        p0 = self._init_params(t, y)

        result = minimize(
            self._nlpost, p0, args=(t, y),
            method='L-BFGS-B',
            options={'maxiter': 20000, 'ftol': 1e-10, 'disp': False}
        )
        self.params_ = result.x
        self.fitted_ = self._predict(result.x, t)
        self.t_ = t
        self.y_ = y

        #  Residual standard deviation for uncertainty
        self.sigma_ = np.exp(result.x[-1])
        return self

    # ──────────────────────────────────────────────
    #  Predict
    # ──────────────────────────────────────────────
    def predict(self, t_new):
        """Return point forecast for new time points."""
        return self._predict(self.params_, np.asarray(t_new, dtype=float))

    def predict_with_uncertainty(self, t_new, alpha=0.05):
        """Return (yhat, yhat_lower, yhat_upper)."""
        t_new = np.asarray(t_new, dtype=float)
        y_pred = self._predict(self.params_, t_new)
        z = norm.ppf(1.0 - alpha / 2.0)
        half_width = z * self.sigma_
        return y_pred, y_pred - half_width, y_pred + half_width

    def predict_components(self, t_new):
        """Return (yhat, trend, weekly_seasonality, yearly_seasonality)."""
        t_new = np.asarray(t_new, dtype=float)
        n = len(t_new)
        k = self.params_[0]
        m = self.params_[1]
        delta = self.params_[2:2 + self.n_changepoints]

        #  Trend
        A = np.zeros((n, self.n_changepoints))
        for j, s in enumerate(self.changepoints_):
            A[:, j] = (t_new >= s).astype(float)
        gamma = -self.changepoints_ * delta
        trend = (k + A @ delta) * t_new + (m + A @ gamma)

        #  Seasonality
        offset = 2 + self.n_changepoints
        nw = 2 * self.weekly_order
        ny = 2 * self.yearly_order
        beta_w = self.params_[offset:offset + nw]
        beta_y = self.params_[offset + nw:offset + nw + ny]

        f_w = self._make_fourier_features(t_new, 7.0, self.weekly_order)
        f_y = self._make_fourier_features(t_new, 365.25, self.yearly_order)

        weekly = f_w @ beta_w
        yearly = f_y @ beta_y

        y_pred = trend + weekly + yearly
        return y_pred, trend, weekly, yearly

    # ──────────────────────────────────────────────
    #  Component plots
    # ──────────────────────────────────────────────
    def plot_components(self, t_new=None):
        """
        Three-panel decomposition: trend, weekly seasonality,
        yearly seasonality.

        Parameters
        ----------
        t_new : array-like or None
            Time points for trend plot.  If None, uses training time.
        """
        if t_new is None:
            t_new = self.t_

        y_pred, trend, weekly, yearly = self.predict_components(t_new)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

        #  Panel 1: Trend
        axes[0].plot(t_new, trend, color='tab:blue', linewidth=1.5)
        if self.changepoints_ is not None and len(self.changepoints_) > 0:
            for cp in self.changepoints_:
                axes[0].axvline(cp, color='gray', linestyle='--',
                                alpha=0.4, linewidth=0.8)
            axes[0].text(t_new[-1], trend[-1], 'Trend',
                         fontsize=10, va='center')
        axes[0].set_ylabel('Trend')
        axes[0].set_title('Prophet Component Decomposition')
        axes[0].grid(True, alpha=0.3)

        #  Panel 2: Weekly seasonality
        doy = np.linspace(0, 7, 200)
        f_w = self._make_fourier_features(doy, 7.0, self.weekly_order)
        offset = 2 + self.n_changepoints
        nw = 2 * self.weekly_order
        beta_w = self.params_[offset:offset + nw]
        weekly_pattern = f_w @ beta_w
        axes[1].plot(doy, weekly_pattern, color='tab:green', linewidth=1.5)
        axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_ylabel('Weekly')
        axes[1].set_xlabel('Day of week')
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu',
                                  'Fri', 'Sat', 'Sun'])
        axes[1].grid(True, alpha=0.3)

        #  Panel 3: Yearly seasonality
        doy = np.linspace(0, 365, 365)
        f_y = self._make_fourier_features(doy, 365.25, self.yearly_order)
        ny = 2 * self.yearly_order
        beta_y = self.params_[offset + nw:offset + nw + ny]
        yearly_pattern = f_y @ beta_y
        axes[2].plot(doy, yearly_pattern, color='tab:red', linewidth=1.5)
        axes[2].axhline(0, color='gray', linestyle='-', alpha=0.5)
        axes[2].set_ylabel('Yearly')
        axes[2].set_xlabel('Day of year')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes


# =====================================================================
#  使用示例（合成数据）
# =====================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')      # 无头环境兼容

    # --- 1. 合成数据：线性趋势 + 周季节 + 噪声 ---
    np.random.seed(42)
    days = np.arange(0, 730, 1.0)                     # 2 年日数据
    trend_true = 10.0 + 0.008 * days                   # 线性趋势
    weekly_true = 1.5 * np.sin(2 * np.pi * days / 7)   # 周季节
    noise = np.random.randn(len(days)) * 0.6
    y = trend_true + weekly_true + noise

    # --- 2. 训练自定义 ProphetModel ---
    print("=== 训练自定义 ProphetModel ===")
    model = ProphetModel(
        n_changepoints=20,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        weekly_order=3,
        yearly_order=0           # 合成数据无年季节，设为 0 减少参数量
    )
    model.fit(days, y)

    # --- 3. 预测 ---
    future = np.arange(730, 800, 1.0)                  # 未来 70 天
    y_pred, y_low, y_high = model.predict_with_uncertainty(future, alpha=0.05)
    y_pred_train = model.predict(days)

    # --- 4. 评估 ---
    train_rmse = np.sqrt(np.mean((y - y_pred_train) ** 2))
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Estimated sigma: {model.sigma_:.4f}")

    # --- 5. 可视化 ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(days, y, 'k.', alpha=0.3, markersize=1.5, label='Observed')
    ax.plot(days, y_pred_train, color='tab:blue', linewidth=1.5, label='Fitted')
    ax.plot(future, y_pred, color='tab:red', linewidth=1.5, label='Forecast')
    ax.fill_between(future, y_low, y_high,
                    color='tab:red', alpha=0.15, label='95% CI')
    ax.axvline(730, color='gray', linestyle='--', alpha=0.6, label='Forecast start')
    ax.set_xlabel('Day')
    ax.set_ylabel('y')
    ax.set_title('ProphetModel — Forecast with Uncertainty')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('E:/wuyi/数学建模半自动/research-assistant/knowledge/algorithm-repository/figures/prophet_forecast.png',
                dpi=150, bbox_inches='tight')
    print("Figure saved to prophet_forecast.png")

    # --- 6. 组件分解 ---
    fig2, _ = model.plot_components(np.concatenate([days, future]))
    plt.savefig('E:/wuyi/数学建模半自动/research-assistant/knowledge/algorithm-repository/figures/prophet_components.png',
                dpi=150, bbox_inches='tight')
    print("Component plot saved to prophet_components.png")
    plt.show()
```

### 基于 fbprophet / prophet 库的生产用法

```python
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# ---- 准备数据（Prophet 要求两列: ds=日期, y=数值） ----
np.random.seed(42)
dates = pd.date_range('2022-01-01', periods=730, freq='D')
trend = 10.0 + 0.008 * np.arange(730)
weekly = 1.5 * np.sin(2 * np.pi * np.arange(730) / 7)
y = trend + weekly + np.random.randn(730) * 0.6

df = pd.DataFrame({'ds': dates, 'y': y})

# ---- 基础模型 ----
model = Prophet(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    seasonality_mode='additive',
    interval_width=0.95
)
model.add_country_holidays('US')   # 加入美国节假日
model.fit(df)

# ---- 预测 ----
future = model.make_future_dataframe(periods=60, freq='D')
forecast = model.predict(future)

# ---- 组件图 ----
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# ---- 交叉验证 ----
df_cv = cross_validation(
    model,
    initial='365 days',
    period='90 days',
    horizon='180 days',
    parallel='processes'
)
df_metrics = performance_metrics(df_cv)
print(df_metrics.head())

# ---- 自定义季节项 / 额外回归量 ----
# model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# model.add_regressor('external_feature', prior_scale=5.0, mode='additive')

# ---- 乘法季节项（适合波动随趋势增长的数据） ----
# model = Prophet(seasonality_mode='multiplicative')

# ---- 饱和增长（逻辑斯蒂，需指定 cap） ----
# df['cap'] = 100
# model = Prophet(growth='logistic')
# model.fit(df)
```

---

## 参考文献

1. Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37–45. DOI: 10.1080/00031305.2017.1380080
2. Prophet 官方文档 — https://facebook.github.io/prophet/
3. Prophet GitHub 仓库 — https://github.com/facebook/prophet
4. Stan Development Team. (2023). Stan Modeling Language Users Guide and Reference Manual. https://mc-stan.org/
