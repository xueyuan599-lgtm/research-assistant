# ARIMA/SARIMA — 差分自回归移动平均模型

- **来源**: Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- **DOI**: 10.1002/9781118619193
- **方法类别**: 时间序列 / 单变量预测 / 参数模型

## 数学设定

### AR(p) — 自回归模型

$$
X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2)
$$

当前值表示为过去 $p$ 个观测值的线性组合加上白噪声。使用 **backshift 算子** $B$（$B^k X_t = X_{t-k}$）：

$$
(1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p) X_t = c + \varepsilon_t
$$

简记为 $\phi(B) X_t = c + \varepsilon_t$。

### MA(q) — 移动平均模型

$$
X_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
$$

当前值表示为当前和过去 $q$ 个白噪声冲击的加权和：

$$
X_t = \mu + (1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q) \varepsilon_t
$$

简记为 $X_t = \mu + \theta(B) \varepsilon_t$。

### ARMA(p,q) — 自回归移动平均模型

$$
X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t
$$

backshift 形式：

$$
\phi(B) X_t = c + \theta(B) \varepsilon_t
$$

### ARIMA(p,d,q) — 差分自回归移动平均模型

对 $X_t$ 进行 $d$ 阶差分使其平稳，然后对差分序列拟合 ARMA：

$$
W_t = (1 - B)^d X_t
$$

$$
\phi(B) W_t = c + \theta(B) \varepsilon_t
$$

$$
\phi(B)(1 - B)^d X_t = c + \theta(B) \varepsilon_t
$$

当 $d=1$ 时：$W_t = X_t - X_{t-1}$（一阶差分常用于经济金融数据）。
当 $d=2$ 时：$W_t = (X_t - X_{t-1}) - (X_{t-1} - X_{t-2}) = X_t - 2X_{t-1} + X_{t-2}$。

### SARIMA(p,d,q)(P,D,Q)$_s$ — 季节差分自回归移动平均模型

在 ARIMA 基础上引入季节成分（$s$ 为季节周期长度，如 $s=12$ 月数据，$s=4$ 季数据）：

$$
\Phi_P(B^s) \phi_p(B) (1-B)^d (1-B^s)^D X_t = c + \Theta_Q(B^s) \theta_q(B) \varepsilon_t
$$

其中：

- $\phi_p(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ — 非季节 AR 多项式
- $\theta_q(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ — 非季节 MA 多项式
- $\Phi_P(B^s) = 1 - \Phi_1 B^s - \Phi_2 B^{2s} - \cdots - \Phi_P B^{Ps}$ — 季节 AR 多项式
- $\Theta_Q(B^s) = 1 + \Theta_1 B^s + \Theta_2 B^{2s} + \cdots + \Theta_Q B^{Qs}$ — 季节 MA 多项式
- $(1-B)^d$ — 非季节差分算子
- $(1-B^s)^D$ — 季节差分算子

乘积形式意味着模型的 AR 部分包含非季节滞后与季节滞后的所有交互项，MA 部分同理。

### 平稳性条件

AR 特征方程 $\phi(B) = 0$ 的所有根的模大于 1（即根在单位圆外）。等价地，反特征方程 $1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$ 的所有根的模小于 1。

- **AR(1)**: $|\phi_1| < 1$
- **AR(2)**: $\phi_1 + \phi_2 < 1$, $\phi_2 - \phi_1 < 1$, $|\phi_2| < 1$

### 可逆性条件

MA 特征方程 $\theta(B) = 0$ 的所有根的模大于 1。这是保证 MA 模型可以唯一表示为无限阶 AR 形式的条件，也是参数估计可识别的必要条件。

### ACF / PACF 识别

| 模型 | ACF | PACF |
|------|-----|------|
| AR(p) | 指数衰减或正弦波动（拖尾） | 在滞后 $p$ 后截尾 |
| MA(q) | 在滞后 $q$ 后截尾 | 指数衰减（拖尾） |
| ARMA(p,q) | 拖尾（$q-p$ 步后开始） | 拖尾（$p-q$ 步后开始） |

### 信息准则

用于模型选择（定阶）的常用准则，其中 $k = p + q + P + Q + 1$（含截距项），$n$ 为有效样本量，$\hat{L}$ 为极大似然值：

$$
\text{AIC} = -2\log(\hat{L}) + 2k
$$

$$
\text{BIC} = -2\log(\hat{L}) + k\log(n)
$$

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

CSS-MLE 下: $\text{AIC} \approx n\log(\text{SSE}/n) + 2k$，忽略常数项不影响模型比较。

## 关键假设

1. **平稳性（或通过差分达到平稳）**：序列均值、方差不随时间变化（趋势/季节成分已通过差分去除）
2. **可逆性**：MA 多项式根在单位圆外，保证唯一表示
3. **白噪声残差**：残差无自相关（Ljung-Box 检验 $p > 0.05$），否则模型阶数不足
4. **正态性（可选）**：残差正态分布对点估计不是必需的，但影响预测区间和 $t$ 检验的有效性
5. **线性性**：ARIMA 捕捉线性依赖关系，不能自动处理非线性模式（需用 ARCH/GARCH 或非线性时间序列模型）
6. **等间隔采样**：要求观测时间间隔均匀，不规则间隔数据需预处理

## 适用场景

- **单变量时间序列预测**：经济指标（GDP、CPI、失业率）、金融序列（股价、汇率）、需求预测
- **含趋势和季节性的序列**：经适当差分后 ARIMA/SARIMA 是经典选择
- **短期预测**：ARIMA 在 1-3 步预测中通常表现稳健
- **作为预测 baseline**：几乎所有时间序列预测任务的第一个对照模型
- **与外部回归量结合**：ARIMAX / SARIMAX 可引入外生变量

### 不适用

- **多变量关系**：多变量时间序列用 VAR / VARMA
- **不规则间隔数据**：需先插值为等间隔
- **极长期预测**：预测区间随步长快速扩大，预测均值回归到序列均值
- **高噪声信号**：信噪比极低时 ARIMA 效果有限
- **非线性模式**：波动率聚类用 GARCH，复杂非线性用深度学习时序模型
- **高维预测**：大量相关序列同时预测用 GVAR / 因子模型 / 深度学习

## 实现要点

### 关键参数

| 参数 | 范围 | 说明 |
|------|------|------|
| $p$ | [0, 10] | 非季节自回归阶数，PACF 截尾点 |
| $d$ | [0, 2] | 非季节差分阶数，ADF/KPSS 检验确定 |
| $q$ | [0, 10] | 非季节移动平均阶数，ACF 截尾点 |
| $P$ | [0, 3] | 季节自回归阶数，季节 PACF 截尾点 |
| $D$ | [0, 1] | 季节差分阶数，季节单位根检验确定 |
| $Q$ | [0, 3] | 季节移动平均阶数，季节 ACF 截尾点 |
| $s$ | 4/12/52 | 季节周期长度 |
| $include\_mean$ | True/False | 是否包含截距项 |

### 实现注意事项

1. **定阶策略**：
   - 先定 $d$：ADF 检验（$H_0$: 单位根存在） + KPSS 检验（$H_0$: 平稳）
   - 再定 $(p,q)$：观察差分后的 ACF/PACF；或用 auto-ARIMA 搜索
   - 季节参数 $(P,D,Q)_s$：观察季节性 lag（$s, 2s, \dots$）处的 ACF/PACF

2. **参数估计**：
   - **CSS（条件平方和）**：初始残差设为 0，递推计算残差，最小化 SSE
   - **CSS-MLE**：两步法，先 CSS 再一步 Newton-Raphson 得到精确 MLE
   - **卡尔曼滤波**：State Space 表示，精确 MLE，statsmodels 默认方法

3. **残差诊断**：
   - Ljung-Box Q 检验：残差是否存在剩余自相关
   - Jarque-Bera 检验：残差是否正态
   - Q-Q 图：目视检查正态性
   - ACF/PACF 残差图：无显著自相关结构

4. **预测区间**：
   - 解析法：基于 $\psi$-权重的方差公式 $\text{Var}(e_{t+h|t}) = \sigma^2 \sum_{i=0}^{h-1} \psi_i^2$
   - 模拟法：从残差 Bootstrap 重抽样，生成预测路径的经验分布

5. **模型评估**：
   - 滚动窗口验证（rolling window CV）：固定窗口大小，向前滚动预测
   - 预测评价指标：RMSE、MAE、MAPE、MASE

### 代码

```python
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.optimize import minimize
from scipy.stats import norm, jarque_bera, chi2
import warnings
import matplotlib.pyplot as plt


# ============================================================
# Helper Functions
# ============================================================

def acf(x, nlags):
    """Sample autocorrelation function.

    Parameters
    ----------
    x : array_like, shape (n,)
        Input time series.
    nlags : int
        Number of lags to compute.

    Returns
    -------
    acf_vals : ndarray, shape (nlags+1,)
        ACF values at lags 0, 1, ..., nlags.
    """
    n = len(x)
    xc = x - np.mean(x)
    denom = np.dot(xc, xc)
    if denom == 0:
        return np.ones(nlags + 1)
    acf_vals = np.zeros(nlags + 1)
    acf_vals[0] = 1.0
    for k in range(1, nlags + 1):
        acf_vals[k] = np.dot(xc[k:], xc[:-k]) / denom
    return acf_vals


def pacf(x, nlags):
    """Sample partial autocorrelation function (Durbin-Levinson).

    Parameters
    ----------
    x : array_like, shape (n,)
        Input time series.
    nlags : int
        Number of lags to compute.

    Returns
    -------
    pacf_vals : ndarray, shape (nlags+1,)
        PACF values at lags 0, 1, ..., nlags.
    """
    acf_vals = acf(x, nlags)
    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0

    if nlags < 1:
        return pacf_vals

    # Durbin-Levinson recursion
    phi = np.array([acf_vals[1]])
    pacf_vals[1] = acf_vals[1]

    for k in range(2, nlags + 1):
        rho = acf_vals[1:k + 1]
        num = rho[k - 1]
        denom = 1.0
        for j in range(1, k):
            num -= phi[j - 1] * rho[k - 1 - j]
            denom -= phi[j - 1] * rho[j - 1]
        phi_k = num / denom
        pacf_vals[k] = phi_k

        phi_new = np.zeros(k)
        for j in range(k - 1):
            phi_new[j] = phi[j] - phi_k * phi[k - 2 - j]
        phi_new[k - 1] = phi_k
        phi = phi_new

    return pacf_vals


def _yule_walker(x, order):
    """Estimate AR coefficients via Yule-Walker equations.

    Returns
    -------
    phi : ndarray, shape (order,)
        AR coefficients [φ₁, ..., φ_order].
    sigma2 : float
        Innovation variance estimate.
    """
    n = len(x)
    acf_vals = acf(x, order)
    r = acf_vals[1:order + 1]
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = acf_vals[abs(i - j)]

    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(R, r, rcond=None)[0]

    gamma0 = np.var(x, ddof=0)
    sigma2 = gamma0 * (1 - np.dot(phi, r))
    return phi, max(sigma2, 1e-10)


def _ljung_box(residuals, nlags=None):
    """Ljung-Box Q-test for residual autocorrelation.

    H₀: residuals are independently distributed (no autocorrelation).

    Returns
    -------
    Q_stat : float
        Ljung-Box Q test statistic.
    p_value : float
        P-value of the test.
    """
    n = len(residuals)
    if nlags is None:
        nlags = min(10, n // 5)
    nlags = min(nlags, n - 1)

    resid = residuals - np.mean(residuals)
    acf_vals = np.zeros(nlags)
    denom = np.dot(resid, resid)
    for k in range(1, nlags + 1):
        acf_vals[k - 1] = np.dot(resid[k:], resid[:-k]) / denom

    # Adjust degrees of freedom for small n
    adj = n - np.arange(1, nlags + 1)
    adj = np.maximum(adj, 1)
    Q_stat = n * (n + 2) * np.sum(acf_vals ** 2 / adj)
    p_value = 1.0 - chi2.cdf(Q_stat, nlags)
    return Q_stat, p_value


def _build_seasonal_poly(coeffs, s, sign=-1.0):
    """Build a seasonal polynomial array.

    For AR: Φ(B^s) = 1 - Φ₁B^s - Φ₂B^{2s} - ... → sign=-1.0
    For MA: Θ(B^s) = 1 + Θ₁B^s + Θ₂B^{2s} + ... → sign=+1.0

    Returns
    -------
    arr : ndarray, shape (P*s + 1,)
    """
    P = len(coeffs)
    arr_len = P * s + 1
    arr = np.zeros(arr_len)
    arr[0] = 1.0
    for i, c in enumerate(coeffs):
        lag = (i + 1) * s
        if lag < arr_len:
            arr[lag] = sign * c
    return arr


def _multiply_polys(coeffs_list):
    """Multiply several polynomials (convolve). Each represented as
    coefficient array starting at B^0.

    Returns
    -------
    result : ndarray
        Convolution of all input arrays.
    """
    result = np.array([1.0])
    for c in coeffs_list:
        result = np.convolve(result, c)
    return result


def _roots_outside_unit(poly_coeffs):
    """Check if all roots of the polynomial lie outside the unit circle.

    poly_coeffs[0] is the coefficient for B^0.
    poly represents: c[0] + c[1]*B + c[2]*B^2 + ...
    """
    roots = np.roots(poly_coeffs[::-1])  # numpy expects descending powers
    return np.all(np.abs(roots) > 1.0 - 1e-10)


# ============================================================
# ARIMA Class
# ============================================================

class ARIMA:
    """ARIMA(p,d,q) / SARIMA(p,d,q)(P,D,Q)_s model estimated via CSS-MLE.

    Parameters
    ----------
    p : int
        Non-seasonal AR order.
    d : int
        Non-seasonal differencing order.
    q : int
        Non-seasonal MA order.
    P : int, optional
        Seasonal AR order (default 0).
    D : int, optional
        Seasonal differencing order (default 0).
    Q : int, optional
        Seasonal MA order (default 0).
    s : int, optional
        Seasonal period (default None, required if P>0, D>0, or Q>0).
    include_mean : bool
        Whether to include a constant/intercept term.
    """

    def __init__(self, p, d, q, P=0, D=0, Q=0, s=None, include_mean=True):
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self.P = int(P)
        self.D = int(D)
        self.Q = int(Q)
        self.s = int(s) if s is not None else None
        self.include_mean = include_mean
        self._is_seasonal = (self.P > 0 or self.D > 0 or self.Q > 0)

        # Storage for fitted results
        self.params_ = None
        self.sigma2_ = None
        self.residuals_ = None
        self.fittedvalues_ = None
        self.residuals_diff_ = None
        self.fittedvalues_diff_ = None
        self.ar_full_ = None
        self.ma_full_ = None
        self._psi_weights = None
        self._y_original = None
        self._n_original = None
        self._y_diff = None  # differenced series used for estimation
        self._sse = None

    # ----------------------------------------------------------
    # Differencing
    # ----------------------------------------------------------

    def _apply_differencing(self, y):
        """Apply (1-B)^d (1-B^s)^D differencing."""
        x = y.copy()
        # Regular differencing
        if self.d > 0:
            x = np.diff(x, n=self.d)
        # Seasonal differencing
        if self.D > 0 and self.s is not None:
            for _ in range(self.D):
                x = x[self.s:] - x[:-self.s]
        return x

    def _reconstruct_from_diff(self, y_diff, y_original, n_steps=0, 
                                forecasts_diff=None):
        """Reconstruct original scale from differenced series.

        For in-sample fitted values (one-step ahead):
        - d=0: fitted_orig = fitted_diff
        - d=1: ŷ_t = y_{t-1} + Ŵ_t
        - d=2: ŷ_t = 2y_{t-1} - y_{t-2} + Ŵ_t

        For out-of-sample forecasts:
        - d=0: ŷ_{t+h|t} = Ŵ_{t+h|t}
        - d=1: ŷ_{t+h|t} = y_t + Σ_{i=1}^h Ŵ_{t+i|t}
        - d=2: iterative reconstruction
        """
        n_diff = len(y_diff)
        n_orig = len(y_original)

        if self.d == 0 and self.D == 0:
            return y_diff.copy()

        # In-sample reconstruction (one-step ahead fitted values)
        fitted_orig = np.full(n_orig, np.nan)

        if not self._is_seasonal and self.D == 0:
            # Standard ARIMA reconstruction
            shift = self.d
            for i in range(shift, n_orig):
                if self.d == 1:
                    fitted_orig[i] = y_original[i - 1] + y_diff[i - shift]
                elif self.d == 2:
                    fitted_orig[i] = (2 * y_original[i - 1] 
                                      - y_original[i - 2] + y_diff[i - shift])
                elif self.d == 0:
                    fitted_orig[i] = y_diff[i]
        else:
            # Seasonal or high-order differencing: use cumulative reconstruction
            # This is approximate; for production, use statsmodels
            try:
                cum = y_diff.copy().astype(np.float64)
                for _ in range(self.d):
                    # Pad at the beginning and cumsum
                    cum = np.cumsum(cum)
                # This gives approximate level reconstruction
                fitted_orig = cum
            except Exception:
                fitted_orig = y_diff.copy()

        return fitted_orig

    # ----------------------------------------------------------
    # CSS Objective
    # ----------------------------------------------------------

    def _build_full_polys(self, phi, theta, Phi, Theta):
        """Build full AR and MA polynomials by multiplying regular and seasonal parts.

        Returns
        -------
        ar_full : ndarray
            Full AR polynomial coefficients (ar_full[0] = 1).
        ma_full : ndarray
            Full MA polynomial coefficients (ma_full[0] = 1).
        """
        # AR polynomials
        ar_reg = np.array([1.0] + [-phi[i] for i in range(self.p)])
        if self._is_seasonal and self.P > 0:
            ar_sea = _build_seasonal_poly(Phi, self.s, sign=-1.0)
        else:
            ar_sea = np.array([1.0])
        ar_full = _multiply_polys([ar_reg, ar_sea])

        # MA polynomials
        ma_reg = np.array([1.0] + [theta[i] for i in range(self.q)])
        if self._is_seasonal and self.Q > 0:
            ma_sea = _build_seasonal_poly(Theta, self.s, sign=+1.0)
        else:
            ma_sea = np.array([1.0])
        ma_full = _multiply_polys([ma_reg, ma_sea])

        return ar_full, ma_full

    def _css(self, params, y):
        """Conditional sum of squares.

        CSS recursion: e[t] = y[t] + Σ ar_full[k] y[t-k] - Σ ma_full[k] e[t-k]
        where ar_full[k] already includes the negative sign for AR terms.
        """
        n = len(y)

        # Parse parameters
        idx = 0
        phi = params[idx:idx + self.p] if self.p > 0 else np.array([])
        idx += self.p
        theta = params[idx:idx + self.q] if self.q > 0 else np.array([])
        idx += self.q
        Phi = params[idx:idx + self.P] if self.P > 0 else np.array([])
        idx += self.P
        Theta = params[idx:idx + self.Q] if self.Q > 0 else np.array([])
        idx += self.Q
        mu = params[idx] if self.include_mean else 0.0

        # Build full polynomials
        ar_full, ma_full = self._build_full_polys(phi, theta, Phi, Theta)
        self.ar_full_ = ar_full
        self.ma_full_ = ma_full

        yc = y - mu
        e = np.zeros(n)

        for t in range(n):
            # AR part: + Σ ar_full[k] * y[t-k]
            ar_part = 0.0
            max_ar_lag = min(len(ar_full) - 1, t)
            for k in range(1, max_ar_lag + 1):
                ar_part += ar_full[k] * yc[t - k]

            # MA part: - Σ ma_full[k] * e[t-k]
            ma_part = 0.0
            max_ma_lag = min(len(ma_full) - 1, t)
            for k in range(1, max_ma_lag + 1):
                ma_part += ma_full[k] * e[t - k]

            e[t] = yc[t] + ar_part - ma_part

        return np.sum(e ** 2)

    def _compute_residuals_fitted(self, y, params):
        """Compute residuals and fitted values for the differenced series."""
        n = len(y)

        idx = 0
        phi = params[idx:idx + self.p] if self.p > 0 else np.array([])
        idx += self.p
        theta = params[idx:idx + self.q] if self.q > 0 else np.array([])
        idx += self.q
        Phi = params[idx:idx + self.P] if self.P > 0 else np.array([])
        idx += self.P
        Theta = params[idx:idx + self.Q] if self.Q > 0 else np.array([])
        idx += self.Q
        mu = params[idx] if self.include_mean else 0.0

        ar_full, ma_full = self._build_full_polys(phi, theta, Phi, Theta)

        yc = y - mu
        e = np.zeros(n)
        fitted = np.zeros(n)

        for t in range(n):
            ar_part = 0.0
            max_ar_lag = min(len(ar_full) - 1, t)
            for k in range(1, max_ar_lag + 1):
                ar_part += ar_full[k] * yc[t - k]

            ma_part = 0.0
            max_ma_lag = min(len(ma_full) - 1, t)
            for k in range(1, max_ma_lag + 1):
                ma_part += ma_full[k] * e[t - k]

            # fitted = μ - Σ ar_full[k] y[t-k] + Σ ma_full[k] e[t-k]
            # But note: e[t] = y[t] + ar_part - ma_part
            #        = y[t] + ar_part - ma_part
            # So fitted[t] = y[t] - e[t] = -ar_part + ma_part + ... 
            # Actually: y[t] = fitted[t] + e[t]
            # From: e[t] = y[t] + ar_part - ma_part
            # So: y[t] = e[t] - ar_part + ma_part
            # fitted[t] = -ar_part + ma_part + mu
            fitted[t] = mu - ar_part + ma_part
            e[t] = yc[t] + ar_part - ma_part

        return e, fitted

    def _compute_psi_weights(self, n_weights=50):
        """Compute psi-weights (infinite MA representation) for forecast intervals.

        For ARMA: ψ(B) = θ(B) / φ(B)
        ψ_0 = 1
        ψ_j = Σ_{i=1}^{min(p,j)} φ_i ψ_{j-i} + θ_j   for j >= 1
        where θ_j = 0 for j > q.
        """
        if self.ar_full_ is None or self.ma_full_ is None:
            return None

        # Extract AR and MA coefficients from full polynomials
        # ar_full[0] = 1, ar_full[k] = -φ_k + cross terms
        # For psi-weights, the model is: φ(B) y_t = θ(B) ε_t
        # ψ(B) = θ(B) / φ(B)
        # ψ is the convolution inverse

        psi = np.zeros(n_weights)
        psi[0] = 1.0

        ar_coeffs = -self.ar_full_[1:]  # φ_i = -ar_full[i] (since ar_full[i] = -φ_i)
        ma_coeffs = self.ma_full_[1:]   # θ_i = ma_full[i]

        p_eff = len(ar_coeffs)
        q_eff = len(ma_coeffs)

        for j in range(1, n_weights):
            total = 0.0
            for i in range(1, min(p_eff, j) + 1):
                total += ar_coeffs[i - 1] * psi[j - i]
            if j <= q_eff:
                total += ma_coeffs[j - 1]
            psi[j] = total

        self._psi_weights = psi
        return psi

    def _forecast_variance(self, h, d):
        """Compute forecast error variance for h-step ahead on original scale.

        For d=0: Var(h) = σ² * Σ_{i=0}^{h-1} ψ_i²
        For d=1: Var(h) = σ² * Σ_{i=0}^{h-1} (cumsum(ψ)_i)²
        For d=2: Var(h) = σ² * Σ_{i=0}^{h-1} (double_cumsum(ψ)_i)²
        """
        if self._psi_weights is None:
            self._compute_psi_weights(n_weights=max(50, h + 10))

        total_d = self.d + self.D * (self.s if self.s else 0)
        psi = self._psi_weights[:h]

        # Apply cumulative integration for each differencing order
        for _ in range(total_d):
            psi = np.cumsum(psi)

        return self.sigma2_ * np.sum(psi ** 2)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def fit(self, y):
        """Estimate model parameters via CSS-MLE.

        Parameters
        ----------
        y : array_like, shape (n,)
            Time series data.

        Returns
        -------
        self : ARIMA
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        self._y_original = y.copy()
        self._n_original = len(y)

        # Apply differencing
        y_diff = self._apply_differencing(y)
        self._y_diff = y_diff
        n_eff = len(y_diff)

        if n_eff < max(self.p + self.q + self.P + self.Q, 1):
            raise ValueError(
                f"Effective sample size ({n_eff}) too small "
                f"for model orders (p={self.p}, d={self.d}, q={self.q}, "
                f"P={self.P}, D={self.D}, Q={self.Q})"
            )

        n_params = self.p + self.q + self.P + self.Q
        if self.include_mean:
            n_params += 1

        if n_params == 0:
            # Mean-only model
            mu = np.mean(y_diff)
            self.params_ = np.array([mu]) if self.include_mean else np.array([])
            self.sigma2_ = np.var(y_diff, ddof=0)
            self.residuals_ = y_diff - mu
            self.fittedvalues_diff_ = np.full(n_eff, mu)
            self._sse = np.sum(self.residuals_ ** 2)
            self._reconstruct(y)
            return self

        # Initial parameter guesses
        x0 = []
        bounds = []

        # AR initial values via Yule-Walker
        if self.p > 0:
            try:
                phi_init, _ = _yule_walker(y_diff, self.p)
                x0.extend(phi_init)
            except Exception:
                x0.extend([0.0] * self.p)
            for _ in range(self.p):
                bounds.append((-0.999, 0.999))

        # MA initial values (small)
        if self.q > 0:
            x0.extend([0.0] * self.q)
            for _ in range(self.q):
                bounds.append((-0.999, 0.999))

        # Seasonal AR initial values
        if self.P > 0:
            x0.extend([0.0] * self.P)
            for _ in range(self.P):
                bounds.append((-0.999, 0.999))

        # Seasonal MA initial values
        if self.Q > 0:
            x0.extend([0.0] * self.Q)
            for _ in range(self.Q):
                bounds.append((-0.999, 0.999))

        # Mean
        if self.include_mean:
            x0.append(np.mean(y_diff))
            bounds.append((None, None))

        # Optimize CSS
        x0 = np.array(x0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._css, x0, args=(y_diff,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-8}
            )

            # If L-BFGS-B fails, try Nelder-Mead
            if not result.success:
                try:
                    result = minimize(
                        self._css, x0, args=(y_diff,),
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7}
                    )
                except Exception:
                    pass

        self.params_ = result.x
        self._sse = result.fun
        self.sigma2_ = result.fun / n_eff

        # Compute residuals and fitted values
        resid, fitted = self._compute_residuals_fitted(y_diff, self.params_)
        self.residuals_diff_ = resid
        self.fittedvalues_diff_ = fitted

        # Reconstruct original scale
        self._reconstruct(y)

        # Compute psi-weights
        self._compute_psi_weights(n_weights=50)

        return self

    def _reconstruct(self, y_original):
        """Reconstruct fitted values on original scale from differenced series."""
        if self.d == 0 and self.D == 0:
            self.fittedvalues_ = self.fittedvalues_diff_.copy()
            self.residuals_ = self.residuals_diff_.copy()
            return

        n_diff = len(self.fittedvalues_diff_)
        n_orig = len(y_original)
        fitted_orig = np.full(n_orig, np.nan)
        resid_orig = np.full(n_orig, np.nan)

        if not self._is_seasonal:
            shift = self.d
            for i in range(shift, n_orig):
                if self.d == 1:
                    fitted_orig[i] = y_original[i - 1] + self.fittedvalues_diff_[i - shift]
                elif self.d == 2:
                    fitted_orig[i] = (2 * y_original[i - 1] 
                                      - y_original[i - 2] 
                                      + self.fittedvalues_diff_[i - shift])
                elif self.d == 0:
                    fitted_orig[i] = self.fittedvalues_diff_[i]
                resid_orig[i] = y_original[i] - fitted_orig[i]
        else:
            # For seasonal models, use the differenced fitted values directly
            # and reconstruct approximately
            reconstructed = self._y_diff.copy() - self.residuals_diff_
            # Inverse differencing by cumulative sum (approximate)
            inv = reconstructed.copy()
            for _ in range(self.d):
                inv = np.cumsum(inv)
            # Pad to original length
            fitted_orig = np.full(n_orig, np.nan)
            offset = n_orig - len(inv)
            if offset >= 0:
                fitted_orig[offset:] = inv
                resid_orig[offset:] = y_original[offset:] - fitted_orig[offset:]

        self.fittedvalues_ = fitted_orig
        self.residuals_ = resid_orig

    def predict(self):
        """Return in-sample one-step-ahead fitted values (original scale).

        Returns
        -------
        fitted : ndarray, shape (n_original,)
            Fitted values (NaN for initial observations lost to differencing).
        """
        if self.fittedvalues_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.fittedvalues_

    def forecast(self, n_steps=10, level=0.95):
        """Generate out-of-sample forecasts with prediction intervals.

        Parameters
        ----------
        n_steps : int
            Number of forecast steps.
        level : float
            Confidence level for prediction intervals (default 0.95).

        Returns
        -------
        forecasts : ndarray, shape (n_steps,)
            Point forecasts.
        ci_lower : ndarray, shape (n_steps,)
            Lower bound of prediction interval.
        ci_upper : ndarray, shape (n_steps,)
            Upper bound of prediction interval.
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        y_diff = self._y_diff
        n_diff = len(y_diff)

        # Parse parameters
        idx = 0
        phi = self.params_[idx:idx + self.p] if self.p > 0 else np.array([])
        idx += self.p
        theta = self.params_[idx:idx + self.q] if self.q > 0 else np.array([])
        idx += self.q
        Phi = self.params_[idx:idx + self.P] if self.P > 0 else np.array([])
        idx += self.P
        Theta = self.params_[idx:idx + self.Q] if self.Q > 0 else np.array([])
        idx += self.Q
        mu = self.params_[idx] if self.include_mean else 0.0

        ar_full, ma_full = self._build_full_polys(phi, theta, Phi, Theta)

        # Extend the series for forecasting
        y_ext = np.concatenate([y_diff - mu, np.zeros(n_steps)])
        e_ext = np.concatenate([self.residuals_diff_, np.zeros(n_steps)])
        n_total = len(y_ext)

        # Iterative multi-step forecast
        for t in range(n_diff, n_total):
            ar_part = 0.0
            max_ar_lag = min(len(ar_full) - 1, t)
            for k in range(1, max_ar_lag + 1):
                ar_part += ar_full[k] * y_ext[t - k]

            ma_part = 0.0
            max_ma_lag = min(len(ma_full) - 1, t)
            for k in range(1, max_ma_lag + 1):
                ma_part += ma_full[k] * e_ext[t - k]

            y_ext[t] = -ar_part + ma_part  # forecast of differenced series
            e_ext[t] = 0.0  # future errors = 0 for point forecast

        forecasts_diff = y_ext[n_diff:] + mu

        # Reconstruct on original scale
        forecasts = np.zeros(n_steps)
        if self.d == 0 and self.D == 0:
            forecasts = forecasts_diff
        elif self.d == 1:
            last_obs = self._y_original[-1]
            cumsum = 0.0
            for h in range(n_steps):
                cumsum += forecasts_diff[h]
                forecasts[h] = last_obs + cumsum
        elif self.d == 2:
            y1 = self._y_original[-1]
            y2 = self._y_original[-2]
            cumsum1, cumsum2 = 0.0, 0.0
            for h in range(n_steps):
                cumsum2 += cumsum1
                cumsum1 += forecasts_diff[h]
                forecasts[h] = y1 + (h + 1) * (y1 - y2) + cumsum2
        else:
            # Fallback: cumulative sum reconstruction
            cum = forecasts_diff.copy()
            for _ in range(self.d):
                cum = np.cumsum(cum)
            forecasts = self._y_original[-1] + cum

        # Prediction intervals using psi-weights
        z = norm.ppf(1 - (1 - level) / 2)
        ci_lower = np.zeros(n_steps)
        ci_upper = np.zeros(n_steps)

        for h in range(1, n_steps + 1):
            var_h = self._forecast_variance(h, self.d)
            se_h = np.sqrt(var_h)
            ci_lower[h - 1] = forecasts[h - 1] - z * se_h
            ci_upper[h - 1] = forecasts[h - 1] + z * se_h

        return forecasts, ci_lower, ci_upper

    def diagnostics(self, figsize=(12, 8)):
        """Residual diagnostic plots.

        Plots: (1) Residuals over time, (2) ACF of residuals,
        (3) Q-Q plot, (4) Histogram with normal overlay.

        Also prints Ljung-Box and Jarque-Bera test results.
        """
        if self.residuals_diff_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        resid = self.residuals_diff_
        n = len(resid)

        # Ljung-Box test
        lb_pval = None
        try:
            lb_q, lb_pval = _ljung_box(resid, nlags=min(20, n // 5))
        except Exception:
            lb_q, lb_pval = np.nan, np.nan

        # Jarque-Bera test
        try:
            jb_stat, jb_pval = jarque_bera(resid)
        except Exception:
            jb_stat, jb_pval = np.nan, np.nan

        print("=" * 60)
        print("Residual Diagnostics")
        print("=" * 60)
        print(f"Ljung-Box Q({min(20, n // 5)}): {lb_q:.4f}, p-value: {lb_pval:.4f}")
        print(f"Jarque-Bera: {jb_stat:.4f}, p-value: {jb_pval:.4f}")
        print()

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Residuals over time
        axes[0, 0].plot(resid, 'o-', markersize=3, linewidth=0.8)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')

        # 2. ACF of residuals
        nlags = min(30, n // 4)
        acf_vals = acf(resid, nlags)
        axes[0, 1].bar(range(nlags + 1), acf_vals, width=0.3)
        conf = 1.96 / np.sqrt(n)
        axes[0, 1].axhline(y=conf, color='b', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=-conf, color='b', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
        axes[0, 1].set_title('ACF of Residuals')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('ACF')

        # 3. Q-Q plot
        from scipy.stats import probplot
        probplot(resid, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')

        # 4. Histogram
        axes[1, 1].hist(resid, bins=30, density=True, alpha=0.6, label='Residuals')
        x_range = np.linspace(resid.min(), resid.max(), 100)
        axes[1, 1].plot(x_range, norm.pdf(x_range, np.mean(resid), np.std(resid)),
                        'r-', label='Normal')
        axes[1, 1].set_title('Histogram')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        return fig

    def aic(self):
        """Akaike Information Criterion (CSS-based).

        AIC = n * log(SSE/n) + 2*k
        where k = p + q + P + Q + (1 if include_mean else 0).
        """
        if self._sse is None:
            raise RuntimeError("Model not fitted yet.")
        n_eff = len(self._y_diff)
        k = self.p + self.q + self.P + self.Q + (1 if self.include_mean else 0)
        return n_eff * np.log(self._sse / n_eff) + 2 * k

    def bic(self):
        """Bayesian Information Criterion.

        BIC = n * log(SSE/n) + k * log(n)
        """
        if self._sse is None:
            raise RuntimeError("Model not fitted yet.")
        n_eff = len(self._y_diff)
        k = self.p + self.q + self.P + self.Q + (1 if self.include_mean else 0)
        return n_eff * np.log(self._sse / n_eff) + k * np.log(n_eff)

    def summary(self):
        """Print a summary table of estimated parameters."""
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet.")

        print("=" * 60)
        print(f"ARIMA{self.p, self.d, self.q}", end="")
        if self._is_seasonal:
            print(f"({self.P, self.D, self.Q, self.s})", end="")
        print("  —  CSS-MLE Estimation")
        print("=" * 60)

        idx = 0
        param_names = []
        if self.p > 0:
            for i in range(self.p):
                param_names.append(f'ar.L{i + 1}')
        if self.q > 0:
            for i in range(self.q):
                param_names.append(f'ma.L{i + 1}')
        if self.P > 0:
            for i in range(self.P):
                param_names.append(f'sar.L{i + 1}.S{self.s}')
        if self.Q > 0:
            for i in range(self.Q):
                param_names.append(f'sma.L{i + 1}.S{self.s}')
        if self.include_mean:
            param_names.append('mean')

        print(f"{'Name':<20} {'Coef.':<12}")
        print("-" * 32)
        for name, val in zip(param_names, self.params_):
            print(f"{name:<20} {val:<12.6f}")

        print("-" * 32)
        print(f"{'sigma²':<20} {self.sigma2_:<12.6f}")
        print(f"{'AIC':<20} {self.aic():<12.2f}")
        print(f"{'BIC':<20} {self.bic():<12.2f}")
        print(f"{'n_eff':<20} {len(self._y_diff):<12}")
        print("=" * 60)

        return self


# ============================================================
# Auto-ARIMA
# ============================================================

def auto_arima(y, max_p=5, max_d=2, max_q=5,
               max_P=2, max_D=1, max_Q=2, s=None,
               seasonal=True, ic='aic', stepwise=True,
               trace=False, include_mean=True):
    """Automatic ARIMA order selection via information criterion search.

    Searches over combinations of (p, d, q) and optionally (P, D, Q, s)
    to find the model with the lowest AIC (or BIC).

    Parameters
    ----------
    y : array_like
        Time series data.
    max_p : int
        Maximum non-seasonal AR order.
    max_d : int
        Maximum non-seasonal differencing order.
    max_q : int
        Maximum non-seasonal MA order.
    max_P : int
        Maximum seasonal AR order.
    max_D : int
        Maximum seasonal differencing order.
    max_Q : int
        Maximum seasonal MA order.
    s : int, optional
        Seasonal period. If None, inferred from seasonal=True/False.
    seasonal : bool
        Whether to search seasonal orders.
    ic : str
        Information criterion: 'aic' or 'bic'.
    stepwise : bool
        If True, use stepwise search (faster). If False, full grid search.
    trace : bool
        If True, print progress.
    include_mean : bool
        Include constant in all candidate models.

    Returns
    -------
    best_model : ARIMA
        Best model according to the selected IC.
    best_order : tuple
        (p, d, q) of the best model.
    best_seasonal_order : tuple or None
        (P, D, Q, s) of the best model, if seasonal.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    best_ic = np.inf
    best_model = None
    best_order = None
    best_seasonal = None

    # Determine d first (can be pre-specified or use unit root tests)
    # For simplicity here, search over d values
    d_values = range(max_d + 1)

    # Seasonal detection
    if seasonal and s is not None:
        seasonal_combos = [(P, D, Q) for P in range(max_P + 1)
                           for D in range(max_D + 1)
                           for Q in range(max_Q + 1)]
    elif seasonal:
        seasonal_combos = [(0, 0, 0)]
    else:
        seasonal_combos = [(0, 0, 0)]

    if stepwise:
        # Stepwise search: start simple, expand in promising directions
        best_p, best_d, best_q = 0, 0, 0
        best_P, best_D, best_Q = 0, 0, 0

        # Initial search over d with p=0, q=0
        for d in d_values:
            try:
                model = ARIMA(0, d, 0, include_mean=include_mean).fit(y)
                current_ic = model.aic() if ic == 'aic' else model.bic()
                if current_ic < best_ic:
                    best_ic = current_ic
                    best_d = d
            except Exception:
                continue

        # Search p
        improved = True
        n_iter = 0
        max_iter = 20
        while improved and n_iter < max_iter:
            improved = False
            n_iter += 1

            # Try increasing p
            for p in range(max(0, best_p - 1), min(max_p + 1, best_p + 3)):
                if p == best_p:
                    continue
                try:
                    model = ARIMA(p, best_d, best_q, 
                                  P=best_P, D=best_D, Q=best_Q, s=s,
                                  include_mean=include_mean).fit(y)
                    current_ic = model.aic() if ic == 'aic' else model.bic()
                    if current_ic < best_ic:
                        best_ic = current_ic
                        best_p = p
                        improved = True
                except Exception:
                    continue

            # Try increasing q
            for q in range(max(0, best_q - 1), min(max_q + 1, best_q + 3)):
                if q == best_q:
                    continue
                try:
                    model = ARIMA(best_p, best_d, q,
                                  P=best_P, D=best_D, Q=best_Q, s=s,
                                  include_mean=include_mean).fit(y)
                    current_ic = model.aic() if ic == 'aic' else model.bic()
                    if current_ic < best_ic:
                        best_ic = current_ic
                        best_q = q
                        improved = True
                except Exception:
                    continue

            # Try seasonal
            if seasonal and s is not None:
                for P, D, Q in seasonal_combos:
                    if P == best_P and D == best_D and Q == best_Q:
                        continue
                    try:
                        model = ARIMA(best_p, best_d, best_q,
                                      P=P, D=D, Q=Q, s=s,
                                      include_mean=include_mean).fit(y)
                        current_ic = model.aic() if ic == 'aic' else model.bic()
                        if current_ic < best_ic:
                            best_ic = current_ic
                            best_P, best_D, best_Q = P, D, Q
                            improved = True
                    except Exception:
                        continue

            if trace:
                print(f"  Step {n_iter}: ARIMA({best_p},{best_d},{best_q})"
                      f"({best_P},{best_D},{best_Q})[{s}] — "
                      f"{ic.upper()}: {best_ic:.2f}")

        best_model = ARIMA(best_p, best_d, best_q,
                           P=best_P, D=best_D, Q=best_Q, s=s,
                           include_mean=include_mean).fit(y)
        best_order = (best_p, best_d, best_q)
        best_seasonal = (best_P, best_D, best_Q, s) if seasonal and s else None

    else:
        # Full grid search
        p_values = range(max_p + 1)
        q_values = range(max_q + 1)
        total = len(d_values) * len(p_values) * len(q_values) * len(seasonal_combos)
        count = 0

        for d in d_values:
            for p in p_values:
                for q in q_values:
                    for P, D, Q in seasonal_combos:
                        if trace:
                            count += 1
                            print(f"  [{count}/{total}] Trying "
                                  f"ARIMA({p},{d},{q})({P},{D},{Q})[{s}]...",
                                  end=' ')
                        try:
                            model = ARIMA(p, d, q, P=P, D=D, Q=Q, s=s,
                                          include_mean=include_mean).fit(y)
                            current_ic = model.aic() if ic == 'aic' else model.bic()
                            if trace:
                                print(f"{ic.upper()}: {current_ic:.2f}")
                            if current_ic < best_ic:
                                best_ic = current_ic
                                best_model = model
                                best_order = (p, d, q)
                                best_seasonal = (P, D, Q, s) if seasonal and s else None
                        except Exception as e:
                            if trace:
                                print(f"FAIL: {e}")

    return best_model, best_order, best_seasonal


# ============================================================
# Production Usage: statsmodels
# ============================================================

def statsmodels_example():
    """Example of production-grade ARIMA/SARIMAX using statsmodels.
    
    To use: uncomment and run with your data.

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # SARIMA(1,1,1)(1,1,1)_12
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(method='innovations_mle', maxiter=200)
    print(result.summary())
    
    # Diagnostics
    result.plot_diagnostics(figsize=(12, 8))
    plt.show()
    
    # Forecast
    forecast_result = result.get_forecast(steps=12)
    forecasts = forecast_result.predicted_mean
    ci = forecast_result.conf_int(alpha=0.05)
    
    # Model selection with auto_arima from pmdarima
    # pip install pmdarima
    # import pmdarima as pm
    # model = pm.auto_arima(data, seasonal=True, m=12, 
    #                       stepwise=True, trace=True,
    #                       information_criterion='aic')
    """
    pass


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("ARIMA/SARIMA — From-Scratch Implementation Demo")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. Generate synthetic ARIMA data
    # -------------------------------------------------------
    np.random.seed(42)
    n = 300

    # Generate ARIMA(1,1,1): y_t - y_{t-1} = 0.7*(y_{t-1} - y_{t-2}) 
    #                        + ε_t + 0.3*ε_{t-1}
    ar_coef = 0.7
    ma_coef = 0.3
    errors = np.random.normal(0, 0.5, n + 100)

    # Generate ARMA(1,1) on differences
    w = np.zeros(n + 100)
    for t in range(1, n + 100):
        w[t] = ar_coef * w[t - 1] + errors[t] + ma_coef * errors[t - 1]

    # Cumsum to get ARIMA(1,1,1)
    y = np.cumsum(w)[100:]  # discard burn-in
    y += 10.0  # add a level shift

    print(f"\nGenerated ARIMA(1,1,1) series, n={n}")
    print(f"True params: φ₁={ar_coef}, θ₁={ma_coef}")

    # Split
    train, test = y[:250], y[250:]

    # -------------------------------------------------------
    # 2. Fit ARIMA from scratch
    # -------------------------------------------------------
    print("\n--- Fitting ARIMA(1,1,1) (from scratch) ---")
    model = ARIMA(p=1, d=1, q=1, include_mean=True)
    model.fit(train)
    model.summary()

    # Diagnostics
    print("\nGenerating diagnostic plots...")
    model.diagnostics()

    # Forecast
    print("\n--- Forecasting next 10 steps ---")
    fc, ci_l, ci_u = model.forecast(n_steps=len(test), level=0.95)

    # Evaluate
    valid = ~np.isnan(fc) & ~np.isnan(test)
    if np.any(valid):
        rmse = np.sqrt(np.mean((fc[valid] - test[valid]) ** 2))
        mae = np.mean(np.abs(fc[valid] - test[valid]))
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")

    # -------------------------------------------------------
    # 3. Auto-ARIMA
    # -------------------------------------------------------
    print("\n--- Auto-ARIMA (stepwise search) ---")
    best, order, s_order = auto_arima(
        train, max_p=3, max_d=2, max_q=3, 
        stepwise=True, trace=True
    )
    print(f"\nBest model: ARIMA{order}")
    best.summary()

    # -------------------------------------------------------
    # 4. Plot results
    # -------------------------------------------------------
    plt.figure(figsize=(12, 6))
    time_idx = np.arange(len(y))
    plt.plot(time_idx, y, 'b-', label='Observed', alpha=0.7)
    plt.plot(time_idx[:250], model.predict(), 'r-', 
             label='Fitted (in-sample)', alpha=0.8, linewidth=1.5)
    plt.plot(time_idx[250:], fc, 'g--', label='Forecast', linewidth=2)
    plt.fill_between(time_idx[250:], ci_l, ci_u, 
                     color='g', alpha=0.2, label='95% PI')
    plt.axvline(x=249, color='gray', linestyle=':', alpha=0.7)
    plt.title('ARIMA(1,1,1) — Fitted and Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # -------------------------------------------------------
    # 5. Stationary ARMA example
    # -------------------------------------------------------
    print("\n--- ARMA(2,0) on stationary data ---")
    np.random.seed(123)
    n_arma = 500
    e = np.random.normal(0, 0.5, n_arma)
    x = np.zeros(n_arma)
    for t in range(2, n_arma):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + e[t]

    arma_model = ARIMA(p=2, d=0, q=0, include_mean=False)
    arma_model.fit(x)
    arma_model.summary()
    print(f"True φ: [0.6, -0.3]")

    # ACF/PACF plots for order identification
    nlags = 20
    acf_vals = acf(x, nlags)
    pacf_vals = pacf(x, nlags)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(nlags + 1), acf_vals, width=0.3)
    axes[0].axhline(y=1.96 / np.sqrt(n_arma), color='b', ls='--', alpha=0.5)
    axes[0].axhline(y=-1.96 / np.sqrt(n_arma), color='b', ls='--', alpha=0.5)
    axes[0].set_title('ACF')
    axes[1].bar(range(nlags + 1), pacf_vals, width=0.3)
    axes[1].axhline(y=1.96 / np.sqrt(n_arma), color='b', ls='--', alpha=0.5)
    axes[1].axhline(y=-1.96 / np.sqrt(n_arma), color='b', ls='--', alpha=0.5)
    axes[1].set_title('PACF')
    plt.tight_layout()
    plt.show()
```

### 基于 statsmodels 的生产用法

```python
# pip install statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# ---- 1. 单位根检验 ----
def stationarity_test(x):
    """ADF + KPSS 联合检验"""
    adf_stat, adf_pval = adfuller(x)[:2]
    kpss_stat, kpss_pval = kpss(x, regression='c')[:2]
    print(f"ADF:  stat={adf_stat:.4f}, pval={adf_pval:.4f}")
    print(f"KPSS: stat={kpss_stat:.4f}, pval={kpss_pval:.4f}")
    # ADF H0: unit root exists (p<0.05 → stationary)
    # KPSS H0: stationary (p>0.05 → stationary)
    return adf_pval < 0.05 and kpss_pval > 0.05

# ---- 2. 模型选择与拟合 ----
# 方法 A: 基于 ACF/PACF 手动定阶
plot_acf(diff_series, lags=30)
plot_pacf(diff_series, lags=30)

# 方法 B: 自动定阶
# pip install pmdarima
import pmdarima as pm
auto_model = pm.auto_arima(
    train, seasonal=True, m=12,
    stepwise=True, trace=True,
    information_criterion='aic',
    max_p=5, max_d=2, max_q=5,
    max_P=2, max_D=1, max_Q=2
)
print(auto_model.summary())

# 方法 C: SARIMAX (最灵活)
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=True,
    enforce_invertibility=True,
    trend='c'
)
result = model.fit(method='innovations_mle', maxiter=200, disp=False)
print(result.summary())

# ---- 3. 残差诊断 ----
result.plot_diagnostics(figsize=(12, 8))
plt.show()

# ---- 4. 预测 ----
forecast_result = result.get_forecast(steps=12)
forecasts = forecast_result.predicted_mean
ci = forecast_result.conf_int(alpha=0.05)

# ---- 5. 滚动窗口评估 ----
def rolling_forecast(y, order, seasonal_order, step=1, window=100):
    """滚动窗口预测验证"""
    n = len(y)
    forecasts = []
    actuals = []
    for start in range(0, n - window, step):
        train = y[start:start + window]
        test = y[start + window:start + window + step]
        try:
            m = SARIMAX(train, order=order, 
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False).fit(disp=False, maxiter=100)
            fc = m.forecast(step)
            forecasts.extend(fc)
            actuals.extend(test)
        except Exception:
            continue
    return np.array(forecasts), np.array(actuals)

fc, ac = rolling_forecast(data, (1, 1, 1), (0, 0, 0, 0))
rmse = np.sqrt(np.mean((fc - ac) ** 2))
print(f"Rolling CV RMSE: {rmse:.4f}")
```

## 参考文献

1. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control* (1st ed.). Holden-Day.
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley. DOI: 10.1002/9781118619193
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
4. Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723. DOI: 10.1109/TAC.1974.1100705
5. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
6. Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.). Springer. DOI: 10.1007/978-3-319-29854-2
