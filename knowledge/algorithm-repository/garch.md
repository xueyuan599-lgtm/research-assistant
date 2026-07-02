# GARCH — 广义自回归条件异方差模型

- **来源**: Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of the Royal Statistical Society, Series B*, 48(2), 127–145.
- **补充**: Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987–1007.
- **方法类别**: 时间序列 / 金融计量

## 数学设定

### 模型框架
GARCH 模型刻画时间序列中 **波动率聚集（volatility clustering）** 现象：大波动后跟大波动，小波动后跟小波动。设收益率序列为 $r_t$，其条件均值和条件方差分别为 $\mu_t = \mathbb{E}[r_t \mid \mathcal{F}_{t-1}]$ 和 $\sigma_t^2 = \text{Var}(r_t \mid \mathcal{F}_{t-1})$。

#### ARCH(q) — Engle (1982)
ARCH 模型将条件方差表示为过去 $q$ 期 **平方冲击** 的线性函数：
$$
\varepsilon_t = r_t - \mu_t, \quad \varepsilon_t \mid \mathcal{F}_{t-1} \sim (0, \sigma_t^2)
$$
$$
\sigma_t^2 = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \alpha_2 \varepsilon_{t-2}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2
$$
其中 $\alpha_0 > 0$，$\alpha_i \geq 0 \ \forall i \geq 1$ 保证方差非负。

#### GARCH(p, q) — Bollerslev (1986)
GARCH 在 ARCH 基础上引入 **条件方差的滞后项**，用更少的参数捕捉长记忆性：
$$
\varepsilon_t = \sigma_t \cdot z_t, \quad z_t \sim \text{i.i.d.} \ (0,1)
$$
$$
\sigma_t^2 = \alpha_0 + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2
$$
- $\alpha_0 > 0,\ \alpha_i \geq 0,\ \beta_j \geq 0$（方差非负条件）
- $z_t$ 通常假定为 **标准正态分布** 或 **标准化 t 分布**（自由度 $\nu$）

### 平稳性条件
**协方差平稳性** 要求：
$$
\sum_{i=1}^{q} \alpha_i + \sum_{j=1}^{p} \beta_j < 1
$$
当该条件满足时，**无条件方差** 为常数：
$$
\bar{\sigma}^2 = \frac{\alpha_0}{1 - \sum_{i=1}^q \alpha_i - \sum_{j=1}^p \beta_j}
$$

### 参数估计：MLE
给定观测值 $\{\varepsilon_1, \dots, \varepsilon_T\}$，GARCH 模型通过 **极大似然估计（MLE）** 求解。正态对数似然：

$$
\log L(\theta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ \log(2\pi) + \log(\sigma_t^2) + \frac{\varepsilon_t^2}{\sigma_t^2} \right]
$$

实践中通过 **方差目标化（variance targeting）** 设定 $\alpha_0$ 初始值来加速收敛：
$$
\alpha_0 = \bar{\sigma}^2 \left(1 - \sum \alpha_i - \sum \beta_j\right)
$$
其中 $\bar{\sigma}^2$ 用样本方差 $\hat{\sigma}^2 = \frac{1}{T}\sum \varepsilon_t^2$ 替代。

### 扩展模型

#### EGARCH(p, q) — Nelson (1991)
**指数 GARCH** 用 $\log \sigma_t^2$ 建模，无需非负约束，且能捕捉 **杠杆效应（leverage effect）**：
$$
\log \sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \cdot g(z_{t-i}) + \sum_{j=1}^{p} \beta_j \cdot \log \sigma_{t-j}^2
$$
其中 $g(z_t) = \theta z_t + \gamma \left(|z_t| - \mathbb{E}[|z_t|]\right)$ 是 **非对称冲击函数**：
- $\theta z_t$：符号效应（杠杆效应，$\theta < 0$ 时负面冲击增加波动更多）
- $\gamma(|z_t| - \mathbb{E}[|z_t|])$：幅度效应

#### GJR-GARCH(p, q) — Glosten et al. (1993)
在 GARCH 中显式加入 **非对称项** 来建模杠杆效应：
$$
\sigma_t^2 = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \gamma \cdot I_{t-1} \cdot \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
$$
其中 $I_{t-1} = \mathbb{1}[\varepsilon_{t-1} < 0]$ 为负面冲击指示变量。$\gamma > 0$ 表明负面冲击对波动的影响大于正面冲击。

#### GARCH-M — Engle, Lilien & Robins (1987)
将条件方差 **纳入均值方程**，刻画风险溢价：
$$
r_t = \mu + \lambda \sigma_t^2 + \varepsilon_t, \quad \varepsilon_t \sim (0, \sigma_t^2)
$$
$\lambda > 0$ 表示风险与收益正相关（高风险要求高回报）。

#### t-GARCH — Bollerslev (1987)
用 **标准化 t 分布** 替代正态分布，捕捉金融收益率的 **厚尾特征**：
$$
z_t \sim t_\nu \ \text{(标准化)}, \quad \nu > 2
$$
对数似然需调整 t 分布的密度函数。自由度 $\nu$ 越小尾部越厚，$\nu \to \infty$ 退化为正态。

### 波动率预测（h 步 ahead）
对 GARCH(1,1)，$h$ 步前条件方差预测通过 **递归代入** 得到：
$$
\mathbb{E}[\sigma_{t+h}^2 \mid \mathcal{F}_t] = \alpha_0 + (\alpha_1 + \beta_1) \cdot \mathbb{E}[\sigma_{t+h-1}^2 \mid \mathcal{F}_t]
$$
当 $h \to \infty$，预测收敛到无条件方差 $\bar{\sigma}^2$。

### 新闻影响曲线（News Impact Curve）
固定 $\sigma_t^2$，将 $\sigma_{t+1}^2$ 表示为 $\varepsilon_t$ 的函数：
$$
\sigma_{t+1}^2 = \alpha_0 + \alpha \cdot \varepsilon_t^2 + \beta \sigma_t^2
$$
该曲线直观展示不同符号和大小的冲击如何影响未来波动。GARCH 对称（$\varepsilon_t^2$ 平方化），EGARCH/GJR 非对称。

## 关键假设
- **弱白噪声**：收益率序列 $r_t$ 均值无自相关（或条件均值已正确定义，如 ARMA 滤除）
- **波动率聚集**：残差平方存在序列相关性（条件异方差）
- **条件正态性**：标准化残差 $z_t = \varepsilon_t / \sigma_t$ 服从标准正态或 t 分布（厚尾修正）
- **参数非负**：标准 GARCH 要求 $\alpha_i, \beta_j \geq 0$（EGARCH 免除此约束）
- **协方差平稳**：$\sum \alpha_i + \sum \beta_j < 1$ 确保长期波动收敛
- **杠杆效应**（仅 EGARCH / GJR-GARCH）：负面冲击对波动的增量影响大于正面冲击
- **分布假设**：正态假设下 MLE 即使误设也一致（QMLE），但标准误需稳健修正

## 适用场景
- **金融收益率波动率建模**：股票、汇率、利率等日度/周度收益率的条件方差估计
- **VaR 计算与风险管理**：$\text{VaR}_{t, \alpha} = \mu_t + \sigma_t \cdot q_\alpha$（$q_\alpha$ 为残差分布 $\alpha$ 分位数）
- **期权定价**：波动率是 Black-Scholes 的核心输入，GARCH 提供时变波动率估计
- **波动率预测**：资产波动率的短期预测，用于组合再平衡和风险对冲
- **宏观经济变量**：通胀率、GDP 增长率的波动性建模
- **事件研究**：量化新闻/政策发布对市场波动的影响

### 不适用
- **仅关注条件均值**（波动率不是核心关注点）：此时 ARIMA 或线性回归即可
- **无波动率聚集的序列**：白噪声或常方差序列，GARCH 参数不可识别
- **序列过短（$T < 200$）**：MLE 在小样本下偏差严重，参数估计不稳定
- **高频数据（分钟级/ tick 级）**：市场微观结构噪声使 GARCH 失效，需用 Realized Volatility 或 HAR-RV
- **非平稳波动率**：结构突变或 regime switching 时需用 SWARCH 或 MS-GARCH
- **大量缺失值/不规则时间间隔**：GARCH 对时间间隔等距要求严格

## 实现要点

### 模型定阶
| 阶数 | 典型值 | 选择方法 |
|------|--------|----------|
| $p$（GARCH 阶数） | 1（多数序列够用） | AIC / BIC（网格搜索 $p \in \{1,2\}, q \in \{1,2\}$） |
| $q$（ARCH 阶数） | 1（多数序列够用） | PACF of $\varepsilon_t^2$ 初步诊断 |
| 分布 | 正态 / t / GED | BIC + QQ 图验证尾部拟合 |

- GARCH(1,1) 对绝大多数金融时间序列已足够，高阶情况罕见
- 残差平方的 PACF 在滞后 $q$ 处截断可提示 ARCH 阶数

### 参数约束
- **标准 GARCH**：$\alpha_i, \beta_j \geq 0$ 通过 **Box-Cox 变换** 或 **约束优化** 实现
- **EGARCH**：$\log \sigma_t^2$ 无符号约束，直接无约束优化
- **平稳条件**：$\sum \alpha_i + \sum \beta_j < 1$ 可用非线性约束（如 SLSQP）

### 优化策略
- **方差目标化**：固定 $\alpha_0 = \hat{\sigma}^2(1 - \sum \alpha_i - \sum \beta_j)$，减少一个自由参数
- **优化器**：`L-BFGS-B`（边界约束）、`SLSQP`（等式/不等式约束）、`Nelder-Mead`（无梯度，稳健但慢）
- **初始值**：$\alpha_0$ 用 0.1 倍样本方差，$\alpha_1 \approx 0.1$，$\beta_1 \approx 0.85$
- **数值稳定性**：使用 $\log(\sigma_t^2)$ 而非 $\sigma_t^2$ 更新，避免方差过小

### 分布选择
- **正态 GARCH**：MLE 一致但非有效（QMLE），标准误需 sandwich 估计
- **t-GARCH**：自由度 $\nu$ 通常 $4 \sim 9$，$\nu$ 越小尾部越厚
- **GED-GARCH**：比 t 分布更灵活，形状参数控制尾部厚度

### 诊断检验
- **标准化残差**：$\hat{z}_t = \varepsilon_t / \hat{\sigma}_t$ 应近似 i.i.d.
- **Ljung-Box 检验**：对 $\hat{z}_t$（检验均值方程）和 $\hat{z}_t^2$（检验方差方程）在滞后 $L$ 处做 Q 检验
- **QQ 图**：验证 $\hat{z}_t$ 是否服从假定分布
- **ARCH-LM 检验**：检验残差平方是否仍存在 ARCH 效应（残差应无 ARCH）

### VaR 计算
$$
\text{VaR}_{t, \alpha} = \mu_t + \sigma_t \cdot q_\alpha(z)
$$
- 正态：$q_{0.01} = -2.326$, $q_{0.05} = -1.645$
- t 分布：$q_\alpha = t_{\nu, \alpha} \cdot \sqrt{(\nu-2)/\nu}$（标准化）
- 回测：Kupiec LR 检验（失败率 与 $\alpha$ 的一致性检验）

### 代码

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as t_dist, jarque_bera
from scipy.special import gammaln
import warnings


def _log_likelihood_garch(params, eps, p, q, dist='normal'):
    """GARCH(p,q) 负对数似然"""
    T = len(eps)
    alpha0 = params[0]
    alpha = params[1:q+1]
    beta = params[q+1:q+p+1]
    
    sigma2 = np.full(T, alpha0 / (1 - np.sum(alpha) - np.sum(beta)))
    sigma2[0:max(p, q)] = np.var(eps)  # 初始方差用样本方差
    
    for t in range(max(p, q), T):
        sigma2[t] = alpha0
        for i in range(1, q+1):
            sigma2[t] += alpha[i-1] * eps[t-i]**2
        for j in range(1, p+1):
            sigma2[t] += beta[j-1] * sigma2[t-j]
    
    sigma2 = np.maximum(sigma2, 1e-8)  # 防止数值下溢
    
    if dist == 'normal':
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    elif dist == 't':
        nu = params[-1]
        if nu <= 2:
            return -1e10
        # 标准化 t 分布对数似然
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        ll = np.sum(c - 0.5 * np.log(sigma2) - (nu + 1) / 2 * np.log(1 + eps**2 / (sigma2 * (nu - 2))))
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    
    return -ll if np.isfinite(ll) else 1e10


class GARCH:
    """GARCH(p,q) 波动率模型 — 从头实现
    
    支持正态和学生 t 分布创新，方差目标化加速估计。
    """
    
    def __init__(self, p=1, q=1, dist='normal'):
        if p < 0 or q < 1:
            raise ValueError("GARCH requires p >= 0, q >= 1")
        self.p = p
        self.q = q
        self.dist = dist
        self.params_ = None
        self.sigma2_ = None
        self.eps_ = None
        self._n_params = p + q + 1 + (1 if dist == 't' else 0)
        
    def fit(self, eps, method='L-BFGS-B', disp=False):
        """MLE 估计 GARCH 参数
        
        Parameters
        ----------
        eps : array-like
            均值方程残差序列（需预滤除条件均值）
        method : str
            scipy 优化方法，推荐 'L-BFGS-B' 或 'SLSQP'
        disp : bool
            是否显示优化过程
            
        Returns
        -------
        self : GARCH
        """
        eps = np.asarray(eps, dtype=float)
        self.eps_ = eps
        T = len(eps)
        eps_var = np.var(eps)
        
        # 方差目标化初始值
        n_par = self.q + self.p + 1
        bounds = [(1e-6, None)] * n_par
        
        # 初始值: alpha0 ~ 0.1 * var, alpha ~ 0.1, beta ~ 0.85
        x0 = [eps_var * 0.1] + [0.1] * self.q + [0.85] * self.p
        
        if self.dist == 't':
            x0.append(8.0)
            bounds.append((2.1, None))
        
        # 约束条件: sum(alpha) + sum(beta) < 1
        cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[1:self.q+1]) - np.sum(x[self.q+1:self.q+self.p+1]) - 1e-6})
        
        result = minimize(
            _log_likelihood_garch, x0, args=(eps, self.p, self.q, self.dist),
            method=method, bounds=bounds, constraints=cons,
            options={'disp': disp, 'maxiter': 500}
        )
        
        if not result.success:
            warnings.warn(f"GARCH MLE did not converge: {result.message}")
        
        self.params_ = result.x
        
        # 计算拟合的条件方差
        self.sigma2_ = self._compute_sigma2(eps, self.params_)
        return self
    
    def _compute_sigma2(self, eps, params):
        """给定参数计算条件方差序列"""
        T = len(eps)
        alpha0 = params[0]
        alpha = params[1:self.q+1] if self.q > 0 else []
        beta = params[self.q+1:self.q+self.p+1] if self.p > 0 else []
        
        # 无条件方差初始化
        init_var = np.var(eps)
        sigma2 = np.full(T, init_var)
        
        for t in range(max(self.p, self.q), T):
            s2 = alpha0
            for i in range(1, self.q+1):
                s2 += alpha[i-1] * eps[t-i]**2
            for j in range(1, self.p+1):
                s2 += beta[j-1] * sigma2[t-j]
            sigma2[t] = s2
        
        return np.maximum(sigma2, 1e-8)
    
    def conditional_volatility(self):
        """返回条件波动率序列 sigma_t (in-sample)"""
        if self.sigma2_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return np.sqrt(self.sigma2_)
    
    def residuals(self):
        """返回标准化残差 z_t = eps_t / sigma_t"""
        if self.sigma2_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.eps_ / np.sqrt(self.sigma2_)
    
    def forecast(self, h=10, last_eps=None, last_sigma2=None):
        """向前 h 步波动率预测
        
        Parameters
        ----------
        h : int
            预测步长
        last_eps : array-like, optional
            最近 q 个残差（用于初始化递归）
        last_sigma2 : array-like, optional
            最近 p 个条件方差
            
        Returns
        -------
        forecasts : ndarray, shape (h,)
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        if last_eps is None:
            last_eps = self.eps_[-self.q:] if self.q > 0 else np.array([])
        if last_sigma2 is None:
            last_sigma2 = self.sigma2_[-self.p:] if self.p > 0 else np.array([])
        
        alpha0 = self.params_[0]
        alpha = self.params_[1:self.q+1] if self.q > 0 else []
        beta = self.params_[self.q+1:self.q+self.p+1] if self.p > 0 else []
        
        forecasts = np.zeros(h)
        eps_queue = list(last_eps)
        sigma2_queue = list(last_sigma2)
        
        # 多步预测: E[eps_{t+k}^2 | F_t] = E[sigma_{t+k}^2 | F_t]
        for k in range(h):
            s2 = alpha0
            for i in range(1, self.q+1):
                if len(eps_queue) >= i:
                    s2 += alpha[i-1] * eps_queue[-i]**2
            for j in range(1, self.p+1):
                if len(sigma2_queue) >= j:
                    s2 += beta[j-1] * sigma2_queue[-j]
            
            forecasts[k] = s2
            eps_queue.append(0.0)  # 远期冲击期望为 0
            sigma2_queue.append(s2)
        
        return np.sqrt(forecasts)
    
    def news_impact_curve(self, eps_range=(-0.1, 0.1), n_points=100):
        """生成新闻影响曲线
        
        Parameters
        ----------
        eps_range : tuple
            冲击范围（标准差单位）
        n_points : int
            
        Returns
        -------
        eps_grid : ndarray
        sigma2_grid : ndarray
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        alpha0 = self.params_[0]
        alpha = self.params_[1:self.q+1] if self.q > 0 else []
        beta = self.params_[self.q+1:self.q+self.p+1] if self.p > 0 else []
        
        sigma2_ref = np.mean(self.sigma2_) if self.sigma2_ is not None else 1.0
        
        eps_grid = np.linspace(eps_range[0], eps_range[1], n_points)
        sigma2_grid = np.zeros(n_points)
        
        for i, e in enumerate(eps_grid):
            s2 = alpha0
            s2 += alpha[0] * e**2 if len(alpha) > 0 else 0
            s2 += beta[0] * sigma2_ref if len(beta) > 0 else 0
            sigma2_grid[i] = s2
        
        return eps_grid, sigma2_grid


class EGARCH:
    """EGARCH(1,1) — 指数 GARCH，非对称波动率模型 (Nelson 1991)
    
    在 log 方差空间中建模，无需非负约束，捕捉杠杆效应。
    """
    
    def __init__(self, dist='normal'):
        self.dist = dist
        self.params_ = None
        self.sigma2_ = None
        self.eps_ = None
        
    def _neg_loglik(self, params, eps):
        T = len(eps)
        omega, alpha, theta, gamma, beta = params[0], params[1], params[2], params[3], params[4]
        
        if self.dist == 't':
            nu = params[5]
            if nu <= 2:
                return 1e10
        elif self.dist == 'ged':
            nu = params[5]
            if nu <= 0:
                return 1e10
        
        log_sigma2 = np.zeros(T)
        log_sigma2[0] = np.log(np.var(eps))
        
        # EGARCH 递归
        for t in range(1, T):
            z = eps[t-1] / np.sqrt(np.exp(log_sigma2[t-1]) + 1e-12)
            g_z = theta * z + gamma * (np.abs(z) - np.sqrt(2 / np.pi))
            log_sigma2[t] = omega + alpha * g_z + beta * log_sigma2[t-1]
        
        sigma2 = np.maximum(np.exp(log_sigma2), 1e-12)
        
        if self.dist == 'normal':
            ll = -0.5 * np.sum(np.log(2 * np.pi) + log_sigma2 + eps**2 / sigma2)
        elif self.dist == 't':
            c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            ll = np.sum(c - 0.5 * log_sigma2 - (nu + 1) / 2 * np.log(1 + eps**2 / (sigma2 * (nu - 2))))
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")
        
        return -ll if np.isfinite(ll) else 1e10
    
    def fit(self, eps, method='L-BFGS-B', disp=False):
        eps = np.asarray(eps, dtype=float)
        self.eps_ = eps
        T = len(eps)
        
        # [omega, alpha, theta, gamma, beta, (nu)]
        x0 = [-0.1 * np.log(np.var(eps)), 0.1, 0.0, 0.1, 0.95]
        bounds = [(None, None), (None, None), (None, None), (0, None), (-1, 1)]
        
        if self.dist == 't':
            x0.append(8.0)
            bounds.append((2.1, None))
        
        result = minimize(
            self._neg_loglik, x0, args=(eps,),
            method=method, bounds=bounds,
            options={'disp': disp, 'maxiter': 500}
        )
        self.params_ = result.x
        self.sigma2_ = self._compute_sigma2(eps)
        return self
    
    def _compute_sigma2(self, eps):
        T = len(eps)
        omega, alpha, theta, gamma, beta = (
            self.params_[0], self.params_[1], self.params_[2],
            self.params_[3], self.params_[4]
        )
        log_sigma2 = np.zeros(T)
        log_sigma2[0] = np.log(np.var(eps))
        for t in range(1, T):
            z = eps[t-1] / np.sqrt(np.exp(log_sigma2[t-1]) + 1e-12)
            g_z = theta * z + gamma * (np.abs(z) - np.sqrt(2 / np.pi))
            log_sigma2[t] = omega + alpha * g_z + beta * log_sigma2[t-1]
        return np.maximum(np.exp(log_sigma2), 1e-12)
    
    def conditional_volatility(self):
        if self.sigma2_ is None:
            raise RuntimeError("Model not fitted yet.")
        return np.sqrt(self.sigma2_)
    
    def residuals(self):
        if self.sigma2_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.eps_ / np.sqrt(self.sigma2_)


# =====================
# 诊断工具函数
# =====================

def plot_volatility_clustering(eps, sigma, title="Volatility Clustering"):
    """绘制收益率与条件波动率，直观展示波动聚集"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    axes[0].plot(eps, color='steelblue', linewidth=0.8)
    axes[0].set_ylabel("Returns")
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_title(title)
    
    axes[1].plot(sigma, color='crimson', linewidth=0.8)
    axes[1].set_ylabel("Conditional Volatility")
    axes[1].set_xlabel("Time")
    
    plt.tight_layout()
    return fig


def qq_plot(z, dist='normal', title="QQ Plot"):
    """标准化残差的 QQ 图"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    z_sorted = np.sort(z)
    n = len(z_sorted)
    
    if dist == 'normal':
        theoretical = norm.ppf(np.linspace(0.01, 0.99, n))
    elif dist == 't':
        # 先粗略估计自由度（方法矩估计）
        kurt = np.mean(z**4) / np.mean(z**2)**2
        nu_est = 4 + 6 / (kurt - 3) if kurt > 3.5 else 8
        nu_est = max(4, min(30, nu_est))
        theoretical = t_dist.ppf(np.linspace(0.01, 0.99, n), df=nu_est)
    else:
        theoretical = norm.ppf(np.linspace(0.01, 0.99, n))
    
    ax.scatter(theoretical, z_sorted, s=8, alpha=0.6, color='steelblue')
    ax.plot(theoretical, theoretical, 'r--', linewidth=1)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(title)
    
    # 标注 Jarque-Bera
    jb_stat, jb_p = jarque_bera(z)
    ax.text(0.05, 0.95, f"JB p-value: {jb_p:.4f}", transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


# =====================
# 使用示例
# =====================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # 生成 GARCH(1,1) 数据
    T = 2000
    omega, alpha, beta = 0.05, 0.10, 0.85
    
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * np.random.randn()
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
    
    print(f"True params: omega={omega}, alpha={alpha}, beta={beta}")
    print(f"Unconditional variance: {omega / (1 - alpha - beta):.4f}")
    
    # 拟合 GARCH(1,1)
    model = GARCH(p=1, q=1, dist='normal')
    model.fit(eps, disp=False)
    
    alpha0_est, alpha_est, beta_est = model.params_
    print(f"Estimated:  alpha0={alpha0_est:.4f}, alpha={alpha_est:.4f}, beta={beta_est:.4f}")
    print(f"Persist (alpha+beta): {alpha_est + beta_est:.4f}")
    
    # 条件波动率
    sigma_hat = model.conditional_volatility()
    true_sigma = np.sqrt(sigma2)
    rmse = np.sqrt(np.mean((sigma_hat[100:] - true_sigma[100:])**2))
    print(f"Volatility RMSE (burn-in 100): {rmse:.4f}")
    
    # 向前 20 步预测
    fcst = model.forecast(h=20)
    print(f"First 5 volatility forecasts: {fcst[:5]}")
    
    # 标准化残差
    z = model.residuals()
    print(f"Std residuals: mean={np.mean(z):.4f}, std={np.std(z):.4f}")
    
    # 新闻影响曲线
    eps_grid, sigma2_grid = model.news_impact_curve()
    
    # 可视化
    fig1 = plot_volatility_clustering(eps, sigma_hat, 
                                       "Synthetic GARCH(1,1) — Volatility Clustering")
    plt.show()
    
    fig2 = qq_plot(z, dist='normal', title="QQ Plot — Standardized Residuals")
    plt.show()
    
    # 新闻影响曲线
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(eps_grid, sigma2_grid, 'b-', linewidth=2)
    ax3.set_xlabel(r"$\varepsilon_t$ (Shock)")
    ax3.set_ylabel(r"$\sigma_{t+1}^2$ (Next-period Variance)")
    ax3.set_title("News Impact Curve — GARCH(1,1)")
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    # EGARCH 示例
    print("\n--- EGARCH(1,1) ---")
    egarch = EGARCH(dist='normal')
    egarch.fit(eps, disp=False)
    egarch_sigma = egarch.conditional_volatility()
    egarch_rmse = np.sqrt(np.mean((egarch_sigma[100:] - true_sigma[100:])**2))
    print(f"EGARCH Volatility RMSE: {egarch_rmse:.4f}")
```

### 基于 `arch` 包的生产用法
```python
from arch import arch_model

# 快速拟合 GARCH(1,1) 正态
am = arch_model(eps, vol='Garch', p=1, q=1, dist='normal')
res = am.fit(disp='off')
print(res.summary())

# 条件波动率
sigma_hat = res.conditional_volatility

# 预测
forecasts = res.forecast(horizon=10)
print(forecasts.variance.iloc[-1])

# EGARCH
am_egarch = arch_model(eps, vol='EGARCH', p=1, q=1, dist='t')
res_egarch = am_egarch.fit(disp='off')

# GJR-GARCH
am_gjr = arch_model(eps, vol='GARCH', p=1, o=1, q=1, dist='t')
res_gjr = am_gjr.fit(disp='off')

# VaR 计算（条件正态）
alpha = 0.05
q_norm = norm.ppf(alpha)
var_series = sigma_hat * q_norm
print(f"5% VaR (last 5): {var_series[-5:]}")

# VaR 回测 — Kupiec LR 检验
from arch.utility import covar
failures = (eps[-500:] < var_series[-500:]).mean()
print(f"VaR failure rate: {failures:.4f} (expected {alpha})")
```

## 参考文献
Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of the Royal Statistical Society, Series B*, 48(2), 127–145.

Bollerslev, T. (1987). A Conditionally Heteroskedastic Time Series Model for Speculative Prices and Rates of Return. *The Review of Economics and Statistics*, 69(3), 542–547.

Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987–1007.

Engle, R. F., Lilien, D. M., & Robins, R. P. (1987). Estimating Time Varying Risk Premia in the Term Structure: The ARCH-M Model. *Econometrica*, 55(2), 391–407.

Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779–1801.

Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica*, 59(2), 347–370.
