# VAR/VEC — 向量自回归 / 向量误差修正模型

- **来源**: Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1–48.
- **DOI**: 10.2307/1912017
- **方法类别**: 时间序列 / 多元时间序列 / 计量经济学

## 数学设定

### VAR(p) 模型框架

$$
y_t = c + \Phi_1 y_{t-1} + \Phi_2 y_{t-2} + \cdots + \Phi_p y_{t-p} + \varepsilon_t, \quad t = 1, \dots, T
$$

其中：
- $y_t$ 是 $K \times 1$ 内生变量向量
- $c$ 是 $K \times 1$ 截距向量
- $\Phi_i$ 是 $K \times K$ 系数矩阵（$i = 1,\dots,p$）
- $\varepsilon_t \sim \text{WN}(0, \Sigma)$ 是 $K \times 1$ 白噪声误差向量，$\Sigma$ 为 $K \times K$ 正定协方差矩阵

### 稳定性条件

将 VAR(p) 写为 $Kp$ 维的伴随形式（companion form）：

$$
\xi_t = F \xi_{t-1} + v_t
$$

其中：
$$
\xi_t = \begin{bmatrix} y_t \\ y_{t-1} \\ \vdots \\ y_{t-p+1} \end{bmatrix}_{Kp \times 1}, \quad
F = \begin{bmatrix}
\Phi_1 & \Phi_2 & \cdots & \Phi_{p-1} & \Phi_p \\
I_K & 0 & \cdots & 0 & 0 \\
0 & I_K & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & I_K & 0
\end{bmatrix}_{Kp \times Kp}
$$

**稳定性条件**：伴随矩阵 $F$ 的所有特征值的模长均小于 1，即 $\max_i |\lambda_i(F)| < 1$。此时 VAR(p) 是协方差平稳的，且脉冲响应函数收敛到零。

### OLS 估计（逐方程 OLS）

VAR(p) 的每个方程可独立用 OLS 估计（似不相关回归 SUR，因各方程 RHS 变量相同）：

$$
\hat{\Phi} = (X'X)^{-1} X'Y
$$

其中 $X$ 为 $T^* \times (1 + Kp)$ 设计矩阵（包含截距和 $p$ 阶滞后），$Y$ 为 $T^* \times K$ 因变量矩阵，$T^* = T - p$。

残差协方差矩阵估计：
$$
\hat{\Sigma} = \frac{1}{T^* - Kp - 1} \hat{E}'\hat{E}, \quad \hat{E} = Y - X\hat{\Phi}
$$

### 滞后阶数选择

信息准则（估计 $T^* = T - p$ 个观测值）：

$$
\begin{aligned}
\text{AIC}(p) &= \ln|\hat{\Sigma}| + \frac{2k}{T^*} \\
\text{BIC}(p) &= \ln|\hat{\Sigma}| + \frac{k \ln T^*}{T^*} \\
\text{HQ}(p) &= \ln|\hat{\Sigma}| + \frac{2k \ln(\ln T^*)}{T^*}
\end{aligned}
$$

其中 $k = 1 + Kp$ 为每方程参数个数，$|\hat{\Sigma}|$ 为残差协方差矩阵的行列式。BIC 一致估计真实阶数但更节俭；AIC 渐近效率高但可能过度参数化。通常结合残差自相关检验（如 Portmanteau 检验）综合判断。

### Granger 因果关系检验

检验变量 $y_j$ 是否 Granger 引起 $y_i$：

- $H_0$：在 $y_i$ 方程中，$y_j$ 的所有 $p$ 阶滞后系数联合为零
- 使用 F 统计量（Chow 检验形式）：

$$
F = \frac{(\text{RSS}_r - \text{RSS}_u) / q}{\text{RSS}_u / (T^* - k)} \sim F(q, T^* - k)
$$

其中 $\text{RSS}_u$ 为无约束残差平方和（包含所有变量所有滞后），$\text{RSS}_r$ 为约束残差平方和（剔除 $y_j$ 的 $p$ 阶滞后），$q = p$ 为约束个数，$k = 1 + Kp$ 为无约束模型参数个数。

### 脉冲响应函数 (Impulse Response Function, IRF)

VAR(p) 的向量 MA($\infty$) 表示：

$$
y_t = \mu + \sum_{i=0}^{\infty} \Psi_i \varepsilon_{t-i}, \quad \Psi_0 = I_K
$$

其中 $\Psi_i$ 递推计算：
$$
\Psi_i = \sum_{j=1}^{\min(i, p)} \Phi_j \Psi_{i-j}, \quad i = 1, 2, \dots
$$

**非正交化 IRF**：$\psi_{jk}(h) = \frac{\partial y_{j,t+h}}{\partial \varepsilon_{k,t}} = e_j' \Psi_h e_k$，即 $\Psi_h$ 的第 $(j,k)$ 元素。

**正交化 IRF (OIRF)**：对 $\Sigma$ 进行 Cholesky 分解 $\Sigma = P P'$（$P$ 为下三角矩阵）：

$$
\Theta_h = \Psi_h P, \quad \theta_{jk}(h) = e_j' \Psi_h P e_k
$$

$\theta_{jk}(h)$ 表示对 $y_k$ 施加一个标准差的正交冲击后，$y_j$ 在 $h$ 期后的响应。**变量排序至关重要**：排序靠前的变量当期影响靠后的变量，反之则不能——变量应从最外生到最内生排序。

**累积 IRF**：$\sum_{h=0}^{H} \Psi_h$ 表示冲击的累积效应。

### 预测误差方差分解 (FEVD)

$h$ 步预测误差的正交分解：

$$
y_{t+h} - \hat{y}_{t+h|t} = \sum_{l=0}^{h-1} \Psi_l \varepsilon_{t+h-l} = \sum_{l=0}^{h-1} \Psi_l P v_{t+h-l}
$$

其中 $v_t = P^{-1} \varepsilon_t$，$\text{Cov}(v_t) = I_K$。

变量 $j$ 的 $h$ 步预测误差方差中由变量 $k$ 的冲击解释的比例：

$$
\omega_{jk}(h) = \frac{\sum_{l=0}^{h-1} \theta_{jk}(l)^2}{\sum_{m=1}^{K} \sum_{l=0}^{h-1} \theta_{jm}(l)^2}
$$

$\omega_{jk}(h) \in [0, 1]$，当 $h \to \infty$ 时收敛到某一极限值。

### 协整 (Cointegration)

####  Engle-Granger 两步法（单一协整向量）

适用于 $K=2$ 或单一协整关系：

1. **协整回归**：$y_{1t} = \beta_0 + \beta_1 y_{2t} + u_t$，用 OLS 估计
2. **单位根检验**：对残差 $\hat{u}_t$ 做 ADF 检验（使用 Engle-Granger 临界值）

若 $\hat{u}_t$ 平稳，则 $y_{1t}$ 与 $y_{2t}$ 协整，协整向量为 $(1, -\beta_1)'$。

#### Johansen 迹检验（多协整向量）

考虑 VECM 形式：

$$
\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \varepsilon_t
$$

其中 $\Pi = \alpha \beta'$ 为 $K \times K$ 矩阵：
- $\beta$：$K \times r$ 协整向量矩阵（长期关系）
- $\alpha$：$K \times r$ 调整速度矩阵（向均衡调整）

**Johansen 检验步骤**：

1. 辅助回归获取残差：
   - $R_{0t}$：$\Delta y_t$ 对 $\Delta y_{t-1}, \dots, \Delta y_{t-p+1}$ 回归的残差
   - $R_{1t}$：$y_{t-1}$ 对 $\Delta y_{t-1}, \dots, \Delta y_{t-p+1}$ 回归的残差

2. 计算矩矩阵：
   $$
   S_{ij} = \frac{1}{T} \sum_{t=1}^{T} R_{it} R_{jt}', \quad i, j = 0, 1
   $$

3. 解广义特征值问题：
   $$
   |\lambda S_{11} - S_{10} S_{00}^{-1} S_{01}| = 0
   $$
   得到特征值 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_K$

4. **迹统计量**（检验 $H(r_0): \text{rank}(\Pi) \leq r_0$）：
   $$
   \lambda_{\text{trace}}(r_0) = -T \sum_{i=r_0+1}^{K} \ln(1 - \lambda_i)
   $$

5. **最大特征值统计量**：
   $$
   \lambda_{\max}(r_0) = -T \ln(1 - \lambda_{r_0+1})
   $$

6. 序贯检验：从 $r=0$ 开始，若拒绝则 $r=1$，依此类推，直到不能拒绝

### VECM(p-1) 模型

当变量存在协整关系时，VAR(p) 等价于 VECM(p-1)：

$$
\Delta y_t = \alpha \beta' y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \varepsilon_t
$$

其中：
- $\beta' y_{t-1}$：$r \times 1$ 误差修正项（长期均衡偏差）
- $\alpha$：调整速度矩阵，度量系统向长期均衡的回复速度
- $\Gamma_i = -(\Phi_{i+1} + \cdots + \Phi_p)$：短期动态系数

估计方法：Johansen MLE（简化秩回归），或 Engle-Granger 两步法（先估计 $\beta$，再估计 $\alpha$ 和 $\Gamma_i$）。

## 关键假设

- **平稳性或协整关系**：VAR 要求所有变量平稳；若存在单位根，需确认协整关系后使用 VECM
- **无结构性突变**：模型参数在样本期内稳定（否则需断点检验或时变参数模型）
- **白噪声误差**：$\varepsilon_t$ 无序列自相关、无条件异方差（可用 Portmanteau 检验验证）
- **正确滞后阶数**：滞后阶数设定偏误将导致估计不一致
- **参数稳定性**：样本期内系数不变（Chow 断点检验 / CUSUM 检验验证）
- **正态性**（非必需但影响小样本推断）：误差项联合正态分布使 ML 估计等价于 OLS
- **Cholesky 排序合理性**（用于 IRF/FEVD）：变量排序反映经济理论因果顺序

## 适用场景

- **宏观经济预测**：GDP、通胀、利率、失业率等多变量联动预测
- **政策分析**：货币政策冲击识别（如利率冲击对产出的动态影响）
- **因果关系检验**：变量间的 Granger 因果方向检验
- **传导机制分析**：冲击在经济变量间的传导路径（IRF）
- **金融传染**：跨资产/跨市场的波动率溢出与联动性
- **货币经济学**：货币供给、利率、产出、价格间的动态关系
- **结构分解**：通过 SVAR 识别结构性冲击（需求 vs 供给）

### 不适用

- **单变量时间序列**：使用 ARIMA/ETS 等单变量模型即可
- **无协整的非平稳序列**：先差分至平稳再使用 VAR（差分 VAR，DVAR）
- **变量数多于时间点数**：$K > T^*$ 时参数过度化，需使用正则化方法（如 LASSO-VAR、贝叶斯 VAR）
- **纯预测任务且变量间相关性弱**：单变量模型或独立预测效果更好
- **短期高频数据**（分钟级 tick 数据）：需特殊处理（已实现波动率、UHF 模型）
- **非线性关系**：VAR 假设线性动态，可用 Threshold VAR / Markov Switching VAR

## 实现要点

### 关键诊断与选择

| 项目 | 方法 | 说明 |
|------|------|------|
| 滞后阶数 | AIC / BIC / HQ | BIC 更节俭，AIC 预测更优 |
| 稳定性 | 伴随矩阵特征值 | 所有特征值模长 < 1 |
| 残差自相关 | Portmanteau / LM 检验 | 拒绝则增加滞后阶数 |
| 协整阶数 | Johansen 迹 / 最大特征值 | 比较 MacKinnon-Haug-Michelis 临界值 |
| Granger 因果 | F 检验 / Wald 检验 | 注意滞后阶数敏感性 |
| IRF 识别 | Cholesky / SVAR / 符号约束 | 排序需有经济理论依据 |

### IRF 变量排序原则

Cholesky 分解要求变量从最外生到最内生排序：

1. **最外生**（不受其他变量当期影响）：如政策变量（货币供给、制度指标）
2. **中间**：如名义变量（价格水平）
3. **最内生**（受其他变量当期影响）：如实际变量（产出、就业）

排序敏感性分析：尝试不同排序，检查 IRF 是否稳健。

### SVAR 识别方法

当 Cholesky 递归排序缺乏理论依据时，使用**结构 VAR (SVAR)**：

- **短期约束**（Sims 经典 B 模型）：$\varepsilon_t = B u_t$，需要 $K(K-1)/2$ 个约束
- **长期约束**（Blanchard-Quah）：长期乘子矩阵为下三角
- **符号约束**（Uhlig）：约束 IRF 的符号方向
- **异方差识别**：利用条件异方差实现无需约束的识别

### 引导法 (Bootstrap) IRF 置信区间

为 IRF 构造置信区间（因为 IRF 标准误没有闭合形式）：

1. 估计 VAR(p)，保存系数 $\hat{\Phi}$ 和残差 $\hat{\varepsilon}_t$
2. 对残差进行有放回再抽样，生成 $B$ 个引导样本
3. 对每个引导样本重新估计 VAR 和 IRF
4. 取 percentiles（如 2.5% 和 97.5%）得到 95% 置信区间

### 代码

```python
import numpy as np
from scipy import stats
from itertools import product


# ============================================================
#  Portmanteau 检验 (残差自相关诊断)
# ============================================================

def portmanteau_test(resid, lags=10):
    """
    多元 Portmanteau 检验
    H0: 残差无序列自相关
    
    Parameters
    ----------
    resid : ndarray (T x K)
        模型残差
    lags : int
        检验的滞后阶数
    
    Returns
    -------
    q_stat : float
        Q 统计量
    p_value : float
        p 值
    """
    T, K = resid.shape
    resid = resid - resid.mean(axis=0)
    
    # 自协方差矩阵
    C0 = resid.T @ resid / T  # T x K
    
    q_stat = 0.0
    for h in range(1, lags + 1):
        Ch = resid[h:].T @ resid[:-h] / T  # K x K
        q_stat += T * np.trace(Ch.T @ np.linalg.solve(C0, Ch @ np.linalg.solve(C0, np.eye(K))))
    
    # 自由度: K^2 * (lags - p)
    # 这里不减去 p，作为近似
    df = K**2 * lags
    p_value = 1 - stats.chi2.cdf(q_stat, df)
    
    return q_stat, p_value


# ============================================================
#  VAR 类 — 从零实现
# ============================================================

class VAR:
    """
    向量自回归模型 VAR(p)
    
    使用 OLS 逐方程估计，支持滞后选择、Granger 因果检验、
    脉冲响应 (IRF) 和方差分解 (FEVD)。
    
    Parameters
    ----------
    maxlags : int, optional
        滞后阶数，在 fit() 中可被 ic 覆盖
    """
    
    def __init__(self, maxlags=None):
        self.maxlags = maxlags
        self.coefs_ = []          # list of K×K coefficient matrices, length p
        self.intercept_ = None    # K×1
        self.sigma_u_ = None      # K×K residual covariance
        self.resid_ = None        # T* × K residuals
        self.neqs = None          # K
        self.nobs = None          # T* (effective observations)
        self._y = None            # original data (T×K)
        self._Y = None            # dependent (T*×K)
        self._X = None            # design matrix (T* × (1+Kp))
        self._fitted = None       # in-sample fitted values
    
    # ----------------------------------------------------------
    #  数据准备
    # ----------------------------------------------------------
    
    def _build_regressors(self, y, nlags):
        """
        构造滞后设计矩阵
        
        Returns
        -------
        Y : ndarray (T-p) × K
            因变量
        X : ndarray (T-p) × (1 + K*p)
            设计矩阵: [1, y_{t-1}', y_{t-2}', ..., y_{t-p}']
        """
        T, K = y.shape
        X_rows = []
        for t in range(nlags, T):
            row = [1.0]  # 截距
            for lag in range(1, nlags + 1):
                row.extend(y[t - lag])
            X_rows.append(row)
        X = np.array(X_rows)
        Y = y[nlags:]
        return Y, X
    
    # ----------------------------------------------------------
    #  拟合
    # ----------------------------------------------------------
    
    def fit(self, y, maxlags=None, ic=None):
        """
        用 OLS 估计 VAR(p)
        
        Parameters
        ----------
        y : ndarray (T × K)
            时间序列数据
        maxlags : int, optional
            最大滞后阶数（ic 为 None 时使用此值）
        ic : {'aic', 'bic', 'hq'}, optional
            信息准则自动选阶，会覆盖 maxlags
        
        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        T, K = y.shape
        self.neqs = K
        self._y = y
        
        if ic is not None:
            best_p = self.select_order(y, maxlags or 10, ic)
            self.maxlags = best_p
        else:
            self.maxlags = maxlags
        
        p = self.maxlags
        
        # 构造设计矩阵
        Y, X = self._build_regressors(y, p)
        self.nobs = Y.shape[0]
        self._Y = Y
        self._X = X
        
        # OLS: B = (X'X)^{-1}X'Y
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        # B 的形状: (1+Kp) × K
        self.intercept_ = B[0]  # K×1
        
        self.coefs_ = []
        for i in range(1, p + 1):
            start = 1 + (i - 1) * K
            end = 1 + i * K
            self.coefs_.append(B[start:end].T)  # K×K
        
        # 残差与协方差
        fitted = X @ B
        resid = Y - fitted
        self._fitted = fitted
        self.resid_ = resid
        # 小样本修正
        d = K * p + 1
        self.sigma_u_ = resid.T @ resid / (self.nobs - d)
        
        return self
    
    # ----------------------------------------------------------
    #  滞后阶数选择
    # ----------------------------------------------------------
    
    def select_order(self, y, maxlags, ic='bic'):
        """
        通过信息准则选择滞后阶数
        
        Parameters
        ----------
        y : ndarray (T × K)
        maxlags : int
            最大滞后阶数
        ic : str
            'aic', 'bic', 或 'hq'
        
        Returns
        -------
        best_p : int
        """
        y = np.asarray(y, dtype=float)
        T, K = y.shape
        results = []
        
        for p in range(1, maxlags + 1):
            Y, X = self._build_regressors(y, p)
            nobs = Y.shape[0]
            if nobs <= 0:
                continue
            
            # OLS
            B = np.linalg.lstsq(X, Y, rcond=None)[0]
            resid = Y - X @ B
            sigma = resid.T @ resid / nobs
            
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                continue
            
            k = 1 + K * p  # 每方程参数个数
            
            aic = logdet + 2.0 * k / nobs
            bic = logdet + np.log(nobs) * k / nobs
            hq = logdet + 2.0 * k * np.log(np.log(nobs)) / nobs
            
            results.append((p, aic, bic, hq))
        
        if not results:
            raise ValueError("Cannot select order: insufficient observations.")
        
        ic_map = {'aic': 1, 'bic': 2, 'hq': 3}
        idx = ic_map.get(ic, 2)
        best = min(results, key=lambda x: x[idx])
        return best[0]
    
    # ----------------------------------------------------------
    #  稳定性检验
    # ----------------------------------------------------------
    
    def _companion_matrix(self):
        """构造 Kp × Kp 伴随矩阵"""
        K = self.neqs
        p = self.maxlags
        F = np.zeros((K * p, K * p))
        for i in range(p):
            F[:K, i * K:(i + 1) * K] = self.coefs_[i]
        for i in range(1, p):
            F[i * K:(i + 1) * K, (i - 1) * K:i * K] = np.eye(K)
        return F
    
    def is_stable(self):
        """
        检验 VAR 是否稳定
        
        Returns
        -------
        stable : bool
        eigvals : ndarray
            伴随矩阵特征值
        """
        F = self._companion_matrix()
        eigvals = np.linalg.eigvals(F)
        return bool(np.all(np.abs(eigvals) < 1 - 1e-12)), np.abs(eigvals)
    
    # ----------------------------------------------------------
    #  预测
    # ----------------------------------------------------------
    
    def predict(self):
        """样本内一步预测值"""
        return self._fitted
    
    def forecast(self, y_end, steps):
        """
        动态 h 步预测（递归）
        
        Parameters
        ----------
        y_end : ndarray (p × K) or (T × K)
            最后 p 个观测值（或完整序列，会自动取最后 p 行）
        steps : int
            预测步数
        
        Returns
        -------
        forecasts : ndarray (steps × K)
        """
        y_end = np.asarray(y_end, dtype=float)
        K = self.neqs
        p = self.maxlags
        
        if y_end.ndim == 1:
            y_end = y_end.reshape(1, -1)
        
        history = y_end[-p:].copy()
        forecasts = []
        
        for _ in range(steps):
            # 构建滞后期向量
            x = [1.0]
            for lag in range(1, p + 1):
                x.extend(history[-lag])
            x = np.array(x)
            
            y_pred = self.intercept_.copy()
            for lag in range(p):
                y_pred += self.coefs_[lag] @ history[-(lag + 1)]
            
            forecasts.append(y_pred)
            history = np.vstack([history, y_pred])
        
        return np.array(forecasts)
    
    def forecast_std(self, steps):
        """
        计算预测标准误（h 步预测误差的均方根）
        
        仅考虑来自残差的不确定性，忽略参数估计不确定性。
        
        Parameters
        ----------
        steps : int
        
        Returns
        -------
        se : ndarray (steps × K)
        """
        K = self.neqs
        p = self.maxlags
        P = np.linalg.cholesky(self.sigma_u_)
        
        # 计算 MA 系数
        psi = np.zeros((steps, K, K))
        psi[0] = np.eye(K)
        for h in range(1, steps):
            for j in range(1, min(h, p) + 1):
                psi[h] += self.coefs_[j - 1] @ psi[h - j]
        
        # 预测误差方差 = Σ_{l=0}^{h-1} Ψ_l Σ Ψ_l'
        se = np.zeros((steps, K))
        for h in range(steps):
            var = np.zeros((K, K))
            for l in range(h + 1):
                var += psi[l] @ self.sigma_u_ @ psi[l].T
            se[h] = np.sqrt(np.diag(var))
        
        return se
    
    # ----------------------------------------------------------
    #  Granger 因果检验
    # ----------------------------------------------------------
    
    def granger_causality(self, cause, effect):
        """
        Granger 因果 F 检验
        
        H0: y_cause 不是 y_effect 的 Granger 原因
        （在 effect 方程中，cause 的所有滞后系数联合为零）
        
        Parameters
        ----------
        cause : int
            原因变量的列索引
        effect : int
            结果变量的列索引
        
        Returns
        -------
        f_stat : float
            F 统计量
        p_value : float
            p 值
        df_num : int
            分子自由度
        df_den : int
            分母自由度
        """
        K = self.neqs
        p = self.maxlags
        
        # 无约束模型 RSS (全模型)
        resid_u = self.resid_[:, effect]
        rss_u = np.sum(resid_u ** 2)
        
        # 约束模型：剔除 cause 变量的所有 p 阶滞后
        # 设计矩阵的列结构: [1, y_{t-1}(0..K-1), y_{t-2}(0..K-1), ..., y_{t-p}(0..K-1)]
        cols_keep = [0]  # keep intercept
        for lag in range(p):
            for var in range(K):
                if var != cause:
                    cols_keep.append(1 + lag * K + var)
        
        X_r = self._X[:, cols_keep]
        y_eff = self._Y[:, effect]
        B_r = np.linalg.lstsq(X_r, y_eff, rcond=None)[0]
        resid_r = y_eff - X_r @ B_r
        rss_r = np.sum(resid_r ** 2)
        
        q = p                        # 约束个数
        k_full = 1 + K * p           # 无约束模型参数个数
        T_eff = self.nobs
        df_den = T_eff - k_full
        
        f_stat = ((rss_r - rss_u) / q) / (rss_u / df_den)
        p_value = 1 - stats.f.cdf(f_stat, q, df_den)
        
        return f_stat, p_value, q, df_den
    
    def granger_causality_matrix(self):
        """
        计算所有变量对的 Granger 因果检验
        
        Returns
        -------
        f_mat : ndarray (K × K)
        p_mat : ndarray (K × K)
            f_mat[i, j] 为检验 H0: y_j 非 y_i 的 Granger 原因的 F 值
            p_mat[i, j] 为对应的 p 值
        """
        K = self.neqs
        f_mat = np.zeros((K, K))
        p_mat = np.zeros((K, K))
        for i, j in product(range(K), range(K)):
            if i == j:
                continue
            f_stat, p_val, _, _ = self.granger_causality(j, i)
            f_mat[i, j] = f_stat
            p_mat[i, j] = p_val
        return f_mat, p_mat
    
    # ----------------------------------------------------------
    #  脉冲响应函数 (IRF)
    # ----------------------------------------------------------
    
    def _ma_coefficients(self, steps):
        """
        计算 MA(∞) 表示系数 Ψ_h
        
        Returns
        -------
        psi : ndarray (steps+1) × K × K
            psi[0] = I_K
        """
        K = self.neqs
        p = self.maxlags
        psi = np.zeros((steps + 1, K, K))
        psi[0] = np.eye(K)
        for h in range(1, steps + 1):
            for j in range(1, min(h, p) + 1):
                psi[h] += self.coefs_[j - 1] @ psi[h - j]
        return psi
    
    def irf(self, steps, ordering=None):
        """
        正交化脉冲响应函数 (OIRF)
        
        Parameters
        ----------
        steps : int
            响应期数
        ordering : list of int, optional
            变量排序，默认 [0, 1, ..., K-1]
            最外生变量排最前
        
        Returns
        -------
        oirf : ndarray (steps+1) × K × K
            oirf[h, i, j] = 变量 j 一个标准差正交冲击对变量 i 在 h 期后的响应
        """
        K = self.neqs
        psi = self._ma_coefficients(steps)
        
        # 按 ordering 重排协方差矩阵后进行 Cholesky
        if ordering is None:
            ordering = list(range(K))
        
        sigma_reorder = self.sigma_u_[np.ix_(ordering, ordering)]
        P = np.linalg.cholesky(sigma_reorder)  # 下三角
        
        oirf = np.zeros((steps + 1, K, K))
        for h in range(steps + 1):
            oirf[h][np.ix_(ordering, ordering)] = psi[h][np.ix_(ordering, ordering)] @ P
        
        return oirf
    
    def irf_bootstrap(self, steps, ordering=None, n_boot=500, alpha=0.05):
        """
        用残差引导法 (Residual Bootstrap) 计算 IRF 置信区间
        
        Parameters
        ----------
        steps : int
        ordering : list or None
        n_boot : int
            引导次数
        alpha : float
            显著性水平
        
        Returns
        -------
        oirf : ndarray (steps+1) × K × K
            点估计
        ci_lower : ndarray (steps+1) × K × K
            下界
        ci_upper : ndarray (steps+1) × K × K
            上界
        """
        K = self.neqs
        p = self.maxlags
        T = self._y.shape[0]
        resid = self.resid_
        
        boot_irfs = np.zeros((n_boot, steps + 1, K, K))
        y_fit = self._fitted
        
        for b in range(n_boot):
            # 从残差中有放回抽样
            idx = np.random.randint(0, self.nobs, size=self.nobs)
            boot_resid = resid[idx]
            
            # 生成引导样本
            y_boot = np.zeros((T, K))
            # 初始值来自原始数据的前 p 期
            y_boot[:p] = self._y[:p].copy()
            for t in range(p, T):
                y_boot[t] = self.intercept_.copy()
                for lag in range(p):
                    y_boot[t] += self.coefs_[lag] @ y_boot[t - lag - 1]
                y_boot[t] += boot_resid[t - p]
            
            # 在引导样本上重新估计 VAR
            boot_var = VAR()
            boot_var.fit(y_boot, maxlags=p)
            
            # 检查稳定性
            try:
                stable, _ = boot_var.is_stable()
                if not stable:
                    continue
            except Exception:
                continue
            
            boot_irfs[b] = boot_var.irf(steps, ordering)
        
        # 仅保留成功的引导样本
        valid = ~np.all(np.abs(boot_irfs) < 1e-15, axis=(1, 2, 3))
        boot_irfs = boot_irfs[valid]
        n_valid = boot_irfs.shape[0]
        
        if n_valid < 100:
            raise RuntimeError(f"Only {n_valid} valid bootstrap samples (< 100).")
        
        # 百分位置信区间
        q_low = (alpha / 2) * 100
        q_high = (1 - alpha / 2) * 100
        
        point_est = self.irf(steps, ordering)
        ci_lower = np.percentile(boot_irfs, q_low, axis=0)
        ci_upper = np.percentile(boot_irfs, q_high, axis=0)
        
        return point_est, ci_lower, ci_upper
    
    def plot_irf(self, steps=20, ordering=None, n_boot=0, 
                 titles=None, figsize=(12, 8)):
        """
        绘制脉冲响应函数
        
        Parameters
        ----------
        steps : int
        ordering : list or None
        n_boot : int
            引导次数（0 表示不画置信区间）
        titles : list of str, optional
            变量名称
        figsize : tuple
        """
        import matplotlib.pyplot as plt
        
        K = self.neqs
        oirf = self.irf(steps, ordering)
        
        if n_boot > 0:
            _, ci_low, ci_up = self.irf_bootstrap(steps, ordering, n_boot)
        
        if titles is None:
            titles = [f"Var {i}" for i in range(K)]
        
        fig, axes = plt.subplots(K, K, figsize=figsize)
        for i in range(K):      # 响应变量
            for j in range(K):  # 冲击变量
                ax = axes[i, j]
                x = np.arange(steps + 1)
                ax.plot(x, oirf[:, i, j], 'b-', linewidth=1.5)
                ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                if n_boot > 0:
                    ax.fill_between(x, ci_low[:, i, j], ci_up[:, i, j],
                                    alpha=0.3, color='b')
                if i == K - 1:
                    ax.set_xlabel(f"Shock to {titles[j]}")
                if j == 0:
                    ax.set_ylabel(titles[i])
        
        plt.tight_layout()
        return fig
    
    # ----------------------------------------------------------
    #  方差分解 (FEVD)
    # ----------------------------------------------------------
    
    def fevd(self, steps):
        """
        预测误差方差分解
        
        Parameters
        ----------
        steps : int
            预测步数
        
        Returns
        -------
        fevd_arr : ndarray steps × K × K
            fevd[h, i, j] = 变量 j 冲击解释变量 i 的 h 步预测误差方差的比例
        """
        K = self.neqs
        P = np.linalg.cholesky(self.sigma_u_)
        psi = self._ma_coefficients(steps)
        
        # 正交化 MA 系数 Θ_h = Ψ_h P
        theta = np.zeros_like(psi)
        for h in range(steps + 1):
            theta[h] = psi[h] @ P
        
        # 计算 FEVD
        fevd_arr = np.zeros((steps, K, K))
        for h in range(steps):  # h=0 对应 1 步预测
            for i in range(K):
                # 分子: 各冲击的累积平方贡献
                contrib = np.zeros(K)
                for j in range(K):
                    for l in range(h + 1):
                        contrib[j] += theta[l, i, j] ** 2
                total = contrib.sum()
                if total > 0:
                    fevd_arr[h, i, :] = contrib / total
        
        return fevd_arr


# ============================================================
#  VECM 类 — 协整检验与估计
# ============================================================

class VECM:
    """
    向量误差修正模型 VECM(p-1)
    
    提供 Johansen 迹检验和 VECM 估计。
    
    Parameters
    ----------
    rank : int or None
        协整秩 r（若为 None，由 trace_test 确定）
    """
    
    # Johansen 迹检验临界值（MacKinnon-Haug-Michelis, 1999）
    # 95% 分位数，含截距项的情况
    _CRIT_VAL_95 = {
        1: [3.841, 6.635],        # r=0, r<=1  (K=1, trace vs max-eigen)
        2: [15.495, 19.937],       # r=0, r<=1  when K=2 (trace)
        3: [29.797, 35.175],       # K=3 trace
        4: [47.856, 53.988],       # K=4 trace
        5: [69.819, 77.419],       # K=5 trace
    }
    
    def __init__(self, rank=None):
        self.rank = rank
        self.alpha_ = None     # K × r adjustment speeds
        self.beta_ = None      # K × r cointegrating vectors
        self.gamma_ = None     # list of K×K short-run coefficient matrices
        self.intercept_ = None
        self.resid_ = None
        self.sigma_u_ = None
        self.neqs = None
        self.nobs = None
        self.p = None          # VAR order (VECM uses p-1 lags of differences)
        self.eigvals_ = None
        self.trace_stats_ = None
    
    # ----------------------------------------------------------
    #  Johansen 迹检验
    # ----------------------------------------------------------
    
    def trace_test(self, y, maxlags=5, significance=0.05):
        """
        Johansen 迹检验确定协整秩
        
        Parameters
        ----------
        y : ndarray (T × K)
            时间序列数据
        maxlags : int
            VAR 最大滞后阶数（VECM 使用 maxlags-1 阶差分滞后）
        significance : float
            显著性水平（目前支持 0.05）
        
        Returns
        -------
        r : int
            选择的协整秩
        trace_stats : list of (r, stat, crit_val, reject)
        """
        y = np.asarray(y, dtype=float)
        T, K = y.shape
        self.neqs = K
        
        # 选择滞后阶数（用 VAR 的 BIC）
        var_temp = VAR()
        p = var_temp.select_order(y, maxlags, ic='bic')
        self.p = p
        lag_diff = p - 1  # VECM 差分滞后阶数
        
        # 构造辅助回归的残差
        dy = np.diff(y, axis=0)
        T_eff = dy.shape[0]
        
        # R_0t: Δy_t 对 Δy_{t-1}, ..., Δy_{t-lag_diff} 回归的残差
        # R_1t: y_{t-1} 对 Δy_{t-1}, ..., Δy_{t-lag_diff} 回归的残差
        
        if lag_diff > 0:
            Z = np.column_stack([
                np.vstack([np.zeros((i, K)), dy[:T_eff - i]]) 
                for i in range(1, lag_diff + 1)
            ])
            # 去除前 lag_diff 个观测
            Z_use = Z[lag_diff:]
            dy_use = dy[lag_diff:]
            y_lag1 = y[lag_diff:-1]
        else:
            Z_use = np.ones((T_eff, 1))  # 仅截距
            dy_use = dy
            y_lag1 = y[:-1]
        
        n_eff = dy_use.shape[0]
        
        # 辅助回归
        # R0: Δy 对 Z
        Z_aug = np.column_stack([np.ones(n_eff), Z_use])
        B0 = np.linalg.lstsq(Z_aug, dy_use, rcond=None)[0]
        R0 = dy_use - Z_aug @ B0
        
        # R1: y_{t-1} 对 Z
        B1 = np.linalg.lstsq(Z_aug, y_lag1, rcond=None)[0]
        R1 = y_lag1 - Z_aug @ B1
        
        # 矩矩阵
        S00 = R0.T @ R0 / n_eff
        S01 = R0.T @ R1 / n_eff
        S10 = R1.T @ R0 / n_eff
        S11 = R1.T @ R1 / n_eff
        
        # 广义特征值问题
        # |λ S11 - S10 S00^{-1} S01| = 0
        S00_inv = np.linalg.inv(S00)
        M = S10 @ S00_inv @ S01
        # 用 S11^{-1} M 的特征值
        try:
            S11_inv = np.linalg.inv(S11)
            eigvals = np.linalg.eigvals(S11_inv @ M)
        except np.linalg.LinAlgError:
            # 若 S11 奇异，使用广义特征值
            eigvals = scipy_stats.multi_norm(0).rvs(1000)  # fallback
            raise np.linalg.LinAlgError("S11 is singular.")
        
        # 排序特征值
        eigvals = np.sort(np.real(eigvals))[::-1]
        eigvals = np.clip(eigvals, 0, 1 - 1e-15)
        self.eigvals_ = eigvals
        
        # 迹统计量
        trace_stats = []
        for r in range(K):
            trace_stat = -n_eff * np.sum(np.log(1 - eigvals[r:]))
            # 临界值
            crit_val = self._get_crit_val(K, r, significance)
            reject = trace_stat > crit_val
            trace_stats.append({
                'r_null': r,
                'trace_stat': trace_stat,
                'crit_val': crit_val,
                'reject': reject
            })
        
        self.trace_stats_ = trace_stats
        
        # 选择秩：第一个不能拒绝的 H0
        r_sel = 0
        for entry in trace_stats:
            if not entry['reject']:
                r_sel = entry['r_null']
                break
        else:
            r_sel = K - 1
        
        self.rank = r_sel
        return r_sel, trace_stats
    
    @staticmethod
    def _get_crit_val(K, r, significance=0.05):
        """
        获取 Johansen 迹检验临界值（近似宏村-Haug-Michelis 表）
        
        仅提供标准 case（截距项）
        """
        # 参照 MacKinnon-Haug-Michelis (1999), Table 1 (case 2, intercept in CE)
        trace_crit_95 = {
            1: {0: 3.841, 1: 6.635},
            2: {0: 15.495, 1: 19.937, 2: 24.051},
            3: {0: 29.797, 1: 35.175, 2: 40.886, 3: 46.229},
            4: {0: 47.856, 1: 53.988, 2: 60.362, 3: 66.821, 4: 73.369},
            5: {0: 69.819, 1: 77.419, 2: 84.924, 3: 92.493, 4: 100.145, 5: 107.855},
        }
        
        if K in trace_crit_95 and r in trace_crit_95[K]:
            return trace_crit_95[K][r]
        # 近似外推
        return 20 * K + 5 * r
    
    # ----------------------------------------------------------
    #  VECM 估计
    # ----------------------------------------------------------
    
    def fit(self, y, rank=None):
        """
        估计 VECM(p-1): Δy_t = αβ'y_{t-1} + ΣΓ_iΔy_{t-i} + ε_t
        
        Parameters
        ----------
        y : ndarray (T × K)
        rank : int, optional
            协整秩，若不提供则先执行迹检验
        """
        y = np.asarray(y, dtype=float)
        T, K = y.shape
        self.neqs = K
        
        if rank is not None:
            self.rank = rank
        elif self.rank is None:
            _, _ = self.trace_test(y)
        
        if self.p is None:
            var_temp = VAR()
            self.p = var_temp.select_order(y, 10, ic='bic')
        
        r = self.rank
        p = self.p
        lag_diff = p - 1
        dy = np.diff(y, axis=0)
        n_eff = dy.shape[0]
        
        # 构造差分滞后矩阵
        if lag_diff > 0:
            Z = np.column_stack([
                dy[lag_diff - 1 - i: n_eff - 1 - i]
                for i in range(lag_diff)
            ])
            dy_use = dy[lag_diff:]
            y_lag1 = y[lag_diff:-1]
        else:
            Z = np.ones((n_eff, 1))
            dy_use = dy
            y_lag1 = y[:-1]
        
        n_eff_use = dy_use.shape[0]
        
        # 辅助回归（同 trace_test）
        Z_aug = np.column_stack([np.ones(n_eff_use), Z])
        B0 = np.linalg.lstsq(Z_aug, dy_use, rcond=None)[0]
        R0 = dy_use - Z_aug @ B0
        
        B1 = np.linalg.lstsq(Z_aug, y_lag1, rcond=None)[0]
        R1 = y_lag1 - Z_aug @ B1
        
        # 矩矩阵
        S00 = R0.T @ R0 / n_eff_use
        S01 = R0.T @ R1 / n_eff_use
        S10 = R1.T @ R0 / n_eff_use
        S11 = R1.T @ R1 / n_eff_use
        
        # 求解特征向量（协整向量）
        S00_inv = np.linalg.inv(S00)
        M = S10 @ S00_inv @ S01
        S11_inv = np.linalg.inv(S11)
        
        eigvals, eigvecs = np.linalg.eig(S11_inv @ M)
        
        # 排序
        idx = np.argsort(np.real(eigvals))[::-1]
        eigvecs = eigvecs[:, idx]
        
        # 标准化协整向量: β'S11β = I_r
        beta = np.zeros((K, r))
        for i in range(r):
            b = np.real(eigvecs[:, i])
            # 标准化
            norm = np.sqrt(b @ S11 @ b)
            beta[:, i] = b / norm
        
        self.beta_ = beta
        
        # 估计 EC term
        ec = y_lag1 @ beta  # n_eff_use × r
        
        # 估计 α 和 Γ
        X_vars = [ec]
        if lag_diff > 0:
            for i in range(lag_diff):
                X_vars.append(dy[lag_diff - 1 - i: n_eff - 1 - i])
        X_vars.append(np.ones((n_eff_use, 1)))
        
        X_ecm = np.column_stack(X_vars)
        B_ecm = np.linalg.lstsq(X_ecm, dy_use, rcond=None)[0]
        
        self.alpha_ = B_ecm[:r].T  # K × r
        
        self.gamma_ = []
        for i in range(lag_diff):
            start = r + i * K
            end = r + (i + 1) * K
            self.gamma_.append(B_ecm[start:end].T)
        
        self.intercept_ = B_ecm[-1]
        
        # 残差
        resid = dy_use - X_ecm @ B_ecm
        self.resid_ = resid
        self.sigma_u_ = resid.T @ resid / (n_eff_use - X_ecm.shape[1])
        self.nobs = n_eff_use
        
        return self
    
    def forecast(self, y, steps):
        """
        从 VECM 做动态预测
        内部转换为 VAR(p) 表示后进行预测
        """
        K = self.neqs
        r = self.rank
        p = self.p
        
        # 从 VECM 系数重建 VAR 系数
        # y_t = y_{t-1} + αβ'y_{t-1} + Σ Γ_i (y_{t-i} - y_{t-i-1})
        # 展开得到 VAR 形式
        Phi = []
        # Φ_1 = I_K + αβ' + Γ_1
        Phi1 = np.eye(K) + self.alpha_ @ self.beta_.T
        if len(self.gamma_) > 0:
            Phi1 += self.gamma_[0]
        Phi.append(Phi1)
        
        for i in range(1, p):
            if i - 1 < len(self.gamma_):
                gamma_i = self.gamma_[i - 1]
            else:
                gamma_i = np.zeros((K, K))
            
            if i < len(self.gamma_):
                gamma_ip1 = self.gamma_[i]
            else:
                gamma_ip1 = np.zeros((K, K))
            
            Phi_i = gamma_i - gamma_ip1
            Phi.append(Phi_i)
        
        # 用 VAR 格式预测
        y_end = y[-p:]
        forecasts = []
        history = y_end.copy()
        
        for _ in range(steps):
            y_pred = self.intercept_.copy()
            for lag in range(p):
                y_pred += Phi[lag] @ history[-(lag + 1)]
            forecasts.append(y_pred)
            history = np.vstack([history, y_pred])
        
        return np.array(forecasts)


# ============================================================
#  使用示例 — 三维宏观经济系统
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # --------------------------------------------------------
    # 示例 1: 模拟 VAR(2) 数据并估计
    # --------------------------------------------------------
    print("=" * 60)
    print("示例 1: 模拟 VAR(2) 估计")
    print("=" * 60)
    
    T = 200
    K = 3
    
    # 真实系数
    Phi1_true = np.array([[0.5, 0.1, 0.0],
                          [0.2, 0.6, 0.1],
                          [0.0, 0.3, 0.7]])
    Phi2_true = np.array([[-0.1, 0.0, 0.0],
                          [0.0, -0.1, 0.0],
                          [0.0, 0.0, -0.1]])
    c_true = np.array([0.5, 0.3, 0.2])
    Sigma_true = np.array([[1.0, 0.2, 0.1],
                           [0.2, 1.0, 0.3],
                           [0.1, 0.3, 1.0]])
    
    # 生成数据
    y = np.zeros((T, K))
    for t in range(2, T):
        y[t] = c_true + Phi1_true @ y[t-1] + Phi2_true @ y[t-2] \
               + np.random.multivariate_normal(np.zeros(K), Sigma_true)
    
    # 估计
    var = VAR()
    var.fit(y, maxlags=6, ic='bic')
    
    print(f"选择滞后阶数: p = {var.maxlags}")
    stable, eigvals = var.is_stable()
    print(f"稳定性: {stable} (max |λ| = {eigvals.max():.4f})")
    
    # 样本内拟合优度 (R²)
    resid = var.resid_
    y_center = var._Y - var._Y.mean(axis=0)
    ss_res = np.sum(resid ** 2, axis=0)
    ss_tot = np.sum(y_center ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot
    for i in range(K):
        print(f"  方程 {i} R² = {r2[i]:.4f}")
    
    # 预测
    fc = var.forecast(y, steps=10)
    print(f"\n10 步预测 (前三期):\n{fc[:3]}")
    
    # Granger 因果检验
    print("\nGranger 因果检验 (p 值矩阵):")
    _, p_mat = var.granger_causality_matrix()
    print("  行 i = 响应变量, 列 j = 原因变量")
    print(np.round(p_mat, 4))
    
    # 残差诊断
    q_stat, pv = portmanteau_test(var.resid_, lags=12)
    print(f"\nPortmanteau 检验: Q = {q_stat:.2f}, p = {pv:.4f}")
    
    # IRF
    print("\n计算 IRF (20 期, 变量排序 [0,1,2])...")
    oirf = var.irf(steps=20)
    print(f"  y₀ 对自身冲击的 1 期响应: {oirf[1, 0, 0]:.4f}")
    print(f"  y₀ 对自身冲击的 5 期响应: {oirf[5, 0, 0]:.4f}")
    
    # FEVD
    fevd_arr = var.fevd(steps=20)
    print(f"\nFEVD (20 步预测):")
    for i in range(K):
        print(f"  Var {i}: shock shares = {np.round(fevd_arr[-1, i] * 100, 1)}")
    
    # 绘制 IRF
    fig = var.plot_irf(steps=20, titles=['GDP', 'Inflation', 'Rate'])
    plt.savefig(r"E:\wuyi\数学建模半自动\research-assistant\outputs\var_irf_example.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nIRF 图已保存")
    
    # --------------------------------------------------------
    # 示例 2: 协整系统与 VECM
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("示例 2: 协整系统与 VECM")
    print("=" * 60)
    
    # 生成协整数据: y₁与y₂协整, y₃是随机游走
    # y₁ = β y₂ + u₁, y₂ = y₂_{t-1} + u₂, u₁平稳
    T2 = 300
    y2 = np.cumsum(np.random.randn(T2) * 0.5)
    u1 = np.random.randn(T2) * 0.3  # 平稳
    y1 = 2.0 * y2 + u1
    y3 = np.cumsum(np.random.randn(T2) * 0.5)
    
    y_coint = np.column_stack([y1, y2, y3])
    
    # Johansen 迹检验
    vecm = VECM()
    r, trace_stats = vecm.trace_test(y_coint, maxlags=4)
    print(f"Johansen 迹检验: 选择 r = {r}")
    for entry in trace_stats:
        print(f"  H0: r ≤ {entry['r_null']}, "
              f"Trace = {entry['trace_stat']:.2f}, "
              f"95% CV = {entry['crit_val']:.2f}, "
              f"Reject = {entry['reject']}")
    
    # VECM 估计
    vecm.fit(y_coint, rank=r)
    print(f"\nVECM 估计完成 (p = {vecm.p})")
    print(f"调整速度矩阵 α:\n{np.round(vecm.alpha_, 4)}")
    print(f"协整向量 β:\n{np.round(vecm.beta_, 4)}")
    
    # VECM 预测
    vecm_fc = vecm.forecast(y_coint, steps=10)
    print(f"\nVECM 10 步预测 (前三期):\n{np.round(vecm_fc[:3], 4)}")


# ============================================================
#  基于 statsmodels 的生产用法
# ============================================================

"""
# 安装: pip install statsmodels

# ---- VAR ----
from statsmodels.tsa.api import VAR as SMVAR

model = SMVAR(data)
results = model.fit(maxlags=4, ic='bic')
print(results.summary())

# Granger 因果
# 检验变量 j 是否引起变量 i
test_result = results.test_causality(caused=i, causing=j)
print(test_result)

# IRF
irf = results.irf(periods=20)
irf.plot()

# FEVD
fevd = results.fevd(periods=20)
fevd.plot()

# 预测
forecast = results.forecast(data.values[-results.k_ar:], steps=10)


# ---- VECM ----
from statsmodels.tsa.vector_ar.vecm import VECM as SMVECM, select_order

# 确定滞后阶数
sel = select_order(data, maxlags=8)
print(sel.summary())

# 估计 (假设已知协整秩 r=1)
vecm_results = SMVECM(data, k_ar_diff=sel.aic, coint_rank=1)
vecm_results.fit()
print(vecm_results.summary())

# Johansen 检验
from statsmodels.tsa.vector_ar.vecm import coint_johansen
joh = coint_johansen(data, det_order=0, k_ar_diff=sel.aic)
print(f"Trace stats: {joh.lr1}")
print(f"Max-eigen stats: {joh.lr2}")


# ---- SVAR ----
from statsmodels.tsa.vector_ar.svar_model import SVAR

# 定义短期约束矩阵 A (K×K)
# A ε_t = B u_t, 约束 A 的某些元素为 0
A_restrict = np.array([['E', 'E', 0],
                        ['E', 'E', 'E'],
                        [0, 'E', 'E']], dtype=object)

model_svar = SVAR(data, svar_type='A', A_restrict=A_restrict)
results_svar = model_svar.fit(maxlags=4)
print(results_svar.summary())
"""

## 参考文献

- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1–48.
- Johansen, S. (1988). Statistical Analysis of Cointegration Vectors. *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
- Engle, R. F. & Granger, C. W. J. (1987). Co-Integration and Error Correction: Representation, Estimation, and Testing. *Econometrica*, 55(2), 251–276.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford University Press.
- MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical Distribution Functions of Likelihood Ratio Tests for Cointegration. *Journal of Applied Econometrics*, 14(5), 563–577.
- Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-Spectral Methods. *Econometrica*, 37(3), 424–438.
- Blanchard, O. J. & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances. *American Economic Review*, 79(4), 655–673.
- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure. *Journal of Monetary Economics*, 52(2), 381–419.
