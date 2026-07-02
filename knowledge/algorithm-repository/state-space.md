# State Space Model / Kalman Filter — 状态空间模型与卡尔曼滤波

- **来源**: Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45. / Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- **DOI**: 10.1115/1.3662552
- **方法类别**: 时间序列 / 状态估计 / 贝叶斯滤波

## 数学设定

### 线性高斯状态空间模型 (Linear Gaussian State Space Model)

状态空间模型用一个**潜状态向量** $\mathbf{x}_t$ 描述动态系统的演化，观测变量 $\mathbf{y}_t$ 通过线性变换从潜状态生成：

**状态方程（转移方程）**：
$$
\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_{t-1} + \mathbf{B}_t \mathbf{u}_t + \mathbf{w}_t, \quad
\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)
$$

**观测方程（测量方程）**：
$$
\mathbf{y}_t = \mathbf{H}_t \mathbf{x}_t + \mathbf{v}_t, \quad
\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)
$$

其中 $\mathbf{x}_t \in \mathbb{R}^d$ 为潜状态，$\mathbf{y}_t \in \mathbb{R}^p$ 为观测，$\mathbf{u}_t \in \mathbb{R}^r$ 为可选的控制输入。$\mathbf{F}_t$ 为 $d \times d$ 状态转移矩阵，$\mathbf{H}_t$ 为 $p \times d$ 观测矩阵，$\mathbf{Q}_t$ 和 $\mathbf{R}_t$ 分别为过程噪声和观测噪声协方差矩阵。

### 卡尔曼滤波 (Kalman Filter) — 前向递推

卡尔曼滤波是状态空间模型的核心推断算法，以**预测—更新**（Predict-Update）的递推形式在线估计潜状态。

设 $\hat{\mathbf{x}}_{t|s} = \mathbb{E}[\mathbf{x}_t \mid \mathbf{y}_{1:s}]$，$\mathbf{P}_{t|s} = \mathrm{Var}[\mathbf{x}_t \mid \mathbf{y}_{1:s}]$。

**初始化**：给定初始状态均值和协方差 $\hat{\mathbf{x}}_{0|0}$、$\mathbf{P}_{0|0}$（通常使用扩散先验 $\mathbf{P}_{0|0} = \kappa \mathbf{I}, \kappa \to \infty$）。

**1. 预测步 (Predict)**：
$$
\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}_t \hat{\mathbf{x}}_{t-1|t-1} + \mathbf{B}_t \mathbf{u}_t
$$
$$
\mathbf{P}_{t|t-1} = \mathbf{F}_t \mathbf{P}_{t-1|t-1} \mathbf{F}_t^\top + \mathbf{Q}_t
$$

**2. 更新步 (Update)**：
$$
\boldsymbol{\nu}_t = \mathbf{y}_t - \mathbf{H}_t \hat{\mathbf{x}}_{t|t-1}
\quad \text{(新息 / Innovation)}
$$
$$
\mathbf{S}_t = \mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^\top + \mathbf{R}_t
\quad \text{(新息协方差)}
$$
$$
\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^\top \mathbf{S}_t^{-1}
\quad \text{(卡尔曼增益)}
$$
$$
\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \boldsymbol{\nu}_t
\quad \text{(状态更新)}
$$
$$
\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1}
\quad \text{(协方差更新)}
$$

也可使用 Joseph 形式保证数值稳定性：
$$
\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1} (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)^\top + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^\top
$$

### 卡尔曼平滑 (Kalman Smoother) — Rauch-Tung-Striebel 后向平滑

平滑利用**全部**观测数据 $\mathbf{y}_{1:T}$ 修正每个时间点的状态估计，得到 $\hat{\mathbf{x}}_{t|T}$（比滤波估计更精确）。从 $t = T-1$ 到 $t = 0$ 后向递推：

$$
\mathbf{G}_t = \mathbf{P}_{t|t} \mathbf{F}_{t+1}^\top \mathbf{P}_{t+1|t}^{-1}
\quad \text{(平滑增益)}
$$
$$
\hat{\mathbf{x}}_{t|T} = \hat{\mathbf{x}}_{t|t} + \mathbf{G}_t(\hat{\mathbf{x}}_{t+1|T} - \hat{\mathbf{x}}_{t+1|t})
$$
$$
\mathbf{P}_{t|T} = \mathbf{P}_{t|t} + \mathbf{G}_t(\mathbf{P}_{t+1|T} - \mathbf{P}_{t+1|t})\mathbf{G}_t^\top
$$

### 预测误差分解对数似然 (Prediction Error Decomposition)

状态空间模型的参数 $\boldsymbol{\theta} = \{\mathbf{F}, \mathbf{H}, \mathbf{Q}, \mathbf{R}\}$ 可通过极大似然估计。卡尔曼滤波天然提供了似然的递推分解：

$$
\log L(\boldsymbol{\theta}; \mathbf{y}_{1:T}) = -\frac{1}{2} \sum_{t=1}^{T} \Big[ p \log(2\pi) + \log|\mathbf{S}_t| + \boldsymbol{\nu}_t^\top \mathbf{S}_t^{-1} \boldsymbol{\nu}_t \Big]
$$

其中 $\boldsymbol{\nu}_t$ 和 $\mathbf{S}_t$ 来自前向滤波的更新步。这一分解既是递推似然的计算方式，也为模型诊断提供了依时间的新息分析。

### 参数估计

- **数值 MLE**：对 $\log$ 方差参数化（如 $\log \sigma^2$）使用 L-BFGS-B / Nelder-Mead 优化
- **EM 算法**：利用平滑后的状态估计，在 E 步计算充分统计量，M 步闭式更新 Q、R
- **贝叶斯方法**：对参数设定先验，通过 MCMC / 变分推断联合估计状态和参数

### 扩展方法

**当线性/高斯假设不成立时**，有以下扩展：

| 方法 | 核心思路 | 适用场景 |
|------|---------|---------|
| **EKF** (Extended KF) | 非线性函数在估计点做一阶泰勒展开（Jacobian 线性化） | 弱非线性系统 |
| **UKF** (Unscented KF) | 用确定性 Sigma 点传播非线性，无需 Jacobian | 中等非线性，导数难求 |
| **Particle Filter** (SMC) | 用加权粒子集合近似后验分布，无分布假设 | 强非线性 / 非高斯 / 多模态 |
| **Ensemble KF** (EnKF) | 用蒙特卡洛集合近似协方差 | 高维（如气象数据同化） |
| **Iterated KF** | 在更新步多次线性化 | 强非线性场景 |

## 关键假设

- **线性动力学**（标准 KF）：状态转移和观测映射为线性（EKF/UKF 可放宽）
- **高斯噪声**：过程噪声和观测噪声均为高斯分布（粒子滤波可放宽）
- **模型参数正确设定**：$\mathbf{F}, \mathbf{H}, \mathbf{Q}, \mathbf{R}$ 已知或已被正确估计
- **马尔可夫性**：$\mathbf{x}_t$ 仅依赖于 $\mathbf{x}_{t-1}$（给定 $\mathbf{x}_{t-1}$ 与过去独立）
- **条件独立性**：给定 $\mathbf{x}_t$，$\mathbf{y}_t$ 与过去观测独立
- **噪声互不相关**：$\mathbf{w}_t$ 和 $\mathbf{v}_t$ 在所有时点互不相关

## 适用场景

- **含潜状态的时序数据**：经济状态、心理特质、信号源等不可直接观测变量
- **实时滤波 / 在线估计**：导航定位（GPS/IMU融合）、目标追踪、机器人定位
- **金融计量**：随机波动率（SV）模型、时变因子模型
- **宏观经济即时预测（Nowcasting）**：用混频数据实时估计 GDP 等
- **信号处理**：降噪、信道均衡、语音增强
- **缺失值插补**：状态空间模型天然处理缺失值（跳过更新步即可）
- **结构时间序列分解**：趋势、季节、周期分量的自动分解
- **控制工程**：最优控制中的状态估计（LQG 控制的组成部分）

### 不适用

- **非递推的批量数据**（直接使用标准回归模型即可）
- **完全确定的系统**（无随机噪声，则退化为确定性状态观测器）
- **严重的模型误设**（参数与真实动力学严重不符时，粒子滤波也难以补救，需先修正模型设定）
- **简单的预测任务且无潜变量**（直接使用 ARIMA / Prophet 效率更高）
- **超高维状态**（$d$ 达百万级时需用 EnKF 等降维近似方法）

## 实现要点

### 关键超参数与设计选择

| 参数 / 组件 | 说明 | 推荐初值 / 默认 |
|-------------|------|----------------|
| $\mathbf{F}$（状态转移矩阵） | 状态演化动力学 | 依问题设定（如局部水平模型为 $\mathbf{I}$） |
| $\mathbf{H}$（观测矩阵） | 状态 $\to$ 观测的线性映射 | 依问题设定 |
| $\mathbf{Q}$（过程噪声协方差） | 状态演化的随机波动幅度 | MLE / EM 估计 |
| $\mathbf{R}$（观测噪声协方差） | 测量误差 | MLE / EM 估计 |
| 初始状态均值 $\hat{\mathbf{x}}_{0|-1}$ | 观测前的先验均值 | $\mathbf{0}$（扩散先验） |
| 初始状态协方差 $\mathbf{P}_{0|-1}$ | 观测前的先验不确定性 | $\kappa \mathbf{I}, \kappa = 10^6$ |

### 数值稳定性

1. **使用 Joseph 形式更新协方差**：保证 $\mathbf{P}_{t|t}$ 的对称性和正定性，避免因浮点误差导致的发散
2. **避免矩阵求逆**：计算卡尔曼增益时使用 `np.linalg.solve` 而非 `np.linalg.inv`，即求解 $\mathbf{K}_t \mathbf{S}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^\top$
3. **Square-Root KF**：对 $\mathbf{P}_{t|t-1}$ 做 Cholesky 分解，在平方根域递推，显著提升数值稳健性
4. **slogdet 计算对数似然**：使用 `np.linalg.slogdet` 而非 `np.log(np.linalg.det)` 避免下溢

### 模型诊断

- **新息白噪声检验**：标准化的新息 $\mathbf{S}_t^{-1/2} \boldsymbol{\nu}_t$ 应为 i.i.d. 标准正态。使用 Ljung-Box Q 检验判断残差自相关
- **QQ 图 / 正态性检验**：验证新息的高斯性假设
- **信号噪声比**：$q = \sigma_\eta^2 / \sigma_\varepsilon^2$ 控制滤波的平滑程度（$q \to 0$ 退化为常数，$q \to \infty$ 完全跟随观测）
- **参数可识别性**：部分参数组合可能产生相同似然值，需结合实际意义约束

### 调优经验

1. **从简单模型开始**：先拟合局部水平模型（仅一个潜状态），诊断残差后逐步添加斜率、季节、回归量
2. **扩散先验**：初始协方差设置为大对角矩阵（$\kappa \approx 10^6$），衰减初始状态不确定性的影响
3. **参数变换**：在 MLE 中优化 $\log(\sigma^2)$ 而非 $\sigma^2$，保证正值约束且梯度更平坦
4. **多起始点**：似然面可能多峰，用多个随机起始点运行优化
5. **新息诊断**：最佳模型应有不相关、零均值、常数方差的新息

### 典型结构时间序列组件

| 组件 | 状态方程 | 说明 |
|------|---------|------|
| 局部水平 | $\mu_t = \mu_{t-1} + \eta_t$ | I(1) 趋势，RW 漂移 |
| 局部线性趋势 | $\mu_t = \mu_{t-1} + \nu_{t-1} + \eta_t,\; \nu_t = \nu_{t-1} + \zeta_t$ | 带随机斜率 |
| 季节性（哑元形式） | $\gamma_t = -\sum_{j=1}^{s-1} \gamma_{t-j} + \omega_t$ | 周期 $s$，和为 0 |
| 季节虚拟 | 结合 $s-1$ 个虚拟变量 | 确定性季节 |
| 回归量 | $y_t = \mathbf{z}_t^\top \boldsymbol{\beta} + \varepsilon_t$ | 外生变量 |
| AR($p$) 分量 | $\psi_t = \phi_1 \psi_{t-1} + \cdots + \phi_p \psi_{t-p} + \xi_t$ | 捕捉周期自回归 |

### 代码

```python
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
import matplotlib.pyplot as plt


# ======================================================================
# 1. KalmanFilter 基类 — 线性高斯状态空间模型
# ======================================================================

class KalmanFilter:
    """线性高斯状态空间模型的卡尔曼滤波
    
    模型设定:
        x_t = F @ x_{t-1} + B @ u_t + w_t,   w_t ~ N(0, Q)
        y_t = H @ x_t + v_t,                  v_t ~ N(0, R)
    
    下标约定:
        x_pred[t] = E[x_t | y_{0:t-1}]  (预测状态)
        x_filt[t] = E[x_t | y_{0:t}]    (滤波状态)
        x_smth[t] = E[x_t | y_{0:T-1}]  (平滑状态)
    
    初始状态 initial_state_mean = E[x_0] (无观测时先验),
    initial_state_cov = Var[x_0].
    
    Parameters
    ----------
    F : ndarray, shape (d, d)
        状态转移矩阵
    H : ndarray, shape (p, d)
        观测矩阵
    Q : ndarray, shape (d, d)
        过程噪声协方差矩阵
    R : ndarray, shape (p, p)
        观测噪声协方差矩阵
    B : ndarray, shape (d, r), optional
        控制输入矩阵
    initial_state_mean : ndarray, shape (d,), optional
        初始状态均值 (默认 0 向量)
    initial_state_cov : ndarray, shape (d, d), optional
        初始状态协方差 (默认 1e6 * I, 扩散先验)
    """
    
    def __init__(self, F, H, Q, R, B=None,
                 initial_state_mean=None, initial_state_cov=None):
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.B = np.asarray(B, dtype=float) if B is not None else None
        
        self.dim_state = self.F.shape[0]
        self.dim_obs = self.H.shape[0]
        
        self.initial_state_mean = (
            np.zeros(self.dim_state) if initial_state_mean is None
            else np.asarray(initial_state_mean, dtype=float)
        )
        self.initial_state_cov = (
            np.eye(self.dim_state) * 1e6 if initial_state_cov is None
            else np.asarray(initial_state_cov, dtype=float)
        )
        
        self._reset_storage()
    
    def _reset_storage(self):
        """清除之前存储的结果"""
        self.predicted_state_means = None  # x_{t|t-1}
        self.predicted_state_covs = None   # P_{t|t-1}
        self.filtered_state_means = None   # x_{t|t}
        self.filtered_state_covs = None    # P_{t|t}
        self.smoothed_state_means = None   # x_{t|T}
        self.smoothed_state_covs = None    # P_{t|T}
        self.innovations = None            # nu_t
        self.innovation_covs = None        # S_t
        self.log_likelihood_ = None
    
    def filter(self, y, u=None):
        """前向卡尔曼滤波
        
        Parameters
        ----------
        y : ndarray, shape (T, p)
            观测序列
        u : ndarray, shape (T, r), optional
            控制输入序列
            
        Returns
        -------
        x_filt : ndarray, shape (T, d)
            滤波状态均值
        P_filt : ndarray, shape (T, d, d)
            滤波状态协方差
        """
        T = y.shape[0]
        d, p = self.dim_state, self.dim_obs
        
        # 预分配
        x_pred = np.empty((T, d))
        P_pred = np.empty((T, d, d))
        x_filt = np.empty((T, d))
        P_filt = np.empty((T, d, d))
        innov_arr = np.empty((T, p))
        S_arr = np.empty((T, p, p))
        
        # t=0 的预测 = 初始先验
        x_pred[0] = self.initial_state_mean.copy()
        P_pred[0] = self.initial_state_cov.copy()
        
        llk = 0.0
        
        for t in range(T):
            # ---- 更新步 ----
            innov_arr[t] = y[t] - self.H @ x_pred[t]          # nu_t
            S_arr[t] = self.H @ P_pred[t] @ self.H.T + self.R  # S_t
            
            # 卡尔曼增益: K_t = P_{t|t-1} @ H^T @ S_t^{-1}
            # 使用 solve 而非 inv 以提高稳定性
            K = solve(S_arr[t].T, (P_pred[t] @ self.H.T).T).T
            
            x_filt[t] = x_pred[t] + K @ innov_arr[t]           # x_{t|t}
            
            # Joseph 形式协方差更新 (保证对称正定)
            I_KH = np.eye(d) - K @ self.H
            P_filt[t] = (I_KH @ P_pred[t] @ I_KH.T
                         + K @ self.R @ K.T)
            
            # 对数似然贡献 (预测误差分解)
            sign, logdet = np.linalg.slogdet(S_arr[t])
            if sign <= 0:
                # S_t 非正定, 加微小正则
                S_arr[t] += 1e-8 * np.eye(p)
                sign, logdet = np.linalg.slogdet(S_arr[t])
            llk += -0.5 * (p * np.log(2 * np.pi) + logdet
                           + innov_arr[t] @ solve(S_arr[t], innov_arr[t]))
            
            # ---- 预测下一步 ----
            if t < T - 1:
                x_pred[t + 1] = self.F @ x_filt[t]
                P_pred[t + 1] = (self.F @ P_filt[t] @ self.F.T
                                 + self.Q)
                if self.B is not None and u is not None:
                    x_pred[t + 1] += self.B @ u[t + 1]
        
        # 存储结果
        self.predicted_state_means = x_pred
        self.predicted_state_covs = P_pred
        self.filtered_state_means = x_filt
        self.filtered_state_covs = P_filt
        self.innovations = innov_arr
        self.innovation_covs = S_arr
        self.log_likelihood_ = llk
        
        return x_filt, P_filt
    
    def smooth(self):
        """Rauch-Tung-Striebel 后向平滑
        
        利用全部观测数据 y_{0:T-1} 修正状态估计 (x_{t|T})。
        
        Returns
        -------
        x_smth : ndarray, shape (T, d)
            平滑状态均值
        P_smth : ndarray, shape (T, d, d)
            平滑状态协方差
        """
        if self.filtered_state_means is None:
            raise RuntimeError("必须先调用 filter() 再调用 smooth()")
        
        T = self.filtered_state_means.shape[0]
        d = self.dim_state
        
        x_smth = np.empty((T, d))
        P_smth = np.empty((T, d, d))
        
        # 从最后一个时点初始化
        x_smth[-1] = self.filtered_state_means[-1].copy()
        P_smth[-1] = self.filtered_state_covs[-1].copy()
        
        for t in range(T - 2, -1, -1):
            # 平滑增益 G_t = P_{t|t} @ F^T @ P_{t+1|t}^{-1}
            G = (self.filtered_state_covs[t] @ self.F.T
                 @ np.linalg.inv(self.predicted_state_covs[t + 1]))
            
            diff = x_smth[t + 1] - self.predicted_state_means[t + 1]
            x_smth[t] = self.filtered_state_means[t] + G @ diff
            
            P_smth[t] = (self.filtered_state_covs[t]
                         + G @ (P_smth[t + 1]
                                - self.predicted_state_covs[t + 1]) @ G.T)
        
        self.smoothed_state_means = x_smth
        self.smoothed_state_covs = P_smth
        
        return x_smth, P_smth
    
    def log_likelihood(self, y, u=None):
        """计算预测误差分解对数似然"""
        self.filter(y, u)
        return self.log_likelihood_
    
    def predict(self, y, n_steps, u_future=None):
        """预测未来状态和观测值
        
        Parameters
        ----------
        y : ndarray, shape (T, p)
            已观测数据
        n_steps : int
            预测步数
        u_future : ndarray, shape (n_steps, r), optional
            未来的控制输入
            
        Returns
        -------
        x_forecast : ndarray, shape (n_steps, d)
            预测状态均值
        y_forecast : ndarray, shape (n_steps, p)
            预测观测均值
        P_forecast : ndarray, shape (n_steps, d, d)
            预测状态协方差
        y_cov : ndarray, shape (n_steps, p, p)
            预测观测协方差
        """
        self.filter(y)
        
        x_prev = self.filtered_state_means[-1].copy()
        P_prev = self.filtered_state_covs[-1].copy()
        
        x_forecast = np.empty((n_steps, self.dim_state))
        y_forecast = np.empty((n_steps, self.dim_obs))
        P_forecast = np.empty((n_steps, self.dim_state, self.dim_state))
        y_cov = np.empty((n_steps, self.dim_obs, self.dim_obs))
        
        for j in range(n_steps):
            x_pred = self.F @ x_prev
            P_pred = self.F @ P_prev @ self.F.T + self.Q
            if self.B is not None and u_future is not None:
                x_pred += self.B @ u_future[j]
            
            y_pred = self.H @ x_pred
            
            x_forecast[j] = x_pred
            P_forecast[j] = P_pred
            y_forecast[j] = y_pred
            y_cov[j] = self.H @ P_pred @ self.H.T + self.R
            
            x_prev = x_pred
            P_prev = P_pred
        
        return x_forecast, y_forecast, P_forecast, y_cov
    
    def standardised_innovations(self):
        """返回标准化新息 (用于模型诊断)"""
        if self.innovations is None:
            raise RuntimeError("必须先调用 filter()")
        std_innov = np.empty_like(self.innovations)
        for t in range(len(self.innovations)):
            S_inv = np.linalg.inv(self.innovation_covs[t])
            # Cholesky 标准化: S^{-1/2} @ nu
            L = np.linalg.cholesky(self.innovation_covs[t])
            std_innov[t] = solve(L, self.innovations[t])
        return std_innov


# ======================================================================
# 2. LocalLevelModel — 局部水平模型
# ======================================================================

class LocalLevelModel:
    """局部水平模型 (随机游走 + 噪声)
    
    这是最简状态空间模型，用于直观理解 KF:
        y_t = mu_t + epsilon_t,   epsilon_t ~ N(0, sigma2_obs)
        mu_t = mu_{t-1} + eta_t,  eta_t    ~ N(0, sigma2_level)
    
    信号噪声比 q = sigma2_level / sigma2_obs:
        q -> 0: 滤波趋于常数 (信任先验 > 信任观测)
        q -> 大: 滤波紧密跟随观测 (信任观测 > 信任先验)
    """
    
    def __init__(self, sigma_obs=1.0, sigma_level=0.1):
        self.sigma_obs = sigma_obs
        self.sigma_level = sigma_level
        self.kf_result = None
        self._build_kf()
    
    def _build_kf(self):
        """根据当前参数构造 KalmanFilter 对象"""
        self.F = np.eye(1)
        self.H = np.eye(1)
        self.Q = np.array([[self.sigma_level ** 2]])
        self.R = np.array([[self.sigma_obs ** 2]])
    
    def fit(self, y, method='MLE'):
        """极大似然估计参数
        
        Parameters
        ----------
        y : ndarray, shape (T, 1) or (T,)
            观测序列
        
        Returns
        -------
        self : LocalLevelModel
            包含估计后的参数
        """
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        T = y.shape[0]
        
        def neg_loglik(params):
            log_sig2_obs = params[0]
            log_sig2_level = params[1]
            
            Q = np.array([[np.exp(log_sig2_level)]])
            R = np.array([[np.exp(log_sig2_obs)]])
            
            kf = KalmanFilter(self.F, self.H, Q, R)
            ll = kf.log_likelihood(y)
            return -ll
        
        # 初始值: 观测方差的一半分配给噪声, 十分之一给水平
        obs_var = np.var(y)
        x0 = [np.log(obs_var * 0.5), np.log(obs_var * 0.1)]
        
        result = minimize(neg_loglik, x0, method='L-BFGS-B')
        
        self.sigma_obs = np.sqrt(np.exp(result.x[0]))
        self.sigma_level = np.sqrt(np.exp(result.x[1]))
        self._build_kf()
        return self
    
    def filter(self, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        kf = KalmanFilter(self.F, self.H, self.Q, self.R)
        x_filt, P_filt = kf.filter(y)
        self.kf_result = kf
        return x_filt[:, 0], P_filt[:, 0, 0]
    
    def smooth(self, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        if self.kf_result is None:
            self.filter(y)
        x_smth, P_smth = self.kf_result.smooth()
        return x_smth[:, 0], P_smth[:, 0, 0]
    
    def log_likelihood(self, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        kf = KalmanFilter(self.F, self.H, self.Q, self.R)
        return kf.log_likelihood(y)


# ======================================================================
# 3. StructuralTimeSeries — 结构时间序列 (趋势 + 季节)
# ======================================================================

def _build_structural_matrices(seasonal_period,
                               sigma_level=0.1,
                               sigma_slope=0.01,
                               sigma_seasonal=0.01,
                               sigma_obs=1.0):
    """构建局部线性趋势 + 季节性的状态空间矩阵
    
    状态向量: [level, slope, seasonal_1, seasonal_2, ..., seasonal_{s-1}]^T
    
    Parameters
    ----------
    seasonal_period : int
        季节周期 (如 4=季度, 12=月度)
    """
    s = int(seasonal_period)
    dim_trend, dim_season = 2, max(s - 1, 0)
    dim_state = dim_trend + dim_season
    
    # --- 趋势组件 F_trend, Q_trend, H_trend ---
    # level_t = level_{t-1} + slope_{t-1} + eta_t
    # slope_t = slope_{t-1} + zeta_t
    F = np.zeros((dim_state, dim_state))
    F[:2, :2] = [[1.0, 1.0],
                 [0.0, 1.0]]
    
    # --- 季节性组件 (哑元形式) ---
    # gamma_t = -gamma_{t-1} - gamma_{t-2} - ... - gamma_{t-s+1} + omega_t
    if dim_season > 0:
        F[2, 2:] = -1.0                           # gamma_t 依赖于过去 s-1 项
        if dim_season > 1:
            F[3:, 2:-1] = np.eye(dim_season - 1)  # 移位: gamma_{t-i} -> gamma_{t-i+1}
    
    # --- 观测矩阵: y_t = level_t + seasonal_1_t + epsilon_t ---
    H = np.zeros((1, dim_state))
    H[0, 0] = 1.0    # level
    if dim_season > 0:
        H[0, 2] = 1.0  # seasonal_1
    
    # --- 过程噪声协方差 ---
    Q = np.zeros((dim_state, dim_state))
    Q[0, 0] = sigma_level ** 2
    Q[1, 1] = sigma_slope ** 2
    if dim_season > 0:
        Q[2, 2] = sigma_seasonal ** 2
    
    # --- 观测噪声 ---
    R = np.array([[sigma_obs ** 2]])
    
    return F, H, Q, R


class StructuralTimeSeries:
    """结构时间序列模型 — 趋势 + 季节分解
    
    模型组件:
        - 局部线性趋势 (level + 随机斜率)
        - 季节性 (哑元形式, 可指定周期)
        - 观测噪声
    
    Parameters
    ----------
    seasonal_period : int
        季节周期 (默认 4=季度数据)
    sigma_obs : float
        观测噪声标准差
    sigma_level : float
        水平扰动标准差
    sigma_slope : float
        斜率扰动标准差
    sigma_seasonal : float
        季节性扰动标准差
    """
    
    def __init__(self, seasonal_period=4, sigma_obs=1.0,
                 sigma_level=0.1, sigma_slope=0.01,
                 sigma_seasonal=0.01):
        self.seasonal_period = seasonal_period
        self.sigma_obs = sigma_obs
        self.sigma_level = sigma_level
        self.sigma_slope = sigma_slope
        self.sigma_seasonal = sigma_seasonal
        self.kf_result = None
        self._build_kf()
    
    def _build_kf(self):
        F, H, Q, R = _build_structural_matrices(
            self.seasonal_period,
            sigma_level=self.sigma_level,
            sigma_slope=self.sigma_slope,
            sigma_seasonal=self.sigma_seasonal,
            sigma_obs=self.sigma_obs,
        )
        self.F, self.H, self.Q, self.R = F, H, Q, R
    
    def fit(self, y, method='MLE'):
        """极大似然估计模型参数"""
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        def neg_loglik(params):
            log_s2_obs = params[0]
            log_s2_level = params[1]
            log_s2_slope = params[2]
            log_s2_seasonal = params[3]
            
            F, H, Q, R = _build_structural_matrices(
                self.seasonal_period,
                sigma_level=np.sqrt(np.exp(log_s2_level)),
                sigma_slope=np.sqrt(np.exp(log_s2_slope)),
                sigma_seasonal=np.sqrt(np.exp(log_s2_seasonal)),
                sigma_obs=np.sqrt(np.exp(log_s2_obs)),
            )
            kf = KalmanFilter(F, H, Q, R)
            ll = kf.log_likelihood(y)
            return -ll if np.isfinite(ll) else 1e12
        
        obs_var = np.var(y)
        result = minimize(
            neg_loglik,
            x0=[np.log(obs_var * 0.3),
                np.log(obs_var * 0.1),
                np.log(obs_var * 0.01),
                np.log(obs_var * 0.05)],
            method='L-BFGS-B',
        )
        
        self.sigma_obs = np.sqrt(np.exp(result.x[0]))
        self.sigma_level = np.sqrt(np.exp(result.x[1]))
        self.sigma_slope = np.sqrt(np.exp(result.x[2]))
        self.sigma_seasonal = np.sqrt(np.exp(result.x[3]))
        self._build_kf()
        return self
    
    def filter(self, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        kf = KalmanFilter(self.F, self.H, self.Q, self.R)
        x_filt, P_filt = kf.filter(y)
        self.kf_result = kf
        return x_filt, P_filt
    
    def smooth(self, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        if self.kf_result is None:
            self.filter(y)
        return self.kf_result.smooth()
    
    def decompose(self, y):
        """分解为水平、斜率、季节、残差分量
        
        Returns
        -------
        dict with keys: level, slope, seasonal, residual
        """
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        x_smth, _ = self.smooth(y)
        
        dim_season = max(self.seasonal_period - 1, 0)
        level = x_smth[:, 0]
        slope = x_smth[:, 1] if x_smth.shape[1] > 1 else None
        seasonal = x_smth[:, 2] if dim_season > 0 else None
        residual = (y[:, 0] - level - seasonal) if seasonal is not None else (y[:, 0] - level)
        
        return {'level': level, 'slope': slope,
                'seasonal': seasonal, 'residual': residual}


# ======================================================================
# 4. 使用示例
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # ====== 例 1: 局部水平模型 — 合成数据 ======
    print("=" * 50)
    print("例 1: 局部水平模型 (Local Level Model)")
    print("=" * 50)
    
    # 生成合成数据: y_t = mu_t + epsilon_t, mu_t = mu_{t-1} + eta_t
    T = 120
    true_sigma_obs = 0.8
    true_sigma_level = 0.15
    
    mu_true = np.zeros(T)
    for t in range(1, T):
        mu_true[t] = mu_true[t - 1] + np.random.normal(0, true_sigma_level)
    y_obs = mu_true + np.random.normal(0, true_sigma_obs, T)
    
    # 拟合局部水平模型
    llm = LocalLevelModel()
    llm.fit(y_obs.reshape(-1, 1))
    
    print(f"真实值:   sigma_obs = {true_sigma_obs:.3f}, "
          f"sigma_level = {true_sigma_level:.3f}")
    print(f"估计值:   sigma_obs = {llm.sigma_obs:.3f}, "
          f"sigma_level = {llm.sigma_level:.3f}")
    print(f"信号噪声比 q = {(llm.sigma_level / llm.sigma_obs) ** 2:.4f}")
    
    # 滤波和平滑
    x_filt, P_filt = llm.filter(y_obs.reshape(-1, 1))
    x_smth, P_smth = llm.smooth(y_obs.reshape(-1, 1))
    
    # 预测未来 20 步
    kf = KalmanFilter(llm.F, llm.H, llm.Q, llm.R)
    x_fc, y_fc, P_fc, y_cov = kf.predict(y_obs.reshape(-1, 1), 20)
    
    # 计算拟合优度 (平滑状态 vs 真实状态)
    rmse_filt = np.sqrt(np.mean((x_filt - mu_true) ** 2))
    rmse_smth = np.sqrt(np.mean((x_smth - mu_true) ** 2))
    print(f"滤波 RMSE (vs 真实状态): {rmse_filt:.4f}")
    print(f"平滑 RMSE (vs 真实状态): {rmse_smth:.4f}")
    
    # ====== 绘制例 1 结果 ======
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    t_idx = np.arange(T)
    fc_idx = np.arange(T, T + 20)
    
    # 图1: 滤波 vs 平滑
    ax = axes[0]
    ax.plot(t_idx, y_obs, 'k.', alpha=0.3, markersize=2, label='Observed')
    ax.plot(t_idx, mu_true, 'k-', linewidth=1.5, label='True state')
    ax.plot(t_idx, x_filt, 'b--', linewidth=1.2, label='Filtered (x_{t|t})')
    ax.plot(t_idx, x_smth, 'r-', linewidth=1.5, label='Smoothed (x_{t|T})')
    ax.fill_between(t_idx,
                     x_smth - 1.96 * np.sqrt(P_smth),
                     x_smth + 1.96 * np.sqrt(P_smth),
                     alpha=0.15, color='r', label='95% CI (smoothed)')
    ax.legend(fontsize=8)
    ax.set_title('Local Level Model: 滤波 vs 平滑估计', fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel('Level')
    
    # 图2: 预测 (含预测区间)
    ax = axes[1]
    ax.plot(t_idx, y_obs, 'k.', alpha=0.4, markersize=2, label='Observed')
    ax.plot(t_idx, x_filt, 'b-', linewidth=1, label='Filtered')
    ax.plot(fc_idx, y_fc[:, 0], 'r--', linewidth=1.5, label='Forecast')
    pred_ci = 1.96 * np.sqrt(y_cov[:, 0, 0])
    ax.fill_between(fc_idx,
                     y_fc[:, 0] - pred_ci,
                     y_fc[:, 0] + pred_ci,
                     alpha=0.2, color='r', label='95% Prediction Interval')
    ax.axvline(x=T - 1, color='gray', linestyle=':', alpha=0.7)
    ax.legend(fontsize=8)
    ax.set_title(f'Kalman Filter 预测 (下 {T-1} 步)', fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    # 图3: 标准化新息诊断
    ax = axes[2]
    std_innov = kf.standardised_innovations()
    ax.plot(t_idx, std_innov[:, 0], 'b-', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=1.96, color='r', linestyle=':', linewidth=0.5)
    ax.axhline(y=-1.96, color='r', linestyle=':', linewidth=0.5)
    ax.fill_between(t_idx, -1.96, 1.96, alpha=0.1, color='r')
    ax.set_title('标准化新息 (应近似 N(0,1), 95% 在虚线内)', fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel('Std. Innovation')
    
    plt.tight_layout()
    plt.savefig('outputs/local_level_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存至 outputs/local_level_demo.png\n")
    
    # ====== 例 2: 结构时间序列 — 季节数据 ======
    print("=" * 50)
    print("例 2: 结构时间序列 (趋势 + 季节)")
    print("=" * 50)
    
    # 生成合成季度数据 (s=4)
    np.random.seed(123)
    T2 = 80
    seasonal_period = 4
    
    level_true = np.cumsum(np.random.normal(0, 0.1, T2))
    slope_true = 0.02
    seasonal_true = np.tile([1.0, -0.5, -0.3, -0.2], T2 // 4 + 1)[:T2]
    y_season = level_true + np.arange(T2) * slope_true + seasonal_true \
               + np.random.normal(0, 0.5, T2)
    
    # 拟合结构时间序列
    sts = StructuralTimeSeries(seasonal_period=4)
    sts.fit(y_season)
    
    print(f"估计参数: sigma_obs={sts.sigma_obs:.3f}, "
          f"sigma_level={sts.sigma_level:.3f}, "
          f"sigma_slope={sts.sigma_slope:.4f}, "
          f"sigma_seasonal={sts.sigma_seasonal:.4f}")
    
    # 分解
    comp = sts.decompose(y_season)
    
    # 绘制分解结果
    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    
    axes2[0].plot(y_season, 'k-', linewidth=0.8, label='Observed')
    axes2[0].plot(comp['level'], 'r-', linewidth=1.2, label='Estimated Level')
    axes2[0].set_title('Structural Time Series Decomposition')
    axes2[0].legend(fontsize=8)
    
    axes2[1].plot(comp['slope'], 'g-', linewidth=1)
    axes2[1].axhline(y=slope_true, color='gray', linestyle='--')
    axes2[1].set_title('Slope Component')
    
    if comp['seasonal'] is not None:
        axes2[2].plot(comp['seasonal'], 'b-', linewidth=1, label='Estimated')
        axes2[2].plot(seasonal_true, 'k--', linewidth=0.8, alpha=0.6, label='True')
        axes2[2].legend(fontsize=8)
        axes2[2].set_title('Seasonal Component')
    
    axes2[3].plot(comp['residual'], 'mo-', markersize=2, linewidth=0.5)
    axes2[3].axhline(y=0, color='gray', linestyle='--')
    axes2[3].set_title('Residual')
    axes2[3].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('outputs/structural_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存至 outputs/structural_decomposition.png")
```

### 基于 statsmodels 的生产用法

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ====== 局部水平模型 ======
# 使用 Nile 河流量数据 (经典数据集)
nile = sm.datasets.nile.load_pandas().data['volume']
nile.index = range(len(nile))

# 局部水平模型
mod_ll = sm.tsa.UnobservedComponents(nile, 'local level')
res_ll = mod_ll.fit()
print(res_ll.summary())

# 绘制滤波和平滑估计
fig = res_ll.plot_components()
plt.show()

# ====== 结构时间序列 ======
# 局部线性趋势 + 季节
# mod = sm.tsa.UnobservedComponents(y, 'local linear trend', seasonal=12)
# res = mod.fit()
# print(res.summary())
# res.plot_components()

# ====== 预测 ======
# res_ll.predict(start=len(nile), end=len(nile) + 20)
# res_ll.get_prediction(start=len(nile), end=len(nile) + 20)

# ====== 模型诊断 ======
# from statsmodels.stats.diagnostic import acorr_ljungbox
# std_resid = res_ll.resid / res_ll.sigma
# lb_test = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
# print(lb_test)
```

## 参考文献

Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.

Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.

Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

Anderson, B. D. O., & Moore, J. B. (1979). *Optimal Filtering*. Prentice-Hall.

Rauch, H. E., Striebel, C. T., & Tung, F. (1965). Maximum Likelihood Estimates of Linear Dynamic Systems. *AIAA Journal*, 3(8), 1445–1450.

Julier, S. J., & Uhlmann, J. K. (1997). A New Extension of the Kalman Filter to Nonlinear Systems. *Proceedings of SPIE*, 3068, 182–193.
