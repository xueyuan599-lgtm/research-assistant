# MCMC — 马尔可夫链蒙特卡洛

- **来源**: Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. *The Journal of Chemical Physics*, 21(6), 1087–1092.
- **DOI**: 10.1063/1.1699114
- **方法类别**: 贝叶斯方法 / 计算统计

## 数学设定

### Bayesian 推断框架
MCMC 的核心目标是贝叶斯后验采样。给定数据 $D$ 和参数 $\theta$：

$$
p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)} \propto p(D \mid \theta) \, p(\theta)
$$

其中 $p(D \mid \theta)$ 为似然，$p(\theta)$ 为先验，$p(D)$ 为边际似然（归一化常数，通常不可解析计算）。MCMC 在不显式计算 $p(D)$ 的情况下从 $p(\theta \mid D)$ 采样。

### 马尔可夫链基础
一个马尔可夫链是随机变量序列 $X_1, X_2, X_3, \dots$，满足**马尔可夫性**（无后效性）：

$$
P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_1) = P(X_{t+1} \mid X_t)
$$

- **转移核** $P(x \to x')$：从状态 $x$ 转移到 $x'$ 的概率密度
- **平稳分布** $\pi(x)$：满足全局平衡方程
  $$
  \pi(x') = \sum_{x} \pi(x) \, P(x \to x')
  $$
- **细致平衡**（可逆性）：链满足细致平衡条件时 $\pi$ 为平稳分布
  $$
  \pi(x) \, P(x \to x') = \pi(x') \, P(x' \to x)
  $$
- **遍历性**：无论初始状态如何，链依分布收敛到平稳分布
  $$
  \lim_{t \to \infty} P(X_t \in A \mid X_0 = x_0) = \pi(A), \quad \forall x_0, A
  $$

### Metropolis-Hastings 算法

MH 是最基本的 MCMC 算法，通过提议分布生成候选状态并以特定概率接受/拒绝。

**算法流程**：
1. 初始化 $\theta^{(0)}$
2. 对 $t = 1, \dots, T$：
   - 从提议分布生成候选 $\theta^* \sim q(\theta^* \mid \theta^{(t-1)})$
   - 计算接受概率：
     $$
     \alpha = \min\left(1, \; \frac{\pi(\theta^*)}{\pi(\theta^{(t-1)})} \cdot \frac{q(\theta^{(t-1)} \mid \theta^*)}{q(\theta^* \mid \theta^{(t-1)})}\right)
     $$
   - 以概率 $\alpha$ 接受：$\theta^{(t)} = \theta^*$；否则 $\theta^{(t)} = \theta^{(t-1)}$

**随机游走 Metropolis-Hastings** (RW-MH)：
$$
q(\theta^* \mid \theta^{(t-1)}) = \mathcal{N}(\theta^* \mid \theta^{(t-1)}, \Sigma)
$$
此时提议分布对称，$q(x' \mid x) = q(x \mid x')$，接受概率简化为：
$$
\alpha = \min\left(1, \; \frac{\pi(\theta^*)}{\pi(\theta^{(t-1)})}\right)
$$

### Gibbs 采样

Gibbs 采样是 MH 的一种特例，每步以概率 1 接受。通过依次从每个变量的**满条件分布**采样实现，适用于高维分块更新。

**算法流程**（$d$ 维参数 $\theta = (\theta_1, \dots, \theta_d)$）：
1. 初始化 $\theta^{(0)} = (\theta_1^{(0)}, \dots, \theta_d^{(0)})$
2. 对 $t = 1, \dots, T$：
   - 依次从每个条件分布采样：
     $$
     \theta_1^{(t+1)} \sim p(\theta_1 \mid \theta_2^{(t)}, \theta_3^{(t)}, \dots, \theta_d^{(t)}, D)
     $$
     $$
     \theta_2^{(t+1)} \sim p(\theta_2 \mid \theta_1^{(t+1)}, \theta_3^{(t)}, \dots, \theta_d^{(t)}, D)
     $$
     $$
     \vdots
     $$
     $$
     \theta_d^{(t+1)} \sim p(\theta_d \mid \theta_1^{(t+1)}, \theta_2^{(t+1)}, \dots, \theta_{d-1}^{(t+1)}, D)
     $$

Gibbs 采样的关键优势是**无调参**——只要条件分布是标准分布（正态、Gamma、Beta 等），可直接采样。

### Hamiltonian Monte Carlo (HMC)

HMC 利用梯度信息高效探索高维后验空间，通过引入辅助动量变量和 Hamiltonian 动力学。

**Hamiltonian 系统**：
- 势能（负对数后验）：$U(\theta) = -\log \pi(\theta) = -\log p(D \mid \theta) - \log p(\theta)$
- 动能（动量 $r \sim \mathcal{N}(0, M)$）：$K(r) = \frac{1}{2} r^\top M^{-1} r$
- 总 Hamiltonian：$H(\theta, r) = U(\theta) + K(r)$

**蛙跳积分**（Leapfrog Integration, 步长 $\varepsilon$，$L$ 步）：
$$
r_{t + \varepsilon/2} = r_t - \frac{\varepsilon}{2} \nabla_\theta U(\theta_t)
$$
$$
\theta_{t + \varepsilon} = \theta_t + \varepsilon \, M^{-1} r_{t + \varepsilon/2}
$$
$$
r_{t + \varepsilon} = r_{t + \varepsilon/2} - \frac{\varepsilon}{2} \nabla_\theta U(\theta_{t + \varepsilon})
$$

**Metropolis 校正**：对蛙跳积分 $L$ 步后的 $(\theta^*, r^*)$，计算
$$
\alpha = \min\left(1, \; \exp\left(-H(\theta^*, r^*) + H(\theta^{(t-1)}, r^{(t-1)})\right)\right)
$$

**NUTS** (No-U-Turn Sampler, Hoffman & Gelman 2014)：自动选择蛙跳步数 $L$，当路径开始"折返"（$\rho \cdot (\theta_+ - \theta_-) < 0$）时停止，免除了 $L$ 的手动调参。

### 诊断指标

**有效样本量** (Effective Sample Size, ESS)：
$$
N_{\text{eff}} = \frac{N}{1 + 2 \sum_{k=1}^{\infty} \rho_k}
$$
其中 $\rho_k$ 为滞后 $k$ 的自相关系数。ESS 衡量链中独立信息量，低 ESS 表示链高度自相关，采样效率低。

**Gelman-Rubin 统计量** $\hat{R}$（Potential Scale Reduction Factor）：
$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}, \quad
\hat{V} = \frac{N-1}{N}W + \frac{1}{N}B
$$
其中 $W$ 为链内方差，$B$ 为链间方差。$\hat{R} \to 1$ 表示链收敛，通常要求 $\hat{R} < 1.01$。

**可视化诊断**：
- **Trace Plot**：迭代轨迹图，应呈现"毛虫状"平稳波动，无明显趋势
- **Autocorrelation Plot**：自相关系数随滞后下降越快越好
- **密度图**：多链后验密度应基本重合

## 关键假设
- 目标分布 $\pi(\theta)$ 满足**已知到归一化常数**：$p(\theta \mid D) \propto p(D \mid \theta) p(\theta)$ 可逐点计算
- 提议分布 $q(\theta^* \mid \theta)$ 的支撑集覆盖目标分布的支撑集（不可约性）
- 链运行足够长以达到收敛（遍历性）
- 马尔可夫链是**非周期**且**不可约**的
- HMC 要求目标分布对 $\theta$ 可微（梯度存在）

## 适用场景
- **贝叶斯后验推断**：复杂模型（多层次模型、混合模型、空间模型）的后验采样
- **高维参数空间**：维度 $d$ 较高时，Gibbs 和 HMC 比简单 MH 更高效
- **归一化常数不可解**：几乎所有非共轭贝叶斯模型
- **模型比较**：DIC (偏差信息准则)、WAIC (广泛可用的信息准则)、留一法交叉验证、贝叶斯因子
- **潜变量模型**：主题模型 (LDA)、隐 Markov 模型、项目反应理论 (IRT)
- **缺失数据处理**：通过数据增强将缺失值视为潜变量一同采样

### 不适用
- **确定性优化问题**：应使用梯度下降、牛顿法等
- **极高维度 + 强相关性**（$d > 10^5$ 且参数高度相关）：变分推断 (VI) 或 拉普拉斯近似更可行
- **实时在线推断**：MCMC 计算代价大，VI / 递推贝叶斯更合适
- **共轭先验可直接解析求解**：无需 MCMC，精确后验闭合形式已知
- **数据规模极大**：每次迭代需遍历全部数据，可尝试 SGLD (随机梯度 Langevin 动力学)

## 实现要点

| 参数 | 说明 | 建议 |
|------|------|------|
| 提议方差 $\sigma^2$ (RW-MH) | 控制提议步长 | 目标接受率 $\approx 0.234$ (高维), $0.44$ (一维) |
| 步长 $\varepsilon$ (HMC) | 蛙跳积分步长 | 过大导致高拒绝率，过小导致探索缓慢 |
| 步数 $L$ (HMC) | 蛙跳积分步数 | 过短类似 RW-MH，过长浪费计算。NUTS 自动选择 |
| 链数 $K$ | 独立并行链数 | 建议 $\ge 4$ 条链，用于 $\hat{R}$ 诊断 |
| 燃烧期 $B$ | 丢弃的初始迭代 | 通常 $B = N/2$ 或根据 $\hat{R}$ 判断 |
| 稀释间隔 | 降低链自相关 | 多数情况不需要（现代诊断接受自相关） |

### 工程经验
1. **先粗后精**：先用 Gibbs/RW-MH 快速探测后验形态，再切到 HMC/NUTS 精细采样
2. **多链胜过单链**：4 条链提供 $\hat{R}$ 诊断，可检测收敛失败和多模态问题
3. **$\hat{R} < 1.01$ 是硬门槛**：若未满足，延长燃烧期或调整提议分布
4. **重参数化**：对于层级模型（非中心化参数化），将相关参数变换到近似独立空间可极大提升采样效率
5. **ESS/s 是效率指标**：不仅是总样本量，单位时间的有效样本量才是真正的比较标准
6. **适应性提议**：在燃烧期内自适应调整提议方差（如 Haario 自适应 MH），但收敛后必须固定

### 工具选型
| 工具 | 特点 | 适用场景 |
|------|------|----------|
| PyMC | Python 生态，NUTS 自动调参 | 通用贝叶斯建模 |
| Stan | HMC/NUTS 最成熟实现，C++ 后端 | 复杂层级模型，高维 |
| Pyro | 深度学习+概率编程，支持 VI+SVI | 大规模/深度贝叶斯 |
| NIMBLE | R 生态，灵活的自定义 MCMC | 生态学、遗传学 |
| JAGS | Gibbs 采样，简单易用 | 教学和老模型复现 |

## 完整 Python 代码

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp


# ============================================================
# 1. Metropolis-Hastings 采样器
# ============================================================

class MetropolisHastings:
    """Metropolis-Hastings 采样器（随机游走版本）"""

    def __init__(self, log_target, proposal_std=0.5, random_state=None):
        """
        参数
        ----------
        log_target : callable
            对数目标分布函数 log π(θ)
        proposal_std : float 或 array
            提议正态分布的标准差
        random_state : int
            随机种子
        """
        self.log_target = log_target
        self.proposal_std = np.atleast_1d(proposal_std)
        self.rng = np.random.default_rng(random_state)
        self._trace = []
        self._log_probs = []
        self._n_accepted = 0
        self._n_total = 0

    def sample(self, init, n_iter=5000, burnin=1000, thin=1):
        """
        运行 MH 链

        参数
        ----------
        init : array
            初始参数
        n_iter : int
            总迭代次数
        burnin : int
            燃烧期长度
        thin : int
            稀释间隔

        返回
        -------
        self : MetropolisHastings
        """
        theta = np.asarray(init, dtype=float)
        log_curr = self.log_target(theta)

        self._trace = []
        self._log_probs = []
        self._n_accepted = 0
        self._n_total = 0

        for i in range(n_iter + burnin):
            # 提议新状态
            theta_star = theta + self.rng.normal(0, self.proposal_std, size=theta.shape)
            log_star = self.log_target(theta_star)

            # MH 接受/拒绝
            log_alpha = log_star - log_curr
            if np.log(self.rng.uniform()) < log_alpha:
                theta = theta_star.copy()
                log_curr = log_star
                if i >= burnin:
                    self._n_accepted += 1

            if i >= burnin:
                self._n_total += 1
                if self._n_total % thin == 0:
                    self._trace.append(theta.copy())
                    self._log_probs.append(log_curr)

        self._trace = np.array(self._trace)
        self._log_probs = np.array(self._log_probs)
        return self

    @property
    def trace(self):
        """采样轨迹 (n_samples, n_params)"""
        return self._trace

    @property
    def log_probs(self):
        """对数后验轨迹"""
        return self._log_probs

    @property
    def acceptance_rate(self):
        """接受率"""
        return self._n_accepted / self._n_total if self._n_total > 0 else 0.0

    @property
    def n_params(self):
        return self._trace.shape[1] if self._trace.ndim > 1 else 1

    def mean(self):
        """后验均值"""
        return self._trace.mean(axis=0)

    def std(self):
        """后验标准差"""
        return self._trace.std(axis=0)

    def quantile(self, q):
        """后验分位数"""
        return np.quantile(self._trace, q, axis=0)


# ============================================================
# 2. Gibbs 采样器（以二维正态为例）
# ============================================================

class Gibbs2DGaussian:
    """
    Gibbs 采样器 — 二维正态分布

    目标：p(x, y) = N([x, y] | [μ_x, μ_y], [[σ_xx, σ_xy], [σ_yx, σ_yy]])
    条件分布：
        x|y ~ N(μ_x + σ_xy/σ_yy·(y - μ_y), σ_xx - σ_xy²/σ_yy)
        y|x ~ N(μ_y + σ_xy/σ_xx·(x - μ_x), σ_yy - σ_xy²/σ_xx)
    """

    def __init__(self, mean, cov, random_state=None):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.rng = np.random.default_rng(random_state)

        # 条件分布参数
        mu_x, mu_y = self.mean
        sxx, sxy = self.cov[0, 0], self.cov[0, 1]
        syx, syy = self.cov[1, 0], self.cov[1, 1]

        self.cond_x_mean = lambda y: mu_x + sxy / syy * (y - mu_y)
        self.cond_x_std = np.sqrt(sxx - sxy ** 2 / syy)
        self.cond_y_mean = lambda x: mu_y + sxy / sxx * (x - mu_x)
        self.cond_y_std = np.sqrt(syy - sxy ** 2 / sxx)

        self._trace = []

    def sample(self, init, n_iter=5000, burnin=1000):
        """
        运行 Gibbs 链

        参数
        ----------
        init : (2,) array
            初始值 (x0, y0)
        n_iter : int
            采样迭代次数（燃烧期后）
        burnin : int
            燃烧期长度

        返回
        -------
        self : Gibbs2DGaussian
        """
        x, y = init
        self._trace = []

        total = n_iter + burnin
        for i in range(total):
            # 从条件分布依次采样
            x = self.rng.normal(self.cond_x_mean(y), self.cond_x_std)
            y = self.rng.normal(self.cond_y_mean(x), self.cond_y_std)
            if i >= burnin:
                self._trace.append([x, y])

        self._trace = np.array(self._trace)
        return self

    @property
    def trace(self):
        return self._trace

    @property
    def acceptance_rate(self):
        """Gibbs 总是接受"""
        return 1.0


# ============================================================
# 3. Hamiltonian Monte Carlo (HMC)
# ============================================================

class HMC:
    """Hamiltonian Monte Carlo 采样器"""

    def __init__(self, log_target, grad_log_target, step_size=0.1, n_steps=10,
                 mass=None, random_state=None):
        """
        参数
        ----------
        log_target : callable
            对数目标分布 log π(θ)
        grad_log_target : callable
            对数目标梯度 ∇log π(θ)
        step_size : float
            蛙跳积分步长 ε
        n_steps : int
            蛙跳积分步数 L
        mass : array
            质量矩阵 M（对角元素）
        random_state : int
        """
        self.log_target = log_target
        self.grad_log_target = grad_log_target
        self.epsilon = step_size
        self.L = n_steps
        self.M = np.atleast_1d(mass) if mass is not None else None
        self.rng = np.random.default_rng(random_state)
        self._trace = []
        self._n_accepted = 0
        self._n_total = 0

    def _leapfrog(self, theta, r):
        """蛙跳积分"""
        grad = self.grad_log_target(theta)
        # 半步动量更新
        r_half = r + 0.5 * self.epsilon * grad
        # 整步位置更新
        theta_new = theta + self.epsilon * r_half
        # 半步动量更新（在新位置）
        grad_new = self.grad_log_target(theta_new)
        r_new = r_half + 0.5 * self.epsilon * grad_new
        return theta_new, r_new

    def _hamiltonian(self, theta, r):
        """计算 H = -log π(θ) + ½rᵀM⁻¹r"""
        potential = -self.log_target(theta)
        kinetic = 0.5 * np.sum(r ** 2)  # 假设 M = I
        return potential + kinetic

    def sample(self, init, n_iter=2000, burnin=500):
        """
        运行 HMC 链
        """
        theta = np.asarray(init, dtype=float)
        d = len(theta)
        if self.M is None:
            self.M = np.ones(d)

        self._trace = []
        self._n_accepted = 0
        self._n_total = 0

        total = n_iter + burnin
        for i in range(total):
            # 采样动量 r ~ N(0, M)
            r = self.rng.normal(0, np.sqrt(self.M))

            theta_curr = theta.copy()
            r_curr = r.copy()
            H_curr = self._hamiltonian(theta_curr, r_curr)

            # 蛙跳积分 L 步
            theta_prop, r_prop = theta_curr.copy(), r_curr.copy()
            for _ in range(self.L):
                theta_prop, r_prop = self._leapfrog(theta_prop, r_prop)
            # 反转动量以保证可逆性
            r_prop = -r_prop
            H_prop = self._hamiltonian(theta_prop, r_prop)

            # Metropolis 接受/拒绝
            log_alpha = -H_prop + H_curr  # = -(H_prop - H_curr)
            if np.log(self.rng.uniform()) < log_alpha:
                theta = theta_prop.copy()
                if i >= burnin:
                    self._n_accepted += 1

            if i >= burnin:
                self._n_total += 1
                self._trace.append(theta.copy())

        self._trace = np.array(self._trace)
        return self

    @property
    def trace(self):
        return self._trace

    @property
    def acceptance_rate(self):
        return self._n_accepted / self._n_total if self._n_total > 0 else 0.0

    def mean(self):
        return self._trace.mean(axis=0)

    def std(self):
        return self._trace.std(axis=0)


# ============================================================
# 4. 诊断函数
# ============================================================

def trace_plot(samples, param_names=None, figsize=(10, 6)):
    """
    绘制定量轨迹图 (Trace Plot)

    参数
    ----------
    samples : ndarray, shape (n_chains, n_iter, n_params)
    param_names : list
    """
    n_chains, n_iter, n_params = samples.shape
    if param_names is None:
        param_names = [f'θ_{i}' for i in range(n_params)]

    fig, axes = plt.subplots(n_params, 1, figsize=figsize, squeeze=False)
    colors = plt.cm.Set1(np.linspace(0, 1, n_chains))

    for p in range(n_params):
        ax = axes[p, 0]
        for c in range(n_chains):
            ax.plot(samples[c, :, p], color=colors[c], lw=0.5, alpha=0.7,
                    label=f'Chain {c + 1}' if p == 0 else None)
        ax.set_ylabel(param_names[p], fontsize=12)
        if p == n_params - 1:
            ax.set_xlabel('Iteration', fontsize=12)
        else:
            ax.tick_params(labelbottom=False)
        ax.axhline(y=samples[:, :, p].mean(), color='black', ls='--', lw=1)

    if n_chains > 1:
        axes[0, 0].legend(fontsize=10, ncol=n_chains)

    fig.suptitle('Trace Plot', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def autocorr_plot(samples, max_lag=50, param_names=None, figsize=(10, 6)):
    """
    绘制自相关图

    参数
    ----------
    samples : ndarray, shape (n_chains, n_iter, n_params)
    max_lag : int
    """
    n_chains, n_iter, n_params = samples.shape
    if param_names is None:
        param_names = [f'θ_{i}' for i in range(n_params)]

    fig, axes = plt.subplots(n_params, 1, figsize=figsize, squeeze=False)
    colors = plt.cm.Set1(np.linspace(0, 1, n_chains))

    for p in range(n_params):
        ax = axes[p, 0]
        for c in range(n_chains):
            acf = np.array([1] + [
                np.corrcoef(samples[c, :-k, p], samples[c, k:, p])[0, 1]
                for k in range(1, max_lag + 1)
            ])
            ax.bar(range(max_lag + 1), acf, width=0.4, color=colors[c],
                   alpha=0.5, label=f'Chain {c + 1}' if p == 0 else None)
        ax.axhline(y=0, color='gray', lw=0.5)
        ax.axhline(y=1.96 / np.sqrt(n_iter), color='red', ls='--', lw=0.8)
        ax.axhline(y=-1.96 / np.sqrt(n_iter), color='red', ls='--', lw=0.8,
                   label='95% CI' if p == 0 else None)
        ax.set_ylabel(f'Autocorr({param_names[p]})', fontsize=12)
        if p == n_params - 1:
            ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylim(-0.2, 1.05)

    if n_chains > 1:
        axes[0, 0].legend(fontsize=10)
    fig.suptitle('Autocorrelation Plot', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def gelman_rubin(samples):
    """
    计算 Gelman-Rubin $\hat{R}$ 统计量

    参数
    ----------
    samples : ndarray, shape (n_chains, n_iter, n_params)

    返回
    -------
    r_hat : ndarray, shape (n_params,)
    """
    n_chains, n_iter, n_params = samples.shape

    # 链内方差 W
    chain_means = samples.mean(axis=1)       # (n_chains, n_params)
    chain_vars = samples.var(axis=1, ddof=1) # (n_chains, n_params)
    W = chain_vars.mean(axis=0)               # (n_params,)

    # 链间方差 B
    overall_mean = chain_means.mean(axis=0)   # (n_params,)
    B = n_iter * np.var(chain_means, axis=0, ddof=1)  # (n_params,)

    # 边际后验方差估计
    V_hat = (n_iter - 1) / n_iter * W + (1 / n_iter) * B

    # R-hat
    r_hat = np.sqrt(V_hat / W)
    return r_hat


def ess(samples):
    """
    计算有效样本量 (ESS)

    参数
    ----------
    samples : ndarray, shape (n_chains, n_iter, n_params)

    返回
    -------
    n_eff : ndarray, shape (n_params,)
    """
    n_chains, n_iter, n_params = samples.shape

    n_eff = np.zeros(n_params)
    for p in range(n_params):
        # 合并多链
        flat = samples[:, :, p].flatten()
        N = len(flat)

        # 计算自相关（使用 FFT 加速）
        # 截断到 N//2
        max_lag = min(N // 2, 500)
        acf = np.ones(max_lag + 1)
        for k in range(1, max_lag + 1):
            acf[k] = np.corrcoef(flat[:-k], flat[k:])[0, 1]

        # 找到第一个负自相关处截断（Geyer 截断规则）
        positive_run = 0
        for k in range(1, max_lag + 1):
            if acf[k] > 0:
                positive_run += 1
            else:
                break
        # 取到正自相关结束处
        cutoff = min(positive_run, max_lag)

        # ESS = N / (1 + 2 * sum(rho_k))
        if cutoff > 0:
            n_eff[p] = N / (1 + 2 * np.sum(acf[1:cutoff + 1]))
        else:
            n_eff[p] = N

    return n_eff


# ============================================================
# 5. 使用示例：贝叶斯线性回归
# ============================================================

def bayesian_linear_regression_example():
    """
    贝叶斯线性回归 MCMC 采样示例

    模型：y_i = α + β·x_i + ε_i, ε_i ~ N(0, σ²)
    先验：α ~ N(0, 10²), β ~ N(0, 10²), σ ~ HalfCauchy(0, 2)
    目标：从后验 p(α, β, σ | D) 采样
    """
    print("=" * 60)
    print("贝叶斯线性回归 — MCMC 采样示例")
    print("=" * 60)

    # ---------- 生成合成数据 ----------
    np.random.seed(42)
    N = 100
    true_alpha, true_beta, true_sigma = 1.5, 2.3, 0.8

    x = np.random.uniform(-3, 3, N)
    y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, N)

    print(f"\n真实值: α={true_alpha}, β={true_beta}, σ={true_sigma}")
    print(f"样本量: N={N}")

    # ---------- 定义对数后验 ----------
    def log_posterior(theta):
        alpha, beta, log_sigma = theta
        sigma = np.exp(log_sigma)  # 在无界空间上采样

        # 先验: α ~ N(0, 10²), β ~ N(0, 10²), logσ ~ uniform (improper)
        log_prior = (stats.norm.logpdf(alpha, 0, 10)
                     + stats.norm.logpdf(beta, 0, 10))

        # 似然: y_i ~ N(α + β·x_i, σ²)
        y_pred = alpha + beta * x
        log_lik = np.sum(stats.norm.logpdf(y, y_pred, sigma))

        return log_prior + log_lik

    # 对数后验梯度（用于 HMC）
    def grad_log_posterior(theta):
        alpha, beta, log_sigma = theta
        sigma = np.exp(log_sigma)
        y_pred = alpha + beta * x
        resid = y - y_pred

        grad_alpha = np.sum(resid) / sigma ** 2 - alpha / 100
        grad_beta = np.sum(resid * x) / sigma ** 2 - beta / 100
        grad_log_sigma = np.sum(resid ** 2 / sigma ** 2 - 1)

        return np.array([grad_alpha, grad_beta, grad_log_sigma])

    # ---------- MH 采样 ----------
    print("\n--- Metropolis-Hastings ---")
    mh = MetropolisHastings(
        log_posterior,
        proposal_std=[0.3, 0.3, 0.15],
        random_state=42
    )
    mh.sample(init=[0, 0, 0], n_iter=10000, burnin=2000, thin=2)
    print(f"接受率: {mh.acceptance_rate:.3f}")
    print(f"后验均值: α={mh.mean()[0]:.3f}, β={mh.mean()[1]:.3f}, σ={np.exp(mh.mean()[2]):.3f}")

    # ---------- HMC 采样 ----------
    print("\n--- Hamiltonian Monte Carlo ---")
    hmc = HMC(
        log_posterior,
        grad_log_posterior,
        step_size=0.08,
        n_steps=15,
        random_state=42
    )
    hmc.sample(init=[0, 0, 0], n_iter=4000, burnin=1000)
    print(f"接受率: {hmc.acceptance_rate:.3f}")
    print(f"后验均值: α={hmc.mean()[0]:.3f}, β={hmc.mean()[1]:.3f}, σ={np.exp(hmc.mean()[2]):.3f}")

    # ---------- Gibbs: 二维正态演示 ----------
    print("\n--- Gibbs 采样: 二维正态 ---")
    mean = [2, -1]
    cov = [[1.5, 0.8],
           [0.8, 2.0]]
    gibbs = Gibbs2DGaussian(mean, cov, random_state=42)
    gibbs.sample(init=[0, 0], n_iter=5000, burnin=1000)
    gibbs_mean = gibbs.trace.mean(axis=0)
    gibbs_cov = np.cov(gibbs.trace.T)
    print(f"真实均值: {mean}")
    print(f"Gibbs 均值: {gibbs_mean.round(3)}")
    print(f"真实协方差:\n{np.array(cov)}")
    print(f"Gibbs 协方差:\n{np.round(gibbs_cov, 3)}")

    # ---------- 诊断: 多链 ----------
    print("\n--- 收敛诊断 ---")
    n_chains = 4
    n_iter = 5000
    chains_mh = np.zeros((n_chains, n_iter, 3))

    for c in range(n_chains):
        mh_c = MetropolisHastings(
            log_posterior,
            proposal_std=[0.3, 0.3, 0.15],
            random_state=42 + c
        )
        mh_c.sample(init=[c - 2, c - 2, c - 2], n_iter=n_iter, burnin=2000)
        chains_mh[c] = mh_c.trace

    r_hat = gelman_rubin(chains_mh)
    n_eff = ess(chains_mh)
    param_names = ['α', 'β', 'log σ']
    print(f"{'参数':<8} {'R-hat':<10} {'ESS':<10}")
    print("-" * 30)
    for i, name in enumerate(param_names):
        print(f"{name:<8} {r_hat[i]:<10.4f} {n_eff[i]:<10.0f}")

    # ---------- 可视化 ----------
    fig_trace = trace_plot(chains_mh, param_names)
    fig_autocorr = autocorr_plot(chains_mh, max_lag=40, param_names=param_names)

    # 后验密度 vs 真实值
    fig_density, axes = plt.subplots(1, 3, figsize=(14, 4))
    names_plot = ['α (intercept)', 'β (slope)', 'σ (noise)']
    true_vals = [true_alpha, true_beta, true_sigma]

    for i in range(3):
        ax = axes[i]
        flat = chains_mh[:, :, i].flatten()
        if i == 2:  # sigma 从 log 空间变换回来
            flat = np.exp(flat)
        ax.hist(flat, bins=60, density=True, alpha=0.6, color='steelblue',
                edgecolor='white')
        ax.axvline(true_vals[i], color='red', ls='--', lw=2.5,
                   label=f'True = {true_vals[i]}')
        ax.set_xlabel(names_plot[i], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)

    fig_density.suptitle('Posterior Distribution vs True Values', fontsize=14, y=1.02)
    plt.tight_layout()

    plt.show()
    print("\n✓ 完成！")

    return {
        'mh': mh,
        'hmc': hmc,
        'gibbs': gibbs,
        'r_hat': r_hat,
        'n_eff': n_eff
    }


if __name__ == "__main__":
    results = bayesian_linear_regression_example()
```

### 基于 PyMC 的生产用法

```python
import pymc as pm
import arviz as az

# 贝叶斯线性回归（使用 PyMC 的 NUTS 采样器）
def bayesian_lr_pymc(x, y):
    with pm.Model() as model:
        # 先验
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfCauchy('sigma', beta=2)

        # 似然
        mu = alpha + beta * x
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

        # NUTS 采样（自动调参）
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=42,
            nuts_sampler='nutpie'  # 更快的 NUTS 后端
        )

    # 诊断
    summary = az.summary(trace, round_to=3)
    r_hat = az.rhat(trace)
    ess = az.ess(trace)
    az.plot_trace(trace)
    az.plot_autocorr(trace)

    return trace, summary

# 使用
# trace, summary = bayesian_lr_pymc(x, y)
# print(summary)
```

## 参考文献
1. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. *The Journal of Chemical Physics*, 21(6), 1087–1092.
2. Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97–109.
3. Geman, S., & Geman, D. (1984). Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 6(6), 721–741.
4. Neal, R. M. (1993). Probabilistic Inference Using Markov Chain Monte Carlo Methods. *Technical Report CRG-TR-93-1*, University of Toronto.
5. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623.
7. Brooks, S., Gelman, A., Jones, G. L., & Meng, X. L. (Eds.). (2011). *Handbook of Markov Chain Monte Carlo*. Chapman & Hall/CRC.
8. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.
