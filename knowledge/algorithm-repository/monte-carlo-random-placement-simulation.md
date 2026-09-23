---
title: Monte Carlo Random Placement Simulation — 随机布放导通概率仿真
type:
  - simulation
  - monte-carlo
  - uncertainty-quantification
domain:
  - materials-physics
source: 华数杯 2026 A 题（微构体导电介质）问题二三；蒙特卡洛方法（Metropolis & Ulam, 1949）
---
# Monte Carlo Random Placement Simulation — 随机布放导通概率仿真

- **来源**: 华数杯 2026 A 题问题二三（指定体积分数下随机布放介质，多次仿真统计导通概率）+ 经典 Monte Carlo 仿真与伯努利频率估计
- **方法类别**: 蒙特卡洛 / 随机布放 / 概率估计 / 置信区间 / 方差缩减
- **状态**: 2026-08-20

## 问题设定（华数杯 A 题规则）

在边长 $L=10000\,\text{nm}$ 正方体内，按给定**体积分数** $\phi$ 随机布放介质（A 直圆柱高 $h=5000$ 底面半径 $30$；B 球半径 $200$）。依华数 A 题设定**允许介质相互贯穿、重叠**，故每个介质中心在腔体内独立均匀采样（含边界回卷），位置间不作碰撞剔除。每次布放的"导通与否"由几何渗透连通判定（见 `geometric-percolation-connectivity`）。问题二~四要求统计**导通概率** $P(\text{导通}\,|\,\phi)$。

华数杯目标分数：$\phi \in \{0.50\%,\ 0.60\%,\ 0.70\%,\ 1.00\%\}$。

## 数学设定

### 体积分数 → 介质数量

给定介质单个体积与总体积占比 $\phi$，所需个数：
$$
\phi = \frac{N \cdot V_{\text{unit}}}{L^3} \;\Rightarrow\; N = \left\lfloor \frac{\phi\, L^3}{V_{\text{unit}}} \right\rfloor
$$
- 球：$V_s = \tfrac{4}{3}\pi r^3$；
- 圆柱：$V_c = \pi r^2 h$。

### 伯努利频率估计

每次布放为独立伯努利试验 $X_i \sim \text{Bernoulli}(p)$（$X_i=1$ 导通）。$M$ 次仿真得 $\hat p = \frac{1}{M}\sum_i X_i$，$M\hat p \sim \text{Binomial}(M,p)$。

**Wilson 区间**（小 $p$/小 $n$ 更稳健）：
$$
p \in \frac{1}{1+z^2/M}\Bigl( \hat p + \frac{z^2}{2M} \pm z \sqrt{ \frac{\hat p(1-\hat p)}{M} + \frac{z^2}{4M^2} } \Bigr)
$$
**正态近似**（$M p \ge 5,\, M(1-p)\ge 5$）：
$$
\hat p \pm z \sqrt{\frac{\hat p(1-\hat p)}{M}}, \qquad z=z_{1-\alpha/2}
$$

### 样本数选择（保证精度）

要求"概率 $\ge 0.9$ 得到给定置信"。宽度 $w$ 的 $95\%$ 正态区间所需样本：
$$
M \gtrsim \Bigl( \frac{2\,z\, \sigma}{w}\Bigr)^2, \qquad \sigma = \sqrt{\hat p(1-\hat p)}
$$
$p=0.5$ 时最坏 $\sigma=0.5$。欲 $w=0.01$（1%）需 $M\approx (2\times1.96\times0.5/0.01)^2\approx 3.8\times 10^4$ 次仿真。

### 方差缩减

- **公共随机数（CRN）**：对比不同 $\phi$ 时用同一随机种子序列，消除跨方案随机性，凸显体积分数效应。
- **对偶变量（antithetic）**：对每个位置 $u_i\sim U(0,1)$ 同时采 $1-u_i$，配对平均，方差减半量级。
- **分层采样（stratified）**：沿 $x$ 方向把布放空间分层，每层固定配额采样。
- （若改用不可重叠布放）布放成功率的条件化：无碰撞布放高密度下有失败或耗时风险，须"采样直到成功"并对拒绝损耗计数；华数 A 允许重叠，此顾虑不存在。

## 适用场景
- 华数杯 A 问题二：给定 $\phi$ 估计导通概率 $\hat p$ + 置信区间
- 问题三/四：作为内层 $P(\text{导通}|\phi)$ 的评估器（反问题外层需多次调用）
- 一般随机几何布放 + 物理属性统计（逾渗概率曲线、连通阈值 $\phi_c$）

## 实现要点
- **固定随机种子**：`np.random.default_rng(seed)`，保证可复现（每 $\phi$ 用不同种子子序列，避免跨方案复用）。
- **允许重叠布放**：依华数 A 设定介质可贯穿/重叠，逐中心独立均匀采样，**不做碰撞剔除**——O(n) 即得，可支撑大样本量。若改用"不可重叠"物理（如排除贯穿），才需要拒绝采样并承担高密度下采样失败风险。
- **收敛监测**：随 $M$ 增大 $\hat p$ 应趋于稳定，区间收窄；用 $\Delta \hat p$ 或区间半宽阈值做早停。
- **阈值设定注意**：导通判据均为"距离 $< d_c$"。允许重叠时"重叠即导通"由 $2r$ 隐含（两球心距离 $<2r$ 时 $d<d_c$ 恒成立）；需衔接几何条目对"重叠介质是否导通"的约定。
- **输出**：$\hat p$、Wilson 区间、样本数、随机种子、时间，供复现与后续反问题回拉。

## 代码

可运行示例：给定体积分数与介质类型，随机布放 $N$ 个介质、统计导通概率与 Wilson 区间。

```python
import numpy as np

L = 10000.0; DC = 1.8; SEED = 42

def n_media(phi, r=200.0, kind='sphere'):
    if kind == 'sphere':
        vol = 4/3*np.pi*r**3
    else:                                  # cylinder: r, h
        h = 5000.0; vol = np.pi*r**2*h
    return int(np.floor(phi * L**3 / vol))

def random_placement(n, r, kind):
    """按体积分数随机布放介质中心（简化球模型）。

    华数 A 题明确"允许介质相互贯穿、重叠"，因此**不做碰撞剔除**：
    每个中心在腔体内独立均匀采样（含边界回卷的周期位置），复杂度 O(n)。
    """
    rng = np.random.default_rng(SEED + n)          # 每 phi 用不同种子，避免跨方案复用
    return rng.uniform(0, L, (n, 3))

def percolates(centers, r, dc=DC):
    """最近像邻接 + 端面接触，Union-Find 判导通。"""
    n = len(centers); par = list(range(n+2)); LFT, RGT = n, n+1
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: par[ra]=rb
    # 端面接触：球心到左/右面的距离 ≤ r+dc
    for i,c in enumerate(centers):
        if c[0] - r <= dc: union(i,LFT)
        if L - c[0] - r <= dc: union(i,RGT)
    # 邻接：最近像距离 ≤ 2r+dc（允许重叠则 2r 可被 dc 覆盖，重叠即导通）
    for i in range(n):
        for j in range(i+1,n):
            d = np.linalg.norm(centers[i]-centers[j])
            d = min(d, np.linalg.norm((centers[i]-centers[j])%L))  # 周期回卷
            if d <= 2*r + dc: union(i,j)
    return find(LFT)==find(RGT)

def est_conduct_prob(phi, n_sim=200, r=200.0):
    n = n_media(phi, r)
    hits = 0.0
    for _ in range(n_sim):
        places = random_placement(n, r, 'sphere')
        hits += percolates(places, r)
    p = hits/n_sim
    z = 1.96
    # Wilson 区间
    den = 1 + z*z/n_sim
    cen = (p + z*z/(2*n_sim))/den
    hw = z*np.sqrt(p*(1-p)/n_sim + z*z/(4*n_sim*n_sim))/den
    return p, (cen-hw, cen+hw)

# 允许重叠的关键：布放 O(n)、无拒绝，可提升样本量；演示用 n_sim=200
for phi in [0.005, 0.006, 0.007, 0.010]:
    p, (lo,hi) = est_conduct_prob(phi, n_sim=200)
    print(f"phi={phi*100:.2f}%  N={n_media(phi)}  p={p:.4f}  95%CI=[{lo:.4f},{hi:.4f}]")
```

> **运行说明**：本示例演示"允许重叠布放 + 导通概率估计"的逻辑流程，作为方法参考。
> 对本条目的目标赛题（体积分数 1% 附近）分位数估计较稀疏、`percolates` 为 O(n²)，直接以文内参数运行较慢且需大样本量才能得到非零估计；实际建模时应按赛题数据规模调整 `n_sim`、引入空间分桶/近邻加速，或与 `geometric-percolation-connectivity` 的连通核配合。

## 参考文献
- Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. J. Amer. Stat. Assoc., 44, 335–341.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA, 22, 209–212.
- Owen, A. B. (2013). Monte Carlo Theory, Methods and Examples（方差缩减、公共随机数）.
