---
title: Inverse Problem Root-Finding under Reliability — 可靠性约束下的反问题求解
type:
  - inverse-problem
  - optimization
  - quantile-reliability
  - uncertainty-quantification
domain:
  - materials-physics
source: 华数杯 2026 A 题（微构体导电介质）问题三四；反问题与可靠性优化
---
# Inverse Problem Root-Finding under Reliability — 可靠性约束下的反问题求解

- **来源**: 华数杯 2026 A 题问题三四（由导通概率反求最低填充量 / 最低成本混合配比）+ 反问题、二分/割线求根与双层随机优化
- **方法类别**: 反问题 / 双层优化 / 分位数可靠性约束 / 不确定性量化 / 一维求根
- **状态**: 2026-08-20

## 问题设定（华数杯 A 题规则）

- **问题三**：求满足导通概率 $\ge 90\%$ 的**最低体积分数/填充量**；
- **问题四**：两种介质成本 $c_A=1.05$ 元/μm³、$c_B=0.05$ 元/μm³，求在导通可靠性约束下**总成本最低的混合配比**。

两者都在随机布放（MC 仿真）之上反推输入量，属典型"由概率反求"反问题。

## 数学设定

### 反问题形式化（问题三）

对给定填充量 $x$，定义 $g(x) = P(\text{导通}\,|\,x)$（由 MC 仿真估计）。反问题为求
$$
x^* = \min\{ x : g(x) \ge \alpha \},\qquad \alpha = 0.9
$$
即 $g(x)$ 首次越过可靠性阈值的点。$g$ 在物理上单调不减（更多介质不降低导通概率，逾渗曲线陡增），可视为单调函数求根：
$$
g(x^*) = \alpha
$$

### 求根方法

1. **二分法**（需 $g$ 单调）：维护 $[x_l, x_u]$ 使 $g(x_l)<\alpha<g(x_u)$，迭代中点，$\text{iter}=\lceil\log_2((x_u-x_l)/\epsilon)\rceil$ 次收敛。
2. **割线法 / 反插值**：利用单调 $g$ 做线性插值 $x_{next}=x_l + (x_u-x_l)\tfrac{\alpha-g(x_l)}{g(x_u)-g(x_l)}$，收敛更快但需 $g$ 平滑。
3. 由于 $g$ 每次评估都要跑内层 MC，**评估次数最小化**是精度/预算权衡的核心（二分仅需 $\sim 20$ 次评估可达 6 位精度）。

### 双层结构（问题三/四）

外层优化（求根 / 配比寻优）反复调用**内层 MC**估计 $g(x)$：
$$
\min_{x}\ f(x) \quad \text{s.t.}\quad g(x)\ge \alpha,\ \ g(x)=P(\text{导通}|x)
$$
内层 MC 给 $g$ 带来抽样误差，外层需处理**随机目标**。

### 随机目标的不确定性处理

点估计 $\hat g(x)$ 有方差 $\frac{\hat p(1-\hat p)}{M}$。为满足可靠性约束（保守侧），用**置信下界**替代点估计：
$$
g_{\text{lo}}(x) = \hat g(x) - z \sqrt{\frac{\hat g(x)(1-\hat g(x))}{M}}
$$
求 $x^*=\min\{x: g_{\text{lo}}(x)\ge \alpha\}$ 得到**保守**（偏大）填充量；反之用上界得偏小值。固定外层的 $M$（或按需动态加大）以控制内层噪声对外层收敛的干扰。（嵌套 MC 若内层点数不足，会引入外层偏差。）

### 问题四：多目标成本配比

设 $N$ 个介质中 B（便宜球）占比 $\lambda$，A（贵柱）占比 $1-\lambda$，总成本
$$
f(\lambda) = N\bigl[ (1-\lambda) c_A v_A + \lambda\, c_B v_B \bigr]
$$
约束 $g(\lambda)\ge\alpha$（导通概率）。沿 $\lambda$ 一维搜索（或网格 + 二分），对每可行 $\lambda$ 用 MC 估 $g(\lambda)$，取满足约束的最小成本配比。

## 适用场景
- 华数杯 A 问题三：最低填充量（$P\ge90\%$）
- 华数杯 A 问题四：最低成本混合配比
- 一般"可靠性约束下的最小资源投入 / 反推输入"类问题（结构可靠性、冗余设计、安全裕度）

## 实现要点
- **单调性**：二分要求 $g$ 单调不减；若物理上不严格单调，先做网格粗扫找跨越区间再二分。
- **MC 预算拆分**：外层每点固定 $M_{\text{in}}$（如 500~2000），与置信下界公式联动，保证 $g_{lo}$ 有意义。
- **置信下界优先**：可靠性约束一律用下界，避免抽样侥幸越过阈值导致的"过度自信"解。
- **可复现**：所有内层 MC 用固定主种子 + 外层次序衍生种子，保证外层搜索确定。
- 输出：$x^*$、对应 $\hat g(x^*)$、区间、总 MC 评估次数。

## 代码

可运行示例：用二分法求满足 $P(\text{导通})\ge 0.9$ 的最小填充量（内层用简化伯努利探针替代真实几何布放，演示反问题结构，可替换为真实 MC 评估器）。

```python
import numpy as np

ALPHA = 0.9      # 可靠性阈值
Z = 1.96
M_IN = 600       # 内层 MC 评估次数

def g_hat(x):
    """简化内层 MC：返回给定填充量 x 的导通概率点估计。
    真实场景替换为 Monte Carlo 布放 + 几何渗透判定。"""
    rng = np.random.default_rng(int(x * 1e7))
    return rng.binomial(M_IN, phi_curve(x)) / M_IN

def phi_curve(x):
    """逾渗概率-填充量的 S 形曲线（示例形态，非真实 A 题数据）。"""
    return 1/(1+np.exp(-(x-0.008)*800))

def g_lower(x):
    """置信下界，用于保守满足置信约束。"""
    p = g_hat(x)
    return p - Z*np.sqrt(p*(1-p)/M_IN)

def bisect_min_volume(x_lo=0.0, x_hi=0.02, tol=1e-5):
    """在单调递增区间二分求 min{x: P>=alpha}（用置信下界）。"""
    assert g_lower(x_hi) >= ALPHA, "上界未达到可靠性阈值"
    while x_hi - x_lo > tol:
        m = 0.5*(x_lo+x_hi)
        if g_lower(m) >= ALPHA: x_hi = m
        else: x_lo = m
    return 0.5*(x_lo+x_hi)

x_star = bisect_min_volume()
print(f"最低填充量 x* = {x_star:.5f}  (P>=90%)")
print(f"该点点估计 g = {g_hat(x_star):.4f}, 置信下界 = {g_lower(x_star):.4f}")
```

## 参考文献
- Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. JASA, 44, 335–341.
- Broadbent, S. R., & Hammersley, J. M. (1957). Percolation processes. Proc. Cambridge Philos. Soc., 53, 629–641.
- Owen, A. B. (2013). Monte Carlo Theory, Methods and Examples（嵌套 MC 与外层偏差分析）.
