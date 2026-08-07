# Physarum Network Optimizer (PNO-GWO) — 黏菌管网络-灰狼混合优化算法

> 基于黏菌（Physarum polycephalum）管状网络自适应机制的原创群智能优化算法，
> 融合灰狼（GWO）等级包围机制与 5 种 SOTA 自适应机制。
> **状态**: v3.0 | **更新**: 2026-07-29 | **类别**: 群智能优化 / 生物启发 / 混合算法

## 算法演进

| 版本 | 日期 | 核心机制 | 相对前一版本提升 |
|------|------|---------|----------------|
| v1.0 | 2026-06-10 | PNO 图探索（电导率 + Lévy） | 基线 |
| v2.0 | 2026-07-27 | + GWO 等级包围 + 双模自适应切换 | Sphere 3.8e13×, Ackley 2.5e7× |
| **v3.0** | **2026-07-29** | **+ SHCA + CG-PSR + CMEA + CAS + Cauchy** | **Sphere 104×, Ackley 21×** |

## 生物灵感

黏菌（*Physarum polycephalum*）在寻找食物时构建管状网络。管道的**电导率（厚度）**根据营养流自适应调整：好路径上的管壁增厚，未使用的管道萎缩。网络整体通过这种局部正反馈实现全局最优路径的发现。

## v3.0 核心创新（新增 5 大自适应机制）

### M1: Success-History Conductivity Adaptation (SHCA)
借鉴 SHADE 的成功历史参数自适应，对 PNO 的 (alpha, beta) 参数进行动态采样：
- 维护成功 (alpha, beta) 对的历史记忆体
- 每代从记忆体采样 → 成功时用加权 Lehmer 均值更新记忆体
- 消除手动调参，自动适配不同函数

### M2: Conductivity-Guided Population Size Reduction (CG-PSR)
借鉴 L-SHADE 的种群缩减策略，但加入 PNO 独有信号：
- 计算**电导率熵** $H_{cond}$ 表征图结构的多样性
- 高熵时慢缩减（保持探索），低熵时快缩减（加速收敛）
- 多标准选择移除节点（适应度 + 度中心性 + 停滞次数）

### M3: Conductivity-Modulated External Archive (CMEA)
借鉴 JADE 的外部存档机制：
- 存储被替换的旧解
- 通过**电导率距离**调制重注入概率
- 高电导率区域更积极探索存档，防止早熟收敛

### M6: Conductivity-Adaptive Sigmoid (CAS)
替代 v2.0 的固定时间 sigmoid：
- p_gwo 不再由 t/T 决定，而是响应电导率熵的变化
- $p_{gwo} = \text{sigmoid}(k \cdot (H_{cond} - H_{target}))$
- 更自适应的探索-开发平衡

### M7: Cauchy-Conductivity Mutation
替代高斯扰动的柯西分布变异：
- 重尾分布提供偶尔的大跳跃，更有效逃逸局部最优
- 尺度由**逆电导率** × **停滞次数**调制
- 孤立节点大跳，连接良好节点小修正

## 数学设定

### 管道电导率更新（带 SHCA）
$$
D_{ij}(t+1) = D_{ij}(t) + \alpha_i \cdot \frac{|f_i - f_j|}{\|X_i - X_j\| + \varepsilon} - \beta_i \cdot D_{ij}(t)
$$
其中 $\alpha_i \sim \text{Cauchy}(M_\alpha, 0.1)$, $\beta_i \sim \text{Cauchy}(M_\beta, 0.1)$，$M_\alpha, M_\beta$ 为成功历史记忆体。

### 电导率熵（全局度量）
$$
H_{cond} = -\sum \frac{D_{ij}}{D_{total}} \ln \frac{D_{ij}}{D_{total}}
$$

### 自适应 GWO 概率（CAS）
$$
p_{gwo} = \text{sigmoid}(k \cdot (H_{cond} - H_{target})), \quad H_{target} = 0.8 - 0.5 \cdot t/T
$$

### 种群缩减（CG-PSR）
$$
N_{target} = 4 + (N_{init} - 4) \cdot (1 - \text{FES/maxFES})^{0.8}
$$
缩减速率由 $H_{cond}$ 调制：高熵慢减，低熵快减。

### Cauchy-Conductivity 变异
$$
X_i'(d) = X_i(d) + \text{Cauchy}(0, \gamma_0 \cdot (1 - \bar{D}_i) \cdot (1 + \text{stg}_i / 30))
$$

## 算法参数

| 参数 | v3.0 默认 | 说明 | 自适应？ |
|------|----------|------|---------|
| `n_pop` | 18×dim | 初始种群规模（L-SHADE 惯例） | 是（CG-PSR） |
| `alpha` | 0.5 | 管壁生长率初始值 | 是（SHCA） |
| `beta` | 0.2 | 管道衰减率初始值 | 是（SHCA） |
| `k_neighbors` | 5 | k近邻图连接数 | 否 |
| `use_shca` | True | 启用参数自适应 | — |
| `use_cgpsr` | True | 启用种群缩减 | — |
| `use_cmea` | True | 启用外部存档 | — |
| `use_cas` | True | 启用自适应 sigmoid | — |
| `use_cauchy` | True | 启用柯西变异 | — |
| `use_gcmms` | False | 启用多策略池（实验性） | — |
| `use_clls` | False | 启用局部搜索（实验性） | — |

## 适用场景

- ✅ **多峰函数优化**（Rastrigin, Ackley — 保持 PNO 的探索优势）
- ✅ **需自适应参数的复杂问题**（SHCA 自动适配）
- ✅ **中等维度问题**（30-100D，CG-PSR 优势明显）
- ✅ **需要混合探索-开发的问题**
- ❌ **需要极致单峰精度**（GWO 在 Sphere 上仍领先 125 个数量级）
- ❌ **计算资源极少**（图结构维护有额外开销）

## 基准测试表现（30D, FES=30000, median of 10 runs）

| 函数 | v3.0 | v2.0 | 提升倍数 | 主要贡献机制 |
|------|------|------|---------|------------|
| Sphere | **3.17e-17** | 3.32e-15 | **104×** | SHCA + CAS |
| Rastrigin | 34.84 | 31.05 | 持平 | — |
| Rosenbrock | 28.79 | 28.76 | 持平 | — |
| Ackley | **1.58e-10** | 3.37e-09 | **21×** | SHCA + CG-PSR |
| Griewank | 1.67e-16 | 0.00 | 持平 | — |

> 详细数据见 [benchmark_v3.md](benchmark_v3.md) | 完整测试脚本: `run_benchmark.py`

## 快速使用

```python
import numpy as np
from algorithm import PhysarumNetworkOptimizer

def sphere(x):
    return np.sum(x ** 2)

# 默认 v3.0（5 大自适应机制开启）
pno = PhysarumNetworkOptimizer(
    n_dim=30, bounds=(-100, 100), max_fes=30000
)
best_x, best_fit, convergence = pno.optimize(sphere)
print(f"Best fitness: {best_fit:.6e}")

# 关闭所有自适应机制 = 回退到基本 PNO-GWO
pno_basic = PhysarumNetworkOptimizer(
    use_shca=False, use_cgpsr=False, use_cmea=False,
    use_cas=False, use_cauchy=False
)
```

## 文件结构

```
physarum-network-optimizer/
├── algorithm.py              ← v3.0 核心（5 自适应机制 + 可开关）
├── algorithm_v2.py           ← v2.0 冻结版（对比用）
├── run_benchmark.py          ← 完整基准测试脚本
├── benchmark_v3.md           ← v3.0 基准测试报告
├── benchmark.md              ← v2.0 基准测试报告（历史）
├── README.md                 ← 本文件
├── demo.py                   ← 使用示例
└── test_algorithm.py         ← 单元测试（13 → 13 all pass）
```

## 引用

```
TbagDesign. (2026). Physarum Network Optimizer (PNO-GWO): 
A Slime Mold Tube Network-Grey Wolf Hybrid Optimization Algorithm. 
research-assistant, v3.0.
```
