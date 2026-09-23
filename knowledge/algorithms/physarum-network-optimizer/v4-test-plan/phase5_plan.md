# Phase 5: 冲击 Q2 实验计划

> 目标：补齐全套实验，使论文达到 Applied Soft Computing / EAAI 级别（Q2）
> 预计总工作量：3-4 周（可并行的任务已标注）

---

## 总体架构

```
Phase 5 冲击 Q2
│
├─ 5A 消融实验（P0，2-3 天）
│   ├─ 5A.1 SHCA 记忆消融
│   ├─ 5A.2 外部归档消融
│   ├─ 5A.3 策略池消融
│   ├─ 5A.4 Cauchy 变异消融
│   └─ 5A.5 Nelder-Mead 精炼消融
│
├─ 5B CEC 2017 约束优化测试集（P0，3-5 天）
│   ├─ 5B.1 实现 CEC2017 约束函数（28 个）
│   ├─ 5B.2 跑 PNO-GWO + 6 个对比算法
│   └─ 5B.3 统计分析 + 报告
│
├─ 5C 高维可扩展性（P1，1-2 天，与 5B 并行）
│   ├─ 5C.1 100D 测试
│   ├─ 5C.2 300D 测试
│   └─ 5C.3 500D 测试
│
├─ 5D SOTA 对比算法扩展（P1，2-3 天，与 5B 并行）
│   ├─ 5D.1 LSHADE-cnEPSO
│   ├─ 5D.2 jSO
│   ├─ 5D.3 EA4eig
│   └─ 5D.4 CMODE (多目标差分进化)
│
├─ 5E 计算复杂度分析（P1，1 天）
│   ├─ 5E.1 时间复杂度测量
│   ├─ 5E.2 空间复杂度分析
│   └─ 5E.3 FLOPS 估算
│
└─ 5F 综合报告（最后 1-2 天）
    ├─ 5F.1 合并所有实验结果
    ├─ 5F.2 生成 Phase 5 综合报告
    └─ 5F.3 论文定位建议
```

---

## 5A 消融实验（P0，审稿人必问）

### 5A.1 目标

量化 PNO-GWO 每个核心机制的独立贡献，回答"哪个组件真正有用"。

### 5A.2 消融配置

| 配置名 | SHCA 记忆 | 外部归档 | 策略池 | Cauchy 变异 | Nelder-Mead |
|--------|-----------|---------|--------|-------------|-------------|
| Full (v4.0) | ✅ | ✅ | ✅ | ✅ | ✅ |
| w/o SHCA | ❌ | ✅ | ✅ | ✅ | ✅ |
| w/o Archive | ✅ | ❌ | ✅ | ✅ | ✅ |
| w/o Strategy | ✅ | ✅ | ❌ | ✅ | ✅ |
| w/o Cauchy | ✅ | ✅ | ✅ | ❌ | ✅ |
| w/o NM | ✅ | ✅ | ✅ | ✅ | ❌ |
| Baseline (全去) | ❌ | ❌ | ❌ | ❌ | ❌ |

### 5A.3 实现方案

在 `algorithm_v4.py` 的 `PhysarumNetworkOptimizer.__init__` 中添加配置开关：

```python
class PhysarumNetworkOptimizer:
    def __init__(self, ..., 
                 use_shca=True, use_archive=True, 
                 use_strategy_pool=True, use_cauchy=True, 
                 use_nelder_mead=True):
        self.use_shca = use_shca
        self.use_archive = use_archive
        self.use_strategy_pool = use_strategy_pool
        self.use_cauchy = use_cauchy
        self.use_nelder_mead = use_nelder_mead
```

然后在对应代码段加 `if self.use_xxx:` 守卫。

### 5A.4 测试集

| 类别 | 问题 | 维度 |
|------|------|------|
| Benchmark | Sphere, Rastrigin, Rosenbrock, Ackley, Griewank | 30D |
| Engineering | Welded Beam, Pressure Vessel, Spring | 原始维度 |

- 每个配置 × 每个问题 × 30 次运行
- 总计：7 配置 × 8 问题 × 30 次 = 1680 次运行

### 5A.5 分析指标

| 指标 | 说明 |
|------|------|
| ΔFitness | 移除该组件后适应度的平均退化幅度 |
| ΔFeasibility | 移除该组件后可行率的变化 |
| 统计显著性 | Wilcoxon 检验：Full vs w/o XXX |
| 效应量 | Cohen's d |

### 5A.6 预期输出

- `outputs/phase5_ablation.json` — 原始数据
- `outputs/figures/ablation_heatmap.png` — 组件贡献热力图
- `outputs/report_phase5_ablation.md` — 消融报告

---

## 5B CEC 2017 约束优化测试集（P0，社区基准）

### 5B.1 目标

在 CEC 2017 约束优化竞赛的标准测试集上评估 PNO-GWO，与文献中的 SOTA 结果对比。

### 5B.2 测试函数

CEC 2017 约束优化竞赛包含 **28 个函数**（C01-C28），分为：

| 类型 | 函数编号 | 特征 |
|------|---------|------|
| 可分离 | C01-C03 | 变量可分离，约束简单 |
| 多峰 | C04-C09 | 多局部最优，约束复杂 |
| 混合 | C10-C15 | 变量耦合，非线性约束 |
| 复合 | C16-C21 | 多种特征组合 |
| 高维约束 | C22-C28 | 10D/20D/30D，约束密度高 |

**关键参数：**
- 维度：10D / 20D / 30D（不同函数不同）
- 每个函数独立的可行域
- 已知最优解（竞赛提供参考值）

### 5B.3 实现方案

**选项 A：使用 `opfunu` Python 库**
```bash
pip install opfunu
```
该库提供 CEC 2017 约束优化测试函数的 Python 实现。

**选项 B：自实现**
从 CEC 2017 官方文档实现 28 个函数。工作量较大但可控。

**推荐选项 A**，节省时间，代码经过社区验证。

### 5B.4 对比算法

| 算法 | 来源 | CEC 2017 报告最优 |
|------|------|-------------------|
| PNO-GWO (本文) | — | 待测 |
| GWO | Phase 4 已实现 | 待测 |
| PSO | Phase 4 已实现 | 待测 |
| DE | Phase 4 已实现 | 待测 |
| SHADE | Phase 4 已实现 | 待测 |
| CMA-ES | Phase 4 已实现 | 待测 |
| LSHADE-cnEPSO | 5D 新增 | 文献报告 Top-3 |
| jSO | 5D 新增 | 文献报告 Top-5 |
| EA4eig | 5D 新增 | 文献报告竞争力强 |
| CMODE | 5D 新增 | 文献报告专门处理约束 |

另需从文献中引用 CEC 2017 竞赛的 **官方排名结果** 作为参考。

### 5B.5 运行规格

| 参数 | 值 |
|------|-----|
| 独立运行次数 | 30（CEC 标准） |
| 最大 FES | 200,000（CEC 2017 标准） |
| 记录频率 | 每 1,000 FES 记录一次最优值 |

### 5B.6 分析指标

| 指标 | CEC 标准要求 |
|------|-------------|
| 最优值、均值、标准差 | 每个函数的 30 次运行统计 |
| Friedman 检验 + Nemenyi 事后检验 | 总体排名 |
| Wilcoxon 秩和检验 | 逐函数两两对比 |
| 收敛曲线 | FES vs 最优适应度（28 个函数） |
| 可行率 | 约束满足情况 |

### 5B.7 预期输出

- `outputs/phase5_cec2017.json` — 原始数据
- `outputs/figures/cec_friedman.png` — Friedman 排名图
- `outputs/figures/cec_convergence_*.png` — 28 个收敛曲线
- `outputs/figures/cec_boxplot_*.png` — 28 个箱线图
- `outputs/report_phase5_cec2017.md` — CEC 测试报告

---

## 5C 高维可扩展性测试（P1）

### 5C.1 目标

验证 PNO-GWO 在高维空间（100D/300D/500D）的性能衰减情况。

### 5C.2 测试集

| 问题 | 100D | 300D | 500D |
|------|------|------|------|
| Sphere | ✅ | ✅ | ✅ |
| Rastrigin | ✅ | ✅ | ✅ |
| Rosenbrock | ✅ | ✅ | ✅ |
| Ackley | ✅ | ✅ | ✅ |
| Griewank | ✅ | ✅ | ✅ |

### 5C.3 运行规格

| 参数 | 100D | 300D | 500D |
|------|------|------|------|
| 最大 FES | 100,000 | 300,000 | 500,000 |
| 种群大小 | 100 | 200 | 300 |
| 运行次数 | 30 | 30 | 30 |

### 5C.4 分析指标

| 指标 | 说明 |
|------|------|
| 适应度 vs 维度 | 维度增长时的性能衰减曲线 |
| 运行时间 vs 维度 | 计算成本增长曲线 |
| 与 GWO/CMA-ES 对比 | 高维下相对优势是否保持 |
| 收敛速度 vs 维度 | FES 到达目标精度所需次数 |

### 5C.5 预期输出

- `outputs/phase5_scalability.json`
- `outputs/figures/scalability_fitness.png` — 适应度 vs 维度
- `outputs/figures/scalability_time.png` — 运行时间 vs 维度
- `outputs/report_phase5_scalability.md`

---

## 5D SOTA 对比算法扩展（P1）

### 5D.1 目标

补充 3-4 个当前 SOTA 算法，提升对比实验的说服力和公平性。

### 5D.2 新增算法

| 算法 | 全称 | 特点 | 实现难度 |
|------|------|------|---------|
| **LSHADE-cnEPSO** | L-SHADE with Constrained EPSO | CEC 2017 约束优化竞赛 Top-3 | 中 |
| **jSO** | improved jSO | CEC 2016 冠军，约束能力强 | 中 |
| **EA4eig** | EA with Eigenvector-based crossover | 强于高维约束 | 中 |
| **CMODE** | Constrained Multi-Objective DE | 专门处理约束的 DE 变体 | 中 |

### 5D.3 实现方案

优先查找开源实现：
- GitHub 搜索算法名 + "python" 或 "cec 2017"
- 论文附录中的伪代码自实现
- `opfunu` / `pyMetaheuristic` 等库

每个算法实现为统一接口：
```python
class LSHADEcnEPSO:
    def optimize(self, obj_func, n_dim, bounds, max_fes):
        # ... 
        return best_x, best_fit, convergence_curve
```

### 5D.4 预期输出

- `v4-test-plan/algorithms/lshade_cneeso.py`
- `v4-test-plan/algorithms/jso.py`
- `v4-test-plan/algorithms/ea4eig.py`
- `v4-test-plan/algorithms/cmode.py`

---

## 5E 计算复杂度分析（P1）

### 5E.1 目标

量化 PNO-GWO 的计算成本，回答"5-6x 开销是否值得"。

### 5E.2 分析维度

| 维度 | 方法 |
|------|------|
| **时间复杂度** | 测量不同 N（种群大小）和 D（维度）下的运行时间，拟合 O(f(N,D)) |
| **空间复杂度** | 分析内存使用（图的邻接矩阵、归档、记忆） |
| **FLOPS 估算** | 统计每次迭代的浮点运算次数 |
| **每组件成本** | 测量 SHCA/归档/策略池/Cauchy/NM 各占总时间的比例 |

### 5E.3 实验设计

**时间复杂度测量：**
```
对每个 (N, D) 组合：
    N ∈ {30, 50, 100, 200, 500}
    D ∈ {10, 30, 50, 100, 200, 500}
    运行 5 次，取中位数运行时间
    拟合 T(N, D) = a * N^b * D^c
```

**组件时间占比：**
```
在算法各阶段插入计时器：
    - 图构建时间
    - SHCA 记忆读写时间
    - 归档维护时间
    - 策略选择时间
    - Cauchy 变异时间
    - Nelder-Mead 精炼时间
    - 其他（种群更新、评估等）
```

### 5E.4 预期输出

- `outputs/phase5_complexity.json`
- `outputs/figures/complexity_time_vs_dim.png`
- `outputs/figures/complexity_time_vs_pop.png`
- `outputs/figures/complexity_component_pie.png` — 各组件时间占比饼图
- `outputs/report_phase5_complexity.md`

---

## 5F 综合报告

### 5F.1 内容结构

```
Phase 5 综合报告
├─ 1. 消融实验结论
│   └─ 哪些组件是关键贡献？哪些可以简化？
├─ 2. CEC 2017 排名
│   └─ PNO-GWO 在社区标准基准上的位置
├─ 3. 高维可扩展性结论
│   └─ 维度增长对性能和计算成本的影响
├─ 4. SOTA 对比总结
│   └─ 与最新算法的竞争力
├─ 5. 计算复杂度结论
│   └─ 开销是否值得？优化方向？
├─ 6. 论文定位建议
│   ├─ 基于全部实验结果的目标期刊推荐
│   ├─ 摘要框架
│   └─ 创新点提炼
└─ 7. 后续改进方向
    └─ v4.1 改进建议
```

---

## 执行计划

### 时间线

```
Week 1：
├─ Day 1-2: 实现消融开关 + 跑消融实验（5A）
├─ Day 3-5: 实现 CEC 2017 函数 + 跑 CEC 实验（5B）
└─ Day 3-5: [并行] 实现新 SOTA 算法（5D）

Week 2：
├─ Day 1-3: CEC 实验续跑 + 新算法跑 CEC（5B + 5D）
├─ Day 4-5: 高维测试（5C）
└─ Day 5: 计算复杂度分析（5E）

Week 3：
├─ Day 1-2: 统计分析全部结果
├─ Day 3-4: 撰写综合报告（5F）
└─ Day 5: 论文框架建议
```

### 资源需求

| 资源 | 需求 |
|------|------|
| 计算时间 | CEC 28 函数 × 10 算法 × 30 次 × 200K FES ≈ 8400 次运行 |
| Python 库 | opfunu (CEC 函数), numpy, scipy, matplotlib, seaborn |
| 磁盘 | ~500MB（结果 JSON + 图表 PNG） |
| 人力 | 主要是编码和等待运行，分析可自动化 |

### 依赖关系

```
5A（消融） ─────────────────────────────┐
5B（CEC） ──────────┐                    │
5D（新算法）───────┤→ 5B 用新算法重跑 ──┤
5C（高维） ────────┘                    │
5E（复杂度）──────────────────────────────┤
                                         ↓
                                    5F 综合报告
```

---

## 成功标准

| 指标 | Q2 达标线 | 衡量方式 |
|------|----------|---------|
| CEC 2017 排名 | Top-5 | Friedman 排名 |
| 消融实验 | 每个组件贡献显著 | Wilcoxon p < 0.05 |
| 高维性能 | 500D 下仍优于 PSO/DE | 统计检验 |
| 计算复杂度 | 开销 O(ND²) 或更优 | 拟合分析 |
| SOTA 对比 | 至少在约束问题上 Top-3 | Friedman 排名 |