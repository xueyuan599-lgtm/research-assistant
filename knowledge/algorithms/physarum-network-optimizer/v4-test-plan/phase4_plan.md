# PNO-GWO v4.0 Phase 4: 算法对比与扩展验证

> **目标**: 与主流优化算法进行 head-to-head 对比，验证 v4.0 的竞争力
> **依赖**: Phase 1-3 已完成（诊断→改进→工程验证）

---

## 1. Phase 1-3 成果回顾

| Phase | 内容 | 核心发现 |
|-------|------|---------|
| Phase 1 | v3.5 诊断 | CG-PSR 有害（280x 退化），FES 70% 更优，重启 15 最佳 |
| Phase 2 | v4.0 改进 | Sphere 101x，Griewank 257x，Ackley 3.51x 提升 |
| Phase 3 | 工程验证 | 可行性规则使 5 个工程问题全部 100% 可行率 |

---

## 2. Phase 4 任务规划

### 2.1 算法对比（核心任务）

#### 对比算法清单

| 算法 | 类型 | 实现方式 | 优先级 |
|------|------|---------|--------|
| **GWO** | 经典群智能 | 自实现（已有基础） | ★★★★★ |
| **PSO** | 经典群智能 | `pyswarm` 或自实现 | ★★★★★ |
| **DE/rand/1/bin** | 经典进化 | `scipy.optimize.differential_evolution` | ★★★★★ |
| **SHADE** | 自适应 DE | 自实现 | ★★★★☆ |
| **L-SHADE** | 种群缩减 DE | 自实现 | ★★★★☆ |
| **JADE** | 自适应 DE | 自实现 | ★★★☆☆ |
| **CMA-ES** | 进化策略 | `cma` 库 | ★★★☆☆ |
| **NLopt (COBYLA)** | 确定性局部 | `nlopt` 库 | ★★☆☆☆ |

#### 对比维度

| 维度 | 指标 | 说明 |
|------|------|------|
| 精度 | Best/Median/Std | 30 次运行统计 |
| 收敛速度 | FES 达到目标精度 | 单峰 1e-6，多峰 1e-2 |
| 鲁棒性 | 成功率 | 30 次中达到精度的比例 |
| 计算时间 | Wall-clock time | 含图构建开销 |
| 工程适用性 | 可行率 + Gap% | 5 个工程问题 |

### 2.2 测试集扩展

#### 基准函数（已有）

| 函数 | 维度 | 特征 |
|------|------|------|
| Sphere | 30D, 100D | 单峰、可分 |
| Rastrigin | 30D, 100D | 多峰、可分 |
| Rosenbrock | 30D, 100D | 单峰、不可分 |
| Ackley | 30D, 100D | 多峰、不可分 |
| Griewank | 30D, 100D | 多峰、不可分 |

#### 工程问题（已有）

| 问题 | 维度 | 约束 |
|------|------|------|
| Welded Beam | 4D | 4 |
| Pressure Vessel | 4D | 4 |
| Spring | 3D | 4 |
| Speed Reducer | 7D | 11 |
| Three-Bar Truss | 2D | 3 |

#### 新增测试（建议）

| 类别 | 问题 | 说明 |
|------|------|------|
| **CEC 2017 子集** | F1, F3, F6, F9, F12 | 学术对标（可选） |
| **更多工程问题** | Gear Train, Piston Lever | 扩展到 10+ 问题 |
| **高维工程问题** | 50D, 100D 版本 | 测试扩展性 |
| **真实数据** | IEEE 14/30 节点最优潮流 | 实际应用验证 |

### 2.3 统计检验

| 检验 | 用途 | 工具 |
|------|------|------|
| Wilcoxon 秩和检验 | 两算法对比 | `scipy.stats.wilcoxon` |
| Friedman 检验 | 多算法排名 | `scipy.stats.friedmanchisquare` |
| Nemenyi 事后检验 | 两两比较 | `scipy.stats` |
| 效应量 Cohen's d | 实际差异大小 | 自算 |

---

## 3. 实现计划

### 3.1 文件结构

```
v4-test-plan/
├── phase4_plan.md                    # 本文件
├── algorithms/                       # 对比算法实现
│   ├── __init__.py
│   ├── gwo.py                        # Grey Wolf Optimizer
│   ├── pso.py                        # Particle Swarm Optimization
│   ├── de.py                         # Differential Evolution
│   ├── shade.py                      # SHADE
│   ├── lshade.py                     # L-SHADE
│   └── jaya.py                       # Jaya (简单但有效)
├── phase4_comparison_runner.py       # 对比实验主脚本
├── phase4_statistical_analysis.py    # 统计检验脚本
└── reports/
    └── report_phase4_comparison.md   # 对比报告
```

### 3.2 时间估算

| 任务 | 预计耗时 | 依赖 |
|------|---------|------|
| 实现对比算法（GWO, PSO, DE, SHADE） | 2-3 小时 | 无 |
| 基准函数对比（5 函数 × 2 维度 × 6 算法 × 30 运行） | 3-4 小时 | 算法实现 |
| 工程问题对比（5 问题 × 6 算法 × 30 运行） | 2-3 小时 | 算法实现 |
| 统计检验 + 可视化 | 1-2 小时 | 实验数据 |
| 报告撰写 | 1 小时 | 统计结果 |
| **总计** | **8-12 小时** | — |

---

## 4. 预期产出

### 4.1 代码产出

| 文件 | 说明 |
|------|------|
| `algorithms/*.py` | 5-6 个对比算法实现 |
| `phase4_comparison_runner.py` | 统一对比实验框架 |
| `phase4_statistical_analysis.py` | 统计检验 + 可视化 |

### 4.2 结果产出

| 文件 | 说明 |
|------|------|
| `reports/report_phase4_comparison.md` | 对比报告 |
| `outputs/phase4_comparison_*.json` | 实验数据 |
| `outputs/figures/` | 收敛曲线、箱线图、秩和检验图 |

### 4.3 核心问题回答

1. **PNO-GWO v4.0 相比 GWO 提升多少？** → 验证图结构探索的价值
2. **与自适应 DE（SHADE）相比如何？** → 与 SOTA 对比
3. **在工程问题上是否优于通用优化器？** → 实际应用价值
4. **计算开销是否可接受？** → 图构建的代价

---

## 5. 关键决策点

| 决策 | 选项 | 建议 |
|------|------|------|
| 对比算法数量 | 3 / 5 / 8 | 先做 5 个（GWO, PSO, DE, SHADE, CMA-ES） |
| 是否包含 CEC 2017 | 是 / 否 | 跳过（延续 Phase 1 决策） |
| 高维测试 | 30D / 100D | 先 30D，有时间再 100D |
| 可视化 | 静态 / 交互式 | matplotlib 静态图（可复现） |

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 对比算法实现有 bug | 结果不可信 | 用已知最优解验证每个算法 |
| 计算时间过长 | 实验不可行 | 先跑 10 次，确认后再跑 30 次 |
| 统计检验不显著 | 无法下结论 | 增加运行次数或降低维度 |
| PNO-GWO 表现不佳 | 论文贡献弱 | 分析优势场景（多峰、约束密集） |