# Algorithm Repository — 多标签元数据 Schema

> 本文件是 `algorithm-repository/` 全部条目 frontmatter 的唯一权威规范。
> 所有新条目必须遵循本 schema；旧条目已统一迁移。
> **作用**：让"按题型谱系三层（决策层/计算层/物理层）多标签联合匹配"的检索得以程序化/协议化执行。

## 统一 frontmatter 格式

每个条目 `.md` 文件**第一行必须是 `---`**，紧跟 YAML 块，再 `---`，然后才是正文（原 `# 标题`）。

```yaml
---
title: <条目标题，中英皆可，与正文 # 标题一致>
type:             # 必填 · 方法类型（多标签，≥1 个），取值见下方“type 受控词表”
  - <标签A>
  - <标签B>
domain:           # 必填 · 应用领域（多标签，≥1 个），取值见下方“domain 受控词表”
  - <领域A>
source:           # 可选 · 来源简述（论文/项目），用于索引展示
---

# 标题
……正文原样保留……
```

> **YAML 数组写法**：使用 `- 项` 列表或内联 `[A, B]` 均可，处理器统一读作多标签集合。
> 已用 `---` 作分隔线但无 key 的旧条目，须改为真正的 key: value frontmatter。

## `type` 受控词表（方法族 / 方法类型）

> type 标签直接映射赛题三层谱系（决策层 / 计算层 / 物理层）的匹配维度。
> 标签为固定受控词，取**小写英文**，便于检索协议精确匹配。

### 决策层（解决"最终要什么"——外层目标/求解策略）
| 标签 | 含义 | 典型条目 |
|------|------|---------|
| `optimization` | 优化 / 数学规划（含线性、非线性、混合整数、双层、随机规划） | adaptive-admm, bayesian-dro, pareto-dominance |
| `inverse-problem` | 反问题 / 由目标反求输入 | 华数A 问题三类（库内如 DRO、bilevel 可承接） |
| `quantile-reliability` | 分位数 / 可靠性约束 | 华数A 问题三/四类 |
| `mult(objective` → 见 `multi-objective` | 多目标折中 / Pareto | pareto-dominance |
| `pareto` | Pareto 支配 / 多目标折中 | pareto-dominance |
| `sensitivity` | 灵敏度 / 稳健性 / 不确定性分析 | sensitivity-analysis, conformal |
| `uncertainty-quantification` | 不确定性量化 | conformal*, gaussian-process |

### 计算层（解决"怎么算出来"——内层迭代/求解机制）
| 标签 | 含义 | 典型条目 |
|------|------|---------|
| `simulation` | 仿真（Monte Carlo、随机布放、物理过程模拟） | 华数A 问题二类 |
| `monte-carlo` | Monte Carlo 随机方法 | mcmc, 华数A 问题二 |
| `graph` | 图论 / 网络 / 连通性 | physarum (图搜索), network-causal |
| `connectivity` | 连通性判定 / 渗透 | 华数A 问题一类 |
| `dynamic-programming` | 动态规划 | — |
| `heuristic` | 元启发式 / 群智能 | gray-langurs, physarum |
| `numerical` | 数值解算（ODE/PDE 离散、迭代求解） | fourier-neural-operator, adaptive-admm |
| `statistical-inference` | 统计推断 | causal*, tmle, dml |
| `clustering` | 聚类 | clustering, mofa |
| `dimensionality-reduction` | 降维 / 特征提取 | dimensionality-reduction, fpca |
| `classification` | 分类 | svm, random-forest, xgboost |
| `regression` | 回归 / 拟合 | lasso, random-forest, xgboost, gaussian-process |
| `ensemble-learning` | 集成学习（bagging/boosting/stacking） | random-forest, xgboost, functional-ensemble |
| `forecasting` | 时间序列预测 | arima, prophet, garch, transformer, lstm |
| `generative-modeling` | 生成模型 | flow-matching, autoencoder, alphafold3 |
| `deep-learning` | 深度学习（网络架构） | transformer, lstm, mamba, fno |
| `physics-informed` | 物理信息约束学习（PINN 类） | physics-informed-lstm, dual-level-pi, lstm-pinn-* |
| `sequence-modeling` | 序列建模 | lstm, mamba, transformer |
| `matrix-tensor` | 矩阵 / 张量分解 | tensor-cp-decomposition |
| `bandit-rl` | 强化学习 / 序贯决策 | — |
| `computer-vision` | 计算机视觉任务（分割/检测/自监督视觉） | segment-anything, masked-autoencoder |
| `signal-processing` | 信号/时频处理 | tic-fusionnet, mdstft, state-space |
| `preprocessing` | 数据预处理（缺失值/异常值/缩放编码/防泄漏/插补） | mice |

### 物理层（解决"底层机理是什么"——核心关系建模）
| 标签 | 含义 | 典型条目 |
|------|------|---------|
| `geometry` | 几何（三维空间、距离、形体） | 华数A 问题一~四类（几何内核） |
| `percolation` | 渗透 / 逾渗（导电路径连通性） | 华数A 问题一类 |
| `mechanism` | 物理机理 / 解析关系 | 华数A 问题类、physics-informed |
| `stochastic-process` | 随机过程 | garch, state-space, mcmc |
| `biomolecular` | 生物分子 / 结构生物学 | alphafold3, proteinmpnn |
| `spatial` | 空间 / 时空 | vecchia, scvi, cell2location |
| `causal` | 因果推断 | did, rdd, causal-forest, dml, tmle |
| `functional` | 函数型数据分析 | fpca, functional-ensemble |
| `time-series` | 时间序列领域标签（配合 forecasting） | arima, garch, var, prophet |
| `financial` | 金融 / 计量 | garch, var, did |

## `domain` 受控词表（应用领域）

> domain 偏"领域定位"，辅助按领域检索；type 才是联合匹配主维度。

`biostatistics` · `structural-biology` · `computational-biology` · `finance-econometrics`
· `operations-research` · `computer-vision` · `nlp-llm` · `signal-processing`
· `energy-systems` · `materials-physics`（新增，承接华数A 导电介质/材料仿真）
· `engineering-mechanics` · `generic-ml`（通用机器学习）

## 复合题型 → 三层谱系标签映射（检索协议依据）

赛题一个小问常是融合型。检索时按**三层谱系各取标签**，对 frontmatter 的 `type` 做**联合匹配**（候选须同时命中若干层的关键标签）。

华数杯 A 题示例：
| 小问 | 决策层 | 计算层 | 物理层 | 前三命中 |
|------|--------|--------|--------|---------|
| 问题一（导通判断） | —（直接判定） | `graph` + `connectivity` | `geometry` + `percolation` | 结果应命中几何/图连通类条目 |
| 问题二（导通概率） | `uncertainty-quantification` | `simulation` + `monte-carlo` | `geometry` + `percolation` | 命中 MC / 随机类 |
| 问题三（最低填充量） | `inverse-problem` + `quantile-reliability` | `simulation` + `monte-carlo` | `geometry` + `percolation` | 命中优化 + MC 复合 |
| 问题四（最低成本配比） | `optimization` + `multi-objective` | `simulation` + `monte-carlo` | `geometry` + `percolation` | 命中优化 + 随机复合 |

## 校验清单（新条目提交前）
1. 第一行 `---`，YAML 含 `title` / `type`（≥1）/ `domain`（≥1）
2. `type` 标签全部来自受控词表（可在 `_index.md` 找到词表速查）
3. `domain` 标签来自受控词表
4. 正文 `# 标题` 与 `title` 一致
5. 标签只多不少：宁可真标签偏多（防漏检），不要漏掉真实维度
