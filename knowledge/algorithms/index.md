# 自创算法库 — Pipeline 产出

> **与 `algorithm-repository/` 的区别**：
> - `algorithm-repository/` — 从顶刊提取的**已有方法**实现（阅读/参考用）
> - `algorithms/` — 经由 **Algorithm Design Pipeline** 自创的**新算法**（创意产出）
>
> 当前收录 **2 个自创算法**（2026-09-23 更新）。

## 索引

| 算法名 | 领域 | 创建日期 | 状态 | 文件 |
|--------|------|---------|------|------|
| Physarum Network Optimizer (PNO-GWO) | 群智能优化 | 2026-06-10 → 2026-08-28 | v4.1（对话中） | [README](physarum-network-optimizer/README.md) |
| 残差校正混合函数型集成 (AHFE) | 函数型数据分析 / 集成学习 | 2026-08（研究包） | 完成 | [README](functional-ensemble-regression/README.md) |

### 归档说明（2026-09-23 整理）

两个条目此前均**误置于 `algorithm-repository/`**（"顶刊提取"库）——它们的来源行都标注自创：

- **AHFE**：`git mv` 迁入本目录；原 `algorithm-repository/` 位置只保留反向索引链接（保证
  `functional` / `ensemble-learning` / `regression` 三个标签仍可检索）。
- **PNO**：`algorithm-repository/` 里那份 135 行**内容重复**的引用文档已删除——它复制了数学设定、
  基准表与参数指南，并已实际漂移出错误（把 Rastrigin 退化标成"持平"）。标签（`heuristic` /
  `optimization` / `graph`）随算法本体迁入本目录 README 的 frontmatter。
- **实现代码**：AHFE 的实现此前只存在于 `outputs/functional_ensemble_regression/`，而该路径在
  `.gitignore` 内（2026-09-23 上传 GitHub 时排除），即**自创算法的代码完全未入版本库**。现已归档到
  本目录（见下），`outputs/` 那份保留为本地工作包。

## 各条目实现清单

**PNO-GWO** — [目录](physarum-network-optimizer/)：`algorithm.py`（v3.0 核心）· `algorithm_v2.py`（v2.0 冻结版）
· `algorithm_v4.py` · `run_benchmark.py` · `test_algorithm.py`（13 项）· `benchmark.md` / `benchmark_v3.md`
· `v4-test-plan/`（含 9 份 phase 报告、消融/CEC2017/可扩展性分析脚本、对比算法实现与图表）

**AHFE** — [目录](functional-ensemble-regression/)：`src/`（10 模块：`ahfe.py` 装配与门控 · `simpls.py` ·
`robust_simpls.py` · `functional_features.py` · `gating.py` · `methods.py` · `benchmark.py` · `visualize.py` ·
`shap_analysis.py` · `data_loader.py`）· `tests/`（4 文件）· `benchmark_report.md`（P5 确证基准）·
`model-comparison.md` · `refs_notes.md` · `plan.md` · `outputs/`（`benchmark_results.json` + 3 张图）·
`_p4a_*.py`（7 个门控/oracle 验证探针）

> **单源声明**：本目录为**权威副本**（已入版本库）。`outputs/functional_ensemble_regression/` 为本地工作包
> （未跟踪，且 `ahfe_report.pdf` / `checkpoints/` / `catboost_info/` 等重产物只留本地）。两者若冲突以本目录为准。

## 条目格式

每个算法条目遵循标准结构（允许按算法实际形态扩展，如 PNO 的 `v4-test-plan/`、AHFE 的 `src/`+`tests/`）：

```
algorithms/{algorithm-name}/
├── README.md              # 算法概述（数学定义 + 适用场景）
├── algorithm.py           # 核心实现
├── test_algorithm.py      # 单元测试
├── demo.py                # 使用示例
├── benchmark.md           # 基准对比报告
└── figures/               # 对比图表
```
