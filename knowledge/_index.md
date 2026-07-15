# Knowledge Base

## 加载策略（关键！防止上下文膨胀）

| 位置 | 加载方式 | 适合放什么 | 最大建议 |
|------|---------|-----------|---------|
| `.claude/rules/` | **每次对话自动加载** | 触发条件 + 简短指针 | 每个文件 < 2KB |
| `CLAUDE.md` | **每次对话自动加载** | 项目定位 + 架构概述 | < 5KB |
| `knowledge/_index.md` | **每次对话自动加载** | 标题 + 一行描述 | < 2KB |
| `knowledge/*.md` | **仅在触发/搜索时加载** | 完整方法论、详细经验 | 无限制 |

**核心原则：rules 是"什么时候用什么"，knowledge 是"怎么用"。禁止把方法论细节写入 rules。**

## knowledge/ 目录结构

| 目录/文件 | 说明 | 规模 |
|-----------|------|------|
| `algorithm-repository/` | **SCI 级算法实现库** — 从顶刊提取的方法 + 可运行代码 | **59 个条目**（21 经典 + 38 个 2021-2026 前沿） |
| `optimization-validation-framework.md` | 优化问题求解→验证→交付全流程方法论 | 1 个文件 |
| `project-experience/` | 项目经验沉淀模板（待填充） | 框架就绪 |
| `kaggle/` | Kaggle 竞赛复盘 + 模式模板 | 8 个模式 + 1 个 agent |

> **`algorithms/`**（位于本目录同级）是由 Algorithm Pipeline 自创算法的存放位置，与 `algorithm-repository/`（从顶刊提取）互补。当前为空，等待 Pipeline 首次创建。

### 2021-2026 顶刊前沿方法（`algorithm-repository/` 新增 38 项）

| 领域 | 条目数 | 覆盖期刊 |
|------|--------|---------|
| 运筹优化 | 8 | Operations Research, INFORMS J. Comput., Math. Programming, EJOR, JOTA, SIAM J. Optim. |
| 机器学习 | 7 | NeurIPS, ICML, ICLR, CVPR, ICCV, JMLR, TPAMI |
| 统计学 | 7 | JRSS-B, Ann. Statist., Biometrika, JASA, Statist. Sci. |
| 因果推断 | 8 | J. Causal Inference, Econometrica, JRSS-B, Biometrika, NeurIPS, ICML |
| 生信分析 | 8 | Nature, Nature Methods/Biotech., Science, Genome Biology, PLoS Comp. Bio. |

> 每条含：数学设定（LaTeX）+ 假设表格 + 实现要点 + 可运行 Python 代码 + APA 7th 引用。
> 全部用英文撰写，便于 Agent 直接读取处理。

**经典方法（21 项）**：ARIMA, XGBoost, Random Forest, LSTM, Transformer, DID, RDD, SVM, PCA/t-SNE/UMAP, Lasso/Ridge/ElasticNet, 等 — 见 `algorithm-repository/_index.md` 完整列表。

## 使用方式
- 通过 `/research` 指令由 orchestrator 自动调用相关 agent 查询
- 知识库 agent 提供语义检索和推荐
- 手动查看直接浏览对应 markdown 文件
