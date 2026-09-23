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
| `algorithm-repository/` | **SCI 级算法实现库** — 从顶刊提取的方法 + 可运行代码 | **82 个条目**（26 经典 + 53 个 2021-2026 前沿 + 3 赛题适配） |
| `optimization-validation-framework.md` | 优化问题求解→验证→交付全流程方法论 | 1 个文件 |
| `project-experience/` | 项目经验沉淀模板（待填充） | 框架就绪 |
| `kaggle/` | Kaggle 竞赛复盘 + 模式模板 | 8 个模式 + 1 个 agent |
| `algorithms/` | Algorithm Pipeline 自创算法（与顶刊提取的 `algorithm-repository/` 互补） | 17 个条目 |
| `mcm/` | 数学建模知识库（含 `templates/` 官方格式规范与节点清单） | 10 个文件 |
| `writing/` | 写作判据与范式：正面范式 · 01–06 论证结构 · 期刊尺子 · 07 结构审稿口径 | 4 个文件 |
| `outputs/` | PNO 实验 JSON（历史产物，非知识条目） | — |

### 2021-2026 顶刊前沿方法（`algorithm-repository/` 新增 53 项）

| 领域 | 条目数 | 覆盖期刊 |
|------|--------|---------|
| 运筹优化 | 8 | Operations Research, INFORMS J. Comput., Math. Programming, EJOR, JOTA, SIAM J. Optim. |
| 机器学习 | 8 | NeurIPS, ICML, ICLR, CVPR, ICCV, JMLR, TPAMI |
| 时序融合 | 14 | Scientific Reports, PLOS ONE, IEEE Access, Eng. Struct., NeurIPS, ICLR, AAAI, China Communications |
| 统计学 | 7 | JRSS-B, Ann. Statist., Biometrika, JASA, Statist. Sci. |
| 因果推断 | 8 | J. Causal Inference, Econometrica, JRSS-B, Biometrika, NeurIPS, ICML |
| 生信分析 | 8 | Nature, Nature Methods/Biotech., Science, Genome Biology, PLoS Comp. Bio. |

> 每条含：数学设定（LaTeX）+ 假设表格 + 实现要点 + 可运行 Python 代码 + 参考文献。
> 以中文撰写为主（英文术语保留原文），便于团队直接阅读；`algorithm-repository/_SCHEMA.md` 为 frontmatter 唯一权威规范。

**经典方法（26 项）**：ARIMA, XGBoost, Random Forest, LSTM, Transformer, State Space/Kalman, DID, RDD, SVM, PCA/t-SNE/UMAP, Lasso/Ridge/ElasticNet, 等 — 见 `algorithm-repository/_index.md` 完整列表。

## 使用方式
- 通过 `/research` 指令由 orchestrator 自动调用相关 agent 查询
- 知识库 agent 提供语义检索和推荐
- 手动查看直接浏览对应 markdown 文件
