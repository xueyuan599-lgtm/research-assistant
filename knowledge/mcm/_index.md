# MCM 知识库索引

> 可生长的数学建模竞赛知识库。Agent 通过查询此库来匹配建模范式和算法方案，赛后自动沉淀新模式。

## 知识库结构

```
knowledge/mcm/
├── _index.md                    # 本文件 — 全局索引 + 查询接口 + 建模范式速查
├── patterns/                    # 已验证的竞赛模式（核心）
│   ├── _template.md             # 模式模板
│   └── (按题型积累)
├── competitions/                # 历史赛题记录
│   ├── _template.md             # 赛题复盘模板
│   └── (按年份积累)
├── templates/                   # 论文/代码模板
│   ├── paper-structure.md       # CUMCM 论文结构模板
│   ├── node-checklists.md       # 关键节点/环节自检清单
│   ├── code-structure.md        # 代码目录结构模板
│   ├── format2026.txt           # 《论文格式规范》(2026) 原文（篇幅口径的依据）
│   └── figures/                 # Nature 级图表模板库（含 README 选型表）
└── agent.md                     # 查询入口说明（stub，职责见 agents/knowledge/agent.md）
```

## 查询接口（Agent 调用方式）

> **`problem_profile` 的参数定义以本表为唯一出处**；查询由 `agents/knowledge/agent.md` 执行。

| 参数 | 类型 | 说明 |
|------|------|------|
| problem_type | string | 题型：`optimization` / `prediction` / `evaluation` / `mechanism` / `simulation` |
| sub_questions | int | 小问数量，通常 3-5 |
| data_scale | string | 数据规模：`small` / `medium` / `large` |
| data_types | string[] | 数据类型列表，如 tabular / time_series / spatial |
| modeling_domains | string[] | 涉及的建模领域，如 linear_programming / differential_equations / statistics |

```
输入: problem_profile（字段见上表）
→ 查 knowledge/mcm/patterns/ 中匹配的模式
→ 返回: 推荐建模范式 + 常见方法 + 已知陷阱 + 成功案例
```

## 模式匹配规则

| 匹配维度 | 权重 | 说明 |
|---------|------|------|
| problem_type 精确匹配 | 必须 | optimization/prediction/evaluation/mechanism/simulation |
| 题型结构相似度 | 高 | 小问数量 + 依赖关系 |
| 数据特征匹配 | 中 | 数据规模 + 类型组合 |
| 建模领域匹配 | 高 | LP/NLP/DE/统计/ML/仿真 |

## 生长机制

1. **赛后写入**：由 MCM 主控协调 `agents/knowledge/agent.md` 将成功/失败模式写入 patterns/ 并复盘至 competitions/（本库自身只读，见 `agent.md` stub）
2. **手动注入**：用户可导入历年赛题优秀论文提炼的模式
3. **模式泛化**：同类题型出现 2 次以上 → 抽象为通用 pattern
4. **过期标记**：竞赛范式变化后，旧 pattern 标记 `deprecated` 而非删除

## 建模范式速查

| 题型 | 典型方法 | 参见模式 |
|------|---------|---------|
| 运筹优化（LP/IP/NLP） | pulp, ortools, scipy.optimize, cvxpy, 遗传算法 | patterns/ |
| 微分方程建模 | scipy.integrate, MATLAB ode45, 差分方程 | patterns/ |
| 统计/预测 | ARIMA, Prophet, LSTM, 灰色预测, 回归分析 | patterns/ |
| 数据预处理（通用前置） | MICE 多重插补, SAITS, IsolationForest, Box-Cox/Yeo-Johnson, 目标编码, 防泄漏切分 | patterns/data-preprocessing.md |
| 评价/决策 | 层次分析法(AHP), 熵权法, TOPSIS, 模糊综合评价 | patterns/ |
| 分类/聚类 | SVM, K-means, 决策树, 朴素贝叶斯 | patterns/ |
| 仿真/随机模拟 | Monte Carlo, 元胞自动机, 多智能体 | patterns/ |
| 图论/网络 | Dijkstra, Floyd, 最小生成树, 最大流 | patterns/ |
| 机理分析 | 量纲分析, 平衡方程, 能量守恒, 物理建模 | patterns/ |
