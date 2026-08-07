# Research Assistant — 科研全流程智能辅助

聚焦科学研究全过程痛点，搭建动态智能辅助工具集。

## 场景覆盖
| 场景 | 说明 | 示例 |
|------|------|------|
| 科研知识问答 | 领域知识查询、方法解释、公式推导 | 因果推断方法对比、DID 模型假设解读 |
| 文献智能检索 | 文献搜索、筛选、综述生成 | 基于 BERTopic 的文献自动综述 |
| 论文选题分析 | 研究前沿识别、选题推荐、创新点判断 | 基于 bibliometrix 的选题热力图 |
| 数据处理与可视化 | 清洗、变换、建模、出版级图表 | 实验数据可视化助手、统计报表自动生成 |
| 实验流程优化 | 方案设计、参数调优、敏感性分析 | 模拟实验自动化 pipeline |
| 算法创造 | 研究想法到新算法：形式化→设计→实现→基准→验证入库 | 设计一个异质性处理效应稳健估计量 |
| 数学建模竞赛 | 选题评估→审题→模型→代码→论文全流程 | "国赛C题农作物种植策略" |
| 论文格式与排版 | 模板适配、参考文献格式化、图表规范 | 期刊模板一键排版、LaTeX 编译辅助 |

## 使用方式
```
/research <你的科研需求>    # 启动智能体管线
```

## ⚠️ 任务启动流程（强制执行）

收到任何非 trivial 任务后，在动手之前：
1. **秘书 Agent**（`agents/secretary.md`）先分析 → 输出分解方案
2. 暂停等待用户确认工具/配色/规模/格式
3. 用户确认后 → **Orchestrator**（`agents/orchestrator.md`）调度 Agent 集群
4. 禁止在阶段 0（秘书分解）完成前写任何代码或探查数据

详见 `agents/secretary.md`。

## 架构
```yaml
research-assistant/
├── AGENTS.md                       # 项目说明
├── .claude/rules/
│   ├── 00-scope-boundary.md        # 作用域沙箱（不污染外层）
│   ├── 01-agent-standards.md       # Agent 编写规范
│   ├── 02-academic-writing-standards.md  # 学术写作质量标准
│   ├── 03-kaggle-track.md          # Kaggle 竞赛流水线
    │   └── 04-mcm-track.md            # 数学建模国赛赛道
├── workflows/
│   └── dynamic-workflow.md         # 动态管线协议
├── agents/                         # 科研智能体（核心，40+ 个定义）
│   ├── secretary.md                # 任务分解守门人 — 所有任务的唯一入口
│   ├── orchestrator.md             # 总协调人 — 意图识别 + 管线编排
│   ├── shared-memory-template.md   # 跨 Agent 共享记忆模板
│   ├── literature/                 # 文献检索与综述（3+1 Agent）
│   ├── topic-analysis/             # 选题分析与前沿探测（3+1 Agent）
│   ├── data-viz/                   # 数据处理与可视化（4+1 Agent）
│   ├── experiment/                 # 实验设计与优化（3+1 Agent）
│   ├── algorithm/                  # 算法创造（5+1 Agent）
│   ├── paper-format/               # 论文格式与排版（3+1 Agent）
│   ├── research-qa/                # 科研知识问答（3+1 Agent）
│   ├── kaggle/                     # Kaggle 竞赛（7+1 Agent）
│   ├── mcm/                        # 数学建模国赛（8 Agent）
│   └── knowledge/                  # 知识检索（1 Agent）
├── scripts/                        # 辅助脚本（Python + MATLAB）
├── schemas/                        # 交接 JSON Schema
├── knowledge/                      # 知识库（70 算法条目）
│   ├── algorithm-repository/       # 顶刊算法实现库
│   ├── kaggle/                     # 竞赛模式库
    │   ├── mcm/                       # 数学建模竞赛知识库
│   ├── project-experience/         # 项目经验沉淀
│   └── _index.md                   # 知识库索引
└── outputs/                        # 所有输出落在此处
```

## 工作原则
- **动态管线**：不预设固定流水线，根据输入意图临时组装智能体
- **沙箱隔离**：所有读写局限在 `research-assistant/` 内，不影响父项目
- **工具复用**：按需调用父项目 `.claude/skills/` 中的专业技能（只读）
- **用户参与**：关键决策点可暂停等待用户确认

## 扩展方式
- 加场景 → `agents/` 下新建 Agent `.md` 文件，orchestrator 注册路由
- 改流程 → 编辑 `workflows/dynamic-workflow.md`
- 加约束 → `rules/` 下新增规则
