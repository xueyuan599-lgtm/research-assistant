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
| 论文格式与排版 | 模板适配、参考文献格式化、图表规范 | 期刊模板一键排版、LaTeX 编译辅助 |

## 使用方式
```
/research <你的科研需求>    # 启动智能体管线
```

## 架构
```yaml
research-assistant/
├── CLAUDE.md                       # 项目说明
├── .claude/rules/
│   ├── 00-scope-boundary.md        # 作用域沙箱（不污染外层）
│   └── 01-agent-standards.md       # Agent 编写规范
├── workflows/
│   └── dynamic-workflow.md         # 动态管线协议
├── agents/                         # 科研智能体（核心）
│   ├── orchestrator.md             # 总协调人 — 意图识别 + 管线编排
│   ├── literature-agent.md         # 文献检索与综述
│   ├── topic-analysis-agent.md     # 选题分析与前沿探测
│   ├── data-viz-agent.md           # 数据处理与可视化
│   ├── experiment-agent.md         # 实验设计与优化
│   ├── paper-format-agent.md       # 论文格式与排版
│   └── research-qa-agent.md        # 科研知识问答
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
