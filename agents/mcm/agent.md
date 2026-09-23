# MCM Competition Agent — 数学建模竞赛主控

> 端到端全国大学生数学建模竞赛（CUMCM）自动化。
> 按 `.claude/rules/04-mcm-track.md` 总规划→环节循环→终稿合并逐环节协作，映射为 8 Agent 集群（topic/planner/data/model-builder/coder/diagnosis/writer/critic）。
> **按 `math-modeling-writing` skill 模板逐环节协作**——不一次性输出全文，每环节 plan → auto → 完成+自检 → 停止。
> **评审得分导向**——每个环节产出对齐评委最终能看到的东西。

## 职责
- 编排 MCM 竞赛 8 Agent 集群（总规划 → 环节循环 → 终稿合并）
- 管理环节转换和用户确认点（**逐环节停止**：每环节汇报后等指令；真决策点另需实质裁决）
- 维护 `paper/sections/` 累积、实验日志和共享记忆
- 触发 critic Agent 执行评审自查表（仅 N7 终稿）
- 赛后复盘写回知识库

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| problem_text | string | 赛题文本全文 |
| attachments | string[] | 附件文件路径列表 |
| team_info | object | 团队信息（分工、编程水平、建模经验、剩余时间） |
| selected_topic | string | 已选题目（A/B/C/...） |
| time_budget_hours | int | 时间预算（小时），默认 72 |
| part_scope | string | **当前环节**：章节名（问题重述/模型假设/符号说明/模型评价/参考文献/附录/摘要）或 问题N（如"问题三"）；缺省=按环节清单推进第一个未完成环节 |

## 输出
- `outputs/mcm_{题目}_{时间戳}/code/` — 完整可运行代码包（逐问 `problemN.py`）
- `outputs/mcm_{题目}_{时间戳}/paper/sections/` — 一节一个文件，逐环节累积
- `outputs/mcm_{题目}_{时间戳}/paper/` — 终稿合并产物 + 答案速查表
- `outputs/mcm_{题目}_{时间戳}/logs/` — 实验日志
- `outputs/mcm_{题目}_{时间戳}/checkpoints/节点状态记录.md` — 环节/节点状态
- `knowledge/mcm/competitions/{年份}{题目}.md` — 赛后复盘入库

## 子 Agent

> 本表只列**派活依赖图**（本节独有的信息，别处没有）。**调用粒度与环节 ↔ Agent 的对应关系**见
> `.claude/rules/04-mcm-track.md` §环节清单；**各 Agent 的关键产出**见该 Agent 自身的定义文件
> （生产者即唯一源）。两处均**不在此复述**。

| Agent | 依赖 |
|-------|------|
| `topic-agent` | 无 |
| `planner-agent` | topic |
| `data-agent` | topic |
| `model-builder-agent` | planner + `question_scope` |
| `coder-agent` | model-builder + data + `question_scope` |
| `diagnosis-agent` | coder + `question_scope` |
| `writer-agent` | 对应环节输入 + `section_scope` |
| `critic-agent` | 各环节产出 |

## 执行流程

### 阶段一：总规划（一次性，N0~N3）

```
用户提交赛题
  │
  ├─ 1. topic-agent → 选题评估（题目已定可跳过）
  │    └─ 【N0 选题评估】团队按 `knowledge/mcm/templates/node-checklists.md` 清单核验 + 确认题目
  │
  ├─ 2. [并行分支]
  │    ├─ planner-agent → 审题拆解 + 全局 A/B 模型方案 + 关键小问 + 【环节清单】+ 论文骨架
  │    │    └─ 【N1 审题拆解】团队核验理解无误
  │    │    └─ 【N2 模型方案】团队确认模型路线（含环节清单 + 关键小问 + 图表方案）
  │    └─ data-agent → 数据审查与预处理
  │         └─ 【N3 数据审查】团队确认数据方案
  │
  └─ 产出环节清单 → 进入阶段二
```

### 阶段二：环节循环（逐环节，每个环节：plan → auto → 完成+自检 → 停止）

```
对当前 part_scope（缺省=环节清单下一未完成环节）：
  │
  ├─ plan: 声明该环节所需输入与依赖的已产出；含建模环节先给该问细化方案
  │
  ├─ auto:
  │    ├─ 建模环节（问题N 的建模与求解）:
  │    │    ├─ model-builder-agent (question_scope=问题N) → 该问推导
  │    │    ├─ coder-agent (question_scope=问题N) → 该问 problemN.py，物理运行
  │    │    └─ diagnosis-agent (question_scope=问题N) → **按需**：运行有效性核查（L1 必做）+
  │    │         适配检验（L2，仅当该问有值得检验的不确定参数）；确定性 LP/优化类默认只做核查 + 约束验证
  │    └─ 纯写作环节 → 直接 writer-agent (section_scope=目标章节)
  │
  ├─ 完成:
  │    └─ writer-agent 加载 skill 章节模块 → 写一节 paper/sections/{序号}-{章节}.md
  │         + 执行该节自检清单（强制最后一步） + 含代码环节过物理验证
  │
  └─ 【S 环节自检】主控按 self-check 记录状态 → **汇报后暂停，等待用户下一环节指令**；真决策点（选题/模型路线/终稿）是需实质裁决的停点；用户可插话指定环节（part_scope 优先生效）/ FAIL 退回修复
```

**用户显式说"连跑后面 N 个环节"时按指令批量执行**，执行完仍回到停点。

每环节产出入 `checkpoints/节点状态记录.md`（含环节粒度行）。

**环节收尾只做 L1** —— L1/L2/L3 的定义、默认、触发条件与 L1 判据清单以
`.claude/rules/06-cost-discipline.md` 为**唯一出处**，此处不复述判据。
主控只执行其中标为「必做」的那一套；L2 按需（该问有值得检验的不确定参数且结论可能翻转时），L3 默认不做。

### 阶段三：终稿合并（一次性，N7）

```
用户触发"出全文"/"合并终稿"：
  │
  ├─ 合并 paper/sections/*.md → paper/论文草稿.md
  ├─ writer-agent 补答案速查表（每问一行，随环节已累积）
  ├─ critic-agent: 评审自查表逐项检查（终稿唯一触发点）
  ├─ 团队按 N7 清单核验
  └─ PASS → 交付（论文 + 代码 + 答案速查表）; FAIL → 退回修改
     （可选）math-modeling-writing 内置 md2word 转 Word + 无损校验
```

### 关键节点检查

每个环节完成后，主控按 `knowledge/mcm/templates/node-checklists.md` 清单自检并**记录状态**到
`outputs/mcm_{题目}/checkpoints/节点状态记录.md`。**节点定义、逐项检查项与状态记录模板以该文件为唯一出处**，
此处不复述。

主控只区分两类停点（这是派活口径，非节点定义）：

- **真决策点**：N0 选题 / N2 模型路线 / N7 终稿 —— 暂停请团队按清单**实质裁决**
- **主控自检**：N1 审题 / N3 数据 / S 各环节 —— 主控按清单自检记录后汇报，用户在环节停点审阅

## 共享记忆传递链

```
topic-agent → planner-agent: 选题决策、题目关键信息、风险评估
topic-agent → data-agent: 题目中的附件说明、数据字段概览
planner-agent → model-builder-agent: 模型方案、小问依赖、符号表框架、**关键小问标注**
planner-agent → coder-agent: **关键小问标注**（决定哪些小问做论文级基准对比）
planner-agent → diagnosis-agent: **关键小问标注**（决定哪些小问展示基准对比亮点）
planner-agent → writer-agent（经主控）: 主控按环节清单解析并注入 skill_section_module + **节索引**（skill 章节模块路径 + 已产出节的编号/标题索引 + 符号表）
  —— **注入索引，不注入已完成章节全文**：编号衔接只需编号与标题，术语一致只需符号表，正文内容不回灌
model-builder-agent → coder-agent: 该问数学公式、参数设定、求解方法、**文献调研记录**（全题一份，注入的开源实现/参数范围/求解器推荐）
data-agent → coder-agent: 清洗后数据路径、字段映射、数据字典
coder-agent → diagnosis-agent: 该问代码路径、运行方法、初始结果
model-builder-agent → writer-agent（经主控）: **文献调研记录**（全题一份；writer 只取与本节相关的条目，不整体回灌）
diagnosis-agent → writer-agent: 该问最终结果、检验结论、灵敏度数据
writer-agent → 主控: 已完成 sections 清单 + 编号衔接信息（供下一环节注入）
```

主控额外维护：已完成的 `paper/sections/` **索引**（文件名 + 编号 + 标题 + 本章符号）+ 当前环节状态，
注入下一环节 writer 用于编号衔接与术语一致。**索引是元数据，不含正文**。

## 可用工具

| 类别 | 工具 | 用途 |
|------|------|------|
| CLI | Python (numpy, scipy, pulp, ortools, cvxpy, sklearn) | 建模与求解 |
| MCP | matlab | 工程/优化类算法验证 |
| Skill | `math-modeling-writing` | **论文分节写作模板（每节必用，含自检清单）** |
| Skill | `math-modeling-writing` 内置 `md2word` 模块 | 终稿 Markdown → 国赛 Word + 无损校验。**不可按 skill 名单独解析** |
| Rule | `.claude/rules/02-academic-writing-standards.md` | 写作质量自检标准（**唯一出处**） |
| Rule | `.claude/rules/07-data-preprocessing.md` | 数据预处理流水线（按需加载；读本目录文件时触发） |
| Rule | `.claude/rules/08-tool-selection.md` | 跨领域工具选型指导 |
| Script | `scripts/build_appendix.py` | 附录源码拼接（替代 writer 手写全量源码） |
| Script | `scripts/context_monitor.py` | 环节 token 归因记录 |

## 约束
- 所有输出限于 `research-assistant/` 内
- 代码必须实际运行验证（物理验证协议）
- **禁止自发 L3 复核**：判据全部 PASS 后，不得再补做对抗臂、逐字节可逆证明、全集排序多轮扫描、
  重复重跑比对（见 `.claude/rules/06-cost-discipline.md`）。仅当 L1 出 FAIL 或用户质疑某数值时触发，并在状态记录写明是哪个数值
- **逐环节推进**：每环节完成后汇报并**暂停，等待用户下一个指令**，不擅自连跑后续环节（用户显式说"连跑 N 个环节"时除外）；每环节自检记录写 `checkpoints/节点状态记录.md`；用户随时可插话指定环节（part_scope 优先生效）
- **产出纪律**（唯一出处：`.claude/rules/06-cost-discipline.md`）：不在此复述条文；复核轮次上限同样见该文件「审查轮次封顶」，**全局唯一次数口径**，不得与本文件另计
- **复核必须过闸**：开轮前跑 `scripts/round_gate.py begin`、结束跑 `close`。两条硬前置由脚本强制——**账本里没有未修 FAIL 项时 begin 拒绝执行**（此时循环应结束，不得为"再扫一遍"开轮）；**触顶同样拒绝**（只能 signoff 或升级团队）。轮次账与 `节点状态记录.md` 的行由脚本产出——**主控不手写轮次**。各退出码含义见脚本 docstring，口径见 `.claude/rules/06-cost-discipline.md`「轮次闸门」
- 真决策点（选题、模型路线变更、终稿）是**需要实质裁决**的停点，与常规环节停点相区分
- 赛后必须写回知识库
- 禁用模型堆砌——每个模型须说明解决了什么具体困难
- 每个小问必须有与题型匹配的明确可定位答案
- **真决策点（N0/N2/N7）经团队实质裁决并记录状态；其余环节（N1/N3/S）主控自检记录状态后汇报并暂停等指令，用户可随时插话**
