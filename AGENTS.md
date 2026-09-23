# Research Assistant — 科研全流程智能辅助

聚焦科学研究全过程痛点，搭建动态智能辅助工具集。

<!-- 场景覆盖 / 使用方式 两节仅本文件持有；CLAUDE.md 无对应节（2026-09-20 核实，原「三节逐字一致」声明已失效） -->
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
| 期刊论文结构写作 | 定尺子→标题→摘要→引言→方法→证据映射→讨论→结构审稿 | "投《管理世界》，按它的要求改这篇稿子" |
| 论文格式与排版 | 模板适配、参考文献格式化、图表规范 | 期刊模板一键排版、LaTeX 编译辅助 |

## 使用方式
```
/research <你的科研需求>            # 启动智能体管线（→ secretary → orchestrator）
/implement <研究想法>               # 想法 → 顶刊级实现完整流水线
/preprocess <数据>                  # 数据预处理（表格/时间序列/函数型/面板）
/pathplan <问题>                    # 路径规划诊断-松弛-修复（求解器报 INFEASIBLE 时）
/paper <目标期刊与材料>             # 期刊论文结构写作（定尺子→逐步骤→结构审稿）
/research-prompt-refiner <提示词>   # 科研提示词优化（强制前置检索增强）
```

> 6 个命令定义在 `.claude/commands/`，已随仓库提交（2026-09-20）——见 `## 架构` 末的「入库完整性提醒」。

<!-- SHARED-BLOCK task-tiers v1 — 本节「分级要点」与 CLAUDE.md 同源，实质须一致（措辞可不逐字同）；改动须同批两侧落地 -->
## ⚠️ 任务启动流程（分级执行）

收到任务后先按 **T0~T3 分级**判定编排深度，再决定是否等待确认。
**分级表、编排流程与开工自查的唯一源是 `.claude/rules/06-cost-discipline.md` §零**——
本文件与 `CLAUDE.md` 都只给要点，不另列一份分级表。要点：T0 直接执行；T1 按默认值推进、**无硬停**；
T2 秘书分解 + 一次确认；T3 走专用赛道规则（MCM 赛道为**逐环节停止**，每环节汇报后等指令，
用户可授权连跑 N 个环节，另有真决策点选题/模型路线/终稿需实质裁决）。
用户未在合理时间回复 → 按方案声明的默认值推进并持续汇报，不无限等待。
<!-- /SHARED-BLOCK -->

## 架构

> 下方目录树已按**磁盘实际状态**核对（2026-09-20）。括号内数字为文件/条目计数，改动目录结构时请同步更新。

```yaml
research-assistant/
├── CLAUDE.md                       # 项目配置（Claude Code 自动加载）
├── AGENTS.md                       # 跨工具架构说明（Codex 等读取）
├── README.md                       # 仓库说明与团队交接
├── .claude/
│   ├── rules/          (11)        # 行为规范（唯一规则层；常驻/按需见各文件 frontmatter）
│   │   ├── 00-scope-boundary.md          # 作用域沙箱（不污染外层）
│   │   ├── 01-agent-standards.md         # Agent 编写规范 + 引用与单一源约定
│   │   ├── 02-academic-writing-standards.md  # 学术写作质量标准
│   │   ├── 03-kaggle-track.md            # Kaggle 竞赛流水线
│   │   ├── 04-mcm-track.md               # 数学建模国赛赛道（薄壳）
│   │   ├── 04-mcm-track-detail.md        # MCM 流程细节（按需）
│   │   ├── 05-exploration-budget.md      # 探索型环节预算闸门
│   │   ├── 06-cost-discipline.md         # 任务分级 + 检验分层 + 产出纪律（跨赛道唯一源）
│   │   ├── 07-data-preprocessing.md      # 数据预处理流水线（按需）
│   │   ├── 08-tool-selection.md          # 跨领域工具选型
│   │   └── 09-journal-track.md           # 期刊论文结构写作赛道（薄壳，按需）
│   ├── agents/         (3)         # 子 Agent 工具分档定义（ra-scan / ra-write / ra-build）
│   ├── commands/       (6)         # 斜杠命令：research · implement · preprocess · pathplan · paper · research-prompt-refiner
│   ├── skills/         (17)        # 专业技能（2026-09-20 由父层迁入）
│   ├── settings.template.json      # 配置模板（入库）
│   └── settings.local.json         # 本地配置（.gitignore，不入库）
├── agents/             (58 个 .md) # 科研智能体（核心）
│   ├── secretary.md                # 任务分解守门人 — 所有任务的唯一入口
│   ├── orchestrator.md             # 总协调人 — 意图识别 + 管线编排
│   ├── shared-memory-template.md   # 跨 Agent 共享记忆模板
│   ├── literature/     (4)         # 文献检索与综述：主控 + search / screening / synthesis
│   ├── topic-analysis/ (4)         # 选题分析与前沿探测：主控 + frontier-detection / gap-analysis / recommendation
│   ├── data-viz/       (9)         # 数据处理与可视化：主控 + cleaning / modeling / visualization / interpretation
│   │                               #   + preprocessing-{inspector, cleaner, transformer, validator}
│   ├── experiment/     (4)         # 实验设计与优化：主控 + design / simulation / optimization
│   ├── algorithm/      (6)         # 算法创造：主控 + formalizer / designer / coder / benchmark / validator
│   ├── paper-format/   (4)         # 论文格式与排版：主控 + template / reference / compliance
│   ├── research-qa/    (4)         # 科研知识问答：主控 + method-explanation / formula-derivation / code-demo
│   ├── kaggle/         (8)         # Kaggle 竞赛：主控 + data-explorer / baseline / feature-engineer / model-builder / ensemble / submission / post-mortem
│   ├── mcm/            (9)         # 数学建模国赛：主控 + topic / planner / data / model-builder / coder / diagnosis / writer / critic
│   ├── journal/        (2)         # 期刊论文结构写作：主控 + structure-reviewer（07 结构审稿）
│   └── knowledge/      (1)         # 知识检索
├── scripts/            (12)        # 辅助脚本；核心 = round_gate.py / context_monitor.py / checkpoint.py / build_appendix.py
├── workflows/          (3)         # 工作流协议
│   ├── dynamic-workflow.md         # 动态管线协议
│   ├── codex-claude-collaboration.md  # Codex ↔ Claude Worker 调度协议（执行器缺失，见 secretary.md ⚠️）
│   └── schemas/                    # 交接 JSON Schema
├── knowledge/          (130 个 .md) # 知识库（按需读，不预加载）
│   ├── _index.md                   # 知识库索引
│   ├── algorithm-repository/ (84)  # 顶刊算法实现库
│   ├── algorithms/           (17)  # 新算法条目（index + physarum-network-optimizer 等）
│   ├── kaggle/               (12)  # 竞赛模式库
│   ├── mcm/                  (10)  # 数学建模知识库（含 templates/format2026）
│   ├── writing/               (4)  # 写作判据：正面范式 / 01–06 论证结构 / 期刊尺子 / 07 结构审稿口径
│   ├── project-experience/    (1)  # 项目经验沉淀
│   ├── outputs/                    # PNO 实验 JSON（历史产物，非知识条目）
│   └── optimization-validation-framework.md
├── staging/                        # 临时暂存（未纳入 git；CUMCM 重构中间件，可清）
└── outputs/            (30 个目录)  # 所有输出落在此处
```

**入库完整性提醒**：规则层、技能层与命令层**必须随仓库提交**，否则新克隆拿不到它们，
且所有指向「唯一源」的链接会**静默失效**（无报错、无告警）。

2026-09-20 核对发现以下内容虽被 `.gitignore` 放行却从未 `git add`，已一并补入：
`.claude/rules/` 5 个文件（含 `06-cost-discipline.md` 这一跨赛道唯一源）· `.claude/skills/`(17) ·
`.claude/agents/`(3 档) · `.claude/commands/`(5) · `scripts/round_gate.py` · `scripts/build_appendix.py` ·
`workflows/codex-claude-collaboration.md`。改动本目录结构时请复核本节。

2026-09-21 新增期刊论文结构写作赛道（`.claude/rules/09-journal-track.md` · `agents/journal/`(2) ·
`.claude/commands/paper.md` · `knowledge/writing/` 3 个判据文件），同样须随仓库提交——
**3 个判据文件是 00–07 各步的唯一源，缺任一个则该步判据静默降级为「无标准」**。

## 工作原则
- **动态管线**：不预设固定流水线，根据输入意图临时组装智能体
- **沙箱隔离**：所有读写局限在本仓库内（父目录已不是项目的组成部分，仅剩 `settings*.json` 与一个指针 `CLAUDE.md`）
- **工具复用**：按需调用本仓库 `.claude/skills/` 中的专业技能（17 个，2026-09-20 由父层迁入）
- **用户参与**：仅真决策点（不可逆、影响交付）暂停确认；其余按默认值推进，用户随时可插话。**T3 赛道例外**：MCM 逐环节停止（每环节汇报后等指令，可授权连跑 N 个环节），另有真决策点（选题/模型路线/终稿）需实质裁决，见 `.claude/rules/04-mcm-track.md`

## 扩展方式
- 加场景 → `agents/` 下新建 Agent `.md` 文件，orchestrator 注册路由
- 改流程 → 编辑 `workflows/dynamic-workflow.md`
- 加约束 → `.claude/rules/` 下新增规则
