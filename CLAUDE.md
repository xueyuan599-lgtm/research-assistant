# Research Assistant — 科研全流程智能辅助

> 聚焦科学研究全过程痛点，搭建动态智能辅助工具集。

## 任务启动流程（分级执行）

收到任务后先按 **T0~T3 分级**判定编排深度，再决定是否等待确认。

**分级表、编排流程、禁止行为与开工自查的唯一源是 `.claude/rules/06-cost-discipline.md` §零**——
本文件不另列一份，避免两级规则漂移。一句话要点：T0 直接执行；T1 按默认值推进、**无硬停**；
T2 秘书分解 + 一次确认；T3 走专用赛道规则。用户未在合理时间回复 → 按方案默认值推进并持续汇报。

## 研究实现流水线（T1/T2）

T3 竞赛/论文走专用赛道（`04-mcm-track.md` / `03-kaggle-track.md` / `09-journal-track.md`），**不走本流水线**。
T2 须先经 `agents/secretary.md` 分解确认（§零 阶段 0）；T1 直接进入。下表路径为**仓库根相对**。

| 阶段 | 做什么 | 主责 Agent |
|------|--------|-----------|
| 1 解析与对标 | 确认核心方法；检索该方法在顶刊中的标准实现规范并汇总 | `agents/literature/agent.md` |
| 2 方案设计 | 模型设定 / 识别策略 / 估计方法 / 假设条件 + **模型选型报告** | `agents/algorithm/designer-agent.md` |
| 3 实现 | 顶刊级代码 | `agents/algorithm/coder-agent.md` |
| 4 实验与验证 | 运行完整实验，再做物理验证 | `agents/experiment/` 系列 + `agents/algorithm/validator-agent.md` |
| 5 对抗式 QA | 独立 critic 审查 → 不通过则 fixer 修复 → 回到 critic | `agents/algorithm/validator-agent.md` |
| 6 交付 | 代码 + 结果图表 + 方法描述 + 复现说明 | — |

选型报告的四项必备内容以 `agents/algorithm/designer-agent.md` 为唯一源；默认按推荐型号推进，
仅当用户明确要求确认、或选型涉及不可逆高成本时暂停。阶段 5 的轮次封顶与触顶升级出口见
`06-cost-discipline.md` §二（唯一源）。

## 能力范围

`agents/` 下的智能体覆盖：科研知识问答 · 文献检索与综述 · 选题与前沿分析 · 数据处理与可视化 ·
实验设计与优化 · 算法创造 · 数学建模竞赛 · Kaggle · 期刊论文结构写作 · 论文格式与排版。

**路由以 `agents/secretary.md`（领域识别表）与 `agents/orchestrator.md`（路由表）为唯一源**——
本文件不另列一份，避免两份路由表漂移。

## 架构

完整目录树与 Agent 清单见 `AGENTS.md`。要点：

| 目录 | 用途 |
|------|------|
| `agents/` | 智能体定义（唯一入口 `secretary.md`） |
| `.claude/rules/` | 行为规范；**部分文件按需加载**（见各文件 frontmatter） |
| `knowledge/` | 知识库；**按需读，不预加载** |
| `scripts/` | 辅助脚本（`round_gate.py` / `context_monitor.py` / `build_appendix.py`） |
| `workflows/` | 工作流协议 + 交接 JSON Schema |
| `outputs/` | 所有输出落此处 |

## 工作原则

- **动态管线**：不预设固定流水线，按输入意图临时组装智能体
- **沙箱隔离**：所有读写局限在本仓库内（`.claude/rules/00-scope-boundary.md`）
- **工具复用**：按需调用本仓库 `.claude/skills/` 中的专业技能（17 个）
- **用户参与**：仅真决策点暂停确认；其余按默认值推进，用户随时可插话
- **成本纪律**：任务分级与编排（T0~T3）· 检验分层（L1/L2/L3）· 产出纪律与审查轮次封顶 ·
  子 Agent 工具分档——四者的唯一源均为 `.claude/rules/06-cost-discipline.md`

## 扩展方式

- 加场景 → `agents/` 下新建 Agent `.md`，orchestrator 注册路由（格式规范见 `.claude/rules/01-agent-standards.md`）
- 改流程 → 编辑 `workflows/dynamic-workflow.md`
- 加约束 → `.claude/rules/` 下新增规则
