---
paths:
  - "agents/journal/**/*"
  - "outputs/journal_*/**/*"
---

# Journal Track — 期刊论文结构写作赛道（薄壳）

> 当用户提出期刊论文写作 / 投稿任务时触发。**审稿人导向**——每步产出对齐审稿人真正会查的东西。
>
> **本文件只承载路由与硬约束。** 判据与口径各有唯一源，本文件**不复述**：
>
> | 内容 | 唯一源 |
> |------|--------|
> | 01–06 判据与跨赛道接入映射 | `knowledge/writing/paper-argument-structure.md` |
> | 尺子层、00 七类信息、工作区布局、换期刊迁移 | `knowledge/writing/journal-rules-checklist.md` |
> | 07 三判据、分级、轮次口径 | `knowledge/writing/structure-review-protocol.md` |
> | 句层 / 词层标准 | `.claude/rules/02-academic-writing-standards.md` |
> | 格式合规清单与分级 | `agents/paper-format/compliance-agent.md` |
> | 轮次上限的全局口径 | `.claude/rules/06-cost-discipline.md` §二 |
>
> **按需加载**：本文件只在**读取 `agents/journal/**` 或期刊工作区时**进入上下文。
> 期刊任务必然先读 `agents/journal/agent.md`，故必然触发；非期刊任务不承担这份常驻成本。
> 触发条件与领域路由见 `agents/secretary.md` 与 `agents/orchestrator.md`（常驻）。

## 触发条件

| 触发信号 | 示例 |
|---------|------|
| 关键词 "写论文" / "期刊论文" | "帮我把这份研究写成期刊论文" |
| 关键词 "目标期刊" / "投稿期刊" | "投《管理世界》，按它的要求改" |
| 关键词 "结构审稿" / "审稿意见" | "按审稿人视角审一遍这篇稿子" |
| 明确期刊名 + 稿件 | "这篇投 JFE 还差什么" |

**与 PAPER_FORMAT 的消歧**：需**结构 / 论证 / 标题 / 摘要 / 引言 / 讨论** → 本赛道；
需**排版 / 模板 / 参考文献格式 / 合规** → `paper-format`（`agents/secretary.md` 领域识别表两行并存，不合并）。

## 协作模式（强制）

```
固定尺子（一次性：三份输入 → 00）
  → 步骤循环（逐步骤：01 → … → 06，一步一文件，每步产出后停止等指令）
  → 结构审稿（一次性：07，委派 structure-reviewer-agent）
  → 格式层（一次性：委派 agents/paper-format/agent.md）
```

- **一步一文件**：`00`~`07` 各一个独立文件，文件名逐字固定（见尺子层唯一源 §3）
- **逐步骤停止**：每步完成汇报后**暂停等指令**，不擅自连跑；用户显式说"连跑 N 步"时除外
- **尺子未固定不得动笔**：无 `00-journal-rules.md` → 拒绝执行 01
- **本赛道不写正文**：`my-paper/draft.md` 由用户提供，或委派用户级 `academic-paper`（逻辑名）产出；
  主控**不自行起草全文**，否则会与外部技能形成两份草稿

## 真决策点（需实质裁决，区别于常规步骤停点）

| 节点 | 内容 |
|------|------|
| **J0** | 尺子确认（`00` 产出后：七类信息与未确认项是否需补齐） |
| **J7** | 终稿确认（`07` 审稿清单的致命项如何处理） |

常规步骤停点用户回一句"继续"即可。

## 硬约束

| 约束 | 内容 |
|------|------|
| **判据不复制** | 01–06 判据、00 七类信息、07 三判据一律只写指针；本文件与各 Agent 文件不得列检查项 |
| **外部技能只写逻辑名** | `academic-paper` / `academic-paper-reviewer` / `academic-pipeline` / `aigc-reduce` 均为用户级技能，按 `.claude/rules/01-agent-standards.md` §1 **不写路径** |
| **格式层唯一规则来源** | `agents/paper-format/` 的输入指向本工作区 `source/journal-requirements.md`，**不得另行联网检索同一期刊** |
| **轮次** | 唯一源 `.claude/rules/06-cost-discipline.md` §二；机械闸门 `scripts/round_gate.py`（`S7` 上限 1 轮，FAIL 才按判据分轮） |
| **产出纪律** | README 仅交接关系表 ≤15 行；禁止全文快照、独立跟踪文档（唯一源 `.claude/rules/06-cost-discipline.md` §二） |
| **作用域** | 所有输出限于本仓库内（`.claude/rules/00-scope-boundary.md`） |
| **写作** | 遵守 `.claude/rules/02-academic-writing-standards.md` 禁用词表与 §2 自检协议 |

## 输出目录

```
outputs/journal_{期刊简称}_{时间戳}/
```

结构与命名唯一源见 `knowledge/writing/journal-rules-checklist.md` §3。
**前缀 `journal_` 是本文件 `paths:` 的触发钩子**——命名不统一会让按需加载失效。由 Agent 创建目录，不写生成脚本。

## 可用工具

- **Python**（文档处理、图表核对）
- **officecli MCP**（终稿 DOCX 渲染，一次渲染 + 一次校验，见 `.claude/rules/06-cost-discipline.md` §二）
- **`agents/paper-format/`**（格式与合规，输入指向工作区 `source/`）
- **用户级技能（逻辑名引用）**：`academic-paper`（草稿产出、结构范式、双语摘要）·
  `academic-paper-reviewer`（审稿攻击角度）· `academic-pipeline`（复现审计）· `aigc-reduce`（中文 AI 痕迹判定）

## 与 MCM / Kaggle 赛道的边界

| 维度 | MCM 赛道 | 本赛道 |
|------|---------|--------|
| 读者 | 评委，5–10 分钟/篇 | 审稿人，逐主张查证 |
| 结构 | 无引言章、无讨论章、结论不设独立章节 | 引言 / 讨论均为一等章节 |
| 尺子 | 固定的官方格式规范 | **可换**：只替 `source/` 与 `00` |
| 答案形态 | 每问 30 秒可定位 | 每条结论可回溯到证据与其边界 |

**边界声明是单向的**——本文件不复述 MCM / Kaggle 规则，也不向它们的文件写入本赛道口径。
`knowledge/writing/paper-argument-structure.md` §7 给出 MCM 逐条适用性映射，接入方式是单向指针。
