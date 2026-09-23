# 期刊规则清单（尺子层）

> **本文件是「先定尺子」这一步的唯一源**——期刊论文赛道开工前必须固定的输入、`00-journal-rules.md` 的信息骨架、
> 输出工作区布局，均在此定义。`agents/journal/**` 只写指针，不复制本文件的表头与检查项。
>
> 本文件管**怎么用**；**什么时候用**见 `.claude/rules/09-journal-track.md`（按需加载）。

---

## 1. 固定输入

写作开始前必须固定下列输入，缺任一份不得进入 01。

| 输入 | 落点 | 来源 | 约束 |
|------|------|------|------|
| 目标期刊官方投稿与格式要求 | `source/journal-requirements.md` | 用户提供原件或摘录 | **唯一规则来源** |
| 样本论文原件 | `source/sample-paper.pdf` | 用户提供 | 须为目标期刊**已发表**论文 |
| 样本论文结构笔记 | `source/sample-paper.md` | 由样本论文提取 | 最低内容要求见 §1.1 |
| 研究材料 | `my-paper/materials.md` | 用户提供 | 与结果**分开存放**，不混写 |
| 研究结果 | `my-paper/results.md` | 用户提供 | 数据、图表、回归表、数值 |
| 论文草稿 | `my-paper/draft.md` | 用户提供，或委派逻辑名 `academic-paper` 产出 | 赛道**不写正文** |

**尺子未固定（`source/journal-requirements.md` 缺失）→ 拒绝执行 01。**

### 1.1 样本论文结构笔记的最低内容

结构拆解的口径（结构模式判定、各章字数配比）以用户级 `academic-paper` 的结构范式参考为准（逻辑名引用，不写路径）。
本仓库只规定**笔记的落点与最低内容**：

- 每章标题 + 该章承担的功能（一句话）
- 该章与「问题—证据—结论」链的对应关系
- 篇幅占比（样本可测时记，不可测则留空）

样本只在 `source/sample-paper.md` 留**结构笔记**，不复制原文段落。

---

## 2. 00-journal-rules.md 的七类信息

`00` 是尺子的**提炼产物**，不是原件副本。七类信息缺任一类标「未确认」，**不得留空**。

| # | 类别 | 记录什么 |
|---|------|---------|
| 1 | 期刊定位 | 学科、读者、偏好实证/理论/政策 |
| 2 | 标题要求 | 字数上限、是否允许副标题、是否要求点明方法 |
| 3 | 摘要结构 | 是否结构式、字数、是否要关键词、中英双语要求 |
| 4 | 方法与数据规范 | 数据可得性声明、稳健性要求、识别策略偏好 |
| 5 | 格式要求 | 篇幅、图表数上限、参考文献格式、匿名要求 |
| 6 | 审稿关注点 | 从投稿指南与样本论文的审稿痕迹中提炼 |
| 7 | **未确认项** | 汇总前 6 类中所有未确认条目，供用户补齐 |

第 7 类是本节相对通用投稿指南的核心增量：**把不确定性显式记账**，而非默认已确认。

### 2.1 与格式合规的分工

`00` 记的是**写作侧尺子**（这一篇该怎么组织）。**格式合规执行**委派 `agents/paper-format/agent.md`；
其分级口径见 `agents/paper-format/compliance-agent.md`，**本文件不复制其清单**——两者是不同交付物、不同评审对象。

### 2.2 格式层唯一规则来源

`agents/paper-format/` 的输入 `journal` 与规则来源**指向本工作区 `source/journal-requirements.md`**，
**不得另行联网检索同一期刊**。同一期刊两份规则 = 漂移源。

---

## 3. 输出工作区

### 3.1 命名

```
outputs/journal_{期刊简称}_{时间戳}/
```

例：`outputs/journal_glsj_20260921/`。期刊简称与 `00` 的期刊定位一致。
**前缀 `journal_` 是 `.claude/rules/09-journal-track.md` 的按需加载触发钩子**——命名不统一会让该规则的触发失效。

### 3.2 布局

```
outputs/journal_{期刊简称}_{时间戳}/
├─ README.md              # 仅交接关系表（边界见 §4）
├─ 00-journal-rules.md
├─ 01-title.md
├─ 02-abstract.md
├─ 03-problem-chain.md
├─ 04-method-check.md
├─ 05-evidence-map.md
├─ 06-discussion.md
├─ 07-review.md
├─ source/
│  ├─ journal-requirements.md
│  ├─ sample-paper.pdf
│  └─ sample-paper.md
└─ my-paper/
   ├─ draft.md
   ├─ materials.md
   └─ results.md
```

**文件名逐字固定，不得改名**：换期刊迁移性（§5）依赖文件链逐名不变。

**由 Agent 创建目录，不写生成脚本**——`scripts/round_gate.py` 的状态记录表头已是「本地镜像」并附带手动同步负担；
再引入工作区生成脚本会把布局变成第二个硬编码副本。

---

## 4. README.md 的边界

README **只写交接关系表**：00→01→…→07 的读取顺序，每步读哪些上游文件、写哪个文件。**≤15 行。**

**禁止**写入进度、轮次、状态、待办、时间线。

依据 `.claude/rules/06-cost-discipline.md` §二「禁止独立跟踪文档」——状态由 `01`~`07` 文件的存在性与 git 历史体现。

---

## 5. 换期刊迁移程序

换目标期刊时**只替换三处**，文件链不变：

1. 用新期刊官方要求覆盖 `source/journal-requirements.md`
2. 用新样本论文覆盖 `source/sample-paper.{pdf,md}`
3. 重跑 `00`，产出新的 `00-journal-rules.md`

`01`~`07` 的名称与顺序**不变**；已产出的步骤按新尺子**复核**，不重建工作区。
`my-paper/` 是作者材料，跨期刊**保持不变**。

---

## 6. 本文件不定义什么

| 内容 | 归属 |
|------|------|
| 01–06 的判据 | `knowledge/writing/paper-argument-structure.md` |
| 07 的审稿口径与轮次 | `knowledge/writing/structure-review-protocol.md` |
| 句层/词层标准 | `.claude/rules/02-academic-writing-standards.md` |
| 格式合规清单 | `agents/paper-format/compliance-agent.md` |
| 赛道触发与硬约束 | `.claude/rules/09-journal-track.md` |
