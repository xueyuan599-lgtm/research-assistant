---
name: math-modeling-writing
description: 按国赛数学建模竞赛中文论文风格分章节撰写论文。读取题目文本、Python 代码与运行结果（日志/CSV/JSON/图表），按《国赛写作指南》生成 Markdown 章节草稿（摘要、问题重述、问题分析、模型假设、符号说明、模型的建立与求解、模型评价、参考文献、附录）。当用户需要把建模代码和实验结果改写成论文章节、按赛题结构写指定章节、或需要按优秀论文风格审查/润色章节时使用。
---

# 数学建模写作

## 输入约定

- 题目文本（或题目文件路径）
- Python 代码目录（可选；模型建立与求解、附录需要）
- 运行结果（`.txt/.log/.csv/.json/.png`，结果类章节需要）
- 已有草稿/大纲（可选；用于衔接章节编号）
- 目标章节列表（单节模式只写指定节；整篇模式逐节生成后合并为 `output/论文草稿.md`）

## 章节路由表

用户要哪一节，就读 `references/sections/` 下对应文件，输出到 `output/sections/序号-章节名.md`。

| 章节 | 参考文件 | 状态 |
|---|---|---|
| 摘要 | `sections/abstract.md` | 已实现（含强制反 AI 自检） |
| 问题重述 | `sections/restatement.md` | 已实现（含自检） |
| 问题分析 | `sections/analysis.md` | 已实现（嵌入模型节，C023 全整合） |
| 模型假设 | `sections/assumptions.md` | 已实现（含自检） |
| 符号说明 | `sections/notation.md` | 已实现（含自检） |
| 模型的建立与求解 | `sections/model-solve.md` | 已实现（含自检，逐问一级标题模板） |
| 模型评价 | `sections/evaluation.md` | 已实现（含自检，仅模型的优点/模型的缺点） |
| 参考文献 | `sections/references.md` | 已实现（含自检） |
| 附录 | `sections/appendix.md` | 已实现（含自检） |

结论：不设独立章节，各问结论已包含在模型建立与求解的每问小结中，文末模型评价后直接进入参考文献。

## 硬性规则

1. 只引用输入中真实存在的数值，禁止编造；摘要与结果章节中的每个数字必须与运行结果文件核对。
2. 图表/公式用编号占位（图 1、表 1、式 (1)），并衔接已有论文结构。
3. 输出 Markdown；每节一个独立文件；整篇模式最后合并。
4. 生成每节后必须执行该节参考文件中的自检清单；摘要强制以“反 AI 自检”为最后一步，未通过则重写。
5. 正文不使用非必要小括号；缩写括号仅在术语首次出现时使用一次。
6. 图表引用：每个图/表在正文中至少被引用一次，引用先于图/表出现（如图X所示、见表X），引用后必须解读；图注在下方、表注在上方。详见 `references/typography.md`。

## 工作流

1. 收集输入，识别题目有几个问，以及每问的模型、方法与核心结果。
2. 按目标章节读取对应 `references/sections/*.md`。
3. 生成草稿 → 对照清单自检 → 写出输出文件。

## 参考文件

- `references/guoshi-writing-guide.md` — 国赛写作指南原文（所有章节的规范底稿）
- `references/sections/abstract.md` — 摘要章节模块（功能、输入依赖、输出骨架、写作规则、常见句式、反 AI 自检）
- `references/sections/restatement.md` — 问题重述章节模块（功能、输入依赖、输出骨架、写作规则、常见句式、自检清单）
- `references/sections/analysis.md` — 问题分析模块（嵌入模式：每个问题一级标题开头的 1~2 段分析）
- `references/sections/assumptions.md` — 模型假设章节模块（功能、输入依赖、输出骨架、写作规则、常见句式、自检清单）
- `references/sections/notation.md` — 符号说明章节模块（功能、输入依赖、输出骨架、符号规范速查、常见句式、自检清单）
- `references/sections/model-solve.md` — 模型建立与求解模块（逐问一级标题模板：分析→图→预处理→建模→求解→结果）
- `references/sections/evaluation.md` — 模型评价章节模块（仅模型的优点/模型的缺点）
- `references/sections/references.md` — 参考文献章节模块（GB/T 7714-2015 顺序编码制）
- `references/sections/appendix.md` — 附录章节模块（核心代码、运行结果、补充推导与图表）

## 阶段二：Markdown → Word（md2word）

产出 `output/论文草稿.md` 后，如需转 Word（WPS 可打开），使用本技能内置的 `phase2-md2word` 模块：

```bash
python phase2-md2word/scripts/md2word.py output/论文草稿.md --out 论文草稿.docx
python phase2-md2word/scripts/verify.py output/论文草稿.md 论文草稿.docx
python phase2-md2word/scripts/qa_render_vision.py 论文草稿.docx
powershell -ExecutionPolicy Bypass -File phase2-md2word/scripts/wps_check.ps1 论文草稿.docx
```

版面规则与 `references/typography.md` 一致：全文西文与数字统一 Times New Roman（附录代码英文与数字用 Consolas）；表格基于无样式、宋体五号（内容过宽自动降小五号）、段前 0.5 行、段后 0、行距 1.5 倍；公式居中、无首行缩进、编号右对齐。依赖与换机安装说明见 `phase2-md2word/SKILL.md`（本机已就绪，无需安装）；测试语料在 `phase2-md2word/tests/`。
