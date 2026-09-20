# 基准资源库（Prompt 相关论文与项目）

> 用途：① 实时检索不可用时的兜底；② 构造检索词的种子；③ 命中主题时快速定位并延展检索。
> 验证：所有条目经 arXiv API / GitHub API 实时核验，验证日期 **2026-08-19**。引用以实时检索为准，若出现更新版本以新版为准。

## 一、系统性综述与编写原则（优化稿结构的依据）

| 成果 | 出处 | 要点 |
|---|---|---|
| The Prompt Report: A Systematic Survey of Prompt Engineering Techniques | arXiv 2406.06608（2024） | 58 种提示词技术的系统综述，覆盖上下文工程、零/少样本、推理方法；优化时对照其技术矩阵选择合适的注入技术 |
| Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4 | arXiv 2312.16171（2023） | 26 条提示词编写原则：少用套话、给出示例、用"不/不要"写否定约束、指令分步、明确输出格式等 |

## 二、推理增强技术（注入求解/推理类提示词）

| 成果 | 出处 | 要点 |
|---|---|---|
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | arXiv 2201.11903（2022, Wei 等） | 让模型"先推导后给结果"，显著提升算术/符号推理准确率 → 在提示词中显式要求推导步骤 |
| Self-Consistency Improves Chain of Thought Reasoning in Language Models | arXiv 2203.11171（2022, Wang 等） | 多次采样取多数票/均值 → 数值类任务把采样数 N 写入验收标准 |
| ReAct: Synergizing Reasoning and Acting in Language Models | arXiv 2210.03629（2022, Yao 等） | 推理+工具调用交替 → 需要代码执行/查表/调用求解器的建模任务采用"推理→执行→看结果→再推理"循环 |
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models | arXiv 2305.10601（2023, Yao 等） | 树状搜索多条推理路径 → 多步优化、方案比选类问题提示词引入分支探索+回溯 |
| Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding | arXiv 2401.12954（2024, Suzgun & Kalai） | 主提示词调度多个专门子代理 → 一题多问/模块化任务分解为子任务，各配专门指令 |

## 三、科研/数学推理专项（数学建模、科学研究场景）

| 成果 | 出处 | 要点 |
|---|---|---|
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | arXiv 2402.03300（2024, deepseek 团队） | 数据合成 + 工具集成（代码解释器）提升数学推理 → 提示词要求"用 Python 验证中间结果/结果" |
| DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning | arXiv 2511.22570（2025） | 自验证式数学推理 → 提示词增加"自查步骤"：对最终答案做独立验证 |
| Minerva: Solving Quantitative Reasoning Problems with Language Models | arXiv 2206.14858（2022, Google） | 大规模预训练 + 逐步推理 → 数值题提示词要求"单位换算显式化、中间量命名" |
| ChemCrow: Augmenting Large-Language Models with Chemistry Tools | arXiv 2304.05376（2023） | LLM + 工具循环完成科研实验 → 科研任务提示词显式列出可用工具箱并约束调用边界 |
| The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery | arXiv 2408.06292（2024, Sakana AI） | 全自动科研闭环（idea→实验→论文） → 科研提示词借鉴其"模块拆分 + 实验日志 + 自我审查"结构 |
| SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering | arXiv 2405.15793（2024, Princeton） | 代码 Agent 接口设计 → 代码类提示词给"运行→看报错→修→再运行"的自愈循环及其轮次上限 |

## 四、GitHub 开源项目（实践基线，星标为 2026-08-19 实时值）

| 仓库 | 星标 | 用途 |
|---|---|---|
| dair-ai/Prompt-Engineering-Guide | ★77.6k | Prompt engineering 指南/教程/论文/资源大全，优化时查阅其技术表与提示词示例 |
| langchain-ai/langchain | ★144.5k | Agent 工程平台，工具编排/多步骤任务参考其链式调用模式 |
| anthropics/prompt-eng-interactive-tutorial | ★37.7k | Anthropic 官方交互式教程，适用于 Claude 系目标模型 |
| microsoft/autogen | ★60.5k | Agent 编程框架，多 agent 协作/对话模式参考（一题多问、分工场景） |
| SWE-agent/SWE-agent | ★20.1k | 代码-agent 接口与自愈循环参考 |
| SakanaAI/AI-Scientist | ★14.4k | 全自动科研 Agent 的开源实现参考 |
| deepseek-ai/DeepSeek-Math | ★3.4k | 数学推理开源模型与数据，数学建模任务的推理风格参考 |
| promptslab/Promptify | ★4.6k | Prompt 版本管理与结构化输出，产出格式约束参考 |

## 使用提示

- 本库是**兜底与种子**：正常运行流程仍须执行实时检索（`references/search-protocol.md`）。
- 引用本库条目标注"基线库已验证"；引用实时检索结果标注检索时间与出处链接。
- 命中本库主题后，检索词可基于该条目延展（作者后续工作、引用它的工作、同主题 GitHub 仓库）。