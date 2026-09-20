# 数学建模专项指南

数学建模（竞赛/课程/科研）时，在通用流程之上叠加。本专项之外，配套资源见 `references/prompt-resources.md`（LLM 数学推理研究成果）与 `references/search-protocol.md`（强制检索）。

## 题型识别（据此定制检索词与提示词结构）

- **优化类**：线性/整数规划、启发式 → 检索 `prompt optimization integer programming solver LLM`；提示词要求"目标函数/约束显式写出，求解器选型并说明"。
- **评价类**：熵权法、TOPSIS、AHP → 检索 `TOPSIS 熵权法 python 数学建模 prompt`；提示词要求"正向化、非负、权重和=1"的显式校验步骤。
- **预测类**：时间序列、回归、机器学习 → 检索 `prompt time series forecasting LLM`；提示词要求"滚动回测 + 朴素基线 + 区间输出"。
- **其他**：微分方程、图论、分类 → 按题检索对应模型 + prompt 的组合词。

## 搜索关键词

按「题型 + 模型/算法 + 工具库 + 数学建模」构造，如 `TOPSIS 熵权法 python 数学建模`；LLM 数学推理方向加 `LLM mathematical reasoning prompt` 系列词（种子见 `references/prompt-resources.md` 第三部分）。

## 提示词必备结构

模型假设、符号说明、模型建立与求解、灵敏度/稳健性分析、可复现代码（依赖+种子）。报告按「摘要—问题重述—假设—符号—建模求解—灵敏度—评价推广」组织。

## 一题多问衔接

- 各子问共用同一数据划分并防泄漏；
- 后问依赖前问结果（如预测→优化）时注明不确定性，并对关键输入做 ±10% 类扰动；
- 一题多问提示词可借鉴 Meta-Prompting（arXiv 2401.12954）的模式：拆成多个专门子任务，各自配指令后汇总，避免一条提示词过载。

## LLM 数学推理提示词研究成果（注入优化稿时选用）

| 成果 | 出处（见 resources 基线库） | 注入方式 |
|---|---|---|
| 思维链（CoT） | arXiv 2201.11903 | 提示词要求"先写出建模思路与公式推导，再求解" |
| 自洽性（Self-Consistency） | arXiv 2203.11171 | 数值敏感类题目要求"独立求解 N 次取众数/均值，报告一致性" |
| 工具校验（DeepSeekMath / ReAct） | arXiv 2402.03300 / arXiv 2210.03629 | 提示词要求"关键中间结果用 Python 验证后再继续" |
| 自验证（DeepSeekMath-V2） | arXiv 2511.22570 | 最终答案前增加"独立验证步骤：重新读题核对所求、复核单位与量纲" |

## 已知坑

见 `references/known-pitfalls.md`；小样本分类要求见 `references/machine-learning.md`。

## 实测证据（本项目沉淀）

方法提示直接决定正确性——移除"小样本优先指数平滑"后 Q1 MAPE 7.19%→8.50%；评价类含"熵权+TOPSIS+正向化+非负+权重和=1"的提示词完全复现基准排名；含趋势转折数据上滚动回测 MAPE 0.92%/0.83%（HW/SARIMA）优于单次留出，转折后两模型预测分歧需明示。