# Academic Writing Standards — 学术写作质量标准

> 解决 AI 写作痕迹的核心：不在"改词"，而在改变**句子结构、信息密度、论证节奏**。
> 本文件提供具体可执行的写作标准，所有 Agent 必须遵守。

---

## 0. 核心原则

| 原则 | 说明 |
|------|------|
| **信息密度优先** | 每句话至少传递一个具体信息（系数、方法名、文献、机制），删掉所有可删的词 |
| **用词精准** | 能用"TWFE"不用"传统双向固定效应模型"，能用"ATT"不用"平均处理效应" |
| **句式自然变化** | 长短句交替，不要每句都是"主-谓-宾"结构 |
| **少用过渡词** | 段落之间靠内容逻辑连接，不靠"Furthermore/Moreover/In addition" |

---

## 1. 禁用与慎用词汇表（Kill List）

以下词汇是 AI 痕迹的最强信号，**严禁出现在最终输出中**（综述/报告/论文段落）。

### 1.1 一级禁用（出现即 AI 痕迹）

| 词汇 | 问题 | 替代方案 |
|------|------|---------|
| `Moreover` | 教授公认 99% AI 指标 | 直接开始新句子，或删除 |
| `Furthermore` | 同上，AI 过渡词榜首 | 删除，或换 `We next consider` |
| `Additionally` | 同上 | 删除，用 `Also`（句首）或 `;` 连接 |
| `In addition` | 低信息量填充 | 删除，或换具体连接 |
| `Notably` | 空洞强调 | 直接说事实，让事实自己 notable |
| `Importantly` | 同上 | 删除 |
| `It is worth noting that` | 6 词废话 | 直接说事实 |
| `It is important to` | AI 套话 | 删掉 |
| `plays a crucial role in` | 空洞 | 直接说机制或效应量 |
| `serves as a` | 冗述 | 直接说是什么 |
| `a wide range of` | 模糊概括 | 说具体数量或范围 |
| `a variety of` | 同上 | 同上 |
| `pivotal` | 过度夸张 | 根本别用 |
| `delves` | Science Advances 确认最高 AI 信号词 | `examines` / `investigates` / `studies` |
| `underscores` | 高 AI 信号词 | `shows` / `indicates` / `suggests` |
| `showcasing` | 高 AI 信号词 | `demonstrating` / 直接描述 |
| `intricate` | 高 AI 信号词 | `complex`（如果必须用） |
| `multifaceted` | AI filler | 说具体有几个方面 |
| `landscape` (抽象用法) | "research landscape" 等 | `literature` / `field` |
| `tapestry` | 完全禁止 | — |
| `interplay` | AI 过度使用 | `interaction` / `relationship` |

### 1.2 二级慎用（每篇最多出现 1 次）

| 词汇 | 说明 |
|------|------|
| `leverage` | 用 `use` / `employ` / `utilize` |
| `enhance` | 太泛，说具体怎么提高 |
| `foster` | 同上 |
| `garner` | 非经济学术语 |
| `robust` | 滥用，每篇不超过 2 次 |
| `comprehensive` | 滥用 |
| `significant` | 仅在统计意义上用，不要做普通形容词 |

### 1.3 句式级禁用

| 句式 | 问题 | 替代 |
|------|------|------|
| `It can be observed that...` | 5 词废话 | 直接陈述 |
| `It is noteworthy that...` | 同上 | 直接陈述 |
| `This is because...` | 稚拙 | `because...` |
| `The reason is that...` | 同上 | `because...` |
| `due to the fact that` | 冗述 | `because` / `due to` |
| `In the context of...` | 套话开头 | 直接进入主题 |
| `It is widely known that` | 模糊 | 删掉或加引用 |
| `There is no doubt that` | 过度自信 | `Our results suggest that` |
| `As can be seen from...` | 赘词 | `Figure 1 shows` |

---

## 2. 顶刊写作模式（正面示例）

### 2.1 计量/因果推断类

**模式 A：直接陈述贡献**
> The paper proposes a new framework for estimating the average treatment effect on the treated (ATT) in difference-in-differences (DiD) designs with multiple time periods, variation in treatment timing, and treatment effect heterogeneity.
> — Callaway & Sant'Anna (2021), Journal of Econometrics

特点：一句话密集打包所有关键信息，无废话。

**模式 B：问题→解法**
> To estimate the dynamic effects of an absorbing treatment, researchers often use two-way fixed effects regressions that include leads and lags of the treatment. We show that in settings with variation in treatment timing across units, the coefficient on a given lead or lag can be contaminated by effects from other periods, and apparent pretrends can arise solely from treatment effects heterogeneity.
> — Sun & Abraham (2021), Journal of Econometrics

特点：第一句确立问题场景，第二句给出核心发现。简洁、具体、有冲击力。

**模式 C：直接结论**
> We find that the minimum wage reduces employment in low-wage sectors. The effects are concentrated in industries with the highest exposure to minimum wage increases.
> — 经典 AER 风格

特点：直接说发现，不铺垫、不渲染。

### 2.2 计量经济学写作的核心特征

1. **开头直接**：不用背景铺垫。直接说 "We study..." / "This paper investigates..." / "We propose..."
2. **方法具体**：不说 "advanced methods"，说 "Sun & Abraham (2021) interaction-weighted estimator"
3. **结果量化**：不说 "significant effect"，说 "a 12% increase (β = 0.12, p < 0.01)"
4. **论证紧凑**：一个段落只说一个观点，段首第一句是 topic sentence
5. **引证自然**：`as shown by Callaway & Sant'Anna (2021)` 而非 `(Callaway & Sant'Anna, 2021) shows that`

---

## 3. 句法层面的具体标准

### 3.1 句子开头多样化（但不是靠过渡词）

**好例子：**
- "We estimate..." 
- "Figure 2 plots..."
- "Columns 1–3 report..."
- "Consistent with this mechanism,..."
- "The coefficient on ... is positive and significant at the 1% level."
- "Building on Callaway & Sant'Anna (2021), we..."

**坏例子：**
- "Moreover, the results show that..." — 过渡词开头
- "In addition, we find that..." — 过渡词开头
- "It is also worth noting that..." — 空洞开头

### 3.2 长短句交替

AI 倾向于所有句子长度均匀（15-25 词）。人类写作为了表达复杂关系，自然产生长短变化。

**长句**：用于表达多重关系、条件限制
```
We estimate a staggered difference-in-differences specification that includes 
leads and lags of the treatment indicator, following Sun and Abraham (2021), 
to allow for treatment effect heterogeneity across cohorts.
```

**短句**：用于强调关键结论
```
The results are robust. Placebo tests pass. 
```

### 3.3 信息密度规则

**任何可以被删除而不丢失信息的词，必须删掉。**

| 啰嗦 | 精简 |
|------|------|
| `in order to` | `to` |
| `as a result of` | `because of` |
| `at the same time` | `Meanwhile` / 删除 |
| `in the case of` | `for` / `in` |
| `on the basis of` | `based on` / `from` |
| `a majority of` | `most` |
| `a number of` | `several` / `many` |
| `are found to be` | `are` |
| `has been shown to be` | `is` |

---

## 4. 领域特定指导

### 4.1 计量经济学/因果推断

| 场景 | AI 写法（禁用） | 顶刊写法（遵行） |
|------|---------------|----------------|
| 介绍方法 | "We employ a sophisticated difference-in-differences methodology" | "We estimate a staggered DiD specification (Callaway & Sant'Anna, 2021)" |
| 说结果 | "The results are statistically significant" | "The coefficient is positive and significant at the 5% level (β = 0.047, SE = 0.021)" |
| 说稳健性 | "To ensure the robustness of our findings" | "We assess robustness through..." |
| 平行趋势 | "We conduct a parallel trend test" | "We test for differential pre-trends by including...; the pre-treatment coefficients are jointly insignificant (F = 1.24, p = 0.28)" |
| 安慰剂 | "We perform a placebo test" | "We randomly reassign treatment and re-estimate; the resulting distribution centers on zero" |

### 4.2 机器学习/统计

| 场景 | AI 写法 | 顶刊写法 |
|------|---------|---------|
| 模型选择 | "We choose XGBoost due to its superior performance" | "XGBoost minimizes cross-validated RMSE (CV-RMSE = 0.32) relative to RF (0.38) and LASSO (0.41)" |
| 调参 | "We carefully tune the hyperparameters" | "We tune λ via 5-fold cross-validation on a log-spaced grid of 100 values" |
| 特征工程 | "We perform feature engineering to improve performance" | "We include quadratic terms for X₁, X₂ and their interaction, selected via sequential Bonferroni screening" |
| 泛化 | "The model generalizes well" | "Out-of-sample R² = 0.84, evaluated on the 2022 holdout sample" |

---

## 5. 自检协议（Agent 输出前必须执行）

每次生成文本（综述/报告/论文段落）后，检查以下 5 项：

1. **词汇检查**：全文搜索 `Moreover / Furthermore / Additionally / Notably / Importantly / It is worth / plays a crucial / pivotal / underscores / delves / intricate` —— 有任何出现 → 删除或替换
2. **句长检查**：连续 3 句长度都在 15-25 词 → 改写（拉长一句或缩短一句）
3. **开头检查**：连续 3 个段落以相同方式开头 → 变化
4. **信息密度检查**：每段是否有至少一个具体数字、方法名或引用？→ 没有则补
5. **自然度检查**：读一遍，是否有任何句子听起来像"模板"→ 改掉

---

## 6. 不同输出类型的要求

| 输出类型 | 风格 | 特殊要求 |
|---------|------|---------|
| 文献综述 | 客观、信息密集 | 自然引证，避免"多篇文献指出"；用具体作者+发现 |
| 方法论描述 | 精确、技术性 | 公式+文字并行，不要用文字复述公式 |
| 结果报告 | 简洁、量化 | 小标题直击发现："2.1 基准回归：FDI 提升 12.5%" |
| 结论 | 克制、不夸大 | 一句话总结发现 → 一句话政策含义 → 一句局限 |
| 代码注释 | 极简 | 只解释 why，不解释 what（代码本身说明 what） |
| 实验报告 | 结构化、可复现 | 参数、数据、环境、seed 面面俱到但不过度解释 |

---

## 7. 最终校验

QA 阶段增加一条评分：**写作质量（10 分）**—— 检查上述 5 项自检，每违反一项扣 2 分。

写作质量 < 6 分 → 不能交付，需要重写。
