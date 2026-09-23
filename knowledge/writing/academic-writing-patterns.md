# 学术写作范式库（参考层）

> `.claude/rules/02-academic-writing-standards.md` 的**下沉参考层**。该规则常驻核心已含禁用词表与自检协议；
> 本文件承载**正面范式与领域指导**——写文献综述、方法论描述、结果报告、论文段落前**读一遍**。
> 不写这些体裁时无需读。

---

## 1. 顶刊写作模式（正面示例）

### 1.1 计量/因果推断类

**模式 A：直接陈述贡献**
> The paper proposes a new framework for estimating the average treatment effect on the treated (ATT) in difference-in-differences (DiD) designs with multiple time periods, variation in treatment timing, and treatment effect heterogeneity.
> — Callaway & Sant'Anna (2021), Journal of Econometrics

一句话密集打包所有关键信息，无废话。

**模式 B：问题→解法**
> To estimate the dynamic effects of an absorbing treatment, researchers often use two-way fixed effects regressions that include leads and lags of the treatment. We show that in settings with variation in treatment timing across units, the coefficient on a given lead or lag can be contaminated by effects from other periods, and apparent pretrends can arise solely from treatment effects heterogeneity.
> — Sun & Abraham (2021), Journal of Econometrics

第一句确立问题场景，第二句给出核心发现。简洁、具体、有冲击力。

**模式 C：直接结论**
> We find that the minimum wage reduces employment in low-wage sectors. The effects are concentrated in industries with the highest exposure to minimum wage increases.
> — 经典 AER 风格

直接说发现，不铺垫、不渲染。

### 1.2 计量经济学写作的核心特征

1. **开头直接**：不用背景铺垫。直接 "We study..." / "This paper investigates..." / "We propose..."
2. **方法具体**：不说 "advanced methods"，说 "Sun & Abraham (2021) interaction-weighted estimator"
3. **结果量化**：不说 "significant effect"，说 "a 12% increase (β = 0.12, p < 0.01)"
4. **论证紧凑**：一段只说一个观点，段首第一句是 topic sentence
5. **引证自然**：`as shown by Callaway & Sant'Anna (2021)` 而非 `(Callaway & Sant'Anna, 2021) shows that`

---

## 2. 句法层面的具体标准

### 2.1 句子开头多样化（但不是靠过渡词）

**好例子：** "We estimate..." / "Figure 2 plots..." / "Columns 1–3 report..." /
"Consistent with this mechanism,..." / "The coefficient on ... is positive and significant at the 1% level." /
"Building on Callaway & Sant'Anna (2021), we..."

**坏例子：** "Moreover, the results show that..." / "In addition, we find that..." /
"It is also worth noting that..."（均为过渡词/空洞开头）

### 2.2 长短句交替

AI 倾向所有句子长度均匀（15-25 词）；人类写作自然产生长短变化。

**长句**——用于表达多重关系、条件限制：
```
We estimate a staggered difference-in-differences specification that includes
leads and lags of the treatment indicator, following Sun and Abraham (2021),
to allow for treatment effect heterogeneity across cohorts.
```

**短句**——用于强调关键结论：
```
The results are robust. Placebo tests pass.
```

### 2.3 信息密度规则

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

## 3. 领域特定指导

### 3.1 计量经济学/因果推断

| 场景 | AI 写法（禁用） | 顶刊写法（遵行） |
|------|---------------|----------------|
| 介绍方法 | "We employ a sophisticated difference-in-differences methodology" | "We estimate a staggered DiD specification (Callaway & Sant'Anna, 2021)" |
| 说结果 | "The results are statistically significant" | "The coefficient is positive and significant at the 5% level (β = 0.047, SE = 0.021)" |
| 说稳健性 | "To ensure the robustness of our findings" | "We assess robustness through..." |
| 平行趋势 | "We conduct a parallel trend test" | "We test for differential pre-trends by including...; the pre-treatment coefficients are jointly insignificant (F = 1.24, p = 0.28)" |
| 安慰剂 | "We perform a placebo test" | "We randomly reassign treatment and re-estimate; the resulting distribution centers on zero" |

### 3.2 机器学习/统计

| 场景 | AI 写法 | 顶刊写法 |
|------|---------|---------|
| 模型选择 | "We choose XGBoost due to its superior performance" | "XGBoost minimizes cross-validated RMSE (CV-RMSE = 0.32) relative to RF (0.38) and LASSO (0.41)" |
| 调参 | "We carefully tune the hyperparameters" | "We tune λ via 5-fold cross-validation on a log-spaced grid of 100 values" |
| 特征工程 | "We perform feature engineering to improve performance" | "We include quadratic terms for X₁, X₂ and their interaction, selected via sequential Bonferroni screening" |
| 泛化 | "The model generalizes well" | "Out-of-sample R² = 0.84, evaluated on the 2022 holdout sample" |

---

## 4. 不同输出类型的要求

| 输出类型 | 风格 | 特殊要求 |
|---------|------|---------|
| 文献综述 | 客观、信息密集 | 自然引证，避免"多篇文献指出"；用具体作者+发现 |
| 方法论描述 | 精确、技术性 | 公式+文字并行，不要用文字复述公式 |
| 结果报告 | 简洁、量化 | 小标题直击发现："2.1 基准回归：FDI 提升 12.5%" |
| 结论 | 克制、不夸大 | 一句话总结发现 → 一句话政策含义 → 一句局限 |
| 代码注释 | 极简 | 只解释 why，不解释 what（代码本身说明 what） |
| 实验报告 | 结构化、可复现 | 参数、数据、环境、seed 面面俱到但不过度解释 |

---

## 5. 最终校验

QA 阶段按规则文件中的自检协议逐项检查（词汇/句长/开头/信息密度/自然度）。
**写作质量 < 6 分 → 不能交付，需要重写。**
