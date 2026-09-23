---
title: SHAP — Shapley 值可解释性与特征归因
type:
  - regression
  - classification
  - statistical-inference
domain:
  - generic-ml
---
# SHAP — Shapley 值可解释性与特征归因

- **来源**: Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS)*.
- **DOI**: 10.5555/3295222.3295230
- **方法类别**: 统计建模 / 模型可解释性

## 数学设定

### 问题框架：加性特征归因

一个模型的预测 $f(x)$ 可被分解为**基准值**与各特征的**归因贡献**之和（加性特征归因模型，additive feature attribution）：

$$
f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j
$$

其中 $\phi_0 = \mathbb{E}[f]$ 为全体样本上的平均预测（基准值），$\phi_j$ 为第 $j$ 个特征对本次预测的贡献。所有局部归因满足**局部精确性**（local accuracy）：各特征贡献之和恰等于预测值与基准值的差。

### Shapley 值定义

SHAP 将合作博弈论中的 **Shapley 值**搬运到特征归因。对 $M$ 个特征的特征集合 $\mathcal{F}$，任意子集 $S \subseteq \mathcal{F}$ 的边际贡献由条件期望 $f_S(S)$ 刻画。第 $j$ 个特征的 Shapley 值为所有**特征排列**下该特征边际贡献的平均：

$$
\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!\,(M-|S|-1)!}{M!} \left[ f_{S \cup \{j\}}(S \cup \{j\}) - f_S(S) \right]
$$

这是对 $2^M$ 个子集的加权平均（权重为「把 $j$ 加进子集 $S$ 的排列数」占比）。组合意义：**某特征对预测的贡献 = 它是第一个被加入 $S$ 的特征时边际效应的加权平均**，消除顺序任意性。

### 三个公理（为何 Shapley 唯一）

Shapley 值是满足以下三公理的**唯一**归因方案（Lundberg & Lee 2017; Shapley 1953）：

1. **局部精确性 (Local Accuracy)**: $\sum_j \phi_j = f(x) - \mathbb{E}[f]$
2. **一致性 (Consistency)**: 若两个模型 $f, f'$ 满足「加入特征 $j$ 的边际贡献在 $f'$ 中不减少」，则 $\phi_j(f') \geq \phi_j(f)$—重要性排序与模型单调一致。
3. **缺失性 (Missingness)**: 不参与任何 $f_S$ 的特征贡献为 0。

一致性公理是 SHAP 相对朴素重要性**最关键的优势**：它保证归因排序不会被「增益被交互项掩盖」所扭曲。

### 与 GBDT 内置增益重要性的区别

| 维度 | GBDT gain / cover 重要性 | SHAP |
|------|------------------------|------|
| 符号方向 | 无（仅绝对值） | 有（正/负贡献） |
| 交互归因 | 无（单一标量） | SHAP interaction values 分解交互 |
| 一致性 | 不保证（分裂次数可能误导） | 由一致性公理保证 |
| 可加性 | 无 | 局部精确（逐样本分解） |
| 计算 | 训练时顺带 | 需额外计算（TreeSHAP） |

**核心差异**：GBDT 的 gain 是"这个特征在整个模型训练中被用来分裂时平均带来的损失下降"，是**模型内部**的全局统计；SHAP 是"这个特征对**某个样本**预测值的贡献（含方向）"，是**逐样本 + 可加**的归因。两者排序可能不同——一个特征很少被分裂但每次分裂影响很大（高 gain）也可能在 SHAP 中排名靠前。

### 三类核心可视化

- **summary plot（蜜蜂图/蜂群图）**：横轴为 SHAP 值，每个样本一个点，颜色表示该特征取值高低。展示**特征方向 + 排序 + 取值与贡献的关系**——若红色点（高取值）集中在 SHAP>0 侧，说明"取值越高、贡献越大"。
- **依赖图 (dependence plot)**：单个特征 $x_j$ vs 其 SHAP 值 $\phi_j$，用于查看**边际影响的非线性形状**（是否为阈值、倒 U 等），可选着色第二个特征看交互。
- **force plot**：单个样本的"力量图"，把基准值 $\phi_0$ 往左推、各特征贡献往右拉，直观展示"为什么判成这个值/类别"。

## 关键假设

- **特征独立（近似）**：TreeSHAP 的快路径假设特征近似独立；强相关特征时归因可能不稳，需结合交互值或 Permutation SHAP 核对。
- **加性分解成立**：模型预测可被分解为各特征贡献之和（对树/线性模型成立；对深度网络用 DeepSHAP 近似）。
- **归因是"贡献"而非"因果"**：SHAP 回答"模型把预测归因到哪些输入变化"，**不是**"改变该特征会导致目标怎样变化"的反事实因果。若目标是因果效应，用因果推断方法（DID、因果森林）而非 SHAP。

## 适用场景

- **模型"变量重要性/影响因子分析"**：国赛评价、归因类问题，回答"哪个因素对结果影响最大、方向如何"
- **非线性模型的可解释性兜底**：评委要求解释 GBDT/黑盒模型时，用 SHAP 给出逐样本归因
- **特征方向与符号**：gain 只能给"重要程度"，SHAP 能给"正贡献/负贡献"
- **交互作用探测**：SHAP interaction values 定位特征间交互

### 不适用 / 注意

- **需要因果结论时**：SHAP 是预测归因，不能当因果证据
- **特征高度共线**：归因不稳定，多个近似特征分摊贡献
- **计算量大**：精确 Shapley 是 $O(2^M)$，需用 TreeSHAP（树模型上 $O(TLD^2)$）或近似采样

## 实现要点

### Python 用法（shap 库）

```python
import shap

# 基于树模型（XGBoost / LightGBM）计算
explainer = shap.TreeExplainer(model)        # model 为已训练的 GBDT
shap_values = explainer.shap_values(X)       # 逐样本 SHAP 值，shape=(n, M)

# 全局：summary plot（蜜蜂图）
shap.summary_plot(shap_values, X)            # 中文标签需自绘传给 feature_names

# 全局：均值 abs(SHAP) 排序的特征重要性
import numpy as np
mean_abs = np.abs(shap_values).mean(axis=0)  # 每个特征的平均 |贡献|
feat_rank = np.argsort(mean_abs)[::-1]

# 局部：force plot（单样本）
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])

# 依赖图：单特征边际影响
shap.dependence_plot("feature_name", shap_values, X)
```

注意：sklearn 的 `model.feature_importances_`（gain/cover）是与 SHAP 互补的**模型内部**重要性；国赛建议**两者并列**——用 gain 论证"模型学习用到了谁"，用 SHAP 论证"具体方向与强度"。

## 国赛应用

### "变量重要性 / 影响因子分析"节写作模板（每句带数值）

> 我们以 XGBoost 为基模型，用 SHAP 对影响因子进行逐样本归因。特征 $j$ 的 Shapley 贡献 $\phi_j$ 满足加性分解 $f(x)=\phi_0+\sum_j\phi_j$（局部精确性）。图 X 的 summary plot 显示，特征 A 的平均 $|\phi_A| = 0.xx$，排名第 1，且 SHAP 值随 A 取值增大而升高（Spearman 相关 $\rho = +0.xx$），表明"A 越大、目标越高"；特征 B 均贡献 $|\phi_B| = 0.xx$，方向为负，说明 B 与目标呈反向关系。依赖图显式 A 在阈值 $x_A > x_A^*$ 后贡献趋平，呈非线性（分段边际效应）。

要点：
1. **每个结论配一个真实数值**（均值 |SHAP|、方向、排序、相关度），不写"显著影响"这类空话。
2. **明确讲清"贡献 ≠ 因果"**：写"该分析揭示模型将预测归因于哪些输入变化，属预测归因而非因果推断"——既诚实又堵住评委技术追问。
3. **增益 + SHAP 双证据**：先给 `model.feature_importances_`（gain）做全局排序，再用 SHAP 给方向与逐样本分布，二者互相印证。
4. **与"某某指标最重要"的普通排序文字区分**：SHAP 能回答"重要到什么程度、正负方向、非线性形状"，这是评分看重的分析深度。

## 参考文献

- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS)*.
- Shapley, L. S. (1953). A Value for n-Person Games. In *Contributions to the Theory of Games II*, 307–317. Princeton University Press.
