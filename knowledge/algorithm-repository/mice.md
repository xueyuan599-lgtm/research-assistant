---
title: MICE — 多重插补（链式方程）
type:
  - preprocessing
  - statistical-inference
domain:
  - generic-ml
  - biostatistics
---
# MICE — 链式方程多重插补（Multiple Imputation by Chained Equations）

- **来源**: van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67. | Rubin, D. B. (1976). Inference and Missing Data. *Biometrika*, 63(3), 581–592.
- **DOI**: 10.18637/jss.v045.i03 / 10.1093/biomet/63.3.581
- **方法类别**: 数据处理 / 统计推断

## 数学设定

### 问题框架：三种缺失机制（Rubin 1976）

记完整数据 $Y = (Y_{\text{obs}}, Y_{\text{mis}})$，缺失指示矩阵 $R$（$R_{ij}=1$ 表示观测到 $Y_{ij}$）。缺失机制按缺失概率对**已观测数据**与**未观测数据**的依赖性分为三类：

| 机制 | 全称 | 缺失概率 | 例 |
|------|------|---------|-----|
| **MCAR** | Missing Completely At Random | $P(R \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(R)$ 不依赖任何数据 | 仪器随机故障 |
| **MAR** | Missing At Random | $P(R \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(R \mid Y_{\text{obs}})$ 只依赖已观测 | 高收入者更少填收入（收入与其他已观测相关） |
| **MNAR** | Missing Not At Random | 依赖 $Y_{\text{mis}}$ 本身 | 收入越高越不愿意报告收入 |

**关键**：缺失机制决定方法合法性。MCAR/MAR 下可做多重插补得到无偏推断；MNAR 缺失本身携带信息，任何插补都需额外建模（如 pattern-mixture、选择模型）。国赛第一步必须先做**机制判断**：MCAR 可删行（若缺失率低）；MAR 用 MICE；MNAR 需说明假设并做敏感性分析。

### 链式方程多重插补（MICE / Fully Conditional Specification）

MICE 不用联合分布，而是**逐变量**用条件分布迭代插补。设 $p$ 个含缺失的变量，对每个变量 $Y_k$ 用一个回归模型（以其余变量为自变量）拟合，并在每轮迭代中更新插补值：

**算法（每轮 $t$，共 $m$ 套插补可用于合并）**：
1. 对每个变量先用简单法（均值/中位数）填初始值。
2. 对 $k = 1, \dots, p$，仅用"$Y_k$ 已观测的样本"拟合并得到条件分布 $P(Y_k \mid Y_{\text{other}})$，从中**抽样**填补 $Y_k$ 的缺失值。
3. 重复步骤 2 若干轮（通常 5–10 轮）直到收敛，得到一套完整数据。

对同一算法跑 $m$ 套（常取 $m=5$ 或 $m \geq 20$），得到 $m$ 套完整数据集，这就是**多重插补**——相比单次插补，它显式记录了插补的不确定性。

### Rubin 合并规则（Rubin 1976, 1987）

对 $m$ 套完整数据各自估计目标量 $\hat{Q}_i$（如回归系数、均值），合并估计为取均值：

$$
\bar{Q} = \frac{1}{m} \sum_{i=1}^{m} \hat{Q}_i
$$

合并方差由**组内方差**与**组间方差**组成：

$$
T = \bar{U} + \left(1 + \frac{1}{m}\right) B,\qquad
\bar{U} = \frac{1}{m}\sum_i U_i,\quad
B = \frac{1}{m-1}\sum_i (\hat{Q}_i - \bar{Q})^2
$$

其中 $\bar{U}$ 为平均插补内方差，$B$ 为插补间方差，因子 $(1+1/m)$ 修正有限 $m$。多重插补的推断标准误因此**大于**任何单套数据，正确体现了缺失带来的额外不确定性。

### 与均值/中位数填补的对比（为什么逼不得已才用简单填补）

| 方法 | 问题 | MICE 的改进 |
|------|------|------------|
| 均值填补 | **低估方差**：把缺失都填成均值，分布被压缩，估计方差偏小、标准误被低估、p 值被高估（假阳性） | 从条件分布抽样，保留不确定性 |
| 中位数填补 | 同上，还扭曲分布形状（峰度） | 保留插补间方差 |
| 简单回归填补 | **扭曲相关/回归系数**：系统性地将缺失值向回归直线收缩，削弱变量间关系 | 抽样 plus noise，保留相关性 |
| 单次插补 | 只给一套数据，**无法反映插补不确定性** | $m$ 套数据 + Rubin 合并 |

经验法则：缺失率 < 5% 且 MCAR 时，可考虑删除行；缺失率较高或 MAR 时优先 MICE 多重插补，杜绝"一律填均值"。

## 关键假设

- **缺失机制为 MAR（或 MCAR）**：这是 MICE 合法性的核心；若强烈怀疑 MNAR，需说明并做敏感性分析。
- **条件分布正确设定**：每个变量的条件模型要能捕捉其与其余变量的关系（类别变量用 logistic/多项，连续变量用线性/岭回归）。
- **观测变量足以解释缺失**（MAR 的可忽略性）：已观测特征需包含与缺失相关的信息。

## 适用场景

- **表格数据含广泛缺失**：问卷、判据指标缺测，国赛指标体系类题目
- **时间序列缺失块/稀疏**：配合时序插补（见 PyPOTS/SAITS 条目，若库内存在）
- **需做后续统计推断（回归、相关性、假设检验）**：MICE 提供带正确不确定性的数据，优于一刀切均值
- **多变量联动缺失**：MICE 逐变量条件插补天然处理多变量

### 不适用 / 注意

- **MNAR 缺失**：MICE 不纠正，需敏感性分析或 MNAR 建模
- **超高维（$p > n$）**：每个条件模型需正则化（如 lasso-based mice）
- **极低缺失率的小样本**：可能引入噪声，先评估是否直接删除

## 实现要点

### Python（IterativeImputer）

```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 需先启用
from sklearn.impute import IterativeImputer

# 基于"轮转回归 + 随机"的 MICE 型插补
imputer = IterativeImputer(
    estimator=None,            # 默认 BayesianRidge（线性），可换 RandomForestRegressor
    max_iter=10,               # 链式方程轮数
    random_state=42,           # 可复现
    sample_posterior=True,     # 从后验/误差分布抽样，体现 MAR 不确定性（类 MICE）
)
X_complete = imputer.fit_transform(X)

# 注意：IterativeImputer 是"单套"完整数据；
# 需要真实多重合并时循环 m 次跑不同 seed，或直接用 R 的 mice 包。
```

### R（mice 包，官方实现）

```r
library(mice)
# 1) 缺失结构 + 机制初步判断
md.pattern(data)              # 缺失模式矩阵
# 2) MICE 插补 m 套
imp <- mice(data, m = 5, method = "pmm", seed = 42)   # pmm = 预测均值匹配，稳健
# 3) 用合并规则拟合模型
fit <- with(imp, lm(y ~ x1 + x2))
pooled <- pool(fit)           # Rubin 合并规则
summary(pooled)
```

### 插补质检（缺一不可）

1. **分布对比图**：插补后 $Y_{\text{obs}}$ 与 $Y_{\text{imp}}$ 的密度/箱线图应大致重叠加；MICE 自带 `stripplot/xyplot` 诊断。
2. **关键统计对比**：均值、方差、相关系数在插补前后不应系统性偏移（若偏移明显，说明条件模型误设）。
3. **敏感性分析**：比较"MICE(MAR) vs 简单填均值 vs 删行"三种下游模型的结论是否稳健；MNAR 场景下故意按不利方向偏移插补值，看结论是否翻转。
4. **收敛诊断**：观察插补值随轮次是否稳定（trace plot）。

## 国赛应用

### 缺失值处理章节的完整链：机制判断 → 选法 → 质检 → 敏感性

> 数据存在缺失。我们按 Rubin (1976) 三机制分类进行判断：以缺失指示 $R$ 为因变量做 logistic 回归，各已观测协变量系数联合不显著（伪 $R^2 = 0.02$，p = 0.31），支持 **MAR/MCAR** 假设（$P(R\mid Y_{\text{obs}})$ 近似恒定）。据此采用 **MICE 多重插补**：对 $p$ 个含缺失变量逐变量建条件模型，经 10 轮链式方程迭代、生成 $m=5$ 套完整数据，再用 Rubin 合并规则得到 $\bar{Q} = \frac{1}{m}\sum\hat{Q}_i$、方差 $T = \bar{U} + (1+1/m)B$。

质检：图 X 显示 $Y_{\text{obs}}$ 与 $Y_{\text{imp}}$ 密度高度重叠；插补前后关键变量的相关系数变化 < 0.05，均值偏差 < 1%（未填均值时的"方差压缩"未出现）。

敏感性：将插补方式分别替换为均值填补与整行删除后重跑主模型，核心系数符号与量级一致（变化 < 10%），结论稳健；在 MNAR 侧施加 ±20% 偏移仍不翻转，说明结果对缺失假设不敏感。

要点：
1. **先做机制判断再选法**，这是和"直接 fillna(mean)"拉开差距的关键一步，评委关心你知不知道自己为什么这么处理。
2. 明确写"均值填补会低估方差、扭曲相关，故弃用"（见上表），体现方法选择有理由而非默认。
3. **用 Rubin 合并规则给标准误**，缺失不确定性进入推断，比单套插补更诚实。

## 参考文献

- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67.
- Rubin, D. B. (1976). Inference and Missing Data. *Biometrika*, 63(3), 581–592.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. John Wiley & Sons.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC.
