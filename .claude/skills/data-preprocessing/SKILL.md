---
name: data-preprocessing
description: Unified data preprocessing reference across multiple toolkits (scikit-learn, PyPOTS, sktime, scikit-fda, fdasrsf). Covers data profiling, missing value imputation, outlier detection, encoding, scaling, time-series/functional data preprocessing, pipeline construction, and leakage prevention.
license: MIT license
metadata:
  skill-author: K-Dense Inc.
---

# 数据预处理 Skill

> 全栈数据预处理参考，覆盖 4 种数据类型（表格/时间序列/函数型/面板）× 9 个处理域。
> 与 preprocessing-inspector → preprocessing-cleaner → preprocessing-transformer → preprocessing-validator 四阶段管线无缝衔接。

## 使用场景

- 需要对原始数据进行系统性质量评估和探查
- 需要处理缺失值、异常值或类型不一致
- 需要对分类变量编码、对数值特征缩放
- 处理时间序列数据，需要智能插补或平稳化
- 处理函数型数据，需要平滑、对齐或降维
- 需要构建可复现的预处理流水线
- 需要防止数据泄漏（train/test 污染）
- 需要将预处理参数持久化供推理复用

## 触发关键词

```
数据预处理, 数据清洗, 数据探查, 缺失值, 异常值, 标准化, 归一化, 编码, 插补,
数据质量, 数据类型转换, 重采样, 平稳化, 平滑, 对齐, 流水线, 管道,
imputation, outlier, data profiling, preprocessing pipeline, data cleaning,
missing value, scaler, encoder, Pipeline, ColumnTransformer
```

## 预处理流水线架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Inspector   │───▶│   Cleaner    │───▶│ Transformer  │───▶│  Validator   │
│  (探查)      │    │  (清洗)      │    │  (变换)      │    │  (验证)      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

| 阶段 | Agent 文件 | 对应能力域 |
|------|-----------|-----------|
| 探查 | `research-assistant/agents/data-viz/preprocessing-inspector.md` | ① 数据探查与质量评估 |
| 清洗 | `research-assistant/agents/data-viz/preprocessing-cleaner.md` | ② 缺失值处理、③ 异常值检测 |
| 变换 | `research-assistant/agents/data-viz/preprocessing-transformer.md` | ④ 编码、⑤ 缩放、⑥ 时序、⑦ 函数型 |
| 验证 | `research-assistant/agents/data-viz/preprocessing-validator.md` | —（验证预处理产出） |

## 安装

```bash
uv pip install pandas numpy scikit-learn
uv pip install sktime pypots
uv pip install scikit-fda fdasrsf
uv pip install category-encoders
```

## 快速入门

### 示例 1：表格数据完整预处理

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num = X.select_dtypes(include=np.number).columns.tolist()
cat = X.select_dtypes(include="object").columns.tolist()

pipe = Pipeline([("prep", ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scl", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])), ("clf", RandomForestClassifier(random_state=42))])
pipe.fit(X_train, y_train)
print(f"Test acc: {pipe.score(X_test, y_test):.3f}")
```

### 示例 2：时间序列缺失值插补与平稳化

```python
import pandas as pd, numpy as np
from pypots.imputation import SAITS
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.pipeline import Pipeline

ts = pd.read_csv("sensor.csv", index_col="time", parse_dates=True)
y_train, y_test = temporal_train_test_split(ts["value"], test_size=0.2)
X_3d = y_train.values.reshape(-1, 1, 1)
saits = SAITS(n_steps=X_3d.shape[0], n_features=1, n_layers=2, epochs=30)
saits.fit(X_3d)
y_clean = pd.Series(saits.impute(X_3d)[:, 0, 0])
y_trans = Pipeline([
    ("diff", Differencer(1)),
    ("box", BoxCoxTransformer()),
]).fit_transform(y_clean)
```

## 核心能力

### ① 数据探查与质量评估

数据类型自动检测、缺失模式分析、分布特征概览。

```python
import pandas as pd, missingno as msno
df = pd.read_csv("data.csv")
print(df.info(), df.describe())
msno.matrix(df)
print(df.isnull().mean().sort_values(ascending=False))
```

**See:** `references/tabular-preprocessing.md`

### ② 缺失值处理

覆盖表格/时间序列/函数型数据的多策略插补。

```python
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
SimpleImputer(strategy="median")                         # 均值/中位数/众数/常数
IterativeImputer(max_iter=10, random_state=42)           # 多变量迭代
KNNImputer(n_neighbors=5, weights="distance")            # KNN 插补
```

**See:** `references/tabular-preprocessing.md`

### ③ 异常值检测与处理

```python
from sklearn.ensemble import IsolationForest
from scipy import stats
Q1, Q3 = df[col].quantile([0.25, 0.75]); iqr = Q3 - Q1
out = (df[col] < Q1 - 1.5*iqr) | (df[col] > Q3 + 1.5*iqr)
z = np.abs(stats.zscore(df[col])); out = z > 3
iso = IsolationForest(contamination=0.05, random_state=42)
out = iso.fit_predict(df[num_cols]) == -1
```

**See:** `references/tabular-preprocessing.md`

### ④ 变量编码

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OrdinalEncoder(categories=[["low", "medium", "high"]])
TargetEncoder(cols=["category"])
```

**See:** `references/tabular-preprocessing.md`

### ⑤ 特征缩放

```python
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
    RobustScaler, PowerTransformer, QuantileTransformer)
StandardScaler(); RobustScaler()
PowerTransformer("yeo-johnson")
QuantileTransformer(n_quantiles=1000, output_distribution="normal")
```

**See:** `references/tabular-preprocessing.md`

### ⑥ 时间序列预处理

```python
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.date import DateTimeFeatures
Differencer(lags=1); Differencer(lags=12)
BoxCoxTransformer()
DateTimeFeatures(ts_freq="D", feature_set="all")
```

**See:** `references/time-series-preprocessing.md`

### ⑦ 函数型数据预处理

```python
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.dim_reduction import FPCA
KernelSmoother(kernel_estimator="normal", bandwidth=0.5)
FPCA(n_components=5, centering=True)
```

**See:** `references/functional-data-preprocessing.md`

### ⑧ 流水线构建

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols),
])
pipe = Pipeline([("prep", prep), ("clf", RandomForestClassifier())])
import joblib; joblib.dump(pipe, "pipeline.pkl")
```

**See:** `references/pipeline-and-leakage.md`

### ⑨ 数据泄漏防范

```python
# 错误：对整个数据集 fit 后再拆分
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# 正确：先拆分，训练集 fit，测试集只 transform
X_train, X_test = train_test_split(X, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 更优：用 Pipeline 自动防泄漏
Pipeline([("scl", StandardScaler()), ("clf", RandomForestClassifier())])
```

**See:** `references/pipeline-and-leakage.md`

## Agent 集成

| Agent | 相关能力域 | 技能提供的支持 |
|-------|-----------|---------------|
| `preprocessing-inspector` | ① 数据探查 | 探查工具链与代码模板 |
| `preprocessing-cleaner` | ② 缺失值、③ 异常值 | PyPOTS/IQR/IsolationForest 实现 |
| `preprocessing-transformer` | ④-⑦ 编码/缩放/时序/函数型 | 全部变换操作参考 |
| `preprocessing-validator` | — | 提供预处理产出供验证 |

## 最佳实践

| 实践 | 说明 |
|------|------|
| 先探查再处理 | 先运行 Data Inspector 了解数据状况再决定策略 |
| 拆分再 fit | 预处理参数仅在训练集 fit，测试集只 transform |
| 用 Pipeline 封装 | GridSearchCV 时自动防泄漏 |
| 用 ColumnTransformer | 数值列缩放 + 分类列编码，一份代码完成 |
| 固定随机种子 | random_state=42 确保可复现 |
| 保存预处理参数 | joblib.dump(pipeline, "prep.pkl") 供推理复用 |
| 高缺失率先判断 | 缺失率 > 50% 优先考虑删除列 |

## 常见问题

| 问题 | 解决 |
|------|------|
| 数据泄漏 | Pipeline + 先拆分后 fit |
| Pipeline 缺少 transform | 确保中间步骤是 Transformer 类型 |
| 列名不匹配 | 用名称列表而非索引，remainder=passthrough 调试 |
| 高缺失率 >50% | 优先删除该列或用缺失指示变量 |
| 时序未来数据泄漏 | 使用 TimeSeriesSplit 交叉验证 |
| 目标编码泄漏 | 在 CV fold 内计算或 TargetEncoder(cv=5) |

## 参考文档

| 文件 | 内容 |
|------|------|
| `references/tabular-preprocessing.md` | 表格数据：探查、缺失值、异常值、编码、缩放 |
| `references/time-series-preprocessing.md` | 时间序列：PyPOTS 插补、平稳化、重采样、日期特征 |
| `references/functional-data-preprocessing.md` | 函数型数据：平滑、基展开、SRSF 对齐、FPCA |
| `references/pipeline-and-leakage.md` | Pipeline/ColumnTransformer + 5 种泄漏模式 |

## 关联资源

- [scikit-learn 预处理参考](../scikit-learn/references/preprocessing.md)
- [scikit-learn 流水线参考](../scikit-learn/references/pipelines_and_composition.md)
- [数据预处理管线规则](../../../.claude/rules/07-data-preprocessing.md)
