# 表格数据预处理参考

> 涵盖表格数据的探查、缺失值处理、异常值检测与处理、变量编码、特征缩放。

---

## 1. 数据探查

### 基础信息

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.info())              # 列类型、非空数、内存占用
print(df.describe())          # 数值列的统计摘要（均值、标准差、分位数）
print(df.describe(include="object"))  # 分类列的频次、唯一值
```

### 缺失模式分析

```python
import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df)               # 缺失热图（白色 = 缺失）
msno.bar(df)                  # 每列缺失量柱状图
msno.heatmap(df)              # 缺失相关性热图
plt.savefig("missing_analysis.png")
```

### 数据类型推断与修正

```python
# 查看每列数据类型
print(df.dtypes)

# 类型转换
df["col"] = df["col"].astype("float64")
df["date"] = pd.to_datetime(df["date"])
df["category"] = df["category"].astype("category")

# 混合类型检测：检查 object 列的唯一值数量
for col in df.select_dtypes("object"):
    print(f"{col}: {df[col].nunique()} unique, {df[col].isnull().sum()} missing")
```

---

## 2. 缺失值处理

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 删除（dropna） | 缺失率低，随机缺失 | 简单 | 丢失信息，样本减少 |
| SimpleImputer | 低缺失率，MCAR | 快速，稳定 | 忽略变量关系 |
| IterativeImputer | 多变量相关 | 利用变量间关系 | 计算量大 |
| KNNImputer | 有相似样本 | 距离加权，自然 | 对 K 值敏感 |

### SimpleImputer（简单插补）

```python
from sklearn.impute import SimpleImputer

# 四种策略
SimpleImputer(strategy="mean")     # 均值（默认）
SimpleImputer(strategy="median")   # 中位数（推荐，鲁棒）
SimpleImputer(strategy="most_frequent")  # 众数（分类变量）
SimpleImputer(strategy="constant", fill_value=0)  # 常数填充

# 添加缺失指示变量
from sklearn.impute import MissingIndicator
indicator = MissingIndicator(features="missing-only")
```

### IterativeImputer（多变量迭代插补）

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    max_iter=10,          # 迭代轮数
    random_state=42,
    estimator=None,       # 默认 BayesianRidge，可换 RandomForestRegressor
    initial_strategy="median",
    imputation_order="roman",   # 按列顺序 / ascending / descending
)
X_imputed = imputer.fit_transform(X)
```

### KNNImputer

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5,
    weights="distance",   # "uniform" 等权 / "distance" 距离加权
    metric="nan_euclidean",
)
X_imputed = imputer.fit_transform(X)
```

---

## 3. 异常值检测与处理

| 方法 | 假设 | 适用 | 不适用 |
|------|------|------|--------|
| IQR | 正态/对称分布 | 快速筛查 | 偏态分布 |
| Z-score | 近似正态 | 简单有效 | 小样本，非正态 |
| Modified Z-score | 任何分布 | 鲁棒 | MAD=0 时失效 |
| Isolation Forest | 异常点稀少且不同 | 高维数据 | 异常比例高 |
| LOF | 局部密度差异 | 局部异常 | 全局异常 |

### IQR 方法

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = (df[col] < lower) | (df[col] > upper)

# Winsorization（截尾替换）
df[col] = df[col].clip(lower=lower, upper=upper)
```

### Z-score 方法

```python
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df[col]))
outliers = z > 3            # 常用阈值 2.5~3.5

# 修正 Z-score（基于 MAD，更鲁棒）
median = df[col].median()
mad = np.median(np.abs(df[col] - median))
modified_z = 0.6745 * (df[col] - median) / mad
outliers = np.abs(modified_z) > 3.5
```

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.05,     # 预期异常比例
    random_state=42,
    n_estimators=100,
)
outlier_labels = iso.fit_predict(X)  # -1=异常, 1=正常
outliers = outlier_labels == -1
```

---

## 4. 变量编码

| 编码方法 | 适用变量类型 | 输出维度 | 备注 |
|---------|------------|---------|------|
| OneHotEncoder | 名义变量（无顺序） | k 列 | 默认稀疏，handle_unknown 处理新类别 |
| OrdinalEncoder | 有序变量 | 1 列 | 需指定 categories 顺序 |
| TargetEncoder | 高基数变量（>50） | 1 列 | 需在 CV 内使用防泄漏 |
| FrequencyEncoder | 高基数变量 | 1 列 | 用频次代替类别 |

### OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(
    handle_unknown="ignore",    # 测试集出现新类别时不报错
    sparse_output=False,        # 返回 dense array
    drop="first",               # 避免虚拟变量陷阱（k-1 列）
)
X_encoded = enc.fit_transform(X_cat)
```

### OrdinalEncoder

```python
from sklearn.preprocessing import OrdinalEncoder

# 需手动指定顺序（否则按字母排序）
enc = OrdinalEncoder(categories=[
    ["low", "medium", "high"],
    ["never", "sometimes", "always"],
])
X_encoded = enc.fit_transform(X_cat)
```

### TargetEncoder（目标编码）

```python
from category_encoders import TargetEncoder

enc = TargetEncoder(
    cols=["high_cardinality_col"],
    handle_missing="value",
    handle_unknown="value",
)
# 注意：必须在训练集上 fit，CV 内用 fold 级编码防泄漏
X_encoded = enc.fit_transform(X_train, y_train)
X_test_encoded = enc.transform(X_test)
```

### Frequency Encoding（高频编码）

```python
freq_map = df["col"].value_counts()
df["col_freq"] = df["col"].map(freq_map)
```

---

## 5. 特征缩放

| 缩放器 | 输出范围 | 中心化 | 异常值鲁棒 | 稀疏友好 |
|--------|---------|--------|-----------|---------|
| StandardScaler | 均值 0，标准差 1 | 是 | 否 | 否 |
| MinMaxScaler | [0, 1] 或 [a, b] | 否 | 否 | 否 |
| RobustScaler | 中位数 0，IQR 1 | 是 | 是 | 否 |
| MaxAbsScaler | [-1, 1] | 否 | 否 | 是 |
| Normalizer | 单位范数 | 否 | — | 是 |
| PowerTransformer | 近似正态 | 是（Yeo-Johnson） | 否 | 否 |
| QuantileTransformer | [0,1] 或正态 | 可配置 | 是 | 否 |

### 算法缩放需求矩阵

| 算法 | 必须缩放 | 原因 |
|------|---------|------|
| SVM / SVR | 是 | 特征尺度影响几何间隔 |
| KNN | 是 | 距离计算 |
| PCA / LDA | 是 | 方差最大化受尺度影响 |
| Neural Network | 是 | 梯度更新收敛 |
| Linear / Logistic | 推荐 | 系数可比，正则化作用一致 |
| Tree (RF, GBDT, XGBoost) | 否 | 基于分裂点，尺度不变 |
| Naive Bayes | 否 | 概率计算，尺度不变 |

### 代码示例

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, Normalizer, MaxAbsScaler
)

# Z-score 标准化（默认）
StandardScaler()

# 区间缩放
MinMaxScaler(feature_range=(0, 1))

# 稳健缩放（基于中位数和 IQR）
RobustScaler(quantile_range=(25.0, 75.0))

# 正态化变换
PowerTransformer(method="yeo-johnson")   # 自动处理正值/负值
PowerTransformer(method="box-cox")       # 仅处理正值

# 分位数变换
QuantileTransformer(n_quantiles=1000, output_distribution="normal")

# 单位范数
Normalizer(norm="l2")   # l1 / l2 / max

# 稀疏保留
MaxAbsScaler()
```

---

## 参考

- [scikit-learn 预处理文档](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn 缺失值处理](https://scikit-learn.org/stable/modules/impute.html)
- [category_encoders 文档](https://contrib.scikit-learn.org/category_encoders/)
