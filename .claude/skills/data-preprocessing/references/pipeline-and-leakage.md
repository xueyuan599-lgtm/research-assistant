# 流水线构建与数据泄漏防范

> 涵盖 Pipeline/ColumnTransformer/FeatureUnion 的正确用法，以及 5 种常见数据泄漏模式的错误 vs 正确代码对比。

---

## 1. Pipeline 基础

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 方式一：命名步骤
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression()),
])

# 方式二：make_pipeline（自动命名）
pipe = make_pipeline(StandardScaler(), PCA(10), LogisticRegression())

# 访问步骤
pipe.named_steps["scaler"]
pipe[:-1]  # 子流水线（去掉最后一步）
```

---

## 2. ColumnTransformer（混合类型预处理）

数值列和分类列分别做不同预处理。

```python
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 列选择方式
col = ["age", "income"]           # 名称列表
col = [0, 1, 2]                   # 列索引
col = [True, False, True, ...]    # boolean mask
col = lambda df: df.columns[:3]   # 可调用对象

# 官方方式
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ]), ["age", "income"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["gender", "city"]),
    ],
    remainder="drop",          # 未指定列的处理方式
    # drop=删除, passthrough=保留, estimator=用另一个变换器
)

# 快捷方式
preprocessor = make_column_transformer(
    (StandardScaler(), ["age", "income"]),
    (OneHotEncoder(), ["gender", "city"]),
    remainder="passthrough",
)

# 获取输出列名
preprocessor.set_output(transform="pandas")
X_transformed = preprocessor.fit_transform(X)
print(X_transformed.columns.tolist())
```

---

## 3. FeatureUnion（并行特征提取）

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

union = FeatureUnion([
    ("pca", PCA(n_components=5)),
    ("kbest", SelectKBest(k=10)),
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("union", union),
    ("clf", LogisticRegression()),
])
```

---

## 4. TransformedTargetRegressor

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer

# 对目标变量做变换，预测时自动逆变换
ttr = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=QuantileTransformer(n_quantiles=100,
                                    output_distribution="normal"),
)
ttr.fit(X_train, y_train)
y_pred = ttr.predict(X_test)  # 自动逆变换回原始尺度
```

---

## 5. 参数持久化

```python
import joblib

# 保存完整流水线
joblib.dump(pipe, "full_pipeline.pkl")

# 加载推理
pipe_loaded = joblib.load("full_pipeline.pkl")
y_pred = pipe_loaded.predict(X_new)

# 只保存预处理部分
preprocessor = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scl", StandardScaler()),
])
preprocessor.fit(X_train)
joblib.dump(preprocessor, "preprocessor.pkl")
```

---

## 6. 数据泄漏 5 种模式

### 模式 1：标准缩放泄漏

```python
# 错误：对整个数据集 fit 后拆分
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)       # 泄漏！测试集信息污染训练集
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# 正确：先拆分
X_train, X_test = train_test_split(X, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 仅在训练集 fit
X_test_scaled = scaler.transform(X_test)          # 只用 transform
```

### 模式 2：目标编码泄漏

```python
# 错误：用整个数据集的目标均值编码
enc = TargetEncoder()
X_encoded = enc.fit_transform(X, y)      # 泄漏！

# 正确：在 CV fold 内计算
# -- 用 Pipeline + cross_val_score 自动处理 --
from sklearn.model_selection import cross_val_score
pipe = Pipeline([("enc", TargetEncoder()), ("clf", XGBClassifier())])
scores = cross_val_score(pipe, X, y, cv=5)  # 每折内分别 fit

# 或手动拆分
X_train, X_test = train_test_split(X, test_size=0.2)
enc = TargetEncoder(cols=["cat_col"])
X_train_enc = enc.fit_transform(X_train, y_train)
X_test_enc = enc.transform(X_test)
```

### 模式 3：特征选择泄漏

```python
# 错误：在 CV 外部选择特征
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # 泄漏！
scores = cross_val_score(clf, X_selected, y, cv=5)

# 正确：将特征选择放在 Pipeline 内
pipe = Pipeline([
    ("select", SelectKBest(k=10)),
    ("clf", LogisticRegression()),
])
scores = cross_val_score(pipe, X, y, cv=5)
```

### 模式 4：时序未来数据泄漏

```python
# 错误：在整个序列上计算滚动统计量
df["roll_mean"] = df["value"].rolling(7).mean()   # t 时刻用了 t-6 ~ t 的数据

# 正确：shift 后再 rolling
df["roll_mean"] = df["value"].shift(1).rolling(7).mean()  # 只使用 t-7 ~ t-1

# 正确交叉验证
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # 每折只从历史数据学习
```

### 模式 5：重复数据泄漏

```python
# 错误：train/test 包含重复样本
df = pd.concat([train_df, test_df])
df = df.drop_duplicates()  # 隐含泄漏！

# 正确：分别去重
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()
```

---

## 7. 完整安全 Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. 读取
df = pd.read_csv("data.csv")
X, y = df.drop("target", axis=1), df["target"]

# 2. 拆分（第一步！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 预处理
num = X.select_dtypes(include=np.number).columns.tolist()
cat = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

# 4. 完整流水线（预处理 + 模型）
pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42)),
])

# 5. 训练与评估
pipe.fit(X_train, y_train)
print(f"Train: {pipe.score(X_train, y_train):.3f}")
print(f"Test:  {pipe.score(X_test, y_test):.3f}")

# 6. 交叉验证（安全！Pipeline 自动防泄漏）
scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"CV: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 7. 保存
import joblib
joblib.dump(pipe, "safe_pipeline.pkl")
```

---

## 8. 泄漏排查清单

| 检查点 | 确认 |
|--------|------|
| train_test_split 在所有预处理之前执行 | [ ] |
| StandardScaler / MinMaxScaler 只在训练集 fit | [ ] |
| 目标编码（TargetEncoder）在 CV fold 内计算 | [ ] |
| 特征选择（SelectKBest / RFE）在 Pipeline 内 | [ ] |
| 时间序列滚动统计量使用 shift(1) 后再计算 | [ ] |
| train/test 无重复行 | [ ] |
| Pipeline 中所有步骤都有 transform 方法 | [ ] |
| ColumnTransformer 的 remainder 指定正确 | [ ] |

---

## 参考

- [scikit-learn 流水线文档](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn  ColumnTransformer 文档](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [scikit-learn 数据泄漏指南](https://scikit-learn.org/stable/common_pitfalls.html)
