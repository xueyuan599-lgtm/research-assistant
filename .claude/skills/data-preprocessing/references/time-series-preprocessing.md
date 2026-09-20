# 时间序列预处理参考

> 涵盖时间序列数据的缺失值智能插补、平稳化、平滑与分解、重采样、日期特征提取。

---

## 1. 时间序列基础处理

### 时间索引校验

```python
import pandas as pd

ts = pd.read_csv("sensor.csv", index_col="time", parse_dates=True)

# 检查是否为等间隔
inferred = pd.infer_freq(ts.index)
print(f"Inferred frequency: {inferred}")

# 检查断点
full_range = pd.date_range(ts.index.min(), ts.index.max(), freq=inferred)
missing_idx = full_range.difference(ts.index)
print(f"Missing timestamps: {len(missing_idx)}")

# 检查重复索引
print(f"Duplicate indices: {ts.index.duplicated().sum()}")
```

### 重采样（更改时间频率）

```python
# 降采样（聚合）
ts_down = ts.resample("1h").mean()        # 每小时均值
ts_down = ts.resample("1D").ffill()       # 每日，向前填充

# 升采样（插值）
ts_up = ts.resample("5min").interpolate(method="linear")
ts_up = ts.resample("1min").interpolate(method="cubic")
```

### 多序列对齐到同一时间轴

```python
# 合并多序列后前向填充
aligned = pd.concat([series1, series2], axis=1, join="outer")
aligned = aligned.sort_index().ffill()
```

---

## 2. 缺失值填补（PyPOTS 神经方法）

PyPOTS 提供三种 SOTA 神经网络插补器，尤其适合不规则采样和长时间段缺失。

### SAITS（自注意力插补）

```python
from pypots.imputation import SAITS
import numpy as np

# PyPOTS 要求 3D 输入 [n_samples, n_steps, n_features]
# 对单变量时序：reshape(-1, 1, 1)
X = ts["value"].values.reshape(-1, 1, 1)

saits = SAITS(
    n_steps=X.shape[0],       # 时间步数
    n_features=1,             # 变量维度
    n_layers=2,               # 注意力层数
    d_model=64,               # 注意力维度
    d_inner=128,              # FFN 隐藏维度
    n_head=4,                 # 注意力头数
    dropout=0.1,
    learning_rate=1e-3,
    epochs=50,
    batch_size=32,
)
saits.fit(X)
X_imputed = saits.impute(X)  # 同上 shape [n, 1, 1]
```

### BRITS（双向 RNN 插补）

```python
from pypots.imputation import BRITS

brits = BRITS(
    n_steps=X.shape[0],
    n_features=1,
    rnn_hidden_size=64,
    learning_rate=1e-3,
    epochs=50,
)
brits.fit(X)
X_imputed = brits.impute(X)
```

### CSDI（分数扩散插补）

```python
from pypots.imputation import CSDI

csdi = CSDI(
    n_steps=X.shape[0],
    n_features=1,
    n_layers=4,
    n_heads=4,
    learning_rate=1e-3,
    epochs=100,
)
csdi.fit(X)
X_imputed = csdi.impute(X)
```

### 方法对比

| 方法 | 优势 | 劣势 | 推荐场景 |
|------|------|------|---------|
| SAITS | 并行计算，全局依赖捕捉 | 需足够训练样本 | 长序列、高缺失率 |
| BRITS | 双向时序建模 | 顺序计算，较慢 | 短序列、强时序依赖 |
| CSDI | 生成式分布建模 | 训练最慢 | 极不规则采样 |
| 线性插值 | 极快，无训练 | 忽略全局模式 | 短空缺、规则采样 |

### 传统插值方法

```python
# pandas 内置插值
ts["value"].interpolate(method="linear")       # 线性
ts["value"].interpolate(method="quadratic")     # 二次
ts["value"].interpolate(method="spline", order=3)  # 样条
ts["value"].fillna(method="ffill")              # 向前填充（LOCF）
ts["value"].fillna(method="bfill")              # 向后填充

# sktime 插值
from sktime.transformations.series.impute import Imputer
Imputer(method="linear")      # 支持 linear / ffill / bfill / drift / seasonal
```

---

## 3. 平稳化与变换

### 差分

```python
from sktime.transformations.series.difference import Differencer

# 一阶差分（去除趋势）
Differencer(lags=1)

# 季节差分（去除季节成分）
Differencer(lags=12)    # 月度数据的年度周期
Differencer(lags=4)     # 季度数据的年度周期

# 多阶差分
Differencer(lags=[1, 12])  # 同时一阶 + 季节差分
```

### 方差稳定变换

```python
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.exponent import ExponentTransformer

BoxCoxTransformer()                           # Box-Cox（需正值）
BoxCoxTransformer(sp=12, method="mle")         # 含季节成分
ExponentTransformer(power=0.5)                # 平方根变换
ExponentTransformer(power=0.0, bias=True)     # 对数变换（power=0 = log）
```

### 平稳性检验

```python
from sktime.utils.validation.series import StationarityTests
from statsmodels.tsa.stattools import adfuller, kpss

# ADF 检验（H0: 存在单位根，即非平稳）
adf_stat, adf_pval, _, _, crit_values, _ = adfuller(ts["value"].dropna())
print(f"ADF p-value: {adf_pval:.4f}")
# p < 0.05 → 拒绝 H0 → 平稳

# KPSS 检验（H0: 趋势平稳）
kpss_stat, kpss_pval, _, _ = kpss(ts["value"].dropna(), regression="c")
print(f"KPSS p-value: {kpss_pval:.4f}")
# p < 0.05 → 拒绝 H0 → 非平稳
```

---

## 4. 平滑与分解

### STL 分解

```python
from statsmodels.tsa.seasonal import STL

stl = STL(ts["value"], period=12)   # 月度数据周期=12
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
residual = res.resid
```

### Hampel 滤波器（异常值检测）

```python
from sktime.transformations.series.hampel import HampelFilter

hampel = HampelFilter(
    window_length=10,     # 滑动窗口大小
    n_sigma=3.0,          # 标准差倍数阈值
    return_bool=False,    # True=返回 bool mask, False=替换异常值
)
ts_clean = hampel.fit_transform(ts)
```

---

## 5. 日期与时间特征

### 自动日期特征提取

```python
from sktime.transformations.series.date import DateTimeFeatures

# 自动提取：年、月、周、日、季度、星期几、是否周末等
dt_feat = DateTimeFeatures(ts_freq="D", feature_set="all")
date_features = dt_feat.fit_transform(ts)

# feature_set 选项
# "all"     — 所有可用特征
# "holiday" — 仅节假日特征
# "comprehensive" — 大部分常见特征

# 自定义特征
from sktime.transformations.series.summarize import WindowSummarizer
summarizer = WindowSummarizer(
    lag_feature={
        "lag": [1, 2, 3, 6, 12],
        "mean": [[1, 3], [1, 6]],
        "std": [[1, 4]],
    }
)
```

### 手动特征构造

```python
# 基础日期特征
ts["year"] = ts.index.year
ts["month"] = ts.index.month
ts["weekday"] = ts.index.weekday
ts["quarter"] = ts.index.quarter
ts["is_weekend"] = ts.index.weekday >= 5
ts["day_of_year"] = ts.index.dayofyear

# 滞后值
ts["lag_1"] = ts["value"].shift(1)
ts["lag_7"] = ts["value"].shift(7)

# 滚动统计量
ts["roll_mean_7"] = ts["value"].rolling(7).mean()
ts["roll_std_7"] = ts["value"].rolling(7).std()
ts["roll_max_7"] = ts["value"].rolling(7).max()
```

---

## 6. 交叉验证注意事项

```python
from sklearn.model_selection import TimeSeriesSplit

# TimeSeriesSplit：只使用历史数据训练，未来数据验证
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # 预处理参数必须在每折的训练集内 fit

# ❌ 禁止使用未来数据
# rolling(7).mean() 在计算 t 时刻时，只能用 t-7 ~ t-1 的数据
# ✅ 正确：shift(1) 后再 rolling
df["safe_roll"] = df["value"].shift(1).rolling(7).mean()
```

---

## 参考

- [PyPOTS 文档](https://pypots.readthedocs.io/)
- [sktime 变换器](https://www.sktime.org/en/stable/api_reference/transformations.html)
- [statsmodels 时间序列](https://www.statsmodels.org/stable/tsa.html)
