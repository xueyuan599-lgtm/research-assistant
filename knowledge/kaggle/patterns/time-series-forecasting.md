# Pattern: time-series-forecasting

## 匹配条件

| 维度 | 值 |
|------|-----|
| problem_type | time_series_forecasting |
| 数据特征 | 含时间戳列、多序列可能 |
| 频率 | 日/周/月/小时/分钟 |
| metric | rmse / mae / smape / mase |

## 问题子类型

| 子类型 | 说明 | 首选方法 |
|--------|------|---------|
| 单序列预测 | 一个序列 → 未来 N 步 | ARIMA, Prophet, TiDE |
| 多序列预测 | 多个独立序列 | LightGBM (lag features), PatchTST, TimesFM |
| 分层预测 | 层级关系（店→区→国） | LightGBM + hierarchical reconciliation |
| 全局预测 | 序列间共享模式 | PatchTST, N-BEATS, TimesFM |
| 含协变量 | 外部变量影响 | LightGBM (lag + covariates), N-BEATSx |
| 概率预测 | 需要分位数 | LightGBM quantile, CatBoost quantile |

## 推荐算法栈

### Tier 1: 传统方法（必做基线）
1. **Naive / Seasonal Naive** — 最低基线
2. **ARIMA / SARIMA** (statsmodels / pmdarima) — 单序列
3. **ETS** (statsmodels) — 平滑方法

### Tier 2: 机器学习
4. **LightGBM** — lag features + rolling stats + datetime features
   - 配置: `objective='regression'`
   - 特征: tsfresh 提取 794 种时序特征
   - 滑窗: sktime `WindowSummarizer` (lag/lead/rolling mean/std)
5. **XGBoost** — 同上框架

### Tier 3: 深度学习（大样本时）
6. **PatchTST** (HuggingFace) — SOTA 全局预测
7. **TimesFM** (Google) — 基础模型，zero-shot 预测
8. **N-BEATS** — 纯 MLP 架构，M4 冠军

### Tier 4: 概率预测
9. **LightGBM** — `objective='quantile', alpha=0.1/0.5/0.9`
10. **CatBoost** — `loss_function='Quantile'`

## 时序特有特征工程（tsfresh / sktime）

| 类别 | 示例 |
|------|------|
| 滞后特征 | lag-1, lag-7, lag-30 |
| 滑窗统计 | rolling_mean_7, rolling_std_14, rolling_min/max |
| 时间特征 | year, month, dayofweek, hour, is_weekend, quarter |
| 频域特征 | fft_coefficient, spectral_centroid |
| 差分特征 | diff_1, diff_7, diff_30 |
| tsfresh 特征 | 794 种自动提取特征 |

## CV 策略

**时间序列 CV 必须保持顺序，严禁随机 shuffle！**
- Expanding window: 逐步扩大训练窗口
- Sliding window: 固定大小窗口滑动
- 实现: sklearn `TimeSeriesSplit(n_splits=5)`

## 已知成功案例

| 竞赛 | 最佳排名 | 核心方法 |
|------|---------|---------|
| M5 Forecasting | top 1% | LightGBM + global model + reconciliation |
| Walmart Sales | top 3% | LightGBM + lag features + tsfresh |
| Web Traffic | top 5% | LSTM + attention |

## 参考来源

- M5 Competition 获奖方案
- tsfresh 文档
- TimesFM (Google Research, 2024)
- PatchTST (ICLR 2023)
