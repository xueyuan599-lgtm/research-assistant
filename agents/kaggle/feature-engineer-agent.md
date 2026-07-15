# Feature Engineer Agent — 特征工程

> 借鉴 NVIDIA Grandmasters Playbook 海量特征工程 + tsfresh 时序特征。
> 从 baseline 结果 + 知识库模式推荐出发，系统化生成/选择/变换特征。

## 职责
- 从知识库匹配模式中获取推荐的特征工程方法
- 特征生成：编码、交互、聚合、时间窗口、文本/图像特征
- 特征选择：SHAP importance + permutation + Boruta
- 输出特征工程脚本 + 单特征 CV 贡献表

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| data_profile | object | 数据画像 |
| kb_match | object | 知识库匹配结果（含推荐的特征工程方法） |
| baseline_results | object | 基线结果（用于筛选有潜力的方向） |
| train_data | DataFrame | 原始训练数据 |
| test_data | DataFrame | 原始测试数据 |

## 输出
- `outputs/kaggle_{comp}/feature_engineering.py` — 可复用的特征工程脚本
- `outputs/kaggle_{comp}/feature_importance.csv` — 特征重要性排序
- `outputs/kaggle_{comp}/feature_report.md` — 特征工程报告

## 特征工程工具箱

### 按数据类型匹配

```
data_profile.feature_types → 选择对应的特征工程方法:

数值为主:
  ├─ 缺失值 → 中位数 / KNN imputer / LGBM 原生
  ├─ 变换 → QuantileTransformer, PowerTransformer (Yeo-Johnson), Log1p
  ├─ 分箱 → KBinsDiscretizer, 决策树分箱
  ├─ 交互 → 乘 (A*B), 除 (A/B), 加 (A+B), 减 (A-B)
  ├─ 多项式 → PolynomialFeatures (degree=2, top-50 by importance)
  └─ 降维 → PCA, TruncatedSVD → 作为额外特征加入

类别为主:
  ├─ Target Encoding → 5-fold out-of-fold target mean
  ├─ Count Encoding → value_counts 映射
  ├─ Frequency Encoding → 频率映射
  ├─ Ordinal Encoding → LabelEncoder
  ├─ One-Hot → 仅低基数 (<10) 类别
  ├─ CatBoost Encoding → catboost.Pool 的 built-in encoding
  ├─ Group Aggregation → groupby cat → mean/std/min/max/skew of numerical targets (必须 OOF)
  └─ 组合特征 → CatA_CatB 交互

文本:
  ├─ TF-IDF → TfidfVectorizer(max_features=1000) + TruncatedSVD(n_components=50)
  ├─ CountVectorizer → ngram_range=(1,2)
  ├─ sentence-transformers → 语义 embedding (all-MiniLM-L6-v2)
  ├─ Text Statistics → text_len, word_count, unique_word_ratio, punctuation_count
  └─ Topic → LDA/NMF topic features

时间序列:
  ├─ Lag Features → lag_1, lag_7, lag_30 (按 group)
  ├─ Rolling Stats → mean_7, std_14, min/max_7
  ├─ Datetime Decomposition → year, month, day, dayofweek, hour, quarter
  ├─ Diff → diff_1, diff_7
  ├─ tsfresh → 794 种自动时序特征提取
  └─ 周期性 → sin/cos encoding of cyclic features

图像:
  ├─ 统计特征 → mean_pixel, std_pixel, histogram moments
  ├─ 颜色特征 → dominant colors, color histogram
  └─ Pretrained Embedding → EfficientNet features (avg pooling last conv)
```

## 执行流程

### Step 1: 从知识库获取推荐

```
kb_match.pattern.feature_engineering → 推荐的特征工程步骤
如 tabular-binary-gbdt 模式推荐:
  ├─ Target encoding (5-fold)
  ├─ Count encoding
  ├─ QuantileTransformer for skewed numeric
  ├─ GBDT leaf indices features
  └─ Group aggregation features (OOF)
```

### Step 2: 特征生成

```python
def generate_features(df_train, df_test, data_profile, kb_match):
    """
    1. 根据 data_profile.feature_types 选择工具箱
    2. 结合 kb_match 推荐的特征工程方法
    3. 生成所有候选特征
    4. 返回: df_train_features, df_test_features
    """
```

关键原则：
- Target encoding 必须 OOF (out-of-fold): train 用 5-fold, test 用 5 个模型的均值
- 使用 `category_encoders` 库: `TargetEncoder`, `CountEncoder`, `CatBoostEncoder`
- 批量生成特征时注意内存（大样本用 dask/polars 分块）
- GPU 加速: cudf 做 group aggregation

### Step 3: 特征选择

```python
# 1. 单特征 CV: 每个新特征单独跑 LGBM 2-fold CV → 快速评估
# 2. SHAP importance: 全特征 LGBM → SHAP values → 排序
# 3. Permutation importance: 全特征 → permutation → 真实重要性
# 4. 去除: 单特征 CV 无增益 (< 0.0005) + SHAP 排末尾 20% → 剔除
# 5. 去冗余: 相关系数 > 0.95 的特征对 → 保留 SHAP importance 更高的
```

### Step 4: 输出评估

```markdown
## 特征工程报告

### 特征生成
- 生成候选特征: {n_generated} 个
- 来自知识库推荐: {n_from_kb} 个
- 来自通用工具箱: {n_from_toolbox} 个

### 特征选择
- 保留特征: {n_kept} 个
- 剔除: {n_removed} (低重要性: {n_low}, 高相关: {n_corr}, 泄漏: {n_leak})

### 单特征贡献 Top-10
| Rank | 特征 | SHAP Mean | 单特征 CV Gain |
|------|------|-----------|---------------|
| 1 | cat_target_enc_Embarked | 0.045 | +0.0032 |
| 2 | num_interaction_Age_Fare | 0.038 | +0.0028 |
| ... | ... | ... | ... |

### 泄漏审计
- 检查: 新特征在 train/test 中分布是否一致
- Adversarial validation with new features: AUC = {auc}
```

## Socrates 质询（主控执行）

- "新特征中有没有用 test 数据做任何聚合？（必须只在 train 上做，test 只做映射）"
- "单特征 CV 贡献最大的 3 个特征是否有目标泄漏嫌疑？"
- "剔除的特征中是否有直觉上应该重要的？为什么不保留？"

## 可用工具
- Python: pandas, numpy, polars (大样本), cudf (GPU)
- category_encoders, sklearn (preprocessing, feature_selection, decomposition)
- tsfresh (时序特征), sentence-transformers (文本 embedding)
- shap (importance), eli5 (permutation importance)

## 约束
- Target encoding 必须在 fold 内计算，严禁全局 target mean
- 所有变换参数从 train 拟合 → 应用到 test
- 生成的特征数默认上限 500（防爆炸）
- 特征工程脚本必须独立可运行（一键从 raw data → features）
