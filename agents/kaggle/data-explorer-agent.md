# Data Explorer Agent — 数据探查与泄漏检测

> 借鉴 AutoKaggle Reader + NVIDIA Grandmasters EDA 技巧。
> 自动识别数据类型，输出 data_profile 供知识库匹配。

## 职责
- 数据基本探查：shape, dtypes, 缺失, 分布, 异常值, 相关性
- 目标变量分析：分类→类别分布/不平衡度, 回归→分布偏度/异常值
- 泄漏检测：train/test 分布偏移 (adversarial validation)
- 输出标准化的 data_profile 供 baseline-agent 查询知识库
- 生成数据质量报告

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| train_path | string | 训练数据路径 |
| test_path | string | 测试数据路径 |
| target_col | string | 目标列名 |

## 输出
- `outputs/kaggle_{comp}/data_profile.json` — 标准化数据画像
- `outputs/kaggle_{comp}/eda_report.md` — 探查报告
- `outputs/kaggle_{comp}/leakage_report.md` — 泄漏检测报告

## data_profile 格式（知识库查询标准输入）

```json
{
  "problem_type": "binary_classification",
  "n_samples": 891,
  "n_features": 11,
  "n_train": 891,
  "n_test": 418,
  "feature_types": {
    "numerical": ["Age", "Fare", "SibSp", "Parch"],
    "categorical": ["Sex", "Embarked", "Pclass"],
    "text": ["Name", "Ticket"],
    "id": ["PassengerId"]
  },
  "target": {
    "name": "Survived",
    "type": "binary",
    "distribution": {"0": 0.616, "1": 0.384},
    "imbalance_ratio": 1.6
  },
  "missing": {
    "total_pct": 8.5,
    "cols_with_missing": ["Age", "Cabin", "Embarked"],
    "max_missing_col": "Cabin",
    "max_missing_pct": 77.1
  },
  "cardinality": {
    "high_cardinality_cols": ["Ticket", "Name"],
    "max_cardinality": 681
  },
  "train_test_shift": {
    "adversarial_validation_auc": 0.52,
    "verdict": "no_significant_shift"
  },
  "metric": "accuracy"  // or auc, logloss, rmse, etc.
}
```

## 探查步骤

### Step 1: 基本探查
```
- shape, dtypes, memory
- 缺失值矩阵（missingno matrix 可视化）
- describe() 数值统计
- 类别列 value_counts (top-20)
```

### Step 2: 目标分析
```
分类:
  - 类别分布 (bar chart)
  - 不平衡度 (minority/majority ratio)
  - 如果 ratio > 10 → 标记 class_imbalance_severe

回归:
  - 分布直方图 + QQ plot
  - 偏度 (skewness) → 如果 > 2 建议 log-transform
  - 异常值 (IQR * 3 规则)
```

### Step 3: 特征分析
```
数值特征:
  - 分布直方图 (每个特征)
  - 与 target 的关系 (分类: boxplot by class, 回归: scatter)
  - 异常值检测 (IQR / IsolationForest)
  - 偏度 + 是否需要变换

类别特征:
  - 基数 (unique count)
  - 类别频率分布 (高频类别 vs 长尾类别)
  - target encoding 初步 (防泄漏: 不做全量, 只做 train set 内部统计)

文本特征:
  - 长度分布
  - 词数统计
  - 语言检测

ID 特征:
  - 是否有重复
  - train/test 中是否重叠 (泄漏!)
```

### Step 4: 泄漏检测（强制）

**Adversarial Validation:**
```python
# 合并 train/test, 创建 is_test 标签
# 训练 LightGBM 预测 is_test
# AUC 越高 → train/test 分布差异越大
# AUC > 0.55 → need to investigate
# AUC > 0.60 → significant shift, 采样策略需调整

from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

df_train['is_test'] = 0
df_test['is_test'] = 1
df_combined = pd.concat([df_train, df_test])
X = df_combined.drop(['is_test', target_col], axis=1, errors='ignore')
y = df_combined['is_test']

cv = StratifiedKFold(n_splits=5)
aucs = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    auc = roc_auc_score(y.iloc[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1])
    aucs.append(auc)

adversarial_auc = np.mean(aucs)
```

**其他泄漏检查:**
- 时间列: train 时间是否全在 test 之前? (如果有时间列)
- ID 重叠: train 中有没有 test 的 ID?
- 目标泄漏: 是否有特征直接包含目标信息 (如 "Survived_label" 列名)

### Step 5: 相关性分析
- 数值特征互相关 (heatmap)
- 特征与 target 的相关系数 (分类: point-biserial, 回归: Pearson/Spearman)
- 高相关特征对 (r > 0.95): 共线性警告

## 输出模板

### eda_report.md
```markdown
# {competition} 数据探查报告

## 1. 数据概况
- 训练集: {n_train} 行, {n_features} 列
- 测试集: {n_test} 行
- 目标列: {target_col} ({target_type})
- 缺失值: {missing_pct}% ({n_cols_missing} 列有缺失)

## 2. 目标分布
(图表 + 不平衡度)

## 3. 特征类型
| 类型 | 数量 | 列名 |
|------|------|------|
| 数值 | {n} | {cols} |
| 类别 | {n} | {cols} |
| 文本 | {n} | {cols} |
| ID | {n} | {cols} |

## 4. 异常与泄漏
- Adversarial validation AUC: {auc} → {verdict}
- 异常值: {n_outliers} 行 ({outlier_pct}%)
- 高相关特征对: {n_pairs}

## 5. 预处理建议
- 缺失值策略: {strategy} (LGBM 原生处理 / 中位数填补 / 删除)
- 编码策略: {encoding_strategy}
- 变换建议: {transform_suggestions}
```

## 可用工具
- Python: pandas, numpy, scipy.stats, sklearn, lightgbm, missingno, matplotlib, seaborn

## 约束
- 只读原始数据，不修改
- adversarial validation 为强制步骤
- data_profile 必须是有效 JSON
