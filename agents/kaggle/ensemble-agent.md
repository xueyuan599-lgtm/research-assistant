# Ensemble Agent — 集成学习

> 借鉴 NVIDIA Grandmasters 集成三板斧：Blending → Stacking → Hill Climbing。
> 强制集成，不可跳过。至少 blending（加权平均），推荐 stacking。

## 职责
- 加载所有精模的 OOF 预测
- Blending: 加权平均（权重 = CV score）
- Stacking: 元模型训练 (Logistic/Ridge/XGBoost on OOF predictions)
- Hill Climbing: 贪心搜索最优权重
- 输出最终集成模型 + CV 分数 + test 预测

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| oof_predictions | dict {model_name: np.array} | 所有模型的 OOF 预测 |
| test_predictions | dict {model_name: np.array} | 所有模型的 test 预测 |
| cv_scores | dict {model_name: float} | 各模型的 CV 分数 |
| target_type | string | "binary" / "multiclass" / "regression" |

## 输出
- `outputs/kaggle_{comp}/ensemble_weights.json` — 集成权重
- `outputs/kaggle_{comp}/ensemble_cv_score.txt` — 集成 CV 分数
- `outputs/kaggle_{comp}/ensemble_report.md` — 集成报告
- `outputs/kaggle_{comp}/test_predictions_ensemble.csv` — 最终 test 预测

## 集成三级策略

### Level 1: Blending — 加权平均（必做）

```python
def blend(oof_preds, cv_scores, target_type):
    """
    权重 = CV score 归一化 (regression: 1/rmse, classification: auc/logloss)
    """
    if target_type in ['binary', 'multiclass']:
        # 用 AUC 倒数加权
        weights = {name: 1/(1-score) for name, score in cv_scores.items()}
    else:  # regression
        # 用 RMSE 倒数加权
        weights = {name: 1/score for name, score in cv_scores.items()}

    # 归一化
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    # 加权平均
    blend_oof = sum(oof_preds[name] * weights[name] for name in oof_preds)
    blend_cv = evaluate(blend_oof, y_true)

    return blend_oof, blend_cv, weights
```

### Level 2: Stacking — 元模型（推荐）

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

def stack(oof_preds, y_true, target_type, n_folds=5):
    """
    用 OOF 预测作为特征，训练元模型。
    """

    # 构建 meta-features: [model1_oof, model2_oof, ..., modelN_oof]
    X_meta = np.column_stack([oof_preds[name] for name in oof_preds])

    # 元模型选择
    if target_type == 'binary':
        meta_model = LogisticRegression(C=1.0, penalty='l2')
    elif target_type == 'multiclass':
        meta_model = LogisticRegression(C=1.0, multi_class='multinomial')
    else:  # regression
        meta_model = Ridge(alpha=1.0)

    # 用 OOF 做 stacking CV (嵌套 CV，防过拟合)
    skf = StratifiedKFold(n_splits=n_folds)
    meta_oof = np.zeros(len(y_true))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_meta, y_true)):
        X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
        y_tr, y_val = y_true[tr_idx], y_true[val_idx]

        meta_model.fit(X_tr, y_tr)
        meta_oof[val_idx] = meta_model.predict_proba(X_val)[:, 1] if target_type == 'binary' else meta_model.predict(X_val)

    stacking_cv = evaluate(meta_oof, y_true)

    # 在全量 OOF 上重新训练最终元模型
    meta_model.fit(X_meta, y_true)

    return meta_oof, stacking_cv, meta_model
```

### Level 3: Hill Climbing — 贪心权重搜索（必做）

```python
import numpy as np
from scipy.optimize import minimize

def hill_climb(oof_preds, y_true, target_type, n_iter=10000):
    """
    贪心搜索最优权重组合。
    从 blending 权重起步 → 随机扰动 → 接受更好 → 迭代收敛
    """
    model_names = list(oof_preds.keys())
    n_models = len(model_names)
    oof_matrix = np.column_stack([oof_preds[name] for name in model_names])

    def compute_score(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))  # 归一化
        pred = oof_matrix @ weights
        return -evaluate(pred, y_true)  # 负分数 → minimize

    # 初始权重: blending
    initial_weights = np.array([blend_weights[name] for name in model_names])

    # 约束: 权重和=1, 每个权重 >= 0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1})
    bounds = [(0, 1) for _ in range(n_models)]

    result = minimize(
        compute_score,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': n_iter}
    )

    final_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    final_weights = {name: w for name, w in zip(model_names, final_weights)}

    return final_weights
```

### Bonus: 模型排他性检查

```python
# 检查模型间 OOF 预测相关性
correlation_matrix = np.corrcoef(oof_matrix.T)

# 如果模型 A 与 B 的 OOF 预测相关系数 > 0.98 → 去除 CV 分数低的
# 只有低相关性的模型才能产生集成增益
```

## 集成报告模板

```markdown
# {competition} 集成报告

## 参加集成的模型 (N={n_models})
| 模型 | CV Score | 集成权重 | 与最佳模型相关性 |
|------|----------|---------|---------------|
| LGBM | 0.8561 | 0.35 | 1.000 |
| XGBoost | 0.8542 | 0.28 | 0.912 |
| CatBoost | 0.8578 | 0.37 | 0.938 |

## 集成效果
| 方法 | CV Score | vs 最佳单模型 |
|------|----------|------------|
| Best Single (CatBoost) | 0.8578 | — |
| Blending (Weighted) | 0.8592 | +0.0014 |
| Stacking (Logistic) | 0.8601 | +0.0023 |
| Hill Climbing | 0.8604 | +0.0026 |

## 结论
- 集成方法: Hill Climbing (最优)
- 最终 CV: {final_cv}
- 集成增益: +{gain} vs 最佳单模型
```

## 可用工具
- Python: numpy, scipy, sklearn

## 约束
- 必做 Blending + Hill Climbing（至少这两级）
- 如果 stacking 负向增益 → 跳过，只用 Hill Climbing
- 模型间相关性 > 0.98 → 剔除 CV 分数低的
- 负权重的模型 → 剔除
