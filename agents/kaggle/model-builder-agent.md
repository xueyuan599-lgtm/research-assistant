# Model Builder Agent — 精模构建与调参

> 借鉴 OpenDataSci 自审查迭代 + PiML 自适应记忆。
> **核心创新：超参搜索从知识库推荐的参数范围起步，而非随机初始化。**
> 使用 Optuna 进行 Bayesian optimization，结合先验知识加速收敛。

## 职责
- 接收 baseline Top-3 方向 + 知识库推荐的超参范围
- 对每个方向进行 Optuna 超参搜索
- OOF 预测 + 伪标签 (pseudo-labeling)
- 外部数据扩充（如适用）
- 输出精模 + 最优参数 + CV 分数 + OOF 预测

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| top3_directions | list[object] | baseline 推荐的 Top-3 模型方向 |
| kb_match | object | 知识库匹配结果（含推荐的超参范围） |
| train_features | DataFrame | 特征工程后的训练数据 |
| test_features | DataFrame | 特征工程后的测试数据 |
| cv_strategy | object | CV 切分方案 |
| n_trials_per_model | int | Optuna trials per model，默认 100 |

## 输出
- `outputs/kaggle_{comp}/models/` — 精模权重/配置
- `outputs/kaggle_{comp}/oof_predictions.csv` — OOF 预测（供集成用）
- `outputs/kaggle_{comp}/tuning_log.json` — 调参日志
- `outputs/kaggle_{comp}/model_report.md` — 精模报告

## 执行流程

### Step 1: 知识库驱动的超参初始化

```python
# ❌ 不用随机范围
# params = {"learning_rate": trial.suggest_float("lr", 1e-5, 1.0, log=True)}

# ✅ 用知识库推荐的范围起步
kb_params = kb_match.pattern.suggested_params  # 如 {"learning_rate": "0.01-0.05", "num_leaves": "31-255"}
params = {
    "learning_rate": trial.suggest_float("lr", 0.01, 0.05, log=True),  # 从 kb_params 出发
    "num_leaves": trial.suggest_int("num_leaves", 31, 255),
    "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
    # ... kb_params 提供的范围
}
# 知识库无推荐 → 才用通常文献默认范围
```

### Step 2: 对各 Top-3 方向并行调参

```python
def tune_model(model_class, model_name, kb_params, X, y, cv, n_trials=100):
    """
    1. 用 kb_params 初始化 Optuna search space
    2. Optuna Bayesian optimization (TPE sampler)
    3. Early stopping on validation set
    4. 返回: 最优参数, CV mean/std, OOF predictions
    """

# 并行: LGBM, XGBoost, CatBoost 同时调
# 如果 AutoGluon leaderboard 有其他高分模型 → 也加入
```

### Step 3: 伪标签 (Pseudo-Labeling)

适用条件:
- 有 test 数据（无 label）
- 当前模型 CV score 足够好（CV > baseline + 0.01）
- 样本量不大 (<100K，否则伪标签收益小)

```python
# 1. 用最优模型预测 test
pseudo_labels = best_model.predict(test_features)

# 2. 筛选高置信度样本
if is_classification:
    proba = best_model.predict_proba(test_features)
    confident_mask = np.max(proba, axis=1) > 0.95  # top-5% confidence
else:  # regression
    # 用多模型预测方差衡量置信度
    pred_std = np.std(all_model_preds, axis=0)
    confident_mask = pred_std < np.percentile(pred_std, 20)

# 3. 将置信样本加入训练集，重新训练
train_augmented = pd.concat([train, test[confident_mask].assign(**{target: pseudo_labels[confident_mask]})])

# 4. 重新跑一轮 CV → 验证伪标签是否有正向增益
```

### Step 4: 外部数据扩充（可选）

```
- 搜索 Kaggle 论坛中其他参赛者提到相关公开数据
- 用 mcp-research 找相关外部数据源
- 相似竞赛的 train → 加入当前训练数据
- 注意: 必须验证外部数据与当前 test 分布一致 (adversarial validation)
```

### Step 5: 过拟合检测

```python
# 检查: CV vs train score 差距
overfit_ratio = train_score / cv_score

if overfit_ratio > 1.05:
    # 过拟合 → 增强正则化
    # - 增加 min_child_samples / min_samples_leaf
    # - 降低 num_leaves / max_depth
    # - 增加 reg_alpha / reg_lambda
    # - 增加 subsample / colsample_bytree
```

## 自适应迭代策略（借鉴 PiML）

```
调参循环 (最多 3 轮):
  Round 1: 知识库推荐参数范围 → 100 trials
    → 评估: CV 是否有 > 0.005 的提升 vs baseline?
    → 是: 继续
    → 否: 扩大搜索范围 (kb_params * 2)

  Round 2: 缩窄到最优方向 → 50 trials
    → 评估: CV std 是否在减小?
    → 是: 继续。否: 检查 data leakage / CV split 问题

  Round 3: 微调 → 25 trials (fine grid around best)
    → 输出: 最优参数 + CV 分数
```

## Socrates 质询

- "CV 提升 0.003 是否显著？(check: 是否超过 2*CV_std？)"
- "伪标签的置信样本比例是否合理？(应 <30%)"
- "train score vs CV score 差距 > 0.05 时，用了什么正则化？"

## 可用工具
- Python: optuna, sklearn, xgboost, lightgbm, catboost
- GPU: RAPIDS cuML (if available)

## 约束
- 调参必须用 Optuna (不手工调)
- OOF 预测必须保留（供 ensemble 阶段使用）
- 伪标签必须先验证 CV 增益再确认使用
