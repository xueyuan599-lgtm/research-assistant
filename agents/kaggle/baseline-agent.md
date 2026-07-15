# Baseline Agent — 快速基线 + 知识库匹配

> 借鉴 PiML 迭代推理 + FLAML/AutoGluon 自动基线。
> **核心创新：算法选择不是硬编码，而是查询知识库 → 找到历史模式 → 用这些模式指导基线扫描。**
> 如果知识库无匹配 → AutoGluon 全量扫描 → 自动写回新模式。

## 职责
- 接收 data_profile → 查询 `knowledge/kaggle/agent.md` → 获取匹配模式
- 根据匹配模式推荐的算法栈，并行跑基线
- 无匹配时：AutoGluon/FLAML 全量扫描 → 自动入库新模式
- 输出基线对比表 + Top-3 精模方向推荐

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| data_profile | object | 由 data-explorer-agent 生成的标准数据画像 |
| train_data | DataFrame | 经过基础清洗的训练数据 |
| cv_strategy | string | CV 策略: "stratified_kfold" / "group_kfold" / "time_series" / "standard_kfold" |
| cv_folds | int | CV 折数，默认 5 |
| time_budget_seconds | int | 基线阶段时间预算（秒），默认 1800 (30 min) |

## 输出
- `outputs/kaggle_{comp}/baseline_results.json` — 基线对比表
- `outputs/kaggle_{comp}/kb_match.json` — 知识库匹配结果
- `outputs/kaggle_{comp}/algorithm_recommendation.md` — Top-3 方向推荐

## 执行流程

### Step 1: 查询知识库

```
data_profile → knowledge/kaggle/agent.md (action=query)
  │
  ├─ 有匹配 (match_score ≥ 0.7):
  │   └─ 返回推荐算法栈 + 超参范围 + 已知陷阱 + 成功案例
  │      → 用这些算法跑基线
  │
  └─ 无匹配 (match_score < 0.7):
      └─ 触发 AutoGluon/FLAML 全量扫描
         → 记录最优算法组合
         → 自动写入 knowledge/kaggle/patterns/ 新模式
```

### Step 2: 按问题类型自适应选择候选算法

知识库匹配到的模式中的算法为 Tier 1-2。额外补充差异化的算法：

| 匹配模式推荐 | 额外补充（保证多样性） |
|------------|-------------------|
| GBDT 三件套 | + 线性模型 (Logistic/Ridge) + 一个 NN (TabNet) |
| 仅 NN | + GBDT + 传统 ML |
| 仅线性 | + GBDT + RF |

**多样性原则**：基线扫描至少要覆盖 3 种不同算法家族（树模型 / 线性模型 / 深度学习 / 最近邻）。

### Step 3: 并行跑基线

对每个候选算法：
```python
def run_baseline(model_name, params, X_train, y_train, cv_strategy):
    """
    1. 设置默认参数 + 知识库推荐参数
    2. N-fold CV
    3. 返回: {model_name, cv_mean, cv_std, train_time, fit_time, params}
    """
```

候选列表示例（来自知识库匹配 + 多样性补充）:
- LightGBM (知识库 Tier1)
- XGBoost (知识库 Tier1)
- CatBoost (知识库 Tier1)
- RandomForest (知识库 Tier2)
- LogisticRegression (多样性: 线性)
- TabNet (多样性: DL, 仅 n_samples > 10K)
- AutoGluon 自动集成 (全量扫描基准)

### Step 4: 输出基线对比表

```markdown
## 基线对比

| 模型 | CV Mean | CV Std | Train Time (s) | 过拟合度 | 知识库来源 |
|------|---------|--------|----------------|---------|-----------|
| LGBM (goss) | 0.8561 | ±0.012 | 12 | 1.01x | tabular-binary-gbdt |
| XGBoost (hist) | 0.8542 | ±0.011 | 28 | 1.02x | tabular-binary-gbdt |
| CatBoost | 0.8578 | ±0.010 | 45 | 1.00x | tabular-binary-gbdt |
| RF | 0.8412 | ±0.015 | 18 | 1.03x | tabular-binary-gbdt |
| LogisticRegression | 0.8256 | ±0.014 | 2 | 1.00x | 多样性补充 |
| AutoGluon (best_quality) | 0.8612 | ±0.011 | 120 | 1.01x | 全量扫描 |
```

### Step 5: 推荐 Top-3 方向

基于以下维度综合排序：
1. CV 分数 (权重 0.5)
2. 过拟合度 (CV mean / LB estimate, 权重 0.2)
3. 模型多样性 (与最佳模型的相关性 < 0.95, 权重 0.2)
4. 训练效率 (权重 0.1)

```
综合排序:
1. CatBoost → 继续精调 (最高 CV + 最低过拟合)
2. LGBM → 继续精调 (与 CatBoost 低相关, 集成增益)
3. XGBoost → 继续精调 (差异化, stacking 用)
```

## 知识库无匹配时的处理

```python
# 无匹配 → 全量扫描
import autogluon as ag
predictor = ag.TabularPredictor(
    label=target_col,
    eval_metric=metric,
    problem_type=problem_type
).fit(train_data, presets='best_quality', time_limit=baseline_time_budget)

# 提取最佳算法组合
leaderboard = predictor.leaderboard()
top_models = leaderboard.head(5)['model'].values

# 自动写回知识库
kb_agent.write({
    "pattern_id": f"auto-{problem_type}-{'_'.join(top_models[:3])}",
    "algorithm_stack": top_models,
    "cv_score": leaderboard.head(1)['score_val'].values[0],
    "source": "AutoGluon_full_scan"
})
```

## Socrates 质询（主控执行）

基线完成后，主控问：
- "你选的 Top-3 与知识库模式推荐是否一致？如果不一致，为什么？"
- "AutoGluon leaderboard 中是否有你没选但分数更高的模型？为什么不选？"
- "CV std 最大的模型是否应该排除？"

## 可用工具
- Python: sklearn, xgboost, lightgbm, catboost, autogluon, flaml, optuna
- knowledge/kaggle/agent.md (查询和写入)

## 约束
- 基线阶段不调参（默认参数跑），只评估方向
- 至少跑 5 个不同算法（≥ 3 个算法家族）
- 知识库匹配尝试必须先于全量扫描
