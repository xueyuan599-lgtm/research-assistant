# Pattern: tabular-multiclass-gbdt-stacking

## 匹配条件

| 维度 | 值 |
|------|-----|
| problem_type | multiclass_classification |
| n_samples | 1K – 500K |
| n_features | 10 – 500 |
| feature_types | 混合（数值 + 类别为主） |
| metric | logloss / accuracy / multiclass_auc |
| train_test_shift | < 0.05 |

## 推荐算法栈（按优先级）

### Tier 1: GBDT 三件套（必选）
1. **LightGBM** — 多分类首选
   - 配置: `objective='multiclass'`, `metric='multi_logloss'`, `num_class=N`
   - 注意: 多分类时 `num_class` 必须正确设置
2. **XGBoost** — 差异化模型
   - 配置: `objective='multi:softprob'`, `eval_metric='mlogloss'`, `num_class=N`
3. **CatBoost** — 类别特征场景
   - 配置: `loss_function='MultiClass'`, `eval_metric='MultiClass'`
   - 优势: 类别特征无需编码

### Tier 2: 差异化模型
4. **Random Forest** — 与 GBDT 低相关
5. **TabNet** — 深度学习备选（N ≥ 10K 样本时）

### Tier 3: 线性基线
6. **Multinomial Logistic Regression** (sklearn `LogisticRegression(multi_class='multinomial')`)

## 与二分类的关键差异

| 差异点 | 二分类 | 多分类 |
|--------|--------|--------|
| 类别不平衡 | `scale_pos_weight` | `class_weight='balanced'` |
| 目标编码 | Target mean per class | 需搞多类目标编码 |
| 集成 | Stacking 用 LogisticRegression | Stacking 元模型推荐 XGBoost/LGBM |
| 伪标签 | 高置信度样本加 pseudo label | 小心伪标签放大类别偏斜 |
| 评价 | AUC / LogLoss | LogLoss (cross-entropy) |

## 典型集成策略

```
Tier 1: LGBM + XGBoost + CatBoost (weighted by CV logloss)
Tier 2: + RF
Tier 3: Stacking via LGBM (XGBoost multi:softprob 做 meta-model)
```

## 已知成功案例

| 竞赛 | 最佳排名 | 核心模型 |
|------|---------|---------|
| Otto Group Product | top 3% | XGBoost + NN ensemble |
| Forest Cover Type | top 5% | LGBM + CatBoost stacking |
| Santander Customer | top 10% | LGBM + 大量特征工程 |

## 参考来源

- LGBM Multiclass 官方文档最佳实践
- AutoGluon multiclass 自动集成策略
- 社区共识
