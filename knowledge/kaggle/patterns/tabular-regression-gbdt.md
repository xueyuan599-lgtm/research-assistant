# Pattern: tabular-regression-gbdt-stacking

## 匹配条件

| 维度 | 值 |
|------|-----|
| problem_type | regression |
| n_samples | 1K – 500K |
| n_features | 10 – 500 |
| feature_types | 混合（数值 + 类别为主） |
| metric | rmse / mae / rmsle / mape |
| target_distribution | 偏度 < 2 或 log-transform 后 < 1 |

## 推荐算法栈

### Tier 1: GBDT 三件套
1. **LightGBM** — 回归首选
   - 配置: `objective='regression'` (RMSE) or `objective='regression_l1'` (MAE) or `objective='tweedie'` (保险/计数)
   - 注意: RMSLE 目标需先对 target 做 log1p → 训练回归 → 预测后 expm1
2. **XGBoost**
   - 配置: `objective='reg:squarederror'` or `reg:tweedie`
3. **CatBoost**
   - 配置: `loss_function='RMSE'` or `MAE` or `Quantile:alpha=0.5`

### Tier 2: 差异化
4. **Random Forest** — 回归场景表现好
5. **Ridge / KernelRidge** — 与 GBDT 低相关，集成增益
6. **KNN** — 局部模式补充

### Tier 3: 神经网络
7. **MLPRegressor** / **TabNet** — 大样本时

## 回归特有处理

| 步骤 | 方法 |
|------|------|
| 目标变换 | log1p (右偏), sqrt, Box-Cox, Yeo-Johnson |
| 异常值 | IsolationForest 剔出 + 单独建模或不参与 loss |
| 伪标签 | 高置信度预测 + 重新训练（仅大样本） |
| 校正 | 预测后 Platt scaling 式校正 → 均值对齐 |

## 典型集成策略

```
Tier 1: LGBM + XGBoost + CatBoost  (weighted average by CV RMSE)
Tier 2: + Ridge + RF
Final:   hill climbing → weights
Stacking: Ridge 做 Meta-model
```

## 已知成功案例

| 竞赛 | 最佳排名 | 核心模型 |
|------|---------|---------|
| House Prices | top 2% | LGBM + XGBoost + Ridge stacking + 大量特征工程 |
| Mercari Price | top 5% | LGBM + Ridge + 文本特征 (TF-IDF + SVD) |
| Allstate Claims | top 10% | XGBoost + LGBM + CatBoost ensemble |

## 参考来源

- Kaggle House Prices 顶级方案
- Mercari Price Suggestion Winner Solution
- 社区共识
