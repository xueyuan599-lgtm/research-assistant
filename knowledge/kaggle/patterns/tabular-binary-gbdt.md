# Pattern: tabular-binary-gbdt-stacking

## 匹配条件

| 维度 | 值 |
|------|-----|
| problem_type | binary_classification |
| n_samples | 1K – 500K |
| n_features | 10 – 500 |
| feature_types | 混合（数值 + 类别为主） |
| metric | auc / logloss / accuracy |
| train_test_shift | < 0.05 (adversarial validation AUC) |

## 推荐算法栈（按优先级）

### Tier 1: GBDT 三件套（必选）
1. **LightGBM** — 首选
   - 配置: `boosting='goss'`, `objective='binary'`, `metric='auc'`
   - 优势: 速度快、类别特征原生支持
   - 典型超参: `learning_rate=0.01-0.05, num_leaves=31-255, min_child_samples=20-100`
2. **XGBoost** — 差异化模型
   - 配置: `tree_method='hist'`, `objective='binary:logistic'`, `eval_metric='auc'`
   - 优势: 正则化强、深度树表现好
   - 典型超参: `learning_rate=0.01-0.05, max_depth=3-8, subsample=0.6-0.9, colsample_bytree=0.6-0.9`
3. **CatBoost** — 类别特征专业户
   - 配置: `loss_function='Logloss'`, `eval_metric='AUC'`
   - 优势: 类别特征无需编码、抗过拟合好
   - 典型超参: `learning_rate=0.01-0.05, depth=6-10, l2_leaf_reg=1-10`

### Tier 2: 差异化模型（推荐）
4. **Random Forest** — 与 GBDT 低相关
   - 优势: 与 boosting 系列相关性低，集成后通常有增益
   - 配置: `n_estimators=500-1000, max_depth=10-30`
5. **TabNet** — 深度学习备选
   - 优势: 特征交互自动学习
   - 注意: 样本 <10K 时不推荐

### Tier 3: 线性基线（必须跑，用于对标）
6. **Logistic Regression** — 校准
7. **ElasticNet** — 高维/稀疏场景

## 典型特征工程

| 步骤 | 方法 | 优先级 |
|------|------|--------|
| 缺失值 | LGBM/CatBoost 原生处理 | 必做 |
| 类别编码 | Target encoding (5-fold), Count encoding | 必做 |
| 数值变换 | QuantileTransformer, PowerTransformer, Box-Cox | 推荐 |
| 交互特征 | GBDT leaf indices 作特征, 乘除组合 top-20 | 推荐 |
| 聚合特征 | groupby mean/std/min/max (防泄漏: 用 5-fold target encoding) | 推荐 |
| 降维 | PCA/NMF → 作为额外特征加入 | 可选 |

## 典型集成策略

```
Tier 1: LGBM + XGBoost + CatBoost   (加权平均, weight = CV score)
Tier 2: + RF                        (blending)
Tier 3: + 所有模型 stacking          (meta-model = LogisticRegression)
Final:  hill climbing 搜索最优权重
```

## 已知成功案例

| 竞赛 | 最佳排名 | 核心模型 | 参考方案 |
|------|---------|---------|---------|
| Titanic | top 2% | LGBM + XGBoost stacking | [Kaggle Discussion] |
| Spaceship Titanic | top 3% | CatBoost + LGBM + XGBoost | [Kaggle Discussion] |
| IEEE-CIS Fraud | top 5% | LGBM + NN ensemble | [Kaggle Discussion] |
| Home Credit Default | top 10% | LGBM + CatBoost + 大量特征工程 | [Kaggle Discussion] |

## 常见陷阱

1. **目标泄漏**: target encoding 必须用 out-of-fold
2. **过拟合小样本**: <1000 样本时 → CatBoost + 强正则化，不做 stacking
3. **分布偏移**: adversarial validation AUC > 0.55 → 必须修正采样策略
4. **类别不平衡**: 对 LightGBM 设 `is_unbalance=True` 或 `scale_pos_weight`

## 参考来源

- NVIDIA Kaggle Grandmasters Playbook (2024)
- AutoKaggle (ICLR 2025): 5-agent pipeline
- PiML (PMLR 2025): iterative memory approach
- 社区共识: LGBM+XGBoost+CatBoost 三件套在表格二分类中无替代

---

## 附录：NCAA 竞赛特殊经验

### 1. 双向样本增广

在竞赛预测（如 NCAA 锦标赛）场景中，每场比赛包含两支队伍，标签天然是对称的。通过 **双向样本增广**（win - lose → label=1, lose - win → label=0）可将训练样本翻倍。此方法要求所有特征以两队的**差值**形式输入。适用条件：
- 比赛类问题
- 包含明确的配对结构
- 特征可以表示为差值（对称特征）

### 2. Elo 评分在体育预测中的重要性

Elo 评分在 NCAA 预测中是排名第 1 的特征，远超种子差、Massey 排名差等静态特征。Elo 的优势：
- 动态反映球队的近期状态变化
- 隐含对手强度校准（击败强队获得更多积分）
- 赛季间回归避免过时信息积累
- 可自定义 K 因子区分常规赛和锦标赛

**关键参数**: 初始 Elo=1500, K_reg=20, K_tourney=30, home_adv=100, season_regress=0.75

### 3. 时间序列 CV (expanding window) 应对时间偏移

体育数据存在明显的时间分布偏移（规则变化、球队风格演变、COVID 影响）。Expanding Window CV 可以有效暴露时间不稳定性：

| 现象 | 表现 | 应对 |
|------|------|------|
| 规则/风格演变 | fold-to-fold Brier 波动 | 使用 expanding 而非 sliding window |
| COVID 异常 | Fold 4 Brier 异常升高（0.205） | 标识/降权异常赛季 |
| 近期数据偏移 | 早期 fold vs 晚期 fold 表现不一致 | 增加 fold 数量至 5-6 |

**实现要点**: 按赛季分组，训练集只包含历史赛季。

### 4. 男子 vs 女子数据差异处理

| 差异 | 处理方式 |
|------|---------|
| 男子有 Massey Ordinals（7 系统），女子无 | 女子减少特征数，单独训练模型 |
| 男子数据始于 2003，女子始于 2010 | CV fold 数量不同（男子 5 fold, 女子 4 fold） |
| 女子预测难度更低（Brier ~0.14 vs ~0.18） | 女子集成中 CatBoost 主导（LGBM 权重为 0） |
| 女子样本量更少 | 选择泛化能力更强的模型（CatBoost > LightGBM） |

**关键发现**: 男女数据**必须分开建模**。统一模型在性能上显著劣于分性别模型（女子 Elo 值的方差和均值不同，效率特征分布不同）。

### 5. Massey Ordinals 处理技巧

Massey Ordinals 有 100+ 系统，但许多系统仅覆盖部分赛季。处理要点：
- **仅使用全覆盖系统**（男子: AP, COL, DOL, MOR, POM, USA, WLK），排除不完整系统
- 取每个系统在该赛季的最新排名（max RankingDayNum）
- 聚合统计：mean, std, best, worst, median, spread
- 对缺失值用中位数填充
- Massey 特征收益有限（~0.001-0.002 Brier），如计算资源有限可考虑省略

### 6. 2020 赛季特殊处理

2020 年 NCAA 锦标赛因 COVID-19 取消。必须：
- 排除 2020 年的 tournament 数据（无比赛结果）
- 保留常规赛数据用于特征计算
- 2021 年锦标赛正常但赛程异常，可考虑降权

### 7. 常见陷阱

1. **种子差 ≠ Elo 差**: 种子是锦标赛开始前的固定排名，Elo 是动态更新的。两者相关但 Elo 信息更丰富
2. **滚动窗口期选择**: 3 场窗口过短（噪声大），20 场窗口能更好地反映球队赛季状态
3. **效率计算**: 使用 `poss = FGA - OR + TO + 0.475*FTA` 公式，场均 possessions 差异大时需要标准化
4. **中性场地**: NCAA 锦标赛几乎全在中立场地进行，`is_neutral` 特征无区分度
