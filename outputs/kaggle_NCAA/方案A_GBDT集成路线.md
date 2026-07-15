# 方案 A：GBDT + Elo + 滚动特征 + 集成

> 目标：用标准 Kaggle 套利打法，构建可靠的 NCAA 胜负概率预测模型
> 评价指标：Brier Score（越小越好）
> 数据路径：`outputs/kaggle_NCAA/data/`

---

## 1. 数据加载

### 加载以下文件（男子 M 和女子 W 结构完全相同）

| 文件 | 用途 |
|------|------|
| `{M/W}RegularSeasonCompactResults.csv` | 常规赛结果（Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT） |
| `{M/W}RegularSeasonDetailedResults.csv` | 同上 + 详细统计（FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF） |
| `{M/W}NCAATourneyCompactResults.csv` | 锦标赛结果（同上格式） |
| `{M/W}NCAATourneyDetailedResults.csv` | 锦标赛详细结果 |
| `{M/W}Teams.csv` | 球队信息（TeamID, TeamName, FirstD1Season, LastD1Season） |
| `{M/W}NCAATourneySeeds.csv` | 种子排位（Season, Seed, TeamID） |
| `{M/W}MasseyOrdinals.csv` | **只有男子有** — 第三方排名系统的每日排名（Season, RankingDayNum, SystemName, TeamID, OrdinalRank） |
| `{M/W}TeamConferences.csv` | 球队所属联盟 |
| `SampleSubmissionStage2.csv` | 提交模板（ID, Pred），ID 格式 `{Season}_{TeamA}_{TeamB}` |

**注意：**
- 男子 TeamID 范围 1101-1481，女子 TeamID 范围 3101-34xx
- 2020 赛季没有锦标赛（COVID），跳过该年的锦标赛数据
- Massey Ordinals 只有男子有，女子没有这个文件 → 需要跳过相关特征

---

## 2. 特征工程管线

### 核心设计原则

所有特征必须只使用**该场比赛之前**的信息，严格避免前视偏差（lookahead bias）。

做法：将所有比赛按 `(Season, DayNum)` 排序，逐场处理，先提取特征，再更新状态。

### 2.1 Elo 评分系统

**公式：**
```
Expected(A win vs B) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))

更新规则：
  Elo_A_new = Elo_A + K * (result - expected)
  Elo_B_new = Elo_B + K * ((1 - result) - (1 - expected))
  
  其中 result = 1 if A wins, 0 if B wins
```

**参数：**
- 初始 Elo = 1500（所有球队）
- K = 20（常规赛），K = 30（锦标赛）
- 主场优势修正：主场球队 Elo + 100（仅用于计算 expected，不持久化）
- 赛季重置：每年球队 Elo 收敛回 1500：`Elo_new_season = 0.75 * Elo_end + 0.25 * 1500`

**产出特征：**
- `elo_A`, `elo_B` — 两队赛前 Elo
- `elo_diff` = elo_A - elo_B
- `elo_momentum_A` = 近 5 场 Elo 变化

### 2.2 滚动统计特征

对每个球队，维护一个近 N 场比赛的滑动窗口，多窗口并行：

| 窗口 | 特征 |
|------|------|
| 近 3 场 | 场均得分、场均失分、场均净胜分、胜率 |
| 近 5 场 | 同上 |
| 近 10 场 | 同上 |
| 近 20 场 | 同上 |
| 本季至今 | 同上（赛季累计） |

**产出特征（每队）**：5 个窗口 × 4 个指标 = 20 个特征
**对阵差异特征**：`diff = teamA - teamB` 对每个特征

### 2.3 效率指标（仅当使用 Detailed 数据时）

从详细数据估算每回合效率：

```
Possessions = FGA - OR + TO + 0.475 * FTA  （两队分别算，取平均）

Offensive Efficiency = Points / Possessions * 100
Defensive Efficiency = Points_Allowed / Possessions * 100
Net Efficiency = OE - DE
```

**滚动窗口：** 近 5 场、近 10 场、本季的 OE/DE/NetEfficiency

### 2.4 种子特征

**只在 3 月可用**（锦标赛种子公布后）。对于常规赛期间，种子为缺失。

- `seed_A`, `seed_B` — 种子编号（W01=1, W02=2, ..., Y16=16）
- `seed_diff` = seed_A - seed_B（负数表示 A 种子更好）
- `seed_round` — 种子对应的预期轮次（1号种子预期走6轮，16号预期0轮）
- `is_play_in` — 是否首轮轮空赛（11/16号种子的特殊情况）

**种子→轮次映射：**
```
Seed 1 → round 6 (冠军)
Seed 2 → round 5 (亚军)
Seed 3-4 → round 4 (四强)
Seed 5-8 → round 3 (八强)
Seed 9-12 → round 2 (16强)
Seed 13-16 → round 1 (64强)
```

### 2.5 Massey Ordinals 特征（仅男子）

Massey Ordinals 包含几十种排名系统每天发布的排名。由于系统太多，做以下处理：

1. 只保留在 90% 以上赛季都有数据的稳定系统（约 30-40 个）
2. 对每个系统，取**距离该场比赛最近**的排名
3. 汇总统计：
   - `massey_mean_A` — A 在所有系统中排名均值
   - `massey_std_A` — A 的排名标准差（越小说明各系统看法越一致）
   - `massey_best_A`, `massey_worst_A` — 最佳/最差排名
   - `massey_diff_mean` = massey_mean_A - massey_mean_B

### 2.6 其他上下文特征

- `is_neutral` — 是否中立场地（WLoc == 'N'）
- `is_conference` — 是否同联盟比赛
- `rest_days_A`, `rest_days_B` — 距上一场比赛的天数
- `rest_diff` = rest_days_A - rest_days_B
- `is_tournament` — 是否锦标赛
- `tourney_history_A` — A 过去 5 年在锦标赛的平均轮次

### 2.7 特征汇总

| 特征群 | 数量 | 重要性预估 |
|--------|------|-----------|
| Elo 相关 | 6 | ⭐⭐⭐⭐⭐ |
| 滚动统计差异 | 20 | ⭐⭐⭐⭐ |
| 效率差异 | 6 | ⭐⭐⭐⭐ |
| 种子差异 | 4 | ⭐⭐⭐⭐⭐ |
| Massey Ordinals | 5 | ⭐⭐⭐⭐⭐ |
| 上下文 | 5 | ⭐⭐⭐ |
| **总计** | **~46** | |

---

## 3. 训练数据集构建

### 3.1 数据增广：双向样本

每场比赛产生 **2 个训练样本**（消除 "team1 总是赢家" 的偏差）：

```
样本 1: 
  feature = [elo_diff, rolling_diff_5, efficiency_diff, seed_diff, ...]
           = team_winner_features - team_loser_features
  target = 1

样本 2:
  feature = team_loser_features - team_winner_features
  target = 0
```

这样模型学会的是对称比较：`P(A beats B) = 1 - P(B beats A)`

### 3.2 时间序列验证切分

```
Fold 1: train 1985-2000, val 2001-2005
Fold 2: train 1985-2005, val 2006-2010
Fold 3: train 1985-2010, val 2011-2015
Fold 4: train 1985-2015, val 2016-2020
Fold 5: train 1985-2020, val 2021-2025
```

**注意：切分按 Season 整季切，不能按 DayNum 随机切。**

### 3.3 特征预处理

- 数值特征：不做标准化（树模型不需要）
- 缺失值处理：种子特征在常规赛期间缺失 → 用 -1 填充（树模型能处理）
- Massey Ordinals：赛季早期可能缺失 → 用该赛季中位数填充

---

## 4. 模型训练

### 4.1 CatBoost（主模型）

**Optuna 搜索空间：**
```python
{
    'iterations': trial.suggest_int('iterations', 500, 3000, step=500),
    'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
    'depth': trial.suggest_int('depth', 4, 10),
    'l2_leaf_reg': trial.suggest_float('l2_reg', 1, 10),
    'border_count': trial.suggest_int('border', 32, 255),
    'random_strength': trial.suggest_float('rs', 0.1, 1.0),
    'bagging_temperature': trial.suggest_float('bt', 0.0, 1.0),
}
```

**训练参数：**
- loss_function: 'Logloss'
- eval_metric: 'BrierScore'
- early_stopping_rounds: 50（在验证集上）
- verbose: 0
- Optuna trials: 50

### 4.2 LightGBM（辅模型，与 CatBoost 低相关）

**Optuna 搜索空间：**
```python
{
    'n_estimators': trial.suggest_int('n_est', 500, 3000, step=500),
    'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
    'num_leaves': trial.suggest_int('leaves', 31, 255),
    'min_child_samples': trial.suggest_int('min_child', 20, 100),
    'subsample': trial.suggest_float('sub', 0.7, 1.0),
    'colsample_bytree': trial.suggest_float('col', 0.7, 1.0),
    'reg_alpha': trial.suggest_float('alpha', 0, 1.0),
    'reg_lambda': trial.suggest_float('lambda', 0, 1.0),
}
```

**训练参数：**
- objective: 'binary'
- metric: 'binary_logloss'
- early_stopping_rounds: 50
- Optuna trials: 50

### 4.3 训练流程

```python
for gender in ['M', 'W']:
    # 1. 加载 + 特征工程
    # 2. 5-fold 时间序列 CV
    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        # 3. 训练 CatBoost
        # 4. 训练 LightGBM
        # 5. 记录 OOF 预测
    # 6. 在全量数据上 retrain 最终模型
    # 7. 保存 OOF 预测 + CV Brier
```

**5-fold CV 后得到：**
- 每个训练样本的 OOF 预测（用于集成阶段）
- 平均 CV Brier Score

---

## 5. 集成策略

### 5.1 Blending（加权平均）

```
CatBoost 权重 = 1 / CatBoost_CV_Brier
LightGBM 权重 = 1 / LightGBM_CV_Brier

Blend_Pred = (w_CB * P_CB + w_LGB * P_LGB) / (w_CB + w_LGB)
```

### 5.2 Hill Climbing（贪心权重搜索）

用 scipy.optimize.minimize 搜索最优权重：

```python
目标: minimize Brier(weight * P_CB + (1-weight) * P_LGB)
权重范围: [0, 1]
初始值: 0.5
```

### 5.3 Stacking（可选）

以 CB 和 LGB 的 OOF 预测为特征，训练 LogisticRegression 做元模型：

```python
X_meta = stack([OOF_CB, OOF_LGB], axis=1)
meta = LogisticRegression(C=1.0)
meta.fit(X_meta, y_true)

# 最终预测
P_stack = meta.predict_proba(stack([P_CB_test, P_LGB_test]))[:, 1]
```

### 5.4 模型相关性检查

如果 CatBoost 和 LightGBM 的 OOF 预测相关性 > 0.95：
- 说明它们学到的模式几乎一样 → 集成增益有限
- 仍做集成，但不要期望很大提升

---

## 6. 2026 赛季预测

### 6.1 构建所有对阵

```python
# 筛选 2026 赛季的 D1 球队
teams_2026 = teams[teams['LastD1Season'] >= 2026]

# 生成所有无序对 (A, B) 满足 A < B
pairs = []
for i, team_a in enumerate(teams_2026.TeamID):
    for team_b in teams_2026.TeamID.iloc[i+1:]:
        pairs.append((team_a, team_b))
```

### 6.2 计算特征

对每支球队，计算截至 2026 年 3 月锦标赛种子公布日的特征：
- 最后的 Elo 评分
- 近 10/20 场滚动统计
- 2026 赛季累计数据
- 2026 年种子
- Massey Ordinals 在 3 月的最新排名

### 6.3 预测概率

```python
for (team_a, team_b) in pairs:
    features_a = get_features(team_a, '2026')
    features_b = get_features(team_b, '2026')
    diff = features_a - features_b
    
    p_cb = catboost_model.predict_proba(diff)[:, 1]
    p_lgb = lgb_model.predict_proba(diff)[:, 1]
    p_ensemble = blend(p_cb, p_lgb)
    
    # ID = "2026_{team_a}_{team_b}" (a < b)
    submissions.append({"ID": f"2026_{team_a}_{team_b}", "Pred": p_ensemble})
```

---

## 7. 输出

### 提交文件：`submission_a.csv`

```
ID,Pred
2026_1101_1102,0.45
2026_1101_1103,0.67
...
2026_3101_3102,0.38
...
```

### 实验日志：`solution_a_log.md`

包含：
- 各模型 CV Brier
- 集成后的 CV Brier
- CatBoost 最优参数
- LightGBM 最优参数
- 前 20 个最重要的特征（SHAP importance）
- 男子和女子的性能差异

---

## 8. 预期性能

| 指标 | 预期值 |
|------|--------|
| CatBoost CV Brier | ~0.115-0.125 |
| LightGBM CV Brier | ~0.115-0.126 |
| Ensemble CV Brier | ~0.113-0.123 |
| 运行时间 | 30-60 分钟 |
| 内存需求 | ~4GB |
| Python 依赖 | pandas, numpy, sklearn, lightgbm, catboost, optuna, scipy |

---

## 9. 代码骨架（伪代码结构）

```python
# ============================================================
# solution_a.py — GBDT + Elo + Rolling Features + Ensemble
# ============================================================

DATA_DIR = "outputs/kaggle_NCAA/data"
OUTPUT_DIR = "outputs/kaggle_NCAA"
TARGET_SEASON = 2026

# ---------- 1. LOAD ----------
def load_data(gender):
    reg = pd.read_csv(f"{DATA_DIR}/{gender}RegularSeasonCompactResults.csv")
    tourney = pd.read_csv(f"{DATA_DIR}/{gender}NCAATourneyCompactResults.csv")
    # 可选: detailed = pd.read_csv(...)
    teams = pd.read_csv(f"{DATA_DIR}/{gender}Teams.csv")
    seeds = pd.read_csv(f"{DATA_DIR}/{gender}NCAATourneySeeds.csv")
    return reg, tourney, teams, seeds

# ---------- 2. FEATURES ----------
def compute_elo(reg, tourney):
    # 逐场计算 Elo
    # 返回每场比赛前两队的 Elo
    pass

def compute_rolling_stats(reg, tourney, windows=[3,5,10,20]):
    # 对每支球队，按日期顺序维护滚动窗口
    # 返回每场比赛前的滚动统计
    pass

def compute_seed_features(seeds):
    # 种子 → 数值映射
    pass

def build_features(reg, tourney, seeds, ordinals=None):
    elos = compute_elo(reg, tourney)
    rolling = compute_rolling_stats(reg, tourney)
    seed_feats = compute_seed_features(seeds)
    # 合并所有特征到 game-level
    pass

# ---------- 3. TRAIN ----------
def time_series_cv(seasons, n_folds=5):
    # 时间序列 fold 切分
    pass

def train_catboost(X, y, cv_folds):
    study = optuna.create_study(direction='minimize', 
                                metric=lambda trial: cv_brier(...))
    study.optimize(lambda trial: objective(trial, X, y, cv_folds), n_trials=50)
    return best_model, oof_preds, cv_score

def train_lightgbm(X, y, cv_folds):
    # 同上
    pass

# ---------- 4. ENSEMBLE ----------
def blend(cb_preds, lgb_preds, y):
    # 加权平均 + hill climbing
    pass

def stack(cb_preds, lgb_preds, y):
    # Logistic stacking
    pass

# ---------- 5. PREDICT ----------
def predict_2026(gender, model, teams, seeds):
    # 生成所有对阵预测
    pass

# ---------- MAIN ----------
def main():
    for gender in ['M', 'W']:
        reg, tourney, teams, seeds = load_data(gender)
        ordinals = pd.read_csv(...) if gender == 'M' else None
        X, y = build_features(reg, tourney, seeds, ordinals)
        cv_folds = time_series_cv(X['Season'])
        
        cb_model, cb_oof, cb_cv = train_catboost(X, y, cv_folds)
        lgb_model, lgb_oof, lgb_cv = train_lightgbm(X, y, cv_folds)
        
        ensemble_weights = blend(cb_oof, lgb_oof, y)
        sub = predict_2026(gender, ensemble_models, teams, seeds)
    
    combine_submissions('M', 'W') → submission_a.csv
    print(f"Final CV Brier: {final_brier}")

if __name__ == "__main__":
    main()
```
