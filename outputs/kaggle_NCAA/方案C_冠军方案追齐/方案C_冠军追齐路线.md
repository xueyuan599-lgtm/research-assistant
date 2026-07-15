# 方案 C：冠军方案追齐 — Harrison Horan (Brier 0.1097) 复现路线

> 目标：从 Brier 0.164 → 0.13x，追齐 Top 100（0.1243）
> 核心策略：复现冠军极简特征集（3大特征）+ 外部 KenPom 数据 + 概率校准
> 参考方案：方案A_GBDT集成路线.md（保留 Elo 和滚动统计做模型多样性）

---

## 1. 冠军方案核心洞察

Harrison Horan 的 1st place 方案（Brier 0.1097）之所以成功，不是因为模型复杂，而是因为**特征与目标的高度对齐**：

| 发现 | 对我们的启示 |
|------|------------|
| 只用 3 个特征，其他特征加了反而差 | **特征瘦身**，去掉冗余特征 |
| 自定义 `harry_Rating` 本质是 KenPom 效率评分 | **必须引入 KenPom 类数据** |
| `seed_diff` 在女子中相关性 r=-0.756 | 种子特征极重要，已有 ✅ |
| `opp_qlty_pts_won` 质量分捕获赛程强度 | 可用种子+比赛结果计算 |
| Isotonic Calibration 提升 0.003 Brier | 必须加校准 |
| XGBoost Regression（非分类） | 改用回归目标 |

---

## 2. 特征方案（冠军 3 特征 + 我们补充）

### 2.1 核心特征（冠军原版）

#### 特征 1：`seed_diff`（种子差）
已有（方案A已实现）。夺冠选手的确认：
- 男子 r = -0.594
- 女子 r = -0.756（极强）

维持不变。

#### 特征 2：`harry_rating`（KenPom 效率评分）
冠军的 KenPom-like 评分：
```
Possessions = FGA - OR + TO + (FTA × 0.475)
Pace = 200 / ((Possessions_A + Possessions_B) / 2)
OffEff = (Points / Possessions) × 70
DefEff = (OppPts / OppPossessions) × 70  
NetEff = OffEff - DefEff
```

**实现方式 A（推荐，已预计算）：**
- 下载 Formula Bot 免费 CSV: `kenpom_college_basketball_ratings_2000_2026.csv`
- 字段包括: `AdjEM`（效率差）, `AdjO`（进攻）, `AdjD`（防守）, `Tempo`, `Luck`, `SoS`
- 直接取 `AdjEM_diff` 作为 `harry_rating` 的替代
- 来源: https://www.formulabot.com/datasets/kenpom-march-madness

**实现方式 B（从 DetailedResults 自己算）：**
- 使用方案A已有的详细统计计算 Possessions
- 按冠军公式计算 NetEff
- 注意需要赛季均值调整（KenPom 的核心是"adjusted"——调整了对手强度）
- 精度不如方式A，但可不依赖外部数据

**推荐：A+B 各做一个特征，看相关性。**

#### 特征 3：`opp_qlty_pts_won`（质量分）
冠军公式：
```
# 按种子分档（冠军用 4 档）
TIER_1_SEEDS = [1, 2, 3, 4]     # 顶级强队
TIER_2_SEEDS = [5, 6, 7, 8]     # 中上
TIER_3_SEEDS = [9, 10, 11, 12]  # 中下
TIER_4_SEEDS = [13, 14, 15, 16] # 弱队

# 对每支球队，计算赛季中对各档对手的净胜分
opp_qlty_pts_won = sum(
    margin_vs_tier1 * weight1 +
    margin_vs_tier2 * weight2 +
    margin_vs_tier3 * weight3 +
    margin_vs_tier4 * weight4
)
# 权重递减：tier1 权重最大
```

冠军是手工调权的。我们可以用种子自动分档（方案A已有种子数据）。

### 2.2 补充特征（从方案A保留）

冠军用了极简 3 特征。但我们的模型中 CB 和 LGB 需要更多特征做模型多样性。保留以下经验证有效的特征：

| 保留 | 特征 | 原因 |
|------|------|------|
| ✅ | `elo_diff` | 方案A #1 特征（重要性 2893），与 `harry_rating` 互补 |
| ✅ | `off_eff_diff` | 方案A #3，提供进攻端信息 |
| ✅ | `win_rate_diff` | 方案A #2，基本指标 |
| ❌ | 所有滚动窗口重复特征 | 冠军证明冗余特征有害 |
| ❌ | `rest_days`, `is_neutral` 等上下文 | 重要性低，冠军未使用 |
| ⚠️ | Massey Ordinals | 方案A中仅 #18，可考虑去掉 |

### 2.3 最终特征集（精简版）

**男子（6-8 个特征）：**
```
1. seed_diff          ← 冠军核心 #1
2. harry_rating_diff  ← 冠军核心 #2（KenPom AdjEM_diff）
3. opp_qlty_pts_won_diff ← 冠军核心 #3
4. elo_diff           ← 方案A保留（与 harry_rating 互补）
5. win_rate_diff      ← 基础指标保留
6. off_eff_diff       ← 进攻效率
（可选）massey_mean_diff
（可选）avg_PF_l10_diff
```

**女子（5-6 个特征，无 Massey）：**
```
1. seed_diff          ← 冠军核心 #1（女子 r=-0.756 极强）
2. harry_rating_diff  
3. opp_qlty_pts_won_diff
4. elo_diff
5. win_rate_diff
```

---

## 3. 模型方案

### 3.1 改用回归（冠军做法）

冠军使用的是 **XGBoost Regression** 而不是分类:

```python
# 冠军做法：回归 → 概率
model = xgb.XGBRegressor(
    n_estimators=4000,
    learning_rate=0.003,
    early_stopping_rounds=20,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8
)
# 目标: y = 1 if TeamA beats TeamB else 0
# 预测: 直接输出 0-1 概率
# 不需要 predict_proba
```

**我们的方案：三模型集成**

| 模型 | 任务类型 | 理由 |
|------|---------|------|
| **XGBoost Regressor**（主） | 回归 | 冠军做法，4000 trees |
| **CatBoost**（辅） | Logloss 分类 | 方案A已验证，与 XGB 多样性 |
| **LightGBM**（辅） | binary 分类 | 方案A已验证，与 XGB 多样性 |

### 3.2 参数初始化

**XGBoost（冠军参数为起点）：**
```python
{
    'n_estimators': 4000,
    'learning_rate': 0.003,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'early_stopping_rounds': 20,
    'tree_method': 'gpu_hist'  # GPU加速
}
```

**CatBoost（方案A最佳参数为起点）：**
```python
{
    'iterations': 500,
    'learning_rate': 0.08,
    'depth': 5,
    'l2_leaf_reg': 7.0,
    'task_type': 'GPU'
}
```

**LightGBM（方案A最佳参数为起点）：**
```python
{
    'n_estimators': 2000,
    'learning_rate': 0.016,
    'num_leaves': 225,
    'subsample': 0.98,
    'colsample_bytree': 0.84,
    'device': 'gpu'
}
```

### 3.3 CV 策略：Leave One Season Out

冠军做法：
```
for season in range(2003, 2026):
    if season == 2020: continue  # COVID
    train = all_seasons[all_seasons.Season != season]
    val = all_seasons[all_seasons.Season == season]
    # 训练 + 记录 OOF 预测
```

相比方案A的 expanding window：
- ✅ 样本利用率更高（所有数据点参与训练）
- ✅ 更真实模拟跨赛季泛化
- ⚠️ 2020 赛季跳过（无锦标赛标签）

### 3.4 概率校准（冠军标配）

```python
from sklearn.isotonic import IsotonicRegression

# 步骤 1：从 OOF 预测获取原始分数
# 步骤 2：用 IsotonicRegression 校准
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_oof_preds, y_true)
calibrated_preds = iso.transform(raw_test_preds)

# 冠军 CV Brier 变化: 0.1620 → 0.1590（-0.003）
```

### 3.5 后处理：极端概率 rounding

```python
# 冠军做法
preds = np.clip(preds, 0.03, 0.97)  # 极端值收缩
preds[preds >= 0.97] = 0.995  # 几乎必胜
preds[preds <= 0.03] = 0.005  # 几乎必败
```

---

## 4. 外部数据获取方案

### 4.1 Formula Bot KenPom 数据（免费首选）

| 信息 | 内容 |
|------|------|
| URL | https://www.formulabot.com/datasets/kenpom-march-madness |
| 时间范围 | 2000-2026 |
| 格式 | CSV |
| 核心字段 | AdjEM（效率差）, AdjO（进攻效率）, AdjD（防守效率）, Tempo（节奏）, Luck（运气）, SoS（赛程强度）, NCSOS |
| 费用 | **免费** |

**集成方式：**
```python
kenpom = pd.read_csv("kenpom_college_basketball_ratings_2000_2026.csv")
# 按 Season + TeamID 合并到比赛数据
# 特征: kenpom_AdjEM_diff = TeamA.AdjEM - TeamB.AdjEM
```

### 4.2 质量分数据（纯计算，不依赖外部）

从已有的种子数据计算：
```python
# 对每支球队、每个赛季
# 1. 按对手种子分 4 档
# 2. 计算对各档的场均净胜分
# 3. 加权求和（高档对手权重大）
```

### 4.3 伤病数据（进阶，可选）

冠军从 Rotowire 手动录入。可跳过 v1。

---

## 5. 执行计划

### Step 1：数据准备
```
- 下载 Formula Bot KenPom CSV
- 清洗、按(Season, TeamID) 对齐到比赛数据
- 构建质量分特征
- 验证与 champion 特征的相关系数和单变量 Brier
```

### Step 2：特征构造
```
- 冠军 3 特征：seed_diff / harry_rating_diff / opp_qlty_pts_won_diff
- 方案A保留特征：elo_diff / win_rate_diff / off_eff_diff
- 特征瘦身：去掉滚动窗口冗余特征
- 双向样本增广保留
- 男子/女子分别构造
```

### Step 3：模型训练
```
- XGBoost Regressor（主模型，4000 trees）
- CatBoost（辅模型）
- LightGBM（辅模型）
- CV: Leave One Season Out
- 全 GPU 加速
```

### Step 4：校准 + 集成
```
- Isotonic Regression 校准（每个模型单独校准）
- Blending + Hill Climbing 找最优权重
- 极端概率后处理
```

### Step 5：提交
```
- 生成 submission_a.csv
- 记录 CV Brier vs LB 对比
```

---

## 6. 预期性能

| 阶段 | 预期 Brier | 相比现在的提升 |
|------|-----------|--------------|
| 当前（方案A） | 0.1640 | — |
| Step1+2（特征瘦身+KenPom） | 0.145-0.155 | -0.009~0.019 |
| Step3（XGBoost + Leave One Out CV） | 0.138-0.148 | -0.007~0.016 |
| Step4（Isotonic 校准 + 后处理） | 0.134-0.144 | -0.004~0.004 |
| **最终集成** | **0.130-0.140** | **-0.024~0.034** |

Top 100 线是 0.1243。如果 KenPom 数据质量好 + Isotonic 校准效果好，0.130 是合理目标。

但到 0.124 需要额外的提升（伤病数据、手工调权、冠军级别的细致调参）。

---

## 7. 代码结构

```
方案C_冠军方案追齐/
├── 方案C_冠军追齐路线.md          ← 本文件
├── data/
│   └── kenpom_historical.csv      ← 从 Formula Bot 下载
├── solution_c.py                   ← 完整实现脚本
├── submission_c.csv                ← 提交文件
└── solution_c_log.md               ← 实验日志
```

### solution_c.py 骨架

```python
# ============================================================
# solution_c.py — 冠军方案追齐
# 核心：KenPom 特征 + XGBoost + Isotonic Calibration
# ============================================================

# ---------- 1. 数据加载 ----------
def load_data(gender):
    # 同方案A，加载常规赛/锦标赛/种子数据
    pass

def load_kenpom():
    # 加载 KenPom CSV，按(Season, TeamID)合并
    pass

# ---------- 2. 冠军 3 大特征 ----------
def compute_harry_rating(detailed_results):
    # Possessions = FGA - OR + TO + 0.475*FTA
    # OffEff = (Points/Poss) * 70
    # DefEff = (OppPts/OppPoss) * 70
    # 返回每队每赛季的评分
    pass

def compute_quality_wins(reg_results, seeds):
    # 按种子分4档，计算对各档净胜分的加权和
    pass

def compute_seed_features(seeds):
    # 种子→数值，同方案A
    pass

# ---------- 3. 特征合并 ----------
def build_features(gender, kenpom):
    # 合并 3 核心特征 + elo + win_rate
    # 不用滚动窗口冗余特征
    # 双向样本
    pass

# ---------- 4. 训练：Leave One Season Out ----------
def train_xgboost(X, y, seasons):
    xgb_params = {
        'n_estimators': 4000,
        'learning_rate': 0.003,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist',
        'early_stopping_rounds': 20,
    }
    # Leave One Season Out CV
    for val_season in unique_seasons:
        if val_season == 2020: continue
        X_train = X[X.Season != val_season]
        X_val = X[X.Season == val_season]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        oof_preds[val_season] = model.predict(X_val)
    return model, oof_preds

def train_catboost(X, y, seasons):
    # 同方案A参数，但用 Leave One Season Out CV
    pass

def train_lightgbm(X, y, seasons):
    # 同方案A参数，但用 Leave One Season Out CV
    pass

# ---------- 5. 校准 ----------
def calibrate(oof_preds, y_true, test_preds):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_preds, y_true)
    return iso.transform(test_preds)

# ---------- 6. 集成 ----------
def ensemble(xgb_pred, cb_pred, lgb_pred, y_oof):
    # Hill Climbing 找权重
    # 权重 xgb > cb ≈ lgb（预期）
    pass

# ---------- MAIN ----------
def main():
    kenpom = load_kenpom()
    for gender in ['M', 'W']:
        X, y = build_features(gender, kenpom)
        xgb_model, xgb_oof = train_xgboost(X, y, X['Season'])
        cb_model, cb_oof = train_catboost(X, y, X['Season'])
        lgb_model, lgb_oof = train_lightgbm(X, y, X['Season'])
        
        # 校准
        xgb_cal = calibrate(xgb_oof, y, xgb_test_preds)
        cb_cal = calibrate(cb_oof, y, cb_test_preds)
        lgb_cal = calibrate(lgb_oof, y, lgb_test_preds)
        
        # 集成
        weights = find_weights(xgb_cal, cb_cal, lgb_cal, y)
        final_pred = blend(xgb_cal, cb_cal, lgb_cal, weights)
        
        save_submission(gender, final_pred)
    
    combine('M', 'W') → submission_c.csv
```

---

## 8. 与方案A对比总结

| 维度 | 方案A（当前，Brier=0.164） | 方案C（目标 Brier=0.130） |
|------|--------------------------|-------------------------|
| **特征数量** | 34-40 个 | 5-8 个 |
| **核心特征** | Elo + 滚动窗口 | KenPom + 种子差 + 质量分 |
| **外部数据** | 无 | ✅ KenPom CSV |
| **主模型** | CatBoost + LightGBM | ✅ **XGBoost Regressor** |
| **CV 策略** | Expanding Window 5-fold | ✅ **Leave One Season Out** |
| **校准** | 无 | ✅ **Isotonic Regression** |
| **后处理** | clip [0.05, 0.95] | ✅ **极端值 rounding** |
| **预期 Brier** | 0.164 | **0.130-0.140** |
| **与 Top 100 差距** | -0.040 | **-0.006~0.016** |

---

*方案C依据 Harrison Horan (2026 Kaggle NCAA 1st, Brier=0.1097) 的公开解法纪要整理*
