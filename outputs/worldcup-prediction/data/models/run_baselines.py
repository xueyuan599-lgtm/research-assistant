#!/usr/bin/env python3
"""
世界杯预测 — 基线模型对比
============================================================
构建 4 个基线模型：Elo、Poisson、XGBoost、LightGBM
时间序列分割 + 锦标赛分组 CV 两种验证策略
评估指标：Accuracy、LogLoss、Brier Score、AUC
============================================================
"""

import os, sys, warnings, json, pickle, time
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "features")
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

FULL_CSV = os.path.join(DATA_DIR, "feature_matrix_full.csv")
REDUCED_CSV = os.path.join(DATA_DIR, "feature_matrix.csv")

# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. 加载数据
# ---------------------------------------------------------------------------
print("=" * 60)
print("世界杯预测 — 基线模型对比")
print("=" * 60)

df = pd.read_csv(FULL_CSV, encoding="utf-8-sig")

# drop rows where target is NaN (future matches to predict)
mask_trainable = df["home_win"].notna()
df_full = df[mask_trainable].copy().reset_index(drop=True)
print(f"\n总样本: {len(df)}  |  可训练样本: {len(df_full)}")

# ---------------------------------------------------------------------------
# 2. 构造目标变量
# ---------------------------------------------------------------------------
# 3 分类：0 = 客胜, 1 = 平局, 2 = 主胜
df_full["target"] = np.where(
    df_full["home_win"] == 1, 2,
    np.where(df_full["draw"] == 1, 1, 0)
)
y = df_full["target"].values

# 日期列（时间序列分割）
dates = pd.to_datetime(df_full["date"])
df_full["date_dt"] = dates

print("\n目标分布:")
vc = pd.Series(y).value_counts().sort_index()
for k in [0, 1, 2]:
    label = ["客胜", "平局", "主胜"][k]
    print(f"  {label}: {vc.get(k, 0)} ({100*vc.get(k,0)/len(y):.1f}%)")

# ---------------------------------------------------------------------------
# 3. 特征工程（数值 + 分类编码）
# ---------------------------------------------------------------------------
# 识别列类型
id_cols = ["date", "home_team", "away_team", "tournament", "source",
           "home_score", "away_score", "home_win", "draw", "target", "date_dt"]
feature_cols = [c for c in df_full.columns if c not in id_cols]

# 分类特征
cat_cols = ["home_confederation", "away_confederation"]
num_cols = [c for c in feature_cols if c not in cat_cols]

print(f"\n数值特征: {len(num_cols)}  分类特征: {len(cat_cols)}")

# 处理分类特征 — one‑hot encoding
df_feat = df_full[feature_cols].copy()
df_feat = pd.get_dummies(df_feat, columns=cat_cols, drop_first=False, dtype=float)

# 缺失值填充（用中位数）
for c in df_feat.columns:
    if df_feat[c].isna().any():
        df_feat[c] = df_feat[c].fillna(df_feat[c].median())

X = df_feat.values.astype(np.float64)
feature_names = df_feat.columns.tolist()

print(f"最终特征维度: {X.shape[1]}")

# 保存特征名
with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f)

# ---------------------------------------------------------------------------
# 4. 验证策略
# ---------------------------------------------------------------------------

# 4a. 时间序列分割 (前 80% 训练, 后 20% 验证)
sorted_idx = np.argsort(dates)
n = len(y)
split_ts = int(n * 0.8)
train_ts_idx = sorted_idx[:split_ts]
val_ts_idx = sorted_idx[split_ts:]

X_train_ts = X[train_ts_idx]
y_train_ts = y[train_ts_idx]
X_val_ts = X[val_ts_idx]
y_val_ts = y[val_ts_idx]
dates_train = dates.iloc[train_ts_idx]
dates_val = dates.iloc[val_ts_idx]

print(f"\n--- 时间序列分割 ---")
print(f"训练: {len(X_train_ts)}  ({dates_train.min().date()} ~ {dates_train.max().date()})")
print(f"验证: {len(X_val_ts)}  ({dates_val.min().date()} ~ {dates_val.max().date()})")

# 4b. 锦标赛分组 CV (按 tournament 分组)
from sklearn.model_selection import GroupKFold
from scipy.stats import poisson as sp_poisson

# 用 tournament 作为分组
tournament_groups = df_full["tournament"].values
gkf = GroupKFold(n_splits=5)

# ---------------------------------------------------------------------------
# 5. 评估指标工具函数
# ---------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

def compute_metrics(y_true, y_prob, model_name="", prefix=""):
    """计算多分类评估指标"""
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)

    # Brier Score (multi‑class: average of one‑vs‑rest)
    n_classes = y_prob.shape[1]
    brier_sum = 0.0
    for c in range(n_classes):
        y_bin = (y_true == c).astype(float)
        brier_sum += brier_score_loss(y_bin, y_prob[:, c])
    brier = brier_sum / n_classes

    # AUC (OvR macro)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    return {
        "model": model_name,
        "split": prefix,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "auc_ovr": round(auc, 4),
        "n": len(y_true),
    }

# ---------------------------------------------------------------------------
# 6. MODEL 1: Elo Baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 1: Elo 基线模型")
print("=" * 60)

def elo_expected(elo_diff, home_advantage=100):
    """
    Elo 预期得分（主队胜率）
    使用标准 Elo 公式: E = 1 / (1 + 10^(-(elo_diff+home_adv)/400))
    """
    return 1.0 / (1.0 + 10 ** (-(elo_diff + home_advantage) / 400.0))

def elo_to_probs(elo_diff, home_advantage=100, draw_margin=100):
    """
    将 Elo 差值转换为三分类概率 [客胜, 平局, 主胜]

    方法：使用有序 logit 变换：
      主胜: P(home_win) = 1 / (1 + 10^(-(diff + half_margin)/400))
      客胜: P(away_win) = 1 / (1 + 10^(-(-diff + half_margin)/400))
      平局: 剩余概率
    """
    half_margin = draw_margin
    p_home = 1.0 / (1.0 + 10 ** (-(elo_diff + home_advantage - half_margin) / 400.0))
    p_away = 1.0 / (1.0 + 10 ** (-(-elo_diff - home_advantage - half_margin) / 400.0))
    p_draw = 1.0 - p_home - p_away
    # clip
    p_draw = np.clip(p_draw, 0, 1)
    total = p_home + p_away + p_draw
    p_home /= total
    p_away /= total
    p_draw /= total
    return np.column_stack([p_away, p_draw, p_home])

# 调优 draw_margin 参数（基于训练集）
from scipy.optimize import minimize_scalar

def elo_cv_loss(margin):
    probs = elo_to_probs(X_train_ts[:, feature_names.index("elo_diff")], draw_margin=margin)
    return log_loss(y_train_ts, probs)

print("  [调优] 搜索最优 draw_margin...")
result = minimize_scalar(elo_cv_loss, bounds=(20, 300), method="bounded")
best_margin = result.x
print(f"  最优 draw_margin = {best_margin:.2f}  (CV LogLoss = {result.fun:.4f})")

# 训练集评估
elo_train_probs = elo_to_probs(X_train_ts[:, feature_names.index("elo_diff")], draw_margin=best_margin)
elo_val_probs = elo_to_probs(X_val_ts[:, feature_names.index("elo_diff")], draw_margin=best_margin)

elo_train_metrics = compute_metrics(y_train_ts, elo_train_probs, "Elo", "train_ts")
elo_val_metrics = compute_metrics(y_val_ts, elo_val_probs, "Elo", "val_ts")
print(f"  训练: {elo_train_metrics}")
print(f"  验证: {elo_val_metrics}")

# 锦标赛 GroupKFold CV
elo_group_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=tournament_groups)):
    probs_val = elo_to_probs(X[val_idx, feature_names.index("elo_diff")], draw_margin=best_margin)
    met = compute_metrics(y[val_idx], probs_val, "Elo", f"group_cv_fold{fold}")
    elo_group_scores.append(met)
    print(f"  Group CV Fold {fold}: val_acc={met['accuracy']:.4f}  ll={met['log_loss']:.4f}")

elo_group_avg = {
    "model": "Elo", "split": "group_cv_avg",
    "accuracy": round(np.mean([s["accuracy"] for s in elo_group_scores]), 4),
    "log_loss": round(np.mean([s["log_loss"] for s in elo_group_scores]), 4),
    "brier_score": round(np.mean([s["brier_score"] for s in elo_group_scores]), 4),
    "auc_ovr": round(np.mean([s["auc_ovr"] for s in elo_group_scores]), 4),
    "n": len(X),
}

# ---------------------------------------------------------------------------
# 7. MODEL 2: Poisson Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 2: Poisson 回归")
print("=" * 60)

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

def build_poisson_model(X_tr, y_home_tr, y_away_tr, X_va, team_home_idx, team_away_idx):
    """
    构建双 Poisson 模型：
      home_goals ~ Poisson(lambda_home)
      away_goals ~ Poisson(lambda_away)

    使用历史进攻/防守特征预测 lambda
    """
    # 使用特征: elo_diff, 近期进球/失球, fifa_points_diff
    poisson_feats = [
        "elo_diff", "fifa_points_diff", "fifa_rank_diff",
        "home_avg_goals_for_5", "home_avg_goals_against_5",
        "away_avg_goals_for_5", "away_avg_goals_against_5",
        "home_net_goals_5", "away_net_goals_5",
        "home_weighted_form_5", "away_weighted_form_5",
        "same_confederation", "is_neutral",
        "home_goal_conversion_rate", "away_goal_conversion_rate",
    ]
    avail = [f for f in poisson_feats if f in feature_names]

    X_tr_p = X_tr[:, [feature_names.index(f) for f in avail]]
    X_va_p = X_va[:, [feature_names.index(f) for f in avail]]

    # 标准化
    scaler = StandardScaler()
    X_tr_p = scaler.fit_transform(X_tr_p)
    X_va_p = scaler.transform(X_va_p)

    # 主队进球模型
    model_home = PoissonRegressor(alpha=0.01, max_iter=500)
    model_home.fit(X_tr_p, y_home_tr)

    # 客队进球模型
    model_away = PoissonRegressor(alpha=0.01, max_iter=500)
    model_away.fit(X_tr_p, y_away_tr)

    # 预测
    lambda_home = model_home.predict(X_va_p)
    lambda_away = model_away.predict(X_va_p)

    return lambda_home, lambda_away, model_home, model_away, scaler, avail


def poisson_match_probs(lambda_home, lambda_away, max_goals=10):
    """
    从两个独立泊松分布计算 3 类结果概率
    使用 Dixon-Coles 风格但无 tau 校正（简洁版）
    """
    n = len(lambda_home)
    probs = np.zeros((n, 3))  # [客胜, 平局, 主胜]

    goals = np.arange(max_goals + 1)
    for i in range(n):
        lh = lambda_home[i]
        la = lambda_away[i]
        # Poisson PMF
        pmf_h = sp_poisson.pmf(goals, lh)
        pmf_a = sp_poisson.pmf(goals, la)
        joint = np.outer(pmf_h, pmf_a)
        p_home = np.sum(joint * np.triu(np.ones((max_goals+1, max_goals+1)), k=1))
        p_away = np.sum(joint * np.tril(np.ones((max_goals+1, max_goals+1)), k=-1))
        p_draw = np.sum(joint * np.eye(max_goals+1))
        total = p_home + p_away + p_draw
        probs[i] = [p_away/total, p_draw/total, p_home/total]

    return probs


# 用实际进球数训练 Poisson
y_home = df_full["home_score"].values.astype(float)
y_away = df_full["away_score"].values.astype(float)

# 选定 Poisson 特定特征
avail_poiss = [f for f in [
    "elo_diff", "fifa_points_diff", "fifa_rank_diff",
    "home_avg_goals_for_5", "home_avg_goals_for_10",
    "home_avg_goals_against_5", "home_avg_goals_against_10",
    "away_avg_goals_for_5", "away_avg_goals_for_10",
    "away_avg_goals_against_5", "away_avg_goals_against_10",
    "home_net_goals_5", "away_net_goals_5",
    "home_net_goals_10", "away_net_goals_10",
    "home_weighted_form_5", "away_weighted_form_5",
    "home_weighted_form_10", "away_weighted_form_10",
    "same_confederation", "is_neutral", "is_actual_home",
    "home_goal_conversion_rate", "away_goal_conversion_rate",
    "home_win_rate_5", "away_win_rate_5",
    "home_days_since_last_match", "away_days_since_last_match",
    "match_importance", "k_factor",
] if f in feature_names]
X_tr_p = X_train_ts[:, [feature_names.index(f) for f in avail_poiss]]
X_va_p = X_val_ts[:, [feature_names.index(f) for f in avail_poiss]]
pois_scaler = StandardScaler()
X_tr_p_s = pois_scaler.fit_transform(X_tr_p)
X_va_p_s = pois_scaler.transform(X_va_p)

model_home = PoissonRegressor(alpha=0.1, max_iter=1000)
model_home.fit(X_tr_p_s, y_home[train_ts_idx])
model_away = PoissonRegressor(alpha=0.1, max_iter=1000)
model_away.fit(X_tr_p_s, y_away[train_ts_idx])

lambda_home_tr = model_home.predict(X_tr_p_s)
lambda_away_tr = model_away.predict(X_tr_p_s)
lambda_home_val = model_home.predict(X_va_p_s)
lambda_away_val = model_away.predict(X_va_p_s)

# 诊断
print(f"  主队 lambda: 训练 μ={np.mean(lambda_home_tr):.3f} σ={np.std(lambda_home_tr):.3f}, "
      f"验证 μ={np.mean(lambda_home_val):.3f} σ={np.std(lambda_home_val):.3f}")
print(f"  客队 lambda: 训练 μ={np.mean(lambda_away_tr):.3f} σ={np.std(lambda_away_tr):.3f}, "
      f"验证 μ={np.mean(lambda_away_val):.3f} σ={np.std(lambda_away_val):.3f}")
print(f"  实际: 主队 μ={np.mean(y_home[train_ts_idx]):.3f} σ={np.std(y_home[train_ts_idx]):.3f}, "
      f"客队 μ={np.mean(y_away[train_ts_idx]):.3f} σ={np.std(y_away[train_ts_idx]):.3f}")

pois_train_probs = poisson_match_probs(lambda_home_tr, lambda_away_tr)
pois_val_probs = poisson_match_probs(lambda_home_val, lambda_away_val)

pois_train_metrics = compute_metrics(y_train_ts, pois_train_probs, "Poisson", "train_ts")
pois_val_metrics = compute_metrics(y_val_ts, pois_val_probs, "Poisson", "val_ts")
print(f"  训练: {pois_train_metrics}")
print(f"  验证: {pois_val_metrics}")

# GroupKFold CV
pois_group_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=tournament_groups)):
    yh_tr = y_home[train_idx]
    yh_va = y_away[train_idx]  # 用于训练 away
    ya_va = y_home[val_idx]
    ya_va2 = y_away[val_idx]

    X_tr_f = X[train_idx][:, [feature_names.index(f) for f in avail_poiss]]
    X_va_f = X[val_idx][:, [feature_names.index(f) for f in avail_poiss]]

    scaler_f = StandardScaler()
    X_tr_f_s = scaler_f.fit_transform(X_tr_f)
    X_va_f_s = scaler_f.transform(X_va_f)

    mh = PoissonRegressor(alpha=0.1, max_iter=1000)
    mh.fit(X_tr_f_s, y_home[train_idx])
    ma = PoissonRegressor(alpha=0.1, max_iter=1000)
    ma.fit(X_tr_f_s, y_away[train_idx])

    lh = mh.predict(X_va_f_s)
    la = ma.predict(X_va_f_s)

    probs_f = poisson_match_probs(lh, la)
    met = compute_metrics(y[val_idx], probs_f, "Poisson", f"group_cv_fold{fold}")
    pois_group_scores.append(met)
    print(f"  Group CV Fold {fold}: val_acc={met['accuracy']:.4f}  ll={met['log_loss']:.4f}")

pois_group_avg = {
    "model": "Poisson", "split": "group_cv_avg",
    "accuracy": round(np.mean([s["accuracy"] for s in pois_group_scores]), 4),
    "log_loss": round(np.mean([s["log_loss"] for s in pois_group_scores]), 4),
    "brier_score": round(np.mean([s["brier_score"] for s in pois_group_scores]), 4),
    "auc_ovr": round(np.mean([s["auc_ovr"] for s in pois_group_scores]), 4),
    "n": len(X),
}

# ---------------------------------------------------------------------------
# 8. MODEL 3: XGBoost
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 3: XGBoost (多分类)")
print("=" * 60)

import xgboost as xgb

# 时间序列分割
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    min_child_weight=5,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)

# 早停
eval_set = [(X_train_ts, y_train_ts), (X_val_ts, y_val_ts)]
xgb_model.fit(
    X_train_ts, y_train_ts,
    eval_set=eval_set,
    verbose=False,
)

xgb_train_probs = xgb_model.predict_proba(X_train_ts)
xgb_val_probs = xgb_model.predict_proba(X_val_ts)

xgb_train_metrics = compute_metrics(y_train_ts, xgb_train_probs, "XGBoost", "train_ts")
xgb_val_metrics = compute_metrics(y_val_ts, xgb_val_probs, "XGBoost", "val_ts")
print(f"  训练: {xgb_train_metrics}")
print(f"  验证: {xgb_val_metrics}")

# 锦标赛分组 CV
xgb_group_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=tournament_groups)):
    xgb_fold = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=5,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_fold.fit(X[train_idx], y[train_idx])
    probs_f = xgb_fold.predict_proba(X[val_idx])
    met = compute_metrics(y[val_idx], probs_f, "XGBoost", f"group_cv_fold{fold}")
    xgb_group_scores.append(met)
    print(f"  Group CV Fold {fold}: val_acc={met['accuracy']:.4f}  ll={met['log_loss']:.4f}")

xgb_group_avg = {
    "model": "XGBoost", "split": "group_cv_avg",
    "accuracy": round(np.mean([s["accuracy"] for s in xgb_group_scores]), 4),
    "log_loss": round(np.mean([s["log_loss"] for s in xgb_group_scores]), 4),
    "brier_score": round(np.mean([s["brier_score"] for s in xgb_group_scores]), 4),
    "auc_ovr": round(np.mean([s["auc_ovr"] for s in xgb_group_scores]), 4),
    "n": len(X),
}
print(f"  Group CV Avg: acc={xgb_group_avg['accuracy']}  ll={xgb_group_avg['log_loss']}")

# 特征重要性
xgb_importance = xgb_model.feature_importances_
top_fea_idx = np.argsort(xgb_importance)[::-1][:15]
print("\n  XGBoost 特征重要性 Top-15:")
for i, idx in enumerate(top_fea_idx):
    print(f"    {i+1:2d}. {feature_names[idx]:35s} {xgb_importance[idx]:.4f}")

# ---------------------------------------------------------------------------
# 9. MODEL 4: LightGBM
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 4: LightGBM (多分类)")
print("=" * 60)

import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)

lgb_model.fit(
    X_train_ts, y_train_ts,
    eval_set=[(X_val_ts, y_val_ts)],
    callbacks=[lgb.callback.early_stopping(50), lgb.callback.log_evaluation(0)],
)

lgb_train_probs = lgb_model.predict_proba(X_train_ts)
lgb_val_probs = lgb_model.predict_proba(X_val_ts)

lgb_train_metrics = compute_metrics(y_train_ts, lgb_train_probs, "LightGBM", "train_ts")
lgb_val_metrics = compute_metrics(y_val_ts, lgb_val_probs, "LightGBM", "val_ts")
print(f"  训练: {lgb_train_metrics}")
print(f"  验证: {lgb_val_metrics}")

# 锦标赛分组 CV
lgb_group_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=tournament_groups)):
    lgb_fold = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_fold.fit(X[train_idx], y[train_idx])
    probs_f = lgb_fold.predict_proba(X[val_idx])
    met = compute_metrics(y[val_idx], probs_f, "LightGBM", f"group_cv_fold{fold}")
    lgb_group_scores.append(met)
    print(f"  Group CV Fold {fold}: val_acc={met['accuracy']:.4f}  ll={met['log_loss']:.4f}")

lgb_group_avg = {
    "model": "LightGBM", "split": "group_cv_avg",
    "accuracy": round(np.mean([s["accuracy"] for s in lgb_group_scores]), 4),
    "log_loss": round(np.mean([s["log_loss"] for s in lgb_group_scores]), 4),
    "brier_score": round(np.mean([s["brier_score"] for s in lgb_group_scores]), 4),
    "auc_ovr": round(np.mean([s["auc_ovr"] for s in lgb_group_scores]), 4),
    "n": len(X),
}
print(f"  Group CV Avg: acc={lgb_group_avg['accuracy']}  ll={lgb_group_avg['log_loss']}")

# ---------------------------------------------------------------------------
# 10. 汇总对比
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("基线模型对比汇总")
print("=" * 60)

all_metrics = [
    elo_train_metrics, elo_val_metrics, elo_group_avg,
    pois_train_metrics, pois_val_metrics, pois_group_avg,
    xgb_train_metrics, xgb_val_metrics, xgb_group_avg,
    lgb_train_metrics, lgb_val_metrics, lgb_group_avg,
]

df_results = pd.DataFrame(all_metrics)

# 保存 CSV
csv_path = os.path.join(OUT_DIR, "baseline_comparison.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n已保存: {csv_path}")
print(df_results.to_string(index=False))

# 按验证集 LogLoss 排序
val_results = df_results[df_results["split"] == "val_ts"].sort_values("log_loss")
cv_results = df_results[df_results["split"] == "group_cv_avg"].sort_values("log_loss")

print("\n--- 时间序列验证 Top-2 ---")
print(val_results[["model", "accuracy", "log_loss", "brier_score", "auc_ovr"]].to_string(index=False))

print("\n--- 分组 CV Top-2 ---")
print(cv_results[["model", "accuracy", "log_loss", "brier_score", "auc_ovr"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 11. 推荐结论
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("推荐结论")
print("=" * 60)

best_val_model = val_results.iloc[0]["model"]
best_val_ll = val_results.iloc[0]["log_loss"]
best_cv_model = cv_results.iloc[0]["model"]
best_cv_ll = cv_results.iloc[0]["log_loss"]

print(f"\n时间序列验证最佳: {best_val_model}  (LogLoss={best_val_ll:.4f})")
print(f"分组 CV 最佳: {best_cv_model}  (LogLoss={best_cv_ll:.4f})")

# 综合评估
print("\n综合推荐 Top-2:")
if best_val_model == best_cv_model:
    top1 = best_val_model
    # 找第二好
    second_val = val_results.iloc[1]["model"]
    second_cv = cv_results.iloc[1]["model"]
    top2 = second_val if second_val != top1 else second_cv
else:
    top1 = best_val_model
    top2 = best_cv_model

print(f"  #1 {top1} — 在 {'/'.join([str(v) for v in val_results[val_results['model']==top1].iloc[0].tolist()[:5]])} 等多个维度领先")
print(f"  #2 {top2} — 在 {'/'.join([str(v) for v in val_results[val_results['model']==top2].iloc[0].tolist()[:5]])} 方面表现突出")

# 保存模型
model_dir = os.path.join(OUT_DIR, "saved_models")
os.makedirs(model_dir, exist_ok=True)

# 保存 XGBoost
with open(os.path.join(model_dir, "xgb_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
print(f"  XGBoost 已保存")
# 保存 LightGBM (use pickle)
with open(os.path.join(model_dir, "lgb_model.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)
print(f"  LightGBM 已保存")
# 保存 Poisson
with open(os.path.join(model_dir, "poisson_home.pkl"), "wb") as f:
    pickle.dump(model_home, f)
with open(os.path.join(model_dir, "poisson_away.pkl"), "wb") as f:
    pickle.dump(model_away, f)
with open(os.path.join(model_dir, "poisson_scaler.pkl"), "wb") as f:
    pickle.dump(pois_scaler, f)
with open(os.path.join(model_dir, "poisson_features.pkl"), "wb") as f:
    pickle.dump(avail_poiss, f)
print(f"  Poisson 已保存")

print(f"\n模型已保存至: {model_dir}")
print("完成!")
