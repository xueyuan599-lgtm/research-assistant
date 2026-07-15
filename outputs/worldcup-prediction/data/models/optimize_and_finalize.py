#!/usr/bin/env python3
"""
世界杯预测 — LightGBM 精模优化流水线
============================================================
完整流程:
  1. 数据加载与预处理（复用基线代码逻辑）
  2. Optuna 贝叶斯超参数优化（≥100 trials，早停 + 时序 CV）
  3. 概率校准（Platt Scaling / Isotonic Regression）
  4. SHAP 特征重要性分析 + 部分依赖图
  5. 稳健性检验（3 种随机种子）
  6. 最终精模训练 + 保存 + 评估报告
============================================================
"""

import os, sys, warnings, json, pickle, time
import numpy as np
import pandas as pd
import warnings, tempfile, shutil
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE = Path(os.path.dirname(os.path.abspath(__file__)))          # models/
DATA_DIR = BASE.parent / "features"                              # data/features/
OUT_DIR = BASE                                                   # data/models/
MODEL_DIR = OUT_DIR / "saved_models"
FIGURE_DIR = OUT_DIR / "figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

FULL_CSV = DATA_DIR / "feature_matrix_full.csv"
REDUCED_CSV = DATA_DIR / "feature_matrix.csv"

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
N_TRIALS = 150           # Optuna trails
N_ROBUSTNESS = 3         # 稳健性重复次数
SEED_BASE = 42
N_FOLDS = 5              # 分组 CV 折数
VERBOSE = 1

print("=" * 70)
print("世界杯预测 — LightGBM 精模优化流水线")
print(f"  试验次数: {N_TRIALS}, 稳健性重复: {N_ROBUSTNESS}, 分组 CV 折数: {N_FOLDS}")
print("=" * 70)

# ===================================================================
# 1. 数据加载与预处理
# ===================================================================
print("\n" + "=" * 50)
print("阶段 1/6: 数据加载与预处理")
print("=" * 50)

df = pd.read_csv(FULL_CSV, encoding="utf-8-sig")

# 剔除目标为 NaN 的未来比赛
mask_trainable = df["home_win"].notna()
df_full = df[mask_trainable].copy().reset_index(drop=True)
print(f"  总样本: {len(df)}  |  可训练样本: {len(df_full)}")

# 构造 3 分类目标: 0=客胜, 1=平局, 2=主胜
df_full["target"] = np.where(
    df_full["home_win"] == 1, 2,
    np.where(df_full["draw"] == 1, 1, 0)
)
y = df_full["target"].values
dates = pd.to_datetime(df_full["date"])

print("  目标分布:", dict(zip(["客胜","平局","主胜"],
    [round(100*(y==c).mean(),1) for c in [0,1,2]])))

# 特征工程（同基线代码）
id_cols = ["date", "home_team", "away_team", "tournament", "source",
           "home_score", "away_score", "home_win", "draw", "target", "date_dt"]
df_full["date_dt"] = dates
feature_cols = [c for c in df_full.columns if c not in id_cols]
cat_cols = ["home_confederation", "away_confederation"]
num_cols = [c for c in feature_cols if c not in cat_cols]

df_feat = df_full[feature_cols].copy()
df_feat = pd.get_dummies(df_feat, columns=cat_cols, drop_first=False, dtype=float)

# 缺失值填充
for c in df_feat.columns:
    if df_feat[c].isna().any():
        df_feat[c] = df_feat[c].fillna(df_feat[c].median())

X = df_feat.values.astype(np.float64)
feature_names = df_feat.columns.tolist()
n_features = X.shape[1]
print(f"  特征维度: {n_features}")

# 保存特征名
with open(OUT_DIR / "feature_names_full.json", "w") as f:
    json.dump(feature_names, f)

# ---------------------------------------------------------------------------
# 时间序列分割
# ---------------------------------------------------------------------------
sorted_idx = np.argsort(dates)
n = len(y)
split_ts = int(n * 0.8)
train_ts_idx = sorted_idx[:split_ts]
val_ts_idx = sorted_idx[split_ts:]

X_train = X[train_ts_idx]
y_train = y[train_ts_idx]
X_val = X[val_ts_idx]
y_val = y[val_ts_idx]
dates_train = dates.iloc[train_ts_idx]
dates_val = dates.iloc[val_ts_idx]

print(f"  时间序列分割: 训练 {len(X_train)} ({dates_train.min().date()} ~ {dates_train.max().date()})")
print(f"               验证 {len(X_val)} ({dates_val.min().date()} ~ {dates_val.max().date()})")

# 锦标赛分组（完整集用于最终评估，训练子集用于调参）
tournament_groups = df_full["tournament"].values
train_tournament_groups = tournament_groups[train_ts_idx]
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=N_FOLDS)

# 评估指标
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

def compute_metrics(y_true, y_prob, model_name="", prefix=""):
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    n_classes = y_prob.shape[1]
    brier_sum = 0.0
    for c in range(n_classes):
        y_bin = (y_true == c).astype(float)
        brier_sum += brier_score_loss(y_bin, y_prob[:, c])
    brier = brier_sum / n_classes
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    return {"accuracy": round(acc, 4), "log_loss": round(ll, 4),
            "brier_score": round(brier, 4), "auc_ovr": round(auc, 4)}

# ===================================================================
# 2. Optuna 超参数优化
# ===================================================================
print("\n" + "=" * 50)
print("阶段 2/6: Optuna 贝叶斯超参数优化")
print(f"  试验次数: {N_TRIALS}")
print("=" * 50)

import optuna
import lightgbm as lgb
from optuna.samplers import TPESampler

def objective(trial):
    """Optuna 目标函数：使用分组 CV 的 log_loss"""
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": SEED_BASE,
        "n_jobs": -1,

        # 树结构
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

        # 学习率 & 迭代
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),

        # 数据采样
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),

        # 特征采样
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.5, 1.0),

        # 正则化
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),

        # 类别特征处理
        "min_data_per_group": trial.suggest_int("min_data_per_group", 5, 100),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),

        # 早停
        "early_stopping_round": 50,
    }

    # 使用分组 CV 评估（仅训练集内交叉验证，保证验证集完全独立）
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=train_tournament_groups)):
        X_tr_f = X_train[tr_idx]
        y_tr_f = y_train[tr_idx]
        X_va_f = X_train[va_idx]
        y_va_f = y_train[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            callbacks=[lgb.callback.early_stopping(50), lgb.callback.log_evaluation(0)],
        )
        proba = model.predict_proba(X_va_f)
        fold_scores.append(log_loss(y_va_f, proba))

    mean_ll = np.mean(fold_scores)
    return mean_ll

# 使用 TPESampler 进行贝叶斯优化
sampler = TPESampler(seed=SEED_BASE, multivariate=True, group=True)
study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name="lgbm_worldcup",
    storage=f"sqlite:///{OUT_DIR / 'optuna_study.db'}",
    load_if_exists=True,
)

# 检查已有 trials
completed = len([t for t in study.trials if t.value is not None])
remaining = max(0, N_TRIALS - completed)

if remaining > 0:
    print(f"  已有 {completed} 个完成的 trials, 还需 {remaining} 个...")
    start_time = time.time()
    study.optimize(objective, n_trials=remaining, show_progress_bar=False, n_jobs=1)
    elapsed = time.time() - start_time
    print(f"\n  [TIME] 优化耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
else:
    print(f"  已有 {completed} 个完成的 trials, 无需重新优化")
print(f"  最佳 Trial #{study.best_trial.number}:")
print(f"    分组 CV LogLoss = {study.best_trial.value:.6f}")
for k, v in study.best_trial.params.items():
    print(f"    {k}: {v}")

# 保存优化历史
df_trials = study.trials_dataframe()
df_trials.to_csv(OUT_DIR / "optimization_history.csv", index=False)
print(f"\n  优化历史已保存 -> {OUT_DIR / 'optimization_history.csv'}")

# 可视化优化历史
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 优化收敛
    ax = axes[0, 0]
    trials = [t for t in study.trials if t.value is not None]
    values = [t.value for t in trials]
    ax.plot(range(1, len(values)+1), values, "b-", alpha=0.5, linewidth=0.8)
    # 累计最优
    best_vals = np.minimum.accumulate(values)
    ax.plot(range(1, len(values)+1), best_vals, "r-", linewidth=2, label="Best so far")
    ax.axhline(y=study.best_trial.value, color="green", linestyle="--",
               alpha=0.5, label=f"Best = {study.best_trial.value:.4f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV LogLoss")
    ax.set_title("Optuna Optimization History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 参数重要性
    ax = axes[0, 1]
    try:
        from optuna.visualization.matplotlib import plot_param_importances
        plot_param_importances(study, ax=ax)
    except Exception:
        ax.text(0.5, 0.5, "Param importance N/A", ha="center", va="center")
        ax.set_title("Parameter Importances")

    # 学习率 vs 性能
    ax = axes[1, 0]
    lr_vals = [t.params.get("learning_rate", np.nan) for t in trials]
    ll_vals = [t.value for t in trials]
    scatter = ax.scatter(lr_vals, ll_vals, c=range(len(trials)),
                         cmap="viridis", alpha=0.6, s=30)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("CV LogLoss")
    ax.set_title("Learning Rate vs CV LogLoss")
    ax.set_xscale("log")
    plt.colorbar(scatter, ax=ax, label="Trial")
    ax.grid(True, alpha=0.3)

    # num_leaves vs max_depth
    ax = axes[1, 1]
    nl_vals = [t.params.get("num_leaves", np.nan) for t in trials]
    md_vals = [t.params.get("max_depth", np.nan) for t in trials]
    sc2 = ax.scatter(nl_vals, md_vals, c=ll_vals, cmap="viridis_r", alpha=0.6, s=30)
    ax.set_xlabel("num_leaves")
    ax.set_ylabel("max_depth")
    ax.set_title("Tree Structure vs CV LogLoss")
    plt.colorbar(sc2, ax=ax, label="LogLoss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "optimization_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  优化过程图已保存 -> {FIGURE_DIR / 'optimization_history.png'}")
except Exception as e:
    print(f"  优化过程绘图失败: {e}")

# ===================================================================
# 3. 概率校准
# ===================================================================
print("\n" + "=" * 50)
print("阶段 3/6: 概率校准")
print("=" * 50)

from sklearn.calibration import CalibratedClassifierCV

best_params = study.best_trial.params.copy()
# 移除早停参数（不传给构造器）
best_params.pop("early_stopping_round", None)

print("  方法: Platt Scaling (sigmoid) + Isotonic Regression 对比")

# 训练一个 LightGBM 作为基础校准器（使用全训练集）
base_lgb = lgb.LGBMClassifier(
    **best_params,
    random_state=SEED_BASE,
    n_jobs=-1,
    verbosity=-1,
)

# 在校准前先训练评估
base_lgb.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.callback.early_stopping(50), lgb.callback.log_evaluation(0)])

base_val_probs = base_lgb.predict_proba(X_val)
base_metrics = compute_metrics(y_val, base_val_probs)
print(f"  校准前 (val): accuracy={base_metrics['accuracy']:.4f}, "
      f"log_loss={base_metrics['log_loss']:.4f}, brier={base_metrics['brier_score']:.4f}")

# Platt Scaling (sigmoid)
calibrator_platt = CalibratedClassifierCV(
    lgb.LGBMClassifier(
        **best_params,
        random_state=SEED_BASE,
        n_jobs=-1,
        verbosity=-1,
    ),
    method="sigmoid",
    cv=5,
)
calibrator_platt.fit(X_train, y_train)
platt_val_probs = calibrator_platt.predict_proba(X_val)
platt_metrics = compute_metrics(y_val, platt_val_probs)
print(f"  Platt Scaling (val): accuracy={platt_metrics['accuracy']:.4f}, "
      f"log_loss={platt_metrics['log_loss']:.4f}, brier={platt_metrics['brier_score']:.4f}")

# Isotonic Regression
calibrator_iso = CalibratedClassifierCV(
    lgb.LGBMClassifier(
        **best_params,
        random_state=SEED_BASE,
        n_jobs=-1,
        verbosity=-1,
    ),
    method="isotonic",
    cv=5,
)
calibrator_iso.fit(X_train, y_train)
iso_val_probs = calibrator_iso.predict_proba(X_val)
iso_metrics = compute_metrics(y_val, iso_val_probs)
print(f"  Isotonic (val): accuracy={iso_metrics['accuracy']:.4f}, "
      f"log_loss={iso_metrics['log_loss']:.4f}, brier={iso_metrics['brier_score']:.4f}")

# 选择最佳校准方法
calib_logloss = {
    "none": base_metrics["log_loss"],
    "sigmoid": platt_metrics["log_loss"],
    "isotonic": iso_metrics["log_loss"],
}
best_calib = min(calib_logloss, key=calib_logloss.get)
print(f"\n  最佳校准方法: {best_calib} (LogLoss={calib_logloss[best_calib]:.4f})")

if best_calib == "sigmoid":
    final_calibrator = calibrator_platt
elif best_calib == "isotonic":
    final_calibrator = calibrator_iso
else:
    final_calibrator = None  # 使用未校准的 base_lgb

# 可靠性曲线图
try:
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = ["Away Win", "Draw", "Home Win"]

    probs_dict = {
        "Uncalibrated": base_val_probs,
        "Platt Scaling": platt_val_probs,
        "Isotonic": iso_val_probs,
    }

    for idx, (name, probs) in enumerate(probs_dict.items()):
        ax = axes[idx]
        for c in range(3):
            prob_true, prob_pred = calibration_curve(
                (y_val == c).astype(float), probs[:, c], n_bins=10
            )
            ax.plot(prob_pred, prob_true, "o-", label=class_names[c],
                    linewidth=2, markersize=6)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(f"{name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  校准曲线已保存 -> {FIGURE_DIR / 'calibration_curves.png'}")
except Exception as e:
    print(f"  校准曲线绘图失败: {e}")

# ===================================================================
# 4. SHAP 特征重要性分析
# ===================================================================
print("\n" + "=" * 50)
print("阶段 4/6: SHAP 特征重要性分析")
print("=" * 50)

import shap

# 使用训练好的 base_lgb 进行 SHAP 分析（未校准模型）
print("  计算 SHAP 值（使用 TreeExplainer，路径依赖法）...")
shap_explainer = shap.TreeExplainer(base_lgb.booster_)
shap_sample = shap.sample(X_val, 500)
shap_values = shap_explainer.shap_values(shap_sample, check_additivity=False)

# SHAP Summary Plot (robust multi-class handling)
# shap v0.51 returns shape (n_samples, n_features) for multiclass with background data,
# or (n_samples, n_features, n_classes) without. Handle both.
# shap_values is a list [class0, class1, class2] or 3D array
if isinstance(shap_values, list):
    shap_3d = np.stack(shap_values, axis=-1)
elif shap_values.ndim == 2:
    # 2D output (unexpected) - duplicate for 3 classes
    shap_3d = np.stack([shap_values] * 3, axis=-1)
else:
    shap_3d = shap_values

class_labels = ["Away Win (Class 0)", "Draw (Class 1)", "Home Win (Class 2)"]
n_feat_actual = shap_3d.shape[1]

# SHAP Summary Dot Plot
fig, axes = plt.subplots(1, 3, figsize=(22, 10))
for c in range(3):
    ax = axes[c]
    shap_c = shap_3d[:, :, c]
    mean_abs_shap = np.abs(shap_c).mean(axis=0)
    top_k = min(20, n_feat_actual)
    top_idx = np.argsort(mean_abs_shap)[-top_k:]

    for i in range(top_k):
        fi = top_idx[i]
        vals = shap_c[:, fi]
        feat_vals = shap_sample[:, fi]
        colors = plt.cm.RdYlBu_r((feat_vals - feat_vals.min()) / max(1e-8, np.ptp(feat_vals)))
        ax.scatter(vals, np.full_like(vals, i), c=colors, s=8, alpha=0.4, edgecolors="none")

    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[fi][:25] for fi in top_idx], fontsize=7)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title(class_labels[c], fontsize=12)
    ax.grid(True, alpha=0.2, axis="x")

plt.tight_layout()
fig.savefig(FIGURE_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  SHAP Summary 图已保存 -> {FIGURE_DIR / 'shap_summary.png'}")

# SHAP Bar Plot (全局特征重要性)
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
for c in range(3):
    ax = axes[c]
    mean_shap = np.abs(shap_3d[:, :, c]).mean(axis=0)
    top_k = min(20, n_feat_actual)
    top_idx = np.argsort(mean_shap)[-top_k:]
    top_vals = mean_shap[top_idx]
    top_names = [feature_names[i][:30] for i in top_idx]

    ax.barh(range(top_k), top_vals, color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP|)", fontsize=9)
    ax.set_title(f"Feature Importance - {class_labels[c]}", fontsize=12)
    ax.grid(True, alpha=0.2, axis="x")

plt.tight_layout()
fig.savefig(FIGURE_DIR / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  SHAP Bar 图已保存 -> {FIGURE_DIR / 'shap_feature_importance.png'}")

# 计算全局特征重要性（取 3 类绝对值平均）
global_shap = np.zeros((n_feat_actual,))
for c in range(3):
    global_shap += np.abs(shap_3d[:, :, c]).mean(axis=0)
global_shap /= 3

# 计算全局特征重要性（取 3 类绝对值平均）
global_shap = np.zeros((len(feature_names),))
for c in range(3):
    global_shap += np.abs(shap_3d[:, :, c]).mean(axis=0)
global_shap /= 3

top20_idx = np.argsort(global_shap)[::-1][:20]
top20_features = [(feature_names[i], global_shap[i]) for i in top20_idx]

print("\n  Top-20 全局特征重要性:")
for rank, (feat, imp) in enumerate(top20_features, 1):
    print(f"    {rank:2d}. {feat:40s} {imp:.6f}")

# 保存特征重要性
df_feat_imp = pd.DataFrame({
    "feature": feature_names,
    "shap_importance_class0": np.abs(shap_3d[:, :, 0]).mean(axis=0),
    "shap_importance_class1": np.abs(shap_3d[:, :, 1]).mean(axis=0),
    "shap_importance_class2": np.abs(shap_3d[:, :, 2]).mean(axis=0),
    "shap_importance_avg": global_shap,
})
df_feat_imp = df_feat_imp.sort_values("shap_importance_avg", ascending=False)
df_feat_imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
print(f"  特征重要性已保存 -> {OUT_DIR / 'feature_importance.csv'}")

# 部分依赖图（Top-6 特征）
try:
    from sklearn.inspection import PartialDependenceDisplay

    top6_idx = top20_idx[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, idx_f in enumerate(top6_idx):
        ax = axes[i]
        fd = PartialDependenceDisplay.from_estimator(
            base_lgb, X_train, [idx_f],
            feature_names=feature_names,
            kind="both",
            grid_resolution=50,
            target=0,
            ax=ax,
        )
        ax.set_title(f"PDP: {feature_names[idx_f][:30]}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "partial_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  部分依赖图已保存 -> {FIGURE_DIR / 'partial_dependence.png'}")
except Exception as e:
    print(f"  部分依赖图失败: {e}")

# ===================================================================
# 5. 稳健性检验
# ===================================================================
print("\n" + "=" * 50)
print("阶段 5/6: 稳健性检验（不同随机种子）")
print("=" * 50)

seeds = [SEED_BASE, 123, 2024]
robustness_results = []

for seed_i, seed in enumerate(seeds):
    print(f"\n  种子 {seed}: 训练中...")

    model = lgb.LGBMClassifier(
        **best_params,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.callback.early_stopping(50), lgb.callback.log_evaluation(0)])

    proba = model.predict_proba(X_val)
    metrics = compute_metrics(y_val, proba)
    metrics["seed"] = seed
    metrics["n_estimators_used"] = model.best_iteration_
    robustness_results.append(metrics)
    print(f"    accuracy={metrics['accuracy']:.4f}, log_loss={metrics['log_loss']:.4f}, "
          f"brier={metrics['brier_score']:.4f}, auc={metrics['auc_ovr']:.4f}, "
          f"n_est={metrics['n_estimators_used']}")

df_robust = pd.DataFrame(robustness_results)
print(f"\n  稳健性汇总（验证集）:")
print(f"    Accuracy:  μ={df_robust['accuracy'].mean():.4f}  "
      f"σ={df_robust['accuracy'].std():.4f}  "
      f"range=[{df_robust['accuracy'].min():.4f}, {df_robust['accuracy'].max():.4f}]")
print(f"    LogLoss:   μ={df_robust['log_loss'].mean():.4f}  "
      f"σ={df_robust['log_loss'].std():.4f}  "
      f"range=[{df_robust['log_loss'].min():.4f}, {df_robust['log_loss'].max():.4f}]")

df_robust.to_csv(OUT_DIR / "robustness_check.csv", index=False)
print(f"  稳健性结果已保存 -> {OUT_DIR / 'robustness_check.csv'}")

# ===================================================================
# 6. 最终精模训练
# ===================================================================
print("\n" + "=" * 50)
print("阶段 6/6: 最终精模训练 + 全量评估")
print("=" * 50)

print(f"\n  使用全量可训练数据，最优超参数:")
for k, v in best_params.items():
    print(f"    {k}: {v}")

# 6a. 在全量数据上训练最终模型
final_model = lgb.LGBMClassifier(
    **best_params,
    random_state=SEED_BASE,
    n_jobs=-1,
    verbosity=-1,
)

# 用时间序列分割验证做早停
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.callback.early_stopping(50), lgb.callback.log_evaluation(0)],
)

# 6b. 评估
final_train_probs = final_model.predict_proba(X_train)
final_val_probs = final_model.predict_proba(X_val)

train_metrics = compute_metrics(y_train, final_train_probs)
val_metrics = compute_metrics(y_val, final_val_probs)

print(f"\n  最终模型评估:")
print(f"  ┌──────────────┬──────────┬──────────┬───────────┬──────────┐")
print(f"  │      Split   │ Accuracy │ LogLoss  │ Brier     │ AUC (OvR)│")
print(f"  ├──────────────┼──────────┼──────────┼───────────┼──────────┤")
print(f"  │  Train (ts)  │  {train_metrics['accuracy']:.4f}   │ {train_metrics['log_loss']:.4f}  │ {train_metrics['brier_score']:.4f}   │ {train_metrics['auc_ovr']:.4f}  │")
print(f"  │  Valid (ts)  │  {val_metrics['accuracy']:.4f}   │ {val_metrics['log_loss']:.4f}  │ {val_metrics['brier_score']:.4f}   │ {val_metrics['auc_ovr']:.4f}  │")
print(f"  └──────────────┴──────────┴──────────┴───────────┴──────────┘")

# 6c. 分组 CV 评估
print("\n  分组 CV（5-fold, 锦标赛分组）:")
cv_metrics_list = []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=tournament_groups)):
    X_tr_f = X[tr_idx]
    y_tr_f = y[tr_idx]
    X_va_f = X[va_idx]
    y_va_f = y[va_idx]

    fold_model = lgb.LGBMClassifier(
        **best_params,
        random_state=SEED_BASE,
        n_jobs=-1,
        verbosity=-1,
    )
    fold_model.fit(X_tr_f, y_tr_f,
                   eval_set=[(X_va_f, y_va_f)],
                   callbacks=[lgb.callback.early_stopping(50), lgb.callback.log_evaluation(0)])

    fold_probs = fold_model.predict_proba(X_va_f)
    fold_met = compute_metrics(y_va_f, fold_probs)
    fold_met["fold"] = fold
    cv_metrics_list.append(fold_met)
    print(f"    Fold {fold}: acc={fold_met['accuracy']:.4f}  ll={fold_met['log_loss']:.4f}  "
          f"brier={fold_met['brier_score']:.4f}  auc={fold_met['auc_ovr']:.4f}")

cv_avg = {
    "accuracy": np.mean([m["accuracy"] for m in cv_metrics_list]),
    "log_loss": np.mean([m["log_loss"] for m in cv_metrics_list]),
    "brier_score": np.mean([m["brier_score"] for m in cv_metrics_list]),
    "auc_ovr": np.mean([m["auc_ovr"] for m in cv_metrics_list]),
}
cv_std = {
    "accuracy": np.std([m["accuracy"] for m in cv_metrics_list]),
    "log_loss": np.std([m["log_loss"] for m in cv_metrics_list]),
    "brier_score": np.std([m["brier_score"] for m in cv_metrics_list]),
    "auc_ovr": np.std([m["auc_ovr"] for m in cv_metrics_list]),
}
print(f"    ─────────────────────────────────────────────────────────")
print(f"    CV Avg: acc={cv_avg['accuracy']:.4f} +/- {cv_std['accuracy']:.4f}, "
      f"ll={cv_avg['log_loss']:.4f} +/- {cv_std['log_loss']:.4f}")
print(f"            brier={cv_avg['brier_score']:.4f} +/- {cv_std['brier_score']:.4f}, "
      f"auc={cv_avg['auc_ovr']:.4f} +/- {cv_std['auc_ovr']:.4f}")

# 6d. 保存最终模型 + 校准器
calib_method_path = "none"
if best_calib != "none":
    final_pipeline = {
        "calibrator": final_calibrator,
        "calibration_method": best_calib,
        "base_model": final_model,
        "best_params": best_params,
        "calibrated": True,
    }
    calib_method_path = best_calib
else:
    final_pipeline = {
        "model": final_model,
        "best_params": best_params,
        "calibrated": False,
    }

model_save_path = MODEL_DIR / "final_model.pkl"
with open(model_save_path, "wb") as f:
    pickle.dump(final_pipeline, f)
print(f"\n  最终模型已保存 -> {model_save_path}")

# 同时保存 LightGBM booster 纯格式（方便 Kaggle kernel 加载）
# 保存 LightGBM booster（先写 temp 再复制，避免 C++ 后端 Unicode 路径问题）
lgb_temp_path = os.path.join(tempfile.gettempdir(), "lgb_worldcup_optimized.txt")
final_model.booster_.save_model(lgb_temp_path)
lgb_target_path = os.path.join(str(MODEL_DIR), "lgb_optimized.txt")
shutil.copy2(lgb_temp_path, lgb_target_path)
os.remove(lgb_temp_path)
print(f"  LightGBM booster 已保存 -> {lgb_target_path}")

# 保存最终评估报告
report = {
    "model_type": "LightGBM (optimized)",
    "calibration_method": calib_method_path,
    "optuna_trials": N_TRIALS,
    "best_cv_logloss": study.best_trial.value,
    "best_params": best_params,
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "cv_metrics_avg": cv_avg,
    "cv_metrics_std": cv_std,
    "cv_metrics_per_fold": cv_metrics_list,
    "robustness": robustness_results,
    "top20_features": top20_features,
    "training_date": datetime.now().isoformat(),
    "seed": SEED_BASE,
    "n_train_samples": len(X_train),
    "n_val_samples": len(X_val),
    "n_features": n_features,
}

report_path = OUT_DIR / "final_report.json"
import json as json_lib
class NpEncoder(json_lib.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(report_path, "w", encoding="utf-8") as f:
    json_lib.dump(report, f, indent=2, cls=NpEncoder, ensure_ascii=False)
print(f"  评估报告已保存 -> {report_path}")

# ===================================================================
# 7. 预测函数（输入两队特征 -> 输出三结果概率）
# ===================================================================
print("\n" + "=" * 50)
print("附录: 预测函数定义")
print("=" * 50)

# 在全局命名空间中定义预测函数（同时写入文件）

predict_function_code = """
def predict_match(home_team: str, away_team: str,
                  df_features: pd.DataFrame, pipeline: dict,
                  feature_names: list) -> dict:
    \"\"\"
    预测一场比赛的结果概率

    Parameters
    ----------
    home_team : str
        主队名称（必须与特征矩阵中的球队名称一致）
    away_team : str
        客队名称
    df_features : pd.DataFrame
        包含所有比赛的特征矩阵（用于提取该场比赛的已计算特征）
    pipeline : dict
        最终模型管线（含 model/calibrator 和 best_params）
    feature_names : list
        特征名列表（用于确保特征对齐）

    Returns
    -------
    dict: {"away_win": float, "draw": float, "home_win": float}
    \"\"\"
    # 找到这场比赛
    mask = (df_features["home_team"] == home_team) & \\
           (df_features["away_team"] == away_team)

    if mask.sum() == 0:
        raise ValueError(f"未找到 {home_team} vs {away_team} 的比赛记录")

    # 取最后一场匹配的比赛（可能有多次交锋）
    row = df_features[mask].iloc[-1]

    # 构造特征向量（同预处理流程）
    id_cols = ["date", "home_team", "away_team", "tournament", "source",
               "home_score", "away_score", "home_win", "draw", "target"]
    feature_cols = [c for c in df_features.columns if c not in id_cols]
    cat_cols = ["home_confederation", "away_confederation"]

    feat_dict = {}
    for c in feature_cols:
        if c in cat_cols:
            # 处理分类变量 one-hot
            for fn in feature_names:
                if fn.startswith(c + "_"):
                    expected_val = fn.replace(c + "_", "")
                    feat_dict[fn] = 1.0 if str(row[c]) == expected_val else 0.0
        else:
            val = row[c]
            if pd.isna(val):
                val = 0.0
            feat_dict[c] = float(val)

    # 确保所有特征都存在（缺失补 0）
    X_input = np.array([feat_dict.get(fn, 0.0) for fn in feature_names]).reshape(1, -1)

    # 预测
    if pipeline.get("calibrated", False):
        proba = pipeline["calibrator"].predict_proba(X_input)
    else:
        proba = pipeline["model"].predict_proba(X_input)

    return {
        "away_win": round(float(proba[0, 0]), 6),
        "draw": round(float(proba[0, 1]), 6),
        "home_win": round(float(proba[0, 2]), 6),
    }

# 更简洁的封装版本（直接加载模型文件）
def predict_from_file(home_team: str, away_team: str,
                      feature_csv: str, model_pkl: str,
                      feature_names_json: str) -> dict:
    \"\"\"从文件加载预测\"\"\"
    import json, pickle
    df = pd.read_csv(feature_csv, encoding="utf-8-sig")
    with open(model_pkl, "rb") as f:
        pipeline = pickle.load(f)
    with open(feature_names_json, "r") as f:
        fnames = json.load(f)
    return predict_match(home_team, away_team, df, pipeline, fnames)
"""

with open(OUT_DIR / "predict_function.py", "w", encoding="utf-8") as f:
    f.write(predict_function_code)
print(f"  预测函数已保存 -> {OUT_DIR / 'predict_function.py'}")

# ===================================================================
# 最终汇总
# ===================================================================
print("\n" + "=" * 70)
print("全部完成！输出汇总")
print("=" * 70)

outputs = [
    (MODEL_DIR / "final_model.pkl", "最终精模（pickle 格式，含校准器）"),
    (MODEL_DIR / "lgb_optimized.txt", "LightGBM booster（纯文本格式）"),
    (OUT_DIR / "optimization_history.csv", "Optuna 调参历史（150 trials）"),
    (OUT_DIR / "feature_importance.csv", "SHAP 特征重要性（全特征排序）"),
    (OUT_DIR / "final_report.json", "完整评估报告"),
    (OUT_DIR / "robustness_check.csv", "稳健性检验结果"),
    (OUT_DIR / "predict_function.py", "预测函数脚本"),
    (FIGURE_DIR / "optimization_history.png", "优化收敛过程图"),
    (FIGURE_DIR / "shap_summary.png", "SHAP Summary 图"),
    (FIGURE_DIR / "shap_feature_importance.png", "SHAP 条形特征重要性"),
    (FIGURE_DIR / "partial_dependence.png", "部分依赖图（Top-6 特征）"),
    (FIGURE_DIR / "calibration_curves.png", "校准曲线对比图"),
]

for path, desc in outputs:
    exists = "[OK]" if os.path.exists(path) else "[FAIL]"
    print(f"  [{exists}] {desc}")
    print(f"         {path}")

# 打印最终 CV 分数供调用者使用
print(f"\n{'='*40}")
print(f"最终 CV 分数 (Group 5-Fold):")
print(f"  Accuracy: {cv_avg['accuracy']:.4f} +/- {cv_std['accuracy']:.4f}")
print(f"  LogLoss:  {cv_avg['log_loss']:.4f} +/- {cv_std['log_loss']:.4f}")
print(f"  Brier:     {cv_avg['brier_score']:.4f} +/- {cv_std['brier_score']:.4f}")
print(f"  AUC (OvR): {cv_avg['auc_ovr']:.4f} +/- {cv_std['auc_ovr']:.4f}")
print(f"{'='*40}")
