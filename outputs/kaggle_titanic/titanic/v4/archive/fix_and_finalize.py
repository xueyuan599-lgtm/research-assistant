#!/usr/bin/env python3
"""
Fix script: Recompute metrics (Brier/LogLoss fixed), generate submission & report.
Runs after model_search.py has completed screening + Optuna.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))

# Load data
train = pd.read_csv(os.path.join(BASE, "X_train.csv"))
X = train.drop("Survived", axis=1)
y = train["Survived"].astype(int)
X_test = pd.read_csv(os.path.join(BASE, "X_test.csv"))

# ---------------------------------------------------------------------------
# Load Optuna best params and rebuild models
# ---------------------------------------------------------------------------
with open(os.path.join(BASE, "optuna_results.json")) as f:
    optuna_data = json.load(f)

optuna_models = {}

# GBDT
p = optuna_data["GBDT"]["best_params"]
optuna_models["GBDT_Optuna_Best"] = GradientBoostingClassifier(
    max_depth=p["max_depth"], learning_rate=p["learning_rate"],
    n_estimators=int(p["n_estimators"]),
    subsample=p.get("subsample", 1.0),
    min_samples_leaf=int(p.get("min_samples_leaf", 1)),
    max_features=p.get("max_features", 1.0),
    random_state=SEED
)

# LGB
p = optuna_data["LGB"]["best_params"]
optuna_models["LGB_Optuna_Best"] = LGBMClassifier(
    max_depth=p["max_depth"], learning_rate=p["learning_rate"],
    n_estimators=int(p["n_estimators"]),
    num_leaves=int(p["num_leaves"]),
    reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
    subsample=p.get("subsample", 1.0),
    colsample_bytree=p.get("colsample_bytree", 1.0),
    min_child_samples=int(p.get("min_child_samples", 20)),
    random_state=SEED, verbose=-1, force_col_wise=True
)

# CatBoost
p = optuna_data["CatBoost"]["best_params"]
optuna_models["CatBoost_Optuna_Best"] = CatBoostClassifier(
    depth=p["depth"], learning_rate=p["learning_rate"],
    iterations=int(p["iterations"]),
    l2_leaf_reg=p["l2_leaf_reg"],
    border_count=int(p.get("border_count", 254)),
    random_strength=p.get("random_strength", 1.0),
    random_seed=SEED, verbose=0, allow_writing_files=False
)

# XGB
p = optuna_data["XGB"]["best_params"]
optuna_models["XGB_Optuna_Best"] = XGBClassifier(
    max_depth=p["max_depth"], learning_rate=p["learning_rate"],
    n_estimators=int(p["n_estimators"]),
    reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
    subsample=p.get("subsample", 1.0),
    colsample_bytree=p.get("colsample_bytree", 1.0),
    min_child_weight=int(p.get("min_child_weight", 1)),
    random_state=SEED, eval_metric="logloss", verbosity=0
)

# RF
p = optuna_data["RF"]["best_params"]
optuna_models["RF_Optuna_Best"] = RandomForestClassifier(
    max_depth=p["max_depth"], n_estimators=int(p["n_estimators"]),
    min_samples_leaf=int(p["min_samples_leaf"]),
    min_samples_split=int(p["min_samples_split"]),
    max_features=p.get("max_features", 1.0),
    random_state=SEED, n_jobs=-1
)

# ET
p = optuna_data["ET"]["best_params"]
optuna_models["ET_Optuna_Best"] = ExtraTreesClassifier(
    max_depth=p["max_depth"], n_estimators=int(p["n_estimators"]),
    min_samples_leaf=int(p["min_samples_leaf"]),
    min_samples_split=int(p["min_samples_split"]),
    max_features=p.get("max_features", 1.0),
    random_state=SEED, n_jobs=-1
)

# Top screening models (non-Optuna)
screen_models = {
    "GBDT_d4_lr0.1": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED),
    "LGB_n100_d3_lr0.1": LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, num_leaves=15, random_state=SEED, verbose=-1, force_col_wise=True),
    "CatBoost_it300_d4_lr0.05": CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, verbose=0, allow_writing_files=False),
    "GBDT_d5_lr0.1": GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED),
    "LGB_n200_d3_lr0.05": LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1, force_col_wise=True),
    "CatBoost_it200_d3_lr0.1": CatBoostClassifier(iterations=200, depth=3, learning_rate=0.1, l2_leaf_reg=1, random_seed=SEED, verbose=0, allow_writing_files=False),
    "CatBoost_it200_d4_lr0.05": CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, verbose=0, allow_writing_files=False),
    "GBDT_d3_lr0.05": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED),
    "CatBoost_it300_d5_lr0.03": CatBoostClassifier(iterations=300, depth=5, learning_rate=0.03, l2_leaf_reg=5, random_seed=SEED, verbose=0, allow_writing_files=False),
    "LGB_n200_d4_lr0.1": LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, num_leaves=31, random_state=SEED, verbose=-1, force_col_wise=True),
    "XGB_n200_d4_lr0.1": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, reg_alpha=0, reg_lambda=1, subsample=0.8, random_state=SEED, eval_metric="logloss", verbosity=0),
}

# Combine all models
all_models = {}
all_models.update(optuna_models)
all_models.update(screen_models)

# ---------------------------------------------------------------------------
# Evaluate with cross_val_predict (fixes Brier/LogLoss NaN issue)
# ---------------------------------------------------------------------------
print("=" * 70)
print("  RECOMPUTING METRICS (Brier/LogLoss fixed)")
print("=" * 70)

results = []
seeds = [42, 123, 456]

for name, model in all_models.items():
    t0 = time.time()
    accs, briers, loglosses, aucs = [], [], [], []

    for s in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)

        # Get CV predictions
        y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        from sklearn.model_selection import cross_val_score
        auc_scores = cross_val_score(model, X, y, cv=cv2, scoring="roc_auc", n_jobs=-1)
        acc_scores = cross_val_score(model, X, y, cv=cv2, scoring="accuracy", n_jobs=-1)

        accs.extend(acc_scores)
        aucs.extend(auc_scores)

        # Brier and LogLoss computed from cross_val_predict
        briers.append(brier_score_loss(y, y_prob))
        loglosses.append(log_loss(y, y_prob))

    elapsed = time.time() - t0
    results.append({
        "name": name,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "brier": float(np.mean(briers)),
        "logloss": float(np.mean(loglosses)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "time": elapsed,
    })
    print(f"  {name:<35s} acc={results[-1]['acc_mean']:.4f}±{results[-1]['acc_std']:.4f}  "
          f"Brier={results[-1]['brier']:.4f}  LogLoss={results[-1]['logloss']:.4f}  "
          f"AUC={results[-1]['auc_mean']:.4f}  ({elapsed:.1f}s)")

results.sort(key=lambda r: r["acc_mean"], reverse=True)

# Save rankings
df = pd.DataFrame(results)
df.to_csv(os.path.join(BASE, "model_rankings.csv"), index=False)
print(f"\n[SAVED] model_rankings.csv ({len(results)} models)")

# ---------------------------------------------------------------------------
# Generate submission (best model)
# ---------------------------------------------------------------------------
best_name = results[0]["name"]
best_model = all_models[best_name]

print(f"\n[SUBMIT] Fitting {best_name} on full training data...")
best_model.fit(X, y)
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Get PassengerIds
test_orig_path = os.path.join(BASE, "..", "test.csv")
if os.path.exists(test_orig_path):
    test_orig = pd.read_csv(test_orig_path)
    passenger_ids = test_orig["PassengerId"].values
else:
    passenger_ids = np.arange(892, 892 + len(y_pred))

submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": y_pred})
submission_path = os.path.join(BASE, "submission_best_single.csv")
submission.to_csv(submission_path, index=False)
print(f"[SUBMIT] Saved → submission_best_single.csv")
print(f"[SUBMIT] Survival rate: {y_pred.mean():.3f} ({y_pred.sum()}/{len(y_pred)})")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
lines = []
lines.append("=" * 70)
lines.append("  MODEL SEARCH REPORT -- Titanic V4 Features")
lines.append("=" * 70)
lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"  Features: {X.shape[1]} | Training samples: {X.shape[0]} | Test samples: {X_test.shape[0]}")
lines.append(f"  Models evaluated: {len(results)}")
lines.append("")

lines.append("-" * 70)
lines.append("  FINAL LEADERBOARD")
lines.append("-" * 70)
header = f"{'Rank':<5} {'Model':<35s} {'Accuracy':>16s} {'Brier':>8s} {'LogLoss':>8s} {'AUC':>9s}  {'Time':>7s}"
lines.append(header)
lines.append("-" * len(header))
for i, r in enumerate(results, 1):
    lines.append(
        f"{i:<5} {r['name']:<35s} {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}  "
        f"{r['brier']:.4f}  {r['logloss']:.4f}  {r['auc_mean']:.4f}  {r['time']:.1f}s"
    )

lines.append("")
lines.append("-" * 70)
lines.append("  OPTUNA BEST PARAMETERS")
lines.append("-" * 70)
for fam, info in optuna_data.items():
    lines.append(f"\n  [{fam}] CV Accuracy: {info['best_cv_score']:.5f}")
    lines.append(f"  Params:")
    for k, v in info["best_params"].items():
        if isinstance(v, float):
            lines.append(f"    {k}: {v:.6f}")
        else:
            lines.append(f"    {k}: {v}")

lines.append("")
lines.append("-" * 70)
lines.append("  OVERFITTING DIAGNOSIS")
lines.append("-" * 70)
best = results[0]
gap = best["acc_mean"] - (1 - best["brier"])  # rough overfit indicator via Brier
lines.append(f"  Best model: {best['name']}")
lines.append(f"  CV Accuracy: {best['acc_mean']:.4f}")
lines.append(f"  Brier Score: {best['brier']:.4f} (lower = better calibration)")
lines.append(f"  LogLoss:     {best['logloss']:.4f}")
lines.append(f"  ROC AUC:     {best['auc_mean']:.4f}")
lines.append("")
lines.append("  Interpretation:")
bl = best["brier"]
if bl < 0.10:
    lines.append("  - Brier < 0.10: Excellent calibration, low overfitting risk")
elif bl < 0.15:
    lines.append("  - Brier 0.10-0.15: Good calibration, acceptable")
else:
    lines.append("  - Brier > 0.15: Potential overfitting or poor probability estimates")
lines.append("  - Check learning_curves.png for train vs CV gap at high sample sizes")
lines.append("  - On Titanic (N=891), ensemble models can overfit despite CV")
lines.append("  - Recommendation: use probability averaging (soft voting) from top 3-5 models")

lines.append("")
lines.append("-" * 70)
lines.append("  RECOMMENDATIONS")
lines.append("-" * 70)
lines.append(f"  1. Best single model for submission: {best_name}")
lines.append(f"  2. Kaggle submission: v4/submission_best_single.csv")
lines.append(f"  3. For better LB score: ensemble top 5 via soft voting or stacking")
lines.append(f"  4. Next step: feature ablation to identify redundant features")

report_text = "\n".join(lines)
report_path = os.path.join(BASE, "model_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"[REPORT] Saved → model_report.txt")
print(report_text)
print("\nDone!")
