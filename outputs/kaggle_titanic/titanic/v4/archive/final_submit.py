"""
V4 Final Submission: Multi-seed Top3 Hard Voting + Multi-seed Ensemble
策略：Top3 模型(CatBoost+GBDT+ET) × 5 seeds → Hard Voting → 最终预测
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier)
import warnings
warnings.filterwarnings('ignore')
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load
train = pd.read_csv(os.path.join(script_dir, 'X_train_fixed.csv'))
test = pd.read_csv(os.path.join(script_dir, 'X_test_fixed.csv'))
X = train.drop('Survived', axis=1)
y = train['Survived'].astype(int)
X_test = test

# Clean
for col in X.columns:
    if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce')
for col in X_test.columns:
    if X_test[col].dtype == 'object': X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
X, X_test = X.fillna(0), X_test.fillna(0)
common = [c for c in X.columns if c in X_test.columns]
X, X_test = X[common], X_test[common]

print(f"Features: {X.shape[1]}, Train: {X.shape[0]}, Test: {X_test.shape[0]}")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except: HAS_CAT = False

SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 7777, 8888, 9999]
N_SEEDS = 10

# ── Strategy 1: Multi-seed Top3 Hard Voting ──
# Top3 models: CatBoost_d4, GBDT_d4, ET_d6
all_preds = []

for seed in SEEDS:
    models = [
        ('CatBoost', CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=seed, verbose=0)) if HAS_CAT else None,
        ('GBDT', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=seed)),
        ('ET', ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=seed, n_jobs=-1)),
    ]
    models = [(n, m) for n, m in models if m is not None]
    ens = VotingClassifier(models, voting='hard')
    ens.fit(X, y)
    preds = ens.predict(X_test)
    all_preds.append(preds)
    # Progress
    if (seed_idx := SEEDS.index(seed) + 1) % 2 == 0:
        print(f"  Seed {seed} done ({seed_idx}/{N_SEEDS})")

# Majority vote across all seed ensembles
all_preds = np.array(all_preds)
final_preds = (all_preds.mean(axis=0) >= 0.5).astype(int)
print(f"\nStrategy 1 (Multi-seed Top3 Hard Voting): survived {final_preds.sum()}/{len(final_preds)} ({final_preds.mean():.1%})")

# ── Strategy 2: Multi-seed Individual Models → Aggregated Voting ──
# Train each model with multiple seeds, aggregate all
individual_preds = []
for seed in SEEDS:
    # CatBoost
    if HAS_CAT:
        cb = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=seed, verbose=0)
        cb.fit(X, y)
        individual_preds.append(cb.predict(X_test))
    # GBDT
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=seed)
    gb.fit(X, y)
    individual_preds.append(gb.predict(X_test))
    # ET
    et = ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    et.fit(X, y)
    individual_preds.append(et.predict(X_test))

indiv_arr = np.array(individual_preds)
final_preds_s2 = (indiv_arr.mean(axis=0) >= 0.5).astype(int)
print(f"Strategy 2 (Multi-seed All Models Aggregated): survived {final_preds_s2.sum()}/{len(final_preds_s2)} ({final_preds_s2.mean():.1%})")

# ── Strategy 3: Probabilistic (soft voting equivalent across seeds) ──
all_probas = []
for seed in SEEDS:
    probas = []
    if HAS_CAT:
        cb = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=seed, verbose=0)
        cb.fit(X, y)
        probas.append(cb.predict_proba(X_test)[:, 1])
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=seed)
    gb.fit(X, y)
    probas.append(gb.predict_proba(X_test)[:, 1])
    et = ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    et.fit(X, y)
    probas.append(et.predict_proba(X_test)[:, 1])

    avg_proba = np.mean(probas, axis=0)
    all_probas.append(avg_proba)

mean_proba = np.mean(all_probas, axis=0)
final_preds_s3 = (mean_proba >= 0.5).astype(int)
print(f"Strategy 3 (Multi-seed Avg Probability): survived {final_preds_s3.sum()}/{len(final_preds_s3)} ({final_preds_s3.mean():.1%})")

# For comparison, also check with tuned threshold
for thresh in [0.45, 0.48, 0.50, 0.52, 0.55]:
    preds_t = (mean_proba >= thresh).astype(int)
    print(f"  Threshold {thresh}: survived {preds_t.sum()}/{len(preds_t)} ({preds_t.mean():.1%})")

# ── Save All Submissions ──
def save_sub(preds, name):
    sub = pd.DataFrame({'PassengerId': np.arange(892, 892+len(preds)), 'Survived': preds.astype(int)})
    path = os.path.join(script_dir, f'submission_{name}.csv')
    sub.to_csv(path, index=False)
    return path

paths = []
paths.append(save_sub(final_preds, 'v4_multi_seed_top3'))
paths.append(save_sub(final_preds_s2, 'v4_multi_seed_all'))
paths.append(save_sub(final_preds_s3, 'v4_avg_probability'))

# Also save a threshold-optimized version if it produces ~38% survival
best_t = 0.50
best_diff = abs(final_preds_s3.mean() - 0.38)
for thresh in np.arange(0.40, 0.60, 0.01):
    preds_t = (mean_proba >= thresh).astype(int)
    diff = abs(preds_t.mean() - 0.38)
    if diff < best_diff:
        best_diff = diff
        best_t = thresh

preds_cal = (mean_proba >= best_t).astype(int)
paths.append(save_sub(preds_cal, f'v4_calibrated_t{best_t:.2f}'))
print(f"\nStrategy 4 (Calibrated threshold={best_t:.2f}): survived {preds_cal.sum()}/{len(preds_cal)} ({preds_cal.mean():.1%})")

print(f"\n=== ALL SUBMISSIONS ===")
for p in paths:
    sub = pd.read_csv(p)
    print(f"  {os.path.basename(p)}: survived {sub['Survived'].sum()}/{len(sub)} ({sub['Survived'].mean():.1%})")
print(f"\nFiles saved to: {script_dir}/")
