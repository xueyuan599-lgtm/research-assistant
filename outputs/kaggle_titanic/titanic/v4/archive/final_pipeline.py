"""
V4 Final Pipeline: Optuna Best Re-score + Stacking + Pseudo-Labeling + Submission
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, VotingClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os, json

# ── Paths ──
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, 'X_train_fixed.csv')
test_path = os.path.join(script_dir, 'X_test_fixed.csv')

print(f"Train: {train_path}")
print(f"Test:  {test_path}")

# ── Load ──
train = pd.read_csv(train_path)
X = train.drop('Survived', axis=1)
y = train['Survived'].astype(int)
X_test = pd.read_csv(test_path)

# Clean
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
X = X.fillna(0)
X_test = X_test.fillna(0)

# Align
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

print(f"X: {X.shape}, X_test: {X_test.shape}, y: {y.value_counts().to_dict()}")

# ── CV Setup ──
SEEDS = [42, 123, 456, 789, 1024]

def cv_score_full(model, X, y, seeds=SEEDS, n_folds=10):
    """Multi-seed CV with multiple metrics"""
    acc_scores, brier_scores, auc_scores = [], [], []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        # Accuracy
        acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        acc_scores.extend(acc)
        # AUC
        auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        auc_scores.extend(auc)
    return {
        'accuracy': (np.mean(acc_scores), np.std(acc_scores)),
        'auc': (np.mean(auc_scores), np.std(auc_scores)),
    }

# ── Build Best Models (with Optuna params) ──
print("\n=== Building Optimized Models ===")

# Optuna best params
optuna_params = {
    'GBDT': {'n_estimators': 254, 'max_depth': 4, 'learning_rate': 0.02581, 'subsample': 0.5559},
    'RF': {'n_estimators': 219, 'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 13},
    'ET': {'n_estimators': 104, 'max_depth': 6, 'min_samples_leaf': 8, 'min_samples_split': 7},
}

base_models = {
    'GBDT_Best': GradientBoostingClassifier(**optuna_params['GBDT'], random_state=42),
    'RF_Best': RandomForestClassifier(**optuna_params['RF'], random_state=42, n_jobs=-1),
    'ET_Best': ExtraTreesClassifier(**optuna_params['ET'], random_state=42, n_jobs=-1),
    'GBDT_d4_lr0.1': GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42),
    'RF_d5_l1': RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=1, min_samples_split=5, random_state=42, n_jobs=-1),
    'LR_C3': LogisticRegression(max_iter=5000, C=3.0, random_state=42),
}

# Try XGBoost and CatBoost
try:
    from xgboost import XGBClassifier
    base_models['XGB_d3_lr0.05'] = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, reg_alpha=1, reg_lambda=1, subsample=0.7, random_state=42, verbosity=0)
except: pass

try:
    from catboost import CatBoostClassifier
    base_models['CatBoost_d4_lr0.05'] = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0)
except: pass

# ── Score All ──
print("\n=== Full CV Scoring (10-fold x 5 seeds) ===\n")
scores = {}
for name, model in base_models.items():
    s = cv_score_full(model, X, y)
    scores[name] = s
    print(f"{name:25s} | Acc={s['accuracy'][0]:.4f}±{s['accuracy'][1]:.4f} | AUC={s['auc'][0]:.4f}±{s['auc'][1]:.4f}")

# ── Top Models for Ensemble ──
sorted_models = sorted(scores.items(), key=lambda x: x[1]['accuracy'][0], reverse=True)
print(f"\n=== Top Ranked ===")
for i, (name, s) in enumerate(sorted_models):
    print(f"  {i+1}. {name:25s} Acc={s['accuracy'][0]:.4f}")

# ── Voting Ensembles ──
print(f"\n=== Voting Ensembles ===")

for top_n in [2, 3, 4, 5]:
    top_names = [n for n, _ in sorted_models[:top_n]]
    for vt in ['soft', 'hard']:
        estimators = [(n, base_models[n]) for n in top_names]
        ens = VotingClassifier(estimators, voting=vt)
        s = cv_score_full(ens, X, y)
        print(f"Top{top_n}_Voting_{vt:4s} | Acc={s['accuracy'][0]:.4f}±{s['accuracy'][1]:.4f} | AUC={s['auc'][0]:.4f}")

# ── Stacking ──
print(f"\n=== Stacking Ensemble ===")

# Use top 5 as level-1 estimators
top5_names = [n for n, _ in sorted_models[:5]]
level1_estimators = [(n, base_models[n]) for n in top5_names]

# Level 2: Logistic Regression with strong regularization
for C in [0.01, 0.1, 0.5, 1.0]:
    meta = LogisticRegression(max_iter=5000, C=C, penalty='l2', random_state=42)
    stack = StackingClassifier(level1_estimators, final_estimator=meta, cv=5, stack_method='predict_proba')
    try:
        s = cv_score_full(stack, X, y)
        print(f"Stacking_Top5_LR_C{C} | Acc={s['accuracy'][0]:.4f}±{s['accuracy'][1]:.4f} | AUC={s['auc'][0]:.4f}")
    except Exception as e:
        print(f"Stacking_Top5_LR_C{C} | FAILED: {e}")

# ── Pseudo-Labeling ──
print(f"\n=== Pseudo-Labeling ===")

# Get best ensemble (Top3 Hard Voting based on scores)
best_ens_names = [n for n, _ in sorted_models[:3]]
best_ens = VotingClassifier([(n, base_models[n]) for n in best_ens_names], voting='soft')

# Generate OOF predictions for confidence calibration
print("Generating OOF predictions...")
oof_proba = cross_val_predict(best_ens, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba')
oof_preds = np.argmax(oof_proba, axis=1)
print(f"OOF Accuracy: {accuracy_score(y, oof_preds):.4f}")

# Fit on full data for test prediction
best_ens.fit(X, y)
test_proba = best_ens.predict_proba(X_test)
test_preds_base = best_ens.predict(X_test)
print(f"Base prediction survival rate: {test_preds_base.mean():.3f}")

# Pseudo-label: add high-confidence test samples to training
confidence = np.max(test_proba, axis=1)
high_conf_mask = confidence > 0.90
print(f"High confidence (p>0.90): {high_conf_mask.sum()}/{len(test_proba)} ({high_conf_mask.mean():.1%})")

if high_conf_mask.sum() > 10:
    X_aug = pd.concat([X, X_test[high_conf_mask]], axis=0)
    y_aug = pd.concat([y, pd.Series(test_preds_base[high_conf_mask], index=X_test.index[high_conf_mask])], axis=0)
    print(f"Augmented training: {X_aug.shape[0]} samples (+{high_conf_mask.sum()})")

    best_ens.fit(X_aug, y_aug)
    test_preds_pl = best_ens.predict(X_test)
    print(f"Pseudo-labeled survival rate: {test_preds_pl.mean():.3f}")
else:
    test_preds_pl = test_preds_base
    print("Too few high-confidence samples, skipping pseudo-labeling")

# ── Generate Submissions ──
print(f"\n=== Generating Submissions ===")

base_id_start = 892

# 1. Best single model
best_single_name = sorted_models[0][0]
best_single = base_models[best_single_name]
best_single.fit(X, y)
preds_single = best_single.predict(X_test)
sub1 = pd.DataFrame({'PassengerId': np.arange(base_id_start, base_id_start+len(preds_single)), 'Survived': preds_single.astype(int)})
sub1.to_csv(os.path.join(script_dir, 'submission_single_best.csv'), index=False)
print(f"1. Single best ({best_single_name}): survived {preds_single.sum()}/{len(preds_single)} ({preds_single.mean():.1%})")

# 2. Ensemble (Top3 Soft Voting)
top3_names = [n for n, _ in sorted_models[:3]]
ens3 = VotingClassifier([(n, base_models[n]) for n in top3_names], voting='soft')
ens3.fit(X, y)
preds_ens = ens3.predict(X_test)
sub2 = pd.DataFrame({'PassengerId': np.arange(base_id_start, base_id_start+len(preds_ens)), 'Survived': preds_ens.astype(int)})
sub2.to_csv(os.path.join(script_dir, 'submission_ensemble.csv'), index=False)
print(f"2. Ensemble Top3 Soft: survived {preds_ens.sum()}/{len(preds_ens)} ({preds_ens.mean():.1%})")

# 3. Pseudo-labeled (if applicable)
sub3 = pd.DataFrame({'PassengerId': np.arange(base_id_start, base_id_start+len(test_preds_pl)), 'Survived': test_preds_pl.astype(int)})
sub3.to_csv(os.path.join(script_dir, 'submission_pseudo_labeled.csv'), index=False)
print(f"3. Pseudo-labeled: survived {test_preds_pl.sum()}/{len(test_preds_pl)} ({test_preds_pl.mean():.1%})")

# ── Summary ──
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"Features: {X.shape[1]}")
print(f"Best CV (10-fold x 5 seeds):")
for i, (name, s) in enumerate(sorted_models[:5]):
    print(f"  {i+1}. {name:25s} Acc={s['accuracy'][0]:.4f}±{s['accuracy'][1]:.4f}")

print(f"\nSubmissions saved to {script_dir}/")
print(f"  - submission_single_best.csv")
print(f"  - submission_ensemble.csv")
print(f"  - submission_pseudo_labeled.csv")
print(f"\nExpected LB range: 0.77 - 0.80 (honest modeling with OOF encoding)")
print(f"DONE")
