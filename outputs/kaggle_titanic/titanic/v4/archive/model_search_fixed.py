"""
V4 Fixed Model Search — 使用修复后的 OOF Target Encoding 特征
在 38 个无泄漏特征上搜索最优模型 + Optuna 调参
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os, sys, time, json

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(BASE)  # titanic/
V4 = os.path.join(V4, 'titanic', 'v4') if not os.path.exists(os.path.join(V4, 'v4')) else os.path.join(BASE, '')

# Try to find the actual paths
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, 'X_train_fixed.csv')
test_path = os.path.join(script_dir, 'X_test_fixed.csv')

if not os.path.exists(train_path):
    # Maybe running from v4/
    train_path = os.path.join(os.path.dirname(script_dir), 'v4', 'X_train_fixed.csv')
    test_path = os.path.join(os.path.dirname(script_dir), 'v4', 'X_test_fixed.csv')

print(f"Loading: {train_path}")
print(f"Loading: {test_path}")

# ── Load ──
train = pd.read_csv(train_path)
X = train.drop('Survived', axis=1)
y = train['Survived'].astype(int)
X_test = pd.read_csv(test_path)

# Ensure no object columns
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

# Final NaN check
X = X.fillna(0)
X_test = X_test.fillna(0)

# Align
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

print(f"X_train: {X.shape}, X_test: {X_test.shape}")
print(f"Target rate: {y.mean():.3f}")
print(f"Features: {list(X.columns)}")

# ── CV Setup ──
SEEDS = [42, 123, 456, 789, 1024]
N_FOLDS = 10

def cv_score(model, X, y, seeds=SEEDS, n_folds=N_FOLDS, metrics=['accuracy']):
    """Multi-seed stratified CV"""
    results = {m: [] for m in metrics}
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for metric in metrics:
            if metric == 'accuracy':
                scoring = 'accuracy'
            elif metric == 'neg_brier':
                scoring = 'neg_brier_score'
            elif metric == 'neg_log_loss':
                scoring = 'neg_log_loss'
            elif metric == 'roc_auc':
                scoring = 'roc_auc'
            else:
                continue
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            if 'neg_' in scoring:
                scores = -scores
            results[metric].extend(scores)
    return {m: (np.mean(v), np.std(v)) for m, v in results.items()}

# ── Model Pool ──
models = {}

# Logistic Regression
for C in [0.01, 0.1, 0.3, 0.5, 1.0, 3.0]:
    models[f'LR_C{C}'] = LogisticRegression(max_iter=5000, C=C, penalty='l2', solver='lbfgs', random_state=42)

# Random Forest
for d in [4, 5, 6, 7]:
    for leaf in [1, 3]:
        models[f'RF_d{d}_l{leaf}'] = RandomForestClassifier(
            n_estimators=200, max_depth=d, min_samples_leaf=leaf,
            min_samples_split=max(leaf*2, 5), random_state=42, n_jobs=-1)

# ExtraTrees
for d in [5, 7, None]:
    models[f'ET_d{d}'] = ExtraTreesClassifier(
        n_estimators=200, max_depth=d, min_samples_leaf=3, random_state=42, n_jobs=-1)

# GradientBoosting
for d in [3, 4, 5]:
    for lr in [0.05, 0.1]:
        models[f'GBDT_d{d}_lr{lr}'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=d, learning_rate=lr,
            subsample=0.8, random_state=42)

# KNN
for k in [5, 10, 15, 20]:
    models[f'KNN_k{k}'] = KNeighborsClassifier(n_neighbors=k, weights='distance')

# MLP
for h in [(32,), (64, 32), (32, 16, 8)]:
    h_str = 'x'.join(map(str, h))
    models[f'MLP_h{h_str}'] = MLPClassifier(
        hidden_layer_sizes=h, alpha=0.01, max_iter=2000, random_state=42)

# XGBoost
try:
    from xgboost import XGBClassifier
    for d in [3, 4, 5, 6]:
        for lr in [0.03, 0.05, 0.1]:
            models[f'XGB_d{d}_lr{lr}'] = XGBClassifier(
                n_estimators=200, max_depth=d, learning_rate=lr,
                reg_alpha=1, reg_lambda=1, subsample=0.7, colsample_bytree=0.8,
                random_state=42, verbosity=0)
except ImportError:
    print("XGBoost not available, skipping...")

# CatBoost
try:
    from catboost import CatBoostClassifier
    for d in [3, 4, 5, 6]:
        for lr in [0.03, 0.05, 0.1]:
            models[f'CatBoost_d{d}_lr{lr}'] = CatBoostClassifier(
                iterations=300, depth=d, learning_rate=lr,
                l2_leaf_reg=3, random_seed=42, verbose=0)
except ImportError:
    print("CatBoost not available, skipping...")

# LightGBM
try:
    from lightgbm import LGBMClassifier
    for d in [3, 4, 5, 6]:
        for lr in [0.03, 0.05, 0.1]:
            for nl in [15, 31]:
                models[f'LGB_d{d}_lr{lr}_nl{nl}'] = LGBMClassifier(
                    n_estimators=200, max_depth=d, learning_rate=lr,
                    num_leaves=nl, reg_alpha=0.5, reg_lambda=0.5,
                    subsample=0.7, random_state=42, verbose=0)
except ImportError:
    print("LightGBM not available, skipping...")

print(f"\nTotal models to evaluate: {len(models)}")

# ── Evaluate All Models ──
results = {}
t_start = time.time()
for i, (name, model) in enumerate(models.items()):
    t0 = time.time()
    try:
        scores = cv_score(model, X, y)
        results[name] = {
            'accuracy_mean': scores['accuracy'][0],
            'accuracy_std': scores['accuracy'][1],
            'time': time.time() - t0
        }
        print(f"[{i+1:3d}/{len(models)}] {name:30s} | Acc={scores['accuracy'][0]:.4f}±{scores['accuracy'][1]:.4f} | {results[name]['time']:.1f}s")
    except Exception as e:
        print(f"[{i+1:3d}/{len(models)}] {name:30s} | FAILED: {str(e)[:60]}")
        continue

print(f"\nTotal time: {time.time() - t_start:.1f}s")

# ── Optuna Tuning (Top 6 models) ──
try:
    import optuna
    optuna_available = True
except ImportError:
    print("Optuna not available, skipping hyperparameter tuning...")
    optuna_available = False

if optuna_available and len(results) >= 6:
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True)
    top6_families = list(set([
        name.split('_')[0] for name, _ in sorted_models[:10]
    ]))[:6]
    print(f"\nOptuna tuning for families: {top6_families}")

    optuna_results = {}

    for family in top6_families:
        print(f"\n--- Optuna: {family} (50 trials) ---")

        def objective(trial):
            if family in ['LR']:
                model = LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 5.0, log=True),
                    penalty='l2', solver='lbfgs', max_iter=5000, random_state=42)
            elif family in ['RF', 'ET']:
                cls = RandomForestClassifier if family == 'RF' else ExtraTreesClassifier
                model = cls(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    random_state=42, n_jobs=-1)
            elif family in ['GBDT']:
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 2, 7),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    random_state=42)
            elif family in ['XGB']:
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 2, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
                    reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    random_state=42, verbosity=0)
            elif family in ['CatBoost']:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    iterations=trial.suggest_int('iterations', 50, 500),
                    depth=trial.suggest_int('depth', 2, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    l2_leaf_reg=trial.suggest_int('l2_leaf_reg', 1, 10),
                    random_seed=42, verbose=0)
            elif family in ['LGB']:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 2, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int('num_leaves', 7, 63),
                    reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
                    reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    random_state=42, verbose=0)
            else:
                return 0.5

            # 5-fold CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        optuna_results[family] = {
            'best_params': study.best_params,
            'best_score': study.best_value
        }
        print(f"  Best: {study.best_value:.4f} | Params: {study.best_params}")

    # Save optuna results
    with open(os.path.join(script_dir, 'optuna_results_fixed.json'), 'w') as f:
        json.dump(optuna_results, f, indent=2)
    print(f"\nOptuna results saved to optuna_results_fixed.json")

    # Add Optuna-best models to results
    for family, res in optuna_results.items():
        name = f'{family}_Optuna'
        results[name] = {
            'accuracy_mean': res['best_score'],
            'accuracy_std': 0.0,  # Will re-compute below
            'time': 0.0
        }

# ── Re-score top models with full CV (all seeds) ──
print("\n=== Re-scoring Top Models (10-fold × 5 seeds) ===")
top_models = sorted(results.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True)[:15]

final_results = {}
for name, _ in top_models:
    if name in models:
        model = models[name]
    elif 'Optuna' in name and optuna_available:
        family = name.replace('_Optuna', '')
        params = optuna_results[family]['best_params']
        if family == 'LR':
            model = LogisticRegression(C=params['C'], max_iter=5000, random_state=42)
        elif family == 'RF':
            model = RandomForestClassifier(**{k: v for k, v in params.items() if k != 'n_estimators'},
                                            n_estimators=params.get('n_estimators', 200),
                                            random_state=42, n_jobs=-1)
        elif family == 'XGB':
            from xgboost import XGBClassifier
            model = XGBClassifier(**params, random_state=42, verbosity=0)
        elif family == 'CatBoost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**params, random_seed=42, verbose=0)
        elif family == 'LGB':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**params, random_state=42, verbose=0)
        elif family == 'GBDT':
            model = GradientBoostingClassifier(**params, random_state=42)
        elif family == 'ET':
            model = ExtraTreesClassifier(**{k: v for k, v in params.items() if k != 'n_estimators'},
                                           n_estimators=params.get('n_estimators', 200),
                                           random_state=42, n_jobs=-1)
        else:
            continue
    else:
        continue

    scores = cv_score(model, X, y, seeds=SEEDS, metrics=['accuracy', 'neg_brier', 'neg_log_loss', 'roc_auc'])
    # Fit on full data for Brier/LogLoss
    model.fit(X, y)
    y_proba = model.predict_proba(X)[:, 1]
    final_results[name] = {
        'accuracy_mean': scores['accuracy'][0],
        'accuracy_std': scores['accuracy'][1],
        'brier_mean': scores['neg_brier'][0] if 'neg_brier' in scores else brier_score_loss(y, y_proba),
        'log_loss_mean': scores['neg_log_loss'][0] if 'neg_log_loss' in scores else log_loss(y, y_proba),
        'auc_mean': scores['roc_auc'][0] if 'roc_auc' in scores else roc_auc_score(y, y_proba),
    }
    print(f"{name:30s} | Acc={final_results[name]['accuracy_mean']:.4f}±{final_results[name]['accuracy_std']:.4f} "
          f"| Brier={final_results[name]['brier_mean']:.4f} | AUC={final_results[name]['auc_mean']:.4f}")

# ── Save rankings ──
rankings = pd.DataFrame(final_results).T.sort_values('accuracy_mean', ascending=False)
rankings.to_csv(os.path.join(script_dir, 'model_rankings_fixed.csv'))
print(f"\n=== Model Rankings ===\n{rankings.to_string()}")

# ── Best Model Submission ──
best_name = rankings.index[0]
print(f"\n=== Best Model: {best_name} ===")

# Re-fit best model on full training data
if best_name in models:
    best_model = models[best_name]
elif 'Optuna' in best_name:
    family = best_name.replace('_Optuna', '')
    params = optuna_results[family]['best_params']
    if family == 'LR':
        best_model = LogisticRegression(C=params['C'], max_iter=5000, random_state=42)
    elif family == 'RF':
        best_model = RandomForestClassifier(n_estimators=params.get('n_estimators', 200),
                                             max_depth=params['max_depth'],
                                             min_samples_leaf=params['min_samples_leaf'],
                                             min_samples_split=params['min_samples_split'],
                                             random_state=42, n_jobs=-1)
    elif family == 'XGB':
        from xgboost import XGBClassifier
        best_model = XGBClassifier(**params, random_state=42, verbosity=0)
    elif family == 'CatBoost':
        from catboost import CatBoostClassifier
        best_model = CatBoostClassifier(**params, random_seed=42, verbose=0)
    elif family == 'LGB':
        from lightgbm import LGBMClassifier
        best_model = LGBMClassifier(**params, random_state=42, verbose=0)
    elif family == 'GBDT':
        best_model = GradientBoostingClassifier(**params, random_state=42)
    elif family == 'ET':
        best_model = ExtraTreesClassifier(n_estimators=params.get('n_estimators', 200),
                                           max_depth=params['max_depth'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           min_samples_split=params['min_samples_split'],
                                           random_state=42, n_jobs=-1)

best_model.fit(X, y)
preds = best_model.predict(X_test)
print(f"Predicted survival rate: {preds.mean():.3f} ({preds.sum()}/{len(preds)})")

sub = pd.DataFrame({'PassengerId': np.arange(892, 892+len(preds)), 'Survived': preds.astype(int)})
sub_path = os.path.join(script_dir, 'submission_best_fixed.csv')
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")

# ── Ensemble: Top 5 Voting ──
print(f"\n=== Ensemble: Top 5 Voting ===")

top5_names = rankings.index[:5].tolist()
estimators = []
for i, name in enumerate(top5_names):
    model = models.get(name)
    if model is None and 'Optuna' in name:
        # Reconstruct optuna model
        family = name.replace('_Optuna', '')
        params = optuna_results[family]['best_params']
        if family == 'CatBoost':
            model = CatBoostClassifier(**params, random_seed=42+i, verbose=0)
        elif family == 'XGB':
            model = XGBClassifier(**params, random_state=42+i, verbosity=0)
        elif family == 'LGB':
            model = LGBMClassifier(**params, random_state=42+i, verbose=0)
        elif family == 'RF':
            model = RandomForestClassifier(n_estimators=params.get('n_estimators', 200),
                                           max_depth=params['max_depth'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           min_samples_split=params['min_samples_split'],
                                           random_state=42+i, n_jobs=-1)
        elif family == 'GBDT':
            model = GradientBoostingClassifier(**params, random_state=42+i)
    if model is not None:
        estimators.append((name, model))

for vt in ['soft', 'hard']:
    try:
        ens = VotingClassifier(estimators, voting=vt)
        scores = cv_score(ens, X, y)
        print(f"Top5_Voting_{vt}: Acc={scores['accuracy'][0]:.4f}±{scores['accuracy'][1]:.4f}")
        ens.fit(X, y)
        preds_ens = ens.predict(X_test)
        sub_ens = pd.DataFrame({'PassengerId': np.arange(892, 892+len(preds_ens)), 'Survived': preds_ens.astype(int)})
        sub_ens.to_csv(os.path.join(script_dir, f'submission_ensemble_{vt}.csv'), index=False)
        print(f"  Survival rate: {preds_ens.mean():.3f}")
    except Exception as e:
        print(f"Top5_Voting_{vt}: FAILED - {e}")

print("\n=== DONE ===")
print(f"Outputs in: {script_dir}")
