"""
Titanic V3 — 在V2极简特征上加XGBoost/CatBoost/Ensemble
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. 加载 ──
train = pd.read_csv(os.path.join(BASE, 'train.csv'))
test = pd.read_csv(os.path.join(BASE, 'test.csv'))
test_ids = test['PassengerId']

# ── 2. 极简特征 ──
def engineer(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r'(\w+)\.')
    rare = ['Don','Rev','Dr','Major','Col','Capt','Sir','Lady','Jonkheer','Countess','Mlle','Ms','Mme']
    df['Title'] = df['Title'].replace(rare, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    df['Sex'] = df['Sex'].map({'male':0,'female':1})
    df['Age'] = df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    cols = ['Pclass','Sex','Age','Fare_log','Embarked','Title',
            'FamilySize','IsAlone','HasCabin','SibSp','Parch']
    return df[cols]

X = engineer(train)
y = train['Survived'].astype(int)
X_test = engineer(test)

# ── 3. 多模型多seed CV ──
models = {
    'LR_C0.3': LogisticRegression(max_iter=5000, C=0.3, penalty='l2'),
    'RF_d5': RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, min_samples_split=5, random_state=42),
    'RF_d4': RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, min_samples_split=10, random_state=42),
    'XGB_d3': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, reg_alpha=1, reg_lambda=1, subsample=0.7, random_state=42, verbosity=0),
    'XGB_d4': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, reg_alpha=0.5, reg_lambda=0.5, subsample=0.7, random_state=42, verbosity=0),
    'CatBoost_d3': CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=5, random_seed=42, verbose=0),
    'CatBoost_d4': CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0),
}

seeds = [42, 123, 456, 789, 1024]
results = {}

for name, model in models.items():
    scores = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        s = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        scores.extend(s)
    results[name] = {'mean': np.mean(scores), 'std': np.std(scores)}
    print(f"{name:15s} | 5-seed CV: {results[name]['mean']:.4f} +- {results[name]['std']:.4f}")

# ── 4. 最优 ──
best_name = max(results, key=lambda k: results[k]['mean'])
print(f"\nBest: {best_name} (CV={results[best_name]['mean']:.4f})")

# ── 5. Ensemble ──
sorted_m = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
for topn in [2, 3]:
    names = [n for n, _ in sorted_m[:topn]]
    for vt in ['soft', 'hard']:
        estimators = [(n, models[n]) for n in names]
        ens = VotingClassifier(estimators, voting=vt)
        scores = []
        for seed in seeds:
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            s = cross_val_score(ens, X, y, cv=cv, scoring='accuracy')
            scores.extend(s)
        print(f"Top{topn}_Voting_{vt:4s} | 5-seed CV: {np.mean(scores):.4f} +- {np.std(scores):.4f}")

# ── 6. 生成两份提交 ──

# 6a. 最佳单模型
best_model = models[best_name]
best_model.fit(X, y)
preds_single = best_model.predict(X_test)
print(f"\nSingle ({best_name}): survived {preds_single.sum()}/{len(preds_single)} ({preds_single.mean():.1%})")

sub = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds_single.astype(int)})
sub.to_csv(os.path.join(BASE, 'v2', 'submission_v3_single.csv'), index=False)

# 6b. 最佳 Ensemble (Top2 Voting Hard)
top2 = [n for n, _ in sorted_m[:2]]
ens = VotingClassifier([(n, models[n]) for n in top2], voting='hard')
ens.fit(X, y)
preds_ens = ens.predict(X_test)
print(f"Ensemble (Top2_Hard): survived {preds_ens.sum()}/{len(preds_ens)} ({preds_ens.mean():.1%})")

sub2 = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds_ens.astype(int)})
sub2.to_csv(os.path.join(BASE, 'v2', 'submission_v3_ensemble.csv'), index=False)

print("\nDone: submission_v3_single.csv + submission_v3_ensemble.csv")
