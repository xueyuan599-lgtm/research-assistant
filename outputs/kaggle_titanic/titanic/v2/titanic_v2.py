"""
Titanic V2 — 极简+泛化
只做一件事：产出高分 submission.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── 1. 加载 ──
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv(os.path.join(BASE, 'train.csv'))
test = pd.read_csv(os.path.join(BASE, 'test.csv'))
test_ids = test['PassengerId']

# ── 2. 极简特征工程 ──
def engineer(df, train_df=None):
    df = df.copy()

    # Title
    df['Title'] = df['Name'].str.extract(r'(\w+)\.')
    rare = ['Don','Rev','Dr','Major','Col','Capt','Sir','Lady','Jonkheer','Countess','Mlle','Ms','Mme']
    df['Title'] = df['Title'].replace(rare, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title'] = LabelEncoder().fit_transform(df['Title'])

    # Sex
    df['Sex'] = df['Sex'].map({'male':0,'female':1})

    # Age — 分组中位数
    df['Age'] = df.groupby(['Pclass','Sex'])['Age'].transform(
        lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fare — log1p
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare_log'] = np.log1p(df['Fare'])

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    # Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # 选列
    cols = ['Pclass','Sex','Age','Fare_log','Embarked','Title',
            'FamilySize','IsAlone','HasCabin','SibSp','Parch']
    return df[cols]

X = engineer(train)
y = train['Survived'].astype(int)
X_test = engineer(test)

print(f"特征数: {X.shape[1]}, 训练集: {X.shape[0]}, 测试集: {X_test.shape[0]}")

# ── 3. 多模型 + 多seed CV ──
models = {
    'LR_C0.1': LogisticRegression(max_iter=5000, C=0.1, penalty='l2'),
    'LR_C0.3': LogisticRegression(max_iter=5000, C=0.3, penalty='l2'),
    'RF_d4': RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, min_samples_split=10),
    'RF_d5': RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, min_samples_split=5),
    'RF_d6': RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3, min_samples_split=5),
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
    print(f"{name:12s} | 5-seed CV: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")

# ── 4. 选最优 ──
best_name = max(results, key=lambda k: results[k]['mean'])
print(f"\n最优: {best_name} (CV={results[best_name]['mean']:.4f})")

# ── 5. 试Ensemble ──
# Top-3
sorted_models = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
top3_names = [n for n, _ in sorted_models]

for vt in ['soft','hard']:
    estimators = [(n, models[n]) for n in top3_names]
    ensemble = VotingClassifier(estimators, voting=vt)
    scores = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        s = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
        scores.extend(s)
    print(f"Top3_Voting_{vt:4s} | 5-seed CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# ── 6. 生成提交 ──
best_model = models[best_name]
best_model.fit(X, y)
preds = best_model.predict(X_test)
print(f"预测存活: {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

sub = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds})
sub.to_csv(os.path.join(BASE, 'v2', 'submission_v2.csv'), index=False)
print("\n✅ submission_v2.csv 已生成")
