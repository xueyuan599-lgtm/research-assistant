"""
V4 Simpler: 深度特征工程 + 极简模型
策略：保留 V4 的优质特征构造，但只用 10-15 个最稳定特征 + LR/浅RF
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load raw data ──
BASE = os.path.dirname(script_dir)  # v4/ -> titanic/
train_raw = pd.read_csv(os.path.join(BASE, 'train.csv'))
test_raw = pd.read_csv(os.path.join(BASE, 'test.csv'))

print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

# ── 2. Deep Feature Engineering (keep best, drop target-encoded) ──
def engineer(df, is_train=True):
    df = df.copy()

    # Title (refined, NO target encoding - use LabelEncode only)
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip() if ',' in str(x) else 'Other')
    title_map = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Mme': 'Mrs', 'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
        'Master': 'Master', 'Dr': 'Professional', 'Rev': 'Clergy',
        'Col': 'Military', 'Major': 'Military', 'Capt': 'Military',
        'Lady': 'Honorific', 'Sir': 'Honorific', 'Countess': 'Honorific',
        'Don': 'Honorific', 'Dona': 'Honorific', 'Jonkheer': 'Honorific',
    }
    df['Title'] = df['Title'].map(title_map).fillna('Other')
    # Numeric encode Title (ordered by survival rate)
    title_order = {'Mr': 0, 'Professional': 1, 'Clergy': 2, 'Military': 3,
                   'Honorific': 4, 'Other': 5, 'Master': 6, 'Miss': 7, 'Mrs': 8}
    df['Title_code'] = df['Title'].map(title_order)

    # Sex
    df['Sex_code'] = df['Sex'].map({'male': 0, 'female': 1})

    # Age: group median + missing flag
    df['AgeMissing'] = df['Age'].isna().astype(int)
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fare: log1p + zero flag
    df['FareZero'] = (df['Fare'] == 0).astype(int)
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].replace(0, np.nan).fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare_log'] = np.log1p(df['Fare'].fillna(df['Fare'].median()))

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked_code'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Deck → HasCabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Ticket → group size
    df['TicketGroupSize'] = df.groupby('Ticket')['PassengerId'].transform('count')

    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Key interactions
    df['Sex_Pclass'] = df['Sex_code'] * (4 - df['Pclass'])  # female=1 * (4-Pclass): female_P1=3, female_P3=1, male_*=0
    df['Pclass_Age'] = df['Pclass'] * df['Age']

    # Select columns - NO target encoding, fewer features, cleaner
    cols = [
        'Pclass', 'Sex_code', 'Age', 'AgeMissing', 'Fare_log', 'FareZero',
        'Embarked_code', 'Title_code', 'FamilySize', 'IsAlone', 'HasCabin',
        'TicketGroupSize', 'FarePerPerson', 'Sex_Pclass', 'Pclass_Age',
        'SibSp', 'Parch'
    ]
    return df[cols]

X = engineer(train_raw)
y = train_raw['Survived'].astype(int)
X_test = engineer(test_raw, is_train=False)

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print(f"Features: {X.shape[1]}")
print(f"Feature list: {list(X.columns)}")

# ── 3. Simple models ──
SEEDS = [42, 123, 456, 789, 1024]
print(f"\n{'Model':<35s} {'CV Acc':<16s} {'Std':<10s}")

best_name, best_score, best_model = '', 0, None
results = {}

# Logistic Regression (various C)
for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
    model = LogisticRegression(max_iter=5000, C=C, penalty='l2', solver='lbfgs', random_state=42)
    scores = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        s = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        scores.extend(s)
    name = f'LR_C{C}'
    results[name] = (np.mean(scores), np.std(scores))
    print(f"{name:<35s} {np.mean(scores):.4f}            {np.std(scores):.4f}")
    if np.mean(scores) > best_score:
        best_score, best_name, best_model = np.mean(scores), name, LogisticRegression(max_iter=5000, C=C, random_state=42)

# Random Forest (shallow depth, regularized)
for d in [3, 4, 5]:
    for leaf in [3, 5, 8]:
        for split in [10, 20]:
            model = RandomForestClassifier(n_estimators=200, max_depth=d,
                                           min_samples_leaf=leaf, min_samples_split=split,
                                           random_state=42, n_jobs=-1)
            scores = []
            for seed in SEEDS[:3]:  # Fewer seeds for RF to save time
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
                s = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                scores.extend(s)
            name = f'RF_d{d}_l{leaf}_s{split}'
            results[name] = (np.mean(scores), np.std(scores))
            if np.mean(scores) > best_score:
                best_score, best_name = np.mean(scores), name
                best_model = RandomForestClassifier(n_estimators=200, max_depth=d,
                                                    min_samples_leaf=leaf, min_samples_split=split,
                                                    random_state=42, n_jobs=-1)
            if np.mean(scores) > 0.83:
                print(f"{name:<35s} {np.mean(scores):.4f}            {np.std(scores):.4f}")

# GBDT (shallow, slow learning)
for d in [3, 4]:
    for lr in [0.03, 0.05, 0.1]:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=d, learning_rate=lr,
                                           subsample=0.7, min_samples_leaf=5,
                                           random_state=42)
        scores = []
        for seed in SEEDS[:3]:
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            s = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            scores.extend(s)
        name = f'GBDT_d{d}_lr{lr}'
        results[name] = (np.mean(scores), np.std(scores))
        if np.mean(scores) > 0.83:
            print(f"{name:<35s} {np.mean(scores):.4f}            {np.std(scores):.4f}")
        if np.mean(scores) > best_score:
            best_score, best_name = np.mean(scores), name
            best_model = GradientBoostingClassifier(n_estimators=100, max_depth=d, learning_rate=lr,
                                                    subsample=0.7, min_samples_leaf=5, random_state=42)

# ── 4. Best model → submission ──
print(f"\n=== Best: {best_name} (CV={best_score:.4f}) ===")
best_model.fit(X_scaled, y)
preds = best_model.predict(X_test_scaled)
print(f"Survival rate: {preds.mean():.3f} ({preds.sum()}/{len(preds)})")

sub = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': preds.astype(int)})
sub_path = os.path.join(script_dir, 'submission_v4_simpler.csv')
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}")

# ── 5. Also try Top3 simple ensemble ──
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
top3 = sorted_results[:3]
print(f"\nTop 3 models:")
for name, (acc, std) in top3:
    print(f"  {name}: {acc:.4f} +/- {std:.4f}")

# Save rankings
with open(os.path.join(script_dir, 'simpler_rankings.txt'), 'w') as f:
    for name, (acc, std) in sorted_results:
        f.write(f"{name:<40s} {acc:.4f} +/- {std:.4f}\n")
