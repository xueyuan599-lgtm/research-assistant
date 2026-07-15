"""Final comparison: PCA vs SelectKBest vs Baseline"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
import os, warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(script_dir)

train_raw = pd.read_csv(os.path.join(BASE, 'train.csv'))
test_raw = pd.read_csv(os.path.join(BASE, 'test.csv'))

def engineer(df):
    df = df.copy()
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip() if ',' in str(x) else 'Other')
    title_map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Mme': 'Mrs', 'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
        'Master': 'Master', 'Dr': 'Professional', 'Rev': 'Clergy',
        'Col': 'Military', 'Major': 'Military', 'Capt': 'Military',
        'Lady': 'Honorific', 'Sir': 'Honorific', 'Countess': 'Honorific',
        'Don': 'Honorific', 'Dona': 'Honorific', 'Jonkheer': 'Honorific'}
    df['Title'] = df['Title'].map(title_map).fillna('Other')
    title_order = {'Mr': 0, 'Professional': 1, 'Clergy': 2, 'Military': 3,
                   'Honorific': 4, 'Other': 5, 'Master': 6, 'Miss': 7, 'Mrs': 8}
    df['Title_code'] = df['Title'].map(title_order)
    df['Sex_code'] = df['Sex'].map({'male': 0, 'female': 1})
    df['AgeMissing'] = df['Age'].isna().astype(int)
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['FareZero'] = (df['Fare'] == 0).astype(int)
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].replace(0, np.nan).fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare_log'] = np.log1p(df['Fare'].fillna(df['Fare'].median()))
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked_code'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['TicketGroupSize'] = df.groupby('Ticket')['PassengerId'].transform('count')
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['Sex_Pclass'] = df['Sex_code'] * (4 - df['Pclass'])
    df['Pclass_Age'] = df['Pclass'] * df['Age']
    return df[['Pclass', 'Sex_code', 'Age', 'AgeMissing', 'Fare_log', 'FareZero',
               'Embarked_code', 'Title_code', 'FamilySize', 'IsAlone', 'HasCabin',
               'TicketGroupSize', 'FarePerPerson', 'Sex_Pclass', 'Pclass_Age', 'SibSp', 'Parch']]

X = engineer(train_raw)
y = train_raw['Survived'].astype(int)
X_test = engineer(test_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

SEEDS = [42, 123, 456, 789, 1024]

def cv_multi(model, X_tr):
    scores = []
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        s = cross_val_score(model, X_tr, y, cv=cv, scoring='accuracy')
        scores.extend(s)
    return np.mean(scores), np.std(scores)

print("=" * 65)
print("5-SEED FULL CV COMPARISON")
print("=" * 65)

# 1. Baseline
lr = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
acc, std = cv_multi(lr, X_scaled)
print(f"1. Baseline 17feat + LR:          {acc:.4f} +/- {std:.4f}")
baseline = acc

# 2. PCA
for k in [12, 14, 15, 16]:
    pca = PCA(n_components=k, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_.sum()
    acc, std = cv_multi(LogisticRegression(max_iter=5000, C=1.0, random_state=42), X_pca)
    flag = "*** BEST ***" if acc > baseline else ""
    print(f"2. PCA({k:2d}) var={var:.3f} + LR:     {acc:.4f} +/- {std:.4f} {flag}")

# 3. SelectKBest
for k in [8, 10, 12]:
    skb = SelectKBest(mutual_info_classif, k=k)
    X_skb = skb.fit_transform(X_scaled, y)
    acc, std = cv_multi(LogisticRegression(max_iter=5000, C=1.0, random_state=42), X_skb)
    flag = "*** BEST ***" if acc > baseline else ""
    idx = skb.get_support(indices=True)
    names = [X.columns[i] for i in idx]
    print(f"3. SelectKBest({k:2d}) + LR:         {acc:.4f} +/- {std:.4f} {flag}")
    print(f"   Features: {names}")

# 4. RFE
rf_base = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
for k in [10, 12, 14]:
    rfe = RFE(estimator=rf_base, n_features_to_select=k)
    X_rfe = rfe.fit_transform(X_scaled, y)
    acc, std = cv_multi(LogisticRegression(max_iter=5000, C=1.0, random_state=42), X_rfe)
    flag = "*** BEST ***" if acc > baseline else ""
    idx = rfe.get_support(indices=True)
    names = [X.columns[i] for i in idx]
    print(f"4. RFE({k:2d}) + LR:                {acc:.4f} +/- {std:.4f} {flag}")

# 5. Simple RF (no LR, no dim reduction)
rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=42, n_jobs=-1)
acc, std = cv_multi(rf, X_scaled)
print(f"5. RF_d5_l5 (17feat):            {acc:.4f} +/- {std:.4f}")

# 6. Shallow RF
rf3 = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, min_samples_split=20, random_state=42, n_jobs=-1)
acc, std = cv_multi(rf3, X_scaled)
print(f"6. RF_d3_l8 (ultra-regularized):  {acc:.4f} +/- {std:.4f}")

print()

# === Generate ALL submissions ===
print("=" * 65)
print("GENERATING SUBMISSIONS")
print("=" * 65)

submissions = {}

# A: PCA(14) + LR
pca14 = PCA(n_components=14, random_state=42)
X_pca14 = pca14.fit_transform(X_scaled)
X_test_pca14 = pca14.transform(X_test_scaled)
m = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(X_pca14, y)
preds = m.predict(X_test_pca14)
submissions['pca14_lr'] = preds
print(f"PCA(14)+LR: survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

# B: PCA(15) + LR
pca15 = PCA(n_components=15, random_state=42)
X_pca15 = pca15.fit_transform(X_scaled)
X_test_pca15 = pca15.transform(X_test_scaled)
m = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(X_pca15, y)
preds = m.predict(X_test_pca15)
submissions['pca15_lr'] = preds
print(f"PCA(15)+LR: survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

# C: SelectKBest(10) + LR
skb10 = SelectKBest(mutual_info_classif, k=10)
X_skb10 = skb10.fit_transform(X_scaled, y)
X_test_skb10 = skb10.transform(X_test_scaled)
m = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(X_skb10, y)
preds = m.predict(X_test_skb10)
submissions['skb10_lr'] = preds
print(f"SelectKBest(10)+LR: survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

# D: SelectKBest(12) + LR
skb12 = SelectKBest(mutual_info_classif, k=12)
X_skb12 = skb12.fit_transform(X_scaled, y)
X_test_skb12 = skb12.transform(X_test_scaled)
m = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(X_skb12, y)
preds = m.predict(X_test_skb12)
submissions['skb12_lr'] = preds
print(f"SelectKBest(12)+LR: survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

# E: Baseline 17feat + LR
m = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(X_scaled, y)
preds = m.predict(X_test_scaled)
submissions['17feat_lr'] = preds
print(f"17feat+LR: survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

# F: Ultra-regularized RF
m = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, min_samples_split=20, random_state=42, n_jobs=-1)
m.fit(X_scaled, y)
preds = m.predict(X_test_scaled)
submissions['rf_ultra'] = preds
print(f"RF_d3_l8: survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

# G: RF d5 l5 (top performer from earlier)
m = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=42, n_jobs=-1)
m.fit(X_scaled, y)
preds = m.predict(X_test_scaled)
submissions['rf_d5_l5'] = preds
print(f"RF_d5_l5: survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

# Save all
for name, preds in submissions.items():
    sub = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': preds.astype(int)})
    sub.to_csv(os.path.join(script_dir, f'submission_{name}.csv'), index=False)

print(f"\nAll saved to {script_dir}/")
print("Files: " + ", ".join(f"submission_{n}.csv" for n in submissions))
