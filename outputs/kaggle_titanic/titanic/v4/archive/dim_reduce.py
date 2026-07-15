"""
降维 + 简化模型：PCA / SelectKBest / RFE + LR/RF
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
import warnings
warnings.filterwarnings('ignore')
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(script_dir)

# Load
train_raw = pd.read_csv(os.path.join(BASE, 'train.csv'))
test_raw = pd.read_csv(os.path.join(BASE, 'test.csv'))

# Same feature engineering as simpler_models.py
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
    cols = ['Pclass', 'Sex_code', 'Age', 'AgeMissing', 'Fare_log', 'FareZero',
            'Embarked_code', 'Title_code', 'FamilySize', 'IsAlone', 'HasCabin',
            'TicketGroupSize', 'FarePerPerson', 'Sex_Pclass', 'Pclass_Age', 'SibSp', 'Parch']
    return df[cols]

X = engineer(train_raw)
y = train_raw['Survived'].astype(int)
X_test = engineer(test_raw)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
feature_names = X.columns.tolist()

# CV function
def eval_model(X_tr, model, n_seeds=3):
    scores = []
    for seed in [42, 123, 456][:n_seeds]:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        s = cross_val_score(model, X_tr, y, cv=cv, scoring='accuracy')
        scores.extend(s)
    return np.mean(scores), np.std(scores)

# Baseline (all 17 features)
lr = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
base_acc, base_std = eval_model(X_scaled, lr)
print(f"Baseline (17 features, LR_C1.0): {base_acc:.4f} +/- {base_std:.4f}")

# ── 1. PCA ──
print(f"\n{'='*50}")
print(f"PCA Dimension Reduction")
print(f"{'='*50}")
for k in [3, 5, 8, 10, 12, 15]:
    pca = PCA(n_components=k, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    lr_pca = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    acc, std = eval_model(X_pca, lr_pca)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA({k:2d}) var={var:.3f} | CV={acc:.4f} +/- {std:.4f}")
    if acc > base_acc:
        print(f"    ^^^ BETTER than baseline!")

# ── 2. SelectKBest (Mutual Information) ──
print(f"\n{'='*50}")
print(f"SelectKBest (Mutual Information)")
print(f"{'='*50}")
mi = mutual_info_classif(X_scaled, y, random_state=42)
mi_ranking = sorted(zip(feature_names, mi), key=lambda x: x[1], reverse=True)
print("Top features by MI:")
for i, (name, score) in enumerate(mi_ranking[:15]):
    print(f"  {i+1:2d}. {name:25s} MI={score:.4f}")

for k in [5, 8, 10, 12, 15]:
    selector = SelectKBest(mutual_info_classif, k=k)
    X_sel = selector.fit_transform(X_scaled, y)
    X_test_sel = selector.transform(X_test_scaled)
    selected_idx = selector.get_support(indices=True)
    selected_names = [feature_names[i] for i in selected_idx]
    lr_sel = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    acc, std = eval_model(X_sel, lr_sel)
    flag = " *** BETTER ***" if acc > base_acc else ""
    print(f"  SelectKBest({k:2d}): CV={acc:.4f} +/- {std:.4f}{flag}")
    print(f"    Features: {selected_names}")

# ── 3. RFE ──
print(f"\n{'='*50}")
print(f"RFE (Recursive Feature Elimination)")
print(f"{'='*50}")
for k in [5, 8, 10, 12, 15]:
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=k)
    X_rfe = rfe.fit_transform(X_scaled, y)
    X_test_rfe = rfe.transform(X_test_scaled)
    selected_idx = rfe.get_support(indices=True)
    selected_names = [feature_names[i] for i in selected_idx]
    lr_rfe = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    acc, std = eval_model(X_rfe, lr_rfe)
    flag = " *** BETTER ***" if acc > base_acc else ""
    print(f"  RFE({k:2d}): CV={acc:.4f} +/- {std:.4f}{flag}")
    print(f"    Features: {selected_names}")

# ── 4. Pure top-K by MI + RF importance ──
print(f"\n{'='*50}")
print(f"Combined: Top-K by RF Importance")
print(f"{'='*50}")
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)
rf_imp = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: x[1], reverse=True)
print("Top features by RF importance:")
for i, (name, score) in enumerate(rf_imp[:15]):
    print(f"  {i+1:2d}. {name:25s} imp={score:.4f}")

# Use top-K by RF importance
for k in [5, 8, 10, 12, 15]:
    top_names = [n for n, _ in rf_imp[:k]]
    top_idx = [feature_names.index(n) for n in top_names]
    X_top = X_scaled[:, top_idx]
    X_test_top = X_test_scaled[:, top_idx]
    lr_top = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    acc, std = eval_model(X_top, lr_top)
    flag = " *** BETTER ***" if acc > base_acc else ""
    print(f"  RF_Top({k:2d}): CV={acc:.4f} +/- {std:.4f}{flag}")

# Also try RF directly with top-K (no LR)
for k in [5, 8, 10, 12]:
    top_names = [n for n, _ in rf_imp[:k]]
    top_idx = [feature_names.index(n) for n in top_names]
    X_top = X_scaled[:, top_idx]
    X_test_top = X_test_scaled[:, top_idx]
    rf_top = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                    min_samples_split=10, random_state=42, n_jobs=-1)
    acc, std = eval_model(X_top, rf_top)
    print(f"  RF_Top({k:2d}) + RF_model: CV={acc:.4f} +/- {std:.4f}")

# ── 5. Generate best submission ──
# Try the best approach (likely 10-12 features)
best_k = 12  # default, will override
best_acc = base_acc
best_X = X_scaled
best_X_test = X_test_scaled
best_names = feature_names

# Test top K with both LR and shallow RF
for k in [8, 10, 12]:
    top_names = [n for n, _ in rf_imp[:k]]
    top_idx = [feature_names.index(n) for n in top_names]
    X_top = X_scaled[:, top_idx]

    for model_name, model in [
        ('LR_C1.0', LogisticRegression(max_iter=5000, C=1.0, random_state=42)),
        ('RF_d5_l5', RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=42, n_jobs=-1)),
        ('RF_d4_l5', RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, min_samples_split=10, random_state=42, n_jobs=-1)),
    ]:
        acc, std = eval_model(X_top, model)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_X = X_top
            best_X_test = X_test_scaled[:, top_idx]
            best_names = top_names
            best_model = model
            print(f"\n  NEW BEST: {k} features + {model_name} = {acc:.4f}")

print(f"\n{'='*50}")
print(f"FINAL: {len(best_names)} features")
print(f"CV: {best_acc:.4f}")
print(f"Features: {best_names}")

# Fit & predict
best_model.fit(best_X, y)
preds = best_model.predict(best_X_test)
print(f"Survival rate: {preds.mean():.3f} ({preds.sum()}/{len(preds)})")

sub = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': preds.astype(int)})
sub_path = os.path.join(script_dir, 'submission_v4_dimreduce.csv')
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}")
