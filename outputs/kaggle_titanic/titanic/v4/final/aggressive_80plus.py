"""
V4 Aggressive: 社区验证的 80+ 方案
核心: OOF Family_Survival_Rate + Ticket_Survival_Rate + Title_Survival_Rate
参考: Amnaikram1 (LB 0.8373), wafaahs (Top 7%)
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(script_dir)

train = pd.read_csv(os.path.join(BASE, 'train.csv'))
test = pd.read_csv(os.path.join(BASE, 'test.csv'))
test_ids = test['PassengerId']

y = train['Survived'].astype(int)
all_data = pd.concat([train, test], axis=0, ignore_index=True)

print(f"Train: {train.shape}, Test: {test.shape}, All: {all_data.shape}")

# ============================================================
# FEATURE ENGINEERING (aggressive, community-proven)
# ============================================================

def engineer(df, train_mask=None):
    """train_mask: boolean array, True for training rows"""
    df = df.copy()

    # --- Title (fine-grained) ---
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip() if ',' in str(x) else 'Other')
    title_map = {
        'Mr': 0, 'Mrs': 1, 'Mme': 1, 'Miss': 2, 'Mlle': 2, 'Ms': 2,
        'Master': 3, 'Dr': 4, 'Rev': 5,
        'Col': 6, 'Major': 6, 'Capt': 6,
        'Lady': 7, 'Sir': 7, 'Countess': 7, 'Don': 7, 'Dona': 7, 'Jonkheer': 7,
    }
    df['Title'] = df['Title'].map(title_map).fillna(8)

    # --- Sex ---
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # --- Age: group median ---
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # --- Age binning (10 bins, community standard) ---
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 30, 35, 40, 50, 60, 80],
                          labels=range(9)).astype(int)

    # --- Fare: log + binning ---
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].replace(0, np.nan)
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare_log'] = np.log1p(df['Fare'])
    df['FareBin'] = pd.qcut(df['Fare'], q=10, labels=False, duplicates='drop')
    df['FareBin'] = df['FareBin'].fillna(-1).astype(int)

    # --- Embarked ---
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # --- Family ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # --- Surname (for family grouping) ---
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

    # --- Deck from Cabin ---
    df['Deck'] = df['Cabin'].str[0].fillna('M')
    df['Deck'] = df['Deck'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'M': 8}).fillna(8).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # --- Ticket info ---
    df['TicketFreq'] = df.groupby('Ticket')['PassengerId'].transform('count')
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: ''.join([c for c in str(x).split()[0] if c.isalpha()]))
    df['TicketPrefix'] = df['TicketPrefix'].replace('', 'NUM')

    # --- Fare per person ---
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # --- Key interactions ---
    df['Sex_Pclass'] = df['Sex'] * (4 - df['Pclass'])
    df['Age_Pclass'] = df['Age'] * df['Pclass']

    # --- Name length ---
    df['NameLen'] = df['Name'].str.len()

    return df

all_feat = engineer(all_data)

# ============================================================
# OOF TARGET ENCODING (the key to 80+)
# ============================================================

def oof_target_encode_simple(df, y_series, col, smoothing=10):
    """Simple OOF: global smoothed mean for full dataset"""
    global_mean = y_series.mean()
    cat_mean = y_series.groupby(df.loc[y_series.index, col]).mean()
    cat_count = y_series.groupby(df.loc[y_series.index, col]).count()
    smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
    return df[col].map(smoothed).fillna(global_mean)

# Family survival rate
train_idx = all_feat.index[:891]
test_idx = all_feat.index[891:]

# 1. Family_Survival_Rate
all_feat['FamilySurvRate'] = oof_target_encode_simple(all_feat, y, 'Surname', smoothing=5)
# For rows where Surname only appears in test set, use global mean
test_only_surnames = set(all_feat.loc[test_idx, 'Surname']) - set(all_feat.loc[train_idx, 'Surname'])
all_feat.loc[all_feat['Surname'].isin(test_only_surnames), 'FamilySurvRate'] = y.mean()

# 2. Ticket_Survival_Rate
all_feat['TicketSurvRate'] = oof_target_encode_simple(all_feat, y, 'Ticket', smoothing=10)
test_only_tickets = set(all_feat.loc[test_idx, 'Ticket']) - set(all_feat.loc[train_idx, 'Ticket'])
all_feat.loc[all_feat['Ticket'].isin(test_only_tickets), 'TicketSurvRate'] = y.mean()

# 3. Title_Survival_Rate
all_feat['TitleSurvRate'] = oof_target_encode_simple(all_feat, y, 'Title', smoothing=5)

# 4. TicketPrefix_Survival_Rate
all_feat['PrefixSurvRate'] = oof_target_encode_simple(all_feat, y, 'TicketPrefix', smoothing=10)

# 5. Deck_Survival_Rate
all_feat['DeckSurvRate'] = oof_target_encode_simple(all_feat, y, 'Deck', smoothing=10)

# ============================================================
# OOF cross-validation encoding (per-fold, proper)
# ============================================================
print("Computing proper OOF encoding (5-fold)...")

for col, smooth in [('Surname', 5), ('Ticket', 10), ('Title', 5), ('TicketPrefix', 10), ('Deck', 10)]:
    encoded_col = col + '_OOF'
    all_feat[encoded_col] = y.mean()  # default
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = y.mean()

    for tr_idx, val_idx in kf.split(all_feat.loc[train_idx], y):
        tr_y = y.iloc[tr_idx]
        tr_col = all_feat.loc[tr_idx, col]
        val_col = all_feat.loc[val_idx, col]

        cat_mean = tr_y.groupby(tr_col).mean()
        cat_count = tr_y.groupby(tr_col).count()
        smoothed = (cat_count * cat_mean + smooth * global_mean) / (cat_count + smooth)

        all_feat.loc[val_idx, encoded_col] = val_col.map(smoothed).fillna(global_mean)

    # Test set: use full training encoding
    cat_mean = y.groupby(all_feat.loc[train_idx, col]).mean()
    cat_count = y.groupby(all_feat.loc[train_idx, col]).count()
    smoothed = (cat_count * cat_mean + smooth * global_mean) / (cat_count + smooth)
    all_feat.loc[test_idx, encoded_col] = all_feat.loc[test_idx, col].map(smoothed).fillna(global_mean)

# ============================================================
# SELECT FEATURES
# ============================================================

feature_cols = [
    'Pclass', 'Sex', 'Age', 'AgeBin', 'Fare_log', 'FareBin',
    'Embarked', 'Title', 'FamilySize', 'IsAlone',
    'Deck', 'HasCabin', 'TicketFreq',
    'FarePerPerson', 'Sex_Pclass', 'Age_Pclass', 'NameLen',
    # Simple target encoding
    'FamilySurvRate', 'TicketSurvRate', 'TitleSurvRate', 'PrefixSurvRate', 'DeckSurvRate',
    # OOF target encoding
    'Surname_OOF', 'Ticket_OOF', 'Title_OOF', 'TicketPrefix_OOF', 'Deck_OOF',
]

X_all = all_feat[feature_cols]
# One-hot encode low-cardinality categoricals
for col in ['Embarked', 'AgeBin', 'FareBin']:
    dummies = pd.get_dummies(X_all[col], prefix=col, drop_first=True)
    X_all = pd.concat([X_all, dummies], axis=1)
    X_all.drop(col, axis=1, inplace=True)

X = X_all.iloc[:891].copy()
X_test = X_all.iloc[891:].copy()

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print(f"Features: {X_scaled.shape[1]}")
print(f"X_test NaN: {X_test_scaled.isna().sum().sum()}")

# ============================================================
# MODEL SEARCH
# ============================================================
SEEDS = [42, 123, 456, 789, 1024]

def cv_multi(model, X_tr, seeds=SEEDS):
    scores = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        s = cross_val_score(model, X_tr, y, cv=cv, scoring='accuracy')
        scores.extend(s)
    return np.mean(scores), np.std(scores)

print(f"\n{'='*60}")
print("MODEL SEARCH")
print(f"{'='*60}")

best_name, best_score, best_model = '', 0, None

# RF (various depths)
for d in [4, 5, 6, 7]:
    for leaf in [1, 3, 5]:
        rf = RandomForestClassifier(n_estimators=500, max_depth=d, min_samples_leaf=leaf,
                                     min_samples_split=max(leaf*2, 5), random_state=42, n_jobs=-1)
        acc, std = cv_multi(rf, X_scaled, seeds=SEEDS[:3])
        if acc > 0.84:
            print(f"RF_d{d}_l{leaf}:          {acc:.4f} +/- {std:.4f}")
        if acc > best_score:
            best_score, best_name = acc, f'RF_d{d}_l{leaf}'
            best_model = rf

# GBDT
for d in [3, 4, 5]:
    for lr in [0.03, 0.05, 0.1]:
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=d, learning_rate=lr,
                                         subsample=0.7, min_samples_leaf=5, random_state=42)
        acc, std = cv_multi(gb, X_scaled, seeds=SEEDS[:3])
        if acc > best_score:
            best_score, best_name = acc, f'GBDT_d{d}_lr{lr}'
            best_model = gb
        if acc > 0.84:
            print(f"GBDT_d{d}_lr{lr}:       {acc:.4f} +/- {std:.4f}")

# LR
for C in [0.1, 0.3, 1.0, 3.0]:
    lr = LogisticRegression(max_iter=5000, C=C, random_state=42)
    acc, std = cv_multi(lr, X_scaled, seeds=SEEDS[:3])
    if acc > best_score:
        best_score, best_name = acc, f'LR_C{C}'
        best_model = lr
    print(f"LR_C{C}:             {acc:.4f} +/- {std:.4f}")

# XGBoost
try:
    from xgboost import XGBClassifier
    for d in [3, 4, 5]:
        for lr in [0.03, 0.05]:
            xgb = XGBClassifier(n_estimators=300, max_depth=d, learning_rate=lr,
                                reg_alpha=1, reg_lambda=1, subsample=0.7, random_state=42, verbosity=0)
            acc, std = cv_multi(xgb, X_scaled, seeds=SEEDS[:3])
            if acc > best_score:
                best_score, best_name = acc, f'XGB_d{d}_lr{lr}'
                best_model = xgb
            if acc > 0.84:
                print(f"XGB_d{d}_lr{lr}:        {acc:.4f} +/- {std:.4f}")
except: pass

# CatBoost
try:
    from catboost import CatBoostClassifier
    for d in [3, 4, 5]:
        cb = CatBoostClassifier(iterations=300, depth=d, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0)
        acc, std = cv_multi(cb, X_scaled, seeds=SEEDS[:3])
        if acc > best_score:
            best_score, best_name = acc, f'CatBoost_d{d}'
            best_model = cb
        if acc > 0.84:
            print(f"CatBoost_d{d}:       {acc:.4f} +/- {std:.4f}")
except: pass

# ── Re-score best with full 5 seeds ──
print(f"\n=== Best Model (3-seed): {best_name} = {best_score:.4f} ===")
acc_full, std_full = cv_multi(best_model, X_scaled, seeds=SEEDS)
print(f"5-seed CV: {acc_full:.4f} +/- {std_full:.4f}")

# ── Voting Ensemble (Top 3 by 3-seed CV) ──
print(f"\n=== Voting Ensemble ===")
# Gather top models
top_models_list = []
for name_prefix in ['RF_d5_l1', 'RF_d4_l3', 'GBDT_d4_lr0.1', 'GBDT_d3_lr0.1', 'LR_C1.0', 'LR_C3.0']:
    # Reconstruct top models
    if name_prefix.startswith('RF'):
        d, leaf = 5, 1
        top_models_list.append((name_prefix, RandomForestClassifier(
            n_estimators=500, max_depth=d, min_samples_leaf=leaf, min_samples_split=max(leaf*2,5),
            random_state=42, n_jobs=-1)))
    elif name_prefix.startswith('GBDT_d4'):
        top_models_list.append((name_prefix, GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.7, min_samples_leaf=5, random_state=42)))
    elif name_prefix.startswith('GBDT_d3'):
        top_models_list.append((name_prefix, GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.7, min_samples_leaf=5, random_state=42)))
    elif name_prefix.startswith('LR_C1'):
        top_models_list.append((name_prefix, LogisticRegression(max_iter=5000, C=1.0, random_state=42)))
    elif name_prefix.startswith('LR_C3'):
        top_models_list.append((name_prefix, LogisticRegression(max_iter=5000, C=3.0, random_state=42)))

for top_n in [2, 3, 4, 5, len(top_models_list)]:
    if top_n > len(top_models_list): break
    for vt in ['soft', 'hard']:
        ens = VotingClassifier(top_models_list[:top_n], voting=vt)
        acc, std = cv_multi(ens, X_scaled, seeds=SEEDS[:3])
        if acc > 0.84:
            print(f"Top{top_n}_Voting_{vt}:  {acc:.4f} +/- {std:.4f}")

# ── Generate Submission ──
print(f"\n=== Generating Submission ===")
best_model.fit(X_scaled, y)
preds = best_model.predict(X_test_scaled)
print(f"Best model ({best_name}): survived {preds.sum()}/{len(preds)} ({preds.mean():.1%})")

sub = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds.astype(int)})
sub.to_csv(os.path.join(script_dir, 'submission_aggressive.csv'), index=False)

# Also try: Top 3 models hard voting
# RF_d5_l1 + GBDT_d4_lr0.1 + LR_C1.0
ens3 = VotingClassifier(top_models_list[:3], voting='hard')
ens3.fit(X_scaled, y)
preds3 = ens3.predict(X_test_scaled)
sub3 = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds3.astype(int)})
sub3.to_csv(os.path.join(script_dir, 'submission_aggressive_ensemble.csv'), index=False)
print(f"Top3 Hard Voting: survived {preds3.sum()}/{len(preds3)} ({preds3.mean():.1%})")

# Also try: simple LR with OOF features only (drop simple target encoding)
oof_cols = [c for c in X_scaled.columns if 'OOF' in c or c in
            ['Pclass', 'Sex', 'Age', 'Fare_log', 'Embarked', 'Title', 'FamilySize',
             'IsAlone', 'Deck', 'HasCabin', 'TicketFreq', 'FarePerPerson', 'Sex_Pclass',
             'Age_Pclass', 'NameLen']]
X_oof = X_scaled[oof_cols]
X_test_oof = X_test_scaled[oof_cols]
lr_oof = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
acc, std = cv_multi(lr_oof, X_oof)
print(f"\nLR with OOF-only features: CV={acc:.4f} +/- {std:.4f}")

lr_oof.fit(X_oof, y)
preds_oof = lr_oof.predict(X_test_oof)
sub_oof = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds_oof.astype(int)})
sub_oof.to_csv(os.path.join(script_dir, 'submission_aggressive_oof.csv'), index=False)
print(f"LR OOF-only: survived {preds_oof.sum()}/{len(preds_oof)} ({preds_oof.mean():.1%})")

print(f"\nAll saved to {script_dir}/")
print("  submission_aggressive.csv")
print("  submission_aggressive_ensemble.csv")
print("  submission_aggressive_oof.csv")
