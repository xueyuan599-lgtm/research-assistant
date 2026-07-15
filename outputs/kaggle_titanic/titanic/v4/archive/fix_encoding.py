"""
V4 Fix: Proper Out-of-Fold Target Encoding + Clean Feature Pipeline
修复 Agent C 的 Target Encoding 泄漏问题。
Title_SurvivalRate / TicketSurvRate / Deck_SurvivalRate 必须在 CV fold 内计算。
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else os.path.dirname(os.path.abspath(sys.argv[0]))
BASE = os.path.join(os.path.dirname(BASE), 'titanic') if not os.path.exists(os.path.join(BASE, 'train.csv')) else BASE

# Ensure v4 output dir
os.makedirs(os.path.join(BASE, 'v4'), exist_ok=True)

# ── 1. Load ──
train = pd.read_csv(os.path.join(BASE, 'train.csv'))
test = pd.read_csv(os.path.join(BASE, 'test.csv'))
test_ids = test['PassengerId']

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Survived rate: {train['Survived'].mean():.3f}")

# ── 2. Feature Engineering (NO target leakage) ──

def extract_title(name):
    """Safe title extraction"""
    try:
        parts = name.split(',')
        if len(parts) > 1:
            return parts[1].split('.')[0].strip()
    except:
        pass
    return 'Other'

def engineer_base(df):
    """Basic features without target encoding"""
    df = df.copy()

    # --- Age: group median + missing flag ---
    df['AgeMissing'] = df['Age'].isna().astype(int)
    age_medians = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_medians)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # --- Fare: log1p + zero flag ---
    df['FareZero'] = (df['Fare'] == 0).astype(int)
    fare_medians = df.groupby('Pclass')['Fare'].transform('median')
    df['Fare'] = df['Fare'].fillna(fare_medians)
    df['Fare'] = df['Fare'].replace(0, np.nan)
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare_log'] = np.log1p(df['Fare'])

    # --- Embarked ---
    df['Embarked'] = df['Embarked'].fillna('S')

    # --- Sex ---
    df['Sex_code'] = df['Sex'].map({'male': 0, 'female': 1})

    # --- Title (refined) ---
    df['Title'] = df['Name'].apply(extract_title)
    title_map = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Mme': 'Mrs',
        'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
        'Master': 'Master',
        'Dr': 'Professional',
        'Rev': 'Clergy',
        'Col': 'Military', 'Major': 'Military', 'Capt': 'Military',
        'Lady': 'Honorific', 'Sir': 'Honorific', 'Countess': 'Honorific',
        'Don': 'Honorific', 'Dona': 'Honorific', 'Jonkheer': 'Honorific',
    }
    df['Title'] = df['Title'].map(title_map).fillna('Other')

    # --- Family ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilySize_bin'] = pd.cut(df['FamilySize'], bins=[0, 1, 2, 4, 100],
                                   labels=['Alone', 'Couple', 'Nuclear', 'Large'])

    # --- HasSpouse: Title=Mrs OR (Title=Mr and FamilySize>1) ---
    df['HasSpouse'] = ((df['Title'] == 'Mrs') |
                       ((df['Title'] == 'Mr') & (df['FamilySize'] > 1))).astype(int)

    # --- Deck ---
    df['Deck'] = df['Cabin'].str[0].fillna('M')
    # Merge T (single case) into M
    df['Deck'] = df['Deck'].replace('T', 'M')
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # --- Ticket ---
    df['TicketPrefix'] = df['Ticket'].apply(
        lambda x: ''.join([c for c in str(x).split()[0] if c.isalpha()]) if pd.notna(x) else '')
    df['TicketPrefix'] = df['TicketPrefix'].replace('', 'NUM')
    df['TicketGroupSize'] = df.groupby('Ticket')['PassengerId'].transform('count')

    # --- Fare per person ---
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # --- Interactions ---
    df['Sex_Pclass'] = df['Sex'] + '_P' + df['Pclass'].astype(str)
    df['Pclass_Age'] = df['Pclass'] * df['Age']
    df['Sex_Age'] = df['Sex_code'] * df['Age']
    df['Pclass_Fare'] = df['Pclass'] * df['Fare_log']

    # --- FamilyID for count encoding ---
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    df['FamilyID'] = df['Surname'] + '_' + df['FamilySize'].astype(str)
    df['FamilyID_Count'] = df.groupby('FamilyID')['PassengerId'].transform('count')

    return df

# ── 3. Out-of-Fold Target Encoding ──

def oof_target_encode(train_df, test_df, col, target, smoothing=10, n_folds=5):
    """
    Proper out-of-fold target encoding.
    For each fold: compute encoding on other 4 folds, apply to this fold.
    Test set: use full training data encoding.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoded_col = col + '_SurvivalRate'
    train_df[encoded_col] = 0.0
    global_mean = target.mean()

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_values = np.zeros(len(train_df))

    for tr_idx, val_idx in kf.split(train_df, target):
        tr_target = target.iloc[tr_idx]
        tr_col = train_df[col].iloc[tr_idx]
        val_col = train_df[col].iloc[val_idx]

        # Compute smoothed mean on train folds
        cat_mean = tr_target.groupby(tr_col).mean()
        cat_count = tr_target.groupby(tr_col).count()
        smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)

        # Apply to validation fold
        oof_values[val_idx] = val_col.map(smoothed).fillna(global_mean).values

    train_df[encoded_col] = oof_values

    # Test set: use full training data
    cat_mean = target.groupby(train_df[col]).mean()
    cat_count = target.groupby(train_df[col]).count()
    smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
    test_df[encoded_col] = test_df[col].map(smoothed).fillna(global_mean).values

    return train_df, test_df

# ── 4. Build features ──

train_feat = engineer_base(train)
test_feat = engineer_base(test)

# OOF target encoding for Title, Deck, Ticket (THE CRITICAL FIX)
y = train['Survived'].astype(int)

train_feat, test_feat = oof_target_encode(train_feat, test_feat, 'Title', y, smoothing=10)
train_feat, test_feat = oof_target_encode(train_feat, test_feat, 'Deck', y, smoothing=10)

# For Ticket: use TicketPrefix as proxy (group-level encoding)
# TicketSurvRate is too granular - use TicketGroupSize instead as count feature
# For Ticket-level target encoding, group by TicketPrefix to avoid over-granularity
train_feat, test_feat = oof_target_encode(train_feat, test_feat, 'TicketPrefix', y, smoothing=20)

# ── 5. Encode categoricals ──

def encode_features(train_df, test_df):
    """OneHot + Count encoding, then align columns"""
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # OneHot for low-cardinality categoricals
    onehot_cols = ['Embarked', 'FamilySize_bin', 'Sex_Pclass']
    for col in onehot_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df, dummies], axis=1)

    # Count encoding for high-cardinality categoricals
    count_cols = ['Title', 'Deck', 'TicketPrefix']
    for col in count_cols:
        if col in df.columns:
            counts = df.groupby(col)[col].transform('count') / len(df)
            df[col + '_freq'] = counts

    # Drop non-numeric columns
    drop_cols = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Surname', 'FamilyID',
                 'Embarked', 'FamilySize_bin', 'Sex_Pclass', 'Title', 'Deck', 'TicketPrefix',
                 'Survived']
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Split back
    n_train = len(train_df)
    train_out = df.iloc[:n_train].copy()
    test_out = df.iloc[n_train:].copy()

    # Drop Survived from test (should not exist but just in case)
    if 'Survived' in test_out.columns:
        test_out.drop('Survived', axis=1, inplace=True)

    return train_out, test_out

X_all, X_test_all = encode_features(train_feat, test_feat)
X_all['Survived'] = y.values

# Scale numeric columns
numeric_cols = ['Age', 'Fare_log', 'FarePerPerson', 'FamilySize', 'Pclass_Age',
                'Sex_Age', 'Pclass_Fare', 'TicketGroupSize', 'FamilyID_Count',
                'Title_SurvivalRate', 'Deck_SurvivalRate', 'TicketPrefix_SurvivalRate']

scaler_cols = [c for c in numeric_cols if c in X_all.columns]
scaler = StandardScaler()
X_all[scaler_cols] = scaler.fit_transform(X_all[scaler_cols])

# Apply same scaler to test
test_numeric = [c for c in scaler_cols if c in X_test_all.columns]
if test_numeric:
    X_test_all[test_numeric] = scaler.transform(X_test_all[test_numeric])

# Drop zero-variance columns
for col in X_all.columns:
    if col != 'Survived' and X_all[col].nunique() <= 1:
        X_all.drop(col, axis=1, inplace=True)
        if col in X_test_all.columns:
            X_test_all.drop(col, axis=1, inplace=True)

# Align test columns to train
train_cols = [c for c in X_all.columns if c != 'Survived']
X_test_aligned = X_test_all.reindex(columns=train_cols, fill_value=0)

# ── 6. Save ──
X_all.to_csv(os.path.join(BASE, 'v4', 'X_train_fixed.csv'), index=False)
X_test_aligned.to_csv(os.path.join(BASE, 'v4', 'X_test_fixed.csv'), index=False)

print(f"\n=== Fixed Feature Engineering Complete ===")
print(f"X_train: {X_all.shape} ({X_all.shape[1]-1} features + Survived)")
print(f"X_test:  {X_test_aligned.shape} ({X_test_aligned.shape[1]} features)")
print(f"Target encoded features (OOF): Title_SurvivalRate, Deck_SurvivalRate, TicketPrefix_SurvivalRate")
print(f"\nFiles saved:")
print(f"  v4/X_train_fixed.csv")
print(f"  v4/X_test_fixed.csv")

# Quick sanity check: correlation of target-encoded features with Survived
print(f"\n=== Sanity Check: Correlation with Survived ===")
for col in ['Title_SurvivalRate', 'Deck_SurvivalRate', 'TicketPrefix_SurvivalRate']:
    if col in X_all.columns:
        corr = X_all[col].corr(X_all['Survived'])
        print(f"  {col}: r = {corr:.4f}")

# Check for NaN/Inf
nan_count = X_test_aligned.isna().sum().sum()
inf_count = np.isinf(X_test_aligned.values).sum()
print(f"\nX_test NaN: {nan_count}, Inf: {inf_count}")
assert nan_count == 0, f"Test has {nan_count} NaN values!"
assert inf_count == 0, f"Test has {inf_count} Inf values!"
print("All checks passed!")
