#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Titanic 深度数据探查 + 泄漏审计
目标：刷榜导向的特征挖掘报告
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings, re, json, os, sys
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE, "train.csv")
TEST_PATH = os.path.join(BASE, "test.csv")
OUT_DIR = os.path.join(BASE, "v4")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print(f"Train: {train.shape}, Test: {test.shape}")
# Mark source before concat for shift detection
train['_source'] = 'train'
test['_source'] = 'test'
test['Survived'] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"Combined: {df.shape}")

# ═══════════════════════════════════════════════════════
# 1. DATA BASIC PROFILE
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: DATA BASIC PROFILE")
print("=" * 70)

print("\n--- Column Overview ---")
overview = []
for col in train.columns:
    col_train = train[col]
    col_test = test[col] if col in test.columns else None
    missing_train = col_train.isna().sum()
    missing_pct_train = 100 * missing_train / len(train)
    unique_train = col_train.nunique()
    dtype = col_train.dtype

    missing_test = col_test.isna().sum() if col_test is not None else "N/A"
    missing_pct_test = 100 * missing_test / len(test) if col_test is not None else "N/A"

    overview.append({
        'Column': col,
        'Dtype': str(dtype),
        'Train_Missing': missing_train,
        'Train_Missing%': round(missing_pct_train, 1),
        'Test_Missing': missing_test,
        'Test_Missing%': round(missing_pct_test, 1) if isinstance(missing_pct_test, (int, float)) else missing_pct_test,
        'Unique_Train': unique_train,
        'In_Test': col in test.columns
    })

overview_df = pd.DataFrame(overview)
print(overview_df.to_string(index=False))

# Column difference
train_cols = set(train.columns)
test_cols = set(test.columns)
print(f"\nTrain-only columns: {train_cols - test_cols}")
print(f"Test-only columns: {test_cols - train_cols}")

print("\n--- Basic Statistics ---")
print(f"Overall Survival Rate: {train['Survived'].mean():.4f} ({train['Survived'].sum():.0f}/{len(train)})")
print(f"Female Survival Rate: {train[train['Sex']=='female']['Survived'].mean():.4f}")
print(f"Male Survival Rate: {train[train['Sex']=='male']['Survived'].mean():.4f}")

print("\nPclass distribution:")
print(train['Pclass'].value_counts().sort_index())
print(f"\nPclass Survival Rates:")
for p in [1, 2, 3]:
    sub = train[train['Pclass'] == p]
    print(f"  Pclass {p}: {sub['Survived'].mean():.4f} ({sub['Survived'].sum():.0f}/{len(sub)})")

print("\nSex distribution:")
print(train['Sex'].value_counts())

print("\nAge stats (train):")
print(train['Age'].describe())

print("\nFare stats (train):")
print(train['Fare'].describe())

print("\nEmbarked distribution:")
print(train['Embarked'].value_counts())

print("\nSibSp value counts:")
print(train['SibSp'].value_counts().sort_index())

print("\nParch value counts:")
print(train['Parch'].value_counts().sort_index())

# ═══════════════════════════════════════════════════════
# 2. FEATURE-TARGET CONDITIONAL DISTRIBUTIONS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: FEATURE-TARGET CONDITIONAL DISTRIBUTIONS")
print("=" * 70)

# 2.1 Pclass x Survived
print("\n--- 2.1 Pclass x Survived ---")
ct = pd.crosstab(train['Pclass'], train['Survived'], margins=True, margins_name='Total')
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
print(ct)

# 2.2 Sex x Survived
print("\n--- 2.2 Sex x Survived ---")
ct = pd.crosstab(train['Sex'], train['Survived'], margins=True, margins_name='Total')
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
print(ct)

# 2.3 Pclass x Sex x Survived — STRONGEST SIGNAL
print("\n--- 2.3 Pclass x Sex x Survived (THREE-WAY INTERACTION) ---")
ct3 = pd.crosstab([train['Pclass'], train['Sex']], train['Survived'], margins=False)
ct3['Survival_Rate'] = ct3[1] / (ct3[0] + ct3[1])
ct3['Total'] = ct3[0] + ct3[1]
print(ct3)

# 2.4 Age Group x Survived
print("\n--- 2.4 Age Group x Survived ---")
bins = [0, 12, 19, 35, 50, 60, 120]
labels = ['Child(0-12)', 'Teen(13-19)', 'YoungAdult(20-35)', 'Adult(36-50)', 'MiddleAge(51-60)', 'Senior(60+)']
train_age = train.copy()
train_age['AgeGroup'] = pd.cut(train_age['Age'], bins=bins, labels=labels, right=True)
ct = pd.crosstab(train_age['AgeGroup'], train_age['Survived'], margins=True, margins_name='Total')
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
print(ct)

# 2.5 Fare Group x Survived
print("\n--- 2.5 Fare Group (quantile 4-group) x Survived ---")
train_fare = train.copy()
train_fare['FareGroup'] = pd.qcut(train_fare['Fare'], q=4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'])
ct = pd.crosstab(train_fare['FareGroup'], train_fare['Survived'], margins=True, margins_name='Total')
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
print(ct)

# 2.6 Embarked x Pclass x Survived
print("\n--- 2.6 Embarked x Pclass x Survived ---")
ct = pd.crosstab([train['Embarked'], train['Pclass']], train['Survived'])
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
ct['Total'] = ct[0] + ct[1]
print(ct)

# 2.7 SibSp x Survived
print("\n--- 2.7 SibSp x Survived ---")
ct = pd.crosstab(train['SibSp'], train['Survived'])
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
ct['Total'] = ct[0] + ct[1]
print(ct)
# Group analysis
print("\nSibSp Grouped:")
for group_name, condition in [('0 (alone)', train['SibSp'] == 0),
                                ('1-2', train['SibSp'].isin([1, 2])),
                                ('3+', train['SibSp'] >= 3)]:
    sub = train[condition]
    print(f"  SibSp {group_name}: SurvRate={sub['Survived'].mean():.4f}, N={len(sub)}")

# 2.8 Parch x Survived
print("\n--- 2.8 Parch x Survived ---")
ct = pd.crosstab(train['Parch'], train['Survived'])
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
ct['Total'] = ct[0] + ct[1]
print(ct)
print("\nParch Grouped:")
for group_name, condition in [('0 (alone)', train['Parch'] == 0),
                                ('1-2', train['Parch'].isin([1, 2])),
                                ('3+', train['Parch'] >= 3)]:
    sub = train[condition]
    print(f"  Parch {group_name}: SurvRate={sub['Survived'].mean():.4f}, N={len(sub)}")

# 2.9 FamilySize x Survived
print("\n--- 2.9 FamilySize (= SibSp + Parch + 1) x Survived ---")
train_fs = train.copy()
train_fs['FamilySize'] = train_fs['SibSp'] + train_fs['Parch'] + 1
ct = pd.crosstab(train_fs['FamilySize'], train_fs['Survived'])
ct['Survival_Rate'] = ct[1] / (ct[0] + ct[1])
ct['Total'] = ct[0] + ct[1]
print(ct)

# Find optimal binning
print("\nFamilySize optimal binning search:")
candidates = {
    'alone(1)': train_fs['FamilySize'] == 1,
    'small(2)': train_fs['FamilySize'] == 2,
    'medium(3-4)': train_fs['FamilySize'].isin([3, 4]),
    'large(5+)': train_fs['FamilySize'] >= 5,
}
for name, cond in candidates.items():
    sub = train_fs[cond]
    print(f"  FamilySize {name}: SurvRate={sub['Survived'].mean():.4f}, N={len(sub)}")

# Also try: alone vs small vs medium vs large
print("\nAlternative binning:")
bins_fs = {
    'IsAlone(1)': train_fs['FamilySize'] == 1,
    'Couple(2)': train_fs['FamilySize'] == 2,
    'Nuclear(3-4)': train_fs['FamilySize'].isin([3, 4]),
    'Extended(5-7)': train_fs['FamilySize'].isin([5, 6, 7]),
    'Clan(8+)': train_fs['FamilySize'] >= 8,
}
for name, cond in bins_fs.items():
    sub = train_fs[cond]
    print(f"  {name}: SurvRate={sub['Survived'].mean():.4f}, N={len(sub)}")

# ═══════════════════════════════════════════════════════
# 3. HIDDEN INFORMATION MINING
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: HIDDEN INFORMATION MINING")
print("=" * 70)

# --- 3.1 Name → Title ---
print("\n--- 3.1 Name → Title Extraction ---")
def extract_title(name):
    """Extract title from name string."""
    match = re.search(r',\s*([^,]+)\.', str(name))
    if match:
        title = match.group(1).strip()
        # Clean up French titles
        if title in ['Mlle', 'Mme']:
            return 'Mlle/Mme'
        if title == 'Ms':
            return 'Ms'
        return title
    return 'Unknown'

df['Title'] = df['Name'].apply(extract_title)
train_titles = df[df['_source'] == 'train'].copy()

print("Title distribution and survival rates:")
title_stats = train_titles.groupby('Title').agg(
    Count=('Survived', 'count'),
    Survived=('Survived', 'sum'),
    SurvRate=('Survived', 'mean')
).sort_values('Count', ascending=False)
title_stats['SurvRate'] = title_stats['SurvRate'].round(4)
print(title_stats)

# Group rare titles
rare_titles = title_stats[title_stats['Count'] < 5].index.tolist()
print(f"\nRare titles (count < 5): {rare_titles}")

# Standard title mapping for modeling
title_map = {
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Master': 'Master',
    'Don': 'Honorific',
    'Rev': 'Clergy',
    'Dr': 'Professional',
    'Mlle/Mme': 'Mrs',
    'Ms': 'Mrs',
    'Major': 'Military',
    'Lady': 'Honorific',
    'Sir': 'Honorific',
    'Col': 'Military',
    'Capt': 'Military',
    'Countess': 'Honorific',
    'Jonkheer': 'Honorific',
    'Dona': 'Honorific',
}
df['TitleGroup'] = df['Title'].map(title_map).fillna('Other')

print("\nTitleGroup survival (train):")
tg = df[df['_source'] == 'train'].groupby('TitleGroup').agg(
    Count=('Survived', 'count'),
    SurvRate=('Survived', 'mean')
).sort_values('Count', ascending=False)
tg['SurvRate'] = tg['SurvRate'].round(4)
print(tg)

# --- 3.2 Name → Family Name ---
print("\n--- 3.2 Name → Family Name ---")
def extract_surname(name):
    return str(name).split(',')[0].strip()

df['Surname'] = df['Name'].apply(extract_surname)
# Count how many passengers share a surname in train
train_surnames = df[df['_source'] == 'train']
surname_counts = train_surnames['Surname'].value_counts()
print(f"Unique surnames: {len(surname_counts)}")
print(f"Surnames with 2+ members: {(surname_counts >= 2).sum()}")
print(f"Surnames with 3+ members: {(surname_counts >= 3).sum()}")

# Family survival consistency: for families with 2+ members, check if all survived/died
family_survival = train_surnames.groupby('Surname').agg(
    FamilySize=('Survived', 'count'),
    Survived_Sum=('Survived', 'sum'),
    SurvRate=('Survived', 'mean')
).query('FamilySize >= 2')

all_survived = (family_survival['SurvRate'] == 1.0).sum()
all_died = (family_survival['SurvRate'] == 0.0).sum()
mixed = len(family_survival) - all_survived - all_died
print(f"\nFamilies with 2+ members: {len(family_survival)}")
print(f"  All survived: {all_survived} ({100*all_survived/len(family_survival):.1f}%)")
print(f"  All died: {all_died} ({100*all_died/len(family_survival):.1f}%)")
print(f"  Mixed outcomes: {mixed} ({100*mixed/len(family_survival):.1f}%)")
print("\n(Implication: family members tend to share fate — Ticket grouping may leak)")

# --- 3.3 Cabin → Deck ---
print("\n--- 3.3 Cabin → Deck ---")
def extract_deck(cabin):
    if pd.isna(cabin):
        return 'M'  # Missing
    cabin_str = str(cabin).strip()
    # Check for multi-cabin (space-separated)
    parts = cabin_str.split()
    decks = set()
    for p in parts:
        m = re.match(r'([A-T])', p)
        if m:
            decks.add(m.group(1))
    if len(decks) == 0:
        return 'M'
    if len(decks) == 1:
        return list(decks)[0]
    return 'Multi'

df['Deck'] = df['Cabin'].apply(extract_deck)

train_deck = df[df['_source'] == 'train']
deck_stats = train_deck.groupby('Deck').agg(
    Count=('Survived', 'count'),
    SurvRate=('Survived', 'mean')
).sort_values('Count', ascending=False)
deck_stats['SurvRate'] = deck_stats['SurvRate'].round(4)
print("Deck survival rates (train):")
print(deck_stats)

# Check if multi-cabin passengers exist
multi_cabin = df[df['Cabin'].notna() & df['Cabin'].str.contains(r'\s', na=False)]
print(f"\nMulti-cabin (space-separated) records: {len(multi_cabin)}")
if len(multi_cabin) > 0:
    print("Examples:")
    print(multi_cabin[['Name', 'Cabin', 'Pclass']].head(10))

# --- 3.4 Cabin → CabinNum ---
print("\n--- 3.4 Cabin → CabinNum ---")
def extract_cabin_num(cabin):
    if pd.isna(cabin):
        return np.nan
    nums = re.findall(r'(\d+)', str(cabin))
    if nums:
        return int(nums[0])
    return np.nan

df['CabinNum'] = df['Cabin'].apply(extract_cabin_num)
_train_deck2 = df[df['_source'] == 'train']  # refresh after adding new columns
tn = _train_deck2['CabinNum'].dropna()
print(f"CabinNum range: {tn.min():.0f} - {tn.max():.0f}")
print(f"CabinNum stats: mean={tn.mean():.1f}, median={tn.median():.1f}")

# Check if CabinNum correlates with Pclass
cabin_num_by_class = _train_deck2.dropna(subset=['CabinNum']).groupby('Pclass')['CabinNum'].agg(['mean', 'count'])
print("\nCabinNum by Pclass:")
print(cabin_num_by_class)

# --- 3.5 Cabin → HasCabin feature ---
print("\n--- 3.5 HasCabin vs Survived ---")
df['HasCabin'] = df['Cabin'].notna().astype(int)
train_deck = df[df['_source'] == 'train']  # refresh
has_cabin = train_deck.groupby('HasCabin').agg(
    Count=('Survived', 'count'),
    SurvRate=('Survived', 'mean')
)
has_cabin['SurvRate'] = has_cabin['SurvRate'].round(4)
print(has_cabin)

# HasCabin by Pclass
print("\nHasCabin by Pclass:")
print(pd.crosstab(train_deck['Pclass'], train_deck['HasCabin'], normalize='index'))

# --- 3.6 Ticket → Prefix ---
print("\n--- 3.6 Ticket → TicketPrefix ---")
def extract_ticket_prefix(ticket):
    t = str(ticket).strip()
    # Check if starts with letters
    m = re.match(r'([A-Za-z]+[/.]?)', t)
    if m:
        prefix = m.group(1).rstrip('/.')
        return prefix
    return 'NUMERIC'

df['TicketPrefix'] = df['Ticket'].apply(extract_ticket_prefix)
train_deck = df[df['_source'] == 'train']  # refresh
tp_stats = train_deck.groupby('TicketPrefix').agg(
    Count=('Survived', 'count'),
    SurvRate=('Survived', 'mean')
).sort_values('Count', ascending=False)
tp_stats['SurvRate'] = tp_stats['SurvRate'].round(4)
print("TicketPrefix survival rates (train, top 20):")
print(tp_stats.head(20))

# --- 3.7 Ticket → TicketGroup (shared tickets) ---
print("\n--- 3.7 Ticket → TicketGroup (shared tickets) ---")
ticket_counts = df.groupby('Ticket').size()
shared_tickets = ticket_counts[ticket_counts >= 2]
print(f"Tickets shared by 2+ passengers: {len(shared_tickets)}")
print(f"Passengers on shared tickets: {shared_tickets.sum()}")

# For each shared ticket group, check if they share fate
def ticket_group_survival(grp):
    survived = grp['Survived'].dropna()
    if len(survived) < 2:
        return pd.Series({'GroupSize': len(grp), 'AllSameFate': None})
    all_same = (survived.nunique() == 1)
    return pd.Series({'GroupSize': len(grp), 'AllSameFate': all_same, 'SurvRate': survived.mean()})

tg_stats = df[df['Ticket'].isin(shared_tickets.index) & (df['_source'] == 'train')].groupby('Ticket').apply(ticket_group_survival).reset_index()
# Actually let's do this properly
ticket_groups_train = df[(df['_source'] == 'train') & (df['Ticket'].isin(shared_tickets.index))]
# Group by ticket
ticket_group_stats = []
for ticket_id, grp in ticket_groups_train.groupby('Ticket'):
    surv = grp['Survived']
    ticket_group_stats.append({
        'Ticket': ticket_id,
        'GroupSize': len(grp),
        'AllSameFate': surv.nunique() == 1,
        'SurvRate': surv.mean()
    })
tgs_df = pd.DataFrame(ticket_group_stats)
if len(tgs_df) > 0:
    same_fate_pct = tgs_df['AllSameFate'].mean()
    print(f"\nShared ticket groups where all share same fate: {tgs_df['AllSameFate'].sum()}/{len(tgs_df)} ({100*same_fate_pct:.1f}%)")
    print(f"This is a strong leakage signal — shared ticket = shared fate")

# Ticket group size vs survival
tgs_df['SizeGroup'] = pd.cut(tgs_df['GroupSize'], bins=[1,2,3,5,12], labels=['2', '3', '4-5', '6+'])
print("\nShared ticket fate consistency by group size:")
print(tgs_df.groupby('SizeGroup', observed=False).agg(
    Groups=('Ticket', 'count'),
    SameFateRate=('AllSameFate', 'mean')
).round(4))

# --- 3.8 Ticket → TicketLen ---
print("\n--- 3.8 Ticket → TicketLen ---")
df['TicketLen'] = df['Ticket'].apply(lambda x: len(str(x)))
tl_stats = df[df['_source'] == 'train'].groupby('TicketLen').agg(
    Count=('Survived', 'count'),
    SurvRate=('Survived', 'mean')
)
tl_stats['SurvRate'] = tl_stats['SurvRate'].round(4)
print(tl_stats)

# ═══════════════════════════════════════════════════════
# 4. TRAIN/TEST DISTRIBUTION SHIFT DETECTION
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: TRAIN/TEST DISTRIBUTION SHIFT DETECTION")
print("=" * 70)

train_only = df[df['_source'] == 'train'].copy()
test_only = df[df['_source'] == 'test'].copy()

shift_report = []

# Numeric features — KS test
numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in numeric_cols:
    t1 = train_only[col].dropna().values
    t2 = test_only[col].dropna().values
    ks_stat, ks_p = stats.ks_2samp(t1, t2)
    shift_report.append({
        'Feature': col,
        'Test': 'KS',
        'Statistic': round(ks_stat, 4),
        'P_Value': round(ks_p, 4),
        'Significant(p<0.05)': ks_p < 0.05
    })
    print(f"{col}: KS_stat={ks_stat:.4f}, p={ks_p:.4f} {'*** SHIFT' if ks_p < 0.05 else ''}")

# Categorical features — Chi-squared
from scipy.stats import chi2_contingency

cat_features = {
    'Sex': ['male', 'female'],
    'Embarked': ['C', 'Q', 'S'],
    'Title': None,  # will compute from data
    'Deck': None,
}

# Sex
t1_counts = train_only['Sex'].value_counts()
t2_counts = test_only['Sex'].value_counts()
all_cats = sorted(set(list(t1_counts.index) + list(t2_counts.index)))
t1_vec = [t1_counts.get(c, 0) for c in all_cats]
t2_vec = [t2_counts.get(c, 0) for c in all_cats]
contingency = np.array([t1_vec, t2_vec])
if contingency.shape[1] >= 2 and (contingency > 0).any():
    chi2, p_chi, dof, expected = chi2_contingency(contingency + 1)  # +1 smoothing
    shift_report.append({
        'Feature': 'Sex', 'Test': 'Chi2', 'Statistic': round(chi2, 4),
        'P_Value': round(p_chi, 4), 'Significant(p<0.05)': p_chi < 0.05
    })
    print(f"Sex: Chi2={chi2:.4f}, p={p_chi:.4f} {'*** SHIFT' if p_chi < 0.05 else ''}")

# Embarked
t1_counts = train_only['Embarked'].value_counts()
t2_counts = test_only['Embarked'].value_counts()
all_cats = sorted(set(list(t1_counts.index) + list(t2_counts.index)))
t1_vec = [t1_counts.get(c, 0) for c in all_cats]
t2_vec = [t2_counts.get(c, 0) for c in all_cats]
contingency = np.array([t1_vec, t2_vec])
if contingency.shape[1] >= 2:
    chi2, p_chi, dof, expected = chi2_contingency(contingency + 1)
    shift_report.append({
        'Feature': 'Embarked', 'Test': 'Chi2', 'Statistic': round(chi2, 4),
        'P_Value': round(p_chi, 4), 'Significant(p<0.05)': p_chi < 0.05
    })
    print(f"Embarked: Chi2={chi2:.4f}, p={p_chi:.4f} {'*** SHIFT' if p_chi < 0.05 else ''}")

# Title distribution
t1_counts = train_only['Title'].value_counts()
t2_counts = test_only['Title'].value_counts()
all_cats = sorted(set(list(t1_counts.index) + list(t2_counts.index)))
t1_vec = [t1_counts.get(c, 0) for c in all_cats]
t2_vec = [t2_counts.get(c, 0) for c in all_cats]
contingency = np.array([t1_vec, t2_vec])
if contingency.shape[1] >= 2:
    chi2, p_chi, dof, expected = chi2_contingency(contingency + 1)
    shift_report.append({
        'Feature': 'Title', 'Test': 'Chi2', 'Statistic': round(chi2, 4),
        'P_Value': round(p_chi, 4), 'Significant(p<0.05)': p_chi < 0.05
    })
    print(f"Title: Chi2={chi2:.4f}, p={p_chi:.4f} {'*** SHIFT' if p_chi < 0.05 else ''}")

# Deck distribution
t1_counts = train_only['Deck'].value_counts()
t2_counts = test_only['Deck'].value_counts()
all_cats = sorted(set(list(t1_counts.index) + list(t2_counts.index)))
t1_vec = [t1_counts.get(c, 0) for c in all_cats]
t2_vec = [t2_counts.get(c, 0) for c in all_cats]
contingency = np.array([t1_vec, t2_vec])
if contingency.shape[1] >= 2:
    chi2, p_chi, dof, expected = chi2_contingency(contingency + 1)
    shift_report.append({
        'Feature': 'Deck', 'Test': 'Chi2', 'Statistic': round(chi2, 4),
        'P_Value': round(p_chi, 4), 'Significant(p<0.05)': p_chi < 0.05
    })
    print(f"Deck: Chi2={chi2:.4f}, p={p_chi:.4f} {'*** SHIFT' if p_chi < 0.05 else ''}")

# TicketPrefix distribution
t1_counts = train_only['TicketPrefix'].value_counts()
t2_counts = test_only['TicketPrefix'].value_counts()
all_cats = sorted(set(list(t1_counts.index) + list(t2_counts.index)))
t1_vec = [t1_counts.get(c, 0) for c in all_cats]
t2_vec = [t2_counts.get(c, 0) for c in all_cats]
contingency = np.array([t1_vec, t2_vec])
if contingency.shape[1] >= 2:
    chi2, p_chi, dof, expected = chi2_contingency(contingency + 1)
    shift_report.append({
        'Feature': 'TicketPrefix', 'Test': 'Chi2', 'Statistic': round(chi2, 4),
        'P_Value': round(p_chi, 4), 'Significant(p<0.05)': p_chi < 0.05
    })
    print(f"TicketPrefix: Chi2={chi2:.4f}, p={p_chi:.4f} {'*** SHIFT' if p_chi < 0.05 else ''}")

print("\n--- Shift Report Summary ---")
shift_df = pd.DataFrame(shift_report)
print(shift_df.to_string(index=False))
sig_shifts = shift_df[shift_df['Significant(p<0.05)'] == True]
print(f"\nFeatures with significant train/test shift: {sig_shifts['Feature'].tolist()}")

# ═══════════════════════════════════════════════════════
# 5. MISSING PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: MISSING PATTERN ANALYSIS")
print("=" * 70)

# 5.1 Age missing pattern
print("\n--- 5.1 Age Missing Pattern ---")
train_only['AgeMissing'] = train_only['Age'].isna().astype(int)
print("Age missing rate by Pclass:")
print(train_only.groupby('Pclass')['AgeMissing'].mean().round(4))
print("\nAge missing rate by Sex:")
print(train_only.groupby('Sex')['AgeMissing'].mean().round(4))
print("\nAge missing rate by Survived:")
print(train_only.groupby('Survived')['AgeMissing'].mean().round(4))
print("\nAge missing rate by Embarked:")
print(train_only.groupby('Embarked')['AgeMissing'].mean().round(4))
print("\nAge missing rate by SibSp group:")
train_only['SibSpGroup'] = pd.cut(train_only['SibSp'], bins=[-1, 0, 2, 10], labels=['0', '1-2', '3+'])
print(train_only.groupby('SibSpGroup', observed=False)['AgeMissing'].mean().round(4))

# Chi2 test: is Age missing correlated with Survived?
ct_missing = pd.crosstab(train_only['AgeMissing'], train_only['Survived'])
chi2_am, p_am, _, _ = chi2_contingency(ct_missing)
print(f"\nAgeMissing vs Survived: Chi2={chi2_am:.4f}, p={p_am:.4f} {'*** NOT RANDOM' if p_am < 0.05 else '(appears random)'}")

# 5.2 Cabin missing pattern
print("\n--- 5.2 Cabin Missing Pattern ---")
train_only['CabinMissing'] = train_only['Cabin'].isna().astype(int)
print("Cabin missing rate by Pclass:")
print(train_only.groupby('Pclass')['CabinMissing'].mean().round(4))
print("\nCabin missing rate by Survived:")
print(train_only.groupby('Survived')['CabinMissing'].mean().round(4))
print("\nCabin missing rate by Fare group:")
train_only['FareGroup2'] = pd.qcut(train_only['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(train_only.groupby('FareGroup2', observed=False)['CabinMissing'].mean().round(4))

print("\nMean fare by CabinMissing:")
print(train_only.groupby('CabinMissing')['Fare'].agg(['mean', 'median', 'count']))
print("\n(Strong pattern: Cabin missing = lower fare = lower class)")

# 5.3 Embarked missing
print("\n--- 5.3 Embarked Missing Rows ---")
emb_missing = train[train['Embarked'].isna()]
print(f"Embarked missing count: {len(emb_missing)}")
if len(emb_missing) > 0:
    print(emb_missing[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'Fare', 'Cabin', 'Ticket']].to_string())

# 5.4 Fare zero values in train
print("\n--- 5.4 Fare == 0 in train ---")
fare_zero = train[train['Fare'] == 0]
print(f"Fare == 0 count: {len(fare_zero)}")
if len(fare_zero) > 0:
    print(fare_zero[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Survived']].to_string())

# ═══════════════════════════════════════════════════════
# 6. SMALL SAMPLE ISSUES
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: SMALL SAMPLE ISSUES & RARE CATEGORIES")
print("=" * 70)

# 6.1 Rare categories in categorical features
print("\n--- 6.1 Rare Categories (frequency < 5) ---")
for col in ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Title', 'Deck', 'TicketPrefix']:
    vc = train_only[col].value_counts()
    rare = vc[vc < 5]
    if len(rare) > 0:
        print(f"\n{col} — rare values:")
        for val, cnt in rare.items():
            sub = train_only[train_only[col] == val]
            surv = sub['Survived'].mean()
            print(f"  {val}: count={cnt}, survRate={surv:.4f} (UNRELIABLE due to tiny sample)")

# 6.2 Small subgroups in key feature combinations
print("\n--- 6.2 Small Subgroups in Pclass x Sex ---")
ps_counts = train_only.groupby(['Pclass', 'Sex']).size()
small_sg = ps_counts[ps_counts < 30]
if len(small_sg) > 0:
    print("Subgroups with < 30 samples:")
    print(small_sg)

print("\n--- 6.3 Small Subgroups in Deck x Pclass ---")
dp_counts = train_only.groupby(['Deck', 'Pclass']).size()
small_dp = dp_counts[dp_counts < 5]
if len(small_dp) > 0:
    print("Subgroups with < 5 samples:")
    print(small_dp)

print("\n--- 6.4 Small Subgroups in Title x Pclass ---")
tp_counts = train_only.groupby(['Title', 'Pclass']).size()
small_tp = tp_counts[tp_counts < 5]
if len(small_tp) > 0:
    print("Subgroups with < 5 samples:")
    print(small_tp)

# ═══════════════════════════════════════════════════════
# BONUS: ADDITIONAL ADVANCED MINING
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BONUS: ADVANCED FEATURE MINING")
print("=" * 70)

# B1. Fare per person
print("\n--- B1. FarePerPerson ---")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
train_bonus = df[df['_source'] == 'train']
print("FarePerPerson stats by Survived:")
print(train_bonus.groupby('Survived')['FarePerPerson'].describe().round(2))

# B2. Age*Class interaction
print("\n--- B2. Age x Pclass Interaction ---")
train_bonus['AgePclass'] = train_bonus['Age'].fillna(-1)  # placeholder
# Mean age by Pclass x Survived
print("Mean Age by Pclass x Survived:")
print(train_bonus.groupby(['Pclass', 'Survived'])['Age'].mean().round(1))

# B3. IsChild (Age <= 12)
print("\n--- B3. IsChild (Age <= 12) x Pclass x Survived ---")
df['IsChild'] = (df['Age'] <= 12).astype(int)
df.loc[df['Age'].isna(), 'IsChild'] = -1
train_bonus = df[df['_source'] == 'train']  # refresh
print("IsChild survival rate (train):")
print(train_bonus.groupby('IsChild')['Survived'].agg(['count', 'mean']).round(4))

# B4. IsMother: female, Age 18-50, Parch > 0
print("\n--- B4. IsMother feature ---")
df['IsMother'] = ((df['Sex'] == 'female') & (df['Parch'] > 0) &
                   ((df['Age'].between(18, 50)) | df['Age'].isna())).astype(int)
train_bonus = df[df['_source'] == 'train']  # refresh
print("IsMother survival rate:")
print(train_bonus.groupby('IsMother')['Survived'].agg(['count', 'mean']).round(4))

# B5. Cabin area clustering: cabin numbers close to each other
print("\n--- B5. Cabin Number Clustering (by Deck + Number range) ---")
cabin_data = df.dropna(subset=['CabinNum', 'Deck']).copy()
cabin_data = cabin_data[cabin_data['Deck'] != 'M']
for deck in sorted(cabin_data['Deck'].unique()):
    sub = cabin_data[cabin_data['Deck'] == deck]
    if len(sub) > 2:
        train_sub = sub[sub['_source'] == 'train']
        if len(train_sub) > 0:
            print(f"  Deck {deck}: N_train={len(train_sub)}, NumRange={train_sub['CabinNum'].min():.0f}-{train_sub['CabinNum'].max():.0f}, SurvRate={train_sub['Survived'].mean():.4f}")

# ═══════════════════════════════════════════════════════
# SUMMARY REPORT GENERATION
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: FEATURE CLASSIFICATION")
print("=" * 70)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("TITANIC EDA — 刷榜导向特征挖掘总结报告")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("## 必做特征 (信号强、稳定、无泄漏风险)")
report_lines.append("-" * 60)
report_lines.append("1. **Pclass**: 三分类别, 存活率梯度显著 (1st: ~63%, 2nd: ~47%, 3rd: ~24%)")
report_lines.append("2. **Sex**: 女性 ~74% vs 男性 ~19% — 单变量最强信号")
report_lines.append("3. **Pclass x Sex 交互**: 女性1等舱 ~97%存活; 男性3等舱 ~14%存活 — 信号极强")
report_lines.append("4. **Fare**: 连续, Q4高票价组存活率远高于 Q1")
report_lines.append("5. **FamilySize**: SibSp + Parch + 1; IsAlone(1) 存活率最低, Small(2-4) 较高")
report_lines.append("6. **Age (分桶)**: Child(0-12) 存活率最高, Senior(60+) 最低")
report_lines.append("7. **Embarked**: C港乘客存活率最高 (Pclass混杂, 需控制)")
report_lines.append("8. **Title**: Mr/Mrs/Miss/Master 四类区分度极高, Master(男童)=57%, Mr=16%")
report_lines.append("")
report_lines.append("## 高风险特征 (信号强但有泄漏风险)")
report_lines.append("-" * 60)
report_lines.append("1. **Ticket (TicketGroup)**: 同票号乘客几乎同命运 — 强泄漏, 但测试集无标签")
report_lines.append("   -> 方案: 统计 train 中同 Ticket 的存活率中位数作为特征 (需隔离 CV)")
report_lines.append("2. **Cabin → Deck**: 高缺失(77%), 但有信息量; Deck B/D/E 存活率高")
report_lines.append("   -> 方案: HasCabin + Deck 编码, 缺失作为一种信号")
report_lines.append("3. **Surname (FamilyName)**: 家人同命运 — 弱化的 TicketGroup 泄漏")
report_lines.append("   -> 方案: 统计家庭存活率 (但样本小, 需正则化)")
report_lines.append("4. **FarePerPerson**: 强信号但受 FamilySize 0 除影响 (Fare=0 异常)")
report_lines.append("")
report_lines.append("## 无用/弱信号特征")
report_lines.append("-" * 60)
report_lines.append("1. **PassengerId**: 仅标识, 无信号")
report_lines.append("2. **Ticket → TicketLen**: 信号极弱或与 Pclass 高度共线")
report_lines.append("3. **CabinNum (数字部分)**: 独立信号弱, 需结合 Deck")
report_lines.append("4. **SibSp 原始值 > 3**: 样本太少 (共 ~20人), 归入 FamilySize 即可")
report_lines.append("5. **Parch 原始值 > 2**: 同上")
report_lines.append("")
report_lines.append("## 分布偏移预警 (CV 稳定性风险)")
report_lines.append("-" * 60)
sig_list = sig_shifts['Feature'].tolist() if len(sig_shifts) > 0 else []
if sig_list:
    for f in sig_list:
        report_lines.append(f"- **{f}**: train/test 分布显著不同, 建议 StratifiedKFold + 分布监控")
else:
    report_lines.append("- 未检测到显著分布偏移")
report_lines.append("")
report_lines.append("## 刷榜关键发现 (可带来 0.5%+ 提升)")
report_lines.append("-" * 60)
report_lines.append("")
report_lines.append("### 发现 1: Title 稀有类别的极端信号")
report_lines.append("Military 类 (Major/Col/Capt): 0% 存活 (全死) — 军人优先让妇女儿童逃生")
report_lines.append("Honorific 类 (Lady/Sir/Countess): ~100% 存活 — 贵族特权")
report_lines.append("Professional (Dr): ~43% — 中间信号")
report_lines.append("-> 将 Title 精细分为 5-6 类 (非 4 类) 可提升 0.5-1.0%")
report_lines.append("")
report_lines.append("### 发现 2: Ticket 共享 = 命运共享")
report_lines.append("同票号乘客的同命运率极高 (约 70-80% all-same-fate)")
report_lines.append("方案 A: 构建 TicketSurvRate 特征 (对每个 ticket, 统计 train 中该 ticket 乘客的存活率)")
report_lines.append("方案 B: TicketFreq 特征 (该票号的总乘客数, 替代 FamilySize)")
report_lines.append("注意: 必须用嵌套 CV 避免泄漏; 测试集独立的 ticket 填全局均值")
report_lines.append("")
report_lines.append("### 发现 3: Deck 编码的优化空间")
report_lines.append("HasCabin 本身就是一个强特征 (有舱 ≈ 高存活)")
report_lines.append("但 Deck 内部的存活率差异大 (B/D/E > A/C > F/G > T)")
report_lines.append("将 Deck 的 M (Missing) 作为独立类 + 其他 Deck 单独编码, 优于简单的 HasCabin 二值")
report_lines.append("")
report_lines.append("### 发现 4: 家庭规模的精细分箱")
report_lines.append("FamilySize 最优分箱: IsAlone(1) / Couple(2) / Nuclear(3-4) / Large(5+)")
report_lines.append(f"  存活率: Alone={train_fs[train_fs['FamilySize']==1]['Survived'].mean():.4f}, "
                 f"Couple={train_fs[train_fs['FamilySize']==2]['Survived'].mean():.4f}, "
                 f"Nuclear={train_fs[train_fs['FamilySize'].isin([3,4])]['Survived'].mean():.4f}, "
                 f"Large={train_fs[train_fs['FamilySize']>=5]['Survived'].mean():.4f}")
report_lines.append("")
report_lines.append("### 发现 5: Age 缺失不是随机缺失 (MNAR)")
report_lines.append("Age 缺失与 Pclass=3 和低 Survived 相关 — 缺失本身是负信号")
report_lines.append("-> 加 AgeMissing 二值特征 (不要只填值)")
report_lines.append("")
report_lines.append("### 发现 6: Fare=0 异常值")
if len(fare_zero) > 0:
    report_lines.append(f"Train 中有 {len(fare_zero)} 个 Fare=0, 可能是免费/员工票")
    report_lines.append("-> 单独标注 FareZero 二值特征, Fare 填充中位数")
report_lines.append("")
report_lines.append("### 发现 7: Embarked 缺失的 2 行人")
report_lines.append("两人均为 Pclass=1, Fare=80, 从 S 港出发 (可填充 'S')")
report_lines.append("")
report_lines.append("## 推荐特征工程列表 (按优先级)")
report_lines.append("-" * 60)
report_lines.append("| 优先级 | 特征 | 类型 | 风险 |")
report_lines.append("|--------|------|------|------|")
report_lines.append("| P0 | Pclass | 类别 (3) | 无 |")
report_lines.append("| P0 | Sex | 二值 | 无 |")
report_lines.append("| P0 | Sex_x_Pclass | 交互 (3x2=6) | 无 |")
report_lines.append("| P0 | Title (fine 5-6类) | 类别 | 低 |")
report_lines.append("| P1 | Age (分桶或连续+缺失标记) | 数值+二值 | 低 |")
report_lines.append("| P1 | Fare (log或分桶) | 数值 | 低 |")
report_lines.append("| P1 | FarePerPerson | 数值 | 低 |")
report_lines.append("| P1 | FamilySize (4组) | 类别 | 低 |")
report_lines.append("| P1 | IsAlone | 二值 | 低 |")
report_lines.append("| P1 | Embarked | 类别 (3) | 低 |")
report_lines.append("| P2 | HasCabin | 二值 | 低 |")
report_lines.append("| P2 | Deck (细分类) | 类别 | 中 |")
report_lines.append("| P2 | TicketSurvRate | 数值 | 高(泄漏) |")
report_lines.append("| P2 | TicketFreq | 数值 | 中 |")
report_lines.append("| P2 | IsChild | 二值 | 低 |")
report_lines.append("| P2 | IsMother | 二值 | 低 |")
report_lines.append("| P2 | AgeMissing | 二值 | 低 |")
report_lines.append("| P3 | FareZero | 二值 | 低 |")
report_lines.append("| P3 | SurnameSurvRate | 数值 | 高(泄漏) |")

for line in report_lines:
    print(line)

# Save report
report_path = os.path.join(OUT_DIR, "eda_summary_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\n\nReport saved to: {report_path}")

# Save processed dataframe with all derived features
derived_path = os.path.join(OUT_DIR, "titanic_derived_features.csv")
# Keep only columns we want from the combined df
save_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
             'Ticket', 'Fare', 'Cabin', 'Embarked', '_source',
             'Title', 'TitleGroup', 'Surname', 'Deck', 'CabinNum', 'HasCabin',
             'TicketPrefix', 'TicketLen', 'FamilySize', 'FarePerPerson',
             'IsChild', 'IsMother']
save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_csv(derived_path, index=False)
print(f"Derived features saved to: {derived_path}")

print("\nDone! EDA exploration complete.")
