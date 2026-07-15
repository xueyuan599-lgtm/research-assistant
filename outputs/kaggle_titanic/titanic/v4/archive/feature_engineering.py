#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Titanic V4 Feature Engineering + Feature Selection Pipeline
===========================================================
All feature construction in engineer_features(df, is_train=True).
5-stage feature selection: MI -> Spearman -> Boruta -> Stability -> Permutation.

Run: python feature_engineering.py
Output: v4/X_train.csv, v4/X_test.csv, v4/feature_report.txt
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from boruta import BorutaPy

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(os.path.dirname(BASE_DIR), "train.csv")
TEST_PATH = os.path.join(os.path.dirname(BASE_DIR), "test.csv")


# ============================================================
# PART 1: FEATURE ENGINEERING
# ============================================================

def extract_title_safe(name):
    """Extract title from name, handling edge cases like 'Mrs. Martin (Elizabeth L'."""
    if pd.isna(name):
        return "Unknown"
    # Strategy: split by comma, take the part after comma, split by dot
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    after_comma = parts[1].strip()
    dot_pos = after_comma.find(".")
    if dot_pos == -1:
        return "Unknown"
    title = after_comma[:dot_pos].strip()
    return title


def map_title(title):
    """Map raw title to standardized category."""
    title = title.strip()
    mapping = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Mme": "Mrs",
        "Miss": "Miss",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Master": "Master",
        "Dr": "Professional",
        "Rev": "Clergy",
        "Col": "Military",
        "Major": "Military",
        "Capt": "Military",
        "Lady": "Honorific",
        "Sir": "Honorific",
        "Countess": "Honorific",
        "Don": "Honorific",
        "Dona": "Honorific",
        "Jonkheer": "Honorific",
    }
    return mapping.get(title, "Other")


def engineer_features(df, is_train=True, y=None, fitted_encoders=None, scaler=None):
    """
    Main feature engineering function.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Titanic data (train or test).
    is_train : bool
        Whether this is training data.
    y : pd.Series or None
        Target variable (Survived). Required for train mode for target encoding.
    fitted_encoders : dict or None
        Pre-fitted encoders from training data (for test mode).
    scaler : StandardScaler or None
        Pre-fitted scaler from training data (for test mode).

    Returns
    -------
    X : pd.DataFrame
        Encoded feature matrix.
    encoders : dict
        Fitted encoders (only when is_train=True).
    scaler : StandardScaler
        Fitted scaler (only when is_train=True).
    feature_cols_raw : list
        List of raw feature column names before one-hot expansion.
    onehot_group_map : dict
        Mapping from one-hot column prefix to list of generated columns.
    """
    data = df.copy()

    # ---- Track raw feature names (before encoding) ----
    raw_features = []  # List of (col_name, is_onehot_group, group_name_or_None)

    # ==========================================
    # 1. Missing Value Handling
    # ==========================================

    # AgeMissing indicator
    data["AgeMissing"] = data["Age"].isna().astype(int)
    raw_features.append("AgeMissing")

    # Age: Pclass x Sex group median imputation
    if is_train or (fitted_encoders is not None and "age_medians" in fitted_encoders):
        if is_train:
            age_medians = data.groupby(["Pclass", "Sex"])["Age"].transform("median")
            data["Age"] = data["Age"].fillna(age_medians)
            # Store age medians with ORIGINAL string Sex values (before binary encoding)
            _age_medians_store = data.groupby(["Pclass", "Sex"])["Age"].median().to_dict()
        else:
            for (pclass, sex), median_val in fitted_encoders["age_medians"].items():
                mask = (data["Pclass"] == pclass) & (data["Sex"] == sex)
                data.loc[mask, "Age"] = data.loc[mask, "Age"].fillna(median_val)
        raw_features.append("Age")
    else:
        data["Age"] = data["Age"].fillna(data["Age"].median())
        raw_features.append("Age")

    # Embarked: fill 'S'
    data["Embarked"] = data["Embarked"].fillna("S")

    # Fare: fill missing with Pclass median, log1p transform
    if is_train or (fitted_encoders is not None and "fare_medians" in fitted_encoders):
        if is_train:
            fare_medians = data.groupby("Pclass")["Fare"].transform("median")
            data["Fare"] = data["Fare"].fillna(fare_medians)
        else:
            for pclass, median_val in fitted_encoders["fare_medians"].items():
                mask = data["Pclass"] == pclass
                data.loc[mask, "Fare"] = data.loc[mask, "Fare"].fillna(median_val)
    else:
        data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # FareZero indicator
    data["FareZero"] = (data["Fare"] == 0).astype(int)
    raw_features.append("FareZero")

    # Fare_log
    data["Fare_log"] = np.log1p(data["Fare"])
    raw_features.append("Fare_log")

    # ==========================================
    # 2. A Class: Basic Features
    # ==========================================
    data["Pclass"] = data["Pclass"].astype(int)
    raw_features.append("Pclass")  # ordinal

    data["Sex"] = (data["Sex"] == "female").astype(int)
    raw_features.append("Sex")  # binary

    raw_features.append("Embarked")  # will be OneHot

    # ==========================================
    # 3. B Class: Family Features
    # ==========================================
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    raw_features.append("FamilySize")

    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    raw_features.append("IsAlone")

    # FamilySize_bin
    def bin_familysize(x):
        if x == 1:
            return "Alone"
        elif x == 2:
            return "Couple"
        elif 3 <= x <= 4:
            return "Nuclear"
        else:
            return "Large"

    data["FamilySize_bin"] = data["FamilySize"].apply(bin_familysize)
    raw_features.append("FamilySize_bin")  # will be OneHot

    # SibSp_bin
    def bin_sibsp(x):
        if x == 0:
            return "0"
        elif x == 1:
            return "1"
        else:
            return "2+"

    data["SibSp_bin"] = data["SibSp"].apply(bin_sibsp)
    raw_features.append("SibSp_bin")  # will be OneHot

    # Parch_bin
    def bin_parch(x):
        if x == 0:
            return "0"
        elif x == 1:
            return "1"
        else:
            return "2+"

    data["Parch_bin"] = data["Parch"].apply(bin_parch)
    raw_features.append("Parch_bin")  # will be OneHot

    # ==========================================
    # 4. C Class: Title Features
    # ==========================================
    # Extract raw title
    data["Title_raw"] = data["Name"].apply(extract_title_safe)
    # Map to standardized categories
    data["Title"] = data["Title_raw"].apply(map_title)
    raw_features.append("Title")  # will be Count + Label encoded

    # Title_Rare: frequency < 10 in training
    if is_train:
        title_counts = data["Title"].value_counts()
        rare_titles = set(title_counts[title_counts < 10].index)
    else:
        rare_titles = fitted_encoders.get("rare_titles", set())
    data["Title_Rare"] = data["Title"].isin(rare_titles).astype(int)
    raw_features.append("Title_Rare")

    # IsMaster
    data["IsMaster"] = (data["Title"] == "Master").astype(int)
    raw_features.append("IsMaster")

    # Title_Pclass interaction
    data["Title_Pclass"] = data["Title"].astype(str) + "_P" + data["Pclass"].astype(str)
    raw_features.append("Title_Pclass")  # will be Count encoded

    # TitleGroup (coarser grouping for OneHot)
    def title_group(t):
        if t in ("Mrs", "Miss"):
            return "Female_Honorific"
        elif t == "Mr":
            return "Male_Common"
        elif t == "Master":
            return "Child"
        elif t == "Professional":
            return "Professional"
        elif t == "Clergy":
            return "Clergy"
        elif t == "Military":
            return "Military"
        elif t == "Honorific":
            return "Honorific"
        else:
            return "Other"

    data["TitleGroup"] = data["Title"].apply(title_group)
    raw_features.append("TitleGroup")  # will be OneHot

    # ==========================================
    # 5. HasSpouse & HasChild
    # ==========================================
    # HasSpouse: Title==Mrs OR (Title==Mr and has family)
    data["HasSpouse"] = ((data["Title"] == "Mrs") |
                          ((data["Title"] == "Mr") & (data["FamilySize"] > 1))).astype(int)
    raw_features.append("HasSpouse")

    data["HasChild"] = (data["Parch"] > 0).astype(int)
    raw_features.append("HasChild")

    # ==========================================
    # 6. D Class: Deck/Cabin Features
    # ==========================================
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    raw_features.append("HasCabin")

    def extract_deck(cabin):
        if pd.isna(cabin):
            return "M"
        first_char = str(cabin).strip()[0]
        if first_char.isalpha() and first_char.upper() in "ABCDEFGT":
            return first_char.upper()
        return "M"

    data["Deck"] = data["Cabin"].apply(extract_deck)
    raw_features.append("Deck")  # will be Count encoded

    # CabinShared: same cabin number appears multiple times
    if is_train:
        cabin_counts = data["Cabin"].dropna().value_counts()
        shared_cabins = set(cabin_counts[cabin_counts > 1].index)
    else:
        shared_cabins = fitted_encoders.get("shared_cabins", set())
    data["CabinShared"] = data["Cabin"].apply(
        lambda x: int(pd.notna(x) and x in shared_cabins)
    )
    raw_features.append("CabinShared")

    # Deck_Pclass interaction
    data["Deck_Pclass"] = data["Deck"].astype(str) + "_P" + data["Pclass"].astype(str)
    raw_features.append("Deck_Pclass")  # will be Count encoded

    # CabinNum (numeric part of cabin, binned)
    def extract_cabin_num(cabin):
        if pd.isna(cabin):
            return "Missing"
        nums = re.findall(r"\d+", str(cabin))
        if not nums:
            return "Missing"
        return int(nums[0])

    def bin_cabin_num(x):
        if x == "Missing":
            return "Missing"
        if x <= 30:
            return "0-30"
        elif x <= 60:
            return "31-60"
        elif x <= 100:
            return "61-100"
        else:
            return "100+"

    data["CabinNum"] = data["Cabin"].apply(extract_cabin_num).apply(bin_cabin_num)
    raw_features.append("CabinNum")  # will be OneHot

    # ==========================================
    # 7. E Class: Ticket Features
    # ==========================================
    # TicketPrefix
    def extract_ticket_prefix(ticket):
        ticket_str = str(ticket).strip()
        # Extract leading alphabetic part
        match = re.match(r"^([A-Za-z\.\/]+)", ticket_str)
        if match:
            prefix = match.group(1).rstrip("./")
            return prefix if prefix else "NUM"
        return "NUM"

    data["TicketPrefix_raw"] = data["Ticket"].apply(extract_ticket_prefix)

    # Merge rare prefixes (freq < 5) into "Rare"
    if is_train:
        prefix_counts = data["TicketPrefix_raw"].value_counts()
        rare_prefixes = set(prefix_counts[prefix_counts < 5].index)
    else:
        rare_prefixes = fitted_encoders.get("rare_prefixes", set())

    data["TicketPrefix"] = data["TicketPrefix_raw"].apply(
        lambda x: "Rare" if x in rare_prefixes else x
    )
    raw_features.append("TicketPrefix")  # will be Count encoded

    # TicketLen
    data["TicketLen"] = data["Ticket"].apply(lambda x: len(str(x)))
    raw_features.append("TicketLen")

    # TicketNumLen
    data["TicketNumLen"] = data["Ticket"].apply(
        lambda x: len(re.sub(r"[^0-9]", "", str(x)))
    )
    raw_features.append("TicketNumLen")

    # TicketGroupSize
    ticket_group_sizes = data["Ticket"].value_counts()
    data["TicketGroupSize"] = data["Ticket"].map(ticket_group_sizes)
    raw_features.append("TicketGroupSize")

    # TicketSurvRate: Target Encoding with CV + smoothing
    # We handle this in the target encoding section since it needs y
    # For now, create placeholder (will be overwritten)
    data["TicketSurvRate"] = 0.0
    raw_features.append("TicketSurvRate")

    # TicketSurvivedAll / TicketSurvivedNone
    # These also need y, handled in target encoding section
    data["TicketAllSurvived"] = 0.0
    data["TicketNoneSurvived"] = 0.0
    raw_features.append("TicketAllSurvived")
    raw_features.append("TicketNoneSurvived")

    # ==========================================
    # 8. F Class: Interaction Features
    # ==========================================
    # F1: Sex_Pclass
    data["Sex_Pclass"] = data["Sex"].apply(lambda x: "F" if x == 1 else "M") + "_P" + data["Pclass"].astype(str)
    raw_features.append("Sex_Pclass")  # will be OneHot

    # F2: Sex_Age
    data["Sex_Age"] = data["Sex"] * data["Age"]
    raw_features.append("Sex_Age")

    # F3: Pclass_Age
    data["Pclass_Age"] = data["Pclass"] * data["Age"]
    raw_features.append("Pclass_Age")

    # F4: Pclass_Fare
    data["Pclass_Fare"] = data["Pclass"] * data["Fare_log"]
    raw_features.append("Pclass_Fare")

    # F5: Sex_FamilySize
    data["Sex_FamilySize"] = data["Sex"].apply(lambda x: "F" if x == 1 else "M") + "_FS" + data["FamilySize_bin"].astype(str)
    raw_features.append("Sex_FamilySize")  # will be OneHot

    # F6: Pclass_FamilySize
    data["Pclass_FamilySize"] = "P" + data["Pclass"].astype(str) + "_" + data["FamilySize_bin"].astype(str)
    raw_features.append("Pclass_FamilySize")  # will be OneHot

    # F7: Pclass_Embarked
    data["Pclass_Embarked"] = "P" + data["Pclass"].astype(str) + "_" + data["Embarked"].astype(str)
    raw_features.append("Pclass_Embarked")  # will be OneHot

    # F8: Sex_Embarked
    data["Sex_Embarked"] = data["Sex"].apply(lambda x: "F" if x == 1 else "M") + "_" + data["Embarked"].astype(str)
    raw_features.append("Sex_Embarked")  # will be OneHot

    # F9: Age_Pclass_Sex (3-way interaction, aggressive)
    def bin_age(age):
        if pd.isna(age):
            return "Unknown"
        if age <= 12:
            return "Child"
        elif age <= 18:
            return "Teen"
        elif age <= 35:
            return "YoungAdult"
        elif age <= 55:
            return "Adult"
        else:
            return "Senior"

    data["Age_bin"] = data["Age"].apply(bin_age)
    data["Age_Pclass_Sex"] = (
        data["Age_bin"].astype(str) + "_P" + data["Pclass"].astype(str) + "_" +
        data["Sex"].apply(lambda x: "F" if x == 1 else "M")
    )
    raw_features.append("Age_Pclass_Sex")  # will be OneHot

    # ==========================================
    # 9. G Class: Aggregate Statistics
    # ==========================================

    # G1: FarePerPerson
    data["FarePerPerson"] = data["Fare"] / data["FamilySize"]
    raw_features.append("FarePerPerson")

    # G2: Pclass_FareMean (within CV fold for train, global for test)
    if is_train:
        data["Pclass_FareMean"] = data.groupby("Pclass")["Fare"].transform("mean")
    else:
        pclass_fare_map = fitted_encoders.get("pclass_fare_mean", {})
        data["Pclass_FareMean"] = data["Pclass"].map(pclass_fare_map)
    raw_features.append("Pclass_FareMean")

    # G3: Pclass_FareRank (within Pclass, quantile)
    if is_train:
        data["Pclass_FareRank"] = data.groupby("Pclass")["Fare"].transform(
            lambda x: x.rank(pct=True)
        )
    else:
        pclass_fare_quantiles = fitted_encoders.get("pclass_fare_quantiles", {})
        data["Pclass_FareRank"] = 0.0
        for pclass, quantile_df in pclass_fare_quantiles.items():
            mask = data["Pclass"] == pclass
            data.loc[mask, "Pclass_FareRank"] = data.loc[mask, "Fare"].apply(
                lambda f: np.searchsorted(quantile_df["fare_sorted"], f) / max(len(quantile_df), 1)
            )
    raw_features.append("Pclass_FareRank")

    # G4: Title_SurvivalRate (Target Encoding, handled in target encoding section)
    data["Title_SurvivalRate"] = 0.0
    raw_features.append("Title_SurvivalRate")

    # G5: Deck_SurvivalRate (Target Encoding, handled in target encoding section)
    data["Deck_SurvivalRate"] = 0.0
    raw_features.append("Deck_SurvivalRate")

    # G6: FamilyID (surname + family size)
    data["Surname"] = data["Name"].apply(lambda n: n.split(",")[0].strip() if pd.notna(n) else "Unknown")
    data["FamilyID"] = data["Surname"].astype(str) + "_" + data["FamilySize"].astype(str)

    # FamilyID count
    if is_train:
        fid_counts = data["FamilyID"].value_counts()
        data["FamilyID_Count"] = data["FamilyID"].map(fid_counts)
    else:
        fid_counts_map = fitted_encoders.get("familyid_counts", {})
        data["FamilyID_Count"] = data["FamilyID"].map(fid_counts_map).fillna(1)
    raw_features.append("FamilyID_Count")

    # ==========================================
    # 10. Target Encoding (LOOCV + smoothing)
    # ==========================================
    if is_train and y is not None:
        global_mean = y.mean()

        # --- TicketSurvRate: 5-fold CV Target Encoding ---
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        ticket_surv_rate = np.zeros(len(data))
        smoothing_ticket = 20

        for train_idx, val_idx in skf.split(data, y):
            y_train_fold = y.iloc[train_idx]
            ticket_fold = data.iloc[train_idx]["Ticket"]
            ticket_val = data.iloc[val_idx]["Ticket"]

            # Compute ticket stats from training fold
            ticket_df = pd.DataFrame({"Ticket": ticket_fold, "Survived": y_train_fold})
            ticket_stats = ticket_df.groupby("Ticket")["Survived"].agg(["mean", "count"])
            ticket_stats.columns = ["mean", "n"]

            for i, idx in enumerate(val_idx):
                t = ticket_val.iloc[i]
                if t in ticket_stats.index:
                    n_cat = ticket_stats.loc[t, "n"]
                    cat_mean = ticket_stats.loc[t, "mean"]
                    smoothed = (n_cat * cat_mean + smoothing_ticket * global_mean) / (n_cat + smoothing_ticket)
                else:
                    smoothed = global_mean
                ticket_surv_rate[idx] = smoothed

        data["TicketSurvRate"] = ticket_surv_rate

        # TicketSurvivedAll / TicketSurvivedNone
        ticket_group_stats = data.groupby("Ticket")["TicketSurvRate"].transform("mean")
        data["TicketAllSurvived"] = ((ticket_group_stats > 0.95) & (data["TicketGroupSize"] > 1)).astype(int)
        data["TicketNoneSurvived"] = ((ticket_group_stats < 0.05) & (data["TicketGroupSize"] > 1)).astype(int)

        # --- Title_SurvivalRate: LOOCV Target Encoding ---
        title_surv_rate = np.zeros(len(data))
        smoothing_title = 10
        # Compute group stats from y (not data - Survived was already dropped)
        title_stats_df = pd.DataFrame({"Title": data["Title"], "Survived": y.values})
        title_stats = title_stats_df.groupby("Title")["Survived"].agg(["mean", "count"])

        titles_arr = data["Title"].values
        for i in range(len(data)):
            t = titles_arr[i]
            n_cat = title_stats.loc[t, "count"]
            if n_cat > 1:
                # LOOCV: exclude self
                cat_mean_loo = (title_stats.loc[t, "mean"] * n_cat - y.iloc[i]) / (n_cat - 1)
                smoothed = ((n_cat - 1) * cat_mean_loo + smoothing_title * global_mean) / (n_cat - 1 + smoothing_title)
            else:
                smoothed = global_mean
            title_surv_rate[i] = smoothed

        data["Title_SurvivalRate"] = title_surv_rate

        # --- Deck_SurvivalRate: LOOCV Target Encoding ---
        deck_surv_rate = np.zeros(len(data))
        smoothing_deck = 10
        deck_stats_df = pd.DataFrame({"Deck": data["Deck"], "Survived": y.values})
        deck_stats = deck_stats_df.groupby("Deck")["Survived"].agg(["mean", "count"])

        decks_arr = data["Deck"].values
        for i in range(len(data)):
            d = decks_arr[i]
            n_cat = deck_stats.loc[d, "count"]
            if n_cat > 1:
                cat_mean_loo = (deck_stats.loc[d, "mean"] * n_cat - y.iloc[i]) / (n_cat - 1)
                smoothed = ((n_cat - 1) * cat_mean_loo + smoothing_deck * global_mean) / (n_cat - 1 + smoothing_deck)
            else:
                smoothed = global_mean
            deck_surv_rate[i] = smoothed

        data["Deck_SurvivalRate"] = deck_surv_rate

    elif not is_train and fitted_encoders is not None:
        global_mean_enc = fitted_encoders.get("global_surv_mean", 0.3838)

        # Test set: use training set statistics
        ticket_stats_map = fitted_encoders.get("ticket_surv_stats", {})
        smoothing_ticket = 20
        ticket_surv_values = []
        for t in data["Ticket"]:
            if t in ticket_stats_map:
                n_cat, cat_mean = ticket_stats_map[t]
                smoothed = (n_cat * cat_mean + smoothing_ticket * global_mean_enc) / (n_cat + smoothing_ticket)
            else:
                smoothed = global_mean_enc
            ticket_surv_values.append(smoothed)
        data["TicketSurvRate"] = ticket_surv_values

        title_stats_map = fitted_encoders.get("title_surv_stats", {})
        smoothing_title = 10
        title_surv_values = []
        for t in data["Title"]:
            if t in title_stats_map:
                n_cat, cat_mean = title_stats_map[t]
                smoothed = (n_cat * cat_mean + smoothing_title * global_mean_enc) / (n_cat + smoothing_title)
            else:
                smoothed = global_mean_enc
            title_surv_values.append(smoothed)
        data["Title_SurvivalRate"] = title_surv_values

        deck_stats_map = fitted_encoders.get("deck_surv_stats", {})
        smoothing_deck = 10
        deck_surv_values = []
        for d in data["Deck"]:
            if d in deck_stats_map:
                n_cat, cat_mean = deck_stats_map[d]
                smoothed = (n_cat * cat_mean + smoothing_deck * global_mean_enc) / (n_cat + smoothing_deck)
            else:
                smoothed = global_mean_enc
            deck_surv_values.append(smoothed)
        data["Deck_SurvivalRate"] = deck_surv_values

        # TicketAllSurvived / TicketNoneSurvived for test
        ticket_group = data.groupby("Ticket")["TicketSurvRate"].transform("mean")
        data["TicketAllSurvived"] = ((ticket_group > 0.95) & (data["TicketGroupSize"] > 1)).astype(int)
        data["TicketNoneSurvived"] = ((ticket_group < 0.05) & (data["TicketGroupSize"] > 1)).astype(int)

    # ==========================================
    # 11. Encoding Pipeline
    # ==========================================
    # Pass age_medians_store to encode_features so it uses string Sex keys
    extra_fit = {}
    if is_train:
        extra_fit["age_medians"] = _age_medians_store

    result = encode_features(
        data, raw_features, is_train, y, fitted_encoders, scaler, extra_fit
    )
    if is_train:
        X, fitted, onehot_group_map, scaler_fitted = result
        return X, fitted, onehot_group_map, scaler_fitted
    else:
        X, _, onehot_group_map, _ = result
        return X, None, onehot_group_map, None


def encode_features(data, raw_features, is_train, y, fitted_encoders, scaler, extra_fit=None):
    """
    Encode all features and return numerical matrix.

    Encoding order:
    1. Binary: 0/1 (already done)
    2. Ordinal: Pclass 1/2/3
    3. OneHot: low cardinality (< 8 categories)
    4. Count/Frequency Encoding: high cardinality (> 8 categories)
    5. Target Encoding: already computed
    6. StandardScaler: continuous numeric columns
    """
    encoded_parts = []
    column_names = []
    onehot_group_map = {}  # prefix -> [column names]
    fitted = {} if is_train else None
    scaler_out = None

    # Identify which columns to encode
    # Some are already binary/numeric, some need encoding

    # Collect columns that need OneHot encoding
    onehot_candidates = [
        "Embarked", "FamilySize_bin", "SibSp_bin", "Parch_bin",
        "TitleGroup", "CabinNum", "Sex_Pclass", "Sex_FamilySize",
        "Pclass_FamilySize", "Pclass_Embarked", "Sex_Embarked",
        "Age_Pclass_Sex"
    ]

    # Collect columns that need Count Encoding (high cardinality)
    count_encode_candidates = [
        "Title", "Title_Pclass", "Deck", "Deck_Pclass", "TicketPrefix"
    ]

    # Binary columns (already encoded as 0/1)
    binary_cols = [
        "Sex", "AgeMissing", "FareZero", "IsAlone", "Title_Rare",
        "IsMaster", "HasSpouse", "HasChild", "HasCabin", "CabinShared",
        "TicketAllSurvived", "TicketNoneSurvived"
    ]

    # Numeric columns (use StandardScaler)
    numeric_cols = [
        "Age", "Fare_log", "FamilySize", "TicketLen", "TicketNumLen",
        "TicketGroupSize", "TicketSurvRate", "Sex_Age", "Pclass_Age",
        "Pclass_Fare", "FarePerPerson", "Pclass_FareMean",
        "Pclass_FareRank", "Title_SurvivalRate", "Deck_SurvivalRate",
        "FamilyID_Count"
    ]

    # Ordinal columns
    ordinal_cols = ["Pclass"]

    # Build encoded matrix step by step
    df_encoded = data.copy()

    # Step 1: Binary (already done in engineer_features)
    pass

    # Step 2: Ordinal (already numeric 1/2/3)
    pass

    # Step 3: OneHot Encoding
    for col in onehot_candidates:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=int)
            # Track group for Spearman dedup
            onehot_group_map[col] = list(dummies.columns)
            encoded_parts.append(dummies)
            column_names.extend(dummies.columns)

    # Step 4: Count/Frequency Encoding
    for col in count_encode_candidates:
        if col in df_encoded.columns:
            if is_train:
                counts = df_encoded[col].value_counts()
                encoded_vals = df_encoded[col].map(counts).values.reshape(-1, 1)
                if fitted is not None:
                    fitted[f"count_{col}"] = counts
            else:
                count_map = fitted_encoders.get(f"count_{col}", {})
                encoded_vals = df_encoded[col].map(count_map).fillna(1).values.reshape(-1, 1)

            temp_df = pd.DataFrame(encoded_vals, columns=[f"{col}_count"], index=df_encoded.index)
            encoded_parts.append(temp_df)
            column_names.append(f"{col}_count")

    # Step 5: Binary features
    for col in binary_cols:
        if col in df_encoded.columns:
            temp_df = pd.DataFrame(df_encoded[col].values.reshape(-1, 1),
                                    columns=[col], index=df_encoded.index)
            encoded_parts.append(temp_df)
            column_names.append(col)

    # Step 6: Ordinal features
    for col in ordinal_cols:
        if col in df_encoded.columns:
            temp_df = pd.DataFrame(df_encoded[col].values.reshape(-1, 1),
                                    columns=[col], index=df_encoded.index)
            encoded_parts.append(temp_df)
            column_names.append(col)

    # Step 7: Numeric features
    numeric_features_df = pd.DataFrame(index=df_encoded.index)
    numeric_feature_cols = []
    for col in numeric_cols:
        if col in df_encoded.columns:
            numeric_features_df[col] = df_encoded[col].astype(float)
            numeric_feature_cols.append(col)

    # StandardScaler
    if is_train:
        scaler_out = StandardScaler()
        numeric_scaled = scaler_out.fit_transform(numeric_features_df)
    else:
        scaler_out = scaler
        numeric_scaled = scaler_out.transform(numeric_features_df)

    numeric_scaled_df = pd.DataFrame(
        numeric_scaled, columns=numeric_feature_cols, index=df_encoded.index
    )
    encoded_parts.append(numeric_scaled_df)
    column_names.extend(numeric_feature_cols)

    # Concatenate all encoded parts
    X = pd.concat(encoded_parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]  # Remove duplicate columns

    # Store final column order
    if is_train:
        fitted["feature_columns"] = list(X.columns)
        fitted["rare_titles"] = set(data["Title"].value_counts()[data["Title"].value_counts() < 10].index)
        fitted["rare_prefixes"] = set(data["TicketPrefix_raw"].value_counts()[data["TicketPrefix_raw"].value_counts() < 5].index)
        fitted["shared_cabins"] = set(data["Cabin"].dropna().value_counts()[data["Cabin"].dropna().value_counts() > 1].index)
        # Use age_medians from extra_fit (computed with string Sex values, before binary encoding)
        fitted["age_medians"] = extra_fit.get("age_medians", {}) if extra_fit else {}
        fitted["fare_medians"] = data.groupby("Pclass")["Fare"].median().to_dict()
        fitted["pclass_fare_mean"] = data.groupby("Pclass")["Fare"].mean().to_dict()

        # Store fare quantile boundaries for test set
        pclass_fare_quantiles = {}
        for pclass, group in data.groupby("Pclass"):
            pclass_fare_quantiles[pclass] = pd.DataFrame({
                "fare_sorted": sorted(group["Fare"].values)
            })
        fitted["pclass_fare_quantiles"] = pclass_fare_quantiles

        fitted["familyid_counts"] = data["FamilyID"].value_counts().to_dict()
        fitted["global_surv_mean"] = y.mean() if y is not None else 0.3838

        # Store ticket-level stats for test set (use y, not data - Survived was dropped)
        if y is not None:
            ticket_stats_df = pd.DataFrame({"Ticket": data["Ticket"], "Survived": y.values})
            ticket_stats = ticket_stats_df.groupby("Ticket")["Survived"].agg(["mean", "count"])
            fitted["ticket_surv_stats"] = {t: (row["count"], row["mean"])
                                           for t, row in ticket_stats.iterrows()}

            title_stats_df = pd.DataFrame({"Title": data["Title"], "Survived": y.values})
            title_stats = title_stats_df.groupby("Title")["Survived"].agg(["mean", "count"])
            fitted["title_surv_stats"] = {t: (row["count"], row["mean"])
                                          for t, row in title_stats.iterrows()}

            deck_stats_df = pd.DataFrame({"Deck": data["Deck"], "Survived": y.values})
            deck_stats = deck_stats_df.groupby("Deck")["Survived"].agg(["mean", "count"])
            fitted["deck_surv_stats"] = {d: (row["count"], row["mean"])
                                         for d, row in deck_stats.iterrows()}

        return X, fitted, onehot_group_map, scaler_out
    else:
        return X, None, onehot_group_map, scaler_out


# ============================================================
# PART 2: FEATURE SELECTION
# ============================================================

def onehot_parent(col_name, onehot_group_map):
    """Return the parent feature name if col is a one-hot dummy."""
    for parent, children in onehot_group_map.items():
        if col_name in children:
            return parent
    return None


def feature_selection(X, y, onehot_group_map, output_dir):
    """
    5-stage feature selection.

    Returns selected feature list.
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TITANIC V4 FEATURE SELECTION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Total candidate features: {X.shape[1]}")
    report_lines.append("")

    current_features = list(X.columns)
    n_total = len(current_features)

    # ==========================================
    # Stage 1: Mutual Information Filter
    # ==========================================
    report_lines.append("-" * 50)
    report_lines.append("Stage 1: Mutual Information Filter (MI > 0.01)")
    report_lines.append("-" * 50)

    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X.columns, "mi": mi}).sort_values("mi", ascending=False)

    stage1_kept = mi_df[mi_df["mi"] > 0.01]["feature"].tolist()
    stage1_dropped = mi_df[mi_df["mi"] <= 0.01]["feature"].tolist()

    report_lines.append(f"Kept: {len(stage1_kept)} features")
    report_lines.append(f"Dropped: {len(stage1_dropped)} features")
    if stage1_dropped:
        for f in stage1_dropped:
            mi_val = mi_df[mi_df["feature"] == f]["mi"].values[0]
            report_lines.append(f"  - {f}: MI={mi_val:.6f}")
    report_lines.append("")

    X_stage1 = X[stage1_kept]
    current_features = stage1_kept

    # ==========================================
    # Stage 2: Spearman Correlation Dedup
    # ==========================================
    report_lines.append("-" * 50)
    report_lines.append("Stage 2: Spearman Correlation Dedup (|r| > 0.85)")
    report_lines.append("-" * 50)

    corr_matrix = X_stage1.corr(method="spearman")

    # Build current MI map for tie-breaking
    mi_map = {f: v for f, v in zip(X.columns, mi)}

    # Identify target-encoded features (contain Survived info in their construction)
    # These have artificially inflated MI and should NOT be allowed to out-compete
    # clean features like Sex, Pclass, etc.
    target_encoded_patterns = [
        "Title_SurvivalRate", "Deck_SurvivalRate", "TicketSurvRate",
        "TicketAllSurvived", "TicketNoneSurvived"
    ]

    def is_target_encoded(feature_name):
        for pattern in target_encoded_patterns:
            if pattern in feature_name:
                return True
        return False

    # Identify protected clean features that should never be dropped by derived features
    protected_clean = {"Sex", "Pclass", "Age", "Fare_log", "IsAlone", "FamilySize",
                       "AgeMissing", "FareZero", "HasCabin", "IsMaster", "HasSpouse",
                       "HasChild", "Title_Rare", "CabinShared"}

    to_drop = set()
    checked = set()

    for i, f1 in enumerate(current_features):
        for j, f2 in enumerate(current_features):
            if j <= i:
                continue
            pair = tuple(sorted([f1, f2]))
            if pair in checked:
                continue
            checked.add(pair)

            if abs(corr_matrix.loc[f1, f2]) > 0.85:
                # Check if one of them is a one-hot dummy
                parent1 = onehot_parent(f1, onehot_group_map)
                parent2 = onehot_parent(f2, onehot_group_map)

                # If same onehot group, don't drop (structural correlation)
                if parent1 is not None and parent1 == parent2:
                    continue

                # Rule 1: Never dedup between a target-encoded feature and a clean feature
                # Target-encoded features have artificially high MI from Survived leakage
                f1_is_te = is_target_encoded(f1)
                f2_is_te = is_target_encoded(f2)
                if f1_is_te != f2_is_te:
                    report_lines.append(f"  {f1} vs {f2}: r={corr_matrix.loc[f1, f2]:.3f}, SKIP (target-encoded vs clean, keep both)")
                    continue

                # Rule 2: Protected clean features should not be dropped by derived features
                f1_is_protected = f1 in protected_clean
                f2_is_protected = f2 in protected_clean
                if f1_is_protected != f2_is_protected:
                    # Drop the non-protected one
                    if f1_is_protected:
                        to_drop.add(f2)
                        report_lines.append(f"  {f1} (protected) vs {f2}: r={corr_matrix.loc[f1, f2]:.3f}, drop {f2}")
                    else:
                        to_drop.add(f1)
                        report_lines.append(f"  {f2} (protected) vs {f1}: r={corr_matrix.loc[f1, f2]:.3f}, drop {f1}")
                    continue

                # Rule 3: For ties between two derived features, drop the one with lower MI
                mi1 = mi_map.get(f1, 0)
                mi2 = mi_map.get(f2, 0)

                if mi1 >= mi2:
                    to_drop.add(f2)
                    report_lines.append(f"  {f1} vs {f2}: r={corr_matrix.loc[f1, f2]:.3f}, drop {f2} (MI={mi2:.4f} < {mi1:.4f})")
                else:
                    to_drop.add(f1)
                    report_lines.append(f"  {f1} vs {f2}: r={corr_matrix.loc[f1, f2]:.3f}, drop {f1} (MI={mi1:.4f} < {mi2:.4f})")

    # Post-hoc: ensure protected features are never in to_drop
    to_drop = to_drop - protected_clean

    stage2_kept = [f for f in current_features if f not in to_drop]
    report_lines.append(f"Kept: {len(stage2_kept)} features")
    report_lines.append(f"Dropped: {len(to_drop)} features")
    report_lines.append("")

    X_stage2 = X_stage1[stage2_kept]
    current_features = stage2_kept

    # ==========================================
    # Stage 3: Boruta
    # ==========================================
    report_lines.append("-" * 50)
    report_lines.append("Stage 3: Boruta (RF, n_estimators=200, max_depth=5)")
    report_lines.append("-" * 50)

    rf_boruta = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
    )

    try:
        boruta = BorutaPy(
            rf_boruta, n_estimators="auto", perc=100, max_iter=100,
            random_state=RANDOM_STATE, verbose=0
        )
        boruta.fit(X_stage2.values, y.values)

        boruta_decision = boruta.support_  # True = Accepted
        boruta_weak = boruta.support_weak_  # True = Tentative

        stage3_accepted = [f for f, d in zip(current_features, boruta_decision) if d]
        stage3_tentative = [f for f, d in zip(current_features, boruta_weak) if d and not boruta_decision[np.where(np.array(current_features) == f)[0][0]]]
        stage3_rejected = [f for f in current_features
                          if f not in stage3_accepted and f not in stage3_tentative]

        # Fix: compute tentative properly
        boruta_decisions = []
        for i, f in enumerate(current_features):
            if boruta_decision[i]:
                boruta_decisions.append("Accepted")
            elif boruta_weak[i]:
                boruta_decisions.append("Tentative")
            else:
                boruta_decisions.append("Rejected")

        stage3_accepted = [f for f, d in zip(current_features, boruta_decisions) if d == "Accepted"]
        stage3_tentative = [f for f, d in zip(current_features, boruta_decisions) if d == "Tentative"]
        stage3_rejected = [f for f, d in zip(current_features, boruta_decisions) if d == "Rejected"]

        report_lines.append(f"Accepted: {len(stage3_accepted)}")
        report_lines.append(f"Tentative: {len(stage3_tentative)}")
        report_lines.append(f"Rejected: {len(stage3_rejected)}")

    except Exception as e:
        report_lines.append(f"Boruta failed: {e}. Falling back to custom implementation.")

        # Custom Boruta fallback
        n_shadow = 5
        rf_custom = RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
        )

        X_shadow = X_stage2.values.copy()
        shadow_names = []
        for i in range(X_shadow.shape[1]):
            for s in range(n_shadow):
                shadow_col = np.random.permutation(X_shadow[:, i])
                X_shadow = np.column_stack([X_shadow, shadow_col])
                shadow_names.append(f"shadow_{i}_{s}")

        rf_custom.fit(X_shadow, y.values)
        importances = rf_custom.feature_importances_

        real_importance = importances[:X_stage2.shape[1]]
        shadow_importance = importances[X_stage2.shape[1]:]

        # Max shadow importance per original feature
        shadow_max_per_feature = []
        for i in range(X_stage2.shape[1]):
            shadow_idx_start = i * n_shadow
            shadow_max_per_feature.append(
                np.max(shadow_importance[shadow_idx_start:shadow_idx_start + n_shadow])
            )

        shadow_max = np.max(shadow_max_per_feature)

        boruta_decisions = []
        stage3_accepted = []
        stage3_rejected = []
        stage3_tentative = []

        for i, f in enumerate(current_features):
            if real_importance[i] > shadow_max:
                boruta_decisions.append("Accepted")
                stage3_accepted.append(f)
            else:
                boruta_decisions.append("Rejected")
                stage3_rejected.append(f)

        report_lines.append(f"Custom Boruta - Accepted: {len(stage3_accepted)}")
        report_lines.append(f"Custom Boruta - Rejected: {len(stage3_rejected)}")

    report_lines.append(f"Top 5 by importance:")
    for i, f in enumerate(current_features):
        if boruta_decisions[i] == "Accepted":
            imp = rf_boruta.feature_importances_[i] if 'rf_boruta' in dir() else importances[i]
            report_lines.append(f"  {f}: importance={imp:.6f}")
    report_lines.append("")

    X_stage3 = X_stage2[list(current_features)]  # Keep all for next stages

    # ==========================================
    # Stage 4: Stability Selection
    # ==========================================
    report_lines.append("-" * 50)
    report_lines.append("Stage 4: Stability Selection (L1-LR, 100 bootstrap x 0.7)")
    report_lines.append("-" * 50)

    n_bootstrap = 100
    sample_frac = 0.7
    n_samples = int(len(X_stage3) * sample_frac)

    selection_counts = np.zeros(len(current_features))

    for b in range(n_bootstrap):
        idx = np.random.RandomState(RANDOM_STATE + b).choice(
            len(X_stage3), size=n_samples, replace=False
        )
        X_boot = X_stage3.values[idx]
        y_boot = y.values[idx]

        lr = LogisticRegression(
            penalty="l1", C=0.5, solver="saga", max_iter=5000,
            random_state=RANDOM_STATE + b
        )
        lr.fit(X_boot, y_boot)
        selection_counts += (np.abs(lr.coef_[0]) > 1e-6).astype(int)

    selection_probs = selection_counts / n_bootstrap

    stage4_kept = [f for f, p in zip(current_features, selection_probs) if p > 0.6]
    stage4_dropped = [f for f, p in zip(current_features, selection_probs) if p <= 0.6]

    report_lines.append(f"Kept (prob > 0.6): {len(stage4_kept)}")
    report_lines.append(f"Dropped: {len(stage4_dropped)}")
    for f, p in sorted(zip(current_features, selection_probs), key=lambda x: -x[1]):
        status = "KEPT" if p > 0.6 else "DROPPED"
        if status == "DROPPED" or p < 0.8:
            report_lines.append(f"  {f}: prob={p:.2f} [{status}]")
    report_lines.append("")

    # ==========================================
    # Stage 5: Permutation Importance
    # ==========================================
    report_lines.append("-" * 50)
    report_lines.append("Stage 5: Permutation Importance (RF, n_repeats=20)")
    report_lines.append("-" * 50)

    rf_perm = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_perm.fit(X_stage3, y)

    perm_result = permutation_importance(
        rf_perm, X_stage3, y, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
    )

    perm_mean = perm_result.importances_mean
    perm_std = perm_result.importances_std

    stage5_kept = []
    stage5_dropped = []
    for i, f in enumerate(current_features):
        if perm_mean[i] - 2 * perm_std[i] > 0:
            stage5_kept.append(f)
        else:
            stage5_dropped.append(f)

    report_lines.append(f"Kept (mean - 2*std > 0): {len(stage5_kept)}")
    report_lines.append(f"Dropped: {len(stage5_dropped)}")
    for f, m, s in sorted(
        zip(current_features, perm_mean, perm_std),
        key=lambda x: -x[1]
    ):
        status = "KEPT" if m - 2 * s > 0 else "DROPPED"
        if status == "DROPPED" or m < 0.02:
            report_lines.append(f"  {f}: mean={m:.4f}, std={s:.4f}, mean-2std={m-2*s:.4f} [{status}]")
    report_lines.append("")

    # ==========================================
    # Final Decision: Three-way intersection + 2-way candidates
    # ==========================================
    report_lines.append("=" * 70)
    report_lines.append("FINAL FEATURE SET DECISION")
    report_lines.append("=" * 70)

    # Determine which features passed each stage
    stage3_pass = set(stage3_accepted)
    # Tentative features are borderline - treat as passed for intersection
    stage3_pass = stage3_pass.union(set(stage3_tentative))
    stage4_pass = set(stage4_kept)
    stage5_pass = set(stage5_kept)

    # Count votes for each feature
    all_candidates = set(current_features)
    vote_counts = {}
    for f in all_candidates:
        votes = 0
        if f in stage3_pass:
            votes += 1
        if f in stage4_pass:
            votes += 1
        if f in stage5_pass:
            votes += 1
        vote_counts[f] = votes

    # Three-way intersection (3 votes)
    final_core = sorted([f for f, v in vote_counts.items() if v == 3])

    # Two-way candidates (2 votes) for manual review
    two_way = sorted([f for f, v in vote_counts.items() if v == 2])

    # One-way (1 vote) - usually dropped
    one_way = sorted([f for f, v in vote_counts.items() if v == 1])

    # Zero votes - dropped
    zero_votes = sorted([f for f, v in vote_counts.items() if v == 0])

    report_lines.append(f"\nThree-way intersection (core, 3/3): {len(final_core)} features")
    for f in final_core:
        report_lines.append(f"  + {f}")

    report_lines.append(f"\nTwo-way candidates (borderline, 2/3): {len(two_way)} features")
    for f in two_way:
        flags = []
        if f in stage3_pass:
            flags.append("Boruta")
        if f in stage4_pass:
            flags.append("Stability")
        if f in stage5_pass:
            flags.append("Permutation")
        report_lines.append(f"  ~ {f} [passed: {', '.join(flags)}]")

    report_lines.append(f"\nDropped (1/3 or 0/3): {len(one_way) + len(zero_votes)} features")
    for f in one_way:
        flags = []
        if f in stage3_pass:
            flags.append("Boruta")
        if f in stage4_pass:
            flags.append("Stability")
        if f in stage5_pass:
            flags.append("Permutation")
        report_lines.append(f"  - {f} [passed only: {', '.join(flags)}]")
    for f in zero_votes:
        report_lines.append(f"  - {f} [no selection method passed]")

    # Decision: core + two-way candidates (manual review passes all)
    # In automated pipeline, we keep core + two-way features for robustness
    final_features = final_core + two_way

    report_lines.append(f"\n{'=' * 70}")
    report_lines.append(f"FINAL FEATURE SET: {len(final_features)} features")
    report_lines.append(f"  Core (3/3): {len(final_core)}")
    report_lines.append(f"  Borderline (2/3): {len(two_way)}")
    report_lines.append(f"  Total selected: {len(final_features)}")
    report_lines.append(f"{'=' * 70}")

    # Build stage summary
    stage_summary = {
        "total": n_total,
        "stage1": len(stage1_kept),
        "stage2": len(stage2_kept),
        "stage3_accepted": len(stage3_accepted),
        "stage3_tentative": len(stage3_tentative),
        "stage4": len(stage4_kept),
        "stage5": len(stage5_kept),
        "core": len(final_core),
        "borderline": len(two_way),
        "final": len(final_features),
    }

    # Write report
    report_path = os.path.join(output_dir, "feature_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nFeature selection report saved to: {report_path}")

    return final_features, report_lines, stage_summary


# ============================================================
# PART 3: MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("TITANIC V4 — FEATURE ENGINEERING + SELECTION PIPELINE")
    print("=" * 70)

    output_dir = BASE_DIR

    # ---- Load data ----
    print("\n[1/5] Loading data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    print(f"  Train: {train_raw.shape}")
    print(f"  Test: {test_raw.shape}")

    y_train = train_raw["Survived"].copy()
    train_raw = train_raw.drop(columns=["Survived"])

    # ---- Engineer features (train) ----
    print("\n[2/5] Engineering features (train)...")
    X_train_full, fitted_encoders, onehot_group_map, scaler = engineer_features(
        train_raw, is_train=True, y=y_train
    )
    print(f"  Candidate features: {X_train_full.shape[1]}")

    # ---- Feature selection ----
    print("\n[3/5] Running feature selection...")
    selected_features, report_lines, stage_summary = feature_selection(
        X_train_full, y_train, onehot_group_map, output_dir
    )

    # ---- Engineer features (test) with fitted encoders ----
    print("\n[4/5] Engineering features (test)...")
    X_test_full, _, _, _ = engineer_features(
        test_raw, is_train=False, fitted_encoders=fitted_encoders, scaler=scaler
    )

    # Ensure same columns as training
    train_cols = fitted_encoders["feature_columns"]
    # Keep only selected features
    selected_train_cols = [c for c in train_cols if c in selected_features]

    # For test, reindex to match training columns (fill missing with 0)
    X_test_aligned = X_test_full.reindex(columns=selected_train_cols, fill_value=0)

    # Add Survived back to train for saving
    X_train_final = X_train_full[selected_train_cols].copy()
    X_train_final["Survived"] = y_train.values

    X_test_final = X_test_aligned.copy()

    # ---- Save outputs ----
    print("\n[5/5] Saving outputs...")

    train_out_path = os.path.join(output_dir, "X_train.csv")
    test_out_path = os.path.join(output_dir, "X_test.csv")

    X_train_final.to_csv(train_out_path, index=False)
    X_test_final.to_csv(test_out_path, index=False)

    # ---- Print summary ----
    s = stage_summary
    print("\n" + "=" * 70)
    print("=== Feature Engineering Summary ===")
    print(f"Total candidate features: {s['total']}")
    print(f"After MI filter:           {s['stage1']} features")
    print(f"After Spearman dedup:      {s['stage2']} features")
    print(f"After Boruta:              {s['stage3_accepted']} accepted, {s['stage3_tentative']} tentative")
    print(f"After Stability Selection: {s['stage4']} features")
    print(f"After Permutation Importance: {s['stage5']} features")
    print(f"Final feature set ({s['final']} features):")
    print(f"  Core (3/3 intersection): {s['core']}")
    print(f"  Borderline (2/3):        {s['borderline']}")
    for i, f in enumerate(selected_features, 1):
        print(f"  {i:2d}. {f}")
    print(f"\n=== Saved ===")
    print(f"X_train: {X_train_final.shape} -> v4/X_train.csv")
    print(f"X_test: {X_test_final.shape} -> v4/X_test.csv")
    print("=" * 70)

    return X_train_final, X_test_final, selected_features


if __name__ == "__main__":
    main()
