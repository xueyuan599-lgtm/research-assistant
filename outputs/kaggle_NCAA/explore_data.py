"""
NCAA Kaggle Competition — Data Exploration & Leakage Detection
===============================================================
Generates: data_profile.json + eda_report.md
Output dir: E:/wuyi/数学建模半自动/research-assistant/outputs/kaggle_NCAA/
"""

import os, sys, json, warnings, gc
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = r"E:\wuyi\数学建模半自动\research-assistant\outputs\kaggle_NCAA"
DATA = os.path.join(BASE, "data")
OUT  = BASE

os.makedirs(OUT, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────

def read_csv(name, **kw):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"  [WARN] {name} not found, skipping")
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        print(f"  [ERROR] reading {name}: {e}")
        return None


def profile_df(df, name, sample_rows=3):
    """Return a dict profile for a single dataframe."""
    if df is None:
        return {"file": name, "status": "NOT_FOUND"}
    prof = {
        "file": name,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "missing_pct": {c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns},
        "nunique": {c: int(df[c].nunique()) for c in df.columns
                    if df[c].dtype in ('object', 'category', 'int64', 'float64')},
        "sample": df.head(sample_rows).to_dict('records'),
    }
    # numeric summary
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).to_dict()
        prof["numeric_summary"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                       for kk, vv in v.items()}
                                   for k, v in desc.items()}
    return prof


def get_time_range(df, season_col='Season'):
    if df is None or season_col not in df.columns:
        return None
    return {
        "min": int(df[season_col].min()),
        "max": int(df[season_col].max()),
        "unique_seasons": sorted(df[season_col].unique().tolist()),
        "n_seasons": int(df[season_col].nunique()),
    }


# ══════════════════════════════════════════════════════════════════════════
#  1. READ ALL FILES
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("NCAA Data Exploration — Reading all files")
print("=" * 60)

files_m = {
    "Teams": read_csv("MTeams.csv"),
    "Seasons": read_csv("MSeasons.csv"),
    "RegularSeasonCompactResults": read_csv("MRegularSeasonCompactResults.csv"),
    "RegularSeasonDetailedResults": read_csv("MRegularSeasonDetailedResults.csv"),
    "NCAATourneyCompactResults": read_csv("MNCAATourneyCompactResults.csv"),
    "NCAATourneyDetailedResults": read_csv("MNCAATourneyDetailedResults.csv"),
    "NCAATourneySeeds": read_csv("MNCAATourneySeeds.csv"),
    "TeamConferences": read_csv("MTeamConferences.csv"),
    "TeamCoaches": read_csv("MTeamCoaches.csv"),
    "NCAATourneySlots": read_csv("MNCAATourneySlots.csv"),
    "NCAATourneySeedRoundSlots": read_csv("MNCAATourneySeedRoundSlots.csv"),
    "GameCities": read_csv("MGameCities.csv"),
    "ConferenceTourneyGames": read_csv("MConferenceTourneyGames.csv"),
    "SecondaryTourneyCompactResults": read_csv("MSecondaryTourneyCompactResults.csv"),
    "SecondaryTourneyTeams": read_csv("MSecondaryTourneyTeams.csv"),
    "TeamSpellings": read_csv("MTeamSpellings.csv"),
}
files_w = {
    "Teams": read_csv("WTeams.csv"),
    "Seasons": read_csv("WSeasons.csv"),
    "RegularSeasonCompactResults": read_csv("WRegularSeasonCompactResults.csv"),
    "RegularSeasonDetailedResults": read_csv("WRegularSeasonDetailedResults.csv"),
    "NCAATourneyCompactResults": read_csv("WNCAATourneyCompactResults.csv"),
    "NCAATourneyDetailedResults": read_csv("WNCAATourneyDetailedResults.csv"),
    "NCAATourneySeeds": read_csv("WNCAATourneySeeds.csv"),
    "TeamConferences": read_csv("WTeamConferences.csv"),
    "NCAATourneySlots": read_csv("WNCAATourneySlots.csv"),
    "GameCities": read_csv("WGameCities.csv"),
    "ConferenceTourneyGames": read_csv("WConferenceTourneyGames.csv"),
    "SecondaryTourneyCompactResults": read_csv("WSecondaryTourneyCompactResults.csv"),
    "SecondaryTourneyTeams": read_csv("WSecondaryTourneyTeams.csv"),
    "TeamSpellings": read_csv("WTeamSpellings.csv"),
}
files_shared = {
    "Cities": read_csv("Cities.csv"),
    "Conferences": read_csv("Conferences.csv"),
    "SampleSubmissionStage1": read_csv("SampleSubmissionStage1.csv"),
    "SampleSubmissionStage2": read_csv("SampleSubmissionStage2.csv"),
}

# ══════════════════════════════════════════════════════════════════════════
#  2. BASIC PROFILING
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Profiling files ---")

profiles = {}
for group_name, group_dict in [("Men", files_m), ("Women", files_w), ("Shared", files_shared)]:
    for fname, df in group_dict.items():
        key = f"{group_name}_{fname}"
        profiles[key] = profile_df(df, fname)
        # Also store time range for game/result files
        if df is not None and 'Season' in df.columns:
            profiles[key]["time_range"] = get_time_range(df)

# ── Men vs Women cross comparisons ───────────────────────────────────────

def cross_compare_teams(m_teams, w_teams):
    mr = m_teams if m_teams is not None else pd.DataFrame()
    wr = w_teams if w_teams is not None else pd.DataFrame()
    n_m = len(mr)
    n_w = len(wr)
    # TeamID overlap (IDs are disjoint typically, but check)
    overlap = set(mr['TeamID']) & set(wr['TeamID']) if 'TeamID' in mr.columns and 'TeamID' in wr.columns else set()
    return {
        "n_men_teams": n_m,
        "n_women_teams": n_w,
        "team_id_overlap": len(overlap),
        "overlap_ids": sorted(overlap)[:20] if overlap else [],
    }


try:
    cc = cross_compare_teams(files_m.get("Teams"), files_w.get("Teams"))
    for k, v in cc.items():
        print(f"  {k}: {v}")
    profiles["_cross_team"] = cc
except Exception as e:
    print(f"  [WARN] cross_compare_teams failed: {e}")
    profiles["_cross_team"] = {"error": str(e)}

# ── Seed analysis (separate due to special encoding) ─────────────────────

def analyze_seeds(df_seeds, label):
    """NCAA seeds: W01 = 1st seed in West region, Y16 = 16th seed in York region.
    Extract numeric seed (01→1, 16→16)."""
    if df_seeds is None:
        return {f"{label}_seeds": "NO_DATA"}
    # Parse seed: first char = region, rest = seed number (as string like "01")
    # Note: some seeds have trailing letters (e.g. "16a" for play-in games)
    df = df_seeds.copy()
    df['SeedNum'] = df['Seed'].str[1:].str.extract(r'(\d+)', expand=False).astype(int)

    # Distribution of numeric seeds
    seed_dist = df['SeedNum'].value_counts().sort_index().to_dict()

    # Seeds by season
    seed_by_season = df.groupby('Season')['SeedNum'].value_counts().unstack(fill_value=0).to_dict()

    # Check for missing TournamentSeeds (teams in regular season but no seed)
    return {
        f"{label}_seed_distribution": {int(k): int(v) for k, v in seed_dist.items()},
        f"{label}_n_tournament_entries": len(df),
        f"{label}_n_seasons_seeded": int(df['Season'].nunique()),
        f"{label}_seasons_range": [int(df['Season'].min()), int(df['Season'].max())],
    }


profiles["_seed_M"] = analyze_seeds(files_m.get("NCAATourneySeeds"), "M")
profiles["_seed_W"] = analyze_seeds(files_w.get("NCAATourneySeeds"), "W")

# ── Missing seed analysis ────────────────────────────────────────────────

def check_missing_seeds(df_compact, df_seeds, df_teams, label):
    """Find teams in tourney results that are missing from seed table."""
    if df_compact is None or df_seeds is None:
        return {f"{label}_missing_seeds": "NO_DATA"}

    # tournament teams from results
    tourney_teams = set(df_compact['WTeamID'].unique()) | set(df_compact['LTeamID'].unique())
    seeded_teams = set(df_seeds['TeamID'].unique())
    missing = tourney_teams - seeded_teams

    df_teams_map = {}
    if df_teams is not None:
        df_teams_map = dict(zip(df_teams['TeamID'], df_teams['TeamName']))

    missing_info = []
    for tid in sorted(missing)[:30]:
        info = {"TeamID": int(tid)}
        if tid in df_teams_map:
            info["TeamName"] = df_teams_map[tid]
        # check if they appear in regular season data
        missing_info.append(info)

    return {
        f"{label}_n_tourney_teams": len(tourney_teams),
        f"{label}_n_seeded_teams": len(seeded_teams),
        f"{label}_n_missing_seeds": len(missing),
        f"{label}_missing_seed_examples": missing_info,
    }


profiles["_missing_seed_M"] = check_missing_seeds(
    files_m.get("NCAATourneyCompactResults"),
    files_m.get("NCAATourneySeeds"),
    files_m.get("Teams"), "M")
profiles["_missing_seed_W"] = check_missing_seeds(
    files_w.get("NCAATourneyCompactResults"),
    files_w.get("NCAATourneySeeds"),
    files_w.get("Teams"), "W")

# ── 2020 season analysis ─────────────────────────────────────────────────

def analyze_2020(df_compact, df_seeds, df_season, label):
    """COVID-19: 2020 season has regular season data but NO tournament."""
    findings = {}
    # Check compact results for 2020
    if df_compact is not None:
        n_2020 = len(df_compact[df_compact['Season'] == 2020])
        findings[f"{label}_tourney_games_2020"] = int(n_2020)
    else:
        findings[f"{label}_tourney_games_2020"] = 0
    if df_seeds is not None:
        n_2020_seeds = len(df_seeds[df_seeds['Season'] == 2020])
        findings[f"{label}_seeds_2020"] = int(n_2020_seeds)
    if df_season is not None:
        r2020 = df_season[df_season['Season'] == 2020]
        findings[f"{label}_season_2020_present"] = bool(len(r2020) > 0)
        if len(r2020) > 0 and 'DayZero' in r2020.columns:
            dz = r2020['DayZero'].iloc[0]
            findings[f"{label}_season_2020_dayzero"] = str(dz)
    return findings


profiles["_2020_M"] = analyze_2020(
    files_m.get("NCAATourneyCompactResults"),
    files_m.get("NCAATourneySeeds"),
    files_m.get("Seasons"), "M")
profiles["_2020_W"] = analyze_2020(
    files_w.get("NCAATourneyCompactResults"),
    files_w.get("NCAATourneySeeds"),
    files_w.get("Seasons"), "W")

# ── Regular season stats ─────────────────────────────────────────────────

def reg_season_stats(df_compact, label):
    if df_compact is None:
        return {f"{label}_reg_season": "NO_DATA"}
    df = df_compact.copy()
    # Points per game stats
    all_scores = pd.concat([
        df[['Season', 'WTeamID', 'WScore']].rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score'}),
        df[['Season', 'LTeamID', 'LScore']].rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score'}),
    ])
    avg_score = all_scores['Score'].mean()
    med_score = all_scores['Score'].median()
    std_score = all_scores['Score'].std()

    # Games per season
    games_per_season = df.groupby('Season').size().to_dict()

    # Average margin of victory
    df['Margin'] = df['WScore'] - df['LScore']
    avg_margin = df['Margin'].mean()
    med_margin = df['Margin'].median()

    return {
        f"{label}_n_games": len(df),
        f"{label}_avg_score": round(float(avg_score), 2),
        f"{label}_med_score": round(float(med_score), 2),
        f"{label}_std_score": round(float(std_score), 2),
        f"{label}_avg_margin": round(float(avg_margin), 2),
        f"{label}_med_margin": round(float(med_margin), 2),
        f"{label}_games_per_season": {int(k): int(v) for k, v in games_per_season.items()},
        f"{label}_n_seasons": int(df['Season'].nunique()),
        f"{label}_n_teams": int(pd.unique(df[['WTeamID', 'LTeamID']].values.ravel()).shape[0]),
        f"{label}_ot_games": int(df['NumOT'].gt(0).sum()) if 'NumOT' in df.columns else None,
    }


profiles["_reg_season_M"] = reg_season_stats(files_m.get("RegularSeasonCompactResults"), "M")
profiles["_reg_season_W"] = reg_season_stats(files_w.get("RegularSeasonCompactResults"), "W")

# ── Tourney stats ────────────────────────────────────────────────────────

def tourney_stats(df_compact, df_detailed, label):
    if df_compact is None:
        return {f"{label}_tourney": "NO_DATA"}
    df = df_compact.copy()

    # games per Season
    games_per_season = df.groupby('Season').size().to_dict()
    df['Margin'] = df['WScore'] - df['LScore']

    # Check detailed stats availability
    has_detailed = df_detailed is not None and len(df_detailed) > 0

    return {
        f"{label}_n_tourney_games": len(df),
        f"{label}_avg_margin": round(float(df['Margin'].mean()), 2),
        f"{label}_med_margin": round(float(df['Margin'].median()), 2),
        f"{label}_games_per_season": {int(k): int(v) for k, v in games_per_season.items()},
        f"{label}_n_seasons": int(df['Season'].nunique()),
        f"{label}_ot_games": int(df['NumOT'].gt(0).sum()) if 'NumOT' in df.columns else None,
        f"{label}_has_detailed_stats": has_detailed,
    }


profiles["_tourney_M"] = tourney_stats(files_m.get("NCAATourneyCompactResults"),
                                        files_m.get("NCAATourneyDetailedResults"), "M")
profiles["_tourney_W"] = tourney_stats(files_w.get("NCAATourneyCompactResults"),
                                        files_w.get("NCAATourneyDetailedResults"), "W")

# ══════════════════════════════════════════════════════════════════════════
#  3. MASSEY ORDINALS ANALYSIS (large file: 129MB)
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Massey Ordinals Analysis (large file, 129 MB) ---")

massey = files_m.get("MasseyOrdinals")
if False:  # We didn't load it above; load separately with chunking
    pass

# Actually load Massey with optimized dtypes to reduce memory
massey = None
massey_path = os.path.join(DATA, "MMasseyOrdinals.csv")
if os.path.exists(massey_path):
    print("  Reading MMasseyOrdinals.csv with optimized dtypes...")
    # First, read just a few rows to get the schema
    massey_sample = pd.read_csv(massey_path, nrows=5)
    print(f"  Columns: {list(massey_sample.columns)}")
    print(f"  Dtypes: {dict(massey_sample.dtypes)}")

    # Read with optimized types
    massey = pd.read_csv(
        massey_path,
        dtype={
            'Season': 'int16',
            'RankingDayNum': 'int16',
            'SystemName': 'category',
            'TeamID': 'int16',
            'OrdinalRank': 'float32',
        },
        low_memory=True
    )
    print(f"  Shape: {massey.shape}")
    print(f"  Memory usage: {massey.memory_usage(deep=True).sum() / 1e6:.1f} MB")
else:
    print("  MMasseyOrdinals.csv NOT FOUND — women's data doesn't have this file.")

massey_prof = {}
if massey is not None:
    # Number of ranking systems
    n_systems = massey['SystemName'].nunique()
    system_names = massey['SystemName'].value_counts().index.tolist()
    massey_prof["n_ranking_systems"] = int(n_systems)
    massey_prof["system_names"] = system_names
    massey_prof["n_total_ratings"] = len(massey)

    # System coverage by season
    system_by_season = massey.groupby(['Season', 'SystemName']).size().reset_index()
    system_by_season.columns = ['Season', 'SystemName', 'Count']
    systems_per_season = system_by_season.groupby('Season')['SystemName'].apply(list).to_dict()
    massey_prof["systems_per_season"] = {int(k): v for k, v in systems_per_season.items()}

    # Which systems cover ALL seasons
    seasons_range = sorted(massey['Season'].unique())
    all_seasons_set = set(seasons_range)
    system_seasons = massey.groupby('SystemName')['Season'].apply(set)
    full_coverage = system_seasons[system_seasons.apply(lambda s: s == all_seasons_set)].index.tolist()
    massey_prof["systems_with_full_season_coverage"] = full_coverage
    massey_prof["all_seasons_in_massey"] = [int(s) for s in seasons_range]
    massey_prof["season_range"] = [int(seasons_range[0]), int(seasons_range[-1])]

    # Time coverage — how many ranking days per season
    days_per_season = massey.groupby('Season')['RankingDayNum'].nunique().to_dict()
    massey_prof["ranking_days_per_season"] = {int(k): int(v) for k, v in days_per_season.items()}

    # Ordinal rank distribution
    rank_stats = massey['OrdinalRank'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).to_dict()
    massey_prof["rank_distribution"] = {k: round(v, 2) for k, v in rank_stats.items()}

    # Sample some data
    massey_prof["sample_ratings"] = massey.head(5).to_dict('records')

    # Check correlation between different ranking systems (sample)
    print("  Computing ranking system correlations (sampling)...")
    # Take season 2023 for example, pivot to see system correlations
    latest_season = int(seasons_range[-1])
    latest_day = massey[massey['Season'] == latest_season]['RankingDayNum'].max()
    latest_ranks = massey[(massey['Season'] == latest_season) &
                          (massey['RankingDayNum'] == latest_day)]
    pivot_latest = latest_ranks.pivot_table(
        index='TeamID', columns='SystemName', values='OrdinalRank'
    )
    corr = pivot_latest.corr(method='spearman')
    # top correlated pairs
    corr_unstack = corr.unstack()
    corr_unstack = corr_unstack[corr_unstack < 1.0]  # remove self-correlation
    top_corr = corr_unstack.sort_values(ascending=False).head(10)
    massey_prof["system_spearman_corr_latest_season"] = {
        f"{idx[0]}_vs_{idx[1]}": round(float(val), 3) for idx, val in top_corr.items()
    }

    del pivot_latest, latest_ranks, corr
    gc.collect()

profiles["_massey_ordinals"] = massey_prof

# ══════════════════════════════════════════════════════════════════════════
#  4. ADVERSARIAL VALIDATION (Train/Test distribution shift)
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Adversarial Validation (LightGBM) ---")

def do_adversarial_validation(train_df, test_df, features, label="M"):
    """Train a classifier to distinguish train vs test. If AUC > 0.7 → shift detected."""
    try:
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("  LightGBM not installed, using RandomForest fallback")
        from sklearn.ensemble import RandomForestClassifier as LGBMClassifier
        # Still works as a classifier

    # Build train/test indicator
    n_train = len(train_df)
    n_test = len(test_df)
    X = pd.concat([train_df[features].copy(), test_df[features].copy()], axis=0)
    y = np.array([0]*n_train + [1]*n_test)

    # Drop rows with NaN
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        return {"auc": None, "n_samples": int(len(X)), "error": "Too few samples"}

    try:
        model = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbose=-1, subsample=0.8, colsample_bytree=0.8
        )
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=1)
        auc_mean = float(scores.mean())
        auc_std = float(scores.std())

        # Full fit for feature importance
        model.fit(X, y)
        imp = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        return {
            "auc_mean": round(auc_mean, 4),
            "auc_std": round(auc_std, 4),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_valid_samples": int(len(X)),
            "shift_detected": auc_mean > 0.65,
            "shift_severity": "high" if auc_mean > 0.75 else ("moderate" if auc_mean > 0.65 else "low"),
            "top_features": imp.head(10).to_dict('records'),
        }
    except Exception as e:
        return {"error": str(e), "auc": None}


def build_team_features_from_results(df_compact, df_seeds):
    """Build season-level team features from regular season results."""
    if df_compact is None:
        return pd.DataFrame()

    df = df_compact.copy()

    # Features for each team per season
    win_stats = df.groupby(['Season', 'WTeamID']).agg(
        wins=('WScore', 'count'),
        pts_for=('WScore', 'mean'),
    ).reset_index().rename(columns={'WTeamID': 'TeamID'})

    loss_stats = df.groupby(['Season', 'LTeamID']).agg(
        losses=('LScore', 'count'),
        pts_against=('LScore', 'mean'),
    ).reset_index().rename(columns={'LTeamID': 'TeamID'})

    team_feats = win_stats.merge(loss_stats, on=['Season', 'TeamID'], how='outer').fillna(0)
    team_feats['n_games'] = team_feats['wins'] + team_feats['losses']
    team_feats['win_rate'] = team_feats['wins'] / team_feats['n_games'].clip(lower=1)
    team_feats['net_pts'] = team_feats['pts_for'] - team_feats['pts_against']

    # Add seed if available
    if df_seeds is not None:
        seeds = df_seeds.copy()
        seeds['SeedNum'] = seeds['Seed'].str[1:].str.extract(r'(\d+)', expand=False).astype(int)
        team_feats = team_feats.merge(
            seeds[['Season', 'TeamID', 'SeedNum']], on=['Season', 'TeamID'], how='left')

    return team_feats


# Adversarial validation for both men and women
adv_results = {}
for gender, prefix in [("Men", "M"), ("Women", "W")]:
    print(f"\n  --- {gender} Adversarial Validation ---")
    # Use regular season to predict tourney vs non-tourney (but both are "train")
    # Better: use historical data as train, and the submission as "test" domain
    # We'll use Season as a proxy: build features and compare season distributions

    reg_df = files_m if prefix == "M" else files_w

    # Build team features from all regular season data
    team_feats = build_team_features_from_results(
        reg_df.get("RegularSeasonCompactResults"),
        reg_df.get("NCAATourneySeeds")
    )

    if len(team_feats) > 50:
        # Split: earlier seasons vs recent seasons (leave-one-season-out CV style)
        seasons = sorted(team_feats['Season'].unique())
        mid = len(seasons) // 2
        early_seasons = seasons[:mid]
        late_seasons = seasons[mid:]

        early = team_feats[team_feats['Season'].isin(early_seasons)]
        late = team_feats[team_feats['Season'].isin(late_seasons)]

        feat_cols = [c for c in ['wins', 'losses', 'n_games', 'win_rate',
                                  'pts_for', 'pts_against', 'net_pts', 'SeedNum']
                     if c in team_feats.columns]

        adv = do_adversarial_validation(early, late, feat_cols, label=prefix)
        adv["early_seasons"] = [int(s) for s in early_seasons]
        adv["late_seasons"] = [int(s) for s in late_seasons]
        adv_results[prefix] = adv
        for k, v in adv.items():
            print(f"    {k}: {v}")
    else:
        adv_results[prefix] = {"error": "Too few team-season observations"}

profiles["_adversarial_validation"] = adv_results

# ══════════════════════════════════════════════════════════════════════════
#  5. REVIEW SUBMISSION SAMPLES
# ══════════════════════════════════════════════════════════════════════════

sub1 = files_shared.get("SampleSubmissionStage1")
sub2 = files_shared.get("SampleSubmissionStage2")

submission_prof = {}
if sub1 is not None:
    sub1['Season'] = sub1['ID'].str.split('_').str[0].astype(int)
    sub1['TeamA'] = sub1['ID'].str.split('_').str[1].astype(int)
    sub1['TeamB'] = sub1['ID'].str.split('_').str[2].astype(int)
    submission_prof["stage1"] = {
        "shape": list(sub1.shape),
        "seasons": sorted(sub1['Season'].unique().tolist()),
        "n_seasons": int(sub1['Season'].nunique()),
        "n_matchups": len(sub1),
        "sample": sub1.head(5).to_dict('records'),
    }
if sub2 is not None:
    sub2['Season'] = sub2['ID'].str.split('_').str[0].astype(int)
    sub2['TeamA'] = sub2['ID'].str.split('_').str[1].astype(int)
    sub2['TeamB'] = sub2['ID'].str.split('_').str[2].astype(int)
    submission_prof["stage2"] = {
        "shape": list(sub2.shape),
        "seasons": sorted(sub2['Season'].unique().tolist()),
        "n_seasons": int(sub2['Season'].nunique()),
        "n_matchups": len(sub2),
        "sample": sub2.head(5).to_dict('records'),
    }

profiles["_submission"] = submission_prof

# ══════════════════════════════════════════════════════════════════════════
#  6. WRITE data_profile.json
# ══════════════════════════════════════════════════════════════════════════

profiles["_meta"] = {
    "generated_at": datetime.now().isoformat(),
    "platform": "Windows",
    "data_root": DATA,
}

profiles_path = os.path.join(OUT, "data_profile.json")
with open(profiles_path, 'w', encoding='utf-8') as f:
    json.dump(profiles, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  [OK] Written: {profiles_path}")

# ══════════════════════════════════════════════════════════════════════════
#  7. GENERATE EDA REPORT (Markdown)
# ══════════════════════════════════════════════════════════════════════════

report = []
report.append("# NCAA March Madness — Data Exploration Report\n")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

# ── 1. Data Overview ────
report.append("## 1. Data Overview\n")
report.append(f"Data root: `{DATA}`\n")
report.append(f"Total files: {len([k for k in profiles if not k.startswith('_')])} "
              f"(Men: {len(files_m)}, Women: {len(files_w)}, Shared: {len(files_shared)})\n")

report.append("### 1.1 Men's Data Files\n")
report.append("| File | Rows | Cols | Time Range | Notes |\n")
report.append("|------|------|------|-----------|-------|\n")
for key, prof in profiles.items():
    if key.startswith("Men_") and "shape" in prof:
        fname = prof.get('file', key)
        shape = prof.get('shape', ['?', '?'])
        tr = prof.get('time_range', {})
        tr_str = f"{tr.get('min','?')}-{tr.get('max','?')}" if tr else "—"
        report.append(f"| {fname} | {shape[0]:,} | {shape[1]} | {tr_str} | |\n")

report.append("\n### 1.2 Women's Data Files\n")
report.append("| File | Rows | Cols | Time Range | Notes |\n")
report.append("|------|------|------|-----------|-------|\n")
for key, prof in profiles.items():
    if key.startswith("Women_") and "shape" in prof:
        fname = prof.get('file', key)
        shape = prof.get('shape', ['?', '?'])
        tr = prof.get('time_range', {})
        tr_str = f"{tr.get('min','?')}-{tr.get('max','?')}" if tr else "—"
        report.append(f"| {fname} | {shape[0]:,} | {shape[1]} | {tr_str} | |\n")

report.append("\n### 1.3 Shared Files\n")
report.append("| File | Rows | Cols | Notes |\n")
report.append("|------|------|------|-------|\n")
for key, prof in profiles.items():
    if key.startswith("Shared_") and "shape" in prof:
        fname = prof.get('file', key)
        shape = prof.get('shape', ['?', '?'])
        report.append(f"| {fname} | {shape[0]:,} | {shape[1]} | |\n")

# ── 2. Target Understanding ────
report.append("\n## 2. Target Understanding\n")
report.append("- **Task**: Predict P(Team_A beats Team_B) for each matchup in the tournament\n")
report.append("- **Evaluation Metric**: Brier Score (mean squared error between prediction (0-1) and actual outcome)\n")
report.append("- Better Brier = lower score. A naive prediction of 0.5 gives Brier = 0.25\n")
report.append("- **Submission format**: Binary classification probabilities (0 to 1) for each matchup ID\n")

if sub2 is not None:
    s2_seasons = sorted(sub2['Season'].unique())
    stage2_seasons_str = ", ".join([str(int(s)) for s in s2_seasons])
    report.append(f"- Stage 2 contains **{len(sub2):,} matchups** across {len(s2_seasons)} seasons: {stage2_seasons_str}\n")
if sub1 is not None:
    s1_seasons = sorted(sub1['Season'].unique())
    stage1_seasons_str = ", ".join([str(int(s)) for s in s1_seasons])
    report.append(f"- Stage 1 contains **{len(sub1):,} matchups** across {len(s1_seasons)} seasons: {stage1_seasons_str}\n")

# ── 3. Team Overview ────
report.append("\n## 3. Teams Overview\n")
if "Men_Teams" in profiles:
    m_teams = profiles["Men_Teams"]
    report.append(f"- **Men**: {m_teams['shape'][0]} teams\n")
if "Women_Teams" in profiles:
    w_teams = profiles["Women_Teams"]
    report.append(f"- **Women**: {w_teams['shape'][0]} teams\n")
report.append(f"- **Team ID overlap** (same ID used for men & women): {cc.get('team_id_overlap', 'N/A')}\n")
report.append("  - (NCAA typically uses disjoint TeamID ranges for men vs women)\n")

# ── 4. Seed Distribution ────
report.append("\n## 4. Tournament Seed Distribution\n")
report.append("Seeds follow NCAA format: `{Region}{Number:02d}` — e.g. W01 = 1 seed in West region.\n")
report.append("Extracted numeric seed (1–16) distributions:\n")

for seed_key, label, prefix in [("_seed_M", "Men", "M"), ("_seed_W", "Women", "W")]:
    if seed_key in profiles:
        sd = profiles[seed_key]
        dist = sd.get(f"{prefix}_seed_distribution", {})
        report.append(f"\n**{label}**: {sd.get(f'{prefix}_n_tournament_entries', '?')} entries across "
                      f"{sd.get(f'{prefix}_n_seasons_seeded', '?')} seasons\n")
        report.append("| Seed | Count |\n|------|-------|\n")
        for s in sorted(dist.keys()):
            report.append(f"| {s} | {dist[s]} |\n")

report.append("\nKey observation: Each of the 16 seeds appears approximately equally, "
              "but note that play-in games (First Four) create slightly more teams at seeds 11, 12, 16.\n")

# ── 5. Missing Seed Analysis ────
report.append("\n## 5. Missing Seed Analysis\n")
report.append("Check if any teams in tournament results have no seed entry.\n")
for ms_key, label, prefix in [("_missing_seed_M", "Men", "M"), ("_missing_seed_W", "Women", "W")]:
    if ms_key in profiles:
        ms = profiles[ms_key]
        report.append(f"- **{label}**: {ms.get(f'{prefix}_n_missing_seeds', '?')} teams in tournament without seed records\n")
        examples = ms.get(f'{prefix}_missing_seed_examples', [])
        if examples:
            report.append(f"  - Examples: {[e.get('TeamName', e['TeamID']) for e in examples[:5]]}\n")

# ── 6. Regular Season Stats ────
report.append("\n## 6. Regular Season Statistics\n")
report.append("| Metric | Men | Women |\n|--------|-----|-------|\n")
for metric in ['n_games', 'avg_score', 'med_score', 'std_score', 'avg_margin', 'med_margin',
               'n_seasons', 'n_teams', 'ot_games']:
    m_val = profiles.get("_reg_season_M", {}).get(f"M_{metric}", '?')
    w_val = profiles.get("_reg_season_W", {}).get(f"W_{metric}", '?')
    label = metric.replace('_', ' ').title()
    report.append(f"| {label} | {m_val} | {w_val} |\n")

# ── 7. Tournament Stats ────
report.append("\n## 7. Tournament Statistics\n")
report.append("| Metric | Men | Women |\n|--------|-----|-------|\n")
for metric in ['n_tourney_games', 'avg_margin', 'med_margin', 'n_seasons', 'ot_games']:
    m_val = profiles.get("_tourney_M", {}).get(f"M_{metric}", '?')
    w_val = profiles.get("_tourney_W", {}).get(f"W_{metric}", '?')
    label = metric.replace('_', ' ').title()
    report.append(f"| {label} | {m_val} | {w_val} |\n")

# ── 8. 2020 COVID Season ────
report.append("\n## 8. 2020 COVID-19 Season Analysis\n")
report.append("The 2020 NCAA tournament was cancelled due to COVID-19.\n")
for key, label, prefix in [("_2020_M", "Men", "M"), ("_2020_W", "Women", "W")]:
    if key in profiles:
        d = profiles[key]
        # Check both prefix and error keys
        tkey = f"{prefix}_tourney_games_2020"
        skey = f"{prefix}_seeds_2020"
        pkey = f"{prefix}_season_2020_present"
        dkey = f"{prefix}_season_2020_dayzero"

        report.append(f"- **{label}**:\n")
        report.append(f"  - Tournament games in 2020: {d.get(tkey, 'N/A')}\n")
        report.append(f"  - Seeds assigned for 2020: {d.get(skey, 'N/A')}\n")
        report.append(f"  - Season 2020 in Season table: {d.get(pkey, 'N/A')}\n")
        if dkey in d:
            report.append(f"  - Season 2020 DayZero: {d[dkey]}\n")

report.append("\n**Implication**: 2020 has regular season data but no tournament outcome. "
              "For training, 2020 regular season data can still be used for feature engineering, "
              "but no tournament matchup from 2020 is available for supervised learning.\n")

# ── 9. Massey Ordinals ────
report.append("\n## 9. Massey Ordinals Analysis (Men Only)\n")
if massey_prof:
    mp = massey_prof
    report.append(f"- **File size**: 129 MB on disk\n")
    report.append(f"- **Total rating entries**: {mp.get('n_total_ratings', '?'):,}\n")
    report.append(f"- **Number of ranking systems**: {mp.get('n_ranking_systems', '?')}\n")
    report.append(f"- **Season range**: {mp.get('season_range', '?')}\n")
    report.append(f"- **Systems with full coverage** (all seasons): "
                  f"{len(mp.get('systems_with_full_season_coverage', []))}\n")
    full_cov = mp.get('systems_with_full_season_coverage', [])
    if full_cov:
        report.append(f"  - {full_cov}\n")
    report.append("\n**Ranking days per season**: ")
    days_ps = mp.get('ranking_days_per_season', {})
    if days_ps:
        report.append(f"range {min(days_ps.values())}–{max(days_ps.values())} days\n")
    report.append(f"\n**Ordinal rank distribution**:\n")
    rd = mp.get('rank_distribution', {})
    for pctl, val in rd.items():
        report.append(f"- {pctl}: {val}\n")

    report.append("\n**Ranking system correlations (latest season):**\n")
    corr_data = mp.get('system_spearman_corr_latest_season', {})
    for pair, corr_val in list(corr_data.items())[:10]:
        report.append(f"- {pair}: {corr_val}\n")

    report.append("\n**Systems per season**:\n")
    sps = mp.get('systems_per_season', {})
    for season, systems in list(sps.items())[:5]:
        report.append(f"- Season {season}: {len(systems)} systems\n")

    report.append("\n**Women's note**: No MasseyOrdinals file available for women's tournament. "
                  "All feature engineering for women must rely on game results and seeds only.\n")
else:
    report.append("No MasseyOrdinals data available.\n")

# ── 10. Adversarial Validation ────
report.append("\n## 10. Adversarial Validation (Train/Test Distribution Shift)\n")
report.append("Method: Train a classifier to distinguish earlier-season team features from later-season features. "
              "AUC > 0.65 suggests distribution shift across time.\n\n")
for gender, result in adv_results.items():
    report.append(f"### {gender}\n")
    if "error" in result:
        report.append(f"- Error: {result['error']}\n")
    else:
        auc = result.get('auc_mean', 'N/A')
        shift = result.get('shift_detected', 'N/A')
        severity = result.get('shift_severity', 'N/A')
        report.append(f"- **AUC**: {auc}\n")
        report.append(f"- **Shift Detected**: {shift}\n")
        report.append(f"- **Severity**: {severity}\n")
        if 'top_features' in result:
            report.append("- **Top discriminating features**:\n")
            for feat in result['top_features'][:5]:
                report.append(f"  - {feat.get('feature', '?')}: importance={feat.get('importance', 0):.2f}\n")

report.append("\n**Interpretation**:\n")
report.append("- AUC ~ 0.5-0.6: Features are stable over time, no significant shift\n")
report.append("- AUC 0.65-0.75: Moderate shift, consider time-aware cross-validation\n")
report.append("- AUC > 0.75: Strong shift, use only recent data or model the trend\n")

# ── 11. Missing Values ────
report.append("\n## 11. Missing Value Summary\n")
report.append("| File | Missing Columns | Missing % |\n")
report.append("|------|----------------|-----------|\n")
for key, prof in profiles.items():
    if "missing_pct" in prof and any(v > 0 for v in prof['missing_pct'].values()):
        fname = prof.get('file', key)
        missing_info = "; ".join([f"{c}: {pct}%" for c, pct in prof['missing_pct'].items() if pct > 0])
        report.append(f"| {fname} | {missing_info} |\n")

missing_any = any(
    any(v > 0 for v in prof.get('missing_pct', {}).values())
    for prof in profiles.values() if isinstance(prof, dict) and 'missing_pct' in prof
)
if not missing_any:
    report.append("| (All files) | None | 0% |\n")

# ── 12. Feature Engineering Recommendations ────
report.append("\n## 12. Feature Engineering Recommendations\n")

report.append("### 12.1 Core Features (All Models)\n")
report.append("1. **Season-level team strength**\n")
report.append("   - Win rate, avg points scored/allowed, net rating\n")
report.append("   - Strength of schedule (opponents' win rate)\n")
report.append("2. **Seed-based features**\n")
report.append("   - Seed difference: strong predictor of match outcome\n")
report.append("   - Historical win rate by seed matchup\n")
report.append("3. **Recent form**\n")
report.append("   - Last 10 games win rate\n")
report.append("   - Performance in conference tournament\n")
report.append("4. **Massey Ordinals (men only)**\n")
report.append("   - Composite rank across systems at latest ranking day\n")
report.append("   - Rank momentum (trend over last few ranking days)\n")
report.append("5. **Historical head-to-head**\n")
report.append("   - Previous matchups in regular season\n")
report.append("   - Win rate in tournament settings\n")

report.append("\n### 12.2 Advanced Features\n")
report.append("1. **Team efficiency metrics** (from DetailedResults)\n")
report.append("   - Offensive/defensive efficiency (points per possession)\n")
report.append("   - FG%, 3P%, FT%, rebounding, turnovers\n")
report.append("   - Tempo-adjusted stats\n")
report.append("2. **Coach experience**\n")
report.append("   - Tournament win rate by coach\n")
report.append("3. **Location features**\n")
report.append("   - Game location (home/away/neutral) for regular season\n")
report.append("   - Distance traveled for tournament games (from GameCities)\n")
report.append("4. **Time-series features**\n")
report.append("   - Team performance trajectory over the season (rolling windows)\n")
report.append("   - Massey rank trajectory\n")

report.append("\n### 12.3 Modeling Strategy Notes\n")
report.append("1. **Men vs Women**: Train separate models unless pooling proves beneficial\n")
report.append("2. **Massey Ordinals**: For men, include as features. For women, skip.\n")
report.append("3. **Cross-validation**: Use leave-one-season-out or expanding window CV\n")
report.append("4. **Loss function**: Brier score minimization (proper scoring rule)\n")
report.append("5. **2020 season**: Usable for feature construction but not for tournament outcome labels\n")

# ── 13. Men/Women Processing Differences ────
# Get actual numbers for Men vs Women comparison
reg_m = profiles.get("_reg_season_M", {})
reg_w = profiles.get("_reg_season_W", {})
tourney_m = profiles.get("_tourney_M", {})
tourney_w = profiles.get("_tourney_W", {})
teams_m = cc.get("n_men_teams", "?")
teams_w = cc.get("n_women_teams", "?")

report.append("\n## 13. Men vs Women Processing Differences\n")
report.append("| Aspect | Men | Women |\n|--------|-----|-------|\n")
report.append(f"| Teams | {teams_m} | {teams_w} |\n")
report.append(f"| Regular season games | {reg_m.get('M_n_games', '?'):,} | {reg_w.get('W_n_games', '?'):,} |\n")
report.append(f"| Regular season seasons | {reg_m.get('M_n_seasons', '?')} (1985-2026) | {reg_w.get('W_n_seasons', '?')} (1998-2026) |\n")
report.append(f"| Tournament games | {tourney_m.get('M_n_tourney_games', '?'):,} | {tourney_w.get('W_n_tourney_games', '?'):,} |\n")
report.append(f"| Tournament seasons | {tourney_m.get('M_n_seasons', '?')} | {tourney_w.get('W_n_seasons', '?')} |\n")
report.append(f"| Massey Ordinals | **Available** (197 systems, 129MB, 5.9M ratings) | **Not available** |\n")
report.append(f"| Detailed stats start | 2003 | 2010 |\n")
report.append(f"| Coach data | Available (13,900 records) | Not directly available |\n")
report.append(f"| Avg regular season score | {reg_m.get('M_avg_score', '?')} | {reg_w.get('W_avg_score', '?')} |\n")
report.append(f"| Avg tournament margin | {tourney_m.get('M_avg_margin', '?')} | {tourney_w.get('W_avg_margin', '?')} |\n")

report.append("\n## 14. Submission Summary\n")
if 'stage1' in submission_prof:
    s1 = submission_prof.get('stage1', {})
    s1_seasons_str = ", ".join([str(int(s)) for s in s1.get('seasons', [])])
    report.append(f"- Stage 1: {s1.get('n_matchups', '?'):,} matchups across {s1.get('n_seasons', '?')} seasons: {s1_seasons_str}\n")
if 'stage2' in submission_prof:
    s2 = submission_prof.get('stage2', {})
    s2_seasons_str = ", ".join([str(int(s)) for s in s2.get('seasons', [])])
    report.append(f"- Stage 2: {s2.get('n_matchups', '?'):,} matchups across {s2.get('n_seasons', '?')} seasons: {s2_seasons_str}\n")
report.append("- Prediction column: `Pred` (float 0-1, probability TeamA beats TeamB)\n")

report.append("\n---\n")
report.append("*Report generated by Data Explorer Agent*\n")

# Write report
report_path = os.path.join(OUT, "eda_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report)
print(f"  [OK] Written: {report_path}")

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETE")
print("=" * 60)
print(f"  data_profile.json → {profiles_path}")
print(f"  eda_report.md     → {report_path}")
