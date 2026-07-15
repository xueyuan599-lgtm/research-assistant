#!/usr/bin/env python
"""
solution_a.py — GBDT Ensemble Approach for Kaggle NCAA
CatBoost + LightGBM + Optuna Tuning + Blending

Architecture:
  1. Data loading (compact + detailed + seeds + Massey)
  2. Elo rating system (chronological)
  3. Team-season features (rolling windows, efficiency, Massey aggregations)
  4. Dual-sample training data construction
  5. Expanding window cross-validation
  6. CatBoost + LightGBM with Optuna hyperparameter tuning
  7. Blending ensemble (weighted by 1/Brier)
  8. Pseudo-labeling (optional)
  9. Final submission generation

Usage:
  python solution_a.py
"""

import os, sys, gc, warnings, time, json, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

warnings.filterwarnings('ignore')
np.random.seed(42)

try:
    import optuna
except ImportError:
    optuna = None

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = "E:/wuyi/数学建模半自动/research-assistant/outputs/kaggle_NCAA/data"
OUTPUT_DIR = "E:/wuyi/数学建模半自动/research-assistant/outputs/kaggle_NCAA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

M_START = 2003    # Men: detailed stats available from this year
W_START = 2010    # Women: detailed stats available from this year
MASSEY_SYSTEMS = ['AP', 'COL', 'DOL', 'MOR', 'POM', 'USA', 'WLK']

N_OPTUNA_TRIALS = 50
N_FOLDS = 5

# Save experiment log
LOG = []


# ─── Helpers ─────────────────────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    LOG.append(msg)


def parse_seed(seed_str):
    """Convert seed string like 'W01', 'X16a' to numeric 1-16."""
    digits = ''.join(c for c in str(seed_str) if c.isdigit())
    try:
        return int(digits)
    except:
        return 16


def brier(y_true, y_pred):
    from sklearn.metrics import brier_score_loss
    return brier_score_loss(y_true, y_pred)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_data(gender='M'):
    """Load all data for a given gender."""
    prefix = 'M' if gender == 'M' else 'W'
    base = DATA_DIR
    tables = {}

    log(f"Loading {gender} data...")

    tables['teams'] = pd.read_csv(f"{base}/{prefix}Teams.csv")
    tables['seeds'] = pd.read_csv(f"{base}/{prefix}NCAATourneySeeds.csv")

    # Regular season
    tables['reg_compact'] = pd.read_csv(f"{base}/{prefix}RegularSeasonCompactResults.csv")
    tables['reg_detailed'] = pd.read_csv(f"{base}/{prefix}RegularSeasonDetailedResults.csv")

    # Tournament
    tables['tourney_compact'] = pd.read_csv(f"{base}/{prefix}NCAATourneyCompactResults.csv")
    det_path = f"{base}/{prefix}NCAATourneyDetailedResults.csv"
    tables['tourney_detailed'] = pd.read_csv(det_path) if os.path.exists(det_path) else None

    # Conferences
    tables['conferences'] = pd.read_csv(f"{base}/{prefix}TeamConferences.csv")

    # Gender-specific data
    if gender == 'M':
        log("  Loading Massey ordinals (7 systems)...")
        massey = pd.read_csv(f"{base}/MMasseyOrdinals.csv")
        massey = massey[massey['SystemName'].isin(MASSEY_SYSTEMS)].copy()
        tables['massey'] = massey
    else:
        tables['massey'] = None

    return tables


def load_submission():
    """Load the Stage 2 submission template."""
    log("Loading submission template...")
    sub = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage2.csv")
    return sub


# ═══════════════════════════════════════════════════════════════════════════
# ELO RATING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
def compute_elo(reg_games, tourney_games=None, start_year=2003,
                initial_elo=1500.0, K_reg=20, K_tourney=30,
                home_adv=100, season_regress=0.75):
    """
    Compute Elo ratings chronologically across all games.

    Returns:
      elo_df: DataFrame with columns [Season, TeamID, Elo]
              giving the pre-tournament / end-of-season Elo for each team-season.
    """
    log("  Computing Elo ratings...")

    # Filter by start year
    reg = reg_games[reg_games['Season'] >= start_year].copy()
    if tourney_games is not None:
        tourney = tourney_games[tourney_games['Season'] >= start_year].copy()
        all_games = pd.concat([reg, tourney], ignore_index=True)
    else:
        all_games = reg.copy()

    all_games = all_games.sort_values(['Season', 'DayNum']).reset_index(drop=True)

    current_elo = {}          # team_id -> current elo
    current_season = None
    pre_tourney_elos = []     # list of dicts for output

    # We also track if we're in tournament territory
    season_max_reg_day = {}   # max regular-season day per season

    # First pass: find the max regular season day per season
    reg_season_max = reg.groupby('Season')['DayNum'].max().to_dict()

    for _, game in all_games.iterrows():
        season = int(game['Season'])
        day = int(game['DayNum'])
        wteam = int(game['WTeamID'])
        lteam = int(game['LTeamID'])
        wloc = game['WLoc'] if 'WLoc' in game.index else 'N'

        # Season transition
        if season != current_season:
            if current_season is not None and current_season in reg_season_max:
                # Record pre-tournament Elo for previous season
                for tid, elo in current_elo.items():
                    pre_tourney_elos.append({'Season': current_season,
                                             'TeamID': tid,
                                             'Elo': round(elo, 1)})

            # Regression
            new_elo = {}
            for tid, elo in current_elo.items():
                new_elo[tid] = season_regress * elo + (1 - season_regress) * initial_elo
            current_elo = new_elo
            current_season = season

        # Get current Elo
        elo_w = current_elo.get(wteam, initial_elo)
        elo_l = current_elo.get(lteam, initial_elo)

        # Home court adjustment (for expected score only)
        if wloc == 'H':
            elo_w_adj, elo_l_adj = elo_w + home_adv, elo_l
        elif wloc == 'A':
            elo_w_adj, elo_l_adj = elo_w, elo_l + home_adv
        else:
            elo_w_adj, elo_l_adj = elo_w, elo_l

        # Expected score
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l_adj - elo_w_adj) / 400.0))

        # Determine K
        max_reg_day = reg_season_max.get(season, 132)
        is_tourney = day > max_reg_day
        K = K_tourney if is_tourney else K_reg

        # Update
        current_elo[wteam] = elo_w + K * (1.0 - expected_w)
        current_elo[lteam] = elo_l + K * (0.0 - expected_w)

    # Final season
    if current_season is not None and current_season in reg_season_max:
        for tid, elo in current_elo.items():
            pre_tourney_elos.append({'Season': current_season,
                                     'TeamID': tid,
                                     'Elo': round(elo, 1)})

    elo_df = pd.DataFrame(pre_tourney_elos)
    elo_df = elo_df.drop_duplicates(subset=['Season', 'TeamID']).reset_index(drop=True)
    log(f"    Elo computed: {len(elo_df)} team-season entries")
    return elo_df


# ═══════════════════════════════════════════════════════════════════════════
# TEAM STATS (compact + detailed)
# ═══════════════════════════════════════════════════════════════════════════
def compute_team_features(reg_compact, reg_detailed, start_year, gender='M'):
    """
    Compute per-team-season features from regular season data.

    Features:
      - Points scored/allowed averages (season, last 3/5/10/20 games)
      - Win rate (season, last 3/5/10/20 games)
      - Efficiency metrics (from detailed stats): OE, DE, NE
      - Number of games played

    Returns: DataFrame indexed by (Season, TeamID) with feature columns.
    """
    log(f"  Computing team features ({gender})...")

    # Filter
    reg = reg_compact[reg_compact['Season'] >= start_year].copy()

    # ─── Unify to per-team-per-game format ───
    # Winner side
    w = reg[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore']].copy()
    w.columns = ['Season', 'DayNum', 'TeamID', 'PF', 'OpponentID', 'PA']
    w['Win'] = 1

    # Loser side
    l = reg[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore']].copy()
    l.columns = ['Season', 'DayNum', 'TeamID', 'PF', 'OpponentID', 'PA']
    l['Win'] = 0

    all_games = pd.concat([w, l], ignore_index=True)
    all_games = all_games.sort_values(['Season', 'TeamID', 'DayNum']).reset_index(drop=True)

    # ─── Season-level aggregates ───
    season_stats = all_games.groupby(['Season', 'TeamID']).agg(
        num_games=('Win', 'count'),
        avg_PF=('PF', 'mean'),
        avg_PA=('PA', 'mean'),
        win_rate=('Win', 'mean'),
    ).reset_index()

    season_stats['net_avg'] = season_stats['avg_PF'] - season_stats['avg_PA']

    # ─── Rolling window aggregates (last 3, 5, 10, 20 games) ───
    # For each team-season, take the last N games and compute means
    def add_rolling(grp, n_suffix):
        """Add rolling features for a given window size suffix (3,5,10,20,None)."""
        result = {}
        if n_suffix is None:
            # Use ALL games (=season stats)
            return {
                f'win_rate': grp['Win'].mean(),
                f'avg_PF': grp['PF'].mean(),
                f'avg_PA': grp['PA'].mean(),
            }
        n = int(n_suffix)
        recent = grp.tail(n) if len(grp) >= n else grp
        return {
            f'win_rate_l{n}': recent['Win'].mean(),
            f'avg_PF_l{n}': recent['PF'].mean(),
            f'avg_PA_l{n}': recent['PA'].mean(),
        }

    windows = [3, 5, 10, 20]
    rolling_features = []

    for (season, tid), grp in all_games.groupby(['Season', 'TeamID']):
        row = {'Season': season, 'TeamID': tid}
        for n in windows:
            recent = grp.tail(n) if len(grp) >= n else grp
            row[f'win_rate_l{n}'] = recent['Win'].mean()
            row[f'avg_PF_l{n}'] = recent['PF'].mean()
            row[f'avg_PA_l{n}'] = recent['PA'].mean()
        rolling_features.append(row)

    rolling_df = pd.DataFrame(rolling_features)

    # Merge season + rolling
    features = season_stats.merge(rolling_df, on=['Season', 'TeamID'], how='left')

    # ─── Efficiency features (from detailed stats) ───
    if reg_detailed is not None:
        det = reg_detailed[reg_detailed['Season'] >= start_year].copy()
        if len(det) > 0:
            log(f"    Computing efficiency features ({len(det)} games)...")

            # Compute possessions and efficiency per game
            # For each game: team = winner, opponent = loser
            det_w = det.copy()
            det_w['TeamID'] = det_w['WTeamID']
            det_w['OpponentID'] = det_w['LTeamID']
            det_w['PF'] = det_w['WScore']
            det_w['PA'] = det_w['LScore']
            det_w['Win'] = 1
            det_w['FGA'] = det_w['WFGA']
            det_w['OR'] = det_w['WOR']
            det_w['TO'] = det_w['WTO']
            det_w['FTA'] = det_w['WFTA']
            det_w['OppFGA'] = det_w['LFGA']
            det_w['OppOR'] = det_w['LOR']
            det_w['OppTO'] = det_w['LTO']
            det_w['OppFTA'] = det_w['LFTA']

            det_l = det.copy()
            det_l['TeamID'] = det_l['LTeamID']
            det_l['OpponentID'] = det_l['WTeamID']
            det_l['PF'] = det_l['LScore']
            det_l['PA'] = det_l['WScore']
            det_l['Win'] = 0
            det_l['FGA'] = det_l['LFGA']
            det_l['OR'] = det_l['LOR']
            det_l['TO'] = det_l['LTO']
            det_l['FTA'] = det_l['LFTA']
            det_l['OppFGA'] = det_l['WFGA']
            det_l['OppOR'] = det_l['WOR']
            det_l['OppTO'] = det_l['WTO']
            det_l['OppFTA'] = det_l['WFTA']

            det_all = pd.concat([det_w, det_l], ignore_index=True)
            det_all['poss'] = (det_all['FGA'] - det_all['OR'] + det_all['TO']
                               + 0.475 * det_all['FTA'])
            det_all['opp_poss'] = (det_all['OppFGA'] - det_all['OppOR']
                                   + det_all['OppTO'] + 0.475 * det_all['OppFTA'])
            det_all['avg_poss'] = (det_all['poss'] + det_all['opp_poss']) / 2
            det_all['avg_poss'] = det_all['avg_poss'].clip(lower=1)  # avoid div by 0
            det_all['off_eff'] = det_all['PF'] / det_all['avg_poss'] * 100
            det_all['def_eff'] = det_all['PA'] / det_all['avg_poss'] * 100
            det_all['net_eff'] = det_all['off_eff'] - det_all['def_eff']

            # Season aggregates for efficiency
            eff_season = det_all.groupby(['Season', 'TeamID']).agg(
                off_eff=('off_eff', 'mean'),
                def_eff=('def_eff', 'mean'),
                net_eff=('net_eff', 'mean'),
            ).reset_index()

            # Rolling windows for efficiency
            eff_rolling = []
            det_all_sorted = det_all.sort_values(['Season', 'TeamID', 'DayNum'])
            for (season, tid), grp in det_all_sorted.groupby(['Season', 'TeamID']):
                row = {'Season': season, 'TeamID': tid}
                for n in [3, 5, 10]:
                    recent = grp.tail(n) if len(grp) >= n else grp
                    row[f'off_eff_l{n}'] = recent['off_eff'].mean()
                    row[f'def_eff_l{n}'] = recent['def_eff'].mean()
                    row[f'net_eff_l{n}'] = recent['net_eff'].mean()
                eff_rolling.append(row)

            eff_rolling_df = pd.DataFrame(eff_rolling)

            # Merge efficiency into features
            features = features.merge(eff_season, on=['Season', 'TeamID'], how='left')
            features = features.merge(eff_rolling_df, on=['Season', 'TeamID'], how='left')

    log(f"    Team features: {features.shape[0]} team-seasons, {features.shape[1]} columns")
    return features


# ═══════════════════════════════════════════════════════════════════════════
# MASSEY FEATURES (men only)
# ═══════════════════════════════════════════════════════════════════════════
def compute_massey_features(massey_df, gender='M'):
    """
    Compute per-team-season Massey ordinal features.
    For each team-season, compute the mean/std/best/worst across the 7 systems.
    Returns: DataFrame indexed by (Season, TeamID).
    """
    if massey_df is None or gender != 'M':
        log("  Massey features: skipping (women or no data)")
        return None

    log("  Computing Massey features (7 systems)...")

    # For each team-season, we want the most recent ranking from each system
    # before the tournament (i.e., highest RankingDayNum per system-season-team)
    massey = massey_df.copy()
    massey = massey.sort_values(['Season', 'TeamID', 'SystemName', 'RankingDayNum'])
    latest = massey.groupby(['Season', 'TeamID', 'SystemName']).last().reset_index()

    # Pivot to get wide format: each system as a column
    # Then compute aggregate stats
    agg = latest.groupby(['Season', 'TeamID']).agg(
        massey_mean=('OrdinalRank', 'mean'),
        massey_std=('OrdinalRank', 'std'),
        massey_best=('OrdinalRank', 'min'),
        massey_worst=('OrdinalRank', 'max'),
        massey_median=('OrdinalRank', 'median'),
        massey_num_systems=('SystemName', 'count'),
    ).reset_index()

    # Rank diff (best - worst) = spread
    agg['massey_spread'] = agg['massey_worst'] - agg['massey_best']

    log(f"    Massey features: {len(agg)} team-seasons")
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# BUILD TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════
def build_training_data(gender='M'):
    """
    Build the training dataset from historical tournament games.

    For each tournament game, create 2 dual samples:
      - (winner - loser, label=1)
      - (loser - winner, label=0)

    Features: difference of team-level features, plus contextual features.

    Returns:
      X: DataFrame of features
      y: numpy array of labels (0/1)
      groups: numpy array of game IDs (for grouped CV)
      metadata: dict with additional info
    """
    log(f"\n{'='*60}")
    log(f"Building training data for {gender}")
    log(f"{'='*60}")

    prefix = 'M' if gender == 'M' else 'W'
    start_year = M_START if gender == 'M' else W_START

    # ─── Load data ───
    tables = load_data(gender)
    reg_compact = tables['reg_compact']
    reg_detailed = tables['reg_detailed']
    tourney_compact = tables['tourney_compact']
    seeds = tables['seeds']
    confs = tables['conferences']
    massey_df = tables['massey']

    # ─── Elo ───
    elo_df = compute_elo(reg_compact, tourney_compact, start_year)

    # ─── Team features ───
    team_feat = compute_team_features(reg_compact, reg_detailed, start_year, gender)

    # ─── Massey features (men only) ───
    massey_feat = compute_massey_features(massey_df, gender)

    # ─── Seeds ───
    seeds = seeds[seeds['Season'] >= start_year].copy()
    seeds['seed_num'] = seeds['Seed'].apply(parse_seed)

    # ─── Filter tournament games ───
    tourney = tourney_compact[tourney_compact['Season'] >= start_year].copy()
    tourney = tourney[tourney['Season'] != 2020].copy()  # No 2020 tournament
    log(f"  Tournament games: {len(tourney)}")

    # ─── Merge in all features ───
    # For each game, we need features for both teams
    # Team A = winner, Team B = loser (for sample 1)
    # Then also Team A = loser, Team B = winner (for sample 2)

    rows = []  # List of feature dicts

    for _, game in tourney.iterrows():
        season = int(game['Season'])
        wteam = int(game['WTeamID'])
        lteam = int(game['LTeamID'])

        # Get features for winner and loser
        def get_feat_dict(tid):
            """Get all features for a team-season."""
            fd = {}

            # Elo
            elo_row = elo_df[(elo_df['Season'] == season) & (elo_df['TeamID'] == tid)]
            fd['elo'] = elo_row['Elo'].values[0] if len(elo_row) > 0 else 1500.0

            # Team features
            tf = team_feat[(team_feat['Season'] == season) & (team_feat['TeamID'] == tid)]
            if len(tf) > 0:
                tf = tf.iloc[0]
                for col in team_feat.columns:
                    if col not in ['Season', 'TeamID']:
                        fd[col] = tf[col]
            else:
                # Defaults for missing team (shouldn't happen for tournament teams)
                fd['num_games'] = 0
                fd['avg_PF'] = 65.0
                fd['avg_PA'] = 65.0
                fd['win_rate'] = 0.5
                fd['net_avg'] = 0.0
                for n in [3, 5, 10, 20]:
                    # Check if these columns exist (they might not if features not computed)
                    fd[f'win_rate_l{n}'] = 0.5
                    fd[f'avg_PF_l{n}'] = 65.0
                    fd[f'avg_PA_l{n}'] = 65.0

            # Seed
            seed_row = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == tid)]
            fd['seed'] = seed_row['seed_num'].values[0] if len(seed_row) > 0 else 16

            # Massey (men only)
            if massey_feat is not None:
                mf = massey_feat[(massey_feat['Season'] == season) & (massey_feat['TeamID'] == tid)]
                if len(mf) > 0:
                    mf = mf.iloc[0]
                    for col in ['massey_mean', 'massey_std', 'massey_best',
                                'massey_worst', 'massey_median', 'massey_spread']:
                        fd[col] = mf[col] if col in mf.index else np.nan

            # Conference
            conf_row = confs[(confs['Season'] == season) & (confs['TeamID'] == tid)]
            fd['conference'] = conf_row['ConfAbbrev'].values[0] if len(conf_row) > 0 else 'UNK'

            return fd

        f_w = get_feat_dict(wteam)
        f_l = get_feat_dict(lteam)

        # Rest days
        # Days since last game for each team
        reg_season = reg_compact[(reg_compact['Season'] == season) &
                                 (reg_compact['DayNum'] < game['DayNum'])]
        last_w = reg_season[(reg_season['WTeamID'] == wteam) | (reg_season['LTeamID'] == wteam)]
        last_l = reg_season[(reg_season['WTeamID'] == lteam) | (reg_season['LTeamID'] == lteam)]
        rest_w = game['DayNum'] - last_w['DayNum'].max() if len(last_w) > 0 else 7
        rest_l = game['DayNum'] - last_l['DayNum'].max() if len(last_l) > 0 else 7
        rest_w = max(rest_w, 1)
        rest_l = max(rest_l, 1)

        # Game-level features
        is_neutral = 1 if game['WLoc'] == 'N' else 0
        same_conf = 1 if f_w.get('conference', '') == f_l.get('conference', '') else 0

        # ─── Create feature diff (winner - loser) → label=1 ───
        row1 = {
            'season': season,
            'game_id': f"{season}_{game['DayNum']}_{wteam}_{lteam}",
            'team_a': wteam,
            'team_b': lteam,
            'label': 1,
        }

        # Add diff features
        diff_keys = ['elo', 'avg_PF', 'avg_PA', 'win_rate', 'net_avg', 'seed',
                     'num_games']
        for n in [3, 5, 10, 20]:
            diff_keys.extend([f'avg_PF_l{n}', f'avg_PA_l{n}', f'win_rate_l{n}'])
        # Efficiency
        eff_keys = ['off_eff', 'def_eff', 'net_eff']
        for n in [3, 5, 10]:
            eff_keys.extend([f'off_eff_l{n}', f'def_eff_l{n}', f'net_eff_l{n}'])
        diff_keys.extend(eff_keys)
        # Massey
        massey_keys = ['massey_mean', 'massey_std', 'massey_best',
                       'massey_worst', 'massey_median', 'massey_spread']
        if massey_feat is not None:
            diff_keys.extend(massey_keys)

        for key in diff_keys:
            val_w = f_w.get(key, np.nan)
            val_l = f_l.get(key, np.nan)
            if val_w is not None and val_l is not None and not (isinstance(val_w, str)):
                row1[key + '_diff'] = val_w - val_l
            else:
                row1[key + '_diff'] = np.nan

        # Context features (same for both directions)
        row1['rest_diff'] = rest_w - rest_l
        row1['is_neutral'] = is_neutral
        row1['same_conf'] = same_conf

        # ─── Sample 2: (loser - winner) → label=0 ───
        row2 = row1.copy()
        row2['team_a'] = lteam
        row2['team_b'] = wteam
        row2['label'] = 0
        row2['game_id'] = f"{season}_{game['DayNum']}_{lteam}_{wteam}"
        for key in diff_keys:
            row2[key + '_diff'] = -(row1.get(key + '_diff', np.nan) or 0)
        row2['rest_diff'] = rest_l - rest_w

        rows.append(row1)
        rows.append(row2)

    df = pd.DataFrame(rows)
    log(f"  Training samples: {len(df)} ({len(df)//2} games × 2 directions)")

    # Separate features, targets, groups
    feature_cols = [c for c in df.columns if c not in [
        'season', 'game_id', 'team_a', 'team_b', 'label']]
    X = df[feature_cols].copy()
    y = df['label'].values
    groups = df['season'].values  # Use season as group for CV
    game_ids = df['game_id'].values

    # Fill NaN with median
    for col in X.columns:
        if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            X[col] = X[col].fillna(X[col].median())

    log(f"  Feature matrix: {X.shape}")
    log(f"  Feature columns: {feature_cols}")

    return X, y, groups, {
        'feature_cols': feature_cols,
        'df_raw': df,
        'game_ids': game_ids,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BUILD TEST DATA (2026 predictions)
# ═══════════════════════════════════════════════════════════════════════════
def build_test_data(gender='M'):
    """
    Build the test dataset for 2026 predictions.

    Returns:
      X_test: DataFrame of features (same columns as training)
      test_ids: array of submission IDs (e.g., "2026_1101_1102")
      team_a_ids, team_b_ids: arrays of team IDs
    """
    log(f"\n{'='*60}")
    log(f"Building test data for {gender} (2026)")
    log(f"{'='*60}")

    prefix = 'M' if gender == 'M' else 'W'
    start_year = M_START if gender == 'M' else W_START

    # ─── Load data (same pipeline) ───
    tables = load_data(gender)
    reg_compact = tables['reg_compact']
    reg_detailed = tables['reg_detailed']
    seeds = tables['seeds']
    confs = tables['conferences']
    massey_df = tables['massey']

    # We only need 2026 team features
    # Compute Elo on all data (to get 2026 Elo)
    # But we don't have tournament data for 2026, so use only regular season
    elo_df = compute_elo(reg_compact, None, start_year)

    # Team features
    team_feat = compute_team_features(reg_compact, reg_detailed, start_year, gender)

    # Massey
    massey_feat = compute_massey_features(massey_df, gender)

    # Seeds (2026)
    seeds = seeds.copy()
    seeds['seed_num'] = seeds['Seed'].apply(parse_seed)

    # ─── Get all 2026 test pairs from submission ───
    sub = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage2.csv")
    test_ids = sub['ID'].values

    # Filter by gender: men IDs < 2000, women IDs >= 3000
    team_pairs = []
    filtered_ids = []
    for tid_str in test_ids:
        parts = tid_str.split('_')
        t1, t2 = int(parts[1]), int(parts[2])
        if gender == 'M' and t1 < 2000 and t2 < 2000:
            team_pairs.append((t1, t2))
            filtered_ids.append(tid_str)
        elif gender == 'W' and t1 >= 3000 and t2 >= 3000:
            team_pairs.append((t1, t2))
            filtered_ids.append(tid_str)

    log(f"  Test pairs: {len(team_pairs)}")

    # ─── Build features for each pair ───
    rows = []
    season_2026 = 2026

    for t1, t2 in team_pairs:
        def get_fd(tid):
            fd = {}

            # Elo
            elo_row = elo_df[(elo_df['Season'] == season_2026) &
                             (elo_df['TeamID'] == tid)]
            fd['elo'] = elo_row['Elo'].values[0] if len(elo_row) > 0 else 1500.0

            # Team features
            tf = team_feat[(team_feat['Season'] == season_2026) &
                           (team_feat['TeamID'] == tid)]
            if len(tf) > 0:
                tf = tf.iloc[0]
                for col in team_feat.columns:
                    if col not in ['Season', 'TeamID']:
                        fd[col] = tf[col]
            else:
                fd['num_games'] = 0
                fd['avg_PF'] = 65.0
                fd['avg_PA'] = 65.0
                fd['win_rate'] = 0.5
                fd['net_avg'] = 0.0

            # Seed (2026)
            seed_row = seeds[(seeds['Season'] == season_2026) &
                             (seeds['TeamID'] == tid)]
            fd['seed'] = seed_row['seed_num'].values[0] if len(seed_row) > 0 else 16

            # Massey
            if massey_feat is not None:
                mf = massey_feat[(massey_feat['Season'] == season_2026) &
                                 (massey_feat['TeamID'] == tid)]
                if len(mf) > 0:
                    mf = mf.iloc[0]
                    for col in ['massey_mean', 'massey_std', 'massey_best',
                                'massey_worst', 'massey_median', 'massey_spread']:
                        fd[col] = mf[col] if col in mf.index else np.nan

            # Conference
            conf_row = confs[(confs['Season'] == season_2026) &
                             (confs['TeamID'] == tid)]
            fd['conference'] = conf_row['ConfAbbrev'].values[0] if len(conf_row) > 0 else 'UNK'

            return fd

        f1 = get_fd(t1)
        f2 = get_fd(t2)

        row = {'team_a': t1, 'team_b': t2}

        # Diff features (same keys as training)
        diff_keys = ['elo', 'avg_PF', 'avg_PA', 'win_rate', 'net_avg', 'seed', 'num_games']
        for n in [3, 5, 10, 20]:
            diff_keys.extend([f'avg_PF_l{n}', f'avg_PA_l{n}', f'win_rate_l{n}'])
        eff_keys = ['off_eff', 'def_eff', 'net_eff']
        for n in [3, 5, 10]:
            eff_keys.extend([f'off_eff_l{n}', f'def_eff_l{n}', f'net_eff_l{n}'])
        diff_keys.extend(eff_keys)
        massey_keys = ['massey_mean', 'massey_std', 'massey_best',
                       'massey_worst', 'massey_median', 'massey_spread']
        if massey_feat is not None:
            diff_keys.extend(massey_keys)

        for key in diff_keys:
            val1 = f1.get(key, np.nan)
            val2 = f2.get(key, np.nan)
            if val1 is not None and val2 is not None and not (isinstance(val1, str)):
                row[key + '_diff'] = val1 - val2
            else:
                row[key + '_diff'] = np.nan

        row['rest_diff'] = 0
        row['is_neutral'] = 1
        same_conf = 1 if f1.get('conference', '') == f2.get('conference', '') else 0
        row['same_conf'] = same_conf

        rows.append(row)

    df_test = pd.DataFrame(rows)
    log(f"  Test features: {df_test.shape}")

    # Use same feature columns as training (will be aligned later)
    return df_test, filtered_ids


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION FOLDS (expanding window by year)
# ═══════════════════════════════════════════════════════════════════════════
def get_cv_folds(gender='M'):
    """Return list of (train_years, val_years) tuples."""
    if gender == 'M':
        folds = [
            (list(range(2003, 2009)), list(range(2009, 2013))),
            (list(range(2003, 2013)), list(range(2013, 2017))),
            (list(range(2003, 2017)), list(range(2017, 2020))),
            (list(range(2003, 2020)), list(range(2021, 2024))),
            (list(range(2003, 2024)), list(range(2024, 2026))),
        ]
    else:
        folds = [
            (list(range(2010, 2014)), list(range(2014, 2017))),
            (list(range(2010, 2017)), list(range(2017, 2020))),
            (list(range(2010, 2020)), list(range(2021, 2023))),
            (list(range(2010, 2023)), list(range(2023, 2026))),
        ]
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (CatBoost + LightGBM with Optuna)
# ═══════════════════════════════════════════════════════════════════════════
def train_catboost_cv(X, y, groups, gender='M', n_trials=30):
    """
    Train CatBoost with Optuna hyperparameter tuning using expanding window CV.
    Returns: trained model (on full data), best params, CV scores.
    """
    log(f"\n  --- CatBoost Training ({gender}) ---")

    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        log("  CatBoost not installed, skipping")
        return None, {}, [np.nan]*N_FOLDS

    folds = get_cv_folds(gender)
    feature_cols = X.columns.tolist()

    # Check GPU availability
    gpu_available = False
    try:
        test_model = CatBoostClassifier(task_type='GPU', iterations=1, verbose=0)
        test_model.fit(X.iloc[:10], groups[:10])
        gpu_available = True
        test_model._save()
        del test_model
        gc.collect()
        log("    GPU is available for CatBoost")
    except Exception:
        log("    GPU not available for CatBoost, using CPU")

    task_type = 'GPU' if gpu_available else 'CPU'

    # ─── Optuna objective ───
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000, step=500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'task_type': task_type,
            'loss_function': 'Logloss',
            'eval_metric': 'BrierScore',
            'verbose': 0,
            'random_seed': 42,
            'early_stopping_rounds': 50,
        }

        cv_scores = []
        for train_years, val_years in folds:
            train_mask = np.isin(groups, train_years)
            val_mask = np.isin(groups, val_years)

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue

            X_tr = X[train_mask]
            y_tr = y[train_mask]
            X_va = X[val_mask]
            y_va = y[val_mask]

            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)

            y_pred = model.predict_proba(X_va)[:, 1]
            score = brier(y_va, y_pred)
            cv_scores.append(score)

        return np.mean(cv_scores) if cv_scores else 1.0

    if optuna is not None:
        log(f"    Optuna tuning with {n_trials} trials...")
        study = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        log(f"    Best CatBoost params: {best_params}")
        log(f"    Best CV Brier: {study.best_value:.5f}")
    else:
        log("    Optuna not available, using default params")
        best_params = {
            'iterations': 1500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'border_count': 128,
            'random_strength': 0.5,
            'bagging_temperature': 0.5,
        }

    # ─── Train final model with best params (on all CV folds, get OOF) ───
    oof_preds = np.zeros(len(X))
    valid_mask = np.zeros(len(X), dtype=bool)
    cv_scores = []
    models = []

    best_params['task_type'] = task_type
    best_params['loss_function'] = 'Logloss'
    best_params['eval_metric'] = 'BrierScore'
    best_params['verbose'] = 0
    best_params['random_seed'] = 42
    best_params['early_stopping_rounds'] = 50

    for train_years, val_years in folds:
        train_mask = np.isin(groups, train_years)
        val_mask = np.isin(groups, val_years)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        X_va = X[val_mask]
        y_va = y[val_mask]
        idx_va = np.where(val_mask)[0]

        model = CatBoostClassifier(**best_params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)

        y_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[idx_va] = y_pred
        valid_mask[idx_va] = True

        score = brier(y_va, y_pred)
        cv_scores.append(score)
        models.append(model)

        log(f"      Fold: train {train_years[0]}-{train_years[-1]} -> "
            f"val {val_years[0]}-{val_years[-1]}: Brier={score:.5f}")

    mean_cv = np.mean(cv_scores)
    log(f"    CatBoost CV Brier: {mean_cv:.5f} (mean of {len(cv_scores)} folds)")

    # Train on full data
    log(f"    Training CatBoost on full data...")
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y, verbose=0)

    return final_model, best_params, cv_scores, oof_preds, mean_cv, valid_mask


def train_lightgbm_cv(X, y, groups, gender='M', n_trials=30):
    """
    Train LightGBM with Optuna hyperparameter tuning using expanding window CV.
    """
    log(f"\n  --- LightGBM Training ({gender}) ---")

    try:
        import lightgbm as lgb
    except ImportError:
        log("  LightGBM not installed, skipping")
        return None, {}, [np.nan]*N_FOLDS, None, np.nan

    folds = get_cv_folds(gender)

    # Check GPU availability
    gpu_available = False
    try:
        test_data = lgb.Dataset(X.iloc[:10], label=groups[:10])
        test_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
                       'objective': 'binary', 'verbose': -1}
        gpu_model = lgb.train(test_params, test_data, num_boost_round=2)
        gpu_available = True
        log("    GPU is available for LightGBM")
        del gpu_model, test_data
        gc.collect()
    except Exception:
        log("    GPU not available for LightGBM, using CPU")

    device = 'gpu' if gpu_available else 'cpu'

    # ─── Optuna objective ───
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'device': device,
            'verbosity': -1,
            'seed': 42,
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }

        cv_scores = []
        for train_years, val_years in folds:
            train_mask = np.isin(groups, train_years)
            val_mask = np.isin(groups, val_years)

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue

            X_tr = X[train_mask]
            y_tr = y[train_mask]
            X_va = X[val_mask]
            y_va = y[val_mask]

            # Use callbacks for early stopping
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, first_metric_only=True),
                           lgb.log_evaluation(0)]
            )

            y_pred = model.predict_proba(X_va)[:, 1]
            score = brier(y_va, y_pred)
            cv_scores.append(score)

        return np.mean(cv_scores) if cv_scores else 1.0

    if optuna is not None:
        log(f"    Optuna tuning with {n_trials} trials...")
        study = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        log(f"    Best LightGBM params: {best_params}")
        log(f"    Best CV Brier: {study.best_value:.5f}")
    else:
        log("    Optuna not available, using default params")
        best_params = {
            'n_estimators': 1500,
            'learning_rate': 0.05,
            'num_leaves': 64,
            'min_child_samples': 50,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

    # ─── Train with CV to get OOF ───
    oof_preds = np.zeros(len(X))
    valid_mask = np.zeros(len(X), dtype=bool)
    cv_scores = []
    models = []

    cfg = {**best_params,
           'objective': 'binary',
           'metric': 'binary_logloss',
           'device': device,
           'verbosity': -1,
           'seed': 42}

    for train_years, val_years in folds:
        train_mask = np.isin(groups, train_years)
        val_mask = np.isin(groups, val_years)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        X_va = X[val_mask]
        y_va = y[val_mask]
        idx_va = np.where(val_mask)[0]

        model = lgb.LGBMClassifier(**cfg)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, first_metric_only=True),
                       lgb.log_evaluation(0)]
        )

        y_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[idx_va] = y_pred
        valid_mask[idx_va] = True

        score = brier(y_va, y_pred)
        cv_scores.append(score)
        models.append(model)

        log(f"      Fold: train {train_years[0]}-{train_years[-1]} -> "
            f"val {val_years[0]}-{val_years[-1]}: Brier={score:.5f}")

    mean_cv = np.mean(cv_scores)
    log(f"    LightGBM CV Brier: {mean_cv:.5f} (mean of {len(cv_scores)} folds)")

    # Final model
    log(f"    Training LightGBM on full data...")
    final_model = lgb.LGBMClassifier(**cfg)
    final_model.fit(X, y,
                    callbacks=[lgb.log_evaluation(0)])

    return final_model, best_params, cv_scores, oof_preds, mean_cv, valid_mask


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_models(cb_oof, lgb_oof, cb_cv, lgb_cv, y_true, gender='M',
                    cb_valid_mask=None, lgb_valid_mask=None):
    """
    Ensemble CatBoost and LightGBM using weighted average.

    Weights = (1/Brier) / sum(1/Brier) for available models.

    Parameters:
      cb_oof / lgb_oof: OOF predictions (may have zeros for never-validated samples)
      cb_cv / lgb_cv: list of per-fold CV Brier scores
      cb_valid_mask / lgb_valid_mask: boolean mask of samples that were in a validation fold

    Returns:
      weights, model_names, ensemble_brier (computed only on validated samples)
    """
    log(f"\n  --- Ensemble ({gender}) ---")

    available_models = {}
    oof_preds = {}
    valid_masks = {}

    if cb_oof is not None:
        available_models['catboost'] = np.mean(cb_cv) if cb_cv else np.nan
        oof_preds['catboost'] = cb_oof
        valid_masks['catboost'] = cb_valid_mask if cb_valid_mask is not None else np.ones(len(y_true), dtype=bool)
    if lgb_oof is not None:
        available_models['lightgbm'] = np.mean(lgb_cv) if lgb_cv else np.nan
        oof_preds['lightgbm'] = lgb_oof
        valid_masks['lightgbm'] = lgb_valid_mask if lgb_valid_mask is not None else np.ones(len(y_true), dtype=bool)

    if len(available_models) == 0:
        log("    No models available for ensemble!")
        return None, [], np.nan

    # Compute weights inversely proportional to Brier score
    scores = list(available_models.values())
    weights = np.array([1.0 / max(s, 0.01) for s in scores])
    weights = weights / weights.sum()

    model_names = list(available_models.keys())
    log(f"    Model scores: {dict(zip(model_names, [f'{s:.5f}' for s in scores]))}")
    log(f"    Weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")

    # Combined valid mask (samples validated by at least one model)
    combined_valid = np.zeros(len(y_true), dtype=bool)
    for name in model_names:
        combined_valid |= valid_masks[name]

    # Compute ensemble OOF only on validated samples
    ensemble_oof = np.zeros(len(y_true), dtype=np.float64)
    for i, name in enumerate(model_names):
        ensemble_oof += weights[i] * oof_preds[name]

    n_valid = combined_valid.sum()
    if n_valid > 0:
        ensemble_brier = brier(y_true[combined_valid], ensemble_oof[combined_valid])
        log(f"    Ensemble OOF Brier (on {n_valid} validated samples): {ensemble_brier:.5f}")
    else:
        ensemble_brier = np.mean(scores)
        log(f"    No validated samples found, using mean CV Brier: {ensemble_brier:.5f}")

    # Hill climbing: optimize weights for minimum Brier (on validated samples only)
    if len(model_names) >= 2 and n_valid > 100:
        try:
            def objective(w):
                w = np.array(w, dtype=np.float64)
                w = w / w.sum()
                pred = np.zeros(len(y_true), dtype=np.float64)
                for i, name in enumerate(model_names):
                    pred += w[i] * oof_preds[name]
                return brier(y_true[combined_valid], pred[combined_valid])

            n = len(model_names)
            result = minimize(
                objective,
                x0=weights,
                bounds=[(0, 1)] * n,
                constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                method='SLSQP',
            )

            if result.success:
                opt_weights = result.x / result.x.sum()
                opt_brier = result.fun
                log(f"    Optimized weights: {dict(zip(model_names, [f'{w:.3f}' for w in opt_weights]))}")
                log(f"    Optimized Brier: {opt_brier:.5f}")

                if opt_brier < ensemble_brier:
                    weights = opt_weights
                    ensemble_brier = opt_brier
                    ensemble_oof = np.zeros(len(y_true), dtype=np.float64)
                    for i, name in enumerate(model_names):
                        ensemble_oof += weights[i] * oof_preds[name]

        except Exception as e:
            log(f"    Hill climbing failed: {e}")

    log(f"    Final ensemble OOF Brier: {ensemble_brier:.5f}")
    return weights, model_names, ensemble_brier
    return weights, model_names, ensemble_brier


# ═══════════════════════════════════════════════════════════════════════════
# PSEUDO-LABELING
# ═══════════════════════════════════════════════════════════════════════════
def pseudo_labeling(models, weights, model_names, gender, X_train, y_train,
                    X_test, threshold=0.9):
    """
    Pseudo-label high-confidence test predictions and retrain.
    """
    log(f"\n  --- Pseudo-Labeling ({gender}) ---")

    # Get ensemble predictions on test
    test_preds = np.zeros(X_test.shape[0])
    for i, name in enumerate(model_names):
        model = models.get(name)
        if model is not None:
            test_preds += weights[i] * model.predict_proba(X_test)[:, 1]

    # High confidence: > 0.9 or < 0.1
    high_conf_mask = (test_preds >= threshold) | (test_preds <= (1 - threshold))
    n_pseudo = high_conf_mask.sum()
    log(f"    High-confidence test samples: {n_pseudo} / {len(test_preds)}")

    if n_pseudo < 50:
        log(f"    Too few pseudo-labels, skipping")
        return models, test_preds

    # Create pseudo-labeled data
    pseudo_labels = (test_preds >= threshold).astype(int)
    X_pseudo = X_test[high_conf_mask].copy()
    y_pseudo = pseudo_labels[high_conf_mask]

    # Combine with training data
    X_combined = np.vstack([X_train, X_pseudo])
    y_combined = np.concatenate([y_train, y_pseudo])

    log(f"    Retraining on combined data ({len(X_combined)} samples)...")

    # Retrain each model
    for i, name in enumerate(model_names):
        if name == 'catboost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**models.get('catboost_params', {}))
            model.fit(X_combined, y_combined, verbose=0)
            models[name] = model
        elif name == 'lightgbm':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**models.get('lightgbm_params', {}))
            model.fit(X_combined, y_combined,
                      callbacks=[lgb.log_evaluation(0)])
            models[name] = model

    # Updated predictions
    test_preds = np.zeros(X_test.shape[0])
    for i, name in enumerate(model_names):
        model = models.get(name)
        if model is not None and name in models:
            test_preds += weights[i] * models[name].predict_proba(X_test)[:, 1]

    return models, test_preds


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Run full pipeline for one gender
# ═══════════════════════════════════════════════════════════════════════════
def run_gender_pipeline(gender='M', n_optuna_trials=50):
    """
    Run the full pipeline for one gender:
      1. Build training data
      2. Build test data
      3. Train CatBoost + LightGBM with Optuna
      4. Ensemble
      5. Pseudo-label (optional)
      6. Return test predictions and metadata
    """
    log(f"\n{'='*60}")
    log(f"RUNNING PIPELINE FOR GENDER: {gender}")
    log(f"{'='*60}")

    X, y, groups, metadata = build_training_data(gender)
    df_test, test_ids = build_test_data(gender)

    # Align feature columns
    feature_cols = metadata['feature_cols']
    # Make sure test has same columns as training
    test_features = [c for c in feature_cols if c in df_test.columns]
    missing = [c for c in feature_cols if c not in df_test.columns]
    if missing:
        log(f"  Missing test columns: {missing}")
        for c in missing:
            df_test[c] = 0.0

    X_train = X[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = y

    # ─── Train CatBoost ───
    cb_result = train_catboost_cv(X_train, y_train, groups, gender, n_optuna_trials)
    if cb_result is not None and cb_result[0] is not None:
        cb_model, cb_params, cb_cv, cb_oof, cb_mean_cv, cb_valid_mask = cb_result
    else:
        cb_model = cb_params = cb_cv = cb_oof = None
        cb_mean_cv = np.nan
        cb_valid_mask = None

    # ─── Train LightGBM ───
    lgb_result = train_lightgbm_cv(X_train, y_train, groups, gender, n_optuna_trials)
    if lgb_result is not None and lgb_result[0] is not None:
        lgb_model, lgb_params, lgb_cv, lgb_oof, lgb_mean_cv, lgb_valid_mask = lgb_result
    else:
        lgb_model = lgb_params = lgb_cv = lgb_oof = None
        lgb_mean_cv = np.nan
        lgb_valid_mask = None

    # ─── Ensemble ───
    ensemble_weights, ensemble_names, ensemble_brier = ensemble_models(
        cb_oof, lgb_oof, cb_cv, lgb_cv, y_train, gender,
        cb_valid_mask=cb_valid_mask, lgb_valid_mask=lgb_valid_mask
    )

    # ─── Predict on test data ───
    log(f"\n  Making test predictions ({gender})...")
    test_preds = np.zeros(X_test.shape[0])

    if 'catboost' in ensemble_names and cb_model is not None:
        log(f"    CatBoost predicting...")
        cb_test = cb_model.predict_proba(X_test)[:, 1]
        idx = ensemble_names.index('catboost')
        test_preds += ensemble_weights[idx] * cb_test

    if 'lightgbm' in ensemble_names and lgb_model is not None:
        log(f"    LightGBM predicting...")
        lgb_test = lgb_model.predict_proba(X_test)[:, 1]
        idx = ensemble_names.index('lightgbm')
        test_preds += ensemble_weights[idx] * lgb_test

    # Collect metadata for logging
    meta = {
        'gender': gender,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_cols),
        'cb_cv_scores': cb_cv if cb_cv is not None else [],
        'lgb_cv_scores': lgb_cv if lgb_cv is not None else [],
        'cb_params': cb_params if cb_params else {},
        'lgb_params': lgb_params if lgb_params else {},
        'ensemble_brier': ensemble_brier,
        'test_preds': test_preds,
        'test_ids': test_ids,
        'ensemble_weights': ensemble_weights.tolist() if ensemble_weights is not None else [],
        'ensemble_names': ensemble_names,
        'feature_cols': feature_cols,
    }

    # Add feature importances if available
    if lgb_model is not None:
        try:
            lgb_importances = lgb_model.feature_importances_
            feat_imp = sorted(zip(feature_cols, lgb_importances), key=lambda x: x[1], reverse=True)
            meta['lgb_importances'] = feat_imp
        except:
            meta['lgb_importances'] = []
    else:
        meta['lgb_importances'] = []

    return meta


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    log(f"=" * 60)
    log(f"NCAA KAGGLE SOLUTION A — GBDT INTEGRATION APPROACH")
    log(f"=" * 60)
    log(f"Data: {DATA_DIR}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # ─── Process Men ───
    try:
        log(f"\n>>> RUNNING MEN'S PIPELINE")
        m_meta = run_gender_pipeline('M', n_optuna_trials=N_OPTUNA_TRIALS)
        results['M'] = m_meta
    except Exception as e:
        log(f"ERROR in men's pipeline: {e}")
        import traceback
        traceback.print_exc()
        results['M'] = None

    # ─── Process Women ───
    try:
        log(f"\n{'='*60}")
        log(f">>> RUNNING WOMEN'S PIPELINE")
        w_meta = run_gender_pipeline('W', n_optuna_trials=N_OPTUNA_TRIALS)
        results['W'] = w_meta
    except Exception as e:
        log(f"ERROR in women's pipeline: {e}")
        import traceback
        traceback.print_exc()
        results['W'] = None

    # ─── Generate Submission ───
    log(f"\n{'='*60}")
    log(f"GENERATING SUBMISSION")
    log(f"{'='*60}")

    sub = load_submission()
    id_to_pred = {}

    for gender, meta in results.items():
        if meta is None:
            log(f"  No results for {gender}, using 0.5")
            continue

        test_ids = meta['test_ids']
        test_preds = meta['test_preds']
        log(f"  Processing {gender}: {len(test_ids)} test pairs")

        for i, tid in enumerate(test_ids):
            pred = float(test_preds[i])
            # Clip to [0.05, 0.95] for numerical stability
            pred = max(0.05, min(0.95, pred))
            id_to_pred[tid] = pred

    # Fill submission
    sub['Pred'] = sub['ID'].map(id_to_pred).fillna(0.5)

    # Save
    sub_path = os.path.join(OUTPUT_DIR, 'submission_a.csv')
    sub.to_csv(sub_path, index=False)
    log(f"  Saved submission to: {sub_path}")
    log(f"  Coverage: {sub['Pred'].notna().sum()} / {len(sub)} predictions")

    # ─── Save detailed log ───
    log_path = os.path.join(OUTPUT_DIR, 'solution_a_log.md')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("# Solution A — Experiment Log\n\n")
        f.write(f"**Run Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Duration:** {(time.time() - t_start) / 60:.1f} minutes\n\n")

        for gender in ['M', 'W']:
            meta = results.get(gender)
            f.write(f"## {gender} Results\n\n")
            if meta is None:
                f.write("Pipeline failed.\n\n")
                continue

            f.write(f"- Training samples: {meta['n_train']}\n")
            f.write(f"- Test samples: {meta['n_test']}\n")
            f.write(f"- Features: {meta['n_features']}\n\n")

            # CV scores
            f.write("### CV Brier Scores\n\n")
            if meta['cb_cv_scores']:
                cb_scores = meta['cb_cv_scores']
                f.write(f"- CatBoost: mean={np.mean(cb_scores):.5f}, "
                       f"scores={[f'{s:.5f}' for s in cb_scores]}\n")
            if meta['lgb_cv_scores']:
                lgb_scores = meta['lgb_cv_scores']
                f.write(f"- LightGBM: mean={np.mean(lgb_scores):.5f}, "
                       f"scores={[f'{s:.5f}' for s in lgb_scores]}\n")
            f.write(f"- Ensemble OOF Brier: {meta['ensemble_brier']:.5f}\n\n")

            # Best params
            f.write("### CatBoost Best Parameters\n\n")
            if meta['cb_params']:
                for k, v in meta['cb_params'].items():
                    f.write(f"- {k}: {v}\n")
            f.write("\n")

            f.write("### LightGBM Best Parameters\n\n")
            if meta['lgb_params']:
                for k, v in meta['lgb_params'].items():
                    f.write(f"- {k}: {v}\n")
            f.write("\n")

            # Feature importance (if available)
            f.write("### Top 20 Features (LightGBM importance)\n\n")
            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|-----------|\n")
            if 'lgb_importances' in meta and meta['lgb_importances']:
                for r, (feat, imp) in enumerate(meta['lgb_importances'][:20]):
                    f.write(f"| {r+1} | {feat} | {imp} |\n")
            f.write("\n")

            # Ensemble info
            f.write(f"### Ensemble\n\n")
            f.write(f"- Model weights: {dict(zip(meta['ensemble_names'], [f'{w:.3f}' for w in meta['ensemble_weights']]))}\n")
            f.write(f"- Ensemble Brier: {meta['ensemble_brier']:.5f}\n\n")

        # Summary
        f.write("## Summary\n\n")
        for gender in ['M', 'W']:
            meta = results.get(gender)
            if meta:
                f.write(f"**{gender}:** Ensemble Brier = {meta['ensemble_brier']:.5f}\n")

        f.write(f"\n*Generated by solution_a.py*\n")

    log(f"  Saved log to: {log_path}")
    log(f"\n{'='*60}")
    log(f"Total time: {(time.time() - t_start) / 60:.1f} minutes")
    log(f"DONE!")
    log(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()
