#!/usr/bin/env python3
"""
2026 世界杯 Monte Carlo 锦标赛仿真
=====================================
从当前实际赛程出发，使用 LightGBM 精模预测每场剩余比赛，
模拟 10000 次完整世界杯，统计各队夺冠/四强/八强概率。
"""

import os, sys, json, pickle, warnings, random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ============================================================
# 0. 路径配置
# ============================================================
BASE = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data")
OUT_DIR = BASE / "simulation"
FOOTBALL_DIR = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/football_data")

OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PKL = BASE / "models/saved_models/final_model.pkl"
FEATURE_NAMES_JSON = BASE / "models/feature_names_full.json"
FEATURE_MATRIX_CSV = BASE / "features/feature_matrix_full.csv"
TEAM_FEATURES_CSV = BASE / "features/team_current_features.csv"

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("2026 世界杯 Monte Carlo 锦标赛仿真")
print("=" * 60)

# 1a 模型和特征
print("\n[1a] 加载模型...")
with open(FEATURE_NAMES_JSON, "r") as f:
    FEATURE_NAMES = json.load(f)

with open(MODEL_PKL, "rb") as f:
    PIPELINE = pickle.load(f)

print(f"    模型类型: {type(PIPELINE['model']).__name__}")
print(f"    特征数量: {len(FEATURE_NAMES)}")

# 1b 球队特征
print("\n[1b] 加载球队特征...")
team_features = pd.read_csv(TEAM_FEATURES_CSV, encoding="utf-8-sig")
team_features.set_index("team", inplace=True)
print(f"    球队数量: {len(team_features)}")

# 1b2 加载完整特征矩阵（用于提取模拟阶段缺失的特征）
print("\n[1b2] 加载完整特征矩阵，计算球队级缺失特征...")
feature_matrix = pd.read_csv(FEATURE_MATRIX_CSV, encoding="utf-8-sig")
feature_matrix['date'] = pd.to_datetime(feature_matrix['date'])

# 从完整特征矩阵中提取每个球队的最新特征值
def compute_team_extra_features(fm: pd.DataFrame) -> dict:
    """从特征矩阵计算 team_features.csv 中缺失的球队级特征"""
    extra = {}
    all_teams = set(fm['home_team'].unique()) | set(fm['away_team'].unique())
    for team in all_teams:
        # 球队作为主队和客队的比赛，按时间排序
        home_rows = fm[fm['home_team'] == team].sort_values('date')
        away_rows = fm[fm['away_team'] == team].sort_values('date')
        # 取最新一条记录
        if len(home_rows) > 0 and len(away_rows) > 0:
            latest_h = home_rows.iloc[-1]
            latest_a = away_rows.iloc[-1]
            latest = latest_h if latest_h['date'] >= latest_a['date'] else latest_a
            prefix = 'home' if (isinstance(latest.get('home_team'), str) and latest['home_team'] == team) else 'away'
        elif len(home_rows) > 0:
            latest = home_rows.iloc[-1]
            prefix = 'home'
        elif len(away_rows) > 0:
            latest = away_rows.iloc[-1]
            prefix = 'away'
        else:
            continue
        extra[team] = {
            'goals_conceded_std_5': float(latest.get(f'{prefix}_goals_conceded_std_5', 1.5)),
            'goals_conceded_std_10': float(latest.get(f'{prefix}_goals_conceded_std_10', 1.5)),
            'wc_avg_goals_for': float(latest.get(f'{prefix}_wc_avg_goals_for', 1.2)),
            'wc_avg_goals_against': float(latest.get(f'{prefix}_wc_avg_goals_against', 1.2)),
            'days_since_last_match': float(latest.get(f'{prefix}_days_since_last_match', 4.0)),
        }
    return extra

team_extra_features = compute_team_extra_features(feature_matrix)
print(f"    已计算 {len(team_extra_features)} 支球队的缺失特征")

# 1c 比赛数据
print("\n[1c] 加载世界杯赛程...")
wc2026 = pd.read_csv(FOOTBALL_DIR / "wc2026_matches.csv")
wc2026['date'] = pd.to_datetime(wc2026['date'])
print(f"    比赛数量: {len(wc2026)}")

# 1d 小组积分榜
print("\n[1d] 加载小组积分榜...")
standings = pd.read_csv(FOOTBALL_DIR / "wc2026_group_standings.csv")
print(f"    小组记录: {len(standings)}")

# ============================================================
# 2. 解析当前赛程状态
# ============================================================
print("\n[2] 解析赛程状态...")

# 已完成的比赛
completed = wc2026[wc2026['status'] == 'Completed'].copy()
print(f"    已完成比赛: {len(completed)}")

# 标记各阶段比赛
group_matches = completed[completed['stage'] == 'Group Stage']
ro32_matches = completed[completed['stage'] == 'Round of 32']
ro16_matches = completed[completed['stage'] == 'Round of 16']
qf_matches = completed[completed['stage'] == 'Quarter-finals']

print(f"      小组赛: {len(group_matches)}")
print(f"      32强: {len(ro32_matches)}")
print(f"      16强: {len(ro16_matches)}")

# 已确定的出线球队（从小组积分榜获得）
groups = standings['group'].unique()
group_winners = {}
group_runners_up = {}
for g in groups:
    grp = standings[standings['group'] == g].sort_values('position')
    group_winners[g] = grp.iloc[0]['team']
    group_runners_up[g] = grp.iloc[1]['team']

print(f"\n    小组第一: {dict(list(group_winners.items())[:6])}...")
print(f"    小组第二: {dict(list(group_runners_up.items())[:6])}...")

# RO32 晋级球队
ro32_winners = {}
for _, row in ro32_matches.iterrows():
    home = row['home_team']
    away = row['away_team']
    hs = row['home_score']
    aws = row['away_score']
    if hs > aws:
        winner = home
    elif aws > hs:
        winner = away
    else:
        # 平局 → 点球（数据没记录点球结果，根据RO16推断）
        # 从下一轮推断
        winner = None
    ro32_winners[f"{row['home_team']}_vs_{row['away_team']}"] = (home, away, winner)

# RO16 晋级球队
ro16_winners = {}
for _, row in ro16_matches.iterrows():
    home = row['home_team']
    away = row['away_team']
    hs = row['home_score']
    aws = row['away_score']
    if hs > aws:
        winner = home
    elif aws > hs:
        winner = away
    else:
        winner = None  # 需要从下一轮推断
    ro16_winners[f"{row['home_team']}_vs_{row['away_team']}"] = (home, away, winner)

# 推断点球结果（从下一轮参赛队）
# RO32 平局推断
ro32_pk_winners = {}
for match_key, (h, a, w) in ro32_winners.items():
    if w is not None:
        continue
    # 检查下一轮谁出现了
    for _, r16_row in ro16_matches.iterrows():
        if r16_row['home_team'] == h or r16_row['home_team'] == a or \
           r16_row['away_team'] == h or r16_row['away_team'] == a:
            if r16_row['home_team'] == h or r16_row['away_team'] == h:
                ro32_pk_winners[match_key] = h
            else:
                ro32_pk_winners[match_key] = a
            break

# RO16 平局推断
ro16_pk_winners = {}
for match_key, (h, a, w) in ro16_winners.items():
    if w is not None:
        continue
    # 查看 QF 参赛队
    for _, qf_row in qf_matches.iterrows():
        if qf_row['home_team'] == h or qf_row['home_team'] == a:
            if qf_row['home_team'] == h:
                ro16_pk_winners[match_key] = h
            else:
                ro16_pk_winners[match_key] = a
            break
        if qf_row['away_team'] == h or qf_row['away_team'] == a:
            if qf_row['away_team'] == h:
                ro16_pk_winners[match_key] = h
            else:
                ro16_pk_winners[match_key] = a
            break

print(f"\n    RO32 点球推断: {ro32_pk_winners}")
print(f"    RO16 点球推断: {ro16_pk_winners}")

# 当前 8 强
remaining_8 = set()
for _, row in ro16_matches.iterrows():
    h, a = row['home_team'], row['away_team']
    hs, aws = row['home_score'], row['away_score']
    if hs > aws:
        remaining_8.add(h)
    elif aws > hs:
        remaining_8.add(a)

# 加上点球胜者
for winner in ro16_pk_winners.values():
    remaining_8.add(winner)

print(f"\n    当前 8 强: {sorted(remaining_8)}")

# ============================================================
# 3. 定义剩余比赛对阵
# ============================================================
print("\n[3] 构建剩余比赛对阵...")

# 根据实际赛程，剩余比赛对阵如下：
# QF:    France vs Morocco
#        Norway vs England
#        Spain vs Belgium
#        Argentina vs Switzerland
# SF:    Winner(FRA/MAR) vs Winner(NOR/ENG)
#        Winner(ESP/BEL) vs Winner(ARG/SUI)
# Final: Winner(SF1) vs Winner(SF2)

REMAINING_MATCHES = [
    # (stage, home, away)
    ("QF", "France", "Morocco"),
    ("QF", "Norway", "England"),
    ("QF", "Spain", "Belgium"),
    ("QF", "Argentina", "Switzerland"),
]

# 这些已从赛程推断
print(f"    剩余 QF 场次: {len(REMAINING_MATCHES)}")

# ============================================================
# 4. 构建预测特征
# ============================================================
print("\n[4] 构建匹配特征...")

def build_match_features(home_team, away_team, team_df, is_neutral=1, team_extra=None):
    """为一场比赛构建所有模型特征"""
    if team_extra is None:
        team_extra = {}

    # 获取两队特征
    if home_team not in team_df.index or away_team not in team_df.index:
        print(f"    WARNING: 球队不在特征数据中: {home_team} vs {away_team}")
        return None

    ht = team_df.loc[home_team]
    at = team_df.loc[away_team]

    # 获取缺失的球队级特征
    ht_extra = team_extra.get(home_team, {})
    at_extra = team_extra.get(away_team, {})

    feat = {}

    # Elo
    feat['elo_home_pre'] = float(ht['elo_rating'])
    feat['elo_away_pre'] = float(at['elo_rating'])
    feat['elo_diff'] = feat['elo_home_pre'] - feat['elo_away_pre']

    # 场地
    feat['is_neutral'] = float(is_neutral)
    feat['is_actual_home'] = 0.0  # 世界杯中立场地

    # 主队近期表现
    feat['home_win_rate_5'] = float(ht['win_rate_5'])
    feat['home_win_rate_10'] = float(ht['win_rate_10'])
    feat['home_avg_goals_for_5'] = float(ht['avg_goals_for_5'])
    feat['home_avg_goals_for_10'] = float(ht['avg_goals_for_10'])
    feat['home_avg_goals_against_5'] = float(ht['avg_goals_against_5'])
    feat['home_avg_goals_against_10'] = float(ht['avg_goals_against_10'])
    feat['home_net_goals_5'] = float(ht['net_goals_5'])
    feat['home_net_goals_10'] = float(ht['net_goals_10'])
    feat['home_goal_conversion_rate'] = float(ht['goal_conversion_rate'])
    feat['home_total_matches_played'] = float(ht['total_matches_played'])

    # 客队近期表现
    feat['away_win_rate_5'] = float(at['win_rate_5'])
    feat['away_win_rate_10'] = float(at['win_rate_10'])
    feat['away_avg_goals_for_5'] = float(at['avg_goals_for_5'])
    feat['away_avg_goals_for_10'] = float(at['avg_goals_for_10'])
    feat['away_avg_goals_against_5'] = float(at['avg_goals_against_5'])
    feat['away_avg_goals_against_10'] = float(at['avg_goals_against_10'])
    feat['away_net_goals_5'] = float(at['net_goals_5'])
    feat['away_net_goals_10'] = float(at['net_goals_10'])
    feat['away_goal_conversion_rate'] = float(at['goal_conversion_rate'])
    feat['away_total_matches_played'] = float(at['total_matches_played'])

    # 加权近况（使用 win_rate 作为近似）
    feat['home_weighted_form_5'] = float(ht['win_rate_5'])
    feat['home_weighted_form_10'] = float(ht['win_rate_10'])
    feat['away_weighted_form_5'] = float(at['win_rate_5'])
    feat['away_weighted_form_10'] = float(at['win_rate_10'])

    # 进球稳定性
    feat['home_goals_conceded_std_5'] = ht_extra.get('goals_conceded_std_5', min(3.0, float(ht['avg_goals_against_5']) * 0.6 + 0.5))
    feat['home_goals_conceded_std_10'] = ht_extra.get('goals_conceded_std_10', min(3.0, float(ht['avg_goals_against_10']) * 0.6 + 0.5))
    feat['away_goals_conceded_std_5'] = at_extra.get('goals_conceded_std_5', min(3.0, float(at['avg_goals_against_5']) * 0.6 + 0.5))
    feat['away_goals_conceded_std_10'] = at_extra.get('goals_conceded_std_10', min(3.0, float(at['avg_goals_against_10']) * 0.6 + 0.5))

    # FIFA 排名
    feat['home_fifa_rank'] = float(ht['fifa_rank'])
    feat['away_fifa_rank'] = float(at['fifa_rank'])
    feat['fifa_rank_diff'] = feat['away_fifa_rank'] - feat['home_fifa_rank']  # 与训练一致：正=主队排名更好
    feat['home_fifa_points'] = float(ht['fifa_points'])
    feat['away_fifa_points'] = float(at['fifa_points'])
    feat['fifa_points_diff'] = feat['home_fifa_points'] - feat['away_fifa_points']

    # 排名变化
    feat['home_rank_change_3m'] = float(ht['rank_change_3m'])
    feat['home_rank_change_6m'] = float(ht['rank_change_6m'])
    feat['home_rank_change_12m'] = float(ht['rank_change_12m'])
    feat['away_rank_change_3m'] = float(at['rank_change_3m'])
    feat['away_rank_change_6m'] = float(at['rank_change_6m'])
    feat['away_rank_change_12m'] = float(at['rank_change_12m'])

    # 世界杯历史
    feat['home_wc_matches_played'] = float(ht['wc_matches_played'])
    feat['home_wc_win_rate'] = float(ht['wc_win_rate'])
    # 世界杯进球数据
    feat['home_wc_avg_goals_for'] = ht_extra.get('wc_avg_goals_for', 1.2)
    feat['home_wc_avg_goals_against'] = ht_extra.get('wc_avg_goals_against', 1.2)
    feat['away_wc_matches_played'] = float(at['wc_matches_played'])
    feat['away_wc_win_rate'] = float(at['wc_win_rate'])
    feat['away_wc_avg_goals_for'] = at_extra.get('wc_avg_goals_for', 1.2)
    feat['away_wc_avg_goals_against'] = at_extra.get('wc_avg_goals_against', 1.2)

    # 是否同联合会
    home_conf = str(ht['confederation'])
    away_conf = str(at['confederation'])
    feat['same_confederation'] = 1.0 if home_conf == away_conf else 0.0

    # 距上场比赛天数
    feat['home_days_since_last_match'] = ht_extra.get('days_since_last_match', 4.0)
    feat['away_days_since_last_match'] = at_extra.get('days_since_last_match', 4.0)

    # K 因子与比赛重要性
    feat['k_factor'] = 60.0
    feat['match_importance'] = 1.0

    # 联合会 one-hot
    confs = ['AFC', 'CAF', 'CONCACAF', 'CONMEBOL', 'OFC', 'UEFA', 'Unknown']
    for c in confs:
        feat[f'home_confederation_{c}'] = 1.0 if home_conf == c else 0.0
        feat[f'away_confederation_{c}'] = 1.0 if away_conf == c else 0.0

    return feat


def predict_match_proba(home_team, away_team, team_df, pipeline, feature_names, team_extra=None):
    """预测一场比赛的主胜/平/客胜概率"""

    feat = build_match_features(home_team, away_team, team_df, team_extra=team_extra)
    if feat is None:
        # Fallback: Elo-based prediction
        return elo_fallback(home_team, away_team, team_df)

    # 构造特征向量（按 feature_names 顺序）
    X = np.array([feat.get(fn, 0.0) for fn in feature_names]).reshape(1, -1)

    # 预测
    try:
        if pipeline.get("calibrated", False):
            proba = pipeline["calibrator"].predict_proba(X)
        else:
            proba = pipeline["model"].predict_proba(X)
    except Exception as e:
        print(f"    模型预测失败: {e}, 使用 Elo fallback")
        return elo_fallback(home_team, away_team, team_df)

    return {
        "away_win": float(proba[0, 0]),
        "draw": float(proba[0, 1]),
        "home_win": float(proba[0, 2]),
    }


def elo_fallback(home_team, away_team, team_df):
    """Elo 基础概率预测（fallback）"""
    if home_team not in team_df.index or away_team not in team_df.index:
        return {"away_win": 0.33, "draw": 0.33, "home_win": 0.34}

    elo_h = float(team_df.loc[home_team, 'elo_rating'])
    elo_a = float(team_df.loc[away_team, 'elo_rating'])

    # Elo 预期得分
    exp_h = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))
    exp_a = 1.0 - exp_h

    # 转换成三分制概率
    # 平局概率与 Elo 差值负相关
    draw_prob = 0.25 - 0.08 * abs(exp_h - 0.5) / 0.5
    draw_prob = max(0.12, min(0.35, draw_prob))

    home_win = exp_h * (1.0 - draw_prob)
    away_win = exp_a * (1.0 - draw_prob)

    return {"away_win": away_win, "draw": draw_prob, "home_win": home_win}


# 测试预测
test_matches = [
    ("France", "Morocco"),
    ("Norway", "England"),
    ("Spain", "Belgium"),
    ("Argentina", "Switzerland"),
]

print("\n    比赛概率预测:")
for h, a in test_matches:
    proba = predict_match_proba(h, a, team_features, PIPELINE, FEATURE_NAMES, team_extra=team_extra_features)
    print(f"      {h} vs {a}: 主胜 {proba['home_win']:.3f}, "
          f"平 {proba['draw']:.3f}, 客胜 {proba['away_win']:.3f}")

# ============================================================
# 5. Monte Carlo 模拟
# ============================================================
print("\n" + "=" * 60)
print("[5] Monte Carlo 模拟 (N=10000)")
print("=" * 60)

N_SIMULATIONS = 10000
np.random.seed(42)
random.seed(42)

# 统计追踪
champion_counts = {}
semifinal_counts = {}
quarterfinal_counts = {}
semifinal_pairs = {}  # 四强组合

# 所有参赛队
all_teams_in_knockout = set()
for _, row in ro16_matches.iterrows():
    all_teams_in_knockout.add(row['home_team'])
    all_teams_in_knockout.add(row['away_team'])
# 加上 QF 中的队
for _, row in qf_matches.iterrows():
    all_teams_in_knockout.add(row['home_team'])
    all_teams_in_knockout.add(row['away_team'])

all_teams_in_knockout = sorted(all_teams_in_knockout)
for t in all_teams_in_knockout:
    champion_counts[t] = 0
    semifinal_counts[t] = 0
    quarterfinal_counts[t] = 0

# 缓存各队对阵各队的概率
proba_cache = {}

def get_cached_proba(home, away, team_extra=None):
    key = (home, away)
    if key not in proba_cache:
        proba_cache[key] = predict_match_proba(home, away, team_features, PIPELINE, FEATURE_NAMES, team_extra=team_extra)
    return proba_cache[key]


def simulate_knockout_match(home, away):
    """模拟一场淘汰赛，返回胜者"""
    proba = get_cached_proba(home, away, team_extra_features)

    r = random.random()
    if r < proba['home_win']:
        return home
    elif r < proba['home_win'] + proba['draw']:
        # 平局 → 加时 + 点球
        # 点球近似：略偏向强队
        elo_h = float(team_features.loc[home, 'elo_rating'])
        elo_a = float(team_features.loc[away, 'elo_rating'])
        pk_skill_h = elo_h / (elo_h + elo_a)
        if random.random() < pk_skill_h:
            return home
        else:
            return away
    else:
        return away


def simulate_tournament():
    """模拟一次完整世界杯（从当前 8 强开始）"""

    # QF
    qf_winners = {}
    for stage, h, a in REMAINING_MATCHES:
        winner = simulate_knockout_match(h, a)
        qf_winners[(h, a)] = winner

    # 确定四强
    fra_vs_mar_winner = qf_winners[("France", "Morocco")]
    nor_vs_eng_winner = qf_winners[("Norway", "England")]
    esp_vs_bel_winner = qf_winners[("Spain", "Belgium")]
    arg_vs_sui_winner = qf_winners[("Argentina", "Switzerland")]

    semifinalists = [fra_vs_mar_winner, nor_vs_eng_winner, esp_vs_bel_winner, arg_vs_sui_winner]

    # SF
    sf1_winner = simulate_knockout_match(fra_vs_mar_winner, nor_vs_eng_winner)
    sf2_winner = simulate_knockout_match(esp_vs_bel_winner, arg_vs_sui_winner)

    finalists = [sf1_winner, sf2_winner]

    # Final
    champion = simulate_knockout_match(sf1_winner, sf2_winner)

    return {
        'quarterfinalists': semifinalists,  # 实际是 QF 参赛队
        'semifinalists': semifinalists,
        'finalists': finalists,
        'champion': champion
    }


# 运行模拟
for sim_idx in range(N_SIMULATIONS):
    if (sim_idx + 1) % 1000 == 0:
        print(f"    模拟进度: {sim_idx+1}/{N_SIMULATIONS}")

    try:
        result = simulate_tournament()

        champ = result['champion']
        semis = result['semifinalists']

        champion_counts[champ] += 1
        for t in semis:
            semifinal_counts[t] += 1

        # 四强组合
        semis_key = tuple(sorted(semis))
        semifinal_pairs[semis_key] = semifinal_pairs.get(semis_key, 0) + 1

    except Exception as e:
        print(f"    Sim {sim_idx+1} 出错: {e}")
        continue

print(f"\n    完成 {N_SIMULATIONS} 次模拟")

# ============================================================
# 6. 结果统计
# ============================================================
print("\n" + "=" * 60)
print("[6] 结果统计")
print("=" * 60)

# 6a 夺冠概率
print("\n--- 夺冠概率 Top 10 ---")
champion_probs = [(t, c/N_SIMULATIONS*100) for t, c in champion_counts.items()]
champion_probs.sort(key=lambda x: x[1], reverse=True)

champion_df_rows = []
for rank, (team, prob) in enumerate(champion_probs[:20], 1):
    print(f"  {rank:2d}. {team:20s} {prob:5.2f}%")
    champion_df_rows.append({"rank": rank, "team": team, "champion_probability_pct": prob})

champion_df = pd.DataFrame(champion_df_rows)
champion_df.to_csv(OUT_DIR / "champion_probability.csv", index=False, encoding="utf-8-sig")
print(f"\n  已保存: champion_probability.csv")

# 6b 四强概率
print("\n--- 四强概率 Top 10 ---")
semifinal_probs = [(t, c/N_SIMULATIONS*100) for t, c in semifinal_counts.items()]
semifinal_probs.sort(key=lambda x: x[1], reverse=True)

semifinal_df_rows = []
for rank, (team, prob) in enumerate(semifinal_probs[:20], 1):
    print(f"  {rank:2d}. {team:20s} {prob:5.2f}%")
    semifinal_df_rows.append({"rank": rank, "team": team, "semifinal_probability_pct": prob})

semifinal_df = pd.DataFrame(semifinal_df_rows)
semifinal_df.to_csv(OUT_DIR / "semifinal_probability.csv", index=False, encoding="utf-8-sig")
print(f"\n  已保存: semifinal_probability.csv")

# 6c 四强组合频率
print("\n--- 四强组合频率 Top 10 ---")
semifinal_pairs_sorted = sorted(semifinal_pairs.items(), key=lambda x: x[1], reverse=True)
for rank, (combo, count) in enumerate(semifinal_pairs_sorted[:10], 1):
    prob = count / N_SIMULATIONS * 100
    teams_str = ", ".join(combo)
    print(f"  {rank:2d}. [{teams_str}] {prob:.2f}%")

# 6d 各队半决赛对阵概率
print("\n--- 半决赛对阵概率 ---")
# 上区 (France/Morocco winner vs Norway/England winner)
# 下区 (Spain/Belgium winner vs Argentina/Switzerland winner)

# 半决赛对阵统计
sf_matchups = {}

# 重新模拟，记录半决赛对阵
np.random.seed(42)
for _ in range(N_SIMULATIONS):
    qf_w = {}
    for stage, h, a in REMAINING_MATCHES:
        qf_w[(h, a)] = simulate_knockout_match(h, a)

    upper_a = qf_w[("France", "Morocco")]
    upper_b = qf_w[("Norway", "England")]
    lower_a = qf_w[("Spain", "Belgium")]
    lower_b = qf_w[("Argentina", "Switzerland")]

    sf_upper = tuple(sorted([upper_a, upper_b]))
    sf_lower = tuple(sorted([lower_a, lower_b]))

    sf_matchups["upper_" + "_vs_".join(sf_upper)] = sf_matchups.get("upper_" + "_vs_".join(sf_upper), 0) + 1
    sf_matchups["lower_" + "_vs_".join(sf_lower)] = sf_matchups.get("lower_" + "_vs_".join(sf_lower), 0) + 1

for matchup, count in sorted(sf_matchups.items(), key=lambda x: x[1], reverse=True):
    prob = count / N_SIMULATIONS * 100
    region = "上区" if matchup.startswith("upper") else "下区"
    teams = matchup.split("_", 1)[1].replace("_vs_", " vs ")
    print(f"  {region}: {teams} → {prob:.1f}%")

# ============================================================
# 7. 可视化
# ============================================================
print("\n[7] 生成可视化...")

# 7a 冠军概率直方图
fig, ax = plt.subplots(figsize=(14, 8))

top12_champ = champion_probs[:12]
teams = [t[0] for t in top12_champ]
probs = [t[1] for t in top12_champ]
colors = plt.cm.RdYlGn(np.array(probs) / max(probs))[::-1]

bars = ax.barh(range(len(teams)), probs, color=colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(teams)))
ax.set_yticklabels(teams, fontsize=11)
ax.set_xlabel("夺冠概率 (%)", fontsize=12)
ax.set_title("2026 世界杯夺冠概率 (Monte Carlo 模拟 10,000 次)", fontsize=14, fontweight='bold')

for i, (bar, prob) in enumerate(zip(bars, probs)):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{prob:.1f}%', va='center', fontsize=10, fontweight='bold')

ax.invert_yaxis()
ax.set_xlim(0, max(probs) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "champion_probability_histogram.png", dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: champion_probability_histogram.png")

# 7b 四强概率
fig, ax = plt.subplots(figsize=(14, 8))

top12_semi = semifinal_probs[:12]
teams_s = [t[0] for t in top12_semi]
probs_s = [t[1] for t in top12_semi]
colors_s = plt.cm.Blues(np.array(probs_s) / max(probs_s))

bars = ax.barh(range(len(teams_s)), probs_s, color=colors_s, edgecolor='white', height=0.7)
ax.set_yticks(range(len(teams_s)))
ax.set_yticklabels(teams_s, fontsize=11)
ax.set_xlabel("四强概率 (%)", fontsize=12)
ax.set_title("2026 世界杯四强概率 (Monte Carlo 模拟 10,000 次)", fontsize=14, fontweight='bold')

for i, (bar, prob) in enumerate(zip(bars, probs_s)):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{prob:.1f}%', va='center', fontsize=10, fontweight='bold')

ax.invert_yaxis()
ax.set_xlim(0, max(probs_s) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "semifinal_probability_histogram.png", dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: semifinal_probability_histogram.png")

# 7c 淘汰赛预测对阵图
print("\n  生成淘汰赛预测对阵图...")
fig, ax = plt.subplots(figsize=(20, 24))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')

# 颜色方案
COLOR_QF = '#4A90D9'
COLOR_SF = '#E67E22'
COLOR_F = '#E74C3C'
COLOR_CHAMP = '#F1C40F'
COLOR_BG = '#F8F9FA'
COLOR_LINE = '#7F8C8D'

# 布局坐标 (x, y) - 中心点
# 左列：QF 参赛队 (x=1), QF 结果 (x=4)
# 中列：SF (x=8), SF 结果 (x=11)
# 右列：Final (x=15), 冠军 (x=18)

layout = {
    # QF 左上
    "France_QF": (1, 24), "Morocco_QF": (1, 23),
    "Norway_QF": (1, 21), "England_QF": (1, 20),
    "Spain_QF": (1, 18), "Belgium_QF": (1, 17),
    "Argentina_QF": (1, 15), "Switzerland_QF": (1, 14),

    # QF winners
    "QF1_winner": (4, 23.5), "QF2_winner": (4, 20.5),
    "QF3_winner": (4, 17.5), "QF4_winner": (4, 14.5),

    # SF
    "SF1_team1": (8, 23), "SF1_team2": (8, 21),
    "SF2_team1": (8, 18), "SF2_team2": (8, 16),

    # SF winners
    "SF1_winner": (11, 22), "SF2_winner": (11, 17),

    # Final
    "Final_team1": (15, 20), "Final_team2": (15, 19),
    "Champion": (18, 19.5),
}

# 种子队伍（前 8 强）
top8_teams = ["France", "Morocco", "Norway", "England", "Spain", "Belgium", "Argentina", "Switzerland"]

# 使用模拟结果确定最可能的对阵
# 获取 QF 最可能胜者
qf_most_likely = {}
for stage, h, a in REMAINING_MATCHES:
    proba = get_cached_proba(h, a)
    if proba['home_win'] >= proba['away_win'] and proba['home_win'] >= proba['draw']:
        qf_most_likely[(h, a)] = h
    elif proba['away_win'] >= proba['home_win'] and proba['away_win'] >= proba['draw']:
        qf_most_likely[(h, a)] = a
    else:
        # 平局概率最高 → Elo 决定
        elo_h = float(team_features.loc[h, 'elo_rating'])
        elo_a = float(team_features.loc[a, 'elo_rating'])
        qf_most_likely[(h, a)] = h if elo_h > elo_a else a

# 预测结果
pred_QF1_w = qf_most_likely[("France", "Morocco")]
pred_QF2_w = qf_most_likely[("Norway", "England")]
pred_QF3_w = qf_most_likely[("Spain", "Belgium")]
pred_QF4_w = qf_most_likely[("Argentina", "Switzerland")]

pred_SF1 = simulate_knockout_match(pred_QF1_w, pred_QF2_w)
np.random.seed(42)
# Reset: actually let's use probabilities
proba_SF1 = get_cached_proba(pred_QF1_w, pred_QF2_w)
proba_SF2 = get_cached_proba(pred_QF3_w, pred_QF4_w)

pred_SF1_w = pred_QF1_w if proba_SF1['home_win'] + proba_SF1['draw']/2 > 0.5 else pred_QF2_w
pred_SF2_w = pred_QF3_w if proba_SF2['home_win'] + proba_SF2['draw']/2 > 0.5 else pred_QF4_w

# 绘制标题
ax.text(10, 27.5, "2026 世界杯淘汰赛预测对阵图", ha='center', va='center',
        fontsize=18, fontweight='bold', fontfamily='sans-serif')

# 绘制各轮标签
ax.text(1, 26.5, "四分之一决赛", ha='center', fontsize=13, fontweight='bold', color=COLOR_QF)
ax.text(8, 25.5, "半决赛", ha='center', fontsize=13, fontweight='bold', color=COLOR_SF)
ax.text(15, 22.5, "决赛", ha='center', fontsize=13, fontweight='bold', color=COLOR_F)
ax.text(18, 22.5, "冠军", ha='center', fontsize=13, fontweight='bold', color=COLOR_CHAMP)

def draw_team_box(ax, x, y, name, is_winner=False, color='#3498DB'):
    """画球队框"""
    box_w, box_h = 2.0, 0.6
    rect = mpatches.FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color if is_winner else '#ECF0F1',
                                    edgecolor=color if is_winner else '#BDC3C7',
                                    linewidth=2 if is_winner else 1)
    ax.add_patch(rect)
    ax.text(x, y, name, ha='center', va='center', fontsize=10,
            fontweight='bold' if is_winner else 'normal',
            color='white' if is_winner else '#2C3E50')

def draw_line(ax, x1, y1, x2, y2, color=COLOR_LINE, lw=1.5):
    """画连线"""
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=0)

# QF 参赛队
draw_team_box(ax, 1, 24, "France")
draw_team_box(ax, 1, 23, "Morocco")
draw_team_box(ax, 1, 21, "Norway")
draw_team_box(ax, 1, 20, "England")
draw_team_box(ax, 1, 18, "Spain")
draw_team_box(ax, 1, 17, "Belgium")
draw_team_box(ax, 1, 15, "Argentina")
draw_team_box(ax, 1, 14, "Switzerland")

# QF 连线
draw_line(ax, 2, 24, 2, 23)
draw_line(ax, 2, 24, 4, 23.5)
draw_line(ax, 2, 23, 4, 23.5)
draw_line(ax, 2, 21, 2, 20)
draw_line(ax, 2, 21, 4, 20.5)
draw_line(ax, 2, 20, 4, 20.5)
draw_line(ax, 2, 18, 2, 17)
draw_line(ax, 2, 18, 4, 17.5)
draw_line(ax, 2, 17, 4, 17.5)
draw_line(ax, 2, 15, 2, 14)
draw_line(ax, 2, 15, 4, 14.5)
draw_line(ax, 2, 14, 4, 14.5)

# QF 胜者
draw_team_box(ax, 4, 23.5, f"→ {pred_QF1_w}", is_winner=True, color=COLOR_QF)
draw_team_box(ax, 4, 20.5, f"→ {pred_QF2_w}", is_winner=True, color=COLOR_QF)
draw_team_box(ax, 4, 17.5, f"→ {pred_QF3_w}", is_winner=True, color=COLOR_QF)
draw_team_box(ax, 4, 14.5, f"→ {pred_QF4_w}", is_winner=True, color=COLOR_QF)

# QF → SF 连线
draw_line(ax, 5, 23.5, 6, 23)
draw_line(ax, 5, 23.5, 8, 23)
draw_line(ax, 5, 20.5, 6, 21)
draw_line(ax, 5, 20.5, 8, 21)
draw_line(ax, 5, 17.5, 6, 18)
draw_line(ax, 5, 17.5, 8, 18)
draw_line(ax, 5, 14.5, 6, 16)
draw_line(ax, 5, 14.5, 8, 16)

# SF 参赛队
draw_team_box(ax, 8, 23, f"{pred_QF1_w}", color=COLOR_SF)
draw_team_box(ax, 8, 21, f"{pred_QF2_w}", color=COLOR_SF)
draw_team_box(ax, 8, 18, f"{pred_QF3_w}", color=COLOR_SF)
draw_team_box(ax, 8, 16, f"{pred_QF4_w}", color=COLOR_SF)

# SF 连线
draw_line(ax, 9, 23, 9, 21)
draw_line(ax, 9, 23, 11, 22)
draw_line(ax, 9, 21, 11, 22)
draw_line(ax, 9, 18, 9, 16)
draw_line(ax, 9, 18, 11, 17)
draw_line(ax, 9, 16, 11, 17)

# SF 胜者
draw_team_box(ax, 11, 22, f"→ {pred_SF1_w}", is_winner=True, color=COLOR_SF)
draw_team_box(ax, 11, 17, f"→ {pred_SF2_w}", is_winner=True, color=COLOR_SF)

# SF → Final 连线
draw_line(ax, 12, 22, 13, 20)
draw_line(ax, 12, 22, 15, 20)
draw_line(ax, 12, 17, 13, 19)
draw_line(ax, 12, 17, 15, 19)

# Final
draw_team_box(ax, 15, 20, f"{pred_SF1_w}")
draw_team_box(ax, 15, 19, f"{pred_SF2_w}")

draw_line(ax, 16, 20, 16, 19)
draw_line(ax, 16, 20, 18, 19.5)
draw_line(ax, 16, 19, 18, 19.5)

# 冠军（最高概率的）
champ = champion_probs[0][0]
draw_team_box(ax, 18, 19.5, f"⭐ {champ}", is_winner=True, color=COLOR_CHAMP)

# 概率标注
# QF 概率
for stage, h, a in REMAINING_MATCHES:
    proba = get_cached_proba(h, a)
    status = "客胜" if proba['away_win'] > proba['home_win'] and proba['away_win'] > proba['draw'] else \
             "平局" if proba['draw'] > proba['home_win'] and proba['draw'] > proba['away_win'] else \
             "主胜"
    print(f"    {h} vs {a}: {status} ({max(proba.values())*100:.1f}%)")

plt.tight_layout()
plt.savefig(OUT_DIR / "tournament_bracket.png", dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: tournament_bracket.png")

# ============================================================
# 8. 仿真报告
# ============================================================
print("\n[8] 生成仿真报告...")

report = f"""# 2026 世界杯 Monte Carlo 仿真报告

## 模拟设置
- **模拟次数**: {N_SIMULATIONS:,}
- **预测模型**: LightGBM (Optuna 优化, CV LogLoss={PIPELINE.get('cv_logloss', 0.855):.3f})
- **仿真时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
- **当前阶段**: 四分之一决赛

## 当前赛程状态

### 已完成的比赛
- 小组赛: 48 场 ✓
- 32 强: 16 场 ✓
- 16 强: 8 场 ✓
- 8 强参赛队: {', '.join(sorted(remaining_8))}

### 剩余比赛
- 四分之一决赛 (4 场)
- 半决赛 (2 场)
- 决赛 (1 场)

## 夺冠概率 Top 10
| 排名 | 球队 | 夺冠概率 |
|------|------|---------|
"""
for rank, (team, prob) in enumerate(champion_probs[:10], 1):
    report += f"| {rank} | {team} | {prob:.2f}% |\n"

report += f"""
## 四强概率 Top 10
| 排名 | 球队 | 四强概率 |
|------|------|---------|
"""
for rank, (team, prob) in enumerate(semifinal_probs[:10], 1):
    report += f"| {rank} | {team} | {prob:.2f}% |\n"

report += f"""
## 四分之一决赛预测

| 对阵 | 主胜 | 平局 | 客胜 |
|------|------|------|------|
"""
for stage, h, a in REMAINING_MATCHES:
    proba = get_cached_proba(h, a)
    report += f"| {h} vs {a} | {proba['home_win']*100:.1f}% | {proba['draw']*100:.1f}% | {proba['away_win']*100:.1f}% |\n"

report += f"""
## 半决赛对阵概率

| 半区 | 对阵 | 概率 |
|------|------|------|
"""
for matchup, count in sorted(sf_matchups.items(), key=lambda x: x[1], reverse=True):
    prob = count / N_SIMULATIONS * 100
    region = "上区" if matchup.startswith("upper") else "下区"
    teams = matchup.split("_", 1)[1].replace("_vs_", " vs ")
    report += f"| {region} | {teams} | {prob:.1f}% |\n"

report += f"""
## 四强组合频率 Top 5
| 排名 | 组合 | 概率 |
|------|------|------|
"""
for rank, (combo, count) in enumerate(semifinal_pairs_sorted[:5], 1):
    prob = count / N_SIMULATIONS * 100
    teams_str = ", ".join(combo)
    report += f"| {rank} | {teams_str} | {prob:.2f}% |\n"

report += f"""
## 预测冠军
**{champion_probs[0][0]}** 是最大夺冠热门 (概率 {champion_probs[0][1]:.2f}%)。
"""
if len(champion_probs) > 1:
    report += f"\n**{champion_probs[1][0]}** 是第二热门 (概率 {champion_probs[1][1]:.2f}%)。\n"

report += """
## 方法论说明

1. **模型**: 使用历史数据训练的 LightGBM 模型，包含 Elo 评分、FIFA 排名、近期状态、世界杯历史表现等 72 维特征
2. **预测**: 对每场未进行的比赛预测 主胜/平/客胜 概率
3. **模拟**: 使用 Monte Carlo 方法，每次模拟从当前实际赛程出发
4. **淘汰赛**: 平局后按概率偏向强队进行点球模拟
5. **随机种子**: 42 (保证可复现)

## 输出文件
- `champion_probability.csv` — 各队夺冠概率
- `semifinal_probability.csv` — 各队四强概率
- `champion_probability_histogram.png` — 夺冠概率直方图
- `semifinal_probability_histogram.png` — 四强概率直方图
- `tournament_bracket.png` — 淘汰赛预测对阵图
"""

with open(OUT_DIR / "simulation_summary.md", "w", encoding="utf-8") as f:
    f.write(report)

print("  已保存: simulation_summary.md")
print("\n" + "=" * 60)
print("仿真完成!")
print("=" * 60)
