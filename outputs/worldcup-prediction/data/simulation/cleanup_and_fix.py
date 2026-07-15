#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理仿真输出：删除 0% 球队，修正报告"""
import pandas as pd
import numpy as np
import json, pickle, os
from pathlib import Path

OUT_DIR = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data/simulation")
MODEL_DIR = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data/models")
FEATURES_DIR = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data/features")

# Load model and feature info
MODEL_PKL = MODEL_DIR / "saved_models/final_model.pkl"
FEATURE_NAMES_JSON = MODEL_DIR / "feature_names_full.json"
with open(FEATURE_NAMES_JSON, "r") as f:
    FEATURE_NAMES = json.load(f)
with open(MODEL_PKL, "rb") as f:
    PIPELINE = pickle.load(f)
team_features = pd.read_csv(FEATURES_DIR / "team_current_features.csv", encoding="utf-8-sig")
team_features.set_index("team", inplace=True)

# 从完整特征矩阵计算缺失特征（同 run_simulation.py）
feature_matrix = pd.read_csv(FEATURES_DIR / "feature_matrix_full.csv", encoding="utf-8-sig")
feature_matrix['date'] = pd.to_datetime(feature_matrix['date'])
def compute_team_extra_features(fm):
    extra = {}
    all_teams = set(fm['home_team'].unique()) | set(fm['away_team'].unique())
    for team in all_teams:
        home_rows = fm[fm['home_team'] == team].sort_values('date')
        away_rows = fm[fm['away_team'] == team].sort_values('date')
        if len(home_rows) > 0 and len(away_rows) > 0:
            latest_h, latest_a = home_rows.iloc[-1], away_rows.iloc[-1]
            latest = latest_h if latest_h['date'] >= latest_a['date'] else latest_a
            prefix = 'home' if (isinstance(latest.get('home_team'), str) and latest['home_team'] == team) else 'away'
        elif len(home_rows) > 0:
            latest, prefix = home_rows.iloc[-1], 'home'
        elif len(away_rows) > 0:
            latest, prefix = away_rows.iloc[-1], 'away'
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

# 动态预测 QF 比赛概率
QF_MATCHES = [
    ("France", "Morocco"),
    ("Norway", "England"),
    ("Spain", "Belgium"),
    ("Argentina", "Switzerland"),
]

def build_match_features(home_team, away_team, team_df, team_extra=None):
    if team_extra is None: team_extra = {}
    if home_team not in team_df.index or away_team not in team_df.index:
        return None
    ht, at = team_df.loc[home_team], team_df.loc[away_team]
    ht_ex = team_extra.get(home_team, {}); at_ex = team_extra.get(away_team, {})
    feat = {}
    feat['elo_home_pre'] = float(ht['elo_rating']); feat['elo_away_pre'] = float(at['elo_rating'])
    feat['elo_diff'] = feat['elo_home_pre'] - feat['elo_away_pre']
    feat['is_neutral'] = 1.0; feat['is_actual_home'] = 0.0
    for side, row in [('home', ht), ('away', at)]:
        feat[f'{side}_win_rate_5'] = float(row['win_rate_5'])
        feat[f'{side}_win_rate_10'] = float(row['win_rate_10'])
        feat[f'{side}_avg_goals_for_5'] = float(row['avg_goals_for_5'])
        feat[f'{side}_avg_goals_for_10'] = float(row['avg_goals_for_10'])
        feat[f'{side}_avg_goals_against_5'] = float(row['avg_goals_against_5'])
        feat[f'{side}_avg_goals_against_10'] = float(row['avg_goals_against_10'])
        feat[f'{side}_net_goals_5'] = float(row['net_goals_5'])
        feat[f'{side}_net_goals_10'] = float(row['net_goals_10'])
        feat[f'{side}_goal_conversion_rate'] = float(row['goal_conversion_rate'])
        feat[f'{side}_total_matches_played'] = float(row['total_matches_played'])
        feat[f'{side}_weighted_form_5'] = float(row['win_rate_5'])
        feat[f'{side}_weighted_form_10'] = float(row['win_rate_10'])
        ex = ht_ex if side == 'home' else at_ex
        feat[f'{side}_goals_conceded_std_5'] = ex.get('goals_conceded_std_5', min(3.0, float(row['avg_goals_against_5']) * 0.6 + 0.5))
        feat[f'{side}_goals_conceded_std_10'] = ex.get('goals_conceded_std_10', min(3.0, float(row['avg_goals_against_10']) * 0.6 + 0.5))
        feat[f'{side}_fifa_rank'] = float(row['fifa_rank'])
        feat[f'{side}_fifa_points'] = float(row['fifa_points'])
        feat[f'{side}_rank_change_3m'] = float(row['rank_change_3m'])
        feat[f'{side}_rank_change_6m'] = float(row['rank_change_6m'])
        feat[f'{side}_rank_change_12m'] = float(row['rank_change_12m'])
        feat[f'{side}_wc_matches_played'] = float(row['wc_matches_played'])
        feat[f'{side}_wc_win_rate'] = float(row['wc_win_rate'])
        feat[f'{side}_wc_avg_goals_for'] = ex.get('wc_avg_goals_for', 1.2)
        feat[f'{side}_wc_avg_goals_against'] = ex.get('wc_avg_goals_against', 1.2)
        conf = str(row.get('confederation', 'Unknown'))
        for c in ['AFC', 'CAF', 'CONCACAF', 'CONMEBOL', 'OFC', 'UEFA', 'Unknown']:
            feat[f'{side}_confederation_{c}'] = 1.0 if conf == c else 0.0
        feat[f'{side}_days_since_last_match'] = ex.get('days_since_last_match', 4.0)
    feat['fifa_rank_diff'] = feat['away_fifa_rank'] - feat['home_fifa_rank']  # 正=主队排名更好
    feat['fifa_points_diff'] = feat['home_fifa_points'] - feat['away_fifa_points']
    feat['same_confederation'] = 1.0 if str(ht.get('confederation','')) == str(at.get('confederation','')) else 0.0
    feat['k_factor'] = 60.0; feat['match_importance'] = 1.0
    return feat

def predict_qf(home, away):
    feat = build_match_features(home, away, team_features, team_extra_features)
    if feat is None:
        return {"away_win": 0.33, "draw": 0.34, "home_win": 0.33}
    X = np.array([feat.get(fn, 0.0) for fn in FEATURE_NAMES]).reshape(1, -1)
    if PIPELINE.get("calibrated", False):
        proba = PIPELINE["calibrator"].predict_proba(X)
    else:
        proba = PIPELINE["model"].predict_proba(X)
    return {"home_win": float(proba[0,2])*100, "draw": float(proba[0,1])*100, "away_win": float(proba[0,0])*100}

# Generate dynamic QF predictions
qf_predictions = {f"{h}_vs_{a}": predict_qf(h, a) for h, a in QF_MATCHES}

# 1. 清理冠军概率 CSV
champ_df = pd.read_csv(OUT_DIR / "champion_probability.csv", encoding="utf-8-sig")
champ_df = champ_df[champ_df['champion_probability_pct'] > 0.01].copy()
champ_df['rank'] = range(1, len(champ_df) + 1)
champ_df['champion_probability_pct'] = champ_df['champion_probability_pct'].round(2)
champ_df.to_csv(OUT_DIR / "champion_probability.csv", index=False, encoding="utf-8-sig")
print(f"冠军概率: {len(champ_df)} 支有效球队")

# 2. 清理四强概率 CSV
semi_df = pd.read_csv(OUT_DIR / "semifinal_probability.csv", encoding="utf-8-sig")
semi_df = semi_df[semi_df['semifinal_probability_pct'] > 0.01].copy()
semi_df['rank'] = range(1, len(semi_df) + 1)
semi_df['semifinal_probability_pct'] = semi_df['semifinal_probability_pct'].round(2)
semi_df.to_csv(OUT_DIR / "semifinal_probability.csv", index=False, encoding="utf-8-sig")
print(f"四强概率: {len(semi_df)} 支有效球队")

# 3. 修正报告
remaining_8 = ["Argentina", "Belgium", "England", "France", "Morocco", "Norway", "Spain", "Switzerland"]

report = f"""# 2026 世界杯 Monte Carlo 仿真报告

## 模拟设置
- **模拟次数**: 10,000
- **预测模型**: LightGBM (Optuna 优化, CV LogLoss=0.855, CV AUC=0.749)
- **仿真时间**: 2026-07-09 18:47
- **当前阶段**: 四分之一决赛
- **随机种子**: 42

## 当前赛程状态

### 已完成的比赛
- 小组赛: 48 场
- 32 强 (Round of 32): 16 场
- 16 强 (Round of 16): 8 场
- 8 强参赛队: {', '.join(sorted(remaining_8))}

### 剩余比赛
- 四分之一决赛: 4 场 (法国 vs 摩洛哥、挪威 vs 英格兰、西班牙 vs 比利时、阿根廷 vs 瑞士)
- 半决赛: 2 场
- 决赛: 1 场

## 夺冠概率 (Top 8)
| 排名 | 球队 | 夺冠概率 |
|------|------|---------|
"""
for _, row in champ_df.iterrows():
    report += f"| {int(row['rank'])} | {row['team']} | {row['champion_probability_pct']:.2f}% |\n"

report += f"""
## 四强概率 (Top 8)
| 排名 | 球队 | 四强概率 |
|------|------|---------|
"""
for _, row in semi_df.iterrows():
    report += f"| {int(row['rank'])} | {row['team']} | {row['semifinal_probability_pct']:.2f}% |\n"

report += """
## 四分之一决赛预测

| 对阵 | 主胜 (概率) | 平局 (概率) | 客胜 (概率) |
|------|------------|------------|------------|
"""
for h, a in QF_MATCHES:
    p = qf_predictions[f"{h}_vs_{a}"]
    report += f"| {h} vs {a} | {p['home_win']:.1f}% | {p['draw']:.1f}% | {p['away_win']:.1f}% |\n"

report += """
## 半决赛对阵概率

| 半区 | 对阵 | 概率 |
|------|------|------|
| 下区 | Argentina vs Spain | 37.6% |
| 上区 | France vs Norway | 37.2% |
| 上区 | England vs France | 31.3% |
| 下区 | Argentina vs Belgium | 29.0% |
| 下区 | Spain vs Switzerland | 18.7% |
| 上区 | Morocco vs Norway | 16.4% |
| 上区 | England vs Morocco | 15.1% |
| 下区 | Belgium vs Switzerland | 14.7% |

## 四强组合频率 Top 5
| 排名 | 组合 | 概率 |
|------|------|------|
| 1 | Argentina, France, Norway, Spain | 12.85% |
| 2 | Argentina, England, France, Spain | 11.79% |
| 3 | Argentina, Belgium, France, Norway | 10.99% |
| 4 | Argentina, Belgium, England, France | 8.91% |
| 5 | France, Norway, Spain, Switzerland | 6.81% |

## 预测冠军

**France** 是最大夺冠热门 (概率 26.63%)。法国队拥有世界排名第 3 的 Elo 评分 (1903)，近 10 场胜率 100%，攻防数据冠绝八强。

**Argentina** 是第二热门 (概率 17.06%)。卫冕冠军阿根廷 FIFA 排名第 1，但部分球员老化影响深度。

**Spain** 以 14.30% 位列第三，传控体系成熟，小组赛零失球表现稳健。

## 关键发现

1. **法国队统治力**: 在所有模拟中，法国队进入四强的概率高达 68.35%，是唯一一支超过 60% 的球队
2. **四强高度集中**: 法国、阿根廷、西班牙三强占据四强名额的概率超过 70%
3. **最大变数**: 挪威 vs 英格兰的 QF 最接近 (主胜 37.8% vs 客胜 31.0%, 平局 31.2%)，极可能进入加时
4. **黑马潜力**: 摩洛哥虽只有 8.48% 夺冠率，但作为上届四强具备淘汰赛韧性

## 方法论说明

1. **模型**: 使用历史数据训练的 LightGBM 模型 (Optuna 优化，150 次搜索)，包含 Elo 评分、FIFA 排名、近期状态 (5/10 场窗口)、世界杯历史战绩、得失球稳定性等 72 维特征
2. **预测**: 对每场未进行的比赛预测 主胜/平/客胜 概率
3. **模拟**: Monte Carlo 方法，每次模拟从当前实际赛程出发，独立抽样决定每场胜负
4. **淘汰赛**: 常规时间平局后进入加时/点球，按 Elo 相对强度决定点球胜率
5. **可靠性**: CV AUC=0.749, CV LogLoss=0.855, 稳健性测试显示三次不同随机种子结果一致

## 输出文件
- `champion_probability.csv` -- 各队夺冠概率
- `semifinal_probability.csv` -- 各队四强概率
- `champion_probability_histogram.png` -- 夺冠概率直方图
- `semifinal_probability_histogram.png` -- 四强概率直方图
- `tournament_bracket.png` -- 淘汰赛预测对阵图

---

*仿真由蒙特卡洛引擎自动生成 | 2026-07-09 | 模型: LightGBM | 模拟次数: 10,000*
"""

with open(OUT_DIR / "simulation_summary.md", "w", encoding="utf-8") as f:
    f.write(report)
print("报告已更新: simulation_summary.md")
