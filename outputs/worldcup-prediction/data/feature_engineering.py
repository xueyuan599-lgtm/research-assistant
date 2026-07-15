#!/usr/bin/env python3
"""
世界杯预测 — 特征工程流水线
===========================
输入：football_data/ 下的原始数据
输出：
  1. features/feature_matrix.csv          — 每场比赛的特征向量
  2. features/feature_description.md      — 特征说明文档
  3. features/team_current_features.csv   — 各球队最新特征值

特征类别：
  - 动态 Elo 评分（含主场优势调整、赛事类型 K 因子）
  - 近期状态特征（最近 5/10 场：胜率、进球、失球、净胜球、加权表现）
  - 攻防能力指标（进攻强度、防守稳定性、进球转换率）
  - FIFA 排名特征（当前排名、变化趋势、排名差距）
  - 大赛经验（世界杯历史战绩）
  - 其他特征（比赛间隔、主客场、所属洲际、赛事重要性）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 路径配置
# ============================================================
DATA_DIR = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/football_data")
OUT_DIR  = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ============================================================
# 0b. 球队名称映射（FIFA 排名中的命名与比赛数据不同）
# ============================================================
TEAM_NAME_MAPPING = {
    'United States': 'USA',
    'Iran': 'IR Iran',
    'Czech Republic': 'Czechia',
    'DR Congo': 'Congo DR',
    'Cape Verde': 'Cabo Verde',
    'Ivory Coast': "Côte d'Ivoire",
    'South Korea': 'Korea Republic',
    'Turkey': 'Türkiye',
    'China': 'China PR',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'São Tomé and Príncipe': 'São Tomé and Príncipe',  # same
}


def normalize_team_name(name: str) -> str:
    """统一球队名称，使比赛数据与 FIFA 排名数据匹配"""
    return TEAM_NAME_MAPPING.get(name, name)


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("世界杯预测特征工程流水线")
print("=" * 60)

# 1a. 国际比赛数据（宽格式）
matches = pd.read_csv(DATA_DIR / "international_matches_2021_2026.csv")
matches['date'] = pd.to_datetime(matches['date'])
matches = matches.sort_values('date').reset_index(drop=True)

# 1b. 长格式比赛数据（每行=一支球队的视角）
matches_long = pd.read_csv(DATA_DIR / "matches_long_format_2021_2026.csv")
matches_long['date'] = pd.to_datetime(matches_long['date'])
matches_long = matches_long.sort_values('date').reset_index(drop=True)

# 1c. FIFA 排名数据
rankings = pd.read_csv(DATA_DIR / "fifa_rankings_2021_2026.csv")
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
rankings = rankings.sort_values('rank_date').reset_index(drop=True)

# 1d. WC2026 比赛
wc2026 = pd.read_csv(DATA_DIR / "wc2026_matches.csv")
wc2026['date'] = pd.to_datetime(wc2026['date'])

# 1e. WC2026 小组积分榜
wc_standings = pd.read_csv(DATA_DIR / "wc2026_group_standings.csv")

print(f"国际比赛记录:           {len(matches):>6,}")
print(f"长格式比赛记录:         {len(matches_long):>6,}")
print(f"FIFA 排名记录:          {len(rankings):>6,}")
print(f"WC2026 比赛:            {len(wc2026):>6,}")
print(f"WC2026 小组记录:        {len(wc_standings):>6,}")

# ============================================================
# 2. 赛事类型 → K 因子映射
# ============================================================
# 世界杯正赛
WC_TOURNAMENTS = {'FIFA World Cup'}
# 洲际杯赛
CONTINENTAL_CUPS = {
    'UEFA Euro', 'Copa America', 'African Cup of Nations', 'AFC Asian Cup',
    'Gold Cup', 'Oceania Nations Cup', 'UEFA Nations League',
    'CONCACAF Nations League', 'Arab Cup', 'COSAFA Cup', 'AFF Championship',
    'SAFF Cup', 'ASEAN Championship', 'Pacific Games', 'Gulf Cup',
    'CAFA Nations Cup', 'EAFF Championship', 'Baltic Cup', 'King\'s Cup',
    'Kirin Cup', 'CONIFA World Football Cup', 'CONIFA Africa Football Cup',
    'CONIFA Asia Cup', 'Muratti Vase', 'Mauritius Four Nations Cup',
    'Indian Ocean Island Games', 'Mapinduzi Cup', 'Navruz Cup',
    'Merdeka Tournament', 'Intercontinental Cup', 'Marianas Cup',
    'CONIFA South America Football Cup', 'Tri-Nations Series',
    'Tri Nation Tournament', 'Three Nations Cup', 'Tri-Nations Cup',
    'South Asian Super Cup', 'Morocco Capital of African Football',
    'Al Ain International Cup', 'Jordan International Tournament',
    'Canadian Shield', 'Mukuru 4 Nations', 'Diamond Jubilee International Football Tournament',
    'FIFA Series', 'CONCACAF Series', 'Unity Cup', 'Outrigger Challenge Cup',
    'Soccer Ashes', 'MSG Prime Minister\'s Cup', 'Island Games',
    'CONMEBOL UEFA Cup of Champions',
}

# 需要匹配的洲际锦标赛关键词
CONTINENTAL_KEYWORDS = [
    'Nations League', 'Euro', 'Copa América', 'Copa America',
    'Asian Cup', 'Gold Cup', 'African Cup', 'Nations Cup',
    'Copa', 'Cup of Nations', 'Championship'
]


def get_k_factor(tournament: str) -> int:
    """根据赛事类型返回 K 因子"""
    t = str(tournament).strip()

    # 世界杯
    if t in WC_TOURNAMENTS:
        return 40

    # 洲际杯赛
    if t in CONTINENTAL_CUPS:
        return 30

    # 检查关键词
    t_lower = t.lower()
    for kw in CONTINENTAL_KEYWORDS:
        if kw.lower() in t_lower:
            return 30

    # 世界杯预选赛 / 各洲预选赛
    if 'qualification' in t_lower or 'qualifier' in t_lower:
        return 20

    # 友谊赛
    if 'friendly' in t_lower:
        return 10

    # 其他（默认）
    return 20


# ============================================================
# 3. 球队所属洲际映射（从 FIFA 排名数据提取）
# ============================================================
def build_team_confederation_map(rankings_df: pd.DataFrame) -> dict:
    """从排名数据构建球队→洲际映射，取最新记录"""
    latest_date = rankings_df['rank_date'].max()
    latest = rankings_df[rankings_df['rank_date'] == latest_date]
    rank_conf = dict(zip(latest['team_name'], latest['confederation']))

    # 添加名称映射后的别名
    for match_name, rank_name in TEAM_NAME_MAPPING.items():
        if rank_name in rank_conf:
            rank_conf[match_name] = rank_conf[rank_name]

    return rank_conf


team_conf_map = build_team_confederation_map(rankings)

# 补充未覆盖的球队
all_teams_in_data = set(matches['home_team'].unique()) | set(matches['away_team'].unique())
for team in all_teams_in_data:
    if team not in team_conf_map:
        normalized = normalize_team_name(team)
        if normalized != team and normalized in team_conf_map:
            team_conf_map[team] = team_conf_map[normalized]
        else:
            # 尝试从排名历史中查找
            team_rankings = rankings[rankings['team_name'] == normalized]
            if len(team_rankings) > 0:
                team_conf_map[team] = team_rankings.iloc[-1]['confederation']
            else:
                team_conf_map[team] = 'Unknown'


# ============================================================
# 4. FIFA 排名处理函数
# ============================================================
def get_rank_before_date(team: str, d: pd.Timestamp, rank_df: pd.DataFrame) -> dict:
    """获取某支球队在某日期前的最新 FIFA 排名"""
    rank_name = normalize_team_name(team)
    team_ranks = rank_df[rank_df['team_name'] == rank_name]
    team_ranks = team_ranks[team_ranks['rank_date'] <= d]
    if len(team_ranks) == 0:
        return {'rank': 999, 'prev_rank': 999, 'total_points': 0, 'prev_points': 0}

    latest = team_ranks.iloc[-1]
    return {
        'rank': latest['rank'],
        'prev_rank': latest['prev_rank'],
        'total_points': latest['total_points'],
        'prev_points': latest['prev_points'],
    }


def get_rank_changes(team: str, d: pd.Timestamp, rank_df: pd.DataFrame) -> dict:
    """计算 3个月/6个月/12个月的排名变化"""
    rank_name = normalize_team_name(team)
    team_ranks = rank_df[rank_df['team_name'] == rank_name]
    team_ranks = team_ranks[team_ranks['rank_date'] <= d]
    if len(team_ranks) == 0:
        return {'rank_change_3m': 0, 'rank_change_6m': 0, 'rank_change_12m': 0}

    current_rank = team_ranks.iloc[-1]['rank']

    # 找到各时间窗口前的排名
    results = {'rank_change_3m': 0, 'rank_change_6m': 0, 'rank_change_12m': 0}
    windows = {'rank_change_3m': 90, 'rank_change_6m': 180, 'rank_change_12m': 365}

    for key, days in windows.items():
        target_date = d - timedelta(days=days)
        past_ranks = team_ranks[team_ranks['rank_date'] >= target_date]
        if len(past_ranks) > 0:
            # 取目标日期后最近的排名
            past_rank = past_ranks.iloc[0]
            results[key] = past_rank['prev_rank'] - current_rank  # 正 = 上升，负 = 下降
        else:
            results[key] = 0

    return results


# ============================================================
# 5. 动态 Elo 评分系统
# ============================================================
class EloRatingSystem:
    """世界杯预测 Elo 评分系统"""

    INITIAL_RATING = 1500.0
    HOME_ADVANTAGE = 100.0  # 主场优势等价 Elo 分

    def __init__(self):
        self.ratings = {}  # team -> rating

    def get_rating(self, team: str) -> float:
        """获取球队当前 Elo 分（首次出现初始化为 1500）"""
        if team not in self.ratings:
            self.ratings[team] = self.INITIAL_RATING
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
        """计算 A 队的预期得分"""
        diff = rating_b - rating_a - home_advantage
        return 1.0 / (1.0 + 10.0 ** (diff / 400.0))

    def update(self, team_a: str, team_b: str, score_a: float, score_b: float,
               k_factor: int, is_neutral: bool) -> tuple:
        """
        更新两队 Elo 评分
        返回更新后的两队评分
        """
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)

        # 主场优势调整
        home_adv = 0 if is_neutral else self.HOME_ADVANTAGE

        # 计算预期得分
        ea = self.expected_score(ra, rb, home_adv)
        eb = self.expected_score(rb, ra, -home_adv)

        # 实际结果（胜负平转换）
        if score_a > score_b:
            sa, sb = 1.0, 0.0
        elif score_a < score_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        # 目标差调整（进球差越大，调整越大，上限 1.5 倍）
        goal_diff = abs(score_a - score_b)
        goal_margin = min(goal_diff, 3) * 0.1667 + 1.0  # 1.0 ~ 1.5

        # 更新评分
        new_ra = ra + k_factor * goal_margin * (sa - ea)
        new_rb = rb + k_factor * goal_margin * (sb - eb)

        self.ratings[team_a] = new_ra
        self.ratings[team_b] = new_rb

        return new_ra, new_rb


# ============================================================
# 6. 近期状态计算
# ============================================================
def compute_rolling_features(long_df: pd.DataFrame) -> dict:
    """
    为每支球队、每个比赛日计算滚动特征。
    返回: {(team, date) -> {feature_dict}}
    使用长格式数据，确保只使用历史信息（不包含当前比赛）。
    """
    print("\n计算近期状态特征...")

    feature_store = {}  # {(team, date): features}

    # 按球队分组，每支球队内部按日期排序
    for team, group in long_df.groupby('team'):
        group = group.sort_values('date').reset_index(drop=True)

        if len(group) == 0:
            continue

        # 为每场比赛计算特征（只使用该场比赛之前的数据）
        for i in range(len(group)):
            row = group.iloc[i]
            current_date = row['date']

            # 获取之前的比赛（严格早于当前日期）
            past_matches = group.iloc[:i]

            if len(past_matches) == 0:
                # 第一场比赛，无历史数据
                feature_store[(team, current_date)] = {
                    'win_rate_5': 0.0,
                    'win_rate_10': 0.0,
                    'avg_goals_for_5': 0.0,
                    'avg_goals_for_10': 0.0,
                    'avg_goals_against_5': 0.0,
                    'avg_goals_against_10': 0.0,
                    'net_goals_5': 0.0,
                    'net_goals_10': 0.0,
                    'weighted_form_5': 0.0,
                    'weighted_form_10': 0.0,
                    'goals_conceded_std_5': 0.0,
                    'goals_conceded_std_10': 0.0,
                    'goal_conversion_rate': 0.0,
                    'total_matches_played': 0,
                }
                continue

            # 计算最近 N 场（使用过去所有场次，但特征名保持 5/10）
            def _rolling_stats(matches_subset, n: int):
                if len(matches_subset) == 0:
                    return {
                        'win_rate': 0.0,
                        'avg_for': 0.0,
                        'avg_against': 0.0,
                        'net': 0.0,
                        'weighted_form': 0.0,
                        'conceded_std': 0.0,
                        'conv_rate': 0.0,
                    }

                sub = matches_subset.tail(n)
                n_actual = len(sub)

                # 胜负判定（从球队视角：进球多则胜）
                wins = (sub['goals_for'] > sub['goals_against']).sum()
                draws = (sub['goals_for'] == sub['goals_against']).sum()

                win_rate = wins / n_actual
                avg_for = sub['goals_for'].mean()
                avg_against = sub['goals_against'].mean()
                net = avg_for - avg_against

                # 加权表现（近期权重更高）
                # 使用指数权重：最近一场权重最高
                weights = np.exp(np.linspace(0, 1, n_actual))
                weights = weights / weights.sum()
                weighted_points = weights * (wins / n_actual * 3 + draws / n_actual * 1)
                weighted_form = weighted_points.sum() * n_actual  # 缩放回可比较范围

                # 防守稳定性（失球标准差）
                conceded_std = sub['goals_against'].std()

                # 进球转换率 = 进球数 / (进球+失球)
                total_gf = sub['goals_for'].sum()
                total_ga = sub['goals_against'].sum()
                conv_rate = total_gf / (total_gf + total_ga) if (total_gf + total_ga) > 0 else 0.0

                return {
                    'win_rate': win_rate,
                    'avg_for': avg_for,
                    'avg_against': avg_against,
                    'net': net,
                    'weighted_form': weighted_form,
                    'conceded_std': conceded_std,
                    'conv_rate': conv_rate,
                }

            stats_5 = _rolling_stats(past_matches, 5)
            stats_10 = _rolling_stats(past_matches, 10)
            all_stats = _rolling_stats(past_matches, len(past_matches))

            feature_store[(team, current_date)] = {
                'win_rate_5': stats_5['win_rate'],
                'win_rate_10': stats_10['win_rate'],
                'avg_goals_for_5': stats_5['avg_for'],
                'avg_goals_for_10': stats_10['avg_for'],
                'avg_goals_against_5': stats_5['avg_against'],
                'avg_goals_against_10': stats_10['avg_against'],
                'net_goals_5': stats_5['net'],
                'net_goals_10': stats_10['net'],
                'weighted_form_5': stats_5['weighted_form'],
                'weighted_form_10': stats_10['weighted_form'],
                'goals_conceded_std_5': stats_5['conceded_std'],
                'goals_conceded_std_10': stats_10['conceded_std'],
                'goal_conversion_rate': all_stats['conv_rate'],
                'total_matches_played': len(past_matches),
            }

    print(f"  已计算 {len(feature_store)} 个球队-日期特征点")
    return feature_store


# ============================================================
# 7. 比赛间隔计算
# ============================================================
def compute_match_intervals(long_df: pd.DataFrame) -> dict:
    """计算每支球队每场比赛距离上一场比赛的天数"""
    print("\n计算比赛间隔...")

    intervals = {}  # {(team, date): days_since_last_match}

    for team, group in long_df.groupby('team'):
        group = group.sort_values('date').reset_index(drop=True)

        for i in range(len(group)):
            row = group.iloc[i]
            current_date = row['date']

            if i == 0:
                intervals[(team, current_date)] = 999  # 无历史，设为较大值
            else:
                prev_date = group.iloc[i - 1]['date']
                delta = (current_date - prev_date).days
                intervals[(team, current_date)] = delta

    return intervals


# ============================================================
# 8. 世界杯历史经验特征
# ============================================================
def compute_world_cup_experience(long_df: pd.DataFrame, matches_df: pd.DataFrame) -> dict:
    """
    计算球队世界杯经验特征。
    由于数据只覆盖 2021-2026，我们使用数据中的世界杯比赛来计算经验。
    """
    print("\n计算世界杯经验特征...")

    # 找出数据中的所有世界杯比赛
    wc_matches = matches_df[matches_df['tournament'].str.contains('FIFA World Cup', na=False)]
    # 排除预选赛
    wc_finals = wc_matches[~wc_matches['tournament'].str.contains('qualification', case=False, na=False)].copy()

    exp_features = {}  # {team: {wc_appearances, wc_matches_played, wc_wins, ...}}

    for team in long_df['team'].unique():
        # 球队为参赛方且比赛为世界杯正赛
        team_wc = long_df[
            (long_df['team'] == team) &
            (long_df['tournament'].str.contains('FIFA World Cup', na=False)) &
            (~long_df['tournament'].str.contains('qualification', case=False, na=False))
            ]

        wc_matches_played = len(team_wc)
        wc_wins = len(team_wc[team_wc['goals_for'] > team_wc['goals_against']])
        wc_draws = len(team_wc[team_wc['goals_for'] == team_wc['goals_against']])
        wc_losses = len(team_wc[team_wc['goals_for'] < team_wc['goals_against']])
        wc_goals_for = team_wc['goals_for'].sum()
        wc_goals_against = team_wc['goals_against'].sum()

        if wc_matches_played > 0:
            wc_win_rate = wc_wins / wc_matches_played
            wc_avg_goals_for = wc_goals_for / wc_matches_played
            wc_avg_goals_against = wc_goals_against / wc_matches_played
        else:
            wc_win_rate = 0.0
            wc_avg_goals_for = 0.0
            wc_avg_goals_against = 0.0

        exp_features[team] = {
            'wc_matches_played': wc_matches_played,
            'wc_wins': wc_wins,
            'wc_draws': wc_draws,
            'wc_losses': wc_losses,
            'wc_win_rate': wc_win_rate,
            'wc_avg_goals_for': wc_avg_goals_for,
            'wc_avg_goals_against': wc_avg_goals_against,
        }

    print(f"  已计算 {len(exp_features)} 支球队的世界杯经验")
    return exp_features


# ============================================================
# 9. 主特征构建流水线
# ============================================================
def build_feature_matrix(
    matches_df: pd.DataFrame,
    long_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    wc2026_df: pd.DataFrame,
    elo_system: EloRatingSystem,
    rolling_features: dict,
    match_intervals: dict,
    wc_exp: dict,
) -> pd.DataFrame:
    """
    构建特征矩阵的核心流水线。
    按时间顺序处理每场比赛，计算所有特征。
    """
    print("\n" + "=" * 60)
    print("构建特征矩阵...")
    print("=" * 60)

    # 合并所有比赛（包括 WC2026）
    all_matches = []

    # 处理国际比赛
    for _, row in matches_df.iterrows():
        all_matches.append({
            'date': row['date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': float(row['home_score']) if pd.notna(row['home_score']) else np.nan,
            'away_score': float(row['away_score']) if pd.notna(row['away_score']) else np.nan,
            'tournament': row['tournament'],
            'neutral': bool(row['neutral']),
            'country': row['country'],
            'source': 'historical',
        })

    # 处理 WC2026 比赛
    for _, row in wc2026_df.iterrows():
        all_matches.append({
            'date': row['date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': float(row['home_score']) if pd.notna(row['home_score']) else np.nan,
            'away_score': float(row['away_score']) if pd.notna(row['away_score']) else np.nan,
            'tournament': 'FIFA World Cup 2026',
            'neutral': True,  # WC 在中立场地
            'country': row['venue_country'],
            'source': 'wc2026',
        })

    matches_all = pd.DataFrame(all_matches)
    matches_all = matches_all.sort_values(['date', 'source']).reset_index(drop=True)

    # 特征存储
    features_list = []

    # 跟踪已处理的球队-比赛日（避免重复处理长格式查找）
    processed_teams = set()

    total = len(matches_all)
    for idx, match in matches_all.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"  处理进度: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)")

        date = match['date']
        home = match['home_team']
        away = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
        tournament = match['tournament']
        is_neutral = match['neutral']
        country = match['country']
        source = match['source']

        # ---- 特征 1: Elo 评分 ----
        elo_home = elo_system.get_rating(home)
        elo_away = elo_system.get_rating(away)
        elo_diff = elo_home - elo_away

        # ---- 特征 2: 近期状态 ----
        def _get_rolling(team, dt):
            """获取球队在日期前的滚动特征。尝试精确匹配，否则用最近的。"""
            key = (team, dt)
            if key in rolling_features:
                return rolling_features[key]

            # 查找该球队最近的特征点
            team_features = [(t, f) for (t, d), f in rolling_features.items() if t == team and d < dt]
            if team_features:
                # 按日期排序，取最近
                team_features.sort(key=lambda x: x[0][1], reverse=True)
                return team_features[0][1]

            # 无历史数据
            return {
                'win_rate_5': 0.0, 'win_rate_10': 0.0,
                'avg_goals_for_5': 0.0, 'avg_goals_for_10': 0.0,
                'avg_goals_against_5': 0.0, 'avg_goals_against_10': 0.0,
                'net_goals_5': 0.0, 'net_goals_10': 0.0,
                'weighted_form_5': 0.0, 'weighted_form_10': 0.0,
                'goals_conceded_std_5': 0.0, 'goals_conceded_std_10': 0.0,
                'goal_conversion_rate': 0.0,
                'total_matches_played': 0,
            }

        home_form = _get_rolling(home, date)
        away_form = _get_rolling(away, date)

        # ---- 特征 3: 比赛间隔 ----
        def _get_interval(team, dt):
            key = (team, dt)
            if key in match_intervals:
                return match_intervals[key]
            # 找最近间隔
            team_intervals = [(t, v) for (t, d), v in match_intervals.items() if t == team and d < dt]
            if team_intervals:
                team_intervals.sort(key=lambda x: x[0][1], reverse=True)
                return team_intervals[0][1]
            return 999

        home_interval = _get_interval(home, date)
        away_interval = _get_interval(away, date)

        # ---- 特征 4: FIFA 排名 ----
        home_rank_info = get_rank_before_date(home, date, rankings_df)
        away_rank_info = get_rank_before_date(away, date, rankings_df)

        home_rank = home_rank_info['rank']
        away_rank = away_rank_info['rank']
        rank_diff = away_rank - home_rank  # 负 = 主队排名更高

        home_rank_change = get_rank_changes(home, date, rankings_df)
        away_rank_change = get_rank_changes(away, date, rankings_df)

        home_points = home_rank_info['total_points']
        away_points = away_rank_info['total_points']
        points_diff = home_points - away_points

        # ---- 特征 5: 所属洲际 ----
        home_conf = team_conf_map.get(home, 'Unknown')
        away_conf = team_conf_map.get(away, 'Unknown')
        same_conf = 1.0 if home_conf == away_conf else 0.0

        # ---- 特征 6: 世界杯经验 ----
        home_wc = wc_exp.get(home, {
            'wc_matches_played': 0, 'wc_wins': 0, 'wc_draws': 0, 'wc_losses': 0,
            'wc_win_rate': 0.0, 'wc_avg_goals_for': 0.0, 'wc_avg_goals_against': 0.0
        })
        away_wc = wc_exp.get(away, {
            'wc_matches_played': 0, 'wc_wins': 0, 'wc_draws': 0, 'wc_losses': 0,
            'wc_win_rate': 0.0, 'wc_avg_goals_for': 0.0, 'wc_avg_goals_against': 0.0
        })

        # ---- 特征 7: 赛事特征 ----
        k_factor = get_k_factor(tournament)
        is_home_flag = 0.0 if is_neutral else 1.0

        # 是否为真正的主场（在国家比赛即为真主场）
        is_actual_home = 1.0 if (not is_neutral and home == country) else 0.0

        # ---- 出价特征向量 ----
        feature_row = {
            # 标识
            'date': date,
            'home_team': home,
            'away_team': away,
            'tournament': tournament,
            'source': source,

            # 目标变量
            'home_score': home_score,
            'away_score': away_score,
            'home_win': 1.0 if (not np.isnan(home_score) and not np.isnan(away_score) and home_score > away_score) else
                        0.0 if (not np.isnan(home_score) and not np.isnan(away_score)) else np.nan,
            'draw': 1.0 if (not np.isnan(home_score) and not np.isnan(away_score) and home_score == away_score) else
                    0.0 if (not np.isnan(home_score) and not np.isnan(away_score)) else np.nan,

            # Elo 评分
            'elo_home_pre': round(elo_home, 1),
            'elo_away_pre': round(elo_away, 1),
            'elo_diff': round(elo_diff, 1),

            # 主场优势
            'is_neutral': 1.0 if is_neutral else 0.0,
            'is_actual_home': is_actual_home,

            # 近期状态（主队）
            'home_win_rate_5': home_form['win_rate_5'],
            'home_win_rate_10': home_form['win_rate_10'],
            'home_avg_goals_for_5': home_form['avg_goals_for_5'],
            'home_avg_goals_for_10': home_form['avg_goals_for_10'],
            'home_avg_goals_against_5': home_form['avg_goals_against_5'],
            'home_avg_goals_against_10': home_form['avg_goals_against_10'],
            'home_net_goals_5': home_form['net_goals_5'],
            'home_net_goals_10': home_form['net_goals_10'],
            'home_weighted_form_5': home_form['weighted_form_5'],
            'home_weighted_form_10': home_form['weighted_form_10'],
            'home_goals_conceded_std_5': home_form['goals_conceded_std_5'],
            'home_goals_conceded_std_10': home_form['goals_conceded_std_10'],
            'home_goal_conversion_rate': home_form['goal_conversion_rate'],
            'home_total_matches_played': home_form['total_matches_played'],

            # 近期状态（客队）
            'away_win_rate_5': away_form['win_rate_5'],
            'away_win_rate_10': away_form['win_rate_10'],
            'away_avg_goals_for_5': away_form['avg_goals_for_5'],
            'away_avg_goals_for_10': away_form['avg_goals_for_10'],
            'away_avg_goals_against_5': away_form['avg_goals_against_5'],
            'away_avg_goals_against_10': away_form['avg_goals_against_10'],
            'away_net_goals_5': away_form['net_goals_5'],
            'away_net_goals_10': away_form['net_goals_10'],
            'away_weighted_form_5': away_form['weighted_form_5'],
            'away_weighted_form_10': away_form['weighted_form_10'],
            'away_goals_conceded_std_5': away_form['goals_conceded_std_5'],
            'away_goals_conceded_std_10': away_form['goals_conceded_std_10'],
            'away_goal_conversion_rate': away_form['goal_conversion_rate'],
            'away_total_matches_played': away_form['total_matches_played'],

            # FIFA 排名特征
            'home_fifa_rank': home_rank,
            'away_fifa_rank': away_rank,
            'fifa_rank_diff': rank_diff,
            'home_fifa_points': round(home_points, 1),
            'away_fifa_points': round(away_points, 1),
            'fifa_points_diff': round(points_diff, 1),
            'home_rank_change_3m': home_rank_change['rank_change_3m'],
            'home_rank_change_6m': home_rank_change['rank_change_6m'],
            'home_rank_change_12m': home_rank_change['rank_change_12m'],
            'away_rank_change_3m': away_rank_change['rank_change_3m'],
            'away_rank_change_6m': away_rank_change['rank_change_6m'],
            'away_rank_change_12m': away_rank_change['rank_change_12m'],

            # 世界杯经验
            'home_wc_matches_played': home_wc['wc_matches_played'],
            'home_wc_win_rate': home_wc['wc_win_rate'],
            'home_wc_avg_goals_for': home_wc['wc_avg_goals_for'],
            'home_wc_avg_goals_against': home_wc['wc_avg_goals_against'],
            'away_wc_matches_played': away_wc['wc_matches_played'],
            'away_wc_win_rate': away_wc['wc_win_rate'],
            'away_wc_avg_goals_for': away_wc['wc_avg_goals_for'],
            'away_wc_avg_goals_against': away_wc['wc_avg_goals_against'],

            # 洲际特征
            'home_confederation': home_conf,
            'away_confederation': away_conf,
            'same_confederation': same_conf,

            # 比赛间隔
            'home_days_since_last_match': home_interval,
            'away_days_since_last_match': away_interval,

            # 赛事特征
            'k_factor': k_factor,
            'match_importance': k_factor / 40.0,  # 归一化到 [0.25, 1.0]
        }

        features_list.append(feature_row)

        # ---- 更新 Elo 评分（仅对已知结果的比赛） ----
        if not np.isnan(home_score) and not np.isnan(away_score):
            elo_system.update(
                home, away,
                home_score, away_score,
                k_factor, is_neutral
            )

    # 转为 DataFrame
    feature_df = pd.DataFrame(features_list)
    return feature_df


# ============================================================
# 10. 生成球队当前特征快照
# ============================================================
def build_team_current_features(feature_df: pd.DataFrame, elo_system: EloRatingSystem) -> pd.DataFrame:
    """生成每支球队当前（最新）的特征快照"""
    print("\n生成球队当前特征快照...")

    # 找到所有球队
    home_teams = set(feature_df['home_team'].unique())
    away_teams = set(feature_df['away_team'].unique())
    all_teams = home_teams | away_teams

    team_features = []

    for team in sorted(all_teams):
        # 球队作为主队和客队的最近比赛
        home_matches = feature_df[feature_df['home_team'] == team].sort_values('date')
        away_matches = feature_df[feature_df['away_team'] == team].sort_values('date')

        # 取最新的比赛特征
        latest_home = home_matches.iloc[-1] if len(home_matches) > 0 else None
        latest_away = away_matches.iloc[-1] if len(away_matches) > 0 else None

        # 综合最新的特征
        if latest_home is not None and latest_away is not None:
            latest = latest_home if latest_home['date'] >= latest_away['date'] else latest_away
        elif latest_home is not None:
            latest = latest_home
        elif latest_away is not None:
            latest = latest_away
        else:
            continue

        # 获取 Elo 评分
        elo = elo_system.get_rating(team)
        conf = team_conf_map.get(team, 'Unknown')

        # 从最新特征中提取球队特征
        is_home_side = (latest['home_team'] == team)
        prefix = 'home' if is_home_side else 'away'

        team_features.append({
            'team': team,
            'confederation': conf,
            'elo_rating': round(elo, 1),
            'latest_match_date': latest['date'],
            'win_rate_5': latest[f'{prefix}_win_rate_5'],
            'win_rate_10': latest[f'{prefix}_win_rate_10'],
            'avg_goals_for_5': latest[f'{prefix}_avg_goals_for_5'],
            'avg_goals_for_10': latest[f'{prefix}_avg_goals_for_10'],
            'avg_goals_against_5': latest[f'{prefix}_avg_goals_against_5'],
            'avg_goals_against_10': latest[f'{prefix}_avg_goals_against_10'],
            'net_goals_5': latest[f'{prefix}_net_goals_5'],
            'net_goals_10': latest[f'{prefix}_net_goals_10'],
            'goal_conversion_rate': latest[f'{prefix}_goal_conversion_rate'],
            'fifa_rank': latest[f'{prefix}_fifa_rank'],
            'fifa_points': latest[f'{prefix}_fifa_points'],
            'rank_change_3m': latest[f'{prefix}_rank_change_3m'],
            'rank_change_6m': latest[f'{prefix}_rank_change_6m'],
            'rank_change_12m': latest[f'{prefix}_rank_change_12m'],
            'wc_matches_played': latest[f'{prefix}_wc_matches_played'],
            'wc_win_rate': latest[f'{prefix}_wc_win_rate'],
            'total_matches_played': latest[f'{prefix}_total_matches_played'],
        })

    team_df = pd.DataFrame(team_features)
    return team_df


# ============================================================
# 11. 特征描述生成
# ============================================================
def generate_feature_description() -> str:
    """生成特征说明文档"""
    return """# 世界杯预测特征说明文档

## 概述
本特征矩阵基于 2021-2026 年国际足球比赛数据构建，包含 6 大类特征，
用于预测足球比赛结果。特征全部基于比赛前的历史信息构建，无前瞻偏差。

## 特征类别

### 1. 动态 Elo 评分（3 个特征）
| 特征名 | 说明 | 范围 |
|--------|------|------|
| `elo_home_pre` | 主队赛前 Elo 评分 | ~1300-1900 |
| `elo_away_pre` | 客队赛前 Elo 评分 | ~1300-1900 |
| `elo_diff` | Elo 评分差值（主-客） | ~-600~600 |

**计算方法**：
- 所有球队初始 Elo = 1500
- K 因子：世界杯=40, 洲际杯=30, 预选赛=20, 友谊赛=10
- 主场优势调整 +100 Elo
- 进球差调整：进球差每多 1 球，调整幅度增加 16.67%（上限 1.5 倍）
- 更新公式：新评分 = 旧评分 + K * 进球差调整 * (实际得分 - 预期得分)

### 2. 近期状态特征（28 个特征）
| 特征名 | 说明 | 范围 |
|--------|------|------|
| `{side}_win_rate_5/10` | 最近 5/10 场胜率 | 0~1 |
| `{side}_avg_goals_for_5/10` | 最近 5/10 场均进球 | 0~5 |
| `{side}_avg_goals_against_5/10` | 最近 5/10 场均失球 | 0~5 |
| `{side}_net_goals_5/10` | 最近 5/10 场净胜球 | -5~5 |
| `{side}_weighted_form_5/10` | 加权表现（指数衰减权重） | 0~3 |
| `{side}_goal_conversion_rate` | 历史进球转换率 | 0~1 |
| `{side}_total_matches_played` | 总比赛场次 | 0~200+ |

**计算方法**：
- {side} 为 home（主队）或 away（客队）
- 严格使用赛前历史数据，无前瞻偏差
- 加权表现使用指数权重：越近的比赛权重越高

### 3. FIFA 排名特征（12 个特征）
| 特征名 | 说明 | 范围 |
|--------|------|------|
| `{side}_fifa_rank` | FIFA 当前排名 | 1~211 |
| `{side}_fifa_points` | FIFA 当前积分 | 0~2000 |
| `fifa_rank_diff` | 排名差（客-主） | -210~210 |
| `fifa_points_diff` | 积分差（主-客） | -2000~2000 |
| `{side}_rank_change_3m` | 3 个月排名变化 | -100~100 |
| `{side}_rank_change_6m` | 6 个月排名变化 | -150~150 |
| `{side}_rank_change_12m` | 12 个月排名变化 | -200~200 |

**计算方法**：
- 使用比赛日之前的最新 FIFA 排名
- 排名变化 = 过去排名 - 当前排名（正 = 上升）
- 3/6/12 个月窗口分别计算

### 4. 世界杯经验特征（8 个特征）
| 特征名 | 说明 | 范围 |
|--------|------|------|
| `{side}_wc_matches_played` | 世界杯正赛出场次数（2021-26） | 0~20+ |
| `{side}_wc_win_rate` | 世界杯胜率 | 0~1 |
| `{side}_wc_avg_goals_for` | 世界杯场均进球 | 0~4 |
| `{side}_wc_avg_goals_against` | 世界杯场均失球 | 0~4 |

### 5. 洲际与对阵特征（5 个特征）
| 特征名 | 说明 | 范围 |
|--------|------|------|
| `home_confederation` | 主队所属洲际 | 分类 |
| `away_confederation` | 客队所属洲际 | 分类 |
| `same_confederation` | 是否同洲际 | 0/1 |
| `is_neutral` | 是否中立场地 | 0/1 |
| `is_actual_home` | 是否真正主场 | 0/1 |

### 6. 比赛节奏特征（4 个特征）
| 特征名 | 说明 | 范围 |
|--------|------|------|
| `home_days_since_last_match` | 主队距上场比赛天数 | 0~999 |
| `away_days_since_last_match` | 客队距上场比赛天数 | 0~999 |
| `k_factor` | 赛事 K 因子 | 10/20/30/40 |
| `match_importance` | 比赛重要性（归一化） | 0.25~1.0 |

## 目标变量
| 变量 | 说明 | 值 |
|------|------|----|
| `home_score` | 主队进球数 | 整数 |
| `away_score` | 客队进球数 | 整数 |
| `home_win` | 主队胜（含平负为 0） | 0/1/NaN |
| `draw` | 平局 | 0/1/NaN |

## 数据规模
- 总样本数：所有 2021-2026 国际比赛 + WC2026
- 特征维度：60+ 数值特征 + 分类特征
- WC2026 未来比赛的目标变量为 NaN

## 使用建议
1. **特征选择**：优先使用 Elo 相关特征、近期状态、FIFA 排名
2. **处理分类**：洲际特征建议 one-hot 编码
3. **缺失处理**：WC2026 未来比赛的 home_score/away_score 为 NaN（预测目标）
4. **时间序列**：严格按时间划分训练/测试集，避免前瞻偏差
"""


# ============================================================
# 12. 主执行流程
# ============================================================
def main():
    print("=" * 60)
    print("阶段 1/4: 初始化 Elo 评分系统")
    elo_system = EloRatingSystem()

    print("\n阶段 2/4: 预计算球队级特征")

    # 近期状态特征
    rolling_features = compute_rolling_features(matches_long)

    # 比赛间隔
    match_intervals = compute_match_intervals(matches_long)

    # 世界杯经验
    wc_exp = compute_world_cup_experience(matches_long, matches)

    # 合并长格式中的 WC2026 数据（如有）
    # 将 WC2026 比赛转换为长格式用于特征计算
    wc_long_rows = []
    for _, row in wc2026.iterrows():
        # Skip matches with no teams
        if pd.isna(row['home_team']) or pd.isna(row['away_team']):
            continue
        # Home perspective
        wc_long_rows.append({
            'date': row['date'],
            'team': row['home_team'],
            'opponent': row['away_team'],
            'goals_for': float(row['home_score']) if pd.notna(row['home_score']) else np.nan,
            'goals_against': float(row['away_score']) if pd.notna(row['away_score']) else np.nan,
            'match_type': 'home',
            'tournament': 'FIFA World Cup 2026',
            'country': row['venue_country'],
        })
        # Away perspective
        wc_long_rows.append({
            'date': row['date'],
            'team': row['away_team'],
            'opponent': row['home_team'],
            'goals_for': float(row['away_score']) if pd.notna(row['away_score']) else np.nan,
            'goals_against': float(row['home_score']) if pd.notna(row['home_score']) else np.nan,
            'match_type': 'away',
            'tournament': 'FIFA World Cup 2026',
            'country': row['venue_country'],
        })

    if wc_long_rows:
        wc_long_df = pd.DataFrame(wc_long_rows)
        wc_long_df['date'] = pd.to_datetime(wc_long_df['date'])
        # 合并到长格式（WC2026 已完成比赛也可用于后续特征计算）
        matches_long_combined = pd.concat([matches_long, wc_long_df], ignore_index=True)
        matches_long_combined = matches_long_combined.sort_values('date').reset_index(drop=True)

        # 重新计算滚动特征（含 WC2026 已完赛场次）
        rolling_features_wc = compute_rolling_features(matches_long_combined)
        match_intervals_wc = compute_match_intervals(matches_long_combined)
        wc_exp_wc = compute_world_cup_experience(matches_long_combined, matches)
    else:
        rolling_features_wc = rolling_features
        match_intervals_wc = match_intervals
        wc_exp_wc = wc_exp
        matches_long_combined = matches_long

    print(f"\n阶段 3/4: 构建特征矩阵")
    feature_df = build_feature_matrix(
        matches, matches_long_combined, rankings,
        wc2026, elo_system,
        rolling_features_wc, match_intervals_wc, wc_exp_wc
    )

    print(f"\n阶段 4/4: 保存输出")

    # 1. 特征矩阵（不含分类列，用于建模）
    feature_matrix_cols = [c for c in feature_df.columns
                           if c not in ['date', 'home_team', 'away_team', 'tournament',
                                        'source', 'home_confederation', 'away_confederation']]
    # 保存完整版
    feature_df.to_csv(OUT_DIR / "feature_matrix_full.csv", index=False, encoding='utf-8-sig')

    # 保存数值版（只含数值特征 + 目标 + 标识）
    numeric_df = feature_df[feature_matrix_cols].copy()
    numeric_df.to_csv(OUT_DIR / "feature_matrix.csv", index=False, encoding='utf-8-sig')

    # 2. 球队当前特征快照
    team_features = build_team_current_features(feature_df, elo_system)
    team_features.to_csv(OUT_DIR / "team_current_features.csv", index=False, encoding='utf-8-sig')

    # 3. 特征描述文档
    description = generate_feature_description()
    with open(OUT_DIR / "feature_description.md", 'w', encoding='utf-8') as f:
        f.write(description)

    # ============================================================
    # 输出摘要
    # ============================================================
    print("\n" + "=" * 60)
    print("特征工程完成！")
    print("=" * 60)

    print(f"\n特征矩阵形状: {feature_df.shape}")
    print(f"数值特征矩阵形状: {numeric_df.shape}")
    print(f"球队当前特征: {team_features.shape}")

    # 统计特征数量
    id_cols = ['date', 'home_team', 'away_team', 'tournament', 'source',
               'home_confederation', 'away_confederation']
    target_cols = ['home_score', 'away_score', 'home_win', 'draw']
    feature_cols = [c for c in numeric_df.columns if c not in target_cols]

    print(f"\n特征统计:")
    print(f"  总特征数量（含标识和目标）: {len(numeric_df.columns)}")
    print(f"  纯预测特征: {len(feature_cols)}")
    print(f"  WC2026 比赛: {len(feature_df[feature_df['source'] == 'wc2026'])} 场")
    print(f"  WC2026 待预测: {feature_df[(feature_df['source'] == 'wc2026') & (feature_df['home_score'].isna())].shape[0]} 场")

    print(f"\n特征列表:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:3d}. {col}")

    # 展示球队当前特征
    print(f"\n球队当前特征 TOP 20:")
    team_sorted = team_features.sort_values('elo_rating', ascending=False)
    for _, row in team_sorted.head(20).iterrows():
        print(f"  {row['team']:25s} | Elo={row['elo_rating']:5.0f} | Rank={int(row['fifa_rank']):3d} | "
              f"WinRate5={row['win_rate_5']:.2f} | WC Played={int(row['wc_matches_played']):2d}")

    print(f"\n输出文件:")
    print(f"  特征矩阵:      {OUT_DIR / 'feature_matrix.csv'}")
    print(f"  完整特征矩阵:  {OUT_DIR / 'feature_matrix_full.csv'}")
    print(f"  球队当前特征:  {OUT_DIR / 'team_current_features.csv'}")
    print(f"  特征说明:      {OUT_DIR / 'feature_description.md'}")

    return feature_df, team_features


if __name__ == '__main__':
    result = main()
