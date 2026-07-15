#!/usr/bin/env python3
"""
国际足球数据集 (2021-2026) 完整数据探查脚本
输出：数据质量报告 + 可视化图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 配置
# ============================================================
DATA_DIR = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/football_data")
OUT_DIR  = Path("E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data/exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 清除 matplotlib 字体缓存
import matplotlib.font_manager as fm
cache_dir = matplotlib.get_cachedir()
for f in os.listdir(cache_dir):
    if f.startswith('fontlist'):
        os.remove(os.path.join(cache_dir, f))
fm._load_fontmanager(try_read_cache=False)

# 中文字体设置 — 强制使用 Noto Sans SC / Microsoft YaHei
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 配色方案
C_DRAW     = '#8E8E8E'
C_HOME_WIN = '#2E86AB'
C_AWAY_WIN = '#A23B72'
C_TOTAL    = '#F18F01'
C_MAIN     = '#2E86AB'

np.random.seed(42)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("国际足球数据集 (2021-2026) 数据探查报告")
print("=" * 60)

# 1a. 国际比赛数据
matches = pd.read_csv(DATA_DIR / "international_matches_2021_2026.csv")
matches['date'] = pd.to_datetime(matches['date'])
matches['total_goals'] = matches['home_score'] + matches['away_score']

# 1b. 长格式比赛数据
matches_long = pd.read_csv(DATA_DIR / "matches_long_format_2021_2026.csv")
matches_long['date'] = pd.to_datetime(matches_long['date'])

# 1c. FIFA 排名
rankings = pd.read_csv(DATA_DIR / "fifa_rankings_2021_2026.csv")
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])

# 1d. WC2026 比赛
wc2026 = pd.read_csv(DATA_DIR / "wc2026_matches.csv")
wc2026['date'] = pd.to_datetime(wc2026['date'])
if 'home_score' in wc2026.columns:
    wc2026['total_goals'] = pd.to_numeric(wc2026['home_score'], errors='coerce') + \
                            pd.to_numeric(wc2026['away_score'], errors='coerce')

# 1e. WC2026 小组积分榜
wc_standings = pd.read_csv(DATA_DIR / "wc2026_group_standings.csv")

print(f"\n{'='*60}")
print(f"已加载 {len(matches):,} 条国际比赛记录")
print(f"已加载 {len(matches_long):,} 条长格式比赛记录")
print(f"已加载 {len(rankings):,} 条 FIFA 排名记录")
print(f"已加载 {len(wc2026)} 条 2026 世界杯比赛记录")
print(f"已加载 {len(wc_standings)} 条小组积分记录")
print(f"{'='*60}")

# ============================================================
# 2. 基本统计量
# ============================================================
# 时间覆盖
date_min = matches['date'].min()
date_max = matches['date'].max()
date_span = (date_max - date_min).days
unique_teams = pd.unique(pd.concat([matches['home_team'], matches['away_team']]))
unique_tournaments = sorted(matches['tournament'].unique())
unique_countries = matches['country'].nunique()

print(f"\n--- 基本统计 ---")
print(f"时间范围: {date_min.date()} 至 {date_max.date()} (共 {date_span} 天)")
print(f"涉及球队数: {len(unique_teams)}")
print(f"赛事类型数: {len(unique_tournaments)}")
print(f"举办国家数: {unique_countries}")

# 赛事类型分布
tournament_counts = matches['tournament'].value_counts()
print(f"\n--- 赛事类型 Top 20 ---")
for t, c in tournament_counts.head(20).items():
    print(f"  {t}: {c} 场 ({c/len(matches)*100:.1f}%)")

# 主客场胜率
home_wins = ((matches['home_score'] > matches['away_score'])).sum()
away_wins = ((matches['home_score'] < matches['away_score'])).sum()
draws     = ((matches['home_score'] == matches['away_score'])).sum()
total     = len(matches)
print(f"\n--- 主客场胜率 ---")
print(f"  主队胜: {home_wins} ({home_wins/total*100:.1f}%)")
print(f"  客队胜: {away_wins} ({away_wins/total*100:.1f}%)")
print(f"  平局:   {draws} ({draws/total*100:.1f}%)")

# 进球分布
goals_stats = matches['total_goals'].describe()
print(f"\n--- 进球分布 (总进球/场) ---")
print(f"  均值: {goals_stats['mean']:.3f}")
print(f"  标准差: {goals_stats['std']:.3f}")
print(f"  中位数: {goals_stats['50%']:.1f}")
print(f"  最大值: {int(goals_stats['max'])}")
print(f"  0 进球比例: {(matches['total_goals']==0).mean()*100:.1f}%")

# 中立场地分析
neutral_matches = matches[matches['neutral'] == True]
non_neutral = matches[matches['neutral'] == False]
print(f"\n--- 中立场地 ---")
print(f"  中立场地比赛: {len(neutral_matches)} ({len(neutral_matches)/total*100:.1f}%)")
if len(neutral_matches) > 0:
    n_home_win = (neutral_matches['home_score'] > neutral_matches['away_score']).mean()
    print(f"  中立场地主队胜率: {n_home_win*100:.1f}%")

# ============================================================
# 3. 缺失值分析
# ============================================================
print(f"\n--- 缺失值分析 ---")
for name, df in [("international_matches", matches),
                  ("fifa_rankings", rankings),
                  ("wc2026_matches", wc2026),
                  ("wc2026_standings", wc_standings),
                  ("matches_long", matches_long)]:
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    if missing.sum() > 0:
        print(f"\n  {name}:")
        for col in missing[missing > 0].index:
            print(f"    {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
    else:
        print(f"  {name}: 无缺失值 [OK]")

# ============================================================
# 4. 时间分布分析
# ============================================================
matches['year_month'] = matches['date'].dt.to_period('M')
monthly_counts = matches.groupby('year_month').size()

print(f"\n--- 月度比赛分布 ---")
print(f"  最高月比赛数: {monthly_counts.max()} ({monthly_counts.idxmax()})")
print(f"  最低月比赛数: {monthly_counts.min()} ({monthly_counts.idxmin()})")
print(f"  月均值: {monthly_counts.mean():.1f} ± {monthly_counts.std():.1f}")

# 年度统计
matches['year'] = matches['date'].dt.year
yearly = matches.groupby('year').agg(
    matches=('date', 'count'),
    avg_goals=('total_goals', 'mean'),
    teams=('home_team', lambda x: len(pd.unique(pd.concat([
        matches.loc[x.index, 'home_team'],
        matches.loc[x.index, 'away_team']
    ]))))
)
print(f"\n--- 年度比赛统计 ---")
print(yearly.to_string())

# ============================================================
# 5. 潜在数据泄漏检查
# ============================================================
print(f"\n--- 数据泄漏检查 ---")

# 5a. 检查未来日期（超过当前时间的数据）
now = pd.Timestamp.now()
future_matches = matches[matches['date'] > now]
print(f"\n  5a. 未来比赛: {len(future_matches)} 场")
if len(future_matches) > 0:
    print(f"      最晚比赛: {future_matches['date'].max().date()}")
    print(f"      赛事类型: {future_matches['tournament'].value_counts().to_dict()}")

# 5b. 检查分数异常值
for score_col in ['home_score', 'away_score']:
    extreme = matches[matches[score_col] >= 10]
    if len(extreme) > 0:
        print(f"\n  5b. 极端比分 ({score_col} >= 10): {len(extreme)} 场")
        for _, row in extreme.head(10).iterrows():
            print(f"      {row['date'].date()} {row['home_team']} {int(row['home_score'])}-{int(row['away_score'])} {row['away_team']} ({row['tournament']})")

# 5c. 主客场重复（同一场比赛在两种角色出现）
home_away_check = matches.groupby(['date', 'home_team', 'away_team']).size()
duplicates = home_away_check[home_away_check > 1]
print(f"\n  5c. 完全重复的比赛记录: {len(duplicates)} 组")

# 5d. 中立场地但 country=home_team 逻辑检查 (当home_team和country不同时才是真中立的)
if 'neutral' in matches.columns:
    pseudo_neutral = matches[(matches['neutral'] == True) & (matches['home_team'] == matches['country'])]
    print(f"\n  5d. 标记为中立但实际在主队国比赛: {len(pseudo_neutral)} 场")

# 5e. Fifa Rankings 检查：同一日期同一球队是否有重复
rank_dup = rankings.groupby(['rank_date', 'team_name']).size()
rank_dups = rank_dup[rank_dup > 1]
print(f"\n  5e. 排名数据同一日期-球队重复: {len(rank_dups)} 条")

# 5f. WC2026 已完成的比赛 vs 积分榜一致性
completed_wc = wc2026[wc2026['status'] == 'Completed']
if len(completed_wc) > 0:
    print(f"\n  5f. WC2026 已完成比赛: {len(completed_wc)} 场")

# ============================================================
# 6. 异常值检测
# ============================================================
print(f"\n--- 异常值检测 ---")

# 6a. IQR 方法检测进球异常值
Q1 = matches['total_goals'].quantile(0.25)
Q3 = matches['total_goals'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR
outliers = matches[matches['total_goals'] > upper_bound]
print(f"\n  6a. 总进球异常值 (>{upper_bound:.1f}): {len(outliers)} 场")
if len(outliers) > 0:
    print(f"      最高进球: {int(outliers['total_goals'].max())}")
    for _, row in outliers.head(10).iterrows():
        print(f"      {row['date'].date()} {row['home_team']} {int(row['home_score'])}-{int(row['away_score'])} {row['away_team']} ({row['tournament']})")

# 6b. FIFA 排名异常值
rankings['rank_change'] = rankings['prev_rank'] - rankings['rank']
large_jump = rankings[rankings['rank_change'].abs() >= 50]
print(f"\n  6b. FIFA 排名大幅变动 (>=50位): {len(large_jump)} 条")
if len(large_jump) > 0:
    for _, row in large_jump.head(10).iterrows():
        print(f"      {row['rank_date'].date()} {row['team_name']}: {int(row['prev_rank'])}→{int(row['rank'])} ({int(row['rank_change']):+d})")

# 6c. 同一天同一球队是否打多场比赛
team_match_dates = matches_long.groupby(['date', 'team']).size()
multi_matches = team_match_dates[team_match_dates > 1]
print(f"\n  6c. 同一天同一球队多场比赛: {len(multi_matches)} 例")
if len(multi_matches) > 0:
    print(f"      例如:")
    for (d, t), cnt in multi_matches.head(5).items():
        print(f"      {d.date()} {t}: {cnt} 场")

# ============================================================
# 7. 世界杯预测相关分析
# ============================================================
print(f"\n--- 世界杯预测相关分析 ---")

# 7a. 世界杯历史战绩（含预选赛）
wc_matches = matches[matches['tournament'].str.contains('World Cup', case=False, na=False)]
print(f"\n  世界杯相关比赛: {len(wc_matches)} 场")
if len(wc_matches) > 0:
    # 确定哪些球队将参加2026世界杯
    wc2026_teams = wc_standings['team'].unique()
    wc_teams_in_data = [t for t in wc2026_teams if t in unique_teams]
    missing_teams = [t for t in wc2026_teams if t not in unique_teams]
    print(f"  2026世界杯参赛队在历史数据中出现的数量: {len(wc_teams_in_data)}/{len(wc2026_teams)}")
    if missing_teams:
        print(f"  未找到的球队: {missing_teams}")

# 7b. 各洲际球队实力分析
conf_mapping = {}
for _, row in rankings.iterrows():
    conf_mapping[row['team_name']] = row['confederation']
# 取最新排名
latest_rank_date = rankings['rank_date'].max()
latest_rankings = rankings[rankings['rank_date'] == latest_rank_date]
conf_stats = latest_rankings.groupby('confederation').agg(
    teams=('team_name', 'count'),
    avg_rank=('rank', 'mean'),
    median_rank=('rank', 'median'),
    avg_points=('total_points', 'mean')
).round(1)
print(f"\n  各洲际排名统计 (截至 {latest_rank_date.date()}):")
print(conf_stats.to_string())

# 7c. 主队优势量化 - 分赛事类型
tournament_home_adv = matches.groupby('tournament').apply(
    lambda x: (x['home_score'] > x['away_score']).mean()
).sort_values(ascending=False)
print(f"\n  各赛事主队胜率 Top 10:")
for t, p in tournament_home_adv.head(10).items():
    n = len(matches[matches['tournament'] == t])
    print(f"    {t}: {p*100:.1f}% ({n} 场)")

# 7d. 进球趋势（随时间）
matches['year_qtr'] = matches['date'].dt.to_period('Q')
quarterly_goals = matches.groupby('year_qtr')['total_goals'].mean()
print(f"\n  进球趋势（按季度均值）:")
print(f"    最高: {quarterly_goals.max():.3f} ({quarterly_goals.idxmax()})")
print(f"    最低: {quarterly_goals.min():.3f} ({quarterly_goals.idxmin()})")

# ============================================================
# 8. 生成可视化图表
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print(f"{'='*60}")

# --- 图1: 进球分布直方图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左: 总进球分布
ax = axes[0]
bins = range(0, int(matches['total_goals'].max()) + 2)
ax.hist(matches['total_goals'], bins=bins, color=C_MAIN, edgecolor='white', alpha=0.85)
ax.set_xlabel('Total Goals per Match', fontsize=12)
ax.set_ylabel('Number of Matches', fontsize=12)
ax.set_title('Distribution of Total Goals per Match\n(2021–2026)', fontsize=13, fontweight='bold')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
mean_goals = matches['total_goals'].mean()
ax.axvline(mean_goals, color=C_TOTAL, linestyle='--', linewidth=1.5,
           label=f'Mean = {mean_goals:.2f}')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 右: 主客进球对比
ax = axes[1]
ax.hist(matches['home_score'], bins=range(0, 12), alpha=0.7, color=C_HOME_WIN,
        edgecolor='white', label='Home Goals', density=True)
ax.hist(matches['away_score'], bins=range(0, 12), alpha=0.7, color=C_AWAY_WIN,
        edgecolor='white', label='Away Goals', density=True)
ax.set_xlabel('Goals', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Home vs Away Goal Distribution\n(2021–2026)', fontsize=13, fontweight='bold')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "goals_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ goals_distribution.png")

# --- 图2: 赛事类型饼图 ---
fig, ax = plt.subplots(figsize=(10, 8))
tournament_top = tournament_counts.head(8)
others_count = tournament_counts[8:].sum()
plot_data = pd.concat([tournament_top, pd.Series({'Others': others_count})])

# 生成配色
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4', '#E94F37', '#5D576B', '#8E8E8E']

wedges, texts, autotexts = ax.pie(
    plot_data.values,
    labels=None,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors[:len(plot_data)],
    pctdistance=0.85,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)
)

# 添加图例
legend_labels = [f'{name}  ({count:,})' for name, count in plot_data.items()]
ax.legend(wedges, legend_labels, title="Tournament Type",
          loc="center left", bbox_to_anchor=(-0.1, 0, 0.5, 1), fontsize=9, title_fontsize=11)

ax.set_title('Match Distribution by Tournament Type\n(2021–2026)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(OUT_DIR / "tournament_pie.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ tournament_pie.png")

# --- 图3: 时间覆盖线图 ---
fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2, 1]})

# 上: 月度比赛数量
ax = axes[0]
monthly_idx = [str(m) for m in monthly_counts.index]
ax.fill_between(range(len(monthly_counts)), monthly_counts.values, alpha=0.3, color=C_MAIN)
ax.plot(range(len(monthly_counts)), monthly_counts.values, color=C_MAIN, linewidth=1.2, marker='', alpha=0.9)
# 滚动均值
rolling_mean = monthly_counts.rolling(12, min_periods=1).mean()
ax.plot(range(len(monthly_counts)), rolling_mean.values, color=C_TOTAL, linewidth=2,
        linestyle='--', label=f'12-month MA ({rolling_mean.mean():.0f})')
ax.axhline(monthly_counts.mean(), color='gray', linestyle=':', alpha=0.7,
           label=f'Overall mean ({monthly_counts.mean():.0f})')

# 标注重要赛事
important_dates = {
    '2022-11': '2022 World Cup',
    '2024-06': 'Euro 2024',
    '2024-06': 'Copa América 2024',
    '2026-06': '2026 World Cup'
}
for date_str, label in important_dates.items():
    if date_str in monthly_idx:
        idx = monthly_idx.index(date_str)
        ax.annotate(label, xy=(idx, monthly_counts.values[idx]),
                    xytext=(idx, monthly_counts.values[idx] + monthly_counts.std()),
                    fontsize=8, ha='center', color='#C73E1D',
                    arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=0.8))

ax.set_xticks(range(0, len(monthly_counts), 6))
ax.set_xticklabels([monthly_idx[i] for i in range(0, len(monthly_counts), 6)], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Matches per Month', fontsize=12)
ax.set_title('Monthly Match Volume (2021–2026)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# 下: 季度场均进球趋势
ax = axes[1]
qtr_idx = [str(m) for m in quarterly_goals.index]
ax.bar(range(len(quarterly_goals)), quarterly_goals.values, color=C_HOME_WIN, alpha=0.8, width=0.7)
ax.axhline(quarterly_goals.mean(), color='gray', linestyle=':', alpha=0.7,
           label=f'Overall mean ({quarterly_goals.mean():.3f})')
# 滚动均值
qtr_rolling = quarterly_goals.rolling(4, min_periods=1).mean()
ax.plot(range(len(quarterly_goals)), qtr_rolling.values, color=C_TOTAL, linewidth=2,
        linestyle='--', label='4-quarter MA')

ax.set_xticks(range(len(quarterly_goals)))
ax.set_xticklabels(qtr_idx, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Avg Goals per Match', fontsize=12)
ax.set_title('Quarterly Average Goals per Match', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "timeline_coverage.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ timeline_coverage.png")

# --- 图4: 主客场胜率对比（分赛事） ---
fig, ax = plt.subplots(figsize=(12, 6))
top_tournaments = tournament_counts.head(15).index.tolist()
tournament_data = []
for t in top_tournaments:
    subset = matches[matches['tournament'] == t]
    if len(subset) < 10:
        continue
    home_win_pct = (subset['home_score'] > subset['away_score']).mean() * 100
    draw_pct = (subset['home_score'] == subset['away_score']).mean() * 100
    away_win_pct = (subset['home_score'] < subset['away_score']).mean() * 100
    tournament_data.append({
        'tournament': t,
        'home_win': home_win_pct,
        'draw': draw_pct,
        'away_win': away_win_pct,
        'n': len(subset)
    })

tdf = pd.DataFrame(tournament_data).sort_values('home_win')
y_pos = range(len(tdf))
ax.barh(y_pos, tdf['home_win'].values, height=0.6, color=C_HOME_WIN, label=f'Home Win (avg {tdf["home_win"].mean():.0f}%)')
ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(tdf['tournament'], tdf['n'])], fontsize=8)
ax.set_xlabel('Win Percentage (%)', fontsize=12)
ax.set_title('Home Team Win Rate by Tournament Type\n(2021–2026)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "home_win_rate_by_tournament.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ home_win_rate_by_tournament.png")

# --- 图5: FIFA 排名趋势 (Top 20 球队) ---
fig, ax = plt.subplots(figsize=(14, 7))
top20_teams = latest_rankings.sort_values('rank').head(20)['team_name'].tolist()
rank_pivot = rankings[rankings['team_name'].isin(top20_teams)].pivot_table(
    index='rank_date', columns='team_name', values='rank'
)
rank_pivot = rank_pivot.sort_index()
for team in top20_teams:
    if team in rank_pivot.columns:
        ax.plot(rank_pivot.index, rank_pivot[team], label=team, linewidth=1.5)
ax.invert_yaxis()
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('FIFA Ranking', fontsize=12)
ax.set_title('FIFA Ranking Trends — Top 20 Teams (2021–2026)', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "fifa_ranking_trends.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fifa_ranking_trends.png")

# --- 图6: WC2026 淘汰赛对阵图 ---
fig, ax = plt.subplots(figsize=(14, 10))
# 筛选淘汰赛
knockout_stages = ['Round of 32', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final']
knockout = wc2026[wc2026['stage'].isin(knockout_stages)].copy()
stages_order = {s: i for i, s in enumerate(knockout_stages)}

if len(knockout) > 0:
    knockout['stage_order'] = knockout['stage'].map(stages_order)
    knockout = knockout.sort_values(['stage_order', 'date'])

    # 为每个阶段创建可视化矩阵
    for i, stage in enumerate(knockout_stages):
        stage_matches = knockout[knockout['stage'] == stage]
        for j, (_, row) in enumerate(stage_matches.iterrows()):
            x_base = i * 2.5
            y_base = len(knockout_stages) - i + j * 1.2 - len(stage_matches) / 2
            # 画比赛卡片
            color = '#4CAF50' if row['status'] == 'Completed' else '#E0E0E0'
            score_text = f"{int(row['home_score'])}:{int(row['away_score'])}" if pd.notna(row['home_score']) else "?:?"
            rect = plt.Rectangle((x_base - 0.9, y_base - 0.4), 1.8, 0.8,
                                facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x_base, y_base + 0.15, f"{row['home_team']}", ha='center', va='center', fontsize=7, fontweight='bold')
            ax.text(x_base, y_base - 0.15, f"{row['away_team']}", ha='center', va='center', fontsize=7)
            ax.text(x_base + 0.9, y_base, score_text, ha='center', va='center', fontsize=8, fontweight='bold')

    # 阶段标题
    for i, stage in enumerate(knockout_stages):
        ax.text(i * 2.5, len(knockout_stages) + 1.5, stage, ha='center', fontsize=10, fontweight='bold')

ax.set_xlim(-1.5, len(knockout_stages) * 2.5 + 1)
ax.set_ylim(-1, len(knockout_stages) + 2)
ax.axis('off')
ax.set_title('2026 FIFA World Cup — Knockout Stage Bracket', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(OUT_DIR / "wc2026_knockout_bracket.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ wc2026_knockout_bracket.png")

# --- 图7: WC2026 小组积分分布 ---
fig, ax = plt.subplots(figsize=(14, 6))
groups = wc_standings['group'].unique()
bar_width = 0.2
x = np.arange(len(groups))
for pos, (label, color) in enumerate([(1, '#FFD700'), (2, '#C0C0C0'), (3, '#CD7F32'), (4, '#8E8E8E')]):
    pts = wc_standings[wc_standings['position'] == pos + 1].set_index('group')['Pts']
    pts = pts.reindex(groups).fillna(0)
    bars = ax.bar(x + pos * bar_width, pts.values, bar_width,
                  label=f'{pos+1}{["st","nd","rd","th"][pos]} Place', color=color, alpha=0.85)
    # 在柱上标球队名
    teams = wc_standings[wc_standings['position'] == pos + 1].set_index('group')['team']
    teams = teams.reindex(groups)
    for xi, (p, t) in enumerate(zip(pts.values, teams.values)):
        if p > 0:
            ax.text(xi + pos * bar_width, p + 0.3, t, ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(groups, fontsize=10)
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Points', fontsize=12)
ax.set_title('2026 FIFA World Cup — Group Stage Points Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "wc2026_group_points.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ wc2026_group_points.png")

# ============================================================
# 9. 汇总报告
# ============================================================
print(f"\n{'='*60}")
print("数据探查完成！")
print(f"{'='*60}")

report = {
    "data_summary": {
        "total_matches": int(len(matches)),
        "total_teams": int(len(unique_teams)),
        "total_tournaments": int(len(unique_tournaments)),
        "host_countries": int(unique_countries),
        "date_range": f"{date_min.date()} 至 {date_max.date()}",
        "date_span_days": int(date_span)
    },
    "match_outcomes": {
        "home_win_pct": round(home_wins/total*100, 1),
        "away_win_pct": round(away_wins/total*100, 1),
        "draw_pct": round(draws/total*100, 1)
    },
    "goals_stats": {
        "mean_per_match": round(float(goals_stats['mean']), 3),
        "median": int(goals_stats['50%']),
        "std": round(float(goals_stats['std']), 3),
        "max": int(goals_stats['max']),
        "zero_goal_pct": round((matches['total_goals']==0).mean()*100, 1)
    },
    "data_quality": {
        "missing_values_international_matches": int(matches.isnull().sum().sum()),
        "missing_values_rankings": int(rankings.isnull().sum().sum()),
        "missing_values_wc2026": {k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in wc2026.isnull().sum().to_dict().items()},
        "duplicate_matches": int(len(duplicates)),
        "future_matches": int(len(future_matches))
    }
}

with open(OUT_DIR / "exploration_report.json", 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n探查报告 JSON 已保存至:", OUT_DIR / "exploration_report.json")
print(f"\n生成图表列表:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
