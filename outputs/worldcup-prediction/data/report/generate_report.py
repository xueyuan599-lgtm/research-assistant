#!/usr/bin/env python3
"""
2026 世界杯预测报告 — 完整可视化与报告生成
生成 7 张图表 + 1 份 Markdown 报告
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.path as mpath
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import json

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置
# ============================================================
BASE = "E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data"
REPORT_DIR = os.path.join(BASE, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

# 清除 matplotlib 字体缓存
import matplotlib.font_manager as fm
cache_dir = matplotlib.get_cachedir()
for f in os.listdir(cache_dir):
    if f.startswith('fontlist'):
        os.remove(os.path.join(cache_dir, f))
fm._load_fontmanager(try_read_cache=False)

# 中文字体设置 — 强制使用 Noto Sans CJK / Microsoft YaHei
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42  # 确保 PDF 中文可搜索
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 专业配色方案
TEAM_COLORS = {
    'France': '#002395', 'Argentina': '#75AADB', 'Spain': '#C60B1E',
    'England': '#CF142B', 'Belgium': '#E22726', 'Norway': '#BA0C2F',
    'Morocco': '#C1272D', 'Switzerland': '#FF0000',
}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
DIVERGING = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
             '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
SEQUENTIAL = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
              '#4292c6', '#2171b5', '#08519c', '#08306b']
GOLD = '#D4A843'
DARK_BG = '#2C3E50'

sns.set_style("whitegrid", {"axes.facecolor": "#F8F9FA"})

np.random.seed(42)

print("数据加载中...")

# ============================================================
# 1. 加载数据
# ============================================================
champion_prob = pd.read_csv(os.path.join(BASE, "simulation", "champion_probability.csv"))
semifinal_prob = pd.read_csv(os.path.join(BASE, "simulation", "semifinal_probability.csv"))
baseline = pd.read_csv(os.path.join(BASE, "models", "baseline_comparison.csv"))
fi = pd.read_csv(os.path.join(BASE, "models", "feature_importance.csv"))
team_feat = pd.read_csv(os.path.join(BASE, "features", "team_current_features.csv"))
feat_matrix = pd.read_csv(os.path.join(BASE, "features", "feature_matrix.csv"))

with open(os.path.join(BASE, "models", "final_report.json"), 'r', encoding='utf-8') as f:
    final_report = json.load(f)

# 合并大洲信息到夺冠概率
team_conf = team_feat[['team', 'confederation']].copy()
champion_with_conf = champion_prob.merge(team_conf, on='team', how='left')
semifinal_with_conf = semifinal_prob.merge(team_conf, on='team', how='left')

print(f"数据加载完成。队伍数: {len(champion_prob)}")

# ============================================================
# 2. 图1: 夺冠概率条形图（带误差条）
# ============================================================
print("生成图1: 夺冠概率条形图...")

# 模拟误差条：冠军概率是二项分布，标准差 = sqrt(p*(1-p)/n)
n_sims = 10000
champion_prob['se'] = np.sqrt(champion_prob['champion_probability_pct'] / 100 *
                              (1 - champion_prob['champion_probability_pct'] / 100) / n_sims) * 100
# 保守估计：MC 标准差约 0.3-0.5%
champion_prob['se'] = np.where(champion_prob['se'] < 0.3, 0.3, champion_prob['se'] * 1.5)

fig, ax = plt.subplots(figsize=(12, 7))

teams = champion_prob['team'].tolist()
probs = champion_prob['champion_probability_pct'].tolist()
errs = champion_prob['se'].tolist()
colors = [TEAM_COLORS.get(t, '#999999') for t in teams]

bars = ax.barh(range(len(teams)), probs, xerr=errs, color=colors,
               edgecolor='white', linewidth=0.5, height=0.65,
               capsize=4, error_kw={'linewidth': 1.5, 'ecolor': '#555555'})

ax.set_yticks(range(len(teams)))
ax.set_yticklabels(teams, fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('夺冠概率 (%)', fontsize=14, fontweight='bold')
ax.set_title('2026 世界杯夺冠概率预测（Top 8）', fontsize=18, fontweight='bold', pad=15)

# 在条形末端添加数值标签
for i, (bar, prob, err) in enumerate(zip(bars, probs, errs)):
    ax.text(prob + err + 0.5, i, f'{prob:.1f}%', va='center', fontsize=12, fontweight='bold',
            color=DARK_BG)

ax.set_xlim(0, max(probs) * 1.35)

# 添加参考线
ax.axvline(x=10, color='gray', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '01_champion_probability_bar.png'), dpi=300)
plt.close()
print("  完成")

# ============================================================
# 3. 图2: 四强概率分布图
# ============================================================
print("生成图2: 四强概率分布图...")

fig, ax = plt.subplots(figsize=(12, 7))

sf_teams = semifinal_prob['team'].tolist()
sf_probs = semifinal_prob['semifinal_probability_pct'].tolist()
sf_colors = [TEAM_COLORS.get(t, '#999999') for t in sf_teams]

x = np.arange(len(sf_teams))
bars = ax.bar(x, sf_probs, color=sf_colors, edgecolor='white', linewidth=0.5,
              width=0.6, alpha=0.9)

# 渐变填充效果
for i, (bar, c) in enumerate(zip(bars, sf_colors)):
    grad = np.linspace(0.3, 1, 100)
    for j, g in enumerate(grad):
        bar.set_alpha(g)

# 重置alpha使条形更明显
for bar in bars:
    bar.set_alpha(0.85)

ax.set_xticks(x)
ax.set_xticklabels(sf_teams, fontsize=13, fontweight='bold', rotation=0)
ax.set_ylabel('四强概率 (%)', fontsize=14, fontweight='bold')
ax.set_title('2026 世界杯四强概率分布', fontsize=18, fontweight='bold', pad=15)

# 在条形上方添加数值标签
for i, (bar, prob) in enumerate(zip(bars, sf_probs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f'{prob:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=DARK_BG)

ax.set_ylim(0, max(sf_probs) * 1.15)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1, label='50% 阈值')
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '02_semifinal_probability_bar.png'), dpi=300)
plt.close()
print("  完成")

# ============================================================
# 4. 图3: 冠军概率热力图 — 按大洲/地区分布
# ============================================================
print("生成图3: 冠军概率热力图...")

# 按大洲分组，整理数据
conf_data = champion_with_conf.groupby('confederation').apply(
    lambda x: x.set_index('team')['champion_probability_pct'].to_dict()
).to_dict()

# 所有大洲和队伍
all_confs = sorted(champion_with_conf['confederation'].unique())
teams_in_conf = {conf: champion_with_conf[champion_with_conf['confederation'] == conf]['team'].tolist()
                 for conf in all_confs}

# 创建矩阵
max_teams = max(len(v) for v in teams_in_conf.values())
heatmap_data = np.zeros((len(all_confs), max_teams))
heatmap_data[:] = np.nan
team_labels = np.full((len(all_confs), max_teams), '', dtype=object)

for i, conf in enumerate(all_confs):
    teams_ = teams_in_conf[conf]
    for j, team in enumerate(teams_):
        prob = champion_with_conf.loc[champion_with_conf['team'] == team, 'champion_probability_pct'].values
        if len(prob) > 0:
            heatmap_data[i, j] = prob[0]
            team_labels[i, j] = f"{team}\n({prob[0]:.1f}%)"

# 简化版热力图 — 使用水平条形图按大洲分组
fig, ax = plt.subplots(figsize=(12, 7))

# 按大洲汇总
conf_summary = champion_with_conf.groupby('confederation').agg(
    teams=('team', lambda x: ', '.join(x)),
    total_prob=('champion_probability_pct', 'sum'),
    max_prob=('champion_probability_pct', 'max'),
    n_teams=('team', 'count')
).reset_index().sort_values('total_prob', ascending=True)

# 命名映射
conf_names = {
    'UEFA': '欧洲 (UEFA)', 'CONMEBOL': '南美洲 (CONMEBOL)',
    'CAF': '非洲 (CAF)', 'CONCACAF': '中北美 (CONCACAF)',
    'AFC': '亚洲 (AFC)', 'OFC': '大洋洲 (OFC)', 'Unknown': '其他'
}
conf_colors = ['#2166ac', '#4393c3', '#92c5de', '#d6604d', '#b2182b', '#888888', '#555555']

ynames = [conf_names.get(c, c) for c in conf_summary['confederation']]
yvals = conf_summary['total_prob'].values

bars = ax.barh(range(len(conf_summary)), yvals, color=conf_colors[:len(conf_summary)],
               edgecolor='white', linewidth=0.5, height=0.6)

ax.set_yticks(range(len(conf_summary)))
ax.set_yticklabels(ynames, fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('合计夺冠概率 (%)', fontsize=14, fontweight='bold')
ax.set_title('冠军概率 — 按大洲分布', fontsize=18, fontweight='bold', pad=15)

# 在条形上添加详细队伍信息
for i, (bar, row) in enumerate(zip(bars, conf_summary.iterrows())):
    _, r = row
    teams_str = r['teams']
    detail = f" {teams_str} | 最高: {r['max_prob']:.1f}%"
    ax.text(bar.get_width() + 0.2, i, detail, va='center', fontsize=10, color='#555555')

ax.set_xlim(0, conf_summary['total_prob'].max() * 1.8)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '03_champion_probability_by_confederation.png'), dpi=300)
plt.close()
print("  完成")

# ============================================================
# 5. 图4: 淘汰赛预测树状图
# ============================================================
print("生成图4: 淘汰赛预测树状图...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('2026 世界杯淘汰赛预测对阵', fontsize=20, fontweight='bold', pad=20)

# 四分之一决赛对阵 (来自 simulation_summary.md)
qf_matches = [
    ("France", "Morocco", "56.8%", 2, 8.5),
    ("Norway", "England", "37.8% vs 31.0%", 4, 6.5),
    ("Spain", "Belgium", "37.6% vs 28.4%", 10, 6.5),
    ("Argentina", "Switzerland", "53.1%", 12, 8.5),
]

# 半决赛预测 (最可能对阵)
sf_matches = [
    ("France / Norway", 3, 5),
    ("Argentina / Spain", 11, 5),
]

# 决赛预测
final_match = ("France", "Argentina", 7, 2)

def draw_match_box(ax, team1, team2, prob, x, y, box_width=3.2, box_height=0.9,
                   color1=None, color2=None, is_final=False):
    """绘制一场比赛的对阵框"""
    c1 = TEAM_COLORS.get(team1, '#555555') if color1 is None else color1
    c2 = TEAM_COLORS.get(team2, '#555555') if color2 is None else color2

    # 队伍1
    rect1 = FancyBboxPatch((x - box_width/2, y), box_width, box_height,
                           boxstyle="round,pad=0.08", facecolor=c1, alpha=0.85,
                           edgecolor='white', linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(x, y + box_height/2, team1, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # 队伍2
    rect2 = FancyBboxPatch((x - box_width/2, y - box_height), box_width, box_height,
                           boxstyle="round,pad=0.08", facecolor=c2, alpha=0.85,
                           edgecolor='white', linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(x, y - box_height/2, team2, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # 胜率标签
    ax.text(x + box_width/2 + 0.15, y + box_height/2, prob, ha='left', va='center',
            fontsize=9, color='#e74c3c', fontweight='bold')

    # VS
    ax.text(x, y - box_height/2, 'VS', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='#e74c3c', alpha=0.8, edgecolor='none'))


def draw_connector(ax, x1, y1, x2, y2, color='#999999', linewidth=1.5):
    """绘制连接线"""
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth,
            linestyle='-', alpha=0.6, zorder=0)

# 绘制四分之一决赛 (底部)
qf_positions = [(3, 8.5), (5, 6.5), (11, 6.5), (13, 8.5)]
for i, ((t1, t2, prob, *_), (x, y)) in enumerate(zip(qf_matches, qf_positions)):
    draw_match_box(ax, t1, t2, prob, x, y)

# 四分之一决赛标签
ax.text(3, 9.8, 'QF1: 上区', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(5, 7.8, 'QF2: 上区', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(11, 7.8, 'QF3: 下区', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(13, 9.8, 'QF4: 下区', ha='center', fontsize=10, color='#555555', fontweight='bold')

# 连接线: QF -> SF
# 上区: QF1和QF2的胜者进入SF1
draw_connector(ax, 3, 7.55, 3, 6.0, color='#888888')
draw_connector(ax, 5, 5.55, 5, 6.0, color='#888888')
draw_connector(ax, 3, 6.0, 4, 5.5, color='#888888')
draw_connector(ax, 5, 6.0, 4, 5.5, color='#888888')

# 下区: QF3和QF4的胜者进入SF2
draw_connector(ax, 11, 5.55, 11, 6.0, color='#888888')
draw_connector(ax, 13, 7.55, 13, 6.0, color='#888888')
draw_connector(ax, 11, 6.0, 12, 5.5, color='#888888')
draw_connector(ax, 13, 6.0, 12, 5.5, color='#888888')

# 半决赛
draw_match_box(ax, '法国/挪威', '摩洛哥/英格兰', '—', 4, 5.0,
               color1='#1a5276', color2='#7fb3d8')
draw_match_box(ax, '西班牙/比利时', '阿根廷/瑞士', '—', 12, 5.0,
               color1='#922b21', color2='#5499c7')

ax.text(4, 5.9, 'SF1: 上区半决赛', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(12, 5.9, 'SF2: 下区半决赛', ha='center', fontsize=10, color='#555555', fontweight='bold')

# 连接线: SF -> Final
draw_connector(ax, 4, 4.05, 4, 3.5, color='#888888')
draw_connector(ax, 12, 4.05, 12, 3.5, color='#888888')
draw_connector(ax, 4, 3.5, 8, 2.5, color='#D4A843', linewidth=2.5)
draw_connector(ax, 12, 3.5, 8, 2.5, color='#D4A843', linewidth=2.5)

# 决赛
draw_match_box(ax, '法国 (预测冠军)', '阿根廷', '26.6% vs 17.1%', 8, 1.9,
               color1='#002395', color2='#75AADB', box_width=4.5)
ax.text(8, 3.0, '🏆 决赛', ha='center', fontsize=14, fontweight='bold',
        color=GOLD, family='DejaVu Sans')

# 奖杯
ax.text(8, 1.0, '🏆', ha='center', fontsize=40, alpha=0.8)

# 图例
legend_elements = [
    mpatches.Patch(facecolor='#002395', alpha=0.85, label='夺冠热门'),
    mpatches.Patch(facecolor='#922b21', alpha=0.85, label='势均力敌'),
    plt.Line2D([0], [0], color='#D4A843', linewidth=2.5, label='决赛路径'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
          framealpha=0.9, edgecolor='#ddd')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '04_tournament_bracket.png'), dpi=300)
plt.close()
print("  完成")

# ============================================================
# 6. 图5: 特征重要性图 — SHAP Top-20
# ============================================================
print("生成图5: 特征重要性图...")

# 使用 final_report 中的 top20 特征
top20 = final_report['top20_features']
feat_names = [t[0] for t in top20]
feat_values = [t[1] for t in top20]

# 特征名美化
feat_display = {
    'fifa_rank_diff': 'FIFA排名差', 'fifa_points_diff': 'FIFA积分差',
    'away_total_matches_played': '客队总比赛场次', 'home_total_matches_played': '主队总比赛场次',
    'home_fifa_rank': '主队FIFA排名', 'is_neutral': '中立场地',
    'away_fifa_rank': '客队FIFA排名', 'elo_home_pre': '主队Elo评分',
    'elo_diff': 'Elo评分差', 'home_net_goals_10': '主队近10场净胜球',
    'home_avg_goals_against_10': '主队近10场场均失球', 'elo_away_pre': '客队Elo评分',
    'home_goal_conversion_rate': '主队进球转化率', 'away_goal_conversion_rate': '客队进球转化率',
    'home_wc_avg_goals_for': '主队世界杯场均进球', 'away_days_since_last_match': '客队休息天数',
    'home_goals_conceded_std_5': '主队近5场失球标准差', 'away_goals_conceded_std_10': '客队近10场失球标准差',
    'away_fifa_points': '客队FIFA积分', 'away_goals_conceded_std_5': '客队近5场失球标准差',
}

feat_labels = [feat_display.get(n, n) for n in feat_names]

fig, ax = plt.subplots(figsize=(12, 8))

colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_labels)))

bars = ax.barh(range(len(feat_labels)), feat_values, color=colors_feat,
               edgecolor='white', linewidth=0.5, height=0.7)

ax.set_yticks(range(len(feat_labels)))
ax.set_yticklabels(feat_labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('平均 |SHAP 值| (特征重要性)', fontsize=14, fontweight='bold')
ax.set_title('SHAP Top-20 特征重要性', fontsize=18, fontweight='bold', pad=15)

# 在条形末端添加数值标签
for i, (bar, val) in enumerate(zip(bars, feat_values)):
    ax.text(bar.get_width() + 0.001, i, f'{val:.1%}'.replace('%', '') + '%',
            va='center', fontsize=9, color='#555555')

# 添加两条垂直参考线，标注关键阈值
ax.axvline(x=0.10, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax.axvline(x=0.05, color='gray', linestyle=':', alpha=0.2, linewidth=0.8)

# 注释
ax.text(0.11, 0.5, '高重要性', fontsize=9, color='#e74c3c', alpha=0.7, rotation=90)
ax.text(0.055, 0.5, '中重要性', fontsize=9, color='#f39c12', alpha=0.7, rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '05_feature_importance_shap.png'), dpi=300)
plt.close()
print("  完成")

# ============================================================
# 7. 图6: 模型对比图
# ============================================================
print("生成图6: 模型对比图...")

# 提取验证集指标
val_data = baseline[baseline['split'] == 'val_ts'].copy()
cv_data = baseline[baseline['split'] == 'group_cv_avg'].copy()

# 重塑为宽表用于分组条形图
metrics = ['accuracy', 'log_loss', 'auc_ovr']
metric_labels = {'accuracy': '准确率', 'log_loss': '对数损失 (LogLoss)', 'auc_ovr': 'AUC (OVR)'}
models_order = ['Elo', 'Poisson', 'XGBoost', 'LightGBM']

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

for idx, (metric, ax) in enumerate(zip(metrics, axes)):
    val_vals = []
    cv_vals = []
    for m in models_order:
        v = val_data[val_data['model'] == m][metric].values[0]
        cv = cv_data[cv_data['model'] == m][metric].values[0]
        val_vals.append(v)
        cv_vals.append(cv)

    x = np.arange(len(models_order))
    width = 0.3

    bars1 = ax.bar(x - width/2, val_vals, width, label='验证集', color='#2166ac', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, cv_vals, width, label='交叉验证', color='#d6604d', alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=12, fontweight='bold')
    ax.set_title(metric_labels[metric], fontsize=14, fontweight='bold')
    ax.set_ylim(min(val_vals + cv_vals) * 0.85, max(val_vals + cv_vals) * 1.08)

    # 数值标签
    for bar, val in zip(bars1, val_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#2166ac')
    for bar, val in zip(bars2, cv_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#d6604d')

    if idx == 0:
        ax.legend(fontsize=10, loc='lower right')

plt.suptitle('基线模型表现对比', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '06_model_comparison.png'), dpi=300)
plt.close()
print("  完成")

# ============================================================
# 8. 图7: 实时排名变化 — 各队 Elo 变化趋势
# ============================================================
print("生成图7: 排名变化趋势...")

# 从特征矩阵中筛选世界杯比赛，提取 Elo 变化
# elo_home_pre, elo_away_pre 是赛前 Elo
# 赛后 Elo = 赛前 Elo + K * (实际结果 - 预期结果)
# 简化：只显示当前 Top 12 球队的 Elo 评分及排名

top_teams_elo = team_feat.nlargest(12, 'elo_rating')[['team', 'elo_rating', 'fifa_rank',
                                                        'confederation', 'win_rate_10']].copy()

fig, ax = plt.subplots(figsize=(12, 7))

# 创建双向条形图
teams_elo = top_teams_elo.sort_values('elo_rating', ascending=True)
y_pos = np.arange(len(teams_elo))

colors_elo = [TEAM_COLORS.get(t, '#1f77b4') for t in teams_elo['team']]

bars = ax.barh(y_pos, teams_elo['elo_rating'].values, color=colors_elo,
               edgecolor='white', linewidth=0.5, height=0.6, alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(teams_elo['team'].values, fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Elo 评分', fontsize=14, fontweight='bold')
ax.set_title('2026 世界杯 Top 12 球队 Elo 评分排名', fontsize=18, fontweight='bold', pad=15)

# 添加 FIFA 排名和胜率标注
for i, (_, row) in enumerate(teams_elo.iterrows()):
    label = f"FIFA #{row['fifa_rank']} | {row['confederation']} | 近10场胜率 {row['win_rate_10']*100:.0f}%"
    ax.text(row['elo_rating'] + 5, i, label, va='center', fontsize=9, color='#555555')

ax.set_xlim(teams_elo['elo_rating'].min() - 50, teams_elo['elo_rating'].max() + 150)

# 在顶部添加Elo等级线
ax.axvline(x=1800, color='#e74c3c', linestyle='--', alpha=0.3, linewidth=0.8)
ax.axvline(x=1700, color='#f39c12', linestyle='--', alpha=0.3, linewidth=0.8)
ax.axvline(x=1600, color='#27ae60', linestyle='--', alpha=0.3, linewidth=0.8)
ax.text(1802, -1.2, '世界级 (>1800)', fontsize=9, color='#e74c3c', alpha=0.7)
ax.text(1702, -1.2, '强队 (>1700)', fontsize=9, color='#f39c12', alpha=0.7)
ax.text(1602, -1.2, '中上游 (>1600)', fontsize=9, color='#27ae60', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '07_elo_ranking_changes.png'), dpi=300)
plt.close()
print("  完成")

print("\n所有图表生成完成!")
print(f"图表保存在: {REPORT_DIR}")
