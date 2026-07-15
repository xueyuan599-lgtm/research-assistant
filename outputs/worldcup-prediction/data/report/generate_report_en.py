#!/usr/bin/env python3
"""
2026 World Cup Prediction Report — English Visualizations & Report Generation
Generates 7 charts + 1 Markdown report (English)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.path as mpath
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import json

warnings.filterwarnings('ignore')

# ============================================================
# 0. Global Settings
# ============================================================
BASE = "E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data"
REPORT_DIR = os.path.join(BASE, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Professional color scheme
TEAM_COLORS = {
    'France': '#002395', 'Argentina': '#75AADB', 'Spain': '#C60B1E',
    'England': '#CF142B', 'Belgium': '#E22726', 'Norway': '#BA0C2F',
    'Morocco': '#C1272D', 'Switzerland': '#FF0000',
}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
GOLD = '#D4A843'
DARK_BG = '#2C3E50'

sns.set_style("whitegrid", {"axes.facecolor": "#F8F9FA"})

np.random.seed(42)

print("Loading data...")

# ============================================================
# 1. Load data
# ============================================================
champion_prob = pd.read_csv(os.path.join(BASE, "simulation", "champion_probability.csv"))
semifinal_prob = pd.read_csv(os.path.join(BASE, "simulation", "semifinal_probability.csv"))
baseline = pd.read_csv(os.path.join(BASE, "models", "baseline_comparison.csv"))
fi = pd.read_csv(os.path.join(BASE, "models", "feature_importance.csv"))
team_feat = pd.read_csv(os.path.join(BASE, "features", "team_current_features.csv"))
feat_matrix = pd.read_csv(os.path.join(BASE, "features", "feature_matrix.csv"))

with open(os.path.join(BASE, "models", "final_report.json"), 'r', encoding='utf-8') as f:
    final_report = json.load(f)

# Merge confederation info
team_conf = team_feat[['team', 'confederation']].copy()
champion_with_conf = champion_prob.merge(team_conf, on='team', how='left')
semifinal_with_conf = semifinal_prob.merge(team_conf, on='team', how='left')

print(f"Data loaded. Teams: {len(champion_prob)}")

# ============================================================
# 2. Figure 1: Champion Probability Bar Chart (with error bars)
# ============================================================
print("Generating Figure 1: Champion Probability...")

n_sims = 10000
champion_prob['se'] = np.sqrt(champion_prob['champion_probability_pct'] / 100 *
                              (1 - champion_prob['champion_probability_pct'] / 100) / n_sims) * 100
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
ax.set_xlabel('Champion Probability (%)', fontsize=14, fontweight='bold')
ax.set_title('2026 World Cup — Champion Probability (Top 8)', fontsize=18, fontweight='bold', pad=15)

for i, (bar, prob, err) in enumerate(zip(bars, probs, errs)):
    ax.text(prob + err + 0.5, i, f'{prob:.1f}%', va='center', fontsize=12, fontweight='bold',
            color=DARK_BG)

ax.set_xlim(0, max(probs) * 1.35)
ax.axvline(x=10, color='gray', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '01_champion_probability_bar.png'), dpi=300)
plt.close()
print("  Done")

# ============================================================
# 3. Figure 2: Semifinal Probability Bar Chart
# ============================================================
print("Generating Figure 2: Semifinal Probability...")

fig, ax = plt.subplots(figsize=(12, 7))

sf_teams = semifinal_prob['team'].tolist()
sf_probs = semifinal_prob['semifinal_probability_pct'].tolist()
sf_colors = [TEAM_COLORS.get(t, '#999999') for t in sf_teams]

x = np.arange(len(sf_teams))
bars = ax.bar(x, sf_probs, color=sf_colors, edgecolor='white', linewidth=0.5,
              width=0.6, alpha=0.9)

for bar in bars:
    bar.set_alpha(0.85)

ax.set_xticks(x)
ax.set_xticklabels(sf_teams, fontsize=13, fontweight='bold', rotation=0)
ax.set_ylabel('Semifinal Probability (%)', fontsize=14, fontweight='bold')
ax.set_title('2026 World Cup — Semifinal Probability Distribution', fontsize=18, fontweight='bold', pad=15)

for i, (bar, prob) in enumerate(zip(bars, sf_probs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f'{prob:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=DARK_BG)

ax.set_ylim(0, max(sf_probs) * 1.15)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1, label='50% threshold')
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '02_semifinal_probability_bar.png'), dpi=300)
plt.close()
print("  Done")

# ============================================================
# 4. Figure 3: Champion Probability by Confederation
# ============================================================
print("Generating Figure 3: Probability by Confederation...")

fig, ax = plt.subplots(figsize=(12, 7))

conf_summary = champion_with_conf.groupby('confederation').agg(
    teams=('team', lambda x: ', '.join(x)),
    total_prob=('champion_probability_pct', 'sum'),
    max_prob=('champion_probability_pct', 'max'),
    n_teams=('team', 'count')
).reset_index().sort_values('total_prob', ascending=True)

conf_labels = {
    'UEFA': 'UEFA (Europe)', 'CONMEBOL': 'CONMEBOL (S. America)',
    'CAF': 'CAF (Africa)', 'CONCACAF': 'CONCACAF (N. America)',
    'AFC': 'AFC (Asia)', 'OFC': 'OFC (Oceania)', 'Unknown': 'Other'
}
conf_colors = ['#2166ac', '#4393c3', '#92c5de', '#d6604d', '#b2182b', '#888888', '#555555']

ynames = [conf_labels.get(c, c) for c in conf_summary['confederation']]
yvals = conf_summary['total_prob'].values

bars = ax.barh(range(len(conf_summary)), yvals, color=conf_colors[:len(conf_summary)],
               edgecolor='white', linewidth=0.5, height=0.6)

ax.set_yticks(range(len(conf_summary)))
ax.set_yticklabels(ynames, fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Total Champion Probability (%)', fontsize=14, fontweight='bold')
ax.set_title('Champion Probability by Confederation', fontsize=18, fontweight='bold', pad=15)

for i, (bar, row) in enumerate(zip(bars, conf_summary.iterrows())):
    _, r = row
    teams_str = r['teams']
    detail = f" {teams_str} | Max: {r['max_prob']:.1f}%"
    ax.text(bar.get_width() + 0.2, i, detail, va='center', fontsize=10, color='#555555')

ax.set_xlim(0, conf_summary['total_prob'].max() * 1.8)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '03_champion_probability_by_confederation.png'), dpi=300)
plt.close()
print("  Done")

# ============================================================
# 5. Figure 4: Tournament Bracket
# ============================================================
print("Generating Figure 4: Tournament Bracket...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('2026 World Cup — Knockout Stage Prediction', fontsize=20, fontweight='bold', pad=20)

# Quarterfinal pairings
qf_matches = [
    ("France", "Morocco", "56.8%", 2, 8.5),
    ("Norway", "England", "37.8% vs 31.0%", 4, 6.5),
    ("Spain", "Belgium", "37.6% vs 28.4%", 10, 6.5),
    ("Argentina", "Switzerland", "53.1%", 12, 8.5),
]

# Semifinal predictions (most likely)
sf_matches = [
    ("France / Norway", 3, 5),
    ("Argentina / Spain", 11, 5),
]

# Final prediction
final_match = ("France", "Argentina", 7, 2)

def draw_match_box(ax, team1, team2, prob, x, y, box_width=3.2, box_height=0.9,
                   color1=None, color2=None, is_final=False):
    """Draw a match box for the bracket"""
    c1 = TEAM_COLORS.get(team1, '#555555') if color1 is None else color1
    c2 = TEAM_COLORS.get(team2, '#555555') if color2 is None else color2

    # Team 1
    rect1 = FancyBboxPatch((x - box_width/2, y), box_width, box_height,
                           boxstyle="round,pad=0.08", facecolor=c1, alpha=0.85,
                           edgecolor='white', linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(x, y + box_height/2, team1, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Team 2
    rect2 = FancyBboxPatch((x - box_width/2, y - box_height), box_width, box_height,
                           boxstyle="round,pad=0.08", facecolor=c2, alpha=0.85,
                           edgecolor='white', linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(x, y - box_height/2, team2, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Win probability label
    ax.text(x + box_width/2 + 0.15, y + box_height/2, prob, ha='left', va='center',
            fontsize=9, color='#e74c3c', fontweight='bold')

    # VS badge
    ax.text(x, y - box_height/2, 'VS', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='#e74c3c', alpha=0.8, edgecolor='none'))


def draw_connector(ax, x1, y1, x2, y2, color='#999999', linewidth=1.5):
    """Draw a connector line"""
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth,
            linestyle='-', alpha=0.6, zorder=0)

# Draw quarterfinals (bottom)
qf_positions = [(3, 8.5), (5, 6.5), (11, 6.5), (13, 8.5)]
for i, ((t1, t2, prob, *_), (x, y)) in enumerate(zip(qf_matches, qf_positions)):
    draw_match_box(ax, t1, t2, prob, x, y)

# QF labels
ax.text(3, 9.8, 'QF1: Upper Bracket', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(5, 7.8, 'QF2: Upper Bracket', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(11, 7.8, 'QF3: Lower Bracket', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(13, 9.8, 'QF4: Lower Bracket', ha='center', fontsize=10, color='#555555', fontweight='bold')

# Connectors: QF -> SF
draw_connector(ax, 3, 7.55, 3, 6.0, color='#888888')
draw_connector(ax, 5, 5.55, 5, 6.0, color='#888888')
draw_connector(ax, 3, 6.0, 4, 5.5, color='#888888')
draw_connector(ax, 5, 6.0, 4, 5.5, color='#888888')

draw_connector(ax, 11, 5.55, 11, 6.0, color='#888888')
draw_connector(ax, 13, 7.55, 13, 6.0, color='#888888')
draw_connector(ax, 11, 6.0, 12, 5.5, color='#888888')
draw_connector(ax, 13, 6.0, 12, 5.5, color='#888888')

# Semifinals
draw_match_box(ax, 'France / Norway', 'Morocco / England', '—', 4, 5.0,
               color1='#1a5276', color2='#7fb3d8')
draw_match_box(ax, 'Spain / Belgium', 'Argentina / Switzerland', '—', 12, 5.0,
               color1='#922b21', color2='#5499c7')

ax.text(4, 5.9, 'SF1: Upper Semifinal', ha='center', fontsize=10, color='#555555', fontweight='bold')
ax.text(12, 5.9, 'SF2: Lower Semifinal', ha='center', fontsize=10, color='#555555', fontweight='bold')

# Connectors: SF -> Final
draw_connector(ax, 4, 4.05, 4, 3.5, color='#888888')
draw_connector(ax, 12, 4.05, 12, 3.5, color='#888888')
draw_connector(ax, 4, 3.5, 8, 2.5, color='#D4A843', linewidth=2.5)
draw_connector(ax, 12, 3.5, 8, 2.5, color='#D4A843', linewidth=2.5)

# Final
draw_match_box(ax, 'France (favorite)', 'Argentina', '26.6% vs 17.1%', 8, 1.9,
               color1='#002395', color2='#75AADB', box_width=4.5)
ax.text(8, 3.0, '🏆 FINAL', ha='center', fontsize=14, fontweight='bold',
        color=GOLD)

# Trophy
ax.text(8, 1.0, '🏆', ha='center', fontsize=40, alpha=0.8)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#002395', alpha=0.85, label='Champion favorite'),
    mpatches.Patch(facecolor='#922b21', alpha=0.85, label='Close matchup'),
    plt.Line2D([0], [0], color='#D4A843', linewidth=2.5, label='Final path'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
          framealpha=0.9, edgecolor='#ddd')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '04_tournament_bracket.png'), dpi=300)
plt.close()
print("  Done")

# ============================================================
# 6. Figure 5: SHAP Feature Importance (Top 20)
# ============================================================
print("Generating Figure 5: Feature Importance...")

top20 = final_report['top20_features']
feat_names = [t[0] for t in top20]
feat_values = [t[1] for t in top20]

feat_display = {
    'fifa_rank_diff': 'FIFA Rank Diff', 'fifa_points_diff': 'FIFA Points Diff',
    'away_total_matches_played': 'Away Total Matches', 'home_total_matches_played': 'Home Total Matches',
    'home_fifa_rank': 'Home FIFA Rank', 'is_neutral': 'Neutral Venue',
    'away_fifa_rank': 'Away FIFA Rank', 'elo_home_pre': 'Home Elo Rating',
    'elo_diff': 'Elo Rating Diff', 'home_net_goals_10': 'Home Net Goals (L10)',
    'home_avg_goals_against_10': 'Home Goals Against (L10)', 'elo_away_pre': 'Away Elo Rating',
    'home_goal_conversion_rate': 'Home Goal Conversion', 'away_goal_conversion_rate': 'Away Goal Conversion',
    'home_wc_avg_goals_for': 'Home WC Avg Goals', 'away_days_since_last_match': 'Away Days Rest',
    'home_goals_conceded_std_5': 'Home Conceded Std (L5)', 'away_goals_conceded_std_10': 'Away Conceded Std (L10)',
    'away_fifa_points': 'Away FIFA Points', 'away_goals_conceded_std_5': 'Away Conceded Std (L5)',
}

feat_labels = [feat_display.get(n, n) for n in feat_names]

fig, ax = plt.subplots(figsize=(12, 8))

colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_labels)))

bars = ax.barh(range(len(feat_labels)), feat_values, color=colors_feat,
               edgecolor='white', linewidth=0.5, height=0.7)

ax.set_yticks(range(len(feat_labels)))
ax.set_yticklabels(feat_labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=14, fontweight='bold')
ax.set_title('SHAP Top-20 Feature Importance', fontsize=18, fontweight='bold', pad=15)

for i, (bar, val) in enumerate(zip(bars, feat_values)):
    ax.text(bar.get_width() + 0.001, i, f'{val:.1%}'.replace('%', '') + '%',
            va='center', fontsize=9, color='#555555')

ax.axvline(x=0.10, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax.axvline(x=0.05, color='gray', linestyle=':', alpha=0.2, linewidth=0.8)

ax.text(0.11, 0.5, 'High Importance', fontsize=9, color='#e74c3c', alpha=0.7, rotation=90)
ax.text(0.055, 0.5, 'Medium Importance', fontsize=9, color='#f39c12', alpha=0.7, rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '05_feature_importance_shap.png'), dpi=300)
plt.close()
print("  Done")

# ============================================================
# 7. Figure 6: Model Comparison
# ============================================================
print("Generating Figure 6: Model Comparison...")

val_data = baseline[baseline['split'] == 'val_ts'].copy()
cv_data = baseline[baseline['split'] == 'group_cv_avg'].copy()

metrics = ['accuracy', 'log_loss', 'auc_ovr']
metric_labels = {'accuracy': 'Accuracy', 'log_loss': 'LogLoss', 'auc_ovr': 'AUC (OVR)'}
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

    bars1 = ax.bar(x - width/2, val_vals, width, label='Validation', color='#2166ac', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, cv_vals, width, label='Cross-Validation', color='#d6604d', alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=12, fontweight='bold')
    ax.set_title(metric_labels[metric], fontsize=14, fontweight='bold')
    ax.set_ylim(min(val_vals + cv_vals) * 0.85, max(val_vals + cv_vals) * 1.08)

    for bar, val in zip(bars1, val_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#2166ac')
    for bar, val in zip(bars2, cv_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#d6604d')

    if idx == 0:
        ax.legend(fontsize=10, loc='lower right')

plt.suptitle('Baseline Model Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '06_model_comparison.png'), dpi=300)
plt.close()
print("  Done")

# ============================================================
# 8. Figure 7: Elo Ranking Trends
# ============================================================
print("Generating Figure 7: Elo Ranking...")

top_teams_elo = team_feat.nlargest(12, 'elo_rating')[['team', 'elo_rating', 'fifa_rank',
                                                        'confederation', 'win_rate_10']].copy()

fig, ax = plt.subplots(figsize=(12, 7))

teams_elo = top_teams_elo.sort_values('elo_rating', ascending=True)
y_pos = np.arange(len(teams_elo))

colors_elo = [TEAM_COLORS.get(t, '#1f77b4') for t in teams_elo['team']]

bars = ax.barh(y_pos, teams_elo['elo_rating'].values, color=colors_elo,
               edgecolor='white', linewidth=0.5, height=0.6, alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(teams_elo['team'].values, fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Elo Rating', fontsize=14, fontweight='bold')
ax.set_title('2026 World Cup — Top 12 Team Elo Ratings', fontsize=18, fontweight='bold', pad=15)

for i, (_, row) in enumerate(teams_elo.iterrows()):
    label = f"FIFA #{row['fifa_rank']} | {row['confederation']} | L10 Win Rate {row['win_rate_10']*100:.0f}%"
    ax.text(row['elo_rating'] + 5, i, label, va='center', fontsize=9, color='#555555')

ax.set_xlim(teams_elo['elo_rating'].min() - 50, teams_elo['elo_rating'].max() + 150)

ax.axvline(x=1800, color='#e74c3c', linestyle='--', alpha=0.3, linewidth=0.8)
ax.axvline(x=1700, color='#f39c12', linestyle='--', alpha=0.3, linewidth=0.8)
ax.axvline(x=1600, color='#27ae60', linestyle='--', alpha=0.3, linewidth=0.8)
ax.text(1802, -1.2, 'World Class (>1800)', fontsize=9, color='#e74c3c', alpha=0.7)
ax.text(1702, -1.2, 'Strong (>1700)', fontsize=9, color='#f39c12', alpha=0.7)
ax.text(1602, -1.2, 'Above Avg (>1600)', fontsize=9, color='#27ae60', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '07_elo_ranking_changes.png'), dpi=300)
plt.close()
print("  Done")

print("\nAll charts generated successfully!")
print(f"Charts saved to: {REPORT_DIR}")
