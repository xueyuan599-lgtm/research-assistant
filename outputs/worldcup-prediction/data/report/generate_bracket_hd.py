#!/usr/bin/env python3
"""Generate a high-resolution tournament bracket figure"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

BASE = "E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data"
REPORT_DIR = os.path.join(BASE, "report")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42

TEAM_COLORS = {
    'France': '#002395', 'Argentina': '#75AADB', 'Spain': '#C60B1E',
    'England': '#CF142B', 'Belgium': '#E22726', 'Norway': '#BA0C2F',
    'Morocco': '#C1272D', 'Switzerland': '#FF0000',
}
GOLD = '#D4A843'

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('#FAFAFA')

# Title
ax.text(11, 13.5, '2026 World Cup — Knockout Stage Prediction',
        ha='center', fontsize=22, fontweight='bold', color='#2C3E50')

def draw_match_box(ax, team1, team2, prob_text, x, y, box_w=3.8, box_h=1.0,
                   color1=None, color2=None, team1_sub=None, team2_sub=None):
    c1 = TEAM_COLORS.get(team1, '#555555') if color1 is None else color1
    c2 = TEAM_COLORS.get(team2, '#555555') if color2 is None else color2

    # Team 1 box
    rect1 = FancyBboxPatch((x - box_w/2, y), box_w, box_h,
                           boxstyle="round,pad=0.1", facecolor=c1, alpha=0.88,
                           edgecolor='white', linewidth=2)
    ax.add_patch(rect1)
    ax.text(x, y + box_h/2, team1, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    if team1_sub:
        ax.text(x, y + box_h/2 - 0.38, team1_sub, ha='center', va='center',
                fontsize=8.5, color='white', alpha=0.75)

    # Team 2 box
    rect2 = FancyBboxPatch((x - box_w/2, y - box_h), box_w, box_h,
                           boxstyle="round,pad=0.1", facecolor=c2, alpha=0.88,
                           edgecolor='white', linewidth=2)
    ax.add_patch(rect2)
    ax.text(x, y - box_h/2, team2, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    if team2_sub:
        ax.text(x, y - box_h/2 + 0.38, team2_sub, ha='center', va='center',
                fontsize=8.5, color='white', alpha=0.75)

    ax.text(x + box_w/2 + 0.2, y + box_h/2, prob_text, ha='left', va='center',
            fontsize=10, color='#C0392B', fontweight='bold')

    ax.plot(x, y - box_h/2, marker='o', markersize=12,
            color='#C0392B', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.text(x, y - box_h/2, 'VS', ha='center', va='center', fontsize=6.5,
            color='white', fontweight='bold', zorder=6)

def draw_connector(ax, x1, y1, x2, y2, color='#95A5A6', lw=2.5, ls='-', alpha=0.5):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, alpha=alpha, zorder=0)

# ===================== QUARTERFINALS =====================
# Main QF boxes
draw_match_box(ax, 'France', 'Morocco', '61.3%', 4, 11.5, box_w=3.8,
               color1='#002395', color2='#C1272D')
draw_match_box(ax, 'Norway', 'England', '22.5% / 55.7%', 6.5, 9.2, box_w=3.8,
               color1='#BA0C2F', color2='#CF142B')
draw_match_box(ax, 'Spain', 'Belgium', '45.5% / 24.7%', 15.5, 9.2, box_w=3.8,
               color1='#C60B1E', color2='#E22726')
draw_match_box(ax, 'Argentina', 'Switzerland', '61.8%', 18, 11.5, box_w=3.8,
               color1='#75AADB', color2='#FF0000')

# QF labels
for x_pos, label_text in [(4, 'QF1'), (6.5, 'QF2'), (15.5, 'QF3'), (18, 'QF4')]:
    ax.text(x_pos, 13.0, label_text, ha='center', fontsize=11, color='#7F8C8D', fontweight='bold')

# ===================== CONNECTORS: QF → SF =====================
# Upper bracket
draw_connector(ax, 4, 10.5, 4, 7.8)
draw_connector(ax, 6.5, 8.2, 6.5, 7.8)
draw_connector(ax, 4, 7.8, 5.25, 7.0)
draw_connector(ax, 6.5, 7.8, 5.25, 7.0)

# Lower bracket
draw_connector(ax, 15.5, 8.2, 15.5, 7.8)
draw_connector(ax, 18, 10.5, 18, 7.8)
draw_connector(ax, 15.5, 7.8, 16.75, 7.0)
draw_connector(ax, 18, 7.8, 16.75, 7.0)

# ===================== SEMIFINALS =====================
draw_match_box(ax, 'France (62%)', 'England (56%)', '—', 5.25, 6.5, box_w=4.2,
               color1='#1A5276', color2='#922B21',
               team1_sub='winner QF1', team2_sub='winner QF2')
draw_match_box(ax, 'Spain (46%)', 'Argentina (62%)', '—', 16.75, 6.5, box_w=4.2,
               color1='#922B21', color2='#1A5276',
               team1_sub='winner QF3', team2_sub='winner QF4')

# SF labels
ax.text(5.25, 8.0, 'SF1: Upper Semifinal', ha='center', fontsize=11, color='#7F8C8D', fontweight='bold')
ax.text(16.75, 8.0, 'SF2: Lower Semifinal', ha='center', fontsize=11, color='#7F8C8D', fontweight='bold')

# ===================== CONNECTORS: SF → FINAL =====================
draw_connector(ax, 5.25, 5.5, 5.25, 4.0)
draw_connector(ax, 16.75, 5.5, 16.75, 4.0)
draw_connector(ax, 5.25, 4.0, 11, 2.7, color=GOLD, lw=3.5)
draw_connector(ax, 16.75, 4.0, 11, 2.7, color=GOLD, lw=3.5)

# ===================== FINAL =====================
draw_match_box(ax, 'France', 'Argentina', '26.6% / 17.1%', 11, 2.1, box_w=5.0,
               color1='#002395', color2='#75AADB')

# Final label
ax.text(11, 3.5, '✦  F I N A L  ✦', ha='center', fontsize=18, fontweight='bold', color=GOLD)

# Trophy icon (text fallback: "(Trophy)" to avoid font issues)
ax.text(11, 0.45, '✦  WORLD CUP  ✦', ha='center', fontsize=16, fontweight='bold',
        color=GOLD, alpha=0.6, style='italic')

# ===================== SIDE INFO =====================
# Left panel: summary stats
info_x = 0.3
info_y = 5.5
ax.text(info_x, info_y, 'TOURNAMENT INFO', fontsize=11, fontweight='bold', color='#2C3E50')
info_lines = [
    '• Stage: Quarterfinals',
    '• 8 teams remaining',
    '• 7 matches to play',
    '• Model: LightGBM + MC',
    '• 10,000 simulations',
    f'• Favorite: France',
]
for j, line in enumerate(info_lines):
    ax.text(info_x, info_y - 0.6 * (j + 1), line, fontsize=9, color='#555555')

# Right panel: semifinal probabilities
sf_probs_x = 19.5
sf_probs_y = 5.5
ax.text(sf_probs_x, sf_probs_y, 'MOST LIKELY SEMIFINALS', fontsize=11, fontweight='bold', color='#2C3E50')
sf_lines = [
    '1. England vs France  — 48.4%',
    '2. Argentina vs Spain  — 44.3%',
    '3. Argentina vs Belgium — 28.2%',
    '4. France vs Norway   — 24.6%',
    '5. England vs Morocco  — 17.8%',
]
for j, line in enumerate(sf_lines):
    ax.text(sf_probs_x, sf_probs_y - 0.6 * (j + 1), line, fontsize=9, color='#555555')

# Top right: most likely final four
ff_x = 19.5
ff_y = 2.5
ax.text(ff_x, ff_y, 'MOST LIKELY FINAL FOUR', fontsize=11, fontweight='bold', color='#2C3E50')
ff_lines = [
    '1. Arg, Eng, Fra, Esp  — 21.1%',
    '2. Arg, Bel, Eng, Fra  — 13.6%',
    '3. Arg, Fra, Nor, Esp  — 10.9%',
]
for j, line in enumerate(ff_lines):
    ax.text(ff_x, ff_y - 0.6 * (j + 1), line, fontsize=9, color='#555555')

# ===================== LEGEND =====================
legend_elements = [
    mpatches.Patch(facecolor='#1A5276', alpha=0.88, label='Strong favorite'),
    mpatches.Patch(facecolor='#922B21', alpha=0.88, label='Close / underdog'),
    plt.Line2D([0], [0], color=GOLD, linewidth=3.5, label='Path to final'),
    plt.Line2D([0], [0], marker='o', markersize=8, color='#C0392B',
               markerfacecolor='#C0392B', label='Win probability'),
]
ax.legend(handles=legend_elements, loc='lower center', fontsize=11,
          framealpha=0.95, edgecolor='#DDD', ncol=4,
          bbox_to_anchor=(0.5, -0.05))

# ===================== SAVE =====================
# Save at multiple resolutions for different uses
plt.savefig(os.path.join(REPORT_DIR, '04_tournament_bracket.png'), dpi=400,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved PNG at 400 DPI")

plt.savefig(os.path.join(REPORT_DIR, '04_tournament_bracket_hd.png'), dpi=600,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved HD PNG at 600 DPI")

plt.close()
print("Done - bracket image regenerated")
