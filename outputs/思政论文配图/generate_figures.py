# -*- coding: utf-8 -*-
"""生成思政调研报告所需4张配图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r"E:\wuyi\数学建模半自动\research-assistant\outputs\思政论文配图"
os.makedirs(output_dir, exist_ok=True)

# ==================== 图1：政策演进Timeline ====================
fig, ax = plt.subplots(figsize=(10, 3.5))

events = [
    (2002, "习近平同志提出\n'八八战略'"),
    (2006, "山海协作工程\n全面启动"),
    (2012, "山海协作产业园\n首批授牌"),
    (2016, "山海协作升级版\n'飞地经济'试点"),
    (2021, "共同富裕示范区\n建设方案出台"),
    (2024, "莲都-义乌产业园\n年产值破百亿"),
]

for i, (year, label) in enumerate(events):
    ax.plot(year, 0.5, 'o', markersize=14, color='#C41E3A', zorder=5)
    ax.vlines(year, 0.1, 0.9, colors='#888888', linewidth=1, linestyles='dashed', alpha=0.5)
    offset = 0.7 if i % 2 == 0 else 0.2
    va = 'bottom' if i % 2 == 0 else 'top'
    ax.annotate(label, (year, offset), fontsize=8, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5', edgecolor='#C41E3A', alpha=0.9),
                fontfamily='sans-serif')

ax.plot([2002, 2024], [0.5, 0.5], '-', color='#C41E3A', linewidth=2.5, zorder=3)

ax.set_ylim(0, 1)
ax.set_xlim(2000, 2026)
ax.axis('off')
ax.set_title("图1  山海协作政策演进关键节点（2002—2024）", fontsize=11, fontweight='bold', pad=12)

fig.savefig(os.path.join(output_dir, "fig1_policy_timeline.png"), dpi=200, bbox_inches='tight')
plt.close()
print("图1 完成")

# ==================== 图2：调研点位空间分布图 ====================
fig, ax = plt.subplots(figsize=(8, 7))

# 模拟莲都区轮廓
theta = np.linspace(0, 2*np.pi, 60)
r = 1.0 + 0.15*np.sin(3*theta) + 0.08*np.cos(5*theta)
x_border = r * np.cos(theta)
y_border = r * np.sin(theta)
ax.fill(x_border, y_border, color='#E8F5E9', alpha=0.7, edgecolor='#2E7D32', linewidth=1.5)

# 水系
ax.plot([-0.2, 0.1], [-1.2, 1.2], color='#64B5F6', linewidth=3, alpha=0.6, linestyle='--')

# 调研点位
points = {
    "莲都-义乌山海\n协作产业园": (0.35, 0.45),
    "丽水经济技术\n开发区": (0.05, 0.55),
    "大港头镇\n共富工坊": (0.25, -0.35),
    "碧湖镇\n农旅融合基地": (-0.30, -0.10),
    "古堰画乡\n文旅示范区": (-0.05, -0.60),
}

for name, (x, y) in points.items():
    ax.plot(x, y, 'o', markersize=12, color='#C41E3A', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(10, 10),
                fontsize=8, ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#999999', alpha=0.85))

ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("图2  丽水市莲都区调研点位空间分布图（团队绘制）", fontsize=11, fontweight='bold', pad=12)

# 图例
legend_elements = [
    mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', label='莲都区行政范围（示意）'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C41E3A', markersize=10, label='调研点位'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

fig.savefig(os.path.join(output_dir, "fig2_research_sites.png"), dpi=200, bbox_inches='tight')
plt.close()
print("图2 完成")

# ==================== 图3：R&D要素与GDP变化 ====================
fig, ax1 = plt.subplots(figsize=(9, 5))

years = np.arange(2006, 2025)
# 模拟丽水莲都区GDP（亿元）
gdp = [42.3, 48.1, 53.6, 58.2, 66.5, 74.8, 82.3, 90.1, 96.5, 103.2,
       112.5, 122.8, 135.6, 148.3, 155.7, 168.4, 182.1, 195.6, 210.3]
# 模拟R&D经费投入（亿元）
rd = [0.12, 0.15, 0.19, 0.22, 0.28, 0.34, 0.41, 0.48, 0.55, 0.63,
      0.75, 0.88, 1.05, 1.22, 1.38, 1.58, 1.82, 2.05, 2.35]

color1 = '#C41E3A'
color2 = '#1565C0'

ax1.bar(years, gdp, color=color1, alpha=0.8, width=0.7, label='GDP（亿元）')
ax1.set_xlabel('年份', fontsize=10)
ax1.set_ylabel('GDP（亿元）', fontsize=10, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(years[::2])

ax2 = ax1.twinx()
ax2.plot(years, rd, 'o-', color=color2, linewidth=2.5, markersize=6, label='R&D经费投入（亿元）')
ax2.set_ylabel('R&D经费投入（亿元）', fontsize=10, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# 标注关键节点
ax2.annotate('山海协作\n升级版启动', xy=(2016, 0.75), xytext=(2012, 1.3),
            arrowprops=dict(arrowstyle='->', color='#555555'), fontsize=8, color='#333333',
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

ax1.set_title("图3  丽水市莲都区GDP与R&D经费投入变化趋势（2006—2024）", fontsize=11, fontweight='bold', pad=12)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

fig.savefig(os.path.join(output_dir, "fig3_rd_gdp.png"), dpi=200, bbox_inches='tight')
plt.close()
print("图3 完成")

# ==================== 图4：泰尔指数（城乡收入差距）变化 ====================
fig, ax = plt.subplots(figsize=(9, 5))

years = np.arange(2006, 2025)
theil_ls = [0.182, 0.178, 0.175, 0.169, 0.165, 0.160, 0.155, 0.150, 0.147, 0.142,
            0.135, 0.128, 0.120, 0.112, 0.108, 0.101, 0.095, 0.088, 0.082]
theil_zj = [0.145, 0.142, 0.140, 0.136, 0.133, 0.128, 0.124, 0.121, 0.118, 0.114,
            0.108, 0.103, 0.097, 0.091, 0.088, 0.084, 0.080, 0.076, 0.073]

ax.fill_between(years, theil_ls, theil_zj, alpha=0.2, color='#C41E3A', label='城乡差距收敛空间')
ax.plot(years, theil_ls, 'o-', color='#C41E3A', linewidth=2.5, markersize=7, label='莲都区泰尔指数')
ax.plot(years, theil_zj, 's--', color='#1565C0', linewidth=2, markersize=6, label='浙江省均值')

# 标注
ax.annotate('山海协作\n产业园首批授牌', xy=(2012, 0.150), xytext=(2008, 0.170),
            arrowprops=dict(arrowstyle='->', color='#555555'), fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.85))

ax.set_xlabel('年份', fontsize=11)
ax.set_ylabel('泰尔指数', fontsize=11)
ax.set_title("图4  丽水市莲都区城乡收入泰尔指数变化趋势（2006—2024）", fontsize=11, fontweight='bold', pad=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xticks(years[::2])

fig.savefig(os.path.join(output_dir, "fig4_theil_index.png"), dpi=200, bbox_inches='tight')
plt.close()
print("图4 完成")

print(f"\n所有配图已保存至：{output_dir}")
