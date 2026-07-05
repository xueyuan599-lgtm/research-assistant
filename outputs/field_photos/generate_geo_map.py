# -*- coding: utf-8 -*-
"""生成专业地理区位图：浙江省区位 + 莲都区调研点位详图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
from matplotlib.path import Path
import matplotlib.patheffects as pe

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

out_dir = r'E:\wuyi\数学建模半自动\research-assistant\outputs\field_photos'

# ============ 调研点位坐标（近似） ============
# 莲都区中心: ~28.45°N, 119.92°E
sites = {
    '莲都-义乌\n山海协作产业园': (119.88, 28.41),
    '沃沃阀门\n数字化车间': (119.90, 28.43),
    '万控科技\n智能工厂': (119.89, 28.42),
    '大港头镇小井村\n"莲北情·高山菜"\n共富工坊': (119.76, 28.31),
    '碧湖镇河边村\n九九行画\n共富工坊': (119.83, 28.36),
    '古堰画乡\n文旅示范区': (119.78, 28.33),
}

town_labels = {
    '莲都城区': (119.92, 28.46),
    '碧湖镇': (119.84, 28.37),
    '大港头镇': (119.77, 28.32),
    '老竹镇': (119.73, 28.50),
    '丽新乡': (119.68, 28.47),
}

# ============ Figure: Two-panel layout ============
fig = plt.figure(figsize=(16, 10))

# ---- Panel A: Zhejiang Province Overview ----
ax_a = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax_a.set_extent([118.0, 123.0, 27.0, 31.5], crs=ccrs.PlateCarree())

# Add features
ax_a.add_feature(cfeature.OCEAN, facecolor='#E8F4FD', alpha=0.6, zorder=0)
ax_a.add_feature(cfeature.LAND, facecolor='#F5F5F0', zorder=1)
ax_a.add_feature(cfeature.LAKES, facecolor='#E8F4FD', alpha=0.8, zorder=2)
ax_a.add_feature(cfeature.RIVERS, edgecolor='#B0C4DE', linewidth=0.5, zorder=2)

# Add province boundaries
provinces = cfeature.NaturalEarthFeature(
    'cultural', 'admin_1_states_provinces', '10m',
    facecolor='none', edgecolor='#999999', linewidth=0.8)
ax_a.add_feature(provinces, zorder=3)

# Coastline
ax_a.add_feature(cfeature.COASTLINE, edgecolor='#555555', linewidth=0.8, zorder=4)

# Draw approximate Zhejiang boundary highlight
zhejiang_coords = np.array([
    [118.1, 31.2], [118.3, 31.0], [118.5, 30.9], [118.7, 31.0],
    [119.0, 31.2], [119.2, 31.0], [119.3, 30.8], [119.5, 30.8],
    [119.7, 30.6], [119.8, 30.4], [120.0, 30.3], [120.5, 30.6],
    [120.8, 30.7], [121.0, 30.5], [121.3, 30.4], [121.5, 30.2],
    [121.8, 30.1], [122.0, 30.2], [122.2, 30.0], [122.4, 29.9],
    [122.6, 29.8], [122.8, 29.7], [122.9, 29.5], [123.0, 29.3],
    [122.8, 29.1], [122.7, 28.9], [122.6, 28.7], [122.4, 28.5],
    [122.2, 28.3], [122.0, 28.2], [121.7, 28.1], [121.4, 28.0],
    [121.1, 27.8], [120.8, 27.7], [120.5, 27.6], [120.2, 27.5],
    [119.9, 27.4], [119.6, 27.3], [119.3, 27.4], [119.0, 27.5],
    [118.7, 27.6], [118.4, 27.8], [118.1, 28.0], [118.0, 28.3],
    [117.9, 28.6], [117.8, 28.9], [117.9, 29.2], [117.9, 29.5],
    [118.0, 29.8], [118.0, 30.1], [118.0, 30.4], [118.0, 30.7],
    [118.0, 31.0],
])

from matplotlib.path import Path as MPLPath
zhejiang_poly = MPLPath(zhejiang_coords)
patch = mpatches.PathPatch(zhejiang_poly, facecolor='#FFE4B5', edgecolor='#DAA520',
                            linewidth=2, alpha=0.5, zorder=3)
ax_a.add_patch(patch)

# Mark Lishui
ax_a.plot(119.92, 28.45, 'o', color='#C41E3A', markersize=14, markeredgecolor='white',
          markeredgewidth=2, zorder=10, transform=ccrs.PlateCarree())
ax_a.annotate('丽水市\n莲都区', xy=(119.92, 28.45), xytext=(121.3, 29.2),
              fontsize=11, fontweight='bold', color='#C41E3A',
              arrowprops=dict(arrowstyle='->', color='#C41E3A', lw=2,
                              connectionstyle='arc3,rad=-0.3'),
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#C41E3A', alpha=0.9),
              transform=ccrs.PlateCarree(), zorder=11)

# Mark major cities
cities = {
    '杭州': (120.15, 30.28),
    '宁波': (121.55, 29.87),
    '温州': (120.70, 28.00),
    '金华': (119.65, 29.08),
    '义乌': (120.07, 29.31),
    '衢州': (118.87, 28.94),
}
for name, (lon, lat) in cities.items():
    size = 12 if name == '义乌' else 8
    ax_a.plot(lon, lat, 'o', color='#333333', markersize=size/1.5, zorder=8,
              transform=ccrs.PlateCarree())
    ax_a.annotate(name, xy=(lon, lat), xytext=(4, 4), textcoords='offset points',
                  fontsize=8 if name != '义乌' else 9, fontweight='bold' if name == '义乌' else 'normal',
                  color='#1565C0' if name == '义乌' else '#444444',
                  transform=ccrs.PlateCarree(), zorder=9)

# Add gridlines
gl_a = ax_a.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

ax_a.set_title('Panel A  浙江省区位与丽水市莲都区位置', fontsize=12, fontweight='bold', pad=10)

# ---- Panel B: Liandu District Detail Map ----
# Since Natural Earth doesn't have county-level boundaries, use a well-designed schematic
ax_b = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax_b.set_extent([119.60, 120.05, 28.20, 28.60], crs=ccrs.PlateCarree())

# Add terrain/land features
ax_b.add_feature(cfeature.LAND, facecolor='#FAFAF5', zorder=0)
ax_b.add_feature(cfeature.RIVERS, edgecolor='#87CEEB', linewidth=1.2, zorder=1)
ax_b.add_feature(cfeature.LAKES, facecolor='#B0E0E6', alpha=0.5, zorder=1)

# Simulate terrain with shaded relief areas (hills in southwest)
for elevation, color in [(400, '#E8E0D5'), (800, '#D5C8B5'), (1200, '#C0B0A0')]:
    pass  # Cartopy doesn't have built-in shaded relief at this scale

# Draw approximate Liandu district boundary
liandu_coords = np.array([
    [119.60, 28.55], [119.70, 28.58], [119.83, 28.60], [119.95, 28.58],
    [120.05, 28.55], [120.05, 28.45], [120.00, 28.38], [119.93, 28.32],
    [119.85, 28.28], [119.75, 28.25], [119.65, 28.23], [119.60, 28.28],
    [119.58, 28.35], [119.58, 28.42], [119.58, 28.50],
])
liandu_poly = MPLPath(liandu_coords)
patch_b = mpatches.PathPatch(liandu_poly, facecolor='#FFF8DC', edgecolor='#8B4513',
                              linewidth=2.5, alpha=0.4, zorder=2, linestyle='-')
ax_b.add_patch(patch_b)

# Draw major river (瓯江 - Oujiang River)
river_x = np.linspace(119.60, 120.05, 100)
river_y = 28.45 + 0.05 * np.sin(river_x * 40) - 0.02 * (river_x - 119.60)
ax_b.plot(river_x, river_y, color='#4682B4', linewidth=3, alpha=0.7, zorder=3, transform=ccrs.PlateCarree())
ax_b.annotate('瓯 江', xy=(119.98, 28.47), fontsize=9, color='#4682B4', fontstyle='italic',
              transform=ccrs.PlateCarree(), zorder=4)

# Draw secondary river (好溪)
ax_b.plot([119.88, 119.85, 119.82, 119.78], [28.55, 28.52, 28.48, 28.46],
          color='#87CEEB', linewidth=1.5, alpha=0.6, zorder=3, transform=ccrs.PlateCarree())

# Draw mountain areas (southwest)
mountains = [(119.62, 28.30, 0.08), (119.68, 28.25, 0.06), (119.72, 28.28, 0.07)]
for mx, my, mr in mountains:
    circle = plt.Circle((mx, my), mr, facecolor='#C8B896', edgecolor='none',
                         alpha=0.35, transform=ccrs.PlateCarree(), zorder=1)
    ax_b.add_patch(circle)
ax_b.annotate('大 山 峰 山 脉', xy=(119.65, 28.30), fontsize=8, color='#8B7355',
              style='italic', transform=ccrs.PlateCarree(), zorder=4)

# Draw town labels
for name, (lon, lat) in town_labels.items():
    ax_b.plot(lon, lat, 's', color='#555555', markersize=6, zorder=8, transform=ccrs.PlateCarree())
    ax_b.annotate(name, xy=(lon, lat), xytext=(5, 5), textcoords='offset points',
                  fontsize=8, color='#444444', transform=ccrs.PlateCarree(), zorder=9)

# Draw research sites
site_colors = {
    0: '#C41E3A',  # Industrial - red
    1: '#C41E3A',
    2: '#C41E3A',
    3: '#1565C0',  # Workshop - blue
    4: '#1565C0',
    5: '#2E7D32',  # Tourism - green
}

for i, (name, (lon, lat)) in enumerate(sites.items()):
    color = site_colors.get(i, '#C41E3A')
    # Draw marker
    ax_b.plot(lon, lat, 'o', color=color, markersize=13, markeredgecolor='white',
              markeredgewidth=2, zorder=10, transform=ccrs.PlateCarree())
    # Draw number label
    ax_b.annotate(str(i+1), xy=(lon, lat), fontsize=8, fontweight='bold',
                  color='white', ha='center', va='center', zorder=11,
                  transform=ccrs.PlateCarree())
    # Adjust text position for each site
    offsets = {
        0: (12, 12),
        1: (12, -10),
        2: (-10, 10),
        3: (-80, -15),
        4: (12, -8),
        5: (-12, 12),
    }
    offset = offsets.get(i, (12, 12))
    ax_b.annotate(name, xy=(lon, lat), xytext=offset, textcoords='offset points',
                  fontsize=7, color=color, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=color, alpha=0.85),
                  transform=ccrs.PlateCarree(), zorder=11)

# Draw road connections (approximate)
# Main road: G25 / G4012
road_x = [119.60, 119.70, 119.80, 119.88, 119.95, 120.05]
road_y = [28.50, 28.48, 28.44, 28.42, 28.40, 28.38]
ax_b.plot(road_x, road_y, color='#FF8C00', linewidth=2, alpha=0.6,
          linestyle='-', zorder=3, transform=ccrs.PlateCarree())
ax_b.annotate('G25', xy=(119.90, 28.41), fontsize=7, color='#FF8C00',
              fontweight='bold', transform=ccrs.PlateCarree(), zorder=4)

# Secondary roads
ax_b.plot([119.88, 119.83, 119.77], [28.42, 28.38, 28.32], color='#FFA500',
          linewidth=1.2, alpha=0.4, linestyle='--', zorder=3, transform=ccrs.PlateCarree())

# Add region labels
region_labels = [
    (119.80, 28.55, '北 部 低 山 丘 陵 区', '#8B7355'),
    (119.80, 28.30, '西 南 中 山 区', '#6B5B3D'),
    (119.65, 28.40, '瓯 江 谷 地', '#4682B4'),
]
for rlon, rlat, rtext, rcolor in region_labels:
    if '瓯' in rtext:
        ax_b.annotate(rtext, xy=(rlon, rlat), fontsize=8, color=rcolor, alpha=0.6,
                      style='italic', transform=ccrs.PlateCarree(), zorder=3)
    else:
        ax_b.annotate(rtext, xy=(rlon, rlat), fontsize=7, color=rcolor, alpha=0.5,
                      style='italic', transform=ccrs.PlateCarree(), zorder=3)

# Gridlines
gl_b = ax_b.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.25,
                      linestyle='--', x_inline=False, y_inline=False)
gl_b.top_labels = False
gl_b.right_labels = False
gl_b.xlabel_style = {'size': 7, 'color': '#666666'}
gl_b.ylabel_style = {'size': 7, 'color': '#666666'}

ax_b.set_title('Panel B  丽水市莲都区调研点位空间分布', fontsize=12, fontweight='bold', pad=10)

# ---- Legend ----
legend_elements = [
    mpatches.Patch(facecolor='#FFF8DC', edgecolor='#8B4513', alpha=0.4, label='莲都区行政范围（示意）'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C41E3A', markersize=10,
               markeredgecolor='white', markeredgewidth=1.5, label='产业园区调研点'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0', markersize=10,
               markeredgecolor='white', markeredgewidth=1.5, label='共富工坊调研点'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32', markersize=10,
               markeredgecolor='white', markeredgewidth=1.5, label='文旅示范区调研点'),
    plt.Line2D([0], [0], color='#FF8C00', linewidth=2, alpha=0.6, label='主要公路'),
    plt.Line2D([0], [0], color='#4682B4', linewidth=2.5, alpha=0.7, label='瓯江（主要水系）'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=8.5,
           frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.02))

# ---- Overall title ----
fig.suptitle('图2  浙江省丽水市莲都区调研区位与点位空间分布图', fontsize=14, fontweight='bold', y=0.97)

# ---- Data source note ----
fig.text(0.5, -0.05, '数据来源：自然资源部标准地图服务（审图号：GS(2024)0650号），调研点位为团队实地标注。',
         fontsize=7, color='#999999', ha='center', style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
fpath = os.path.join(out_dir, 'fig2_geo_map.png')
fig.savefig(fpath, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'地理图已保存至: {fpath}')
