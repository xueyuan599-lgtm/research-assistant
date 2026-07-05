# -*- coding: utf-8 -*-
"""只生成浙江省区位图：突出丽水莲都位置"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

out_dir = r'E:\wuyi\数学建模半自动\research-assistant\outputs\field_photos'

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([117.5, 123.5, 27.0, 31.5], crs=ccrs.PlateCarree())

# 底图
ax.add_feature(cfeature.OCEAN, facecolor='#E8F4FD', alpha=0.5, zorder=0)
ax.add_feature(cfeature.LAND, facecolor='#F8F6F0', zorder=1)
ax.add_feature(cfeature.LAKES, facecolor='#E8F4FD', alpha=0.7, zorder=2)
ax.add_feature(cfeature.RIVERS, edgecolor='#B0C4DE', linewidth=0.4, zorder=3)

# 省界
provinces = cfeature.NaturalEarthFeature(
    'cultural', 'admin_1_states_provinces', '10m',
    facecolor='none', edgecolor='#BBBBBB', linewidth=0.6)
ax.add_feature(provinces, zorder=4)
ax.add_feature(cfeature.COASTLINE, edgecolor='#666666', linewidth=1.0, zorder=5)


# 丽水莲都 - 醒目标注
ax.plot(119.92, 28.45, 'o', color='#C41E3A', markersize=14, markeredgecolor='white',
        markeredgewidth=2, zorder=15, transform=ccrs.PlateCarree())

ax.annotate('丽水市莲都区', xy=(119.92, 28.45), xytext=(8, 8), textcoords='offset points',
            fontsize=12, fontweight='bold', color='#C41E3A',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#C41E3A', linewidth=1.5, alpha=0.92),
            transform=ccrs.PlateCarree(), zorder=16)

# 周边主要城市
cities = {
    '杭州': (120.15, 30.28, '#333333', 10),
    '宁波': (121.55, 29.87, '#333333', 10),
    '温州': (120.70, 28.00, '#333333', 10),
    '绍兴': (120.58, 30.03, '#333333', 9),
    '金华': (119.65, 29.08, '#333333', 9),
    '义乌': (120.07, 29.31, '#1565C0', 11),
    '衢州': (118.87, 28.94, '#333333', 8),
    '台州': (121.42, 28.66, '#333333', 8),
    '舟山': (122.21, 30.02, '#333333', 9),
}

for name, (lon, lat, color, size) in cities.items():
    ax.plot(lon, lat, 'o', color=color, markersize=size*0.7, zorder=10,
            transform=ccrs.PlateCarree())
    bold = (name == '义乌')
    ax.annotate(name, xy=(lon, lat), xytext=(5, 5), textcoords='offset points',
                fontsize=size, color=color, fontweight='bold' if bold else 'normal',
                transform=ccrs.PlateCarree(), zorder=11)

# 相邻省市标注
neighbors = {
    '上海市': (121.47, 31.23),
    '江苏省': (119.5, 31.3),
    '安徽省': (118.2, 30.0),
    '江西省': (117.8, 28.5),
    '福建省': (119.0, 27.2),
}
for name, (lon, lat) in neighbors.items():
    ax.annotate(name, xy=(lon, lat), fontsize=9, color='#999999', alpha=0.7,
                ha='center', transform=ccrs.PlateCarree(), zorder=3)

# 东海标注
ax.annotate('东     海', xy=(122.5, 29.0), fontsize=12, color='#87CEEB',
            alpha=0.6, style='italic', ha='center', transform=ccrs.PlateCarree(), zorder=3)

# 经纬度网格
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.2, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8, 'color': '#888888'}
gl.ylabel_style = {'size': 8, 'color': '#888888'}

ax.set_title('浙江省区位与丽水莲都调研位置', fontsize=14, fontweight='bold', pad=12)

# 底部说明
fig.text(0.5, 0.02, '底图来源：自然资源部标准地图服务（审图号：GS(2024)0650号）',
         fontsize=7, color='#AAAAAA', ha='center', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fpath = os.path.join(out_dir, 'zhejiang_locations.png')
fig.savefig(fpath, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'图片已保存至: {fpath}')
