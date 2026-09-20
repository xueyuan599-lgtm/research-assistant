"""
雷达图 — 多维度指标对比
展示各城市在资源(R)、环境(S)、治理(P)、创新(I) 四个维度的得分
"""

from pyecharts.charts import Radar, Page
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import numpy as np

# ── 维度定义 ──────────────────────────────────────────────────
# 四个维度：资源、环境、治理、创新（每个维度取全省均值）
indicators = [
    {'name': '资源 (R)', 'max': 1},
    {'name': '环境 (S)', 'max': 1},
    {'name': '治理 (P)', 'max': 1},
    {'name': '创新 (I)', 'max': 1},
]

# ── 示例数据（替换为你的实际数据）────────────────────────────────
# 每行: [R, S, P, I] 平均值
cities_data = {
    '杭州市': [0.85, 0.72, 0.78, 0.92],
    '宁波市': [0.78, 0.68, 0.75, 0.85],
    '温州市': [0.55, 0.62, 0.58, 0.52],
    '嘉兴市': [0.65, 0.60, 0.62, 0.58],
    '湖州市': [0.62, 0.65, 0.60, 0.50],
    '绍兴市': [0.58, 0.55, 0.56, 0.55],
    '金华市': [0.60, 0.52, 0.54, 0.62],
    '衢州市': [0.52, 0.58, 0.50, 0.45],
    '舟山市': [0.56, 0.70, 0.52, 0.48],
    '台州市': [0.62, 0.58, 0.60, 0.55],
    '丽水市': [0.68, 0.75, 0.55, 0.42],
}

# 聚类分组（与论文一致）
clusters = {
    '高值领先区': ['杭州市', '宁波市'],
    '中值提升区': ['丽水市', '嘉兴市', '湖州市', '台州市', '金华市'],
    '过渡平衡区': ['舟山市', '衢州市', '绍兴市', '温州市'],
}

CLUSTER_COLORS = {
    '高值领先区': '#D55E00',
    '中值提升区': '#0072B2',
    '过渡平衡区': '#009E73',
}

page = Page(layout=Page.DraggablePageLayout)

for cluster_name, members in clusters.items():
    color = CLUSTER_COLORS[cluster_name]

    radar = (
        Radar(init_opts=opts.InitOpts(
            theme=ThemeType.DARK,
            width="600px", height="500px",
        ))
        .add_schema(
            schema=indicators,
            shape='polygon',
            center=['50%', '55%'],
            radius='65%',
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.06),
            ),
            axisline_opt=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(color='rgba(255,255,255,0.3)')
            ),
            splitline_opt=opts.SplitLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(color='rgba(255,255,255,0.15)'),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=cluster_name,
                pos_left='center',
                title_textstyle_opts=opts.TextStyleOpts(
                    font_size=16, color=color,
                ),
            ),
            tooltip_opts=opts.TooltipOpts(trigger='item'),
            legend_opts=opts.LegendOpts(
                pos_left='left',
                textstyle_opts=opts.TextStyleOpts(font_size=11, color='#ccc'),
            ),
        )
    )

    for city in members:
        values = cities_data.get(city, [0, 0, 0, 0])
        radar.add(
            city,
            [values],
            color=color,
            linestyle_opts=opts.LineStyleOpts(width=1.5, opacity=0.7),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.05),
            label_opts=opts.LabelOpts(is_show=False),
        )

    page.add(radar)

page.render('radar_聚类雷达对比.html')
print('Saved: radar_聚类雷达对比.html')