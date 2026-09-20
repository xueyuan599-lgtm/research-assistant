"""
桑基图 — 障碍因子流量
展示各阶段障碍因子 → 维度的路径与权重
"""

from pyecharts.charts import Sankey
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# ── 数据：因子 → 维度 映射 ────────────────────────────────────
# 替换为你的实际障碍度数据
dim_map = {
    'R1': '资源', 'R2': '资源', 'R3': '资源', 'R4': '资源',
    'S1': '环境', 'S2': '环境', 'S3': '环境', 'S4': '环境',
    'P1': '治理', 'P2': '治理', 'P3': '治理', 'P4': '治理',
    'I1': '创新', 'I2': '创新', 'I3': '创新', 'I4': '创新',
}

# 示例：2025 年各因子障碍度 (%)
obstacle_2025 = {
    'R1': 4.2, 'R2': 4.5, 'R3': 5.3, 'R4': 5.2,
    'S1': 3.7, 'S2': 3.2, 'S3': 5.4, 'S4': 2.9,
    'P1': 2.5, 'P2': 7.5, 'P3': 2.0, 'P4': 5.5,
    'I1': 3.0, 'I2': 11.0, 'I3': 14.0, 'I4': 12.5,
}

# ── 构建 nodes + links ────────────────────────────────────────
nodes = []
node_set = set()

# 因子节点
for factor in obstacle_2025.keys():
    if factor not in node_set:
        nodes.append({'name': factor})
        node_set.add(factor)

# 维度节点
for dim in ['资源', '环境', '治理', '创新']:
    if dim not in node_set:
        nodes.append({'name': dim})
        node_set.add(dim)

# 汇总各维度总障碍度
dim_totals = {}
for factor, val in obstacle_2025.items():
    dim = dim_map[factor]
    dim_totals[dim] = dim_totals.get(dim, 0) + val

links = []
for factor, val in obstacle_2025.items():
    links.append({'source': factor, 'target': dim_map[factor], 'value': round(val, 1)})

# ── 绘图 ──────────────────────────────────────────────────────
sankey = (
    Sankey(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="1200px", height="700px"))
    .add(
        '障碍度',
        nodes,
        links,
        linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.5, color='source'),
        label_opts=opts.LabelOpts(font_size=11, color='#ffffff'),
        node_width=18,
        node_gap=12,
        pos_top='5%',
        pos_bottom='5%',
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title='障碍因子桑基图 (2025年)',
            subtitle='因子 → 维度 流量',
            pos_left='center',
            title_textstyle_opts=opts.TextStyleOpts(font_size=20, color='#ffffff'),
            subtitle_textstyle_opts=opts.TextStyleOpts(font_size=13, color='#aaaaaa'),
        ),
        tooltip_opts=opts.TooltipOpts(trigger='item', trigger_on='mousemove'),
    )
)

sankey.render('sankey_障碍因子流量.html')
print('Saved: sankey_障碍因子流量.html')
