"""
关系图 (Graph) — 聚类关系网络
展示各城市之间的聚类距离关系
"""

from pyecharts.charts import Graph
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# ── 聚类分组 ──────────────────────────────────────────────────
clusters = {
    '高值领先区': {
        'cities': ['杭州市', '宁波市'],
        'color': '#D55E00',
    },
    '中值提升区': {
        'cities': ['丽水市', '嘉兴市', '湖州市', '台州市', '金华市'],
        'color': '#0072B2',
    },
    '过渡平衡区': {
        'cities': ['舟山市', '衢州市', '绍兴市', '温州市'],
        'color': '#009E73',
    },
}

# ── 构建 nodes ────────────────────────────────────────────────
nodes = []
node_idx = {}
i = 0
for cluster_name, info in clusters.items():
    for city in info['cities']:
        nodes.append({
            'name': city,
            'symbolSize': 30,
            'itemStyle': {'color': info['color']},
            'category': cluster_name,
        })
        node_idx[city] = i
        i += 1

# ── 构建 links（同簇内两两相连，不同簇按强度连）────────────────
links = []
strength_map = {
    ('杭州市', '宁波市'): 0.95,
    ('丽水市', '嘉兴市'): 0.60, ('丽水市', '湖州市'): 0.65,
    ('丽水市', '台州市'): 0.75, ('丽水市', '金华市'): 0.70,
    ('嘉兴市', '湖州市'): 0.80, ('嘉兴市', '台州市'): 0.55,
    ('嘉兴市', '金华市'): 0.60, ('湖州市', '台州市'): 0.55,
    ('湖州市', '金华市'): 0.60, ('台州市', '金华市'): 0.65,
    ('舟山市', '衢州市'): 0.50, ('舟山市', '绍兴市'): 0.55,
    ('舟山市', '温州市'): 0.45, ('衢州市', '绍兴市'): 0.60,
    ('衢州市', '温州市'): 0.55, ('绍兴市', '温州市'): 0.65,
}

for (c1, c2), strength in strength_map.items():
    links.append({
        'source': c1,
        'target': c2,
        'value': round(strength, 2),
        'lineStyle': {'width': strength * 4, 'opacity': strength * 0.6, 'curveness': 0.2},
    })

# ── categories ────────────────────────────────────────────────
categories = [
    {'name': '高值领先区', 'itemStyle': {'color': '#D55E00'}},
    {'name': '中值提升区', 'itemStyle': {'color': '#0072B2'}},
    {'name': '过渡平衡区', 'itemStyle': {'color': '#009E73'}},
]

# ── 绘图 ──────────────────────────────────────────────────────
graph = (
    Graph(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="1000px", height="700px"))
    .add(
        '',
        nodes,
        links,
        categories,
        repulsion=400,
        edge_length=220,
        is_draggable=True,
        is_roam=True,
        is_focusnode=True,
        gravity=0.3,
        friction=0.1,
        layout='force',
        label_opts=opts.LabelOpts(
            is_show=True,
            position='right',
            font_size=12,
            color='#ffffff',
        ),
        linestyle_opts=opts.LineStyleOpts(
            color='source',
            opacity=0.4,
            width=1,
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger='item',
            formatter='{b}: {c}',
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title='浙江省城市聚类关系网络',
            subtitle='力导向布局 — 连线越粗关联越强',
            pos_left='center',
            title_textstyle_opts=opts.TextStyleOpts(font_size=20, color='#ffffff'),
            subtitle_textstyle_opts=opts.TextStyleOpts(font_size=13, color='#aaaaaa'),
        ),
        legend_opts=opts.LegendOpts(
            pos_left='5%',
            pos_top='5%',
            textstyle_opts=opts.TextStyleOpts(font_size=12, color='#cccccc'),
        ),
    )
)

graph.render('graph_聚类关系网络.html')
print('Saved: graph_聚类关系网络.html')
