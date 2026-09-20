"""
地理散点 + 涟漪效果图
展示各城市韧性得分的地理分布，带年份时间轴
"""

from pyecharts.charts import Geo, Timeline
from pyecharts import options as opts
from pyecharts.globals import ThemeType, GeoType
import pandas as pd

# ── 城市经纬度（浙江省）─────────────────────────────────────────
CITY_COORDS = {
    '杭州市': [120.155, 30.274],
    '宁波市': [121.544, 29.868],
    '温州市': [120.699, 28.002],
    '嘉兴市': [120.755, 30.747],
    '湖州市': [120.088, 30.893],
    '绍兴市': [120.580, 30.030],
    '金华市': [119.648, 29.079],
    '衢州市': [118.875, 28.937],
    '舟山市': [122.207, 30.016],
    '台州市': [121.421, 28.653],
    '丽水市': [119.923, 28.451],
}

# ── 数据（替换为你的数据源）────────────────────────────────────
df = pd.read_excel('城市生态韧性得分.xlsx', skiprows=1, header=None)
df.columns = ['city', 'year', 'score']

years = sorted(df['year'].unique())

# ── Timeline ──────────────────────────────────────────────────
tl = Timeline(
    init_opts=opts.InitOpts(theme=ThemeType.CHALK, width="1000px", height="700px")
)

for year in years:
    sub = df[df['year'] == year]
    data_pairs = list(zip(sub['city'], [round(v, 3) for v in sub['score']]))

    geo = (
        Geo(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
        .add_schema(
            maptype='浙江',
            itemstyle_opts=opts.ItemStyleOpts(
                area_color='#1a1a2e',
                border_color='#3a3a5c',
            ),
            label_opts=opts.LabelOpts(is_show=True, color='#ccc', font_size=10),
        )
        # 涟漪散点
        .add(
            '韧性得分',
            data_pairs,
            type_=GeoType.EFFECT_SCATTER,
            symbol_size=12,
            color='#ff6b35',
            effect_opts=opts.EffectOpts(scale=8, period=4),
            label_opts=opts.LabelOpts(is_show=False),
        )
        # 热力图背景层
        .add(
            '热力',
            data_pairs,
            type_=GeoType.HEATMAP,
            is_selected=False,
            radius=40,
            blur_size=10,
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f'浙江省城市生态韧性地理分布 ({year}年)',
                pos_left='center',
                title_textstyle_opts=opts.TextStyleOpts(font_size=18, color='#ffffff'),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger='item',
                formatter='{b}: {c}',
            ),
            visualmap_opts=opts.VisualMapOpts(
                is_show=True,
                min_=round(sub['score'].min(), 2),
                max_=round(sub['score'].max(), 2),
                range_color=[
                    '#2d436b', '#3b6e8f', '#48a0a0', '#6bc9a0',
                    '#a8dda0', '#f0e68c', '#f4a460', '#e06c6c',
                ],
                pos_left='left',
                pos_top='center',
            ),
        )
    )

    # 注册坐标
    for city, coords in CITY_COORDS.items():
        geo.add_coordinate(city, coords[0], coords[1])

    tl.add(geo, str(year))

tl.add_schema(
    is_auto_play=True,
    play_interval=1000,
    is_loop_play=False,
    pos_bottom='10px',
)

tl.render('geo_韧性分布.html')
print('Saved: geo_韧性分布.html')
