"""
动态条形竞赛图 (Bar Race)
pyecharts Timeline + Bar

经典的"炫技"图 —— 展示各城市韧性得分排名随年份的演变
"""

from pyecharts.charts import Bar, Timeline
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import pandas as pd

# ── 数据 ──────────────────────────────────────────────────────
# 替换为你的 excel/csv 路径，或直接手填数据
df = pd.read_excel('城市生态韧性得分.xlsx', header=None)
df.columns = ['city', 'year', 'score']

years = sorted(df['year'].unique())
cities = sorted(df['city'].unique())

# ── 颜色映射 ──────────────────────────────────────────────────
COLORS = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00',
          '#56B4E9', '#F0E442', '#000000', '#D55E00', '#0072B2', '#009E73']
city_color = {c: COLORS[i % len(COLORS)] for i, c in enumerate(cities)}

# ── Timeline ──────────────────────────────────────────────────
tl = Timeline(
    init_opts=opts.InitOpts(theme=ThemeType.DARK, width="1000px", height="600px")
)

for year in years:
    sub = df[df['year'] == year].sort_values('score')
    names = sub['city'].tolist()
    scores = [round(v, 3) for v in sub['score'].tolist()]
    colors = [city_color[n] for n in names]

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        .add_xaxis(names)
        .add_yaxis(
            '韧性得分',
            scores,
            itemstyle_opts=opts.ItemStyleOpts(color=colors),
            label_opts=opts.LabelOpts(position='right', font_size=12),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f'浙江省城市生态韧性排名 ({year}年)',
                pos_left='center',
                title_textstyle_opts=opts.TextStyleOpts(font_size=18, color='#ffffff'),
            ),
            tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='shadow'),
            xaxis_opts=opts.AxisOpts(
                name='韧性得分',
                name_location='center',
                axislabel_opts=opts.LabelOpts(font_size=11),
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(font_size=11),
            ),
        )
        .set_series_opts(label_opts=opts.LabelOpts(position='right'))
        .reversal_axis()
    )
    tl.add(bar, str(year))

tl.add_schema(
    is_auto_play=True,
    play_interval=800,
    is_loop_play=False,
    pos_bottom='5px',
    pos_left='center',
    label_opts=opts.LabelOpts(is_show=True, font_size=14),
)

# ── 保存 ──────────────────────────────────────────────────────
tl.render('bar_race_城市韧性排名.html')
print('Saved: bar_race_城市韧性排名.html')
