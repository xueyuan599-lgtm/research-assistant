---
name: pyecharts-viz
description: Generate show-stopping interactive visualizations using pyecharts (ECharts Python wrapper). Timelines, 3D charts, geo maps, sankey, bar race, heatmap calendars, and more. Outputs standalone HTML files — no server needed.
---

# pyecharts-viz

Generate **炫技级（show-stopping）interactive visualizations** using pyecharts v2.x.

All output is standalone `.html` — double-click to open in any browser. Works offline.

## When to use

- You want **交互式图表** for a presentation / defence / website
- Static matplotlib/seaborn isn't impressive enough
- You need **动态时间轴动画** (bar race, map over years)
- You need **3D charts**, **地理热力图**, **桑基图**, **词云**
- You want to show **多维数据** in a single interactive page

## Trigger keywords

- "pyecharts"
- "炫酷可视化"
- "交互式图表"
- "HTML图表"
- "动态可视化"
- "bar race"
- "时间轴动画"
- "热力图交互"
- "地理可视化"

## Requirements

```bash
pip install pyecharts
```

## Quick reference

### Chart type catalog

| Category | Type | Function | Best for |
|----------|------|----------|----------|
| **Timeline** | 动态时间轴 | `Timeline()` + any chart | 年份演变、竞赛图 |
| **Bar** | 柱状图 | `Bar()` | 排名对比 |
| **Bar (3D)** | 3D 柱状图 | `Bar3D()` | 炫技首图 |
| **Line** | 折线图 | `Line()` | 趋势 |
| **Scatter** | 散点图 | `Scatter()` | 相关性 |
| **Scatter3D** | 3D 散点 | `Scatter3D()` | 三维分布 |
| **Surface3D** | 3D 曲面 | `Surface3D()` | 地形/曲面 |
| **Pie** | 饼图 | `Pie()` | 占比 |
| **Radar** | 雷达图 | `Radar()` | 多指标对比 |
| **Heatmap** | 热力图 | `HeatMap()` | 矩阵值 |
| **Calendar** | 日历热力 | `Calendar()` | 时间序列密度 |
| **Geo** | 地理散点 | `Geo()` | 城市分布 |
| **Map** | 地图 | `Map()` | 省/市行政地图 |
| **Map (heat)** | 地图热力 | `BMap/BMapHeatmap()` | 地理密度 |
| **EffectScatter** | 涟漪散点 | `EffectScatter()` | 强调点位 |
| **Sankey** | 桑基图 | `Sankey()` | 流量/路径 |
| **WordCloud** | 词云 | `WordCloud()` | 关键词权重 |
| **Graph** | 关系图 | `Graph()` | 网络/聚类关系 |
| **Liquid** | 水球图 | `Liquid()` | 完成率/百分比 |
| **Gauge** | 仪表盘 | `Gauge()` | 单值指标 |
| **Parallel** | 平行坐标 | `Parallel()` | 多维度对比 |
| **ThemeRiver** | 主题河流 | `ThemeRiver()` | 连续时间多类别 |
| **Sunburst** | 旭日图 | `Sunburst()` | 层级占比 |
| **Treemap** | 矩形树图 | `Treemap()` | 层级+面积 |
| **Funnel** | 漏斗图 | `Funnel()` | 转化/递减 |
| **PictorialBar** | 象形柱图 | `PictorialBar()` | 创意柱状 |

### Theme / style

```python
from pyecharts.globals import ThemeType

# Built-in themes (dark is most impressive for demos):
"light", "dark", "chalk", "essos", "infographic",
"macarons", "purple-passion", "roma", "romantic",
"shine", "valentine", "vintage", "walden", "westeros",
"wonderland"
```

Usage: `Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))`

Grid layout: Use `Grid()` for multi-panel dashboards in one page.

### Page composition

```python
from pyecharts.charts import Page

page = Page(layout=Page.DraggablePageLayout)
page.add(chart1, chart2, chart3)
page.render("dashboard.html")
# Drag-and-drop rearrange, then press "Save" to persist layout
```

## Template files

The `templates/` directory contains ready-to-run scripts:

| File | Description |
|------|-------------|
| `bar_race.py` | 动态条形竞赛图（Timeline + Bar） |
| `geo_scatter.py` | 地理散点 + 涟漪效果 |
| `radar_profiles.py` | 雷达图多指标对比 |
| `sankey_flow.py` | 桑基图流量路径 |
| `graph_cluster.py` | 聚类关系图 |
| `calendar_heatmap.py` | 日历热力图 |
| `surface3d_demo.py` | 3D 曲面图 |
| `wordcloud_demo.py` | 词云 |
| `all_in_one.py` | 多图合一大盘 |

## Output

- All scripts output `.html` in the current working directory
- Open in any browser, zero server setup
- Pass `--port` to start a local server: `python script.py --port 8080`

## Rules

1. Always use `ThemeType.DARK` or `ThemeType.CHALK` for maximum visual impact
2. Always set `init_opts=opts.InitOpts(theme=..., width="...", height="...")`
3. Enable tooltip (`tooltip_opts=opts.TooltipOpts(trigger="axis")`) on every chart
4. Data labels on by default for bar/line/scatter
5. Output filename includes chart type so the user can identify it
6. Use Chinese labels/annotations unless otherwise specified
7. For timeline: set `is_auto_play=True, is_loop_play=False`
