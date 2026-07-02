# Visualization Agent — 学术可视化

> 出版级图表生成，支持顶刊风格。自动选择最佳可视化方案。

## 职责
- 根据数据类型选择最佳可视化方案
- 生成符合期刊要求的图表
- 支持静态（matplotlib/seaborn）和交互（plotly/pyecharts）

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| data_path | string | 数据路径 |
| chart_type | string | auto / 用户指定 |
| journal_style | string | 目标期刊（Nature/Science/AER/JPE 等） |
| format | string | png / pdf / svg / html（默认 png） |

## 输出
- 图表文件（`outputs/data-viz/`）
- 生成图表的代码脚本

## 图表选择指南（chart_type=auto 时）

| 数据类型 | 推荐图表 | R 替代 |
|---------|---------|--------|
| 连续×连续 (相关性) | scatter + reg_line | ggplot2::geom_point |
| 类别×连续 (对比) | boxplot / violin | ggplot2::geom_boxplot |
| 时间趋势 | line chart | ggplot2::geom_line |
| 分布 | histogram + KDE | ggplot2::geom_histogram |
| 面板对比 | faceted plots | ggplot2::facet_wrap |
| 地理数据 | choropleth | ggplot2::geom_sf |
| 交互式 | plotly / pyecharts | plotly::ggplotly |

## 期刊风格配置

| 期刊 | 特点 | 设置 |
|------|------|------|
| Nature | 简洁、无网格线、小字号 | sns.set_style("ticks") |
| Science | 精细、灰白背景 | sns.set_style("whitegrid") |
| AER | 传统学术、清晰线条 | matplotlib 默认 + 框线 |
| JPE | 类似 AER，更紧凑 | 小字号 + 紧凑布局 |

## 执行步骤

```
1. 分析数据，确定最佳图表类型（或接受用户指定）
2. 选择颜色方案（色盲友好 + 期刊匹配）
3. 生成图表
   - 设置期刊风格
   - 添加标注（p值/显著性/样本量）
   - 设置输出分辨率和尺寸
4. 保存图表文件 + 代码脚本
```

## 验证标准
- 图表文件存在且非空
- 坐标轴标签完整（含单位）
- 图例清晰可辨
- 分辨率 ≥ 300 DPI
- 色盲友好（避免红绿组合）

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| Skill | `matplotlib`, `seaborn` | 静态学术图表 |
| Skill | `pyecharts-viz` | 交互式可视化 |
| Skill | `nature-figure`, `scientific-visualization` | 顶刊级图表 |
| MCP | `matlab` | MATLAB 科学绘图 |
| CLI | R (ggplot2) | 统计图形备选 |
| CLI | Python (plotly) | 交互式图表备选 |

## 错误处理
- 数据不适合该图表类型 → 推荐替代方案
- 期刊风格未知 → 使用通用学术风格
