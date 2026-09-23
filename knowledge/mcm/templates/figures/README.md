# MCM Science / Nature 期刊级可视化库

面向 CUMCM、MCM/ICM 与科研论文的可复现静态图表库。默认主题强调信息密度、可编辑矢量输出、色盲与灰度可读性；原粗黑双框教程风格保留为 `legacy` 主题。

规范参考：[Nature Research Figure Guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) 与 [Science/AAAS Research Author Guide](https://spj.science.org/page/research/for-authors)。

## 快速使用

```python
from export_figure import apply_mcm_style, new_figure, save_figure
from template_forecast import make_forecast

apply_mcm_style(theme='journal')
fig, ax = new_figure(width='single', aspect=0.72)
make_forecast(ax, x, observed, forecast, intervals)
save_figure(fig, 'outputs/figures/forecast')
```

旧调用保持有效：`apply_mcm_style()`、所有已有 `make_*()` 和 `plot_from_excel()` 均无需修改。

## 主题与导出

| 设置 | `journal`（默认） | `legacy` |
|---|---|---|
| 用途 | 论文、竞赛正文 | 教程复现、粗黑双框组合图 |
| 字体 | Arial/Helvetica + 中文回退 | Times New Roman/DejaVu Serif |
| 字号 | 正文 5–7 pt；面板编号 8 pt | 11–14 pt |
| 轴线 | 0.25–1 pt，弱网格 | 2 pt 以上完整黑框 |
| 画布 | 89/136/183 mm | 7 × 6.5 inch |

`save_figure()` 支持 PDF、SVG、PNG、TIFF；默认 PDF/SVG/PNG，位图最低 300 dpi。PDF/SVG 保留可编辑文字，TIFF 使用 LZW 压缩。

## 图表选型矩阵

| MCM 问题 | 推荐函数 | 模块 |
|---|---|---|
| 连续趋势与情景 | `make_line_plot()` | `template_line.py` |
| 类别比较与构成 | `make_grouped_bar()` / `make_stacked_bar()` | `template_bar.py` |
| 均值与不确定性 | `make_errorbar()` | `template_errorbar.py` |
| 分布与组间差异 | `make_distribution()` / `make_ecdf()` | `template_distribution.py` |
| 分布对比（箱线） | `make_boxplot()` | `template_distribution.py` |
| 分布+密度（小提琴） | `make_violin()` | `template_distribution.py` |
| 预测与区间 | `make_forecast()` | `template_forecast.py` |
| 回归诊断 | `make_residual_plot()` / `make_qq_plot()` / `make_calibration_plot()` | `template_diagnostics.py` |
| 分类评估 | `make_roc_curve()` / `make_pr_curve()` / `make_confusion_matrix()` | `template_classification.py` |
| 相关或矩阵模式 | `make_heatmap()` | `template_heatmap.py` |
| 预测值与实测值 | `make_scatter_fit()` | `template_scatter.py` |
| 单因素灵敏度 | `make_tornado()` | `template_tornado.py` |
| 多目标优化 | `make_pareto_front()` | `template_pareto.py` |
| 算法迭代 | `make_convergence()` | `template_convergence.py` |
| 增减贡献分解 | `make_waterfall()` | `template_waterfall.py` |
| 多指标评价 | `make_ranked_dot()` / `make_parallel_coordinates()` | `template_multicriteria.py` |
| 评价/方案对比（雷达） | `make_radar()` | `template_radar.py` |
| 高维特征降维 | `make_pca()` | `template_pca.py` |
| 项目调度 | `make_gantt()` | `template_schedule.py` |
| 网络与关键路径 | `make_network()` | `template_network.py` |
| 资源或能量流向 | `make_sankey()` | `template_flow.py` |
| 区域差异与路线 | `make_choropleth()` / `make_route_map()` | `template_map.py` |
| 真实三维轨迹 | `make_3d_trajectory()` | `template_3d.py` |
| 累计量 + 阶段时序 | `make_bar_timeseries()` | `template_bar_timeseries.py` |

不默认提供普通饼图或装饰性 3D 柱图，分别优先使用 100% 堆叠柱和二维比较。雷达图仅在指标同量纲或已标准化时使用（`make_radar()`，指标过多建议用 `make_parallel_coordinates()`）。

## 统计与绘图分离

底层模板只绘制整理后的数据。`stat_helpers.py` 提供：

- `confidence_interval()`：Student-t 置信区间。
- `classification_curves()`：ROC/PR 曲线与 AUC/AP。
- `pareto_mask()`：非支配解判定。
- `standardize_matrix()`：多指标 z-score/min-max 标准化。
- `five_number_summary()`：min/Q1/中位数/Q3/max 五数概括。
- `density_profile()`：Gaussian KDE 的 (grid, pdf) 密度剖面。

## 多面板与质量审查

```python
from export_figure import new_panel_figure, shared_legend
from figure_audit import audit_figure, audit_export

fig, axes = new_panel_figure(2, 2, width='double')
shared_legend(fig, axes)
report = audit_figure(fig, require_units=True)
```

审查项目包括：画布高度、字体、线宽、颜色数量、坐标轴标签、单位提示、位图 DPI、颜色模式和 SVG 可编辑文字。

## Excel 组合图

```python
from template_bar_timeseries import plot_from_excel

plot_from_excel(
    'data.xlsx', 'outputs/emission_composite.png',
    time_col='Time', group_columns=('LF', 'HF'),
    replicate_col=None, cumulative_scale=1.0,
    stage_boundaries=(4, 6),
)
```

该模板保持上下两层完整黑框、内部彩色分隔线、原始散点、误差棒和显著性标记。

## 安装与验证

```bash
pip install -r requirements-figures-core.txt
pip install -r requirements-figures-extras.txt  # 地图与网络
pytest knowledge/mcm/templates/figures/tests -q
python knowledge/mcm/templates/figures/render_all.py
```

`render_all.py` 会临时渲染并审查 20 类模板，生成中英文图册后自动清理单张示例，仅在 `samples/` 保留 `catalog_zh.png` 与 `catalog_en.png`。

## 期刊制图底线

- 画布按最终版面尺寸创建，不事后拉伸。
- 坐标轴标注单位，正文不在图内放长标题。
- 类别超过 5 个时使用分面、排序或合并“其他”。
- 多系列同时使用颜色与线型/标记，避免仅靠颜色区分。
- 连续热图不用彩虹色；有正负中心时使用发散色图。
- 图形与文字优先导出 PDF/SVG；照片和高密度栅格使用 300 dpi 以上 PNG/TIFF。
