# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

FIGURES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES_DIR))


def test_theme_size_export_and_audit(tmp_path):
    from export_figure import apply_mcm_style, figure_size, new_panel_figure, save_figure
    from figure_audit import audit_export, audit_figure

    apply_mcm_style(theme='journal')
    assert figure_size('single')[0] == pytest.approx(89 / 25.4)
    assert figure_size('one_half')[0] == pytest.approx(136 / 25.4)
    with pytest.raises(ValueError):
        apply_mcm_style(theme='unknown')
    with pytest.raises(ValueError):
        figure_size('wide')
    fig, axes = new_panel_figure(1, 2)
    for axis in axes:
        axis.plot([0, 1], [0, 1])
        axis.set(xlabel='X (unit)', ylabel='Y (unit)')
    assert audit_figure(fig).passed
    paths = save_figure(fig, tmp_path / 'figure', formats=('pdf', 'svg', 'png', 'tiff'), close=False)
    assert len(paths) == 4
    for path in paths:
        assert audit_export(path).passed
    plt.close(fig)


def test_stat_helpers():
    from stat_helpers import (classification_curves, confidence_interval,
                              density_profile, five_number_summary,
                              pareto_mask, standardize_matrix)

    mean, low, high = confidence_interval([1, 2, 3, 4])
    assert low < mean < high
    assert pareto_mask([[1, 4], [2, 2], [4, 1], [3, 4]]).tolist() == [True, True, True, False]
    assert standardize_matrix([[1, 2], [3, 4]], 'minmax').shape == (2, 2)
    curves = classification_curves([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert curves['roc_auc'] == pytest.approx(1)
    # 新增统计函数
    assert five_number_summary([1, 2, 3, 4, 5]) == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert five_number_summary([3, 1, 2])[2] == 2.0
    grid, pdf = density_profile([1, 2, 3, 4, 5])
    assert grid.shape == (128,) and pdf.shape == (128,)
    assert np.all(pdf >= 0) and pdf.max() > 0
    # 值域外截断为 0
    edge, edge_pdf = density_profile([1.0, 1.5, 2.0])
    assert edge_pdf[0] == 0.0 and edge_pdf[-1] == 0.0
    # 零方差组退化为全零 pdf
    assert density_profile([5.0, 5.0, 5.0])[1].max() == 0.0
    # 非法输入
    with pytest.raises(ValueError):
        confidence_interval([1])
    with pytest.raises(ValueError):
        pareto_mask([[1, np.nan], [2, 3]])
    with pytest.raises(ValueError):
        classification_curves([1, 1], [0.2, 0.8])
    with pytest.raises(ValueError):
        five_number_summary([1.0])
    with pytest.raises(ValueError):
        five_number_summary([1.0, np.nan])
    with pytest.raises(ValueError):
        density_profile([1.0])
    with pytest.raises(ValueError):
        density_profile([1.0, np.nan, 3.0])


def test_existing_api_smoke():
    from template_bar import make_grouped_bar, make_stacked_bar
    from template_errorbar import make_errorbar
    from template_heatmap import make_heatmap
    from template_line import make_line_plot
    from template_scatter import make_scatter_fit
    from template_tornado import make_tornado

    fig, axes = plt.subplots(2, 3)
    make_line_plot(axes[0, 0], [1, 2, 3], {'A': [2, 3, 5]})
    make_grouped_bar(axes[0, 1], ['A', 'B'], {'X': [1, 2], 'Y': [2, 3]})
    make_stacked_bar(axes[0, 2], ['A', 'B'], {'X': [1, 2], 'Y': [2, 1]})
    make_errorbar(axes[1, 0], [0, 1], [2, 3], [0.2, 0.3])
    make_heatmap(axes[1, 1], [[1, .2], [.2, 1]])
    make_scatter_fit(axes[1, 2], np.arange(6), np.arange(6) + .1)
    plt.close(fig)
    fig, ax = plt.subplots()
    make_tornado(ax, ['A', 'B'], [8, 9], [12, 11], 10)
    plt.close(fig)


def test_new_core_templates_smoke():
    from template_classification import make_confusion_matrix, make_pr_curve, make_roc_curve
    from template_convergence import make_convergence
    from template_diagnostics import make_calibration_plot, make_error_distribution, make_qq_plot, make_residual_plot
    from template_distribution import make_distribution, make_ecdf
    from template_forecast import make_forecast
    from template_multicriteria import make_parallel_coordinates, make_ranked_dot
    from template_pareto import make_pareto_front
    from template_schedule import make_gantt
    from template_waterfall import make_waterfall

    fig, axes = plt.subplots(4, 4)
    groups = {'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]}
    make_distribution(axes[0, 0], groups)
    make_ecdf(axes[0, 1], groups)
    x = np.arange(8); observed = np.arange(8, dtype=float); observed[5:] = np.nan
    forecast = np.arange(8, dtype=float) + .2
    make_forecast(axes[0, 2], x, observed, forecast,
                  {'95% CI': (forecast - 1, forecast + 1)}, split=4.5)
    make_residual_plot(axes[0, 3], np.arange(8), np.linspace(-1, 1, 8))
    make_qq_plot(axes[1, 0], np.linspace(-2, 2, 12))
    make_calibration_plot(axes[1, 1], np.arange(12), np.arange(12) + .3, bins=4)
    make_error_distribution(axes[1, 2], np.linspace(-2, 2, 20))
    make_roc_curve(axes[1, 3], {'M': ([0, .2, 1], [0, .8, 1], .8)})
    make_pr_curve(axes[2, 0], {'M': ([0, .5, 1], [1, .8, .4], .7)}, prevalence=.4)
    make_confusion_matrix(axes[2, 1], [[8, 2], [1, 9]], ['N', 'P'], normalize='true')
    make_pareto_front(axes[2, 2], [1, 2, 3, 4], [5, 3, 2, 4])
    make_convergence(axes[2, 3], [1, 2, 3], {'A': [5, 3, 2]})
    make_waterfall(axes[3, 0], ['A', 'B'], [3, -1], start=10)
    make_ranked_dot(axes[3, 1], ['A', 'B'], [.7, .8])
    make_parallel_coordinates(axes[3, 2], [[1, 2], [2, 1]], ['C1', 'C2'], ['A', 'B'])
    make_gantt(axes[3, 3], ['A', 'B'], [0, 1], [2, 3])
    plt.close(fig)


def test_absorbed_templates_smoke():
    """新增吸收图型：箱线、小提琴、雷达、PCA。"""
    from template_distribution import make_boxplot, make_violin
    from template_pca import make_pca
    from template_radar import make_radar

    groups = {'A': [1, 2, 3, 4, 5], 'B': [2, 3, 4, 5, 6]}
    fig, axes = plt.subplots(1, 2)
    make_boxplot(axes[0], groups)
    make_violin(axes[1], groups)
    plt.close(fig)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    make_radar(ax, ['指标一', '指标二', '指标三'], {'方案A': [0.7, 0.8, 0.6]})
    plt.close(fig)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    X = rng.normal(size=(20, 4))
    make_pca(ax, X, None, group_ids=np.repeat([0, 1], 10))
    plt.close(fig)

    # 非法输入：雷达指标不足 3 个、PCA 样本不足 2 行
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    with pytest.raises(ValueError):
        make_radar(ax, ['仅两个'], {'方案A': [0.5, 0.6]})
    plt.close(fig)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        make_pca(ax, [[1.0]], None)
    plt.close(fig)


@pytest.mark.parametrize('factory', ['distribution', 'forecast', 'confusion', 'schedule'])
def test_new_template_invalid_inputs(factory):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        if factory == 'distribution':
            from template_distribution import make_distribution
            make_distribution(ax, {})
        elif factory == 'forecast':
            from template_forecast import make_forecast
            make_forecast(ax, [1], [1, 2], [1], {})
        elif factory == 'confusion':
            from template_classification import make_confusion_matrix
            make_confusion_matrix(ax, [[1, -1], [2, 3]], ['A', 'B'])
        else:
            from template_schedule import make_gantt
            make_gantt(ax, ['A'], [0], [-1])
    plt.close(fig)


def test_source_inspired_templates_smoke():
    """新增源项目吸收图型：Sobol、搜索轨迹、相图、时空场、情景分解。"""
    from template_phase import make_phase
    from template_scenario import make_scenario_hist
    from template_searchpath import make_searchpath
    from template_sobol import make_sobol
    from template_spacetime import make_spacetime, make_spacetime_compare

    fig, axes = plt.subplots(1, 2)
    make_sobol(axes[0], ['A', 'B', 'C'], [.3, .2, .1], [.4, .3, .15],
               [.03, .03, .02], [.04, .04, .03])
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(100, 10, 300), rng.normal(130, 12, 200)])
    levels = np.concatenate([np.zeros(300, dtype=int), np.ones(200, dtype=int)])
    make_scenario_hist(axes[1], values, levels)
    plt.close(fig)

    fig, ax = plt.subplots()
    pts = np.array([[2.0, 2.0], [1.6, 1.7], [1.2, 1.3], [0.6, 0.8],
                    [0.2, 0.3], [0.05, 0.02]])
    make_searchpath(ax, lambda p: p[:, 0] ** 2 + p[:, 1] ** 2,
                    (-3, 3), (-3, 3), pts)
    plt.close(fig)

    fig, ax = plt.subplots()
    make_phase(ax, lambda x, y: (-y, x), (-2, 2, -2, 2),
               {'轨迹': np.array([[2.0, 0.0], [1.4, 1.4], [0.0, 2.0],
                                  [-1.4, 1.4]])},
               equilibria=[(0.0, 0.0)])
    plt.close(fig)

    x = np.linspace(0, 1, 30)
    t = np.linspace(0, 1, 24)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    field_true = np.sin(xx * np.pi) * np.exp(-tt)
    field_pred = 0.9 * field_true
    axes_c = make_spacetime_compare(field_true, field_pred, x, t)
    plt.close(axes_c[0].figure)
    fig, ax = plt.subplots()
    make_spacetime(ax, field_true, x, t)
    plt.close(fig)

    # 非法输入：S1 > ST、field 形状不匹配、情景层非 0/1
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        make_sobol(ax, ['A'], [.5], [.3])
    plt.close(fig)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        make_spacetime(ax, field_true[:-1], x, t)
    plt.close(fig)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        make_scenario_hist(ax, values, levels + 2)
    plt.close(fig)


REFERENCE_MARK = '虚线参照线'
INTERVAL_MARK = '无配对区间'


def _audit_hits(fig, mark):
    from figure_audit import audit_figure
    return [w for w in audit_figure(fig).warnings if mark in w]


def test_audit_ignores_unhashable_linestyle():
    """LineCollection 的 linestyle 是 [(offset, [on, off])]，元组含 list 不可哈希。

    早期实现用 `in set` 判定虚线，任何虚线 ax.hlines/ax.vlines/fill_between 都会
    抛 TypeError；该异常还能从 save_figure(audit=True) 公共路径逸出。
    """
    from figure_audit import _linestyle_is_dashed
    assert _linestyle_is_dashed('--') and not _linestyle_is_dashed('-')
    assert _linestyle_is_dashed([(0.0, [5.55, 2.4])])
    assert not _linestyle_is_dashed([(0.0, None)])

    for build in (
        lambda ax: ax.hlines([0.5], 0, 1, linestyles='--'),
        lambda ax: ax.vlines([0.5], 0, 1, linestyles=':'),
        lambda ax: ax.fill_between([0, 1], [0, 1], [1, 2], ls='--'),
    ):
        fig, ax = plt.subplots()
        build(ax)
        ax.set(xlabel='t (s)', ylabel='y (m)')
        from figure_audit import audit_figure
        audit_figure(fig)  # 不抛异常即通过
        plt.close(fig)


def test_reference_line_audit():
    """虚线阈值线：无标注告警；邻近文字/图例/角注引值/数据曲线均放过。"""
    def build(ax, **kw):
        ax.plot([0, 1], [0, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='t (s)', ylabel='y (m)')

    fig, ax = plt.subplots(); build(ax); ax.axhline(0.5, ls='--')
    assert _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    fig, ax = plt.subplots(); build(ax); ax.axhline(0.5, ls='--')
    ax.text(0.05, 0.52, '阈值')
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    # 角注远离线，但引用了线自身的数值
    fig, ax = plt.subplots(); build(ax); ax.axvline(0.5, ls='--')
    ax.text(0.99, 0.02, '基准 = 0.5', transform=ax.transAxes)
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    # 角注引用的是别的数值 -> 仍是未标注
    fig, ax = plt.subplots(); build(ax); ax.axvline(0.5, ls='--')
    ax.text(0.99, 0.02, '基准 = 0.9', transform=ax.transAxes)
    assert _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    # 未 draw 的 annotate 必须按数据坐标判定（其 transform 非 transData）
    fig, ax = plt.subplots(); build(ax); ax.axhline(0.5, ls='--')
    ax.annotate('阈值', xy=(0.5, 0.5), xytext=(0.5, 0.5))
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    # 虚线数据曲线、虚线网格、带 label 的参照线都不是阈值线
    fig, ax = plt.subplots(); build(ax); ax.plot([0, 1], [1, 0], ls='--')
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    fig, ax = plt.subplots(); build(ax); ax.grid(ls='--')
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    fig, ax = plt.subplots(); build(ax); ax.axhline(0.5, ls='--', label='基准')
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)

    # eventplot 是数据事件，不是阈值线
    fig, ax = plt.subplots(); build(ax); ax.eventplot([[0.3, 0.7]], linestyles='--')
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)


def test_paired_interval_audit():
    """标注均值却无配对区间的分组柱状图告警；直方图、误差指标、带 CI 均放过。"""
    def bars(ax, x, text):
        ax.bar(x, [1, 2, 3])
        ax.text(x[0], 2.5, text)
        ax.set(xlabel='g', ylabel='s (pt)')

    fig, ax = plt.subplots(); bars(ax, ['A', 'B', 'C'], '均值=2')
    assert _audit_hits(fig, INTERVAL_MARK)
    plt.close(fig)

    # 数值 x 轴（年份等）同样属于分组柱，不能当作直方图跳过
    fig, ax = plt.subplots(); bars(ax, [2020, 2021, 2022], '均值=2')
    assert _audit_hits(fig, INTERVAL_MARK)
    plt.close(fig)

    # CJK 与 mean 粘连时仍应识别
    fig, ax = plt.subplots(); bars(ax, ['A', 'B', 'C'], '总体mean=1.5')
    assert _audit_hits(fig, INTERVAL_MARK)
    plt.close(fig)

    # 直方图 bin 紧贴，不是分组柱
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    ax.hist(rng.normal(size=200), bins=10)
    ax.text(0, 20, '均值=0')
    ax.set(xlabel='v (m)', ylabel='count (n)')
    assert not _audit_hits(fig, INTERVAL_MARK)
    plt.close(fig)

    # 误差指标不是"均值"声称
    fig, ax = plt.subplots(); bars(ax, ['A', 'B', 'C'], 'Mean Absolute Error = 0.32')
    assert not _audit_hits(fig, INTERVAL_MARK)
    plt.close(fig)

    # 已带误差棒则放过
    fig, ax = plt.subplots()
    ax.bar(['A', 'B', 'C'], [1, 2, 3], yerr=[.1, .2, .3])
    ax.text(0, 2.5, '均值=2')
    ax.set(xlabel='g', ylabel='s (pt)')
    assert not _audit_hits(fig, INTERVAL_MARK)
    plt.close(fig)


def test_tornado_corner_caption_is_not_a_false_positive():
    """tornado demo 用角注解释基准线，不得被判为"无标注"。"""
    from template_tornado import make_tornado
    fig, ax = plt.subplots()
    make_tornado(ax, ['A', 'B'], [8, 9], [12, 11], 10)
    assert not _audit_hits(fig, REFERENCE_MARK)
    plt.close(fig)


def test_optional_templates_smoke():
    nx = pytest.importorskip('networkx')
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box
    from template_flow import make_sankey
    from template_map import make_choropleth, make_route_map
    from template_network import make_network

    fig, axes = plt.subplots(1, 3)
    graph = nx.Graph(); graph.add_edge('A', 'B')
    make_network(axes[0], graph, positions={'A': (0, 0), 'B': (1, 0)})
    make_sankey(axes[1], ['A', 'A'], ['B', 'C'], [3, 2])
    regions = gpd.GeoDataFrame({'value': [1, 2]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    make_choropleth(axes[2], regions, 'value')
    plt.close(fig)
    fig, ax = plt.subplots()
    make_route_map(ax, [[0, 0], [1, 1]], [[0, 1]])
    plt.close(fig)
