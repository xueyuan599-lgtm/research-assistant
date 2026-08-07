# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
    from stat_helpers import classification_curves, confidence_interval, pareto_mask, standardize_matrix

    mean, low, high = confidence_interval([1, 2, 3, 4])
    assert low < mean < high
    assert pareto_mask([[1, 4], [2, 2], [4, 1], [3, 4]]).tolist() == [True, True, True, False]
    assert standardize_matrix([[1, 2], [3, 4]], 'minmax').shape == (2, 2)
    curves = classification_curves([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert curves['roc_auc'] == pytest.approx(1)
    with pytest.raises(ValueError):
        confidence_interval([1])
    with pytest.raises(ValueError):
        pareto_mask([[1, np.nan], [2, 3]])
    with pytest.raises(ValueError):
        classification_curves([1, 1], [0.2, 0.8])


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
