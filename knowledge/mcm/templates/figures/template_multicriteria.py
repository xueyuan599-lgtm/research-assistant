# -*- coding: utf-8 -*-
"""Multi-criteria comparison without radar-chart distortion."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def make_ranked_dot(ax, labels, values, xlabel='综合得分', unit='', highlight=0):
    """Draw a sorted dot ranking with direct values."""
    try:
        from .export_figure import style_axis
        from .palette import SCIENCE_LIST
    except ImportError:
        from export_figure import style_axis
        from palette import SCIENCE_LIST
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or len(labels) != values.size or not np.all(np.isfinite(values)):
        raise ValueError('labels 与 values 必须是等长非空有限序列')
    order = np.argsort(values)
    ordered = values[order]; ordered_labels = np.asarray(labels, dtype=object)[order]
    colors = np.full(values.size, '#A9A9A9', dtype=object)
    if highlight is not None:
        original = int(highlight)
        if original < 0 or original >= values.size:
            raise ValueError('highlight 索引越界')
        colors[np.flatnonzero(order == original)[0]] = SCIENCE_LIST[1]
    y = np.arange(values.size)
    ax.hlines(y, 0, ordered, color='#D0D0D0', linewidth=0.8)
    ax.scatter(ordered, y, color=colors, s=24, zorder=3)
    for position, value in zip(y, ordered):
        ax.annotate(f'{value:.2f}', (value, position), xytext=(4, 0), textcoords='offset points', va='center')
    ax.set_yticks(y, ordered_labels)
    ax.set_xlabel(f'{xlabel} ({unit})' if unit else xlabel)
    style_axis(ax, grid='x')
    ax.set_ylim(-0.6, values.size - 0.4)
    return ax


def make_parallel_coordinates(ax, data, criteria, alternatives, normalize=True,
                              palette='science'):
    """Compare alternatives across criteria using parallel coordinates."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
        from .stat_helpers import standardize_matrix
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette
        from stat_helpers import standardize_matrix
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(alternatives), len(criteria)):
        raise ValueError('data 形状必须为 alternatives × criteria')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('data 包含非有限值')
    display = standardize_matrix(matrix, 'minmax') if normalize else matrix
    colors = get_palette(palette, len(alternatives)); x = np.arange(len(criteria))
    styles = ('-', '--', '-.', ':', (0, (4, 1, 1, 1)))
    for index, (name, row) in enumerate(zip(alternatives, display)):
        ax.plot(x, row, color=colors[index], linestyle=styles[index], marker='o',
                markerfacecolor='white', label=name)
    ax.set_xticks(x, criteria)
    ax.set_ylabel('标准化得分' if normalize else '数值')
    if normalize:
        ax.set_ylim(-0.03, 1.03)
    style_axis(ax, grid='both')
    ax.legend(ncol=min(len(alternatives), 3))
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_panel_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_panel_figure, save_figure
    apply_mcm_style(); zh = language == 'zh'
    alternatives = ['方案A', '方案B', '方案C'] if zh else ['Plan A', 'Plan B', 'Plan C']
    criteria = ['成本', '效益', '稳健性', '公平性'] if zh else ['Cost', 'Benefit', 'Robustness', 'Equity']
    data = np.array([[75, 88, 82, 71], [83, 79, 90, 85], [69, 92, 72, 88]])
    fig, axes = new_panel_figure(1, 2, aspect=0.39)
    make_ranked_dot(axes[0], alternatives, [0.78, 0.86, 0.74],
                    xlabel='综合得分' if zh else 'Composite score', highlight=1)
    make_parallel_coordinates(axes[1], data, criteria, alternatives)
    if not zh:
        axes[1].set_ylabel('Normalized score')
    output = THIS_DIR / 'samples' / (stem or ('multicriteria_demo' if zh else 'multicriteria_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
