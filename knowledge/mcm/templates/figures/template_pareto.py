# -*- coding: utf-8 -*-
"""Two-objective Pareto-front template."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def make_pareto_front(ax, objective_x, objective_y, efficient=None,
                      xlabel='目标一', ylabel='目标二', labels=None,
                      minimize=(True, True), highlight=None):
    """Plot feasible solutions and connect the non-dominated frontier."""
    try:
        from .export_figure import style_axis
        from .palette import SCIENCE_LIST
        from .stat_helpers import pareto_mask
    except ImportError:
        from export_figure import style_axis
        from palette import SCIENCE_LIST
        from stat_helpers import pareto_mask
    x = np.asarray(objective_x, dtype=float); y = np.asarray(objective_y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2 or not np.all(np.isfinite(x + y)):
        raise ValueError('两个目标必须是至少含两个有限值的等长一维数组')
    efficient = pareto_mask(np.column_stack([x, y]), minimize=minimize) if efficient is None else np.asarray(efficient, dtype=bool)
    if efficient.shape != x.shape:
        raise ValueError('efficient 形状必须与目标数组一致')
    ax.scatter(x[~efficient], y[~efficient], s=14, color='#B8B8B8', label='可行解', alpha=0.65)
    frontier_order = np.argsort(x[efficient])
    ax.plot(x[efficient][frontier_order], y[efficient][frontier_order], color=SCIENCE_LIST[1],
            marker='o', markerfacecolor='white', label='Pareto 前沿')
    if highlight is not None:
        index = int(highlight)
        if index < 0 or index >= x.size:
            raise ValueError('highlight 索引越界')
        ax.scatter(x[index], y[index], s=45, marker='*', color=SCIENCE_LIST[2], zorder=4)
    if labels is not None:
        if len(labels) != x.size:
            raise ValueError('labels 长度必须与解数量一致')
        for index in np.flatnonzero(efficient):
            ax.annotate(str(labels[index]), (x[index], y[index]), xytext=(3, 3),
                        textcoords='offset points', fontsize=5.5)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style_axis(ax, grid='both')
    ax.legend()
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    rng = np.random.default_rng(8)
    x = np.linspace(10, 100, 45)
    y = 130 - 0.9 * x + 0.006 * x ** 2 + rng.normal(0, 9, x.size)
    zh = language == 'zh'
    fig, ax = new_figure(aspect=0.72)
    make_pareto_front(ax, x, y, xlabel='成本（万元）' if zh else 'Cost (10k CNY)',
                      ylabel='风险（%）' if zh else 'Risk (%)', highlight=14)
    if not zh:
        legend = ax.get_legend(); legend.texts[0].set_text('Feasible'); legend.texts[1].set_text('Pareto front')
    output = THIS_DIR / 'samples' / (stem or ('pareto_demo' if zh else 'pareto_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
