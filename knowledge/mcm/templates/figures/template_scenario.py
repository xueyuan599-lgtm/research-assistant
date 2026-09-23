# -*- coding: utf-8 -*-
"""Simulation-decomposition histogram: output distribution split by scenarios."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_scenario_hist(ax, values, scenario_levels, bin_edges=40,
                       level_labels=('情景一', '情景二'),
                       xlabel='输出指标 (–)', ylabel='频数 (次)',
                       median_line=True, legend_loc='upper left',
                       stacked=True):
    """Stack a histogram of ``values`` split by binary scenario levels.

    ``scenario_levels`` 与 ``values`` 等长，取值 0/1（可传布尔或整数）。
    0=情景一、1=情景二；SimDec 约定情景色取 palette 扩展三色。
    """
    from palette import SCIENCE, SCENARIO_EXTRA

    values = np.asarray(values, dtype=float)
    levels = np.asarray(scenario_levels)
    if values.ndim != 1 or values.size < 2:
        raise ValueError('values 必须是至少两个元素的一维数组')
    if levels.shape != (values.size,):
        raise ValueError('scenario_levels 必须与 values 等长')
    if not np.all(np.isfinite(values)):
        raise ValueError('values 包含 NaN 或无穷值')
    unique = np.unique(levels.astype(int))
    if not set(unique) <= {0, 1}:
        raise ValueError('scenario_levels 必须只包含 0 与 1')

    if len(level_labels) != 2:
        raise ValueError('level_labels 必须恰好两个标签')
    colors = [SCIENCE['蓝'], SCENARIO_EXTRA['情景二']]

    if bin_edges is None:
        bins = None
    elif np.isscalar(bin_edges):
        bins = int(bin_edges)
    else:
        bins = np.asarray(bin_edges, dtype=float)
        if bins.ndim != 1 or bins.size < 2:
            raise ValueError('bin_edges 必须是标量箱数或单调递增数组')
    data = [values[levels.astype(int) == level] for level in (0, 1)]
    if any(group.size == 0 for group in data):
        raise ValueError('至少一个情景层无样本')

    ax.hist(data, bins=bins, stacked=stacked,
            color=colors, edgecolor='white', linewidth=0.35,
            label=list(level_labels), alpha=0.92, zorder=3)
    if median_line:
        span = values.max() - values.min() or 1.0
        for offset, (level, label, color) in enumerate(
                zip((0, 1), level_labels, colors)):
            median = float(np.median(data[level]))
            linestyle = '--' if offset else '-.'
            ax.axvline(median, color=color, linewidth=0.85,
                       linestyle=linestyle, zorder=4,
                       label=f'{label}中位数 = {median:.2f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.02)
    ax.legend(loc=legend_loc)
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.78)
    rng = np.random.default_rng(11)
    zh = language == 'zh'
    # 情景一：需求偏低的正态；情景二：需求偏高 + 厚尾。
    low_demand = rng.normal(118, 12, 900)
    high_demand = rng.normal(142, 17, 600) + rng.exponential(8, 600)
    values = np.concatenate([low_demand, high_demand])
    levels = np.concatenate([np.zeros(900, dtype=int),
                             np.ones(600, dtype=int)])
    make_scenario_hist(
        ax, values, levels,
        level_labels=('情景一：需求偏低', '情景二：需求偏高') if zh
        else ('Scenario 1: low demand', 'Scenario 2: high demand'),
        xlabel='季度利润 (万元)' if zh else 'Quarterly profit (10k CNY)',
        ylabel='频数 (次)' if zh else 'Frequency',
        legend_loc='upper left')
    ax.set_title('利润分布的情景分解（Monte Carlo, n=1500）' if zh
                 else 'Scenario decomposition of profit (Monte Carlo, n=1500)')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('scenario_demo' if zh else 'scenario_demo_en')))


if __name__ == '__main__':
    demo()
