# -*- coding: utf-8 -*-
"""Science-style point-and-interval chart template."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_errorbar(ax, x, means, errors, xlabel='组别', ylabel='数值', unit='',
                  errorbar_type='std', labels=None, connect=False,
                  show_note=True, color='#3B6FB6', grid=True, **kwargs):
    """Draw estimates and uncertainty without implying bar-area magnitude."""
    from export_figure import style_axis

    x = np.asarray(x)
    means = np.asarray(means, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if x.shape != means.shape or errors.shape not in {means.shape, (2, means.size)}:
        raise ValueError('x、means 与 errors 的形状不兼容')
    if np.any(errors < 0):
        raise ValueError('errors 必须为非负数')

    fmt = 'o-' if connect else 'o'
    plot_kw = {
        'fmt': fmt,
        'color': color,
        'ecolor': '#333333',
        'elinewidth': 0.85,
        'capsize': 3,
        'capthick': 0.75,
        'markersize': 4.8,
        'markerfacecolor': 'white',
        'markeredgecolor': color,
        'markeredgewidth': 0.9,
        'zorder': 3,
    }
    plot_kw.update(kwargs)
    ax.errorbar(x, means, yerr=errors, **plot_kw)
    if labels is not None:
        if len(labels) != len(x):
            raise ValueError('labels 长度必须等于 x 长度')
        ax.set_xticks(x, labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'{ylabel} ({unit})' if unit else ylabel)
    style_axis(ax, grid='y' if grid else None)
    if show_note:
        notes = {'std': '均值 ± 标准差', 'se': '均值 ± 标准误', 'ci': '均值与置信区间'}
        ax.text(0.99, 0.02, notes.get(errorbar_type, errorbar_type),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=6.3, color='#6B6B6B')
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.72)
    rng = np.random.default_rng(42)
    zh = language == 'zh'
    scenarios = (['基准情景', '乐观情景', '悲观情景'] if zh
                 else ['Baseline', 'Optimistic', 'Pessimistic'])
    runs = [rng.normal(mean, 8, size=20) for mean in (100, 112, 91)]
    means = [values.mean() for values in runs]
    errors = [1.96 * values.std(ddof=1) / np.sqrt(values.size) for values in runs]
    make_errorbar(ax, np.arange(3), means, errors, xlabel='情景' if zh else 'Scenario',
                  ylabel='总收益' if zh else 'Total return', unit='万元' if zh else '10k CNY', labels=scenarios,
                  errorbar_type='ci')
    ax.set_title('蒙特卡洛结果（95% 置信区间，n = 20）' if zh
                 else 'Monte Carlo result (95% CI, n = 20)')
    if not zh:
        ax.texts[-1].set_text('Mean and confidence interval')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('errorbar_demo' if zh else 'errorbar_demo_en')))


if __name__ == '__main__':
    demo()
