# -*- coding: utf-8 -*-
"""Optimization convergence histories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def make_convergence(ax, iterations, histories, ylabel='目标函数值', xlabel='迭代次数',
                     best_known=None, logy=False, palette='science'):
    """Compare optimization histories with non-colour line distinctions."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette
    iterations = np.asarray(iterations)
    if iterations.ndim != 1 or iterations.size < 2:
        raise ValueError('iterations 至少包含两个值')
    if not histories:
        raise ValueError('histories 不能为空')
    colors = get_palette(palette, len(histories)); styles = ('-', '--', '-.', ':', (0, (4, 1, 1, 1)))
    for index, (name, values) in enumerate(histories.items()):
        values = np.asarray(values, dtype=float)
        if values.shape != iterations.shape or not np.all(np.isfinite(values)):
            raise ValueError(f'{name} 与 iterations 形状不一致或含非有限值')
        if logy and np.any(values <= 0):
            raise ValueError('对数纵轴要求所有历史值大于 0')
        ax.plot(iterations, values, color=colors[index], linestyle=styles[index], label=name)
    if best_known is not None:
        ax.axhline(best_known, color='#555555', linestyle=':', label='已知最优')
    if logy:
        ax.set_yscale('log')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style_axis(ax, grid='y')
    ax.legend()
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style(); x = np.arange(1, 101); rng = np.random.default_rng(2)
    histories = {'遗传算法' if language == 'zh' else 'GA': 20 + 100 * np.exp(-x / 22) + rng.random(100) * 2,
                 '粒子群' if language == 'zh' else 'PSO': 18 + 90 * np.exp(-x / 15) + rng.random(100)}
    fig, ax = new_figure(aspect=0.68)
    make_convergence(ax, x, histories, best_known=18,
                     xlabel='迭代次数' if language == 'zh' else 'Iteration',
                     ylabel='目标函数值' if language == 'zh' else 'Objective')
    output = THIS_DIR / 'samples' / (stem or ('convergence_demo' if language == 'zh' else 'convergence_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
