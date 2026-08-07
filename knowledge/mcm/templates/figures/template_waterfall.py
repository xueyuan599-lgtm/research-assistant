# -*- coding: utf-8 -*-
"""Additive contribution waterfall chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def make_waterfall(ax, labels, contributions, start=0, ylabel='数值', unit='',
                   total_label='最终值', show_values=True):
    """Draw a start-to-end additive bridge with explicit connectors."""
    try:
        from .export_figure import style_axis
        from .palette import SCIENCE_LIST
    except ImportError:
        from export_figure import style_axis
        from palette import SCIENCE_LIST
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 1 or values.size == 0 or len(labels) != values.size or not np.all(np.isfinite(values)):
        raise ValueError('labels 与 contributions 必须为等长非空有限序列')
    cumulative = start + np.r_[0, np.cumsum(values)]
    bottoms = np.minimum(cumulative[:-1], cumulative[1:])
    heights = np.abs(values)
    colors = [SCIENCE_LIST[2] if value >= 0 else SCIENCE_LIST[1] for value in values]
    x = np.arange(values.size + 2)
    ax.bar(0, start, color='#777777', width=0.68, label='起点')
    bars = ax.bar(x[1:-1], heights, bottom=bottoms, color=colors, width=0.68)
    final = cumulative[-1]
    ax.bar(x[-1], final, color=SCIENCE_LIST[0], width=0.68, label=total_label)
    for index in range(values.size + 1):
        ax.plot([x[index] + 0.34, x[index + 1] - 0.34],
                [cumulative[index], cumulative[index]], color='#888888', linewidth=0.55)
    if show_values:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f'{value:+.1f}', ha='center', va='center', fontsize=5.5,
                    color='white' if bar.get_height() > max(heights) * 0.15 else '#222222')
        ax.text(x[-1], final, f'{final:.1f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x, ['起点', *labels, total_label], rotation=25, ha='right')
    ax.set_ylabel(f'{ylabel} ({unit})' if unit else ylabel)
    style_axis(ax, grid='y', zero_line=True)
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style(); zh = language == 'zh'
    labels = ['需求增长', '节能措施', '价格变化', '政策补贴'] if zh else ['Demand', 'Efficiency', 'Price', 'Subsidy']
    fig, ax = new_figure(aspect=0.72)
    make_waterfall(ax, labels, [18, -12, 9, -6], start=80,
                   ylabel='年度成本' if zh else 'Annual cost', unit='万元' if zh else '10k CNY',
                   total_label='最终值' if zh else 'Final')
    if not zh:
        ax.set_xticklabels(['Start', *labels, 'Final'], rotation=25, ha='right')
    output = THIS_DIR / 'samples' / (stem or ('waterfall_demo' if zh else 'waterfall_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
