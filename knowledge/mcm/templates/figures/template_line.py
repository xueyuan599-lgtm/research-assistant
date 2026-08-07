# -*- coding: utf-8 -*-
"""Science-style line chart for trends and scenario comparisons."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_line_plot(ax, x, series, xlabel='时间', ylabel='数值', unit='',
                   legend_loc='best', error=None, error_style='band',
                   direct_labels=False, show_legend=None, grid=True,
                   palette='science', **kwargs):
    """Plot one to five series with non-color distinction and uncertainty.

    ``error`` accepts symmetric arrays or ``(lower, upper)`` error arrays.
    Set ``error_style='bar'`` for discrete measurements; continuous trends
    default to a quiet confidence band.
    """
    from export_figure import style_axis
    from palette import get_palette

    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError('x 必须是一维非空数组')
    if not series:
        raise ValueError('series 不能为空')

    colors = get_palette(palette, len(series))
    markers = ('o', 's', '^', 'D', 'v')
    linestyles = ('-', '--', '-.', ':', (0, (4, 1, 1, 1)))
    markevery = max(1, x.size // 10)

    for index, (name, values) in enumerate(series.items()):
        y = np.asarray(values, dtype=float)
        if y.shape != x.shape:
            raise ValueError(f'序列“{name}”长度与 x 不一致')
        line_kw = {
            'color': colors[index],
            'linestyle': linestyles[index],
            'marker': markers[index],
            'markevery': markevery,
            'markerfacecolor': 'white',
            'markeredgecolor': colors[index],
            'label': name,
            'zorder': 3,
        }
        line_kw.update(kwargs)

        if error and name in error and error_style == 'bar':
            ax.errorbar(x, y, yerr=error[name], capsize=2.2,
                        elinewidth=0.75, capthick=0.75, **line_kw)
        else:
            ax.plot(x, y, **line_kw)
            if error and name in error:
                uncertainty = np.asarray(error[name], dtype=float)
                if uncertainty.ndim == 2 and uncertainty.shape == (2, x.size):
                    lower, upper = y - uncertainty[0], y + uncertainty[1]
                elif uncertainty.shape == y.shape:
                    lower, upper = y - uncertainty, y + uncertainty
                else:
                    raise ValueError(f'序列“{name}”的 error 形状无效')
                ax.fill_between(x, lower, upper, color=colors[index],
                                alpha=0.14, linewidth=0, zorder=1)

        if direct_labels:
            finite = np.flatnonzero(np.isfinite(y))
            if finite.size:
                last = finite[-1]
                ax.annotate(name, (x[last], y[last]), xytext=(4, 0),
                            textcoords='offset points', color=colors[index],
                            fontsize=6.8, va='center', clip_on=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'{ylabel} ({unit})' if unit else ylabel)
    style_axis(ax, grid='y' if grid else None)
    if show_legend is None:
        show_legend = len(series) > 1 and not direct_labels
    if show_legend:
        ax.legend(loc=legend_loc, ncol=min(len(series), 3))
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.72)
    years = np.arange(2019, 2025)
    zh = language == 'zh'
    base_name, improved_name = (('基准方案', '改进方案') if zh
                                else ('Baseline', 'Improved'))
    series = {
        base_name: np.array([120, 128, 135, 142, 150, 158]),
        improved_name: np.array([120, 132, 145, 160, 178, 195]),
    }
    error = {
        base_name: np.array([4, 5, 4, 6, 5, 6]),
        improved_name: np.array([4, 4, 5, 5, 7, 8]),
    }
    make_line_plot(ax, years, series, xlabel='年份' if zh else 'Year',
                   ylabel='收益' if zh else 'Return', unit='万元' if zh else '10k CNY',
                   error=error, legend_loc='upper left')
    ax.set_title('方案收益趋势对比' if zh else 'Scenario return trends')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('line_demo' if zh else 'line_demo_en')))


if __name__ == '__main__':
    demo()
