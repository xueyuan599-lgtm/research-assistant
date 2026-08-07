# -*- coding: utf-8 -*-
"""Science-style grouped and stacked bar chart templates."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def _validate(categories, mapping):
    if not categories or not mapping:
        raise ValueError('categories 和数据映射均不能为空')
    arrays = {}
    for name, values in mapping.items():
        array = np.asarray(values, dtype=float)
        if array.shape != (len(categories),):
            raise ValueError(f'“{name}”长度必须等于 categories 长度')
        arrays[name] = array
    return arrays


def make_grouped_bar(ax, categories, groups, ylabel='数值', unit='',
                     show_values=True, value_format='.1f', palette='science',
                     grid=True, **kwargs):
    """Draw grouped bars with explicit colors, keylines, and smart labels."""
    from export_figure import style_axis
    from palette import get_palette

    groups = _validate(categories, groups)
    colors = get_palette(palette, len(groups))
    hatches = ('', '//', '..', 'xx', '\\\\')
    x = np.arange(len(categories))
    width = min(0.72 / len(groups), 0.28)

    for index, (name, values) in enumerate(groups.items()):
        offset = (index - (len(groups) - 1) / 2) * width
        bar_kw = {
            'color': colors[index],
            'edgecolor': '#333333',
            'linewidth': 0.45,
            'hatch': hatches[index],
            'label': name,
            'zorder': 3,
        }
        bar_kw.update(kwargs)
        bars = ax.bar(x + offset, values, width * 0.92, **bar_kw)
        if show_values and len(categories) * len(groups) <= 24:
            labels = [format(value, value_format) for value in values]
            ax.bar_label(bars, labels=labels, fontsize=6.2, padding=1.5)

    ax.set_xticks(x, categories)
    ax.set_ylabel(f'{ylabel} ({unit})' if unit else ylabel)
    style_axis(ax, grid='y' if grid else None,
               zero_line=any(np.any(values < 0) for values in groups.values()))
    if len(groups) > 1:
        ax.legend(ncol=min(len(groups), 3), loc='upper center')
    return ax


def make_stacked_bar(ax, categories, layers, ylabel='数值', unit='',
                     normalize=False, show_values=False, palette='science',
                     grid=True, **kwargs):
    """Draw stacked or 100% stacked bars with segment keylines."""
    from export_figure import style_axis
    from palette import get_palette

    layers = _validate(categories, layers)
    colors = get_palette(palette, len(layers))
    matrix = np.vstack(list(layers.values()))
    if np.any(matrix < 0):
        raise ValueError('堆叠柱模板仅接受非负数值')
    totals = matrix.sum(axis=0)
    if normalize:
        if np.any(totals == 0):
            raise ValueError('归一化堆叠柱不允许总和为 0 的类别')
        matrix = matrix / totals

    x = np.arange(len(categories))
    bottom = np.zeros(len(categories))
    for index, name in enumerate(layers):
        values = matrix[index]
        bar_kw = {
            'color': colors[index],
            'edgecolor': 'white',
            'linewidth': 0.55,
            'label': name,
            'zorder': 3,
        }
        bar_kw.update(kwargs)
        bars = ax.bar(x, values, bottom=bottom, width=0.68, **bar_kw)
        if show_values:
            for bar, value, base in zip(bars, values, bottom):
                if value >= (0.08 if normalize else max(totals.max() * 0.06, 1e-12)):
                    label = f'{value:.0%}' if normalize else f'{value:.1f}'
                    ax.text(bar.get_x() + bar.get_width() / 2, base + value / 2,
                            label, ha='center', va='center', fontsize=6,
                            color='white' if index in {0, 1, 2} else '#222222')
        bottom += values

    ax.set_xticks(x, categories)
    axis_label = '占比' if normalize and ylabel == '数值' else ylabel
    ax.set_ylabel(f'{axis_label} ({unit})' if unit else axis_label)
    if normalize:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda value, _: f'{value:.0%}')
    style_axis(ax, grid='y' if grid else None)
    ax.legend(ncol=min(len(layers), 3), loc='upper center')
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, figure_size, save_figure
    apply_mcm_style()
    fig, axes = plt.subplots(1, 2, figsize=figure_size('double', 0.38))
    zh = language == 'zh'
    types = ['水浇地', '旱地', '坡地'] if zh else ['Irrigated', 'Dryland', 'Slope']
    base_name, improved_name = (('基准方案', '改进方案') if zh else ('Baseline', 'Improved'))
    make_grouped_bar(axes[0], types, {
        base_name: [150, 120, 90],
        improved_name: [180, 145, 110],
    }, ylabel='利润' if zh else 'Profit', unit='万元' if zh else '10k CNY')
    axes[0].set_title('地块类型利润对比' if zh else 'Profit by land type')
    make_stacked_bar(axes[1], types, {
        ('粮食' if zh else 'Grain'): [0.5, 0.4, 0.6],
        ('经济作物' if zh else 'Cash crop'): [0.3, 0.3, 0.2],
        ('豆类' if zh else 'Legume'): [0.2, 0.3, 0.2],
    }, ylabel='占比' if zh else 'Share', normalize=True, show_values=True)
    axes[1].set_title('作物种植结构' if zh else 'Cropping structure')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('bar_demo' if zh else 'bar_demo_en')))


if __name__ == '__main__':
    demo()
