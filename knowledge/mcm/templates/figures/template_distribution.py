# -*- coding: utf-8 -*-
"""Distribution templates: box, violin, raincloud, and ECDF."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def _groups(mapping):
    if not mapping:
        raise ValueError('groups 不能为空')
    result = {}
    for name, values in mapping.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
            raise ValueError(f'{name} 必须至少包含两个有限值')
        result[str(name)] = array
    return result


def make_distribution(ax, groups, kind='raincloud', ylabel='数值', unit='',
                      show_points=True, show_n=True, palette='science', seed=42):
    """Draw grouped distributions without hiding the raw observations."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette

    groups = _groups(groups)
    if kind not in {'box', 'violin', 'raincloud'}:
        raise ValueError("kind 必须是 'box'、'violin' 或 'raincloud'")
    colors = get_palette(palette, len(groups))
    positions = np.arange(1, len(groups) + 1)
    arrays = list(groups.values())

    if kind in {'violin', 'raincloud'}:
        offsets = positions - 0.12 if kind == 'raincloud' else positions
        violins = ax.violinplot(arrays, positions=offsets, widths=0.56,
                                showmeans=False, showmedians=False,
                                showextrema=False)
        for index, body in enumerate(violins['bodies']):
            body.set_facecolor(colors[index])
            body.set_edgecolor(colors[index])
            body.set_alpha(0.28)
            if kind == 'raincloud':
                vertices = body.get_paths()[0].vertices
                vertices[:, 0] = np.minimum(vertices[:, 0], offsets[index])

    box_positions = positions + (0.08 if kind == 'raincloud' else 0)
    boxes = ax.boxplot(arrays, positions=box_positions, widths=0.20 if kind == 'raincloud' else 0.48,
                       patch_artist=True, showfliers=False, medianprops={'color': '#222222', 'linewidth': 1})
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor('white' if kind == 'raincloud' else color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.9 if kind == 'raincloud' else 0.45)
    for key in ('whiskers', 'caps'):
        for artist in boxes[key]:
            artist.set_color('#555555')

    if show_points:
        rng = np.random.default_rng(seed)
        for position, values, color in zip(positions, arrays, colors):
            center = position + (0.20 if kind == 'raincloud' else 0)
            jitter = rng.uniform(-0.055, 0.055, values.size)
            ax.scatter(np.full(values.size, center) + jitter, values, s=10,
                       facecolor=color, edgecolor='white', linewidth=0.3,
                       alpha=0.78, zorder=3)

    labels = [f'{name}\n(n={len(values)})' if show_n else name
              for name, values in groups.items()]
    ax.set_xticks(positions, labels)
    ax.set_ylabel(f'{ylabel} ({unit})' if unit else ylabel)
    style_axis(ax, grid='y')
    return ax


def make_ecdf(ax, groups, xlabel='数值', unit='', palette='science'):
    """Draw empirical cumulative distributions for robust comparison."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette
    groups = _groups(groups)
    colors = get_palette(palette, len(groups))
    styles = ('-', '--', '-.', ':', (0, (4, 1, 1, 1)))
    for index, (name, values) in enumerate(groups.items()):
        ordered = np.sort(values)
        probability = np.arange(1, ordered.size + 1) / ordered.size
        ax.step(ordered, probability, where='post', color=colors[index],
                linestyle=styles[index], label=name)
    ax.set_xlabel(f'{xlabel} ({unit})' if unit else xlabel)
    ax.set_ylabel('累积概率')
    ax.set_ylim(0, 1.02)
    style_axis(ax, grid='both')
    ax.legend()
    return ax


def make_boxplot(ax, groups, show_points=False, **kwargs):
    """便捷别名：纯箱线图（不叠加散点）。"""
    kwargs.setdefault('show_points', show_points)
    return make_distribution(ax, groups, kind='box', **kwargs)


def make_violin(ax, groups, show_points=False, **kwargs):
    """便捷别名：小提琴图（含内部箱线）。"""
    kwargs.setdefault('show_points', show_points)
    return make_distribution(ax, groups, kind='violin', **kwargs)


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_panel_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_panel_figure, save_figure
    apply_mcm_style()
    rng = np.random.default_rng(7)
    labels = ('方案A', '方案B', '方案C') if language == 'zh' else ('Plan A', 'Plan B', 'Plan C')
    groups = {labels[0]: rng.normal(72, 8, 36), labels[1]: rng.normal(79, 6, 36),
              labels[2]: rng.normal(68, 11, 36)}
    fig, axes = new_panel_figure(1, 2, aspect=0.38)
    make_distribution(axes[0], groups, ylabel='得分' if language == 'zh' else 'Score')
    make_ecdf(axes[1], groups, xlabel='得分' if language == 'zh' else 'Score')
    output = THIS_DIR / 'samples' / (stem or ('distribution_demo' if language == 'zh' else 'distribution_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
