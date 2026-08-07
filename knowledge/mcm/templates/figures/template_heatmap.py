# -*- coding: utf-8 -*-
"""Science-style heatmap for matrices, correlations, and sensitivities."""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def _text_color(rgba):
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return '#FFFFFF' if luminance < 0.52 else '#222222'


def make_heatmap(ax, data, row_labels=None, col_labels=None,
                 cmap='cividis', annot=True, fmt='.2f', center=None,
                 cbar_label='', vmin=None, vmax=None, mask=None,
                 square=True, **kwargs):
    """Draw a perceptually controlled heatmap with contrast-safe labels."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError('data 必须是非空二维数组')
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError('mask 必须与 data 同形')
    else:
        mask = ~np.isfinite(data)
    masked = np.ma.array(data, mask=mask)
    finite = data[~mask]
    if finite.size == 0:
        raise ValueError('data 不包含可绘制的有限值')

    lower = finite.min() if vmin is None else float(vmin)
    upper = finite.max() if vmax is None else float(vmax)
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    if center is not None:
        if not lower < center < upper:
            raise ValueError('center 必须严格位于 vmin 与 vmax 之间')
        norm = TwoSlopeNorm(vmin=lower, vcenter=center, vmax=upper)
    else:
        norm = Normalize(vmin=lower, vmax=upper)

    im = ax.imshow(masked, cmap=cmap, norm=norm,
                   aspect='equal' if square else 'auto', **kwargs)
    if annot:
        for row, col in np.ndindex(data.shape):
            if mask[row, col]:
                continue
            value = data[row, col]
            ax.text(col, row, format(value, fmt), ha='center', va='center',
                    fontsize=6.2, color=_text_color(im.cmap(im.norm(value))))

    if row_labels is not None:
        if len(row_labels) != data.shape[0]:
            raise ValueError('row_labels 长度与矩阵行数不一致')
        ax.set_yticks(np.arange(data.shape[0]), row_labels)
    if col_labels is not None:
        if len(col_labels) != data.shape[1]:
            raise ValueError('col_labels 长度与矩阵列数不一致')
        ax.set_xticks(np.arange(data.shape[1]), col_labels)
        rotation = 35 if any(len(str(label)) > 4 for label in col_labels) else 0
        ax.tick_params(axis='x', rotation=rotation)
        if rotation:
            for label in ax.get_xticklabels():
                label.set_ha('right')
                label.set_rotation_mode('anchor')

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.035,
                              aspect=24)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(labelsize=6.5, width=0.45, length=2)
    if cbar_label:
        cbar.set_label(cbar_label)
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.92)
    rng = np.random.default_rng(7)
    sample = rng.normal(size=(80, 5))
    corr = np.corrcoef(sample.T)
    zh = language == 'zh'
    labels = ['价格', '产量', '成本', '销量', '利润'] if zh else ['Price', 'Yield', 'Cost', 'Sales', 'Profit']
    make_heatmap(ax, corr, row_labels=labels, col_labels=labels,
                 cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                 cbar_label='相关系数' if zh else 'Correlation')
    ax.set_title('指标相关系数矩阵' if zh else 'Indicator correlation matrix')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('heatmap_demo' if zh else 'heatmap_demo_en')))


if __name__ == '__main__':
    demo()
