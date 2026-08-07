# -*- coding: utf-8 -*-
"""Science-style scatter, regression, and calibration template."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_scatter_fit(ax, x, y, xlabel='实测值', ylabel='预测值', unit='',
                     show_r2=True, show_diag=True, show_ci=True,
                     equal_axes=False, point_color='#3B6FB6', grid=True,
                     **kwargs):
    """Draw observations, OLS fit, optional 95% CI, and a 1:1 reference."""
    from export_figure import style_axis

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError('x 与 y 必须是一维同形数组')
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.ptp(x) == 0:
        raise ValueError('回归散点图至少需要 3 个有效点且 x 不能为常数')

    scatter_kw = {
        's': 20,
        'alpha': 0.78,
        'color': point_color,
        'edgecolors': '#222222',
        'linewidths': 0.35,
        'zorder': 3,
    }
    scatter_kw.update(kwargs)
    ax.scatter(x, y, **scatter_kw)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    fitted = slope * x_line + intercept
    ax.plot(x_line, fitted, color='#E07A2D', linewidth=1.2,
            label='线性拟合', zorder=4)

    residuals = y - (slope * x + intercept)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    if show_ci and x.size > 3:
        standard_error = np.sqrt(ss_res / (x.size - 2))
        spread = np.sum((x - x.mean()) ** 2)
        ci = 1.96 * standard_error * np.sqrt(
            1 / x.size + (x_line - x.mean()) ** 2 / spread
        )
        ax.fill_between(x_line, fitted - ci, fitted + ci,
                        color='#E07A2D', alpha=0.14, linewidth=0,
                        label='95% 置信带', zorder=1)

    if show_diag:
        lower = min(x.min(), y.min())
        upper = max(x.max(), y.max())
        ax.plot([lower, upper], [lower, upper], color='#333333',
                linestyle=':', linewidth=0.85, label='1:1 参考线', zorder=2)
        if equal_axes:
            padding = max((upper - lower) * 0.04, 1e-9)
            ax.set_xlim(lower - padding, upper + padding)
            ax.set_ylim(lower - padding, upper + padding)
            ax.set_aspect('equal', adjustable='box')

    if show_r2:
        r2_text = 'NA' if np.isnan(r_squared) else f'{r_squared:.3f}'
        ax.text(0.04, 0.96, f'$R^2$ = {r2_text}\n$n$ = {x.size}',
                transform=ax.transAxes, va='top', ha='left', fontsize=6.8,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82,
                      'pad': 1.5})
    ax.set_xlabel(f'{xlabel} ({unit})' if unit else xlabel)
    ax.set_ylabel(f'{ylabel} ({unit})' if unit else ylabel)
    style_axis(ax, grid='both' if grid else None)
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.88)
    rng = np.random.default_rng(3)
    actual = rng.uniform(0, 100, 48)
    predicted = actual * 0.95 + rng.normal(0, 6, 48)
    zh = language == 'zh'
    make_scatter_fit(ax, actual, predicted, xlabel='实测值' if zh else 'Observed',
                     ylabel='模型预测值' if zh else 'Predicted',
                     unit='万元' if zh else '10k CNY', equal_axes=True)
    ax.set_title('模型预测与实测值对比' if zh else 'Predicted versus observed')
    if not zh:
        legend = ax.get_legend()
        if legend:
            translations = {'线性拟合': 'Linear fit', '95% 置信带': '95% CI', '1:1 参考线': '1:1 reference'}
            for text in legend.texts:
                text.set_text(translations.get(text.get_text(), text.get_text()))
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('scatter_demo' if zh else 'scatter_demo_en')))


if __name__ == '__main__':
    demo()
