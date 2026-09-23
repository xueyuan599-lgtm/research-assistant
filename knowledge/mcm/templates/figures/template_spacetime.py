# -*- coding: utf-8 -*-
"""Spatio-temporal heatmaps u(x, t) for PDE-type fields."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_spacetime(ax, field, x, t, cmap='cividis',
                   xlabel='空间坐标 $x$ (m)', ylabel='时间 $t$ (s)',
                   cbar_label='场量 $u(x,t)$', vmin=None, vmax=None,
                   show_colorbar=True):
    """Render a (len(x), len(t)) u(x, t) grid as a heatmap on one axis."""
    field = np.asarray(field, dtype=float)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if field.ndim != 2 or field.shape != (len(x), len(t)):
        raise ValueError('field 必须是 (len(x), len(t)) 形状的二维数组')
    if x.ndim != 1 or t.ndim != 1 or len(x) < 2 or len(t) < 2:
        raise ValueError('x 与 t 必须是至少两个采样点的一维数组')
    if not np.all(np.isfinite(field)):
        raise ValueError('field 包含 NaN 或无穷值')
    if np.any(np.diff(x) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError('x 与 t 必须单调递增')

    low = float(field.min()) if vmin is None else float(vmin)
    high = float(field.max()) if vmax is None else float(vmax)
    if low == high:
        low, high = low - 0.5, high + 0.5
    im = ax.pcolormesh(x, t, field.T, cmap=cmap, vmin=low, vmax=high,
                       shading='auto', rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(t[0], t[-1])
    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.035,
                                  aspect=22)
        cbar.outline.set_linewidth(0.45)
        cbar.ax.tick_params(labelsize=6.5, width=0.45, length=2)
        if cbar_label:
            cbar.set_label(cbar_label)
    return im


def make_spacetime_compare(field_true, field_pred, x, t, width='double',
                           aspect=0.34, x_label='空间坐标 $x$ (m)',
                           t_label='时间 $t$ (s)',
                           cbar_label='场量 $u(x,t)$',
                           error_label='误差 $u_{\\mathrm{pred}}-u$',
                           titles=('实测场', '模型预测'),
                           show_error=True):
    """True-vs-predicted u(x, t) panels sharing one colorbar.

    色标按两场联合值域共享；``show_error`` 为真时追加第三块误差场
    （发散色 RdBu_r、中心 0，独立色标）。返回三块面板的 axes 数组。
    """
    from export_figure import figure_size

    field_true = np.asarray(field_true, dtype=float)
    field_pred = np.asarray(field_pred, dtype=float)
    if field_true.shape != field_pred.shape:
        raise ValueError('field_true 与 field_pred 必须同形')
    for arr, name in ((x, 'x'), (t, 't')):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.size < 2 or np.any(np.diff(arr) <= 0):
            raise ValueError(f'{name} 必须是单调递增的一维数组（≥2 点）')

    n_fields = 3 if show_error else 2
    fig, axes = plt.subplots(
        1, n_fields, figsize=figure_size(width, aspect),
        sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.24][:n_fields]})

    if show_error:
        fields = (field_true, field_pred, field_pred - field_true)
    else:
        fields = (field_true, field_pred)
    low = min(field_true.min(), field_pred.min())
    high = max(field_true.max(), field_pred.max())
    if low == high:
        low, high = low - 0.5, high + 0.5

    field_names = titles + (None,)
    for index, (ax, field) in enumerate(zip(axes, fields)):
        if index < 2:
            im = make_spacetime(ax, field, x, t, vmin=low, vmax=high,
                                xlabel=x_label, ylabel=t_label if index == 0 else '',
                                show_colorbar=False)
            ax.set_title(field_names[index], fontsize=7, pad=3)
        else:
            err_span = float(np.abs(field).max()) or 1.0
            make_spacetime(ax, field, x, t, cmap='RdBu_r', vmin=-err_span,
                           vmax=err_span, xlabel=x_label, ylabel='',
                           cbar_label=error_label)
            ax.set_title('误差场', fontsize=7, pad=3)
    if not show_error:
        cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03, aspect=22)
        cbar.outline.set_linewidth(0.45)
        cbar.ax.tick_params(labelsize=6, width=0.45, length=2)
        if cbar_label:
            cbar.set_label(cbar_label)
    return axes


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, save_figure
    apply_mcm_style()
    zh = language == 'zh'
    x = np.linspace(0, 2 * np.pi, 120)
    t = np.linspace(0, 4, 90)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    decay = np.exp(-0.18 * tt)
    field_true = np.sin(xx) * decay + 0.08 * np.sin(2 * xx - 1.5 * tt)
    # 模型预测：主波相位略漂移 + 幅值略低，制造可见误差结构。
    field_pred = 0.93 * np.sin(xx - 0.12 * tt) * decay + 0.085 * np.sin(2 * xx - 1.45 * tt)
    axes = make_spacetime_compare(field_true, field_pred, x, t,
                                  x_label='空间坐标 $x$ (m)' if zh else 'Space $x$ (m)',
                                  t_label='时间 $t$ (s)' if zh else 'Time $t$ (s)',
                                  cbar_label='场量 $u(x,t)$' if zh else 'Field $u(x,t)$',
                                  error_label='误差 $u_{pred}-u$' if zh else 'Error',
                                  titles=('实测场', '模型预测') if zh
                                  else ('Observed', 'Predicted'))
    axes[0].figure.suptitle('波动方程解的时空演化与模型误差' if zh
                            else 'Spatio-temporal evolution and model error',
                            fontsize=7.5)
    return save_figure(axes[0].figure, os.path.join(THIS_DIR, 'samples', stem or
                       ('spacetime_demo' if zh else 'spacetime_demo_en')))


if __name__ == '__main__':
    demo()
