# -*- coding: utf-8 -*-
"""Optimizer search trajectory overlaid on the objective contour map."""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_searchpath(ax, func, x_range, y_range, xs, ys=None, grid_n=160,
                    levels=18, cmap='Blues', xlabel='$x_1$', ylabel='$x_2$',
                    phase_labels=('搜索评估', '邻域轮询'), start_label='起点',
                    best_label='最优点', show_colorbar=False, n_colors=5):
    """Draw ``func`` contours with the optimizer's visited points on top.

    ``func`` 接受形状 (N, 2) 的二维数组并返回 (N,) 目标值。``xs``/``ys`` 为
    搜索点序列；``ys`` 缺省时在 ``xs`` 内逐点取 [x, y] 对。返回最优点坐标
    ``(bx, by)`` 便于调用者标注。
    """
    from palette import SCIENCE_LIST

    xs = np.asarray(xs, dtype=float)
    if xs.ndim == 2 and xs.shape[1] >= 2:
        pts = xs[:, :2]
    else:
        if ys is None:
            raise ValueError('xs 为一维序列时必须同时给出 ys')
        pts = np.column_stack([np.asarray(xs, dtype=float),
                               np.asarray(ys, dtype=float)])
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError('搜索点必须形成至少两个 (x, y) 点')
    if not np.all(np.isfinite(pts)):
        raise ValueError('搜索点必须为有限数值')
    for name, lo, hi in ((xlabel, x_range[0], x_range[1]),
                         (ylabel, y_range[0], y_range[1])):
        if not lo < hi:
            raise ValueError(f'{name} 的范围必须满足下限 < 上限')
        if pts[:, 0 if name == xlabel else 1].min() < lo \
                or pts[:, 0 if name == xlabel else 1].max() > hi:
            raise ValueError(f'搜索点超出 {name} 显示范围')

    gx = np.linspace(x_range[0], x_range[1], grid_n)
    gy = np.linspace(y_range[0], y_range[1], grid_n)
    mesh = np.column_stack([np.tile(gx, gy.size), np.repeat(gy, gx.size)])
    fz = np.asarray(func(mesh), dtype=float).reshape(gy.size, gx.size)
    if not np.all(np.isfinite(fz)):
        raise ValueError('目标函数在网格上返回了非有限值')

    values = np.asarray(func(pts), dtype=float)
    best = int(np.argmin(values))
    contour = ax.contourf(gx, gy, fz, levels=levels, cmap=cmap, zorder=1)
    cs = ax.contour(gx, gy, fz, levels=levels, colors='#666666',
                    linewidths=0.3, zorder=2)
    # 保留向量导出里的等高线数值（期刊审稿常要求等高线可读）。
    ax.clabel(cs, inline=True, fontsize=4.8, fmt='%.3g', colors='#444444')

    path = ax.plot(pts[:, 0], pts[:, 1], color='#222222', linewidth=0.75,
                   alpha=0.85, zorder=4)[0]
    for order, point in enumerate(pts):
        frac = order / max(len(pts) - 1, 1)
        color = SCIENCE_LIST[min(int(frac * (n_colors - 1)), n_colors - 1)]
        ax.scatter(point[0], point[1], s=14, color=color, edgecolor='#222222',
                   linewidth=0.35, zorder=5)
    ax.scatter(pts[0, 0], pts[0, 1], s=30, marker='s', color='#FFFFFF',
               edgecolor='#222222', linewidth=0.7, zorder=6, label=start_label)
    ax.scatter(pts[best, 0], pts[best, 1], s=42, marker='*',
               color=SCIENCE_LIST[4], edgecolor='#222222', linewidth=0.4,
               zorder=6, label=best_label)
    handles = [path,
               Line2D([], [], marker='o', linestyle='none', markersize=3.5,
                      markerfacecolor=SCIENCE_LIST[0], markeredgecolor='#222222',
                      label=phase_labels[0]),
               Line2D([], [], marker='o', linestyle='none', markersize=3.5,
                      markerfacecolor=SCIENCE_LIST[3], markeredgecolor='#222222',
                      label=phase_labels[1])]
    ax.legend(handles=handles, loc='upper right', handlelength=1.4)
    if show_colorbar:
        cbar = ax.figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=6, width=0.45, length=2)
        cbar.set_label('目标值 f(x)')
    ax.set_xlabel(xlabel + ' (–)')
    ax.set_ylabel(ylabel + ' (–)')
    return ax, (float(pts[best, 0]), float(pts[best, 1]))


def _ackley(mesh):
    x, y = mesh[:, 0], mesh[:, 1]
    return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + 20 + np.e)


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.92)
    rng = np.random.default_rng(11)
    # 模拟一次“全局搜索 + 局部收缩”的两阶段优化轨迹。
    centers = np.array([[3.0, 2.5], [2.2, 2.0], [1.6, 1.7], [1.1, 1.4],
                        [0.7, 0.9], [0.35, 0.5], [0.08, 0.12]])
    pts, jitter = [], 0.75
    for center in centers:
        pts.append(center + rng.normal(scale=jitter, size=(6, 2)))
        jitter *= 0.55
    pts = np.vstack(pts)
    zh = language == 'zh'
    make_searchpath(ax, _ackley, (-4.5, 4.5), (-4.5, 4.5), pts,
                    xlabel='$x_1$' if not zh else '$x_1$',
                    ylabel='$x_2$',
                    phase_labels=('搜索评估', '邻域轮询') if zh else ('Search', 'Poll'),
                    start_label='起点' if zh else 'Start',
                    best_label='最优点' if zh else 'Best')
    ax.set_title('Ackley 函数上的优化搜索轨迹' if zh
                 else 'Search trajectory on Ackley surface')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('searchpath_demo' if zh else 'searchpath_demo_en')))


if __name__ == '__main__':
    demo()
