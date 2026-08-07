# -*- coding: utf-8 -*-
"""Restrained 3D trajectory template for genuinely spatial evidence."""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_3d_trajectory(ax, trajectories, labels=None, xlabel='X (m)',
                       ylabel='Y (m)', zlabel='Z (m)', view=(22, -58),
                       orthographic=True, equal_scale=True, palette='science',
                       **kwargs):
    """Plot spatial trajectories with orthographic and scale-aware defaults."""
    from palette import get_palette

    if not trajectories:
        raise ValueError('trajectories 不能为空')
    if labels is not None and len(labels) != len(trajectories):
        raise ValueError('labels 长度必须与 trajectories 一致')
    colors = get_palette(palette, len(trajectories))
    linestyles = ('-', '--', '-.', ':', (0, (4, 1, 1, 1)))
    validated = []

    for index, trajectory in enumerate(trajectories):
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3 or trajectory.shape[0] < 2:
            raise ValueError('每条轨迹必须是形状为 (N, 3) 且 N ≥ 2 的数组')
        if not np.all(np.isfinite(trajectory)):
            raise ValueError('轨迹中不能包含 NaN 或无穷值')
        validated.append(trajectory)
        label = labels[index] if labels else None
        line_kw = {
            'color': colors[index],
            'linestyle': linestyles[index],
            'linewidth': 1.3,
            'label': label,
            'zorder': 3,
        }
        line_kw.update(kwargs)
        ax.plot(*trajectory.T, **line_kw)
        ax.scatter(*trajectory[0], s=22, marker='o', facecolor='white',
                   edgecolor=colors[index], linewidth=0.8, depthshade=False)
        ax.scatter(*trajectory[-1], s=28, marker='X', color=colors[index],
                   edgecolor='#222222', linewidth=0.4, depthshade=False)

    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.text2D(0.87, 0.13, ylabel, transform=ax.transAxes,
              ha='center', va='center', fontsize=8)
    ax.text2D(0.985, 0.43, zlabel, transform=ax.transAxes,
              ha='center', va='center', rotation=90, fontsize=8)
    ax.view_init(elev=view[0], azim=view[1])
    if orthographic and hasattr(ax, 'set_proj_type'):
        ax.set_proj_type('ortho')
    if equal_scale and hasattr(ax, 'set_box_aspect'):
        combined = np.vstack(validated)
        ranges = np.ptp(combined, axis=0)
        ranges[ranges == 0] = 1
        ax.set_box_aspect(ranges / ranges.max())

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor('#BFBFBF')
        axis._axinfo['grid'].update({'color': '#D9D9D9', 'linewidth': 0.4,
                                     'linestyle': ':'})
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(pad=-1, labelsize=6.3)
    if labels:
        ax.legend(loc='upper center', ncol=min(len(labels), 3))
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, figure_size, save_figure
    apply_mcm_style()
    fig = plt.figure(figsize=figure_size('double', 0.48))
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 5, 100)
    trajectory_a = np.column_stack([
        50 * t, 30 * np.sin(t), 40 * t - 8 * t ** 2,
    ])
    trajectory_b = np.column_stack([
        50 * t + 15 * np.sin(2 * t), 30 * np.cos(t),
        40 * t - 9 * t ** 2 + 10,
    ])
    zh = language == 'zh'
    make_3d_trajectory(ax, [trajectory_a, trajectory_b],
                       labels=['导弹', '无人机'] if zh else ['Missile', 'Drone'],
                       xlabel='X（m）' if zh else 'X (m)', ylabel='Y（m）' if zh else 'Y (m)',
                       zlabel='高度（m）' if zh else 'Altitude (m)')
    ax.set_title('空间轨迹对比' if zh else 'Spatial trajectories')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('3d_demo' if zh else '3d_demo_en')))


if __name__ == '__main__':
    demo()
