# -*- coding: utf-8 -*-
"""Phase portrait with a vector field for dynamical-system models."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_phase(ax, field, extent, trajectories=None, equilibria=None,
               equilibrium_label='平衡点', density=1.2,
               traj_labels=None, field_color='#9DB8D9',
               traj_colors=None):
    """Overlay measured/simulated trajectories on a streamplot vector field.

    ``field(x, y)`` 接收两个网格数组，返回 ``u, v`` 分量；
    ``trajectories`` 为每条轨迹 (T, 2) 数组列表或 ``{'标签': (T, 2)}`` 字典；
    ``equilibria`` 为 (k, 2) 平衡点坐标（None 元素则跳过该点）。
    """
    from export_figure import style_axis
    from palette import SCIENCE, SCIENCE_LIST

    x = np.linspace(extent[0], extent[1], 220)
    y = np.linspace(extent[2], extent[3], 220)
    xx, yy = np.meshgrid(x, y)
    u, v = field(xx, yy)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != xx.shape or v.shape != xx.shape:
        raise ValueError('field 返回的 u/v 分量必须与网格同形')
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError('field 返回值必须为有限数值')

    if trajectories is not None:
        if isinstance(trajectories, dict):
            items = list(trajectories.items())
        else:
            items = [(None, traj) for traj in trajectories]
        for traj in (t for _, t in items):
            traj = np.asarray(traj, dtype=float)
            if traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] < 2:
                raise ValueError('每条轨迹必须是至少两行的 (T, 2) 数组')
            if not np.all(np.isfinite(traj)):
                raise ValueError('轨迹包含 NaN 或无穷值')

    speed = np.hypot(u, v)
    ax.streamplot(x, y, u, v, color=field_color, linewidth=0.55,
                  density=density, arrowsize=0.7, zorder=1)
    ax.contourf(x, y, np.log1p(speed), levels=12, cmap='Blues', alpha=0.16,
                zorder=0)

    if trajectories is not None:
        for index, (label, traj) in enumerate(items):
            traj = np.asarray(traj, dtype=float)
            color = (SCIENCE['橙'] if len(items) == 1
                     else SCIENCE_LIST[index % len(SCIENCE_LIST)])
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.25,
                    zorder=3, label=label if label else None)
            ax.scatter(traj[0, 0], traj[0, 1], s=16, marker='o', zorder=4,
                       facecolor='white', edgecolor=color, linewidth=0.7)

    if equilibria is not None:
        equilibria = np.asarray(equilibria, dtype=float)
        if equilibria.ndim != 2 or equilibria.shape[1] != 2:
            raise ValueError('equilibria 必须是 (k, 2) 数组')
        for eq_index, point in enumerate(equilibria):
            if np.all(np.isfinite(point)):
                ax.scatter(point[0], point[1], s=44, marker='x', color='#222222',
                           linewidth=1.1, zorder=5,
                           label=equilibrium_label
                           if eq_index == 0 and equilibrium_label else None)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('状态变量 $x_1$ (–)')
    ax.set_ylabel('状态变量 $x_2$ (–)')
    style_axis(ax, grid=None)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right')
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.92)
    zh = language == 'zh'

    def field(x, y):
        return 0.5 * x - x ** 3 / 2 - y, x - 0.4 * y

    t = np.linspace(0, 14, 400)

    def trajectory(z0):
        zs = np.empty((t.size, 2))
        zs[0] = z0
        dt = t[1] - t[0]
        for index in range(1, t.size):
            dx, dy = field(zs[index - 1, 0], zs[index - 1, 1])
            zs[index] = zs[index - 1] + dt * np.array([dx, dy])
        return zs

    if zh:
        trajs = {'轨迹一 (初值 $z_0$=(-2.2, 2.8))': trajectory((-2.2, 2.8)),
                 '轨迹二 (初值 $z_0$=(2.6, -1.6))': trajectory((2.6, -1.6))}
    else:
        trajs = {'Trajectory 1': trajectory((-2.2, 2.8)),
                 'Trajectory 2': trajectory((2.6, -1.6))}
    make_phase(ax, field, (-2.8, 2.8, -2.2, 2.2), trajs,
               equilibria=[(0.0, 0.0)],
               equilibrium_label='平衡点' if zh else 'Equilibrium')
    if not zh:
        ax.set_xlabel('State $x_1$ (–)')
        ax.set_ylabel('State $x_2$ (–)')
    ax.set_title('Duffing 型系统相平面轨迹与向量场' if zh
                 else 'Phase portrait with vector field')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('phase_demo' if zh else 'phase_demo_en')))


if __name__ == '__main__':
    demo()
