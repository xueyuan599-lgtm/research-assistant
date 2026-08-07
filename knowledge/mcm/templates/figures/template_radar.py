# -*- coding: utf-8 -*-
"""雷达图 — 多指标评价 / 方案对比

改编自 academic-figure-skill (github.com/TingxiYu/academic-figure-skill, Apache-2.0)
的 Radar/plot_comparison_radar.py 的 polar 归一化逻辑，按本地 make_* 规范重写。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent


def make_radar(ax, categories, values, unit='', normalize=False,
               fill_alpha=0.15, show_values=False, palette='science'):
    """绘制雷达图（多方案叠加对比）。
    参数:
        ax: 需为 polar 投影（fig.add_subplot(projection='polar')）
        categories: 指标名列表（中文）
        values: dict {方案名: 数值数组}，长度与 categories 一致
        normalize: 是否对各指标归一化到 [0,1]（量纲不同时用）
        show_values: 是否在顶点标注数值
    注意: 雷达图对指标间大小关系敏感，仅适合同量纲或已标准化的评价指标；
          指标过多（>8）时建议改用平行坐标（make_parallel_coordinates）。
    """
    try:
        from .palette import get_palette
    except ImportError:
        from palette import get_palette

    if not categories or len(categories) < 3:
        raise ValueError('雷达图至少需要 3 个评价指标')
    n = len(categories)
    if not values or len(values) < 1:
        raise ValueError('values 不能为空')
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        if arr.shape != (n,) or not np.all(np.isfinite(arr)):
            raise ValueError(f'方案"{name}"的数值长度必须与指标数一致且均为有限值')

    # 角度布局：从顶部开始、顺时针
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.append(angles, angles[0])

    colors = get_palette(palette, len(values))
    for index, (name, vals) in enumerate(values.items()):
        arr = np.asarray(vals, dtype=float)
        if normalize:
            lo, hi = arr.min(), arr.max()
            arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros(n)
        closed = np.append(arr, arr[0])
        ax.plot(angles_closed, closed, color=colors[index], linewidth=1.2,
                label=name)
        ax.fill(angles_closed, closed, color=colors[index], alpha=fill_alpha)
        if show_values:
            for angle, value in zip(angles, arr):
                ax.annotate(f'{value:g}', (angle, value), textcoords='offset points',
                            xytext=(0, 5), ha='center', fontsize=5.5,
                            color=colors[index])

    ax.set_xticks(angles, categories)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    if normalize:
        ax.set_ylim(0, 1.0)
    # 隐藏极坐标的默认网格和边框，用轻量网格
    ax.grid(color='#D9D9D9', linewidth=0.5)
    ax.spines['polar'].set_visible(False)
    if unit:
        ax.set_ylabel(f'({unit})', labelpad=10)
    if len(values) > 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, save_figure
    apply_mcm_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.8), subplot_kw={'projection': 'polar'})
    zh = language == 'zh'
    categories = (['响应速度', '准确性', '稳健性', '成本', '可解释性'] if zh
                  else ['Speed', 'Accuracy', 'Robustness', 'Cost', 'Interpretability'])
    values = {
        '方案A': [0.82, 0.75, 0.70, 0.88, 0.65],
        '方案B': [0.70, 0.88, 0.82, 0.62, 0.78],
    }
    make_radar(ax, categories, values, show_values=True)
    ax.set_title('方案多指标评价' if zh else 'Multi-criteria evaluation')
    return save_figure(fig, THIS_DIR / 'samples' / (stem or
                        ('radar_demo' if zh else 'radar_demo_en')))


if __name__ == '__main__':
    demo()
