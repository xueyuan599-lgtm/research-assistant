# -*- coding: utf-8 -*-
"""PCA 降维可视化 — 高维特征 / 样本分组

改编自 academic-figure-skill (github.com/TingxiYu/academic-figure-skill, Apache-2.0)
的 PCA/plot_PCA.R 的降维与贡献率标注逻辑，按本地 make_* 规范用 Python 重写。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent


def make_pca(ax, X, labels, group_ids=None, group_names=None, standardize=True,
             n_components=2, palette='science'):
    """绘制 PCA 降维散点图。
    参数:
        ax: 2D 坐标轴（n_components=2）；3 维时用 projection='3d'
        X: (n_samples, n_features) 数组
        labels: 每个样本的横轴标签（用于点标注，可传 None）
        group_ids: 可选，每个样本的组别（0..k-1），用于按组着色
        group_names: 可选，组别名称（中文），用于图例
        standardize: 是否标准化（z-score）；特征量纲不同时建议 True
        n_components: 2 或 3
    输出: 在 ax 上绘制散点，返回解释方差比例 (explained_variance_ratio)
    """
    from sklearn.decomposition import PCA
    try:
        from .palette import get_palette
        from .stat_helpers import standardize_matrix
    except ImportError:
        from palette import get_palette
        from stat_helpers import standardize_matrix

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.size == 0 or X.shape[0] < 2:
        raise ValueError('X 必须是至少 2 行 × 若干特征的二维数组')
    if n_components not in (2, 3):
        raise ValueError('n_components 必须是 2 或 3')
    k = min(n_components, X.shape[0], X.shape[1])

    matrix = standardize_matrix(X) if standardize else X
    pca = PCA(n_components=k)
    projected = pca.fit_transform(matrix)
    ratio = pca.explained_variance_ratio_

    scatter_kw = {'s': 16, 'alpha': 0.8, 'edgecolors': 'none'}

    def _scatter(pts, **overrides):
        if k == 2:
            ax.scatter(pts[:, 0], pts[:, 1], **scatter_kw, **overrides)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **scatter_kw, **overrides)

    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        unique = np.unique(group_ids)
        colors = get_palette(palette, len(unique))
        for index, gid in enumerate(unique):
            mask = group_ids == gid
            name = group_names[index] if group_names else f'组{gid}'
            _scatter(projected[mask], color=colors[index], label=name)
        ax.legend()
    else:
        _scatter(projected, color='#3B6FB6')

    axis_names = [f'PC{i + 1} ({ratio[i] * 100:.1f}%)' for i in range(k)]
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    if k == 3:
        ax.set_zlabel(axis_names[2])
    return ax, ratio


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    rng = np.random.default_rng(11)
    zh = language == 'zh'
    centers = [np.array([0, 0, 0, 0, 0]),
               np.array([2, 1, 2, 1, 0]),
               np.array([-1, 2, 0, -1, 1])]
    X = np.vstack([center + rng.normal(0, 0.7, (40, 5)) for center in centers])
    group_ids = np.repeat([0, 1, 2], 40)
    names = (['类别一', '类别二', '类别三'] if zh
             else ['Group 1', 'Group 2', 'Group 3'])
    fig, ax = new_figure(aspect=0.85)
    make_pca(ax, X, None, group_ids=group_ids, group_names=names)
    ax.set_title('PCA 样本分布' if zh else 'PCA sample distribution')
    return save_figure(fig, THIS_DIR / 'samples' / (stem or
                        ('pca_demo' if zh else 'pca_demo_en')))


if __name__ == '__main__':
    demo()
