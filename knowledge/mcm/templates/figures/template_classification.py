# -*- coding: utf-8 -*-
"""Classification evaluation: ROC, precision-recall, and confusion matrix."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def _curve(x, y, name):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError(f'{name} 曲线必须由至少两个等长点组成')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f'{name} 曲线包含非有限值')
    return x, y


def make_roc_curve(ax, curves, xlabel='假阳性率', ylabel='真阳性率'):
    """Plot one or more precomputed ROC curves."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette
    colors = get_palette('science', len(curves))
    for index, (name, values) in enumerate(curves.items()):
        fpr, tpr = _curve(values[0], values[1], 'ROC')
        auc = values[2] if len(values) > 2 else None
        label = f'{name} (AUC={auc:.3f})' if auc is not None else str(name)
        ax.plot(fpr, tpr, color=colors[index], label=label)
    ax.plot([0, 1], [0, 1], color='#777777', linestyle='--', linewidth=0.7)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.01))
    style_axis(ax, grid='both')
    ax.legend()
    return ax


def make_pr_curve(ax, curves, prevalence=None, xlabel='召回率', ylabel='精确率'):
    """Plot one or more precomputed precision-recall curves."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette
    colors = get_palette('science', len(curves))
    for index, (name, values) in enumerate(curves.items()):
        recall, precision = _curve(values[0], values[1], 'PR')
        ap = values[2] if len(values) > 2 else None
        label = f'{name} (AP={ap:.3f})' if ap is not None else str(name)
        ax.plot(recall, precision, color=colors[index], label=label)
    if prevalence is not None:
        ax.axhline(prevalence, color='#777777', linestyle='--', linewidth=0.7)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.01))
    style_axis(ax, grid='both')
    ax.legend()
    return ax


def make_confusion_matrix(ax, matrix, labels, normalize=None, cmap='Blues'):
    """Plot a raw, row-normalized, or column-normalized confusion matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] != len(labels):
        raise ValueError('matrix 必须是与 labels 匹配的方阵')
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError('matrix 必须包含有限非负值')
    display = values.copy()
    if normalize in {'true', 'pred'}:
        axis = 1 if normalize == 'true' else 0
        denominator = display.sum(axis=axis, keepdims=True)
        display = np.divide(display, denominator, out=np.zeros_like(display), where=denominator != 0)
    elif normalize is not None:
        raise ValueError("normalize 必须为 None、'true' 或 'pred'")
    image = ax.imshow(display, cmap=cmap, norm=Normalize(vmin=0, vmax=max(display.max(), 1e-12)))
    threshold = display.max() * 0.55
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            text = f'{display[row, column]:.1%}' if normalize else f'{int(values[row, column])}'
            ax.text(column, row, text, ha='center', va='center',
                    color='white' if display[row, column] > threshold else '#222222')
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel='预测类别', ylabel='真实类别')
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_panel_figure, save_figure
        from .stat_helpers import classification_curves
    except ImportError:
        from export_figure import apply_mcm_style, new_panel_figure, save_figure
        from stat_helpers import classification_curves
    apply_mcm_style()
    rng = np.random.default_rng(4)
    y_true = np.r_[np.zeros(80), np.ones(40)]
    score = np.r_[rng.beta(2, 5, 80), rng.beta(5, 2, 40)]
    data = classification_curves(y_true, score)
    zh = language == 'zh'; model = '模型' if zh else 'Model'
    fig, axes = new_panel_figure(1, 3, aspect=0.33)
    make_roc_curve(axes[0], {model: (data['fpr'], data['tpr'], data['roc_auc'])},
                   '假阳性率' if zh else 'False positive rate',
                   '真阳性率' if zh else 'True positive rate')
    make_pr_curve(axes[1], {model: (data['recall'], data['precision'], data['average_precision'])},
                  prevalence=data['prevalence'], xlabel='召回率' if zh else 'Recall',
                  ylabel='精确率' if zh else 'Precision')
    labels = ['负类', '正类'] if zh else ['Negative', 'Positive']
    make_confusion_matrix(axes[2], [[70, 10], [7, 33]], labels, normalize='true')
    if not zh:
        axes[2].set(xlabel='Predicted', ylabel='Actual')
    output = THIS_DIR / 'samples' / (stem or ('classification_demo' if zh else 'classification_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
