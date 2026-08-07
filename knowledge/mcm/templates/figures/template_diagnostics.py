# -*- coding: utf-8 -*-
"""Regression diagnostic templates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def _paired(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError('输入必须是一维等长数组')
    mask = np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 3:
        raise ValueError('至少需要三个有限配对观测')
    return first[mask], second[mask]


def make_residual_plot(ax, fitted, residuals, xlabel='拟合值', ylabel='残差'):
    """Plot residuals against fitted values with a LOWESS-like binned trend."""
    try:
        from .export_figure import style_axis
        from .palette import SCIENCE_LIST
    except ImportError:
        from export_figure import style_axis
        from palette import SCIENCE_LIST
    fitted, residuals = _paired(fitted, residuals)
    ax.scatter(fitted, residuals, s=13, color=SCIENCE_LIST[0], alpha=0.65,
               edgecolor='white', linewidth=0.3)
    bins = np.array_split(np.argsort(fitted), min(10, max(3, fitted.size // 5)))
    trend_x = [np.mean(fitted[index]) for index in bins if len(index)]
    trend_y = [np.mean(residuals[index]) for index in bins if len(index)]
    ax.plot(trend_x, trend_y, color=SCIENCE_LIST[1], linewidth=1, label='趋势')
    ax.axhline(0, color='#555555', linewidth=0.7)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style_axis(ax, grid='y')
    return ax


def make_qq_plot(ax, residuals, xlabel='理论分位数', ylabel='样本分位数'):
    """Draw a normal Q-Q diagnostic plot."""
    from scipy import stats
    try:
        from .export_figure import style_axis
        from .palette import SCIENCE_LIST
    except ImportError:
        from export_figure import style_axis
        from palette import SCIENCE_LIST
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError('residuals 至少需要三个有限值')
    theoretical, ordered = stats.probplot(values, dist='norm', fit=False)
    slope, intercept = np.polyfit(theoretical, ordered, 1)
    ax.scatter(theoretical, ordered, s=13, color=SCIENCE_LIST[0], alpha=0.7)
    limits = np.array([min(theoretical), max(theoretical)])
    ax.plot(limits, slope * limits + intercept, color='#333333', linestyle='--')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style_axis(ax, grid='both')
    return ax


def make_calibration_plot(ax, predicted, observed, bins=8,
                          xlabel='预测值', ylabel='观测均值'):
    """Compare binned predictions with observed means."""
    try:
        from .export_figure import style_axis
        from .palette import SCIENCE_LIST
    except ImportError:
        from export_figure import style_axis
        from palette import SCIENCE_LIST
    predicted, observed = _paired(predicted, observed)
    if bins < 3:
        raise ValueError('bins 至少为 3')
    edges = np.unique(np.quantile(predicted, np.linspace(0, 1, bins + 1)))
    groups = np.clip(np.digitize(predicted, edges[1:-1]), 0, len(edges) - 2)
    x_mean = np.array([predicted[groups == i].mean() for i in range(len(edges) - 1)])
    y_mean = np.array([observed[groups == i].mean() for i in range(len(edges) - 1)])
    low = min(predicted.min(), observed.min()); high = max(predicted.max(), observed.max())
    ax.plot([low, high], [low, high], color='#666666', linestyle='--', label='理想')
    ax.plot(x_mean, y_mean, color=SCIENCE_LIST[0], marker='o', label='校准')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style_axis(ax, grid='both')
    ax.legend()
    return ax


def make_error_distribution(ax, errors, xlabel='误差'):
    """Draw a normalized error histogram with median and zero references."""
    try:
        from .export_figure import style_axis
        from .palette import SCIENCE_LIST
    except ImportError:
        from export_figure import style_axis
        from palette import SCIENCE_LIST
    values = np.asarray(errors, dtype=float); values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError('errors 至少需要三个有限值')
    ax.hist(values, bins='auto', density=True, color=SCIENCE_LIST[0], alpha=0.45,
            edgecolor='white')
    ax.axvline(0, color='#333333', linestyle='--', label='零误差')
    ax.axvline(np.median(values), color=SCIENCE_LIST[1], label='中位数')
    ax.set(xlabel=xlabel, ylabel='密度')
    style_axis(ax, grid='y')
    ax.legend()
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_panel_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_panel_figure, save_figure
    apply_mcm_style()
    rng = np.random.default_rng(12)
    fitted = np.linspace(10, 100, 80)
    residuals = rng.normal(0, 4 + fitted * 0.025)
    observed = fitted + residuals
    zh = language == 'zh'
    fig, axes = new_panel_figure(2, 2, aspect=0.72)
    make_residual_plot(axes[0, 0], fitted, residuals,
                       '拟合值' if zh else 'Fitted', '残差' if zh else 'Residual')
    make_qq_plot(axes[0, 1], residuals,
                 '理论分位数' if zh else 'Theoretical quantile',
                 '样本分位数' if zh else 'Sample quantile')
    make_calibration_plot(axes[1, 0], fitted, observed,
                          xlabel='预测值' if zh else 'Predicted',
                          ylabel='观测均值' if zh else 'Observed mean')
    make_error_distribution(axes[1, 1], residuals, '误差' if zh else 'Error')
    output = THIS_DIR / 'samples' / (stem or ('diagnostics_demo' if zh else 'diagnostics_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
