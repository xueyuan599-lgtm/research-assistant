# -*- coding: utf-8 -*-
"""Forecast trajectory with nested prediction intervals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def make_forecast(ax, x, observed, forecast, intervals, split=None,
                  xlabel='时间', ylabel='预测值', unit='', observed_label='观测',
                  forecast_label='预测', palette='science'):
    """Plot observed/forecast values and one or more nested intervals."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette
    x = np.asarray(x)
    observed = np.asarray(observed, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if x.ndim != 1 or observed.shape != x.shape or forecast.shape != x.shape:
        raise ValueError('x、observed、forecast 必须是一维等长数组')
    finite_prediction = np.isfinite(forecast)
    if not finite_prediction.any():
        raise ValueError('forecast 至少需要一个有限值')
    colors = get_palette(palette, 2)
    parsed = []
    for label, bounds in intervals.items():
        if len(bounds) != 2:
            raise ValueError(f'{label} 必须提供 lower 和 upper')
        lower, upper = (np.asarray(item, dtype=float) for item in bounds)
        if lower.shape != x.shape or upper.shape != x.shape or np.any(lower > upper):
            raise ValueError(f'{label} 区间形状无效或下界大于上界')
        width = np.nanmean(upper - lower)
        parsed.append((width, str(label), lower, upper))
    for index, (_, label, lower, upper) in enumerate(sorted(parsed, reverse=True)):
        alpha = 0.10 + 0.08 * index
        ax.fill_between(x, lower, upper, where=np.isfinite(lower) & np.isfinite(upper),
                        color=colors[1], alpha=min(alpha, 0.28), linewidth=0,
                        label=label, zorder=1)
    observed_mask = np.isfinite(observed)
    ax.plot(x[observed_mask], observed[observed_mask], color=colors[0], marker='o',
            markerfacecolor='white', label=observed_label, zorder=3)
    ax.plot(x[finite_prediction], forecast[finite_prediction], color=colors[1],
            linestyle='--', marker='s', markevery=max(1, x.size // 10),
            markerfacecolor='white', label=forecast_label, zorder=3)
    if split is not None:
        ax.axvline(split, color='#777777', linestyle=':', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'{ylabel} ({unit})' if unit else ylabel)
    style_axis(ax, grid='y')
    ax.legend(ncol=2)
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    x = np.arange(24)
    truth = 50 + 0.8 * x + 7 * np.sin(x / 3)
    observed = truth.copy(); observed[16:] = np.nan
    forecast = truth + np.r_[np.zeros(16), [1, -1, 2, 1, 3, 2, 4, 3]]
    spread = np.r_[np.full(16, np.nan), np.linspace(4, 10, 8)]
    intervals = {'95% CI': (forecast - spread, forecast + spread),
                 '80% CI': (forecast - spread * 0.6, forecast + spread * 0.6)}
    fig, ax = new_figure(aspect=0.70)
    make_forecast(ax, x, observed, forecast, intervals, split=15.5,
                  xlabel='月份' if language == 'zh' else 'Month',
                  ylabel='需求量' if language == 'zh' else 'Demand',
                  observed_label='观测' if language == 'zh' else 'Observed',
                  forecast_label='预测' if language == 'zh' else 'Forecast')
    output = THIS_DIR / 'samples' / (stem or ('forecast_demo' if language == 'zh' else 'forecast_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
