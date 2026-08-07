# -*- coding: utf-8 -*-
"""Science-style tornado chart for one-at-a-time sensitivity analysis."""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_tornado(ax, params, low, high, baseline, ylabel='参数', unit='',
                 low_label='参数降低', high_label='参数提高', value_format='.1f',
                 show_legend=True):
    """Rank parameters by maximum absolute departure from ``baseline``."""
    from export_figure import style_axis

    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if len(params) == 0 or low.shape != high.shape or low.shape != (len(params),):
        raise ValueError('params、low、high 必须为等长非空序列')
    if not np.all(np.isfinite(np.r_[low, high, baseline])):
        raise ValueError('龙卷风图输入必须为有限数值')

    influence = np.maximum(np.abs(low - baseline), np.abs(high - baseline))
    order = np.argsort(influence)[::-1]
    params = [params[index] for index in order]
    low, high = low[order], high[order]
    low_color, high_color = '#3B6FB6', '#E07A2D'

    all_values = np.r_[low, high, baseline]
    span = max(np.ptp(all_values), max(abs(baseline), 1) * 0.05)
    padding = span * 0.035
    for row, (low_value, high_value) in enumerate(zip(low, high)):
        ax.barh(row, abs(low_value - baseline),
                left=min(low_value, baseline), height=0.54,
                color=low_color, edgecolor='#333333', linewidth=0.4,
                hatch='//', zorder=3)
        ax.barh(row, abs(high_value - baseline),
                left=min(high_value, baseline), height=0.54,
                color=high_color, edgecolor='#333333', linewidth=0.4,
                zorder=3)
        for value in (low_value, high_value):
            is_left = value < baseline
            ax.text(value + (-padding if is_left else padding), row,
                    format(value, value_format), va='center',
                    ha='right' if is_left else 'left', fontsize=6.2,
                    clip_on=False)

    ax.axvline(baseline, color='#222222', linewidth=0.85,
               linestyle='--', zorder=4)
    ax.set_yticks(np.arange(len(params)), params)
    ax.invert_yaxis()
    ax.set_ylim(len(params) - 0.5, -0.9)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f'结果值 ({unit})' if unit else '结果值')
    ax.set_xlim(all_values.min() - span * 0.14, all_values.max() + span * 0.14)
    style_axis(ax, grid='x')
    ax.text(0.99, 0.02, f'基准 = {baseline:{value_format}}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.3, color='#6B6B6B')
    if show_legend:
        handles = [
            Patch(facecolor=low_color, edgecolor='#333333', hatch='//',
                  linewidth=0.4, label=low_label),
            Patch(facecolor=high_color, edgecolor='#333333', linewidth=0.4,
                  label=high_label),
        ]
        ax.legend(handles=handles, ncol=2, loc='upper center',
                  borderaxespad=0.2)
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.86)
    zh = language == 'zh'
    params = (['小麦价格', '化肥成本', '亩产量', '销售数量', '种植面积'] if zh
              else ['Wheat price', 'Fertilizer', 'Yield', 'Sales', 'Area'])
    baseline = 158.0
    low = [148, 152, 150, 155, 153]
    high = [168, 161, 165, 160, 162]
    make_tornado(ax, params, low, high, baseline, unit='万元' if zh else '10k CNY',
                 ylabel='参数' if zh else 'Parameter',
                 low_label='参数降低' if zh else 'Lower input',
                 high_label='参数升高' if zh else 'Higher input')
    ax.set_title('参数灵敏度分析（±10% 扰动）' if zh else 'Sensitivity analysis (±10%)')
    if not zh:
        ax.set_xlabel('Outcome (10k CNY)')
        ax.texts[-1].set_text(f'Baseline = {baseline:.1f}')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('tornado_demo' if zh else 'tornado_demo_en')))


if __name__ == '__main__':
    demo()
