# -*- coding: utf-8 -*-
"""Original-style cumulative-bar and time-series composite figure template."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

COLOR_SCHEMES = {
    1: {
        'LF': '#EA6B5D',
        'HF': '#62BDBB',
        'spine': '#000000',
        'separator': '#E5C51A',
        'dash': '#B6C0CF',
        'panel': '#285296',
        'stages': ('#F8FCFC', '#EDF9F7', '#F5FBFB'),
    },
}


def _as_array(values, name, length=None):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f'{name} 必须是一维非空数组')
    if length is not None and array.size != length:
        raise ValueError(f'{name} 长度必须为 {length}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} 包含非有限值')
    return array


def _label_font(text, size):
    """Use an installed CJK font only when the supplied label needs it."""
    if not any('\u4e00' <= character <= '\u9fff' for character in str(text)):
        return {'fontsize': size, 'fontweight': 'bold'}
    installed = {font.name for font in fm.fontManager.ttflist}
    family = next((name for name in ('Microsoft YaHei', 'SimHei', 'SimSun')
                   if name in installed), 'DejaVu Sans')
    return {'fontproperties': fm.FontProperties(family=family, size=size, weight='bold')}


def significance_mark(p_value):
    """Convert a p value to a conventional significance label."""
    if p_value is None or not np.isfinite(p_value):
        return 'n.s.'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'n.s.'


def prepare_wide_data(df_raw, time_col='Time', group_columns=('LF', 'HF'),
                      replicate_col=None, cumulative_scale=1.0):
    """Summarize a wide table containing one row per time and replicate.

    When ``replicate_col`` is omitted, rows within each time point are matched
    by their order. ``cumulative_scale`` can convert the summed observations to
    the desired cumulative-emission unit.
    """
    required = {time_col, *group_columns}
    missing = required.difference(df_raw.columns)
    if missing:
        raise ValueError(f'Excel 缺少列: {sorted(missing)}')

    frame = df_raw.copy()
    dates = frame[time_col].drop_duplicates().tolist()
    if len(dates) < 2:
        raise ValueError('至少需要两个时间点')
    if replicate_col is None:
        counts = frame.groupby(time_col, sort=False).size()
        if counts.nunique() != 1:
            raise ValueError('各时间点重复数不一致，请提供 replicate_col')
        replicate_col = '_replicate'
        frame[replicate_col] = frame.groupby(time_col, sort=False).cumcount()
    elif replicate_col not in frame.columns:
        raise ValueError(f'Excel 缺少重复编号列: {replicate_col}')

    grouped = frame.groupby(time_col, sort=False)
    means = {}
    errors = {}
    cumulative = {}
    for group in group_columns:
        numeric = pd.to_numeric(frame[group], errors='raise')
        frame[group] = numeric
        means[group] = grouped[group].mean().reindex(dates).to_numpy()
        errors[group] = grouped[group].sem().fillna(0).reindex(dates).to_numpy()
        cumulative[group] = (
            frame.groupby(replicate_col, sort=False)[group].sum().to_numpy()
            * float(cumulative_scale)
        )

    try:
        from scipy import stats
        p_value = float(stats.ttest_ind(
            cumulative[group_columns[0]], cumulative[group_columns[1]],
            equal_var=False, nan_policy='omit',
        ).pvalue)
    except ImportError:
        p_value = np.nan

    return {
        'dates': dates,
        'line_means': means,
        'line_errors': errors,
        'bar_samples': cumulative,
        'p_value': p_value,
    }


def _stage_background(ax, boundaries, colors, y_limits, x_limits):
    edges = [-0.6, *boundaries, x_limits[1]]
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        base = colors[index % len(colors)]
        cmap = LinearSegmentedColormap.from_list(
            f'stage_{index}', ['#FFFFFF', base, '#FFFFFF']
        )
        gradient = np.linspace(0, 1, 256)[None, :]
        ax.imshow(
            gradient, extent=(left, right, y_limits[0], y_limits[1]),
            origin='lower', aspect='auto', cmap=cmap, alpha=0.95,
            interpolation='bicubic', zorder=0,
        )


def make_bar_timeseries(dates, line_means, line_errors, bar_samples,
                        groups=('LF', 'HF'), stage_boundaries=(4, 6),
                        p_value=None, significance=None, scheme_id=1,
                        bar_xlabel=r'N$_2$O cumulative emission (kg·N$_2$O·ha$^{-1}$)',
                        line_ylabel=r'N$_2$O emission flux' + '\n'
                                     r'($\mu$g·m$^{-2}$·h$^{-1}$)',
                        panel_label='(a)', figsize=(7, 6.5)):
    """Draw the reference-style two-panel composite and return the figure."""
    if scheme_id not in COLOR_SCHEMES:
        raise ValueError(f'未知配色方案: {scheme_id}')
    if len(groups) != 2:
        raise ValueError('该版式固定比较两个组')
    dates = list(dates)
    if len(dates) < 2:
        raise ValueError('至少需要两个时间点')

    colors = COLOR_SCHEMES[scheme_id]
    means = {group: _as_array(line_means[group], f'{group} 均值', len(dates))
             for group in groups}
    errors = {group: _as_array(line_errors[group], f'{group} 标准误', len(dates))
              for group in groups}
    samples = {group: _as_array(bar_samples[group], f'{group} 累计值')
               for group in groups}
    boundaries = sorted(float(value) for value in stage_boundaries)
    if any(value <= -0.5 or value >= len(dates) - 0.5 for value in boundaries):
        raise ValueError('stage_boundaries 必须位于时间序列内部')

    rc = {
        'font.family': 'serif',
        'font.serif': [
            'Times New Roman', 'Microsoft YaHei', 'Source Han Sans SC',
            'Noto Sans CJK SC', 'SimSun', 'DejaVu Serif',
        ],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'figure.autolayout': False,
        'figure.constrained_layout.use': False,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize, facecolor='white')
        grid = fig.add_gridspec(
            3, 1, height_ratios=(1.0, 0.34, 2.25), hspace=0.0
        )
        ax_bar = fig.add_subplot(grid[0])
        ax_caption = fig.add_subplot(grid[1])
        ax_line = fig.add_subplot(grid[2])
        ax_bar.set_zorder(3)
        ax_caption.set_zorder(2)
        ax_line.set_zorder(1)

        positions = np.array([1.0, 0.0])
        group_means = np.array([samples[group].mean() for group in groups])
        group_errors = np.array([
            samples[group].std(ddof=1) / np.sqrt(samples[group].size)
            if samples[group].size > 1 else 0.0 for group in groups
        ])
        for position, group, mean, error in zip(
                positions, groups, group_means, group_errors):
            ax_bar.barh(
                position, mean, xerr=error, height=0.55,
                color=colors[group], edgecolor='black', linewidth=2.2,
                error_kw={'elinewidth': 2.2, 'capsize': 5, 'capthick': 2.2},
                zorder=2,
            )
            offsets = np.linspace(-0.13, 0.13, samples[group].size)
            ax_bar.scatter(
                samples[group], position + offsets, s=38,
                facecolor=colors[group], edgecolor='black', linewidth=1.25,
                zorder=4,
            )

        x_max = max(
            np.max(np.concatenate(list(samples.values()))),
            np.max(group_means + group_errors),
        )
        x_limit = x_max * 1.34
        bracket_x = x_max * 1.06
        ax_bar.plot([bracket_x, bracket_x], [0.02, 0.98],
                    color='black', linewidth=2.2, clip_on=False)
        mark = significance or significance_mark(p_value)
        ax_bar.text(bracket_x + x_limit * 0.035, 0.5, mark,
                    fontsize=19, fontweight='bold', ha='left', va='center')
        ax_bar.set_yticks(positions, groups)
        ax_bar.set_xlim(0, x_limit)
        ax_bar.set_ylim(-0.58, 1.58)
        ax_bar.xaxis.set_major_locator(ticker.MaxNLocator(
            nbins=5, steps=[1, 1.5, 2, 2.5, 5, 10], prune='lower'
        ))
        ax_bar.tick_params(axis='both', width=2.2, length=6,
                           labelsize=12, direction='out')
        for label in ax_bar.get_yticklabels():
            label.set_fontweight('bold')
        for spine in ax_bar.spines.values():
            spine.set_color(colors['spine'])
            spine.set_linewidth(2.4)
        ax_bar.spines['bottom'].set_color(colors['separator'])
        ax_bar.spines['bottom'].set_linewidth(3.4)
        ax_bar.text(-0.17, 0.90, panel_label, transform=ax_bar.transAxes,
                    color=colors['panel'], fontsize=20, fontweight='bold',
                    ha='right', va='center', clip_on=False)

        ax_caption.set_xlim(0, 1)
        ax_caption.set_ylim(0, 1)
        ax_caption.set_xticks([])
        ax_caption.set_yticks([])
        ax_caption.patch.set_facecolor('white')
        ax_caption.spines['top'].set_visible(False)
        for side in ('left', 'right', 'bottom'):
            ax_caption.spines[side].set_color(colors['spine'])
            ax_caption.spines[side].set_linewidth(2.4)
        ax_caption.text(
            0.5, 0.34, bar_xlabel, transform=ax_caption.transAxes,
            ha='center', va='center', **_label_font(bar_xlabel, 13),
        )

        x = np.arange(len(dates))
        all_lower = np.concatenate([
            means[group] - errors[group] for group in groups
        ])
        all_upper = np.concatenate([
            means[group] + errors[group] for group in groups
        ])
        data_span = max(float(np.max(all_upper) - np.min(all_lower)), 1.0)
        y_limits = (
            min(0.0, float(np.min(all_lower) - 0.08 * data_span)),
            float(np.max(all_upper) + 0.10 * data_span),
        )
        x_limits = (-0.6, len(dates) - 0.4)
        _stage_background(ax_line, boundaries, colors['stages'], y_limits, x_limits)
        for boundary in boundaries:
            ax_line.axvline(
                boundary, color=colors['dash'], linestyle=(0, (5, 5)),
                linewidth=2.5, zorder=1,
            )

        markers = ('o', 'o')
        for marker, group in zip(markers, groups):
            ax_line.errorbar(
                x, means[group], yerr=errors[group], color=colors[group],
                marker=marker, markersize=8.5, markerfacecolor=colors[group],
                markeredgecolor=colors[group], linewidth=3.0,
                elinewidth=2.3, capsize=5, capthick=2.3,
                label=group, zorder=3,
            )

        ax_line.set_xlim(*x_limits)
        ax_line.set_ylim(*y_limits)
        ax_line.set_xticks(x, dates, rotation=90, ha='center')
        ax_line.set_ylabel(line_ylabel, labelpad=11, **_label_font(line_ylabel, 14))
        ax_line.yaxis.set_major_locator(ticker.MaxNLocator(
            nbins=4, steps=[1, 2, 2.5, 5, 10], prune='upper'
        ))
        ax_line.tick_params(axis='both', width=2.2, length=6,
                            labelsize=11, direction='out')
        for label in (*ax_line.get_xticklabels(), *ax_line.get_yticklabels()):
            label.set_fontweight('bold')
        for spine in ax_line.spines.values():
            spine.set_color(colors['spine'])
            spine.set_linewidth(2.4)
        fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.18)
    return fig, (ax_bar, ax_line)


def plot_advanced_forest_chart(df_real, scheme_id=1, output_path=None,
                               stage_boundaries=(4, 6)):
    """Backward-compatible wrapper for the code structure in the tutorial."""
    fig, axes = make_bar_timeseries(
        dates=df_real['dates'],
        line_means=df_real['line_means'],
        line_errors=df_real['line_errors'],
        bar_samples=df_real['bar_samples'],
        groups=tuple(df_real.get('groups', ('LF', 'HF'))),
        stage_boundaries=stage_boundaries,
        p_value=df_real.get('p_value'),
        significance=df_real.get('significance'),
        scheme_id=scheme_id,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    return fig, axes


def plot_from_excel(excel_path, output_path, time_col='Time',
                    group_columns=('LF', 'HF'), replicate_col=None,
                    cumulative_scale=1.0, stage_boundaries=(4, 6),
                    scheme_id=1):
    """Read Excel, summarize observations, run Welch's test, and save PNG."""
    df_raw = pd.read_excel(excel_path)
    prepared = prepare_wide_data(
        df_raw, time_col=time_col, group_columns=group_columns,
        replicate_col=replicate_col, cumulative_scale=cumulative_scale,
    )
    prepared['groups'] = tuple(group_columns)
    return plot_advanced_forest_chart(
        prepared, scheme_id=scheme_id, output_path=output_path,
        stage_boundaries=stage_boundaries,
    )


def demo(language='zh', stem=None):
    dates = [
        '20 Oct.', '26 Oct.', '31 Oct.', '05 Nov.', '13 Nov.',
        '31 Dec.', '12 Mar.', '20 Mar.', '28 Mar.', '05 Apr.', '15 Apr.',
    ]
    prepared = {
        'dates': dates,
        'line_means': {
            'LF': [18, 48, 27, 6, 4, 10, 8, 24, 82, 57, 10],
            'HF': [22, 45, 32, 25, 5, 11, 33, 39, 152, 79, 14],
        },
        'line_errors': {
            'LF': [4, 8, 5, 3, 2, 3, 3, 8, 11, 9, 4],
            'HF': [5, 7, 6, 4, 2, 3, 5, 7, 18, 10, 5],
        },
        'bar_samples': {
            'LF': [260, 305, 310, 318, 322],
            'HF': [438, 452, 460, 477, 495],
        },
        'groups': ('LF', 'HF'),
        'significance': '**',
    }
    zh = language == 'zh'
    output = THIS_DIR / 'samples' / f"{stem or ('bar_timeseries_demo' if zh else 'bar_timeseries_demo_en')}.png"
    fig, _ = make_bar_timeseries(
        dates=prepared['dates'], line_means=prepared['line_means'],
        line_errors=prepared['line_errors'], bar_samples=prepared['bar_samples'],
        groups=prepared['groups'], stage_boundaries=(4, 6), significance='**',
        bar_xlabel=(r'$N_2O$累计排放（kg·$N_2O$·ha$^{-1}$）' if zh else
                    r'N$_2$O cumulative emission (kg·N$_2$O·ha$^{-1}$)'),
        line_ylabel=(r'$N_2O$排放通量' + '\n' + r'（$\mu$g·m$^{-2}$·h$^{-1}$）' if zh else
                     r'N$_2$O emission flux' + '\n' + r'($\mu$g·m$^{-2}$·h$^{-1}$)'),
    )
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output


if __name__ == '__main__':
    demo()
