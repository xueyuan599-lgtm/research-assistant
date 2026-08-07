# -*- coding: utf-8 -*-
"""Shared styling, sizing, annotation, and publication export helpers."""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 89
DOUBLE_COLUMN_MM = 183
ONE_HALF_COLUMN_MM = 136
MAX_JOURNAL_HEIGHT_MM = 170
DEFAULT_ASPECT = 0.75

CJK_CANDIDATES = [
    'Microsoft YaHei', 'Source Han Sans SC', 'Source Han Sans CN',
    'Noto Sans CJK SC', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei',
]

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_STYLE = THIS_DIR / 'mcm_nature.mplstyle'
LEGACY_STYLE = THIS_DIR / 'mcm_legacy.mplstyle'
THEMES = {
    'journal': DEFAULT_STYLE,
    'legacy': LEGACY_STYLE,
}


def setup_chinese_font():
    """Select the first registered CJK font and return its family name."""
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in installed:
            plt.rcParams['font.sans-serif'] = [name, 'Arial', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return name
    warnings.warn(
        '未找到中文字体；请安装 Microsoft YaHei、Source Han Sans SC '
        '或 Noto Sans CJK SC 以避免中文显示为方块。',
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def apply_mcm_style(style_path=None, palette='science', theme='journal'):
    """Apply a bundled theme while preserving the historical call signature."""
    if theme not in THEMES:
        raise ValueError(f'未知主题: {theme}; 可选 {tuple(THEMES)}')
    plt.style.use(str(style_path or THEMES[theme]))
    setup_chinese_font()
    from palette import apply_palette
    apply_palette(palette)
    return theme


def figure_size(width='single', aspect=DEFAULT_ASPECT):
    """Return figure size in inches for a journal column width."""
    if isinstance(width, (int, float)):
        width_mm = float(width)
    else:
        widths = {
            'single': SINGLE_COLUMN_MM,
            'one_half': ONE_HALF_COLUMN_MM,
            'double': DOUBLE_COLUMN_MM,
        }
        if width not in widths:
            raise ValueError(
                "width 必须是 'single'、'one_half'、'double' 或毫米数值"
            )
        width_mm = widths[width]
    if aspect <= 0:
        raise ValueError('aspect 必须大于 0')
    width_in = width_mm / MM_PER_INCH
    return width_in, width_in * aspect


def new_figure(width='single', aspect=DEFAULT_ASPECT, **subplots_kw):
    """Create a column-width-aware figure and axes."""
    return plt.subplots(figsize=figure_size(width, aspect), **subplots_kw)


def label_panels(axes, labels=None, x=-0.12, y=1.04):
    """Label flattened axes with Nature-style bold lowercase letters."""
    import numpy as np

    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    if labels is None:
        labels = [chr(ord('a') + index) for index in range(len(flat_axes))]
    if len(labels) != len(flat_axes):
        raise ValueError('labels 数量必须与 axes 数量一致')
    for axis, label in zip(flat_axes, labels):
        add_panel_label(axis, str(label), x=x, y=y)
    return axes


def new_panel_figure(nrows=1, ncols=1, width='double', aspect=0.55,
                     panel_labels=True, labels=None, **subplots_kw):
    """Create a compact multi-panel figure with optional automatic labels."""
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figure_size(width, aspect), **subplots_kw
    )
    if panel_labels:
        label_panels(axes, labels=labels)
    return fig, axes


def shared_legend(fig, axes, loc='upper center', ncol=None, **legend_kw):
    """Create one de-duplicated legend shared by multiple axes."""
    import numpy as np

    handles = []
    labels = []
    for axis in np.asarray(axes, dtype=object).reshape(-1):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return None
    return fig.legend(
        handles, labels, loc=loc, ncol=ncol or min(len(labels), 4), **legend_kw
    )


def style_axis(ax, grid='y', zero_line=False):
    """Apply quiet chart scaffolding while preserving left/bottom anchors."""
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    if grid in {'x', 'y', 'both'}:
        axis = 'both' if grid == 'both' else grid
        ax.grid(axis=axis, color='#D9D9D9', linewidth=0.45, alpha=0.65)
    if zero_line:
        ax.axhline(0, color='#6B6B6B', linewidth=0.65, zorder=1)
    ax.margins(x=0.02)
    return ax


def add_panel_label(ax, label, x=-0.12, y=1.04):
    """Add a bold panel label such as ``a`` or ``b``."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight='bold',
            ha='left', va='bottom', clip_on=False)
    return ax


def save_figure(fig, filename, formats=('pdf', 'svg', 'png'), dpi=600,
                bbox_inches='tight', pad_inches=0.04, close=True,
                metadata=None, transparent=False, audit=False):
    """Export editable vector masters and publication-size raster previews."""
    if dpi < 300:
        raise ValueError('期刊位图导出 dpi 不得低于 300')
    base = Path(filename)
    base.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'Creator': 'MCM Science figure templates', **(metadata or {})}
    saved = []
    for fmt in formats:
        normalized = fmt.lower().lstrip('.')
        if normalized not in {'pdf', 'svg', 'png', 'tif', 'tiff'}:
            raise ValueError(f'不支持的导出格式: {fmt}')
        path = base.with_suffix(f'.{normalized}')
        save_kw = {
            'dpi': dpi,
            'bbox_inches': bbox_inches,
            'pad_inches': pad_inches,
            'facecolor': 'white',
            'edgecolor': 'white',
            'transparent': transparent,
        }
        if normalized in {'pdf', 'svg', 'png'}:
            save_kw['metadata'] = metadata
        if normalized in {'tif', 'tiff'}:
            save_kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **save_kw)
        saved.append(str(path))
    if audit:
        from figure_audit import audit_figure
        report = audit_figure(fig)
        if report.errors:
            raise ValueError('图表审查失败: ' + '; '.join(report.errors))
    if close:
        plt.close(fig)
    return saved


if __name__ == '__main__':
    import numpy as np
    apply_mcm_style()
    fig, ax = new_figure()
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), label='正弦')
    ax.plot(x, np.cos(x), linestyle='--', label='余弦')
    ax.set(xlabel='角度 (rad)', ylabel='数值')
    style_axis(ax)
    ax.legend(ncol=2, loc='upper center')
    save_figure(fig, THIS_DIR / 'samples' / '_style_selfcheck')
