# -*- coding: utf-8 -*-
"""Automated checks for publication-figure sizing, styling, and exports."""

from dataclasses import dataclass, field
from pathlib import Path
import re

from matplotlib.colors import to_rgb
import numpy as np


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self):
        return not self.errors

    def add(self, level, message):
        getattr(self, level).append(message)


def _unique_colors(figure):
    colors = []
    for axis in figure.axes:
        for line in axis.lines:
            try:
                color = tuple(np.round(to_rgb(line.get_color()), 3))
            except (TypeError, ValueError):
                continue
            if color not in colors and color not in {(0.0, 0.0, 0.0), (0.2, 0.2, 0.2)}:
                colors.append(color)
    return colors


def audit_figure(figure, theme='journal', require_units=False):
    """Inspect a live Matplotlib figure without mutating it."""
    report = AuditReport()
    width, height = figure.get_size_inches()
    if width <= 0 or height <= 0:
        report.errors.append('画布尺寸无效')
    if theme == 'journal' and height * 25.4 > 170.5:
        report.warnings.append('画布高度超过 Nature 建议的 170 mm')

    for index, axis in enumerate(figure.axes, start=1):
        if not axis.get_visible():
            continue
        if not axis.get_xlabel() and not axis.get_ylabel():
            report.warnings.append(f'轴 {index} 缺少坐标轴标签')
        if require_units:
            for label in (axis.get_xlabel(), axis.get_ylabel()):
                if label and '(' not in label and '（' not in label:
                    report.warnings.append(f'轴 {index} 标签可能缺少括号单位: {label}')
        for line in axis.lines:
            width_pt = float(line.get_linewidth())
            if theme == 'journal' and width_pt < 0.25:
                report.errors.append(f'轴 {index} 存在线宽小于 0.25 pt 的线')
            if theme == 'journal' and width_pt > 1.5:
                report.warnings.append(f'轴 {index} 存在线宽大于 1.5 pt 的线')
        for text in [axis.title, axis.xaxis.label, axis.yaxis.label,
                     *axis.get_xticklabels(), *axis.get_yticklabels(), *axis.texts]:
            if not text.get_text():
                continue
            size = float(text.get_fontsize())
            if theme == 'journal' and size < 5:
                report.errors.append(f'轴 {index} 存在小于 5 pt 的文字')
            if theme == 'journal' and size > 8.5:
                report.warnings.append(f'轴 {index} 存在大于 8.5 pt 的文字')

    colors = _unique_colors(figure)
    if len(colors) > 5:
        report.warnings.append(f'图中使用 {len(colors)} 种折线颜色，建议不超过 5 种')
    return report


def audit_export(path, min_dpi=300):
    """Check a saved raster/vector file for basic publication properties."""
    path = Path(path)
    report = AuditReport()
    if not path.exists() or path.stat().st_size < 1000:
        report.errors.append('导出文件缺失或过小')
        return report
    suffix = path.suffix.lower()
    if suffix in {'.png', '.tif', '.tiff'}:
        from PIL import Image
        with Image.open(path) as image:
            dpi = image.info.get('dpi')
            if dpi and min(dpi) + 1 < min_dpi:
                report.errors.append(f'位图 dpi 低于 {min_dpi}: {dpi}')
            if image.mode not in {'RGB', 'RGBA', 'L'}:
                report.warnings.append(f'非常规颜色模式: {image.mode}')
    elif suffix == '.svg':
        content = path.read_text(encoding='utf-8', errors='ignore')
        if not re.search(r'<text\b', content):
            report.errors.append('SVG 未保留可编辑文本')
    return report
