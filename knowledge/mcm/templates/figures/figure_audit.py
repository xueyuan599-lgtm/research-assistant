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
    info: list[str] = field(default_factory=list)

    @property
    def passed(self):
        return not self.errors

    def add(self, level, message):
        getattr(self, level).append(message)


def _artist_fill_color(artist):
    try:
        rgba = artist.get_facecolor()
        if rgba is None or len(rgba) < 3:
            return None
        color = tuple(np.round(to_rgb(rgba), 3))
    except (TypeError, ValueError):
        return None
    if color in {(0.0, 0.0, 0.0), (0.2, 0.2, 0.2), (1.0, 1.0, 1.0)}:
        return None
    return color


def _unique_colors(figure, include_fill=False):
    """Count unique colors; with include_fill also scan collection/patch fills."""
    colors = []
    fill_colors = []
    for axis in figure.axes:
        for line in axis.lines:
            try:
                color = tuple(np.round(to_rgb(line.get_color()), 3))
            except (TypeError, ValueError):
                continue
            if color not in colors and color not in {(0.0, 0.0, 0.0), (0.2, 0.2, 0.2)}:
                colors.append(color)
        if include_fill:
            for collection in axis.collections:
                color = _artist_fill_color(collection)
                if color and color not in fill_colors:
                    fill_colors.append(color)
            for patch in axis.patches:
                color = _artist_fill_color(patch)
                if color and color not in fill_colors:
                    fill_colors.append(color)
    return fill_colors if include_fill else colors


def _angular_interval(vertices):
    """Return the closed [min, max] theta interval (radians) of a polar polygon.

    Polar fill path vertices are stored as (theta, radius) data coordinates with
    theta in radians, so the angular extent is read directly from column 0.
    """
    angles = [float(v[0]) for v in vertices]
    lo, hi = min(angles), max(angles)
    return lo, hi


def _violin_overlap_report(figure):
    """Best-effort warning when angularly adjacent petals overlap.

    Template petals each carry a unique ``v:{band}`` gid; the meaningful check
    is between petals whose angular centres are neighbours (they are the ones
    that can visibly overlap), regardless of band label.
    """
    from matplotlib.collections import PolyCollection

    findings = []
    for axis in figure.axes:
        petals = []
        for collection in axis.collections:
            if not isinstance(collection, PolyCollection):
                continue
            gid = collection.get_gid()
            if not gid or not str(gid).startswith('v:'):
                continue
            lo, hi = _angular_interval(collection.get_paths()[0].vertices)
            petals.append((str(gid), lo, hi, (lo + hi) / 2))
        petals.sort(key=lambda item: item[3])
        for previous, current in zip(petals, petals[1:]):
            prev_gid, prev_lo, prev_hi = previous[0], previous[1], previous[2]
            cur_gid, cur_lo, cur_hi = current[0], current[1], current[2]
            overlap = max(0.0, min(prev_hi, cur_hi) - max(prev_lo, cur_lo))
            total = max(prev_hi - prev_lo, cur_hi - cur_lo)
            if total > 0 and overlap / total > 0.25:
                findings.append(
                    f'{prev_gid} 与 {cur_gid} 角区间交叠 {overlap / total:.2f}'
                )
    return findings


_DASHED_STYLES = {'--', ':', '-.', 'dashed', 'dotted', 'dashdot'}
# ASCII-only word boundaries: Python's \b treats CJK as word characters, so
# "总体mean=1.5" would slip past \bmean\b. Guard on letters instead.
_MEAN_RE = re.compile(r'(均值|平均|(?<![A-Za-z])mean(?![A-Za-z])|(?<![A-Za-z])average(?![A-Za-z]))', re.IGNORECASE)
_ERROR_METRIC_RE = re.compile(r'(error|误差|MAE|RMSE|MSE|R²|R2)', re.IGNORECASE)
_NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?')
_MAX_GROUP_BARS = 12


def _linestyle_is_dashed(style):
    """Dash detection that survives every shape matplotlib hands back.

    Line2D returns a plain string ('--'). LineCollection returns a per-segment
    list of ``(offset, onoffseq)`` pairs whose onoffseq is itself a list — those
    entries are unhashable, so a plain ``value in set`` test raises TypeError.
    Solid segments carry ``None`` in the second slot; dashed ones a non-empty
    on/off sequence.
    """
    if isinstance(style, str):
        return style in _DASHED_STYLES
    if isinstance(style, (list, tuple)):
        for item in style:
            if isinstance(item, str):
                if item in _DASHED_STYLES:
                    return True
            elif isinstance(item, (list, tuple)):
                if len(item) == 2 and isinstance(item[1], (list, tuple)) and len(item[1]) > 0:
                    return True
                if _linestyle_is_dashed(item):
                    return True
        return False
    return False


def _reference_lines(axis):
    """Yield ``(artist, orientation, coord)`` for threshold-like reference lines.

    ``axhline``/``axvline`` are Line2D anchored on a non-data transform because
    they span the whole axis; ``ax.hlines``/``ax.vlines`` are LineCollections.
    Plotted data series also live in ``axis.lines``, so the transform test is
    what separates a reference line from a dashed data curve. ``EventCollection``
    is a LineCollection subclass holding plotted events, not a threshold, so it
    is excluded too. ``coord`` is the data value the line sits at.
    """
    from matplotlib.collections import EventCollection, LineCollection

    for line in axis.lines:
        try:
            is_reference = line.get_transform() is not axis.transData
        except (AttributeError, TypeError):
            is_reference = False
        if is_reference:
            xy = np.asarray(line.get_xydata(), dtype=float)
            orientation = 'h' if len(set(xy[:, 1])) == 1 else 'v'
            yield line, orientation, xy[0][1] if orientation == 'h' else xy[0][0]
    for collection in axis.collections:
        if not isinstance(collection, LineCollection) or isinstance(collection, EventCollection):
            continue
        segments = collection.get_segments()
        if not segments:
            continue
        first = np.asarray(segments[0], dtype=float)
        orientation = 'h' if len(set(first[:, 1])) == 1 else 'v'
        yield collection, orientation, first[0][1] if orientation == 'h' else first[0][0]


def _artist_anchor_display(artist):
    """First vertex of a reference line, in display coordinates."""
    try:
        if hasattr(artist, 'get_segments') and artist.get_segments():
            vertex = np.asarray(artist.get_segments()[0], dtype=float)[0]
        else:
            vertex = np.asarray(artist.get_xydata(), dtype=float)[0]
        return artist.get_transform().transform(vertex)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _text_quotes_value(axis, value):
    """True when some axis text spells out the reference line's own value.

    A corner caption like "基准 = 158.0" annotates its line semantically even
    though it sits far from it in display space, which a pure proximity test
    cannot see.
    """
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    tolerance = max(abs(target) * 1e-3, 1e-9)
    for text in axis.texts:
        for token in _NUMBER_RE.findall(text.get_text()):
            try:
                if abs(float(token) - target) <= tolerance:
                    return True
            except ValueError:
                continue
    return False


def _has_adjacent_label(axis, artist, orientation, coord):
    """True when an axis text sits close enough to explain a reference line.

    A text drawn in data coordinates is compared in data space directly — that
    path is valid before the figure is drawn, whereas an ``ax.annotate`` text
    transform only resolves to real display coordinates after layout, so reading
    it pre-draw would report a stale position. Everything else (axes-fraction
    captions) is compared in display space, which is layout-independent.
    """
    anchor_display = _artist_anchor_display(artist)
    try:
        if orientation == 'h':
            span_display = abs(axis.bbox.y1 - axis.bbox.y0)
            span_data = abs(axis.get_ylim()[1] - axis.get_ylim()[0])
            index = 1
        else:
            span_display = abs(axis.bbox.x1 - axis.bbox.x0)
            span_data = abs(axis.get_xlim()[1] - axis.get_xlim()[0])
            index = 0
    except (AttributeError, TypeError, ValueError):
        return False
    for text in axis.texts:
        if not text.get_text().strip():
            continue
        try:
            position = text.get_position()
            # An Annotation keeps its own blended transform, so `is transData`
            # is false even when its text was positioned in data coordinates.
            # Its declared coordinate system is the reliable signal.
            coords = getattr(text, 'textcoords', None) or getattr(text, 'xycoords', None)
            if text.get_transform() is axis.transData or coords == 'data':
                if span_data > 0 and abs(float(position[index]) - float(coord)) <= 0.06 * span_data:
                    return True
                continue
            point = text.get_transform().transform(position)
        except (AttributeError, TypeError, ValueError):
            continue
        if anchor_display is not None and span_display > 0:
            if abs(point[index] - anchor_display[index]) <= 0.06 * span_display:
                return True
    return False


def _reference_line_report(figure):
    """Warn about dashed reference lines left without an adjacent annotation.

    A dashed threshold/target/baseline carries decision semantics; with no legend
    label and no text near the line the reader cannot tell which it is. Data
    curves drawn dashed from ``ax.plot`` are excluded by construction (they sit
    on the data transform), and a line counts as annotated when some axis text
    lands within ~6% of the axis span — so a label elsewhere on the figure no
    longer masks an unlabelled threshold.
    """
    findings = []
    for index, axis in enumerate(figure.axes, start=1):
        if not axis.get_visible():
            continue
        for artist, orientation, coord in _reference_lines(axis):
            try:
                if not _linestyle_is_dashed(artist.get_linestyle()):
                    continue
            except (AttributeError, TypeError, ValueError):
                continue
            try:
                label = artist.get_label() or ''
            except (AttributeError, TypeError):
                label = ''
            if label and not label.startswith('_'):
                continue
            if _has_adjacent_label(axis, artist, orientation, coord):
                continue
            if _text_quotes_value(axis, coord):
                continue
            findings.append(
                f'轴 {index} 存在无标注虚线参照线（阈值/目标线须用文字标明含义）'
            )
            break
    return findings


def _paired_interval_report(figure):
    """Warn when a figure claims a mean but draws no paired uncertainty interval.

    Fires only for a small set of bars on a categorical axis — the group
    comparison case, where the reader is invited to rank groups that carry no
    error scale. Histograms (many bins on a continuous axis), count charts, and
    bars that already carry an ErrorbarContainer all stay silent.
    """
    from matplotlib.container import BarContainer, ErrorbarContainer

    findings = []
    for index, axis in enumerate(figure.axes, start=1):
        if not axis.get_visible():
            continue
        labels = [t.get_text() for t in axis.texts]
        claimed = any(_MEAN_RE.search(text) for text in labels)
        # "Mean Absolute Error = 0.32" names an error metric, not a group mean.
        if not claimed or any(_ERROR_METRIC_RE.search(text) for text in labels):
            continue
        if any(isinstance(c, ErrorbarContainer) for c in axis.containers):
            continue
        bars = [
            patch
            for container in axis.containers
            if isinstance(container, BarContainer)
            for patch in container
        ]
        if not bars or len(bars) > _MAX_GROUP_BARS:
            continue
        # Histogram bins sit flush against each other; group bars carry a gap.
        # Testing the gap (rather than whether the x ticks are numeric) keeps
        # numeric-x group charts — years, distances, costs — in scope.
        bars.sort(key=lambda patch: patch.get_x())
        widths = [abs(patch.get_width()) for patch in bars]
        if max(widths) > 0:
            gaps = [
                right.get_x() - (left.get_x() + left.get_width())
                for left, right in zip(bars, bars[1:])
            ]
            if gaps and max(abs(gap) for gap in gaps) <= 0.01 * max(widths):
                continue
        findings.append(
            f'轴 {index} 标注了均值但 {len(bars)} 个柱状点估计无配对区间'
            '（建议叠加 95% CI 误差棒）'
        )
    return findings


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
        is_polar = axis.name == 'polar'
        if is_polar:
            # §7.1.1 polar 专项：r 轴标签非空即满足，跳过通用缺失标签检查。
            if not axis.get_ylabel():
                report.warnings.append(f'极坐标轴 {index} 缺少径向(r)轴标签')
            def _theta_deg(value):
                # matplotlib stores polar thetalim in degrees, but passing
                # degree-valued set_xticks silently rescales it by 180/pi;
                # detect that large flipped magnitude and undo it.
                value = float(value)
                if abs(value) > 360:
                    value = value * (np.pi / 180)
                return value

            span_deg = abs(_theta_deg(axis.get_thetamax()) - _theta_deg(axis.get_thetamin()))
            if span_deg >= 300:
                report.warnings.append(
                    f'轴 {index} θ 跨度 {span_deg:.0f}° 疑似整圆，径向分布建议半圆/扇区化'
                )
            if not axis.get_yticklabels():
                report.warnings.append(f'轴 {index} 缺少径向刻度')
        elif not axis.get_xlabel() and not axis.get_ylabel():
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
    fill_colors = _unique_colors(figure, include_fill=True)
    if fill_colors:
        report.info.append(f'填充色 {len(fill_colors)} 种（折线 {len(colors)} 种）')
    report.warnings.extend(_violin_overlap_report(figure))
    report.warnings.extend(_reference_line_report(figure))
    report.warnings.extend(_paired_interval_report(figure))
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
