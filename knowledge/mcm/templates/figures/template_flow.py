# -*- coding: utf-8 -*-
"""Static vector Sankey/alluvial flow template."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def _depths(nodes, links):
    incoming = defaultdict(set); outgoing = defaultdict(set)
    for source, target, _ in links:
        incoming[target].add(source); outgoing[source].add(target)
    depth = {node: 0 for node in nodes if not incoming[node]}
    pending = set(nodes) - set(depth)
    while pending:
        progressed = False
        for node in list(pending):
            if incoming[node] and all(parent in depth for parent in incoming[node]):
                depth[node] = max(depth[parent] for parent in incoming[node]) + 1
                pending.remove(node); progressed = True
        if not progressed:
            raise ValueError('Sankey 仅支持无环流向网络')
    return depth


def make_sankey(ax, sources, targets, values, node_labels=None,
                palette='science', unit='', node_width=0.035, gap=0.035):
    """Draw a multi-stage Sankey diagram using editable Matplotlib paths."""
    try:
        from .palette import get_palette
    except ImportError:
        from palette import get_palette
    sources = list(map(str, sources)); targets = list(map(str, targets))
    values = np.asarray(values, dtype=float)
    if not sources or len(sources) != len(targets) or values.shape != (len(sources),):
        raise ValueError('sources、targets、values 必须等长且非空')
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError('values 必须为有限正数')
    nodes = list(dict.fromkeys([*sources, *targets])); links = list(zip(sources, targets, values))
    depth = _depths(nodes, links); max_depth = max(depth.values())
    columns = defaultdict(list)
    for node in nodes:
        columns[depth[node]].append(node)
    incoming_total = defaultdict(float); outgoing_total = defaultdict(float)
    for source, target, value in links:
        outgoing_total[source] += value; incoming_total[target] += value
    node_total = {node: max(incoming_total[node], outgoing_total[node]) for node in nodes}
    max_column_total = max(sum(node_total[node] for node in column) for column in columns.values())
    usable_height = 1 - gap * (max(len(column) for column in columns.values()) - 1)
    scale = usable_height / max_column_total
    colors = get_palette(palette, min(len(nodes), 5))
    color_map = {node: colors[index % len(colors)] for index, node in enumerate(nodes)}
    positions = {}
    for level, column in columns.items():
        column_height = sum(node_total[node] * scale for node in column) + gap * (len(column) - 1)
        cursor = (1 + column_height) / 2
        x = 0.03 + (0.94 * level / max(max_depth, 1))
        for node in column:
            height = node_total[node] * scale
            positions[node] = (x, cursor - height, height)
            cursor -= height + gap
    source_offsets = defaultdict(float); target_offsets = defaultdict(float)
    for source, target, value in sorted(links, key=lambda item: (depth[item[0]], item[0], item[1])):
        source_x, source_y, _ = positions[source]; target_x, target_y, _ = positions[target]
        thickness = value * scale
        sy0 = source_y + source_offsets[source]; sy1 = sy0 + thickness
        ty0 = target_y + target_offsets[target]; ty1 = ty0 + thickness
        source_offsets[source] += thickness; target_offsets[target] += thickness
        x0 = source_x + node_width; x1 = target_x; control = (x1 - x0) * 0.45
        vertices = [(x0, sy0), (x0 + control, sy0), (x1 - control, ty0), (x1, ty0),
                    (x1, ty1), (x1 - control, ty1), (x0 + control, sy1), (x0, sy1), (x0, sy0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color_map[source],
                               edgecolor='none', alpha=0.32))
    node_labels = node_labels or {}
    for node, (x, y, height) in positions.items():
        ax.add_patch(Rectangle((x, y), node_width, height, facecolor=color_map[node],
                               edgecolor='white', linewidth=0.5, zorder=3))
        label = node_labels.get(node, node)
        value_text = f' {node_total[node]:g}{unit}' if unit else ''
        ha = 'right' if depth[node] == max_depth else 'left'
        label_x = x - 0.006 if ha == 'right' else x + node_width + 0.006
        ax.text(label_x, y + height / 2, f'{label}{value_text}', ha=ha, va='center', fontsize=6)
    ax.set_xlim(0, 1.08); ax.set_ylim(0, 1); ax.set_axis_off()
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style(); zh = language == 'zh'
    labels = {'能源': '能源', '工业': '工业', '交通': '交通', '有效利用': '有效利用', '损失': '损失'} if zh else {}
    fig, ax = new_figure(width='double', aspect=0.36)
    make_sankey(ax, ['能源', '能源', '工业', '工业', '交通', '交通'],
                ['工业', '交通', '有效利用', '损失', '有效利用', '损失'],
                [65, 35, 44, 21, 24, 11], node_labels=labels, unit='%')
    if not zh:
        english = {'能源': 'Energy', '工业': 'Industry', '交通': 'Transport',
                   '有效利用': 'Useful', '损失': 'Loss'}
        ax.clear()
        make_sankey(ax, ['Energy', 'Energy', 'Industry', 'Industry', 'Transport', 'Transport'],
                    ['Industry', 'Transport', 'Useful', 'Loss', 'Useful', 'Loss'],
                    [65, 35, 44, 21, 24, 11], unit='%')
    output = THIS_DIR / 'samples' / (stem or ('flow_demo' if zh else 'flow_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
