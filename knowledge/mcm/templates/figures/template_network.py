# -*- coding: utf-8 -*-
"""Static network topology and highlighted-path template."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def _networkx():
    try:
        import networkx as nx
    except ImportError as error:
        raise ImportError('网络模板需要可选依赖 networkx：pip install networkx') from error
    return nx


def make_network(ax, graph, positions=None, node_values=None, edge_values=None,
                 highlight_path=None, node_label='节点值', edge_label='边权重',
                 show_labels=True, seed=42, cmap='cividis'):
    """Draw a weighted network with an optional shortest/critical path."""
    nx = _networkx()
    if graph.number_of_nodes() == 0:
        raise ValueError('graph 至少需要一个节点')
    positions = positions or nx.spring_layout(graph, seed=seed)
    if set(positions) != set(graph.nodes):
        raise ValueError('positions 必须覆盖 graph 的全部节点')
    node_values = node_values or {node: 1.0 for node in graph.nodes}
    node_array = np.array([float(node_values.get(node, 0)) for node in graph.nodes])
    if not np.all(np.isfinite(node_array)):
        raise ValueError('node_values 包含非有限值')
    minimum, maximum = float(node_array.min()), float(node_array.max())
    if minimum == maximum:
        maximum = minimum + 1
    node_norm = Normalize(vmin=minimum, vmax=maximum)
    nodes = nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_color=node_array, cmap=cmap,
        vmin=node_norm.vmin, vmax=node_norm.vmax,
        node_size=260, edgecolors='white', linewidths=0.7,
    )
    nodes.set_norm(node_norm)
    regular_edges = list(graph.edges)
    path_edges = set()
    if highlight_path is not None:
        if len(highlight_path) < 2:
            raise ValueError('highlight_path 至少包含两个节点')
        path_edges = set(zip(highlight_path[:-1], highlight_path[1:]))
        if not graph.is_directed():
            path_edges |= {(target, source) for source, target in path_edges}
        if any(source not in graph or target not in graph for source, target in path_edges):
            raise ValueError('highlight_path 包含不存在的节点')
    base_edges = [edge for edge in regular_edges if edge not in path_edges]
    base_kw = {'arrows': graph.is_directed()}
    if graph.is_directed():
        base_kw['arrowsize'] = 9
    nx.draw_networkx_edges(graph, positions, edgelist=base_edges, ax=ax,
                           width=0.8, edge_color='#B8B8B8', **base_kw)
    highlighted = [edge for edge in regular_edges if edge in path_edges]
    if highlighted:
        highlight_kw = {'arrows': graph.is_directed()}
        if graph.is_directed():
            highlight_kw['arrowsize'] = 10
        nx.draw_networkx_edges(graph, positions, edgelist=highlighted, ax=ax,
                               width=2.0, edge_color='#E07A2D', **highlight_kw)
    if show_labels:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=6)
    if edge_values:
        labels = {edge: f'{float(edge_values[edge]):g}' for edge in graph.edges if edge in edge_values}
        nx.draw_networkx_edge_labels(graph, positions, edge_labels=labels, ax=ax,
                                     font_size=5, rotate=False)
    colorbar = ax.figure.colorbar(nodes, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label(node_label)
    ax.set_axis_off()
    ax.margins(0.10)
    return ax


def demo(language='zh', stem=None):
    nx = _networkx()
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style(); graph = nx.DiGraph()
    edges = [('A', 'B', 4), ('A', 'C', 2), ('B', 'D', 3), ('C', 'D', 2),
             ('C', 'E', 5), ('D', 'F', 2), ('E', 'F', 3)]
    graph.add_weighted_edges_from(edges)
    positions = {'A': (0, .5), 'B': (1, 1), 'C': (1, 0), 'D': (2, .75),
                 'E': (2, -.25), 'F': (3, .4)}
    fig, ax = new_figure(width='double', aspect=0.36)
    make_network(ax, graph, positions=positions,
                 node_values={node: graph.degree(node) for node in graph},
                 edge_values={(u, v): d['weight'] for u, v, d in graph.edges(data=True)},
                 highlight_path=['A', 'C', 'D', 'F'],
                 node_label='节点度' if language == 'zh' else 'Node degree')
    output = THIS_DIR / 'samples' / (stem or ('network_demo' if language == 'zh' else 'network_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
