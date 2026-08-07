# -*- coding: utf-8 -*-
"""Optional GeoPandas choropleth and route-map templates."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def _geopandas():
    try:
        import geopandas as gpd
    except ImportError as error:
        raise ImportError('地图模板需要可选依赖：pip install geopandas shapely') from error
    return gpd


def make_choropleth(ax, geodata, value_col, cmap='cividis', legend_label='数值',
                    missing_color='#E6E6E6', edgecolor='white', annotate_col=None):
    """Draw a continuous choropleth from a GeoDataFrame."""
    _geopandas()
    if value_col not in geodata.columns:
        raise ValueError(f'GeoDataFrame 缺少列: {value_col}')
    values = np.asarray(geodata[value_col], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f'{value_col} 不包含有限值')
    geodata.plot(column=value_col, ax=ax, cmap=cmap, edgecolor=edgecolor,
                 linewidth=0.55, missing_kwds={'color': missing_color})
    norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    colorbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                                  fraction=0.035, pad=0.02)
    colorbar.set_label(legend_label)
    if annotate_col is not None:
        if annotate_col not in geodata.columns:
            raise ValueError(f'GeoDataFrame 缺少标注列: {annotate_col}')
        for _, row in geodata.iterrows():
            point = row.geometry.representative_point()
            ax.text(point.x, point.y, str(row[annotate_col]), ha='center', va='center', fontsize=5.5)
    ax.set_axis_off(); ax.set_aspect('equal')
    return ax


def make_route_map(ax, points, routes, boundary=None, labels=None,
                   route_values=None, xlabel='经度', ylabel='纬度'):
    """Plot nodes and one or more precomputed routes over an optional boundary."""
    try:
        from .palette import get_palette
    except ImportError:
        from palette import get_palette
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 2:
        raise ValueError('points 必须是 n×2 坐标数组')
    if not np.all(np.isfinite(coordinates)):
        raise ValueError('points 包含非有限值')
    if boundary is not None:
        _geopandas(); boundary.plot(ax=ax, facecolor='#F5F5F5', edgecolor='#B0B0B0', linewidth=0.6)
    colors = get_palette('science', len(routes))
    for index, route in enumerate(routes):
        route = np.asarray(route, dtype=int)
        if route.ndim != 1 or route.size < 2 or np.any((route < 0) | (route >= len(points))):
            raise ValueError('每条 route 必须是有效节点索引序列')
        width = 1.2 if route_values is None else 0.7 + 1.5 * float(route_values[index]) / max(route_values)
        ax.plot(coordinates[route, 0], coordinates[route, 1], color=colors[index],
                linewidth=width, marker='o', markerfacecolor='white', label=f'Route {index + 1}')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=16, color='#222222', zorder=4)
    if labels is not None:
        if len(labels) != len(points):
            raise ValueError('labels 长度必须与 points 一致')
        for point, label in zip(coordinates, labels):
            ax.annotate(str(label), point, xytext=(3, 3), textcoords='offset points', fontsize=5.5)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    return ax


def demo(language='zh', stem=None):
    gpd = _geopandas()
    from shapely.geometry import box
    try:
        from .export_figure import apply_mcm_style, new_panel_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_panel_figure, save_figure
    apply_mcm_style(); zh = language == 'zh'
    regions = gpd.GeoDataFrame(
        {'region': ['A', 'B', 'C', 'D'], 'value': [42, 65, 28, 81]},
        geometry=[box(0, 1, 1, 2), box(1, 1, 2, 2), box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs='EPSG:4326',
    )
    fig, axes = new_panel_figure(1, 2, aspect=0.38)
    make_choropleth(axes[0], regions, 'value', legend_label='需求指数' if zh else 'Demand index', annotate_col='region')
    points = np.array([[.2, .2], [.3, 1.6], [1.2, 1.7], [1.7, .3], [1.1, .8]])
    make_route_map(axes[1], points, [[0, 4, 2, 1], [0, 3, 4]], boundary=regions,
                   labels=['仓库', 'A', 'B', 'C', 'D'] if zh else ['Depot', 'A', 'B', 'C', 'D'],
                   route_values=[8, 5], xlabel='经度' if zh else 'Longitude', ylabel='纬度' if zh else 'Latitude')
    output = THIS_DIR / 'samples' / (stem or ('map_demo' if zh else 'map_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
