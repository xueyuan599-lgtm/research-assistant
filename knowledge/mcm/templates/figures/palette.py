# -*- coding: utf-8 -*-
"""Science-style color system with color-blind and grayscale safeguards."""

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

INK = '#222222'
MUTED = '#6B6B6B'
GRID = '#D9D9D9'
LIGHT_GRID = '#ECECEC'
WHITE = '#FFFFFF'

SCIENCE = {
    '蓝': '#3B6FB6',
    '橙': '#E07A2D',
    '青': '#2A9D8F',
    '紫': '#8E6C8A',
    '金': '#D6A532',
}
SCIENCE_LIST = list(SCIENCE.values())

# Kept for backward compatibility with existing MCM scripts.
OKABE_ITO = {
    '蓝': '#0072B2',
    '橙': '#E69F00',
    '绿': '#009E73',
    '粉': '#CC79A7',
    '黄': '#F0E442',
    '天蓝': '#56B4E9',
    '金': '#E69F00',
    '红': '#D55E00',
    '紫': '#CC79A7',
    '黑': '#000000',
}
OKABE_ITO_LIST = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7',
    '#56B4E9', '#D55E00', '#F0E442',
]

MCM_SCENARIO = {
    '基准': SCIENCE['蓝'],
    '改进': SCIENCE['橙'],
    '方案一': SCIENCE['蓝'],
    '方案二': SCIENCE['橙'],
    '预测': SCIENCE['橙'],
    '实测': SCIENCE['蓝'],
    '悲观': '#A56A6A',
    '乐观': SCIENCE['青'],
    '基准线': INK,
}

# Tableau 调色板子集：径向小提琴等按类别语义取色使用（Total/Day/Night 固定映射）。
JOURNAL_2 = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
             '#B279A2', '#FFBF79', '#9D755D', '#BAB0AC']

# Paul Tol 色盲安全体系（SciencePlots 抄录）：高对比四色 + 亮色变体，
# 用于线宽受限或需要与 SCIENCE 系区分的场合。
TOL_BRIGHT = {
    '蓝': '#4477AA',
    '青': '#66CCEE',
    '绿': '#228833',
    '黄': '#CCBB44',
    '红': '#EE6677',
    '紫': '#AA3377',
    '灰': '#BBBBBB',
}
TOL_BRIGHT_LIST = list(TOL_BRIGHT.values())

# 情景分解（SimDec 式）扩展语义色：情景一/二在 SCIENCE 蓝基础上错开，
# 第三情景预留金色，保证色盲/灰度下仍可区分。
SCENARIO_EXTRA = {
    '情景一': SCIENCE['蓝'],
    '情景二': SCIENCE['青'],
    '情景三': SCIENCE['金'],
}

SEQUENTIAL_CMAPS = ['cividis', 'viridis', 'magma']
DIVERGING_CMAPS = ['RdBu_r', 'PuOr', 'BrBG']
SCIENCE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'science_blue', ['#F7FAFD', '#C9D9EC', '#7FA6D2', '#3B6FB6', '#17365D']
)


def apply_palette(name='science'):
    """Set a deterministic categorical color cycle."""
    palettes = {
        'science': SCIENCE_LIST,
        'okabe_ito': OKABE_ITO_LIST,
        'mcm': list(dict.fromkeys(MCM_SCENARIO.values())),
        'journal_2': JOURNAL_2,
        'tol_bright': TOL_BRIGHT_LIST,
    }
    if name not in palettes:
        raise ValueError(f'未知调色板: {name}; 可选 {tuple(palettes)}')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palettes[name])


def get_palette(name='science', n=None):
    """Return ``n`` explicit colors without relying on library defaults."""
    palettes = {
        'science': SCIENCE_LIST,
        'okabe_ito': OKABE_ITO_LIST,
        'mcm': list(dict.fromkeys(MCM_SCENARIO.values())),
        'journal_2': JOURNAL_2,
        'tol_bright': TOL_BRIGHT_LIST,
    }
    if name not in palettes:
        raise ValueError(f'未知调色板: {name}; 可选 {tuple(palettes)}')
    colors = palettes[name]
    if n is None:
        return colors.copy()
    if n > len(colors):
        raise ValueError(f'{name} 最多支持 {len(colors)} 个可区分类别，收到 {n}')
    return colors[:n]


if __name__ == '__main__':
    print('Science:', SCIENCE_LIST)
    print('Okabe-Ito:', OKABE_ITO_LIST)
