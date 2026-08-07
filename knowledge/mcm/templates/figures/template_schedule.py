# -*- coding: utf-8 -*-
"""Gantt schedule with groups, completion, milestones, and critical tasks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def make_gantt(ax, tasks, starts, durations, groups=None, completion=None,
               critical=None, milestones=None, xlabel='时间（天）'):
    """Draw a compact scheduling chart from precomputed task timing."""
    try:
        from .export_figure import style_axis
        from .palette import get_palette
    except ImportError:
        from export_figure import style_axis
        from palette import get_palette
    tasks = list(tasks); starts = np.asarray(starts, dtype=float); durations = np.asarray(durations, dtype=float)
    count = len(tasks)
    if count == 0 or starts.shape != (count,) or durations.shape != (count,):
        raise ValueError('tasks、starts、durations 必须等长且非空')
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError('开始时间必须有限且工期必须为正')
    groups = list(groups) if groups is not None else ['任务'] * count
    if len(groups) != count:
        raise ValueError('groups 长度必须与 tasks 一致')
    unique_groups = list(dict.fromkeys(groups)); colors = get_palette('science', len(unique_groups))
    color_map = dict(zip(unique_groups, colors))
    completion = np.ones(count) if completion is None else np.asarray(completion, dtype=float)
    if completion.shape != (count,) or np.any((completion < 0) | (completion > 1)):
        raise ValueError('completion 必须在 0 到 1 之间')
    critical = np.zeros(count, dtype=bool) if critical is None else np.asarray(critical, dtype=bool)
    if critical.shape != (count,):
        raise ValueError('critical 长度必须与 tasks 一致')
    y = np.arange(count)[::-1]
    for index, position in enumerate(y):
        color = color_map[groups[index]]
        ax.barh(position, durations[index], left=starts[index], height=0.58,
                color=color, alpha=0.25, edgecolor='#333333' if critical[index] else 'white',
                linewidth=1 if critical[index] else 0.4)
        ax.barh(position, durations[index] * completion[index], left=starts[index],
                height=0.58, color=color, edgecolor='none')
    if milestones:
        for label, time in milestones.items():
            ax.scatter(time, count - 0.5, marker='D', s=22, color='#222222', clip_on=False)
            ax.annotate(label, (time, count - 0.5), xytext=(0, 5), textcoords='offset points', ha='center')
    ax.set_yticks(y, tasks)
    ax.set_xlabel(xlabel)
    style_axis(ax, grid='x')
    for group, color in color_map.items():
        ax.plot([], [], color=color, linewidth=5, label=group)
    ax.legend(ncol=min(3, len(unique_groups)))
    ax.set_ylim(-0.7, count - 0.15)
    return ax


def demo(language='zh', stem=None):
    try:
        from .export_figure import apply_mcm_style, new_figure, save_figure
    except ImportError:
        from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style(); zh = language == 'zh'
    tasks = ['需求分析', '数据清洗', '模型求解', '稳健性检验', '论文撰写'] if zh else ['Scoping', 'Data cleaning', 'Modeling', 'Validation', 'Writing']
    groups = ['准备', '准备', '建模', '建模', '交付'] if zh else ['Prepare', 'Prepare', 'Model', 'Model', 'Deliver']
    fig, ax = new_figure(width='double', aspect=0.35)
    make_gantt(ax, tasks, [0, 2, 5, 11, 14], [3, 5, 8, 5, 6], groups=groups,
               completion=[1, 1, .85, .5, .25], critical=[0, 1, 1, 1, 1],
               milestones={'中期检查' if zh else 'Review': 13},
               xlabel='时间（天）' if zh else 'Time (days)')
    output = THIS_DIR / 'samples' / (stem or ('schedule_demo' if zh else 'schedule_demo_en'))
    return save_figure(fig, output)


if __name__ == '__main__':
    demo()
