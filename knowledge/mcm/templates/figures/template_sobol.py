# -*- coding: utf-8 -*-
"""Sobol sensitivity indices as grouped first-/total-order bars with CIs."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def make_sobol(ax, params, s1, st, s1_conf=None, st_conf=None,
               s1_label='一阶指数 $S_1$', st_label='总阶指数 $S_T$',
               ylabel='Sobol 敏感度指数 (–)', sort=True, annotate_st=True,
               legend_loc='upper right'):
    """Draw SALib-style grouped bars for first- and total-order indices.

    ``s1_conf``/``st_conf`` 是 95% 置信区间半宽，传入即绘制误差棒。
    """
    from export_figure import style_axis
    from palette import SCIENCE

    params = [str(p) for p in params]
    s1 = np.asarray(s1, dtype=float)
    st = np.asarray(st, dtype=float)
    if not params or s1.shape != (len(params),) or st.shape != (len(params),):
        raise ValueError('params、s1、st 必须为等长非空序列')
    if s1_conf is not None:
        s1_conf = np.asarray(s1_conf, dtype=float)
        if s1_conf.shape != (len(params),):
            raise ValueError('s1_conf 必须与 params 等长')
    if st_conf is not None:
        st_conf = np.asarray(st_conf, dtype=float)
        if st_conf.shape != (len(params),):
            raise ValueError('st_conf 必须与 params 等长')
    values = [s1, st] + ([s1_conf] if s1_conf is not None else []) \
        + ([st_conf] if st_conf is not None else [])
    if not np.all(np.isfinite(np.concatenate(values))):
        raise ValueError('Sobol 指数输入必须为有限数值')
    if np.any(s1 > st + 1e-9):
        raise ValueError('一阶指数 S1 不应大于总阶指数 ST')

    if sort:
        order = np.argsort(st)[::-1]
        params = [params[index] for index in order]
        s1, st = s1[order], st[order]
        if s1_conf is not None:
            s1_conf = s1_conf[order]
        if st_conf is not None:
            st_conf = st_conf[order]

    c_s1, c_st = SCIENCE['蓝'], SCIENCE['橙']
    x = np.arange(len(params))
    error_kw = {'elinewidth': 0.7, 'capthick': 0.7, 'capsize': 2.2}
    ax.bar(x - 0.19, s1, width=0.36, color=c_s1, edgecolor='#333333',
           linewidth=0.4, yerr=s1_conf, error_kw=error_kw, zorder=3,
           label=s1_label)
    ax.bar(x + 0.19, st, width=0.36, color=c_st, edgecolor='#333333',
           linewidth=0.4, yerr=st_conf, error_kw=error_kw, zorder=3,
           label=st_label)

    lower = min(0.0, float((s1 - (s1_conf if s1_conf is not None else 0)).min()))
    upper = float((st + (st_conf if st_conf is not None else 0)).max())
    span = upper - lower
    if annotate_st:
        for index, value in enumerate(st):
            top = value + (st_conf[index] if st_conf is not None else 0.0)
            ax.text(x[index] + 0.19, top + span * 0.03, f'{value:.2f}',
                    ha='center', va='bottom', fontsize=5.8, color='#444444',
                    zorder=4)
    ax.set_xticks(x, params)
    if max(len(p) for p in params) > 4:
        ax.tick_params(axis='x', rotation=20)
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation_mode('anchor')
    ax.set_ylim(lower - span * 0.06, upper + span * 0.2)
    ax.set_ylabel(ylabel)
    style_axis(ax, grid='y', zero_line=True)
    ax.legend(loc=legend_loc, handlelength=1.2)
    return ax


def demo(language='zh', stem=None):
    from export_figure import apply_mcm_style, new_figure, save_figure
    apply_mcm_style()
    fig, ax = new_figure(aspect=0.8)
    zh = language == 'zh'
    params = (['降水量', '气温', '施肥量', '播种密度', '土壤pH'] if zh
              else ['Precip', 'Temp', 'Fertilizer', 'Density', 'Soil pH'])
    s1 = [0.42, 0.18, 0.09, 0.31, 0.05]
    st = [0.51, 0.24, 0.12, 0.40, 0.07]
    s1_conf = [0.035, 0.030, 0.020, 0.032, 0.015]
    st_conf = [0.050, 0.040, 0.028, 0.045, 0.020]
    make_sobol(ax, params, s1, st, s1_conf, st_conf,
               s1_label='一阶指数 $S_1$' if zh else 'First-order $S_1$',
               st_label='总阶指数 $S_T$' if zh else 'Total-order $S_T$',
               ylabel='Sobol 指数 (–)' if zh else 'Sobol index (–)')
    ax.set_title('产量对投入要素的 Sobol 敏感度' if zh
                 else 'Sobol sensitivity of crop yield')
    if not zh:
        ax.set_xlabel('Input parameter')
    return save_figure(fig, os.path.join(THIS_DIR, 'samples', stem or
                       ('sobol_demo' if zh else 'sobol_demo_en')))


if __name__ == '__main__':
    demo()
