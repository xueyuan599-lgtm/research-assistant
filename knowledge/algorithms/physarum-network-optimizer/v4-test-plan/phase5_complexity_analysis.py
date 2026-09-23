"""
Phase 5E 分析: 计算复杂度
=========================
读取 phase5_complexity.json 生成:
  - 时间复杂度的幂律拟合 T(N,D) = a·N^b·D^c
  - outputs/figures/complexity_time_vs_dim.png
  - outputs/figures/complexity_time_vs_pop.png
  - outputs/figures/complexity_component_pie.png
  - reports/report_phase5_complexity.md
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
REPORTS_DIR = BASE_DIR / 'reports'
RESULTS_FILE = OUTPUT_DIR / 'phase5_complexity.json'


def fit_power_law(data):
    """拟合 T = a * N^b * D^c，log-log 线性回归"""
    pts = []
    for key, d in data.items():
        if d.get('median_time') is None:
            continue
        pts.append((d['N'], d['D'], d['median_time']))
    if len(pts) < 3:
        return None, None, None
    X = np.array([[np.log(p[0]), np.log(p[1]), 1] for p in pts])
    y = np.array([np.log(p[2]) for p in pts])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b, c, loga = coef
    return np.exp(loga), b, c


def main():
    data = json.load(open(RESULTS_FILE, 'r', encoding='utf-8'))
    timing = data['timing']
    comp_times = data['component_times']
    comp_marginal = data['component_marginal']

    a, b, c = fit_power_law(timing)

    # 图1: 时间 vs 维度 (固定 N)
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in [30, 50, 100, 200, 500]:
        pts = sorted([(d['D'], d['median_time'])
                      for k, d in timing.items()
                      if d['N'] == N and d['median_time'] is not None])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker='o', label=f'N={N}')
    ax.set_xlabel('维度 D')
    ax.set_ylabel('单次运行时间 (s)')
    ax.set_title('运行时间 vs 维度 (固定 FES=10000)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'complexity_time_vs_dim.png', dpi=150)
    plt.close()

    # 图2: 时间 vs 种群大小 (固定 D)
    fig, ax = plt.subplots(figsize=(8, 5))
    for D in [10, 30, 100, 500]:
        pts = sorted([(d['N'], d['median_time'])
                      for k, d in timing.items()
                      if d['D'] == D and d['median_time'] is not None])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker='s', label=f'D={D}')
    ax.set_xlabel('种群大小 N')
    ax.set_ylabel('单次运行时间 (s)')
    ax.set_title('运行时间 vs 种群大小 (固定 FES=10000)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'complexity_time_vs_pop.png', dpi=150)
    plt.close()

    # 图3: 组件时间占比饼图（Strategy 的负边际=省时，并入党规）
    labels = list(comp_marginal.keys())
    comp_store = {}
    for l in labels:
        m = comp_marginal[l]['marginal_cost']
        # 策略池开启反而省时（负边际），记为正贡献归入 Other 描述
        comp_store[l] = m if m >= 0 else 0.0
    sizes = [comp_store[l] for l in labels]
    other = comp_times['Full'] - sum(sizes)
    total_labels = labels + ['Other(含策略池省时)']
    total_sizes = sizes + [max(other, 0.001)]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(total_sizes, labels=total_labels, autopct='%1.1f%%',
           startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(total_labels))))
    ax.set_title('组件时间占比 (相对 Full 配置)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'complexity_component_pie.png', dpi=150)
    plt.close()

    # 报告
    lines = []
    lines.append('# Phase 5E: 计算复杂度分析\n')
    lines.append('## 1. 实验设置\n')
    lines.append('- 时间测量: 固定 30 代，N ∈ {30,50,100,200,500} × D ∈ {10,30,50,100,200,500}')
    lines.append('- 组件占比: FES=20,000，N=100, D=30，通过移除组件的边际成本估计')
    lines.append('- 目标函数: Sphere\n')

    lines.append('## 2. 时间复杂度拟合\n')
    if a is not None:
        lines.append(f'拟合结果: **T(N,D) ≈ {a:.4f} × N^{{{b:.2f}}} × D^{{{c:.2f}}}**')
        lines.append('')
        lines.append('- 若指数 ≈0.5-1: 线性或亚线性');
    lines.append('')
    lines.append('### 原始测量数据\n')
    lines.append('| N | D | 中位数时间 (s) |')
    lines.append('|---|------|--------------|')
    for k in sorted(timing.keys(), key=lambda x: (timing[x]['N'], timing[x]['D'])):
        d = timing[k]
        t = d['median_time']
        lines.append(f'| {d["N"]} | {d["D"]} | {t if t else "err"} |')

    lines.append('\n## 3. 组件时间占比\n')
    lines.append(f'- Full 配置总时间: {comp_times["Full"]:.3f}s (FES=20,000, N=100, D=30)')
    lines.append('\n| 组件 | 边际成本 (s) | 占比 | 说明 |')
    lines.append('|------|-------------|------|------|')
    for comp in ['SHCA', 'Archive', 'Strategy', 'Cauchy', 'NM']:
        m = comp_marginal[comp]
        pct_full = comp_times['Full']
        raw = comp_times[f'w/o_{comp}'] - comp_times['Full']
        note = ''
        if raw < 0:
            note = '移除后更慢（该组件省时）'
        lines.append(f'| {comp} | {m["marginal_cost"]:.4f} | '
                     f'{m["marginal_cost"]/pct_full*100:.1f}% | {note} |')
    other_cost = sum(comp_marginal[c]['marginal_cost']
                     for c in comp_marginal if comp_marginal[c]['marginal_cost'] >= 0)
    other = comp_times['Full'] - other_cost
    lines.append(f'| 其他(图构建/评估/更新/策略池) | {max(other,0):.4f} | '
                 f'{max(other,0)/comp_times["Full"]*100:.1f}% |'
                 f' 含策略池省时 {abs(comp_marginal["Strategy"]["marginal_cost"]):.4f}s |')

    lines.append('\n![时间vs维度](figures/complexity_time_vs_dim.png)')
    lines.append('\n![时间vs种群](figures/complexity_time_vs_pop.png)')
    lines.append('\n![组件占比](figures/complexity_component_pie.png)')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'report_phase5_complexity.md').write_text('\n'.join(lines),
                                                             encoding='utf-8')
    print(f"报告已生成: {REPORTS_DIR / 'report_phase5_complexity.md'}")
    if a is not None:
        print(f"拟合: T ≈ {a:.4f} × N^{b:.2f} × D^{c:.2f}")


if __name__ == '__main__':
    main()