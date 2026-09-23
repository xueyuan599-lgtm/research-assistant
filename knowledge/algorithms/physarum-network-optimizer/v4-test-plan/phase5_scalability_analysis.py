"""
Phase 5C 分析: 高维可扩展性
===========================
读取 phase5_scalability.json 生成:
  - outputs/figures/scalability_fitness.png
  - outputs/figures/scalability_time.png
  - reports/report_phase5_scalability.md
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
RESULTS_FILE = OUTPUT_DIR / 'phase5_scalability.json'

FUNCS = ['Sphere', 'Rastrigin', 'Rosenbrock', 'Ackley', 'Griewank']
ALGOS = ['PNO-GWO-v4.0', 'GWO', 'PSO', 'DE', 'CMA-ES']
DIMS = [100, 300, 500]


def main():
    data = json.load(open(RESULTS_FILE, 'r', encoding='utf-8'))

    # 重组: results[dim][func][algo] = stats
    results = {}
    for key, res in data.items():
        d = res['dim']
        fn = res['function']
        algo = res['algorithm']
        results.setdefault(d, {}).setdefault(fn, {})[algo] = res['stats']

    # 适应度 vs 维度 (每个算法在所有函数上的几何平均中位数)
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {'PNO-GWO-v4.0': 'o', 'GWO': 's', 'PSO': '^', 'DE': 'v', 'CMA-ES': 'D'}
    for algo in ALGOS:
        medians = []
        for d in DIMS:
            vals = []
            for fn in FUNCS:
                if algo in results[d][fn]:
                    m = results[d][fn][algo]['median']
                    # 归一化: 各函数量级不同，取 log10
                    vals.append(np.log10(max(m, 1e-300)))
            medians.append(np.mean(vals))
        ax.plot(DIMS, medians, marker=markers[algo], label=algo, linewidth=2)
    ax.set_xlabel('维度')
    ax.set_ylabel('平均 log10(中位数适应度)')
    ax.set_title('高维性能衰减 (越低越好)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'scalability_fitness.png', dpi=150)
    plt.close()

    # 运行时间 vs 维度
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ALGOS:
        times = []
        for d in DIMS:
            ts = [results[d][fn][algo]['mean_time'] for fn in FUNCS
                  if algo in results[d][fn]]
            times.append(np.mean(ts))
        ax.plot(DIMS, times, marker=markers[algo], label=algo, linewidth=2)
    ax.set_xlabel('维度')
    ax.set_ylabel('平均运行时间 (s)')
    ax.set_title('计算成本 vs 维度')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scalability_time.png', dpi=150)
    plt.close()

    # 生成报告
    lines = []
    lines.append('# Phase 5C: 高维可扩展性报告\n')
    lines.append('## 1. 实验设置\n')
    lines.append('- 函数: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank')
    lines.append('- 维度: 100D (100K FES), 300D (300K FES), 500D (500K FES)')
    lines.append('- 运行次数: 30')
    lines.append('- 对比: PNO-GWO-v4.0, GWO, PSO, DE, CMA-ES\n')

    for d in DIMS:
        lines.append(f'## {d}D 结果\n')
        lines.append('| 函数 | 算法 | 中位数 | 均值 | 时间(s) |')
        lines.append('|------|------|--------|------|--------|')
        for fn in FUNCS:
            for algo in ALGOS:
                s = results[d][fn].get(algo)
                if s is None:
                    continue
                lines.append(f'| {fn} | {algo} | {s["median"]:.3e} | '
                             f'{s["mean"]:.3e} | {s["mean_time"]:.2f} |')

    # 维度增长趋势
    lines.append('\n## 性能衰减趋势\n')
    lines.append('| 算法 | 100D→300D (log10变化) | 300D→500D (log10变化) |')
    lines.append('|------|----------------------|----------------------|')
    for algo in ALGOS:
        log10_med = []
        for d in DIMS:
            vals = [np.log10(max(results[d][fn][algo]['median'], 1e-300))
                    for fn in FUNCS if algo in results[d][fn]]
            log10_med.append(np.mean(vals))
        lines.append(f'| {algo} | {log10_med[1]-log10_med[0]:+.2f} | '
                     f'{log10_med[2]-log10_med[1]:+.2f} |')

    lines.append('\n![适应度](figures/scalability_fitness.png)')
    lines.append('\n![时间](figures/scalability_time.png)')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'report_phase5_scalability.md').write_text('\n'.join(lines),
                                                              encoding='utf-8')
    print(f"报告已生成: {REPORTS_DIR / 'report_phase5_scalability.md'}")


if __name__ == '__main__':
    main()