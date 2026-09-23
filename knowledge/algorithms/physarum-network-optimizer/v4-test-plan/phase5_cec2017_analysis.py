"""
Phase 5B 分析: CEC 2017 对比统计分析与图表
==========================================
读取 phase5_cec2017.json，计算:
  - 每个函数的 10 次运行统计（已含）
  - Friedman 检验 + 平均排名
  - Wilcoxon 逐函数两两对比（PNO vs 其他）
  - 表格 + 排名图

生成:
  - outputs/figures/cec_friedman.png
  - reports/report_phase5_cec2017.md
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
REPORTS_DIR = BASE_DIR / 'reports'
RESULTS_FILE = OUTPUT_DIR / 'phase5_cec2017.json'

ALGOS = ['PNO-GWO-v4.0', 'GWO', 'PSO', 'DE', 'SHADE', 'CMA-ES',
         'LSHADE-cnEPSO', 'jSO', 'EA4eig', 'CMODE']


def main():
    data = json.load(open(RESULTS_FILE, 'r', encoding='utf-8'))

    # 重组: results[func][algo] = stats
    funcs = {}
    for key, res in data.items():
        fname = res['function']
        algo = res['algorithm']
        funcs.setdefault(fname, {})[algo] = res

    func_names = sorted(funcs.keys())

    # 每个函数每个算法的中位数误差
    median_matrix = np.zeros((len(func_names), len(ALGOS)))
    for i, fn in enumerate(func_names):
        for j, algo in enumerate(ALGOS):
            if algo in funcs[fn]:
                median_matrix[i, j] = funcs[fn][algo]['stats']['median']
            else:
                median_matrix[i, j] = np.inf

    # Friedman 检验: 用中位数矩阵做排名（每函数内部排名）
    ranks = np.zeros_like(median_matrix)
    for i in range(median_matrix.shape[0]):
        ranks[i] = sps.rankdata(median_matrix[i])  # 1=best
    mean_ranks = ranks.mean(axis=0)

    # Friedman 检验: 每函数为块 × 每算法为处理，观测值用 10 runs 均值
    block_means = np.zeros((len(func_names), len(ALGOS)))
    for i, fn in enumerate(func_names):
        for j, algo in enumerate(ALGOS):
            if algo in funcs[fn]:
                block_means[i, j] = funcs[fn][algo]['stats']['mean']
            else:
                block_means[i, j] = np.inf
    friedman_stat, friedman_p = sps.friedmanchisquare(
        *[block_means[:, j] for j in range(len(ALGOS))]
    )

    # 逐配置原始误差数组（供 Wilcoxon 逐函数使用）
    all_data = {}
    for fn in func_names:
        all_data[fn] = {}
        for algo in ALGOS:
            if algo in funcs[fn]:
                all_data[fn][algo] = np.array(funcs[fn][algo]['errors'])

    # Wilcoxon: PNO vs 每个其他算法（逐函数）
    wilcoxon_table = {}
    for algo in ALGOS:
        if algo == 'PNO-GWO-v4.0':
            continue
        wins, losses, ties = 0, 0, 0
        sig_wins = 0
        for fn in func_names:
            pno = np.array(all_data[fn]['PNO-GWO-v4.0'])
            other = np.array(all_data[fn][algo])
            try:
                if np.all(pno == other):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = sps.wilcoxon(pno, other)
            except Exception:
                p = 1.0
            if p < 0.05:
                if pno.mean() < other.mean():
                    wins += 1
                    sig_wins += 1
                else:
                    losses += 1
            else:
                ties += 1
        wilcoxon_table[algo] = {'wins': wins, 'losses': losses, 'ties': ties,
                                'pct_wins': wins / len(func_names)}

    # 生成排名图
    fig, ax = plt.subplots(figsize=(9, 4))
    order = np.argsort(mean_ranks)
    colors = ['#e63946' if ALGOS[i] == 'PNO-GWO-v4.0' else '#457b9d' for i in order]
    ax.barh(range(len(ALGOS)), mean_ranks[order], color=colors)
    ax.set_yticks(range(len(ALGOS)))
    ax.set_yticklabels([ALGOS[i] for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('平均排名 (1=最好)')
    ax.set_title(f'CEC 2017 Friedman 平均排名 (D={10}, {len(func_names)} funcs)', fontsize=11)
    # 标注数值
    for i, idx in enumerate(order):
        ax.text(mean_ranks[idx] + 0.05, i, f'{mean_ranks[idx]:.2f}',
                va='center', fontsize=9)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'cec_friedman.png', dpi=150)
    plt.close()

    # 生成报告
    lines = []
    lines.append('# Phase 5B: CEC 2017 测试集对比实验报告\n')
    lines.append('> 基于 opfunu 库的 CEC 2017 单目标实参优化竞赛套件（无约束版）\n')
    lines.append('\n## 1. 实验设置\n')
    lines.append('- 测试集: CEC 2017 (opfunu 1.0.4: F1, F3-F29 共 %d 个函数;'
                 ' 官方 F2 被排除, F30 该库未提供)' % len(func_names))
    lines.append('- 维度: 10D')
    lines.append('- Max FES: 30,000（社区标准为 100,000；本实验全算法相同预算，排名公平）')
    lines.append('- 独立运行: 10 次')
    lines.append('- 对比算法: PNO-GWO-v4.0, GWO, PSO, DE, SHADE, CMA-ES, '
                 'LSHADE-cnEPSO, jSO, EA4eig, CMODE')
    lines.append('- 指标: 误差 (f(x) - f*)，其中 f* 为官方 bias')
    lines.append('\n⚠️ **注意**: 官方 CEC 2017 约束优化版 (C01-C28) 因依赖官方 MATLAB '
                 '数据文件难以公开复现，此处使用社区标准无约束套件作为基准；'
                 '约束能力由 5A 工程问题（Welded Beam、Pressure Vessel、Spring）覆盖。\n')

    lines.append('\n## 2. Friedman 排名\n')
    lines.append(f'- Friedman 统计量 = {friedman_stat:.2f}, p = {friedman_p:.4g}')
    lines.append(f'- 平均运行时间 (PNO): {np.mean([funcs[fn]["PNO-GWO-v4.0"]["stats"]["mean_time"] for fn in func_names]):.2f}s')
    lines.append('\n| 排名 | 算法 | 平均排名 |')
    lines.append('|------|------|---------|')
    for r, idx in enumerate(order, 1):
        marker = ' **★**' if ALGOS[idx] == 'PNO-GWO-v4.0' else ''
        lines.append(f'| {r} | {ALGOS[idx]}{marker} | {mean_ranks[idx]:.2f} |')

    lines.append('\n## 3. 逐函数 Wilcoxon (PNO vs 其他)\n')
    lines.append('| 算法 | 显著胜 | 显著负 | 平局 | 胜率 |')
    lines.append('|------|-------|-------|------|------|')
    for algo, w in wilcoxon_table.items():
        lines.append(f'| {algo} | {w["wins"]} | {w["losses"]} | '
                     f'{w["ties"]} | {w["pct_wins"]:.0%} |')

    lines.append('\n## 4. 逐函数中位数误差表\n')
    lines.append('| 函数 | ' + ' | '.join(ALGOS) + ' |')
    lines.append('|------|' + '------|' * len(ALGOS))
    for i, fn in enumerate(func_names):
        row = [f'{median_matrix[i, j]:.2e}' if np.isfinite(median_matrix[i, j]) else 'inf'
               for j in range(len(ALGOS))]
        # 标记最优
        best_j = int(np.argmin(median_matrix[i]))
        row[best_j] = f'**{row[best_j]}**'
        lines.append(f'| {fn} | ' + ' | '.join(row) + ' |')

    lines.append('\n## 5. 图\n')
    lines.append('![Friedman 排名](figures/cec_friedman.png)')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'report_phase5_cec2017.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"报告已生成: {REPORTS_DIR / 'report_phase5_cec2017.md'}")
    print(f"Friedman p={friedman_p:.4g}, 平均排名: "
          f"{[(ALGOS[i], round(float(mean_ranks[i]),2)) for i in order]}")
    return mean_ranks, wilcoxon_table


if __name__ == '__main__':
    main()