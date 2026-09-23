"""
Phase 4: Statistical Analysis & Visualization
===============================================
Read comparison results, run Wilcoxon rank-sum test, Friedman test,
generate comparison tables and figures.

Usage:
    python phase4_statistical_analysis.py --input outputs/
    python phase4_statistical_analysis.py --input outputs/phase4_benchmark_*.json
"""

import sys
import os
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats as sp_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False


# ======================================================================
# Statistical Tests
# ======================================================================

def wilcoxon_test(algo1_fits, algo2_fits, alpha=0.05):
    """Wilcoxon rank-sum test (two independent samples)"""
    a = np.array(algo1_fits)
    b = np.array(algo2_fits)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 5 or len(b) < 5:
        return {'statistic': None, 'p_value': None, 'significant': False,
                'effect_size': None, 'winner': 'N/A'}

    try:
        stat, p = sp_stats.ranksums(a, b)
    except Exception:
        return {'statistic': None, 'p_value': None, 'significant': False,
                'effect_size': None, 'winner': 'N/A'}

    pooled_std = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    d = abs(np.mean(a) - np.mean(b)) / (pooled_std + 1e-10)

    if p < alpha:
        winner = 'algo1' if np.median(a) < np.median(b) else 'algo2'
    else:
        winner = 'tie'

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': p < alpha,
        'effect_size': float(d),
        'winner': winner,
        'effect_label': 'large' if d > 0.8 else ('medium' if d > 0.5 else 'small'),
    }


def friedman_test(results_matrix):
    """Friedman test for multiple algorithms across multiple problems.

    Args:
        results_matrix: shape (n_problems, n_algorithms), each element = median fitness
    """
    n_problems, n_algos = results_matrix.shape
    if n_problems < 2 or n_algos < 2:
        return {'statistic': None, 'p_value': None, 'significant': False,
                'rankings': None}

    rankings = np.zeros_like(results_matrix)
    for i in range(n_problems):
        order = np.argsort(results_matrix[i])
        for rank, idx in enumerate(order):
            rankings[i, idx] = rank + 1

    try:
        stat, p = sp_stats.friedmanchisquare(*[rankings[:, j] for j in range(n_algos)])
    except Exception:
        return {'statistic': None, 'p_value': None, 'significant': False,
                'rankings': rankings.mean(axis=0).tolist()}

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': p < 0.05,
        'mean_rankings': rankings.mean(axis=0).tolist(),
        'rankings_per_problem': rankings.tolist(),
    }


def nemenyi_post_hoc(rankings, alpha=0.05):
    """Nemenyi post-hoc test after significant Friedman."""
    n_problems, n_algos = rankings.shape
    mean_ranks = rankings.mean(axis=0)

    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(n_algos, 3.0)
    cd = q * np.sqrt(n_algos * (n_algos + 1) / (6 * n_problems))

    comparisons = {}
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            comparisons[f'{i}_vs_{j}'] = {
                'rank_diff': float(diff),
                'significant': diff > cd,
                'cd': float(cd),
            }

    return {'cd': float(cd), 'mean_rankings': mean_ranks.tolist(), 'comparisons': comparisons}


# ======================================================================
# Visualization
# ======================================================================

def plot_boxplot(results, algo_names, func_name, out_path):
    """Box plot of fitness distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data, labels = [], []

    for algo in algo_names:
        if algo in results:
            fits = [f for f in results[algo]['best_fits'] if np.isfinite(f)]
            if fits:
                data.append(np.log10(np.array(fits) + 1e-30))
                labels.append(algo)

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    ax.set_title(f'{func_name} -- Algorithm Comparison (log10)', fontsize=14)
    ax.set_ylabel('log10(fitness)', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence(convergences, algo_names, func_name, out_path):
    """Convergence curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(algo_names)))

    for algo, color in zip(algo_names, colors):
        if algo in convergences and convergences[algo]:
            conv = convergences[algo]
            y = np.array(conv)
            y = np.where(np.isfinite(y), y, np.nan)
            if len(y) > 100:
                step = max(1, len(y) // 200)
                y = y[::step]
            x = np.linspace(0, 1, len(y))
            ax.semilogy(x, np.abs(y) + 1e-30, label=algo, color=color, linewidth=1.5)

    ax.set_title(f'{func_name} -- Convergence Curves', fontsize=14)
    ax.set_xlabel('FES Ratio', fontsize=12)
    ax.set_ylabel('|fitness| (log)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rank_bar(mean_rankings, algo_names, out_path):
    """Friedman mean rank bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(algo_names)))

    sorted_pairs = sorted(zip(mean_rankings, algo_names, colors))
    ranks_sorted = [p[0] for p in sorted_pairs]
    names_sorted = [p[1] for p in sorted_pairs]
    colors_sorted = [p[2] for p in sorted_pairs]

    bars = ax.barh(range(len(names_sorted)), ranks_sorted, color=colors_sorted)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel('Mean Rank (lower is better)', fontsize=12)
    ax.set_title('Friedman Mean Rankings', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, ranks_sorted):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================================
# Report Generation
# ======================================================================

def generate_report(benchmark_results, engineering_results, output_dir, algo_names):
    """Generate Markdown comparison report."""
    report = []
    report.append("# PNO-GWO v4.0 Phase 4: Algorithm Comparison Report\n")
    report.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- Benchmark ---
    if benchmark_results:
        report.append("\n## 1. Benchmark Functions\n")
        report.append("\n### 1.1 Performance Summary\n")
        header = "| Function | Algorithm | Best | Median | Mean | Std | Success | Time(s) |"
        sep =    "|----------|-----------|------|--------|------|-----|---------|---------|"
        report.append(header)
        report.append(sep)

        for func_name, algos in benchmark_results.items():
            for algo_name in algo_names:
                if algo_name in algos:
                    s = algos[algo_name]['stats']
                    row = (f"| {func_name} | {algo_name} | "
                           f"{s['best']:.4e} | {s['median']:.4e} | "
                           f"{s['mean']:.4e} | {s['std']:.4e} | "
                           f"{s['success_count']}/{s.get('n_valid', 30)} | "
                           f"{s['mean_time']:.2f} |")
                    report.append(row)

        # Wilcoxon
        report.append("\n### 1.2 Wilcoxon Rank-Sum Test (PNO-GWO vs Others)\n")
        report.append("| Function | vs Algorithm | p-value | Sig | Effect | Winner |")
        report.append("|----------|-------------|---------|-----|--------|--------|")

        for func_name, algos in benchmark_results.items():
            if 'PNO-GWO-v4.0' not in algos:
                continue
            pno_fits = algos['PNO-GWO-v4.0']['best_fits']
            for algo_name in algo_names:
                if algo_name == 'PNO-GWO-v4.0' or algo_name not in algos:
                    continue
                test = wilcoxon_test(pno_fits, algos[algo_name]['best_fits'])
                if test['p_value'] is not None:
                    sig = 'Y' if test['significant'] else 'N'
                    winner_map = {'algo1': 'PNO-GWO', 'algo2': algo_name, 'tie': 'Tie'}
                    winner = winner_map.get(test['winner'], 'N/A')
                    report.append(
                        f"| {func_name} | {algo_name} | "
                        f"{test['p_value']:.4f} | {sig} | "
                        f"{test['effect_size']:.2f} ({test['effect_label']}) | {winner} |"
                    )

        # Friedman
        report.append("\n### 1.3 Friedman Test\n")
        func_names = list(benchmark_results.keys())
        n_algos = len(algo_names)
        if len(func_names) >= 2:
            matrix = np.zeros((len(func_names), n_algos))
            for i, fname in enumerate(func_names):
                for j, aname in enumerate(algo_names):
                    if aname in benchmark_results[fname]:
                        matrix[i, j] = benchmark_results[fname][aname]['stats']['median']
                    else:
                        matrix[i, j] = float('inf')

            fried = friedman_test(matrix)
            report.append(f"- Friedman chi2 = {fried['statistic']:.4f}, p = {fried['p_value']:.4f}")
            report.append(f"- Significant: {'Yes (p < 0.05)' if fried['significant'] else 'No (p >= 0.05)'}")
            report.append("\nMean Rankings:")
            for rank, name in sorted(zip(fried['mean_rankings'], algo_names)):
                report.append(f"  - {name}: {rank:.2f}")

    # --- Engineering ---
    if engineering_results:
        report.append("\n## 2. Engineering Problems\n")
        report.append("\n### 2.1 Performance Summary\n")
        header = "| Problem | Algorithm | Best | Median | Feasible | Gap(%) | Time(s) |"
        sep =    "|---------|-----------|------|--------|----------|--------|---------|"
        report.append(header)
        report.append(sep)

        for prob_name, algos in engineering_results.items():
            for algo_name in algo_names:
                if algo_name in algos:
                    s = algos[algo_name]['stats']
                    gap_str = f"{s['median_gap']:.2f}" if np.isfinite(s['median_gap']) else "inf"
                    row = (f"| {prob_name} | {algo_name} | "
                           f"{s['best']:.4e} | {s['median']:.4e} | "
                           f"{s['feasible_count']}/30 | {gap_str} | "
                           f"{s['mean_time']:.2f} |")
                    report.append(row)

    # --- Conclusions ---
    report.append("\n## 3. Key Findings\n")
    report.append("1. **PNO-GWO vs GWO**: Validates the value of graph-structure exploration")
    report.append("2. **PNO-GWO vs SHADE**: Comparison with SOTA adaptive DE")
    report.append("3. **Engineering applicability**: Constraint handling + feasibility rates")
    report.append("4. **Computational overhead**: Cost of graph construction\n")

    report_text = '\n'.join(report)
    report_path = output_dir / 'report_phase4_comparison.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport generated: {report_path}")
    return report_text


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 4 Statistical Analysis')
    parser.add_argument('--input', type=str, default=None, help='Result JSON file or directory')
    parser.add_argument('--auto', action='store_true', help='Auto-find latest results')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = None
    engineering_results = None

    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            bench_files = sorted(input_path.glob('phase4_benchmark_*.json'))
            eng_files = sorted(input_path.glob('phase4_engineering_*.json'))
            if bench_files:
                with open(bench_files[-1], 'r', encoding='utf-8') as f:
                    benchmark_results = json.load(f)
            if eng_files:
                with open(eng_files[-1], 'r', encoding='utf-8') as f:
                    engineering_results = json.load(f)
        elif input_path.is_file():
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Auto-detect type
            sample_key = list(data.keys())[0] if data else ''
            if sample_key in ['Sphere', 'Rastrigin', 'Rosenbrock', 'Ackley', 'Griewank']:
                benchmark_results = data
            else:
                engineering_results = data
    else:
        outputs = base_dir / 'outputs'
        if outputs.exists():
            bench_files = sorted(outputs.glob('phase4_benchmark_*.json'))
            eng_files = sorted(outputs.glob('phase4_engineering_*.json'))
            if bench_files:
                with open(bench_files[-1], 'r', encoding='utf-8') as f:
                    benchmark_results = json.load(f)
            if eng_files:
                with open(eng_files[-1], 'r', encoding='utf-8') as f:
                    engineering_results = json.load(f)

    if benchmark_results is None and engineering_results is None:
        print("No results found. Run phase4_comparison_runner.py first.")
        return

    # Collect algorithm names
    all_algos = set()
    if benchmark_results:
        for v in benchmark_results.values():
            all_algos.update(v.keys())
    if engineering_results:
        for v in engineering_results.values():
            all_algos.update(v.keys())
    all_algos = sorted(all_algos)

    # Generate figures
    if benchmark_results:
        print("\nGenerating benchmark figures...")
        for func_name, algos in benchmark_results.items():
            plot_boxplot(algos, all_algos, func_name, fig_dir / f'boxplot_{func_name}.png')
            convergences = {}
            for aname, data in algos.items():
                if 'median_convergence' in data:
                    convergences[aname] = data['median_convergence']
            if convergences:
                plot_convergence(convergences, all_algos, func_name,
                                 fig_dir / f'convergence_{func_name}.png')

        # Friedman rank bar
        func_names = list(benchmark_results.keys())
        matrix = np.zeros((len(func_names), len(all_algos)))
        for i, fname in enumerate(func_names):
            for j, aname in enumerate(all_algos):
                if aname in benchmark_results[fname]:
                    matrix[i, j] = benchmark_results[fname][aname]['stats']['median']
                else:
                    matrix[i, j] = float('inf')
        fried = friedman_test(matrix)
        if fried['mean_rankings'] is not None:
            plot_rank_bar(fried['mean_rankings'], all_algos, fig_dir / 'friedman_rankings.png')

    # Generate report
    generate_report(benchmark_results, engineering_results, out_dir, all_algos)

    print(f"\nFigures saved to: {fig_dir}")
    print("Analysis complete!")


if __name__ == '__main__':
    main()