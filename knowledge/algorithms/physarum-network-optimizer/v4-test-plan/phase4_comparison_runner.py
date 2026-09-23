"""
Phase 4: PNO-GWO v4.0 算法对比实验
====================================
在基准函数和工程问题上对比 PNO-GWO v4.0 与 5 种主流优化算法。
30 次独立运行，收集统计指标。

用法:
    python phase4_comparison_runner.py                  # 全部实验
    python phase4_comparison_runner.py --benchmark-only  # 仅基准函数
    python phase4_comparison_runner.py --engineering-only # 仅工程问题
    python phase4_comparison_runner.py --runs 5          # 快速测试（5次）
"""

import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np

# 确保能导入同级模块
sys.path.insert(0, str(Path(__file__).parent))

from algorithms import GWO, PSO, DE, SHADE, CMAES
from constraint_handler import get_all_engineering_problems

# 导入 PNO-GWO v4.0
sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithm_v4 import PhysarumNetworkOptimizer


# ======================================================================
# 基准函数定义
# ======================================================================

def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley(x):
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
            + 20 + np.e)

def griewank(x):
    sum_sq = np.sum(x ** 2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1


BENCHMARK_FUNCTIONS = {
    'Sphere':     (sphere,     (-100, 100),   0.0),
    'Rastrigin':  (rastrigin,  (-5.12, 5.12), 0.0),
    'Rosenbrock': (rosenbrock, (-30, 30),     0.0),
    'Ackley':     (ackley,     (-32, 32),     0.0),
    'Griewank':   (griewank,   (-600, 600),   0.0),
}


# ======================================================================
# 算法工厂
# ======================================================================

def create_algorithms(n_dim, bounds, max_fes):
    """创建所有对比算法实例，返回字典"""
    algos = {}

    # PNO-GWO v4.0
    pno = PhysarumNetworkOptimizer(
        n_dim=n_dim, bounds=bounds, max_fes=max_fes,
        use_shca=True, use_cgpsr=False, use_cas=True,
        use_cauchy=True, use_nm=True, use_twophase=True
    )
    algos['PNO-GWO-v4.0'] = pno

    # GWO
    algos['GWO'] = GWO(n_pop=min(18 * n_dim, 200))

    # PSO
    algos['PSO'] = PSO(n_pop=min(18 * n_dim, 200))

    # DE
    algos['DE'] = DE(n_pop=max(10 * n_dim, 50), F=0.5, CR=0.9)

    # SHADE
    algos['SHADE'] = SHADE(n_pop=max(10 * n_dim, 50))

    # CMA-ES
    algos['CMA-ES'] = CMAES(sigma0=0.3)

    return algos


# ======================================================================
# 基准函数实验
# ======================================================================

def run_benchmark_comparison(n_dim=30, max_fes=30000, n_runs=30, verbose=False):
    """在基准函数上运行所有算法对比"""
    results = {}

    for func_name, (func, bounds, optimum) in BENCHMARK_FUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"Benchmark: {func_name} ({n_dim}D) | Optimum: {optimum}")
        print(f"{'='*60}")
        results[func_name] = {}

        algos = create_algorithms(n_dim, bounds, max_fes)

        for algo_name, algo in algos.items():
            print(f"\n  Algorithm: {algo_name}")
            run_results = {
                'best_fits': [],
                'convergences': [],
                'times': [],
            }

            for run in range(n_runs):
                np.random.seed(42 + run)
                try:
                    t0 = time.time()
                    best_x, best_fit, conv = algo.optimize(
                        func, n_dim=n_dim, bounds=bounds, max_fes=max_fes,
                        verbose=False
                    )
                    elapsed = time.time() - t0

                    run_results['best_fits'].append(float(best_fit))
                    run_results['convergences'].append(conv)
                    run_results['times'].append(elapsed)

                    if verbose:
                        print(f"    Run {run+1:2d}: best={best_fit:.6e}  time={elapsed:.2f}s")
                except Exception as e:
                    print(f"    Run {run+1:2d}: ERROR - {e}")
                    traceback.print_exc()
                    run_results['best_fits'].append(float('inf'))
                    run_results['convergences'].append([])
                    run_results['times'].append(0.0)

            # 统计汇总
            fits = np.array(run_results['best_fits'])
            valid = fits[np.isfinite(fits)]
            if len(valid) > 0:
                stats = {
                    'best': float(np.min(valid)),
                    'median': float(np.median(valid)),
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid)),
                    'worst': float(np.max(valid)),
                    'success_count': int(np.sum(np.abs(valid - optimum) < 1e-4)),
                    'success_rate': float(np.sum(np.abs(valid - optimum) < 1e-4) / n_runs),
                    'n_valid': int(len(valid)),
                    'mean_time': float(np.mean(run_results['times'])),
                }
            else:
                stats = {'best': float('inf'), 'median': float('inf'),
                         'mean': float('inf'), 'std': float('inf'),
                         'worst': float('inf'), 'success_count': 0,
                         'success_rate': 0.0, 'n_valid': 0, 'mean_time': 0.0}

            results[func_name][algo_name] = {
                'stats': stats,
                'best_fits': run_results['best_fits'],
                'times': run_results['times'],
                'median_convergence': _get_median_convergence(
                    run_results['best_fits'], run_results['convergences']
                ),
            }

            print(f"    {algo_name}: median={stats['median']:.6e}  "
                  f"std={stats['std']:.6e}  "
                  f"success={stats['success_count']}/{n_runs}  "
                  f"time={stats['mean_time']:.2f}s")

    return results


def _get_median_convergence(best_fits, convergences):
    """获取中位数对应的收敛曲线"""
    if not convergences or not best_fits:
        return []
    fits = np.array(best_fits)
    valid_mask = np.isfinite(fits)
    if not valid_mask.any():
        return []
    median_idx = np.argsort(fits[valid_mask])[len(fits[valid_mask]) // 2]
    valid_indices = np.where(valid_mask)[0]
    return convergences[valid_indices[median_idx]]


# ======================================================================
# 工程问题实验
# ======================================================================

def run_engineering_comparison(max_fes=30000, n_runs=30, verbose=False):
    """在工程问题上运行所有算法对比"""
    problems = get_all_engineering_problems()
    results = {}

    for prob_name, prob in problems.items():
        print(f"\n{'='*60}")
        print(f"Engineering: {prob_name} ({prob.n_dim}D, {len(prob.constraints)} constraints)")
        print(f"Known optimum: {prob.known_optimum}")
        print(f"{'='*60}")
        results[prob_name] = {}

        # 包装目标函数（含约束惩罚）
        def make_obj(problem):
            def obj_func(x):
                obj_val, violation = problem.evaluate(x)
                if violation > 0:
                    return obj_val + 1e6 * violation
                return obj_val
            return obj_func

        obj_func = make_obj(prob)
        bounds = (np.array([b[0] for b in prob.bounds]),
                  np.array([b[1] for b in prob.bounds]))

        algos = create_algorithms(prob.n_dim, bounds, max_fes)

        for algo_name, algo in algos.items():
            print(f"\n  Algorithm: {algo_name}")
            run_results = {
                'best_fits': [],
                'feasible_counts': [],
                'gaps': [],
                'times': [],
            }

            for run in range(n_runs):
                np.random.seed(42 + run)
                try:
                    t0 = time.time()
                    best_x, best_fit, conv = algo.optimize(
                        obj_func, n_dim=prob.n_dim, bounds=bounds,
                        max_fes=max_fes, verbose=False
                    )
                    elapsed = time.time() - t0

                    _, violation = prob.evaluate(best_x)
                    is_feasible = violation <= 1e-6

                    if prob.known_optimum is not None and is_feasible:
                        gap = (best_fit - prob.known_optimum) / abs(prob.known_optimum) * 100
                    else:
                        gap = float('inf')

                    run_results['best_fits'].append(float(best_fit))
                    run_results['feasible_counts'].append(1 if is_feasible else 0)
                    run_results['gaps'].append(float(gap))
                    run_results['times'].append(elapsed)

                    if verbose:
                        feas_str = "Y" if is_feasible else "N"
                        print(f"    Run {run+1:2d}: {feas_str} best={best_fit:.6e} gap={gap:.2f}%")
                except Exception as e:
                    print(f"    Run {run+1:2d}: ERROR - {e}")
                    traceback.print_exc()
                    run_results['best_fits'].append(float('inf'))
                    run_results['feasible_counts'].append(0)
                    run_results['gaps'].append(float('inf'))
                    run_results['times'].append(0.0)

            fits = np.array(run_results['best_fits'])
            feas = np.array(run_results['feasible_counts'])
            gaps = np.array(run_results['gaps'])
            valid_gaps = gaps[np.isfinite(gaps) & (feas == 1)]

            stats = {
                'best': float(np.min(fits[np.isfinite(fits)])) if np.any(np.isfinite(fits)) else float('inf'),
                'median': float(np.median(fits[np.isfinite(fits)])) if np.any(np.isfinite(fits)) else float('inf'),
                'mean': float(np.mean(fits[np.isfinite(fits)])) if np.any(np.isfinite(fits)) else float('inf'),
                'std': float(np.std(fits[np.isfinite(fits)])) if np.any(np.isfinite(fits)) else float('inf'),
                'feasible_count': int(feas.sum()),
                'feasible_rate': float(feas.mean()),
                'mean_gap': float(np.mean(valid_gaps)) if len(valid_gaps) > 0 else float('inf'),
                'median_gap': float(np.median(valid_gaps)) if len(valid_gaps) > 0 else float('inf'),
                'mean_time': float(np.mean(run_results['times'])),
                'known_optimum': prob.known_optimum,
            }

            results[prob_name][algo_name] = {
                'stats': stats,
                'best_fits': run_results['best_fits'],
                'feasible_counts': run_results['feasible_counts'],
                'gaps': run_results['gaps'],
                'times': run_results['times'],
            }

            print(f"    {algo_name}: feasible={stats['feasible_count']}/{n_runs}  "
                  f"median_gap={stats['median_gap']:.2f}%  "
                  f"median={stats['median']:.6e}")

    return results


# ======================================================================
# 主入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 4: PNO-GWO v4.0 Algorithm Comparison')
    parser.add_argument('--runs', type=int, default=30, help='Runs per experiment')
    parser.add_argument('--dim', type=int, default=30, help='Benchmark dimension')
    parser.add_argument('--max-fes', type=int, default=30000, help='Max function evaluations')
    parser.add_argument('--benchmark-only', action='store_true', help='Benchmark functions only')
    parser.add_argument('--engineering-only', action='store_true', help='Engineering problems only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"PNO-GWO v4.0 Phase 4 Comparison Experiment")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Params: dim={args.dim}, max_fes={args.max_fes}, runs={args.runs}")
    print(f"Output: {out_dir}")

    all_results = {}

    # Benchmark
    if not args.engineering_only:
        print("\n" + "=" * 70)
        print("PART 1: Benchmark Functions")
        print("=" * 70)
        benchmark_results = run_benchmark_comparison(
            n_dim=args.dim, max_fes=args.max_fes, n_runs=args.runs,
            verbose=args.verbose
        )
        all_results['benchmark'] = benchmark_results

        bench_file = out_dir / f'phase4_benchmark_{timestamp}.json'
        bench_save = {}
        for fname, algos in benchmark_results.items():
            bench_save[fname] = {}
            for aname, data in algos.items():
                bench_save[fname][aname] = {
                    'stats': data['stats'],
                    'best_fits': data['best_fits'],
                    'times': data['times'],
                }
        with open(bench_file, 'w', encoding='utf-8') as f:
            json.dump(bench_save, f, indent=2, ensure_ascii=False)
        print(f"\nBenchmark results saved: {bench_file}")

    # Engineering
    if not args.benchmark_only:
        print("\n" + "=" * 70)
        print("PART 2: Engineering Problems")
        print("=" * 70)
        eng_results = run_engineering_comparison(
            max_fes=args.max_fes, n_runs=args.runs, verbose=args.verbose
        )
        all_results['engineering'] = eng_results

        eng_file = out_dir / f'phase4_engineering_{timestamp}.json'
        eng_save = {}
        for pname, algos in eng_results.items():
            eng_save[pname] = {}
            for aname, data in algos.items():
                eng_save[pname][aname] = {
                    'stats': data['stats'],
                    'best_fits': data['best_fits'],
                    'feasible_counts': data['feasible_counts'],
                    'gaps': data['gaps'],
                    'times': data['times'],
                }
        with open(eng_file, 'w', encoding='utf-8') as f:
            json.dump(eng_save, f, indent=2, ensure_ascii=False)
        print(f"\nEngineering results saved: {eng_file}")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()