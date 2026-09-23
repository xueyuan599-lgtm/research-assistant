"""
Phase 5E: 计算复杂度分析
========================
量化 PNO-GWO 的计算成本。

1. 时间复杂度: 测量不同 N×D 组合的运行时间，拟合 T(N,D) = a·N^b·D^c
2. 组件时间占比: 图构建 / SHCA / 归档 / 策略选择 / Cauchy / NM / 其他

用法:
    python phase5_complexity_runner.py                # 完整实验
    python phase5_complexity_runner.py --quick        # 快速测试
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from algorithm_v4 import PhysarumNetworkOptimizer

OUTPUT_DIR = BASE_DIR / 'outputs'
RESULTS_FILE = OUTPUT_DIR / 'phase5_complexity.json'


def sphere(x):
    return np.sum(x ** 2)


def run_timing_experiment():
    """时间复杂度的 N×D 组合测试

    固定代数 T=30（而非固定 FES），这样每代成本随 N,D 增长的变化
    才能真实反映算法复杂度（图构建 O(N log N) + 节点更新 O(N·D)）。
    """
    results = {}
    pop_sizes = [30, 50, 100, 200, 500]
    dims = [10, 30, 50, 100, 200, 500]
    fixed_gens = 30
    # FES = N × 30，保证每代对所有节点都更新一遍
    max_fes = 500 * fixed_gens + 1000  # 上界，实际由代数控制

    for N in pop_sizes:
        for D in dims:
            fes_budget = max(N * fixed_gens + N, 1000)
            try:
                algo = PhysarumNetworkOptimizer(
                    n_pop=N, n_dim=D, bounds=(-10, 10),
                    max_fes=fes_budget, verbose=False
                )
                times = []
                for _ in range(5):
                    np.random.seed(1)
                    t0 = time.time()
                    algo.optimize(sphere, n_dim=D, bounds=(-10, 10),
                                  max_fes=fes_budget, verbose=False)
                    times.append(time.time() - t0)
                results[f'N{N}_D{D}'] = {
                    'N': N, 'D': D,
                    'median_time': float(np.median(times)),
                    'times': times,
                    'fes_used': algo.n_fes,
                    'gens': algo._generation,
                }
                print(f"  N={N:4d} D={D:4d}  time={np.median(times):.3f}s  "
                      f"gens={algo._generation}")
            except Exception as e:
                print(f"  N={N:4d} D={D:4d}  ERROR: {e}")
                results[f'N{N}_D{D}'] = {
                    'N': N, 'D': D, 'median_time': None, 'times': [], 'error': str(e)
                }

    return results


def measure_component_times():
    """组件时间占比分析

    通过分别禁用组件来测量其边际成本:
      ΔT(component) = T_full - T(without component)
    """
    from scipy.spatial import KDTree

    N = 100
    D = 30
    max_fes = 20000

    configs = {
        'Full':         dict(use_shca=True,  use_cmea=True,  use_strategy_pool=True,  use_cauchy=True,  use_nm=True),
        'w/o_SHCA':     dict(use_shca=False, use_cmea=True,  use_strategy_pool=True,  use_cauchy=True,  use_nm=True),
        'w/o_Archive':  dict(use_shca=True,  use_cmea=False, use_strategy_pool=True,  use_cauchy=True,  use_nm=True),
        'w/o_Strategy': dict(use_shca=True,  use_cmea=True,  use_strategy_pool=False, use_cauchy=True,  use_nm=True),
        'w/o_Cauchy':   dict(use_shca=True,  use_cmea=True,  use_strategy_pool=True,  use_cauchy=False, use_nm=True),
        'w/o_NM':       dict(use_shca=True,  use_cmea=True,  use_strategy_pool=True,  use_cauchy=True,  use_nm=False),
    }

    times = {}
    for name, cfg in configs.items():
        algo = PhysarumNetworkOptimizer(
            n_pop=N, n_dim=D, bounds=(-10, 10), max_fes=max_fes,
            use_cas=True, use_twophase=True, **cfg
        )
        t_med = []
        for _ in range(3):
            np.random.seed(1)
            t0 = time.time()
            algo.optimize(sphere, n_dim=D, bounds=(-10, 10),
                          max_fes=max_fes, verbose=False)
            t_med.append(time.time() - t0)
        times[name] = float(np.median(t_med))
        print(f"  {name:<12s} time={times[name]:.3f}s")

    # 各组件边际成本
    components = {}
    t_full = times['Full']
    for comp in ['SHCA', 'Archive', 'Strategy', 'Cauchy', 'NM']:
        t_without = times[f'w/o_{comp}']
        marginal = t_full - t_without
        components[comp] = {
            't_full': t_full,
            't_without': t_without,
            'marginal_cost': float(max(marginal, 0)),
            'pct_of_full': float(max(marginal, 0) / t_full * 100),
        }

    return times, components


def main():
    parser = argparse.ArgumentParser(description='Phase 5E: Complexity Analysis')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("5E.1 时间复杂度: N×D 组合测量")
    print("=" * 70)
    timing = run_timing_experiment()

    print("\n" + "=" * 70)
    print("5E.2 组件时间占比")
    print("=" * 70)
    full_times, components = measure_component_times()

    result = {
        'timing': timing,
        'component_times': full_times,
        'component_marginal': components,
        'fitness_function': 'sphere',
        'method': 'T(N,D) = a·N^b·D^c, 固定 FES = 10000 (timing) / 20000 (components)',
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(result, f, indent=1)
    print(f"\n复杂度分析完成，结果保存到 {RESULTS_FILE}")


if __name__ == '__main__':
    main()