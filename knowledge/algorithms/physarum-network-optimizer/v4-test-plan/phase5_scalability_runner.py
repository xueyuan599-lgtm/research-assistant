"""
Phase 5C: 高维可扩展性测试
==========================
验证 PNO-GWO 在 100D/300D/500D 下的性能衰减。

5 基准函数 × 3 维度 × 30 次运行，与 GWO/CMA-ES 对比。

用法:
    python phase5_scalability_runner.py               # 完整实验
    python phase5_scalability_runner.py --runs 5      # 快速测试
    python phase5_scalability_runner.py --resume      # 续跑
"""

import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from algorithms import GWO, CMAES, PSO, DE
from algorithm_v4 import PhysarumNetworkOptimizer

OUTPUT_DIR = BASE_DIR / 'outputs'
RESULTS_FILE = OUTPUT_DIR / 'phase5_scalability.json'
PROGRESS_FILE = OUTPUT_DIR / 'phase5_scalability_progress.json'


# ======================================================================
# 基准函数（向量化）
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


FUNCTIONS = {
    'Sphere':     (sphere,     (-100, 100)),
    'Rastrigin':  (rastrigin,  (-5.12, 5.12)),
    'Rosenbrock': (rosenbrock, (-30, 30)),
    'Ackley':     (ackley,     (-32, 32)),
    'Griewank':   (griewank,   (-600, 600)),
}

# 维度配置
DIM_CONFIGS = {
    100: {'max_fes': 100000},
    300: {'max_fes': 200000},
    500: {'max_fes': 300000},
}


def create_algorithms(n_dim, max_fes):
    algos = {}
    algos['PNO-GWO-v4.0'] = PhysarumNetworkOptimizer(
        n_dim=n_dim, bounds=(-100, 100), max_fes=max_fes,
        use_shca=True, use_cgpsr=False, use_cas=True,
        use_cauchy=True, use_nm=True, use_twophase=True,
        use_cmea=True, use_strategy_pool=True,
        n_pop=min(18 * n_dim, 300)
    )
    algos['GWO'] = GWO(n_pop=min(18 * n_dim, 200))
    algos['PSO'] = PSO(n_pop=min(18 * n_dim, 200))
    algos['DE'] = DE(n_pop=min(max(10 * n_dim, 50), 300), F=0.5, CR=0.9)
    algos['CMA-ES'] = CMAES(sigma0=0.3)
    return algos


def _load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': {}, 'results': {}}


def _save_progress(progress):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description='Phase 5C: Scalability Test')
    parser.add_argument('--runs', type=int, default=30, help='Runs per combo')
    parser.add_argument('--dims', type=str, default='100,300,500',
                        help='Comma-separated dims')
    parser.add_argument('--resume', action='store_true', help='Resume')
    parser.add_argument('--quick', action='store_true', help='Only 100D')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_runs = args.runs
    dims = [100] if args.quick else [int(d) for d in args.dims.split(',')]

    progress = _load_progress() if args.resume else {'completed': {}, 'results': {}}
    results = progress['results']

    print(f"Phase 5C: 高维可扩展性测试")
    print(f"  dims={dims}  runs={n_runs}")

    for ndim in dims:
        cfg = DIM_CONFIGS[ndim]
        max_fes = cfg['max_fes']
        print(f"\n{'='*70}\n维度: {ndim}D  max_fes={max_fes}\n{'='*70}")

        algos = create_algorithms(ndim, max_fes)

        for fname, (func, bounds) in FUNCTIONS.items():
            for algo_name, algo in algos.items():
                key = f"{ndim}D::{fname}::{algo_name}"
                if args.resume and key in progress['completed']:
                    print(f"  跳过 {key}")
                    continue

                fits = []
                times = []
                for run in range(n_runs):
                    np.random.seed(42 + run)
                    try:
                        t0 = time.time()
                        best_x, best_fit, _ = algo.optimize(
                            func, n_dim=ndim, bounds=bounds,
                            max_fes=max_fes, verbose=False
                        )
                        times.append(time.time() - t0)
                        fits.append(float(best_fit))
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        traceback.print_exc()
                        fits.append(float('inf'))
                        times.append(0.0)

                arr = np.array(fits)
                finite = arr[np.isfinite(arr)]
                stats = {
                    'best': float(np.min(finite)) if len(finite) else float('inf'),
                    'median': float(np.median(finite)) if len(finite) else float('inf'),
                    'mean': float(np.mean(finite)) if len(finite) else float('inf'),
                    'std': float(np.std(finite)) if len(finite) else float('inf'),
                    'mean_time': float(np.mean(times)),
                }
                print(f"  {fname:<10s} {algo_name:<14s} "
                      f"median={stats['median']:10.4e}  time={stats['mean_time']:.2f}s")

                results[key] = {
                    'dim': ndim,
                    'function': fname,
                    'algorithm': algo_name,
                    'stats': stats,
                    'best_fits': fits,
                    'times': times,
                }
                progress['completed'][key] = True
                _save_progress(progress)

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\n可扩展性测试完成，结果保存到 {RESULTS_FILE}")


if __name__ == '__main__':
    main()