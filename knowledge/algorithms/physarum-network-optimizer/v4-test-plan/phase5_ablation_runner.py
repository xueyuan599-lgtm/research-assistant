"""
Phase 5A: PNO-GWO v4.0 消融实验
================================
量化每个核心机制对整体性能的独立贡献。

7 配置 × 8 问题 × 30 次运行 = 1680 次运行
配置:
  Full (v4.0)      — 全部组件
  w/o SHCA         — 去掉成功历史记忆
  w/o Archive      — 去掉外部归档
  w/o Strategy     — 去掉策略池（纯 PNO 探索）
  w/o Cauchy       — 去掉 Cauchy 变异/重启
  w/o NM           — 去掉 Nelder-Mead 精化
  Baseline         — 全部关闭

用法:
    python phase5_ablation_runner.py                 # 完整实验
    python phase5_ablation_runner.py --runs 3        # 快速测试
    python phase5_ablation_runner.py --resume        # 续跑
"""

import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np

# 确保能导入同级模块
sys.path.insert(0, str(Path(__file__).parent))

from constraint_handler import get_all_engineering_problems

# 导入 PNO-GWO v4.0
sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithm_v4 import PhysarumNetworkOptimizer


# ======================================================================
# 路径
# ======================================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
REPORTS_DIR = BASE_DIR / 'reports'
RESULTS_FILE = OUTPUT_DIR / 'phase5_ablation.json'
PROGRESS_FILE = OUTPUT_DIR / 'phase5_ablation_progress.json'


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
    'Sphere':     (sphere,     (-100, 100)),
    'Rastrigin':  (rastrigin,  (-5.12, 5.12)),
    'Rosenbrock': (rosenbrock, (-30, 30)),
    'Ackley':     (ackley,     (-32, 32)),
    'Griewank':   (griewank,   (-600, 600)),
}

# 消融配置（对应计划表）
# 5 个机制: SHCA / Archive / Strategy / Cauchy / NM
ABLATION_CONFIGS = {
    'Full':         dict(use_shca=True,  use_cmea=True,  use_strategy_pool=True,  use_cauchy=True,  use_nm=True),
    'w/o_SHCA':     dict(use_shca=False, use_cmea=True,  use_strategy_pool=True,  use_cauchy=True,  use_nm=True),
    'w/o_Archive':  dict(use_shca=True,  use_cmea=False, use_strategy_pool=True,  use_cauchy=True,  use_nm=True),
    'w/o_Strategy': dict(use_shca=True,  use_cmea=True,  use_strategy_pool=False, use_cauchy=True,  use_nm=True),
    'w/o_Cauchy':   dict(use_shca=True,  use_cmea=True,  use_strategy_pool=True,  use_cauchy=False, use_nm=True),
    'w/o_NM':       dict(use_shca=True,  use_cmea=True,  use_strategy_pool=True,  use_cauchy=True,  use_nm=False),
    'Baseline':     dict(use_shca=False, use_cmea=False, use_strategy_pool=False, use_cauchy=False, use_nm=False),
}

# 机制顺序（用于报告和热力图）
MECHANISMS = ['SHCA', 'Archive', 'Strategy', 'Cauchy', 'NM']


# ======================================================================
# 工具函数
# ======================================================================

def _load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': {}, 'results': {}}


def _save_progress(progress):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=1, default=str)


def _make_constrained_obj(prob):
    """包装工程问题为含罚函数的单目标"""
    def obj_func(x):
        obj_val, violation = prob.evaluate(x)
        if violation > 0:
            return obj_val + 1e6 * violation
        return obj_val
    return obj_func


# ======================================================================
# 核心实验
# ======================================================================

def run_single(config_name, config, problem_key, problem_cfg,
               n_runs=30, max_fes=30000, start_run=0):
    """对单个 (配置, 问题) 组合运行 n_runs 次"""
    # 解析问题配置
    if problem_key in BENCHMARK_FUNCTIONS:
        obj_func, bounds = BENCHMARK_FUNCTIONS[problem_key]
        n_dim = 30
        is_engineering = False
    else:
        prob = problem_cfg
        obj_func = _make_constrained_obj(prob)
        bounds = (np.array([b[0] for b in prob.bounds]),
                  np.array([b[1] for b in prob.bounds]))
        n_dim = prob.n_dim
        is_engineering = True

    best_fits = []
    feasible_flags = []
    times = []

    for run in range(start_run, n_runs):
        np.random.seed(42 + run)
        try:
            algo = PhysarumNetworkOptimizer(
                n_dim=n_dim, bounds=bounds, max_fes=max_fes,
                use_cas=True, use_twophase=True,
                **config
            )
            t0 = time.time()
            best_x, best_fit, _ = algo.optimize(
                obj_func, n_dim=n_dim, bounds=bounds, max_fes=max_fes,
                verbose=False
            )
            elapsed = time.time() - t0

            if is_engineering:
                _, violation = prob.evaluate(best_x)
                feasible_flags.append(1 if violation <= 1e-6 else 0)
            else:
                feasible_flags.append(1)

            best_fits.append(float(best_fit))
            times.append(elapsed)
        except Exception as e:
            print(f"    ERROR ({config_name}/{problem_key}/run{run}): {e}")
            traceback.print_exc()
            best_fits.append(float('inf'))
            feasible_flags.append(0)
            times.append(0.0)

    return {
        'best_fits': best_fits,
        'feasible_flags': feasible_flags,
        'times': times,
    }


# ======================================================================
# 主入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 5A: PNO-GWO Ablation Study')
    parser.add_argument('--runs', type=int, default=30, help='Runs per config-problem')
    parser.add_argument('--max-fes', type=int, default=30000, help='Max FES')
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
    parser.add_argument('--only', type=str, default=None, help='Only run specific config name')
    parser.add_argument('--suffix', type=str, default='',
                        help='Output file suffix (for parallel runs)')
    args = parser.parse_args()

    global RESULTS_FILE, PROGRESS_FILE
    if args.suffix:
        RESULTS_FILE = OUTPUT_DIR / f'phase5_ablation_{args.suffix}.json'
        PROGRESS_FILE = OUTPUT_DIR / f'phase5_ablation_progress_{args.suffix}.json'

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    n_runs = args.runs
    max_fes = args.max_fes

    # 构建问题集：5 基准 + 3 工程（Welded Beam, Pressure Vessel, Spring）
    problems = {k: v for k, v in BENCHMARK_FUNCTIONS.items()}
    eng_probs = get_all_engineering_problems()
    problems['welded_beam'] = eng_probs['welded_beam']
    problems['pressure_vessel'] = eng_probs['pressure_vessel']
    problems['spring'] = eng_probs['spring']

    # 进度管理
    progress = _load_progress() if args.resume else {'completed': {}, 'results': {}}
    results = progress['results']

    configs_to_run = {k: v for k, v in ABLATION_CONFIGS.items()
                      if args.only is None or k == args.only}

    print(f"Phase 5A 消融实验")
    print(f"  配置数: {len(configs_to_run)}  问题数: {len(problems)}  "
          f"运行次数: {n_runs}  Max FES: {max_fes}")
    print(f"  总计: {len(configs_to_run) * len(problems) * n_runs} 次运行")

    for config_name, config in configs_to_run.items():
        print(f"\n{'='*70}")
        print(f"配置: {config_name}  {config}")
        print(f"{'='*70}")

        for prob_key, prob_cfg in problems.items():
            key = f"{config_name}::{prob_key}"
            if args.resume and key in progress['completed']:
                print(f"  跳过已完成: {key}")
                continue

            print(f"  Problem: {prob_key}")
            res = run_single(config_name, config, prob_key, prob_cfg,
                             n_runs=n_runs, max_fes=max_fes)

            ndim = 30 if prob_key in BENCHMARK_FUNCTIONS else prob_cfg.n_dim
            fits = np.array(res['best_fits'])
            feas = np.array(res['feasible_flags'])
            finite = fits[np.isfinite(fits)]

            stats = {
                'best': float(np.min(finite)) if len(finite) > 0 else float('inf'),
                'median': float(np.median(finite)) if len(finite) > 0 else float('inf'),
                'mean': float(np.mean(finite)) if len(finite) > 0 else float('inf'),
                'std': float(np.std(finite)) if len(finite) > 0 else float('inf'),
                'feasible_rate': float(feas.mean()),
                'mean_time': float(np.mean(res['times'])),
                'n_dim': ndim,
            }
            print(f"    median={stats['median']:.6e}  feas={stats['feasible_rate']:.2f}  "
                  f"time={stats['mean_time']:.2f}s")

            results[key] = {
                'config': config,
                'problem': prob_key,
                'stats': stats,
                'best_fits': res['best_fits'],
                'feasible_flags': res['feasible_flags'],
                'times': res['times'],
            }
            progress['completed'][key] = True
            _save_progress(progress)

    # 保存汇总结果
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=1)

    print(f"\n消融实验完成，结果已保存到 {RESULTS_FILE}")
    print(f"原始数据: {len(results)} 个 (配置×问题) 组合")


if __name__ == '__main__':
    main()