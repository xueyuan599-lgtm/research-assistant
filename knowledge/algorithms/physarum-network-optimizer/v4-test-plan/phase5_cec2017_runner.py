"""
Phase 5B: CEC 2017 测试套件对比实验
====================================
使用 opfunu 库提供的 CEC 2017 竞赛套件（无约束版，29 个函数）。

注意：官方 CEC 2017 约束版（C01-C28）因需官方 MATLAB 数据文件难以公开
复现，本实验使用 opfunu 1.0.4 提供的 CEC 2017 无约束套件作为基准。
    （实现覆盖 F1、F3-F29 共 28 个函数；官方 F2 被 CEC 竞赛排除，
     F30 在 opfunu 1.0.4 中未提供，因此本测试集为 28 个函数。）
约束能力另由 5A 的工程问题（Welded Beam 等）覆盖。

规格（受限于计算资源，已公开注明）:
  - 维度: 10D
  - Max FES: 30,000
  - 独立运行: 10 次
  - 所有算法使用相同预算，排名对比公平

用法:
    python phase5_cec2017_runner.py                # 完整实验
    python phase5_cec2017_runner.py --runs 3       # 快速测试
    python phase5_cec2017_runner.py --resume       # 续跑
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
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from algorithms import (GWO, PSO, DE, SHADE, CMAES,
                        LSHADEcnEPSO, JSO, EA4eig, CMODE)
from algorithm_v4 import PhysarumNetworkOptimizer

# ======================================================================
# 路径
# ======================================================================
OUTPUT_DIR = BASE_DIR / 'outputs'
RESULTS_FILE = OUTPUT_DIR / 'phase5_cec2017.json'
PROGRESS_FILE = OUTPUT_DIR / 'phase5_cec2017_progress.json'


# ======================================================================
# CEC 2017 函数工厂（opfunu）
# ======================================================================

def build_cec2017_functions(ndim=10):
    """构建 CEC 2017 函数列表（官方竞赛排除 F2）"""
    from opfunu.cec_based import (
        F12017, F32017, F42017, F52017, F62017, F72017, F82017, F92017,
        F102017, F112017, F122017, F132017, F142017, F152017, F162017,
        F172017, F182017, F192017, F202017, F212017, F222017, F232017,
        F242017, F252017, F262017, F272017, F282017, F292017,
    )
    cls_list = [
        F12017, F32017, F42017, F52017, F62017, F72017, F82017, F92017,
        F102017, F112017, F122017, F132017, F142017, F152017, F162017,
        F172017, F182017, F192017, F202017, F212017, F222017, F232017,
        F242017, F252017, F262017, F272017, F282017, F292017,
    ]
    funcs = {}
    for F in cls_list:
        inst = F(ndim=ndim)
        name = inst.name.split(':')[0].strip()  # 'F1'
        bias = getattr(inst, 'f_bias', None)
        if bias is None:
            bias = 100.0 * int(name[1:])
        funcs[name] = {
            'func': inst.evaluate,
            'bias': float(bias),
            'name_full': inst.name,
        }
    return funcs


# ======================================================================
# 算法工厂
# ======================================================================

def create_algorithms(n_dim, max_fes):
    algos = {}
    algos['PNO-GWO-v4.0'] = PhysarumNetworkOptimizer(
        n_dim=n_dim, bounds=(-100, 100), max_fes=max_fes,
        use_shca=True, use_cgpsr=False, use_cas=True,
        use_cauchy=True, use_nm=True, use_twophase=True,
        use_cmea=True, use_strategy_pool=True
    )
    algos['GWO'] = GWO(n_pop=min(18 * n_dim, 200))
    algos['PSO'] = PSO(n_pop=min(18 * n_dim, 200))
    algos['DE'] = DE(n_pop=max(10 * n_dim, 50), F=0.5, CR=0.9)
    algos['SHADE'] = SHADE(n_pop=max(10 * n_dim, 50))
    algos['CMA-ES'] = CMAES(sigma0=0.3)
    algos['LSHADE-cnEPSO'] = LSHADEcnEPSO(n_pop=max(18 * n_dim, 100))
    algos['jSO'] = JSO(n_pop=max(18 * n_dim, 100))
    algos['EA4eig'] = EA4eig(n_pop=max(10 * n_dim, 50))
    algos['CMODE'] = CMODE(n_pop=max(10 * n_dim, 50))
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


# ======================================================================
# 主入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 5B: CEC 2017 Comparison')
    parser.add_argument('--runs', type=int, default=10, help='Runs per algo-function')
    parser.add_argument('--max-fes', type=int, default=30000, help='Max FES')
    parser.add_argument('--ndim', type=int, default=10, help='Dimension')
    parser.add_argument('--resume', action='store_true', help='Resume from progress')
    parser.add_argument('--quick', action='store_true', help='Only F1,F5,F10,F20,F29')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_runs = args.runs
    max_fes = args.max_fes
    ndim = args.ndim

    funcs = build_cec2017_functions(ndim=ndim)
    if args.quick:
        funcs = {k: v for k, v in funcs.items() if k in
                 ('F1', 'F5', 'F10', 'F20', 'F29')}

    algos = create_algorithms(ndim, max_fes)

    progress = _load_progress() if args.resume else {'completed': {}, 'results': {}}
    results = progress['results']

    print(f"Phase 5B: CEC 2017 对比实验 ({len(funcs)} functions × {len(algos)} algos × {n_runs} runs)")
    print(f"  D={ndim}  max_fes={max_fes}")

    for fname, finfo in funcs.items():
        print(f"\n{'='*70}\nFunction: {fname} ({finfo['name_full']})\n{'='*70}")
        for algo_name, algo in algos.items():
            key = f"{fname}::{algo_name}"
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
                        finfo['func'], n_dim=ndim, bounds=(-100, 100),
                        max_fes=max_fes, verbose=False
                    )
                    times.append(time.time() - t0)
                    # 减去 bias → 误差值
                    fits.append(float(best_fit - finfo['bias']))
                except Exception as e:
                    print(f"    ERROR ({fname}/{algo_name}/run{run}): {e}")
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
                'bias': finfo['bias'],
            }
            print(f"  {algo_name:<14s} median={stats['median']:12.4e}  "
                  f"mean={stats['mean']:12.4e}  time={stats['mean_time']:.2f}s")

            results[key] = {
                'function': fname,
                'algorithm': algo_name,
                'stats': stats,
                'errors': fits,
                'times': times,
            }
            progress['completed'][key] = True
            _save_progress(progress)

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nCEC 2017 实验完成，结果保存到 {RESULTS_FILE}")


if __name__ == '__main__':
    main()