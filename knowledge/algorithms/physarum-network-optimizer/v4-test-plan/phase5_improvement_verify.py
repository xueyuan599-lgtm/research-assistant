"""
Phase 5 改进验证 (v4.1): 聚焦测试 — 无需大型测试

依据 Phase 5B (CEC 2017) 实测: PNO-GWO 在旋转/混合/组合函数上
落后 DE 族 (SHADE/jSO/CMODE 排名 1-3) 100-1000×, 而强项位于
未旋转经典函数。v4.1 在策略切换中引入 SHADE 式 pbest-DE 臂。

验证设计 (轻量):
  - 强项防线 (防回归): Sphere/Rastrigin/Rosenbrock/Ackley/Griewank (30D)
                      + CEC F5/F7 (10D)
  - 弱项靶区 (验改进): CEC F1/F10/F15/F16/F27/F29 (10D)
  - 变体: v4.0 (use_pbest_de=False) vs v4.1 (True), 同随机种子
  - 预算: max_fes=30000 (与 5B 一致), 每组合 10 次运行

用法:
    python phase5_improvement_verify.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import scipy.stats as sps
from opfunu.cec_based import (
    F12017, F52017, F72017, F102017, F152017, F162017, F272017, F292017,
)

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from algorithm_v4 import PhysarumNetworkOptimizer

OUTPUT_DIR = BASE_DIR / 'outputs'
REPORTS_DIR = BASE_DIR / 'reports'


# ======================================================================
# 基准: 经典 5 函数 (30D) + CEC 靶区函数 (10D)
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
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

def griewank(x):
    return (np.sum(x ** 2) / 4000
            - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1)

CLASSIC = {
    'Sphere': (sphere, (-100, 100), 30),
    'Rastrigin': (rastrigin, (-5.12, 5.12), 30),
    'Rosenbrock': (rosenbrock, (-30, 30), 30),
    'Ackley': (ackley, (-32, 32), 30),
    'Griewank': (griewank, (-600, 600), 30),
}

CEC = {
    # 弱项靶区 (5B Friedman 失分点 + 陷阱高分点)
    'F1': (F12017, '弱项: 旋转 BIC'), 'F10': (F102017, '弱项: 旋转混合'),
    'F15': (F152017, '弱项: 旋转混合'), 'F16': (F162017, '弱项: 旋转混合'),
    'F27': (F272017, '弱项: 组合陷阱'), 'F29': (F292017, '弱项: 组合大失分'),
    # 强项防线 (5B 已占优)
    'F5': (F52017, '强项: 旋转 Rastrigin'), 'F7': (F72017, '强项: 阶梯'),
}

MAX_FES = 30000
N_RUNS = 10
SEED0 = 42

import argparse


def parse_args():
    ap = argparse.ArgumentParser(description='v4.1 聚焦验证')
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--subset', default='all',
                    help='all | classic | weak | strong | f16f29')
    return ap.parse_args()


def make_variant(use_pbest_de, n_dim, bounds, max_fes):
    return PhysarumNetworkOptimizer(
        n_dim=n_dim, bounds=bounds, max_fes=max_fes,
        use_shca=True, use_cgpsr=False, use_cas=True,
        use_cauchy=True, use_nm=True, use_twophase=True,
        use_cmea=True, use_strategy_pool=True, use_pbest_de=use_pbest_de,
    )


def run_variant(variant, func, n_dim, bounds, max_fes, n_runs, seed0):
    fits, times = [], []
    for run in range(n_runs):
        np.random.seed(seed0 + run)
        t0 = time.time()
        bx, bf, _ = variant.optimize(func, n_dim=n_dim, bounds=bounds,
                                     max_fes=max_fes, verbose=False)
        times.append(time.time() - t0)
        fits.append(float(bf))
    return np.array(fits), np.mean(times)


def cec_err(inst, x):
    """CEC 误差 = f(x) - bias"""
    return float(inst.evaluate(x) - getattr(inst, 'f_bias', 0.0))


def main():
    args = parse_args()
    global N_RUNS
    N_RUNS = args.runs
    results = {}
    rows = []
    subset = args.subset

    # ---- 经典 30D ----
    if subset in ('all', 'classic'):
        for fname, (func, bounds, ndim) in CLASSIC.items():
            row = {'function': fname, 'group': '经典(防回归)', 'dim': ndim}
            for ver, use_pde in [('v4.0', False), ('v4.1', True)]:
                alg = make_variant(use_pde, ndim, bounds, MAX_FES)
                fits, mt = run_variant(alg, func, ndim, bounds, MAX_FES, N_RUNS, SEED0)
                row[f'{ver}_median'] = float(np.median(fits))
                row[f'{ver}_best'] = float(np.min(fits))
                row[f'{ver}_time'] = mt
                results[f'{fname}|{ver}'] = fits.tolist()
            w = sps.wilcoxon(results[f'{fname}|v4.0'], results[f'{fname}|v4.1'])
            row['wilcoxon_p'] = float(w.pvalue)
            row['delta'] = row['v4.1_median'] - row['v4.0_median']
            rows.append(row)

    # ---- CEC 10D (靶区子集) ----
    cec_subsets = {
        'all': list(CEC),
        'weak': [k for k, v in CEC.items() if '弱项' in v[1]],
        'strong': [k for k, v in CEC.items() if '强项' in v[1]],
        'f16f29': ['F16', 'F29'],
    }
    pick = cec_subsets.get(subset, list(CEC))
    for fname in pick:
        F, note = CEC[fname]
        inst = F(ndim=10)          # 实例化一次，绑定 evaluate 复用
        bias = getattr(inst, 'f_bias', None)
        if bias is None:
            bias = 0.0
        row = {'function': fname, 'group': note, 'dim': 10}
        for ver, use_pde in [('v4.0', False), ('v4.1', True)]:
            alg = make_variant(use_pde, 10, (-100, 100), MAX_FES)
            errs = []
            for run in range(N_RUNS):
                np.random.seed(SEED0 + run)
                _, bf, _ = alg.optimize(
                    inst.evaluate, n_dim=10, bounds=(-100, 100),
                    max_fes=MAX_FES, verbose=False,
                )
                errs.append(float(bf - bias))
            row[f'{ver}_median'] = float(np.median(errs))
            row[f'{ver}_best'] = float(np.min(errs))
            results[f'{fname}|{ver}'] = errs
        w = sps.wilcoxon(results[f'{fname}|v4.0'], results[f'{fname}|v4.1'])
        row['wilcoxon_p'] = float(w.pvalue)
        row['delta'] = row['v4.1_median'] - row['v4.0_median']
        rows.append(row)

    # ---- 汇总 ----
    lines = ['# Phase 5 改进验证 (v4.0 vs v4.1)\n']
    lines.append(f'- 预算: max_fes={MAX_FES}, 每组合 {N_RUNS} 次运行, 同随机种子')
    lines.append('- v4.1 = v4.0 + SHADE 式 pbest-DE 臂 (current-to-pbest/1 + 二项交叉'
                 ' + F/CR 成功历史), PNO 探索臂内 30% 概率触发\n')
    lines.append('| 函数 | 类别 | v4.0 中位 | v4.1 中位 | Δ(v4.1-v4.0) | Wilcoxon p |')
    lines.append('|------|------|----------|----------|------------|-----------|')
    for r in rows:
        d = r['delta']
        better = '↓' if d < 0 else ('↑' if d > 0 else '=')
        lines.append(
            f"| {r['function']} | {r['group']} | {r['v4.0_median']:.3e} | "
            f"{r['v4.1_median']:.3e} | {d:.2e} {better} | {r['wilcoxon_p']:.3f} |"
        )

    # 分组结论
    weak = [r for r in rows if '弱项' in r['group']]
    strong = [r for r in rows if '强项' in r['group'] or r['group'] == '经典(防回归)']
    n_w = sum(1 for r in weak if r['delta'] < 0)
    n_s = sum(1 for r in strong if r['delta'] < 0)
    lines.append(f'\n## 结论\n'
                 f'- 弱项靶区 (n={len(weak)}): v4.1 改善 {n_w} 个, 退化 {len(weak)-n_w} 个\n'
                 f'- 强项防线 (n={len(strong)}): v4.1 保持 {n_s} 个相对更好, '
                 f'显著回归 (p<0.1 且 Δ>0) 需人工标注')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'phase5_improvement_verify.json').write_text(
        json.dumps(results, indent=1), encoding='utf-8')
    report_path = REPORTS_DIR / 'report_phase5_improvement.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))
    print(f'\n报告: {report_path}')


if __name__ == '__main__':
    main()