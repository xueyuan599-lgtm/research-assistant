"""
PNO-GWO v4.0 实验框架
=====================
统一接口：支持消融实验、敏感性分析、工程问题对比。

用法:
    python v4_experiment_runner.py --experiment ablation --runs 10
    python v4_experiment_runner.py --experiment sensitivity_fes --runs 10
    python v4_experiment_runner.py --experiment all --runs 30
"""
import sys
import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加算法目录到路径
ALGO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ALGO_DIR))

from algorithm import PhysarumNetworkOptimizer
from algorithm_v4 import PhysarumNetworkOptimizer as PhysarumNetworkOptimizerV4


# ======================================================================
# 1. 测试函数集
# ======================================================================

class BenchmarkFunctions:
    """经典优化测试函数"""

    @staticmethod
    def sphere(x):
        return np.sum(x ** 2)

    @staticmethod
    def rastrigin(x):
        n = len(x)
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    @staticmethod
    def rosenbrock(x):
        return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                   for i in range(len(x) - 1))

    @staticmethod
    def ackley(x):
        n = len(x)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
                + 20 + np.e)

    @staticmethod
    def griewank(x):
        sum_sq = np.sum(x ** 2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_sq - prod_cos + 1

    @staticmethod
    def get_all():
        return {
            'sphere': (BenchmarkFunctions.sphere, (-100, 100)),
            'rastrigin': (BenchmarkFunctions.rastrigin, (-5.12, 5.12)),
            'rosenbrock': (BenchmarkFunctions.rosenbrock, (-30, 30)),
            'ackley': (BenchmarkFunctions.ackley, (-32.768, 32.768)),
            'griewank': (BenchmarkFunctions.griewank, (-600, 600)),
        }


# ======================================================================
# 2. 工程问题集
# ======================================================================

class EngineeringProblems:
    """经典工程设计优化问题"""

    @staticmethod
    def welded_beam(x):
        """焊接梁设计 (4 变量, 4 约束)"""
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        # 目标: 最小化成本
        cost = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
        # 约束
        P = 6000; L = 14; E = 30e6; G = 12e6
        tau_max = 13600; sigma_max = 30000; delta_max = 0.25
        M = P * (L + x2 / 2)
        R = np.sqrt(x2**2 / 4 + ((x1 + x3) / 2)**2)
        J = 2 * (x1 * x2 / np.sqrt(2) * (x2**2 / 12 + ((x1 + x3) / 2)**2))
        tau1 = P / (np.sqrt(2) * x1 * x2)
        tau2 = M * R / J
        tau = np.sqrt(tau1**2 + 2 * tau1 * tau2 * x2 / (2 * R) + tau2**2)
        sigma = 6 * P * L / (x4 * x3**2)
        delta = 4 * P * L**3 / (E * x3**3 * x4)
        Pc = (4.013 * E * np.sqrt(x3**2 * x4**6 / 36) / L**2
              * (1 - x3 / (2 * L) * np.sqrt(E / (4 * G))))
        # 罚函数
        penalty = 0
        if tau - tau_max > 0: penalty += 1e6 * (tau - tau_max)**2
        if sigma - sigma_max > 0: penalty += 1e6 * (sigma - sigma_max)**2
        if delta - delta_max > 0: penalty += 1e6 * (delta - delta_max)**2
        if x1 - x4 > 0: penalty += 1e6 * (x1 - x4)**2
        if 0.10471 * x1**2 + 0.04811 * x3 * x4 * (14 + x2) - 5 > 0:
            penalty += 1e6
        if P - Pc > 0: penalty += 1e6 * (P - Pc)**2
        return cost + penalty

    @staticmethod
    def pressure_vessel(x):
        """压力容器设计 (4 变量, 4 约束)"""
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        # x1=Ts, x2=Th, x3=R, x4=L (x1,x2 为整数倍 0.0625)
        cost = (0.6224 * x1 * x3 * x4
                + 1.7781 * x2 * x3**2
                + 3.1661 * x1**2 * x4
                + 19.84 * x1**2 * x3)
        penalty = 0
        if -x1 + 0.0193 * x3 > 0: penalty += 1e6 * (-x1 + 0.0193 * x3)**2
        if -x2 + 0.00954 * x3 > 0: penalty += 1e6 * (-x2 + 0.00954 * x3)**2
        if -np.pi * x3**2 * x4 - 4/3 * np.pi * x3**3 + 1296000 > 0:
            penalty += 1e6 * (-np.pi * x3**2 * x4 - 4/3 * np.pi * x3**3 + 1296000)**2
        if x4 - 240 > 0: penalty += 1e6 * (x4 - 240)**2
        return cost + penalty

    @staticmethod
    def tension_compression_spring(x):
        """拉压弹簧设计 (3 变量, 4 约束)"""
        x1, x2, x3 = x[0], x[1], x[2]  # d, D, N
        cost = (x3 + 2) * x2 * x1**2
        penalty = 0
        if 1 - x2**3 * x3 / (71785 * x1**4) > 0:
            penalty += 1e6 * (1 - x2**3 * x3 / (71785 * x1**4))**2
        if (4 * x2**2 - x1 * x2) / (12566 * (x2 * x1**3 - x1**4)) + 1 / (5108 * x1**2) - 1 > 0:
            penalty += 1e6
        if 1 - 140.45 * x1 / (x2**2 * x3) > 0:
            penalty += 1e6 * (1 - 140.45 * x1 / (x2**2 * x3))**2
        if (x1 + x2) / 1.5 - 1 > 0:
            penalty += 1e6 * ((x1 + x2) / 1.5 - 1)**2
        return cost + penalty

    @staticmethod
    def speed_reducer(x):
        """减速器设计 (7 变量, 11 约束)"""
        x1, x2, x3, x4, x5, x6, x7 = x[:7]
        cost = (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
                - 1.508 * x1 * (x6**2 + x7**2)
                + 7.4777 * (x6**3 + x7**3)
                + 0.7854 * (x4 * x6**2 + x5 * x7**2))
        penalty = 0
        g = [
            27 / (x1 * x2**2 * x3) - 1,
            397.5 / (x1 * x2**2 * x3**2) - 1,
            1.93 * x4**3 / (x2 * x3 * x6**4) - 1,
            1.93 * x5**3 / (x2 * x3 * x7**4) - 1,
            np.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) / (110 * x6**3) - 1,
            np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) / (85 * x7**3) - 1,
            x2 * x3 / 40 - 1,
            5 * x2 / x1 - 1,
            x1 / (12 * x2) - 1,
            (1.5 * x6 + 1.9) / x4 - 1,
            (1.1 * x7 + 1.9) / x5 - 1,
        ]
        for gi in g:
            if gi > 0:
                penalty += 1e6 * gi**2
        return cost + penalty

    @staticmethod
    def three_bar_truss(x):
        """三杆桁架 (2 变量, 3 约束)"""
        x1, x2 = x[0], x[1]
        P = 2; sigma = 2; L = 1
        cost = (2 * np.sqrt(2) * x1 + x2) * L
        penalty = 0
        g1 = np.sqrt(2) * x1 + x2 / (np.sqrt(2) * x1**2 + 2 * x1 * x2) * P / sigma - 1
        g2 = x2 / (np.sqrt(2) * x1**2 + 2 * x1 * x2) * P / sigma - 1
        g3 = 1 / (np.sqrt(2) * x2 + x1) * P / sigma - 1
        for gi in [g1, g2, g3]:
            if gi > 0:
                penalty += 1e6 * gi**2
        return cost + penalty

    @staticmethod
    def get_classic():
        """返回经典工程问题集"""
        return {
            'welded_beam': (EngineeringProblems.welded_beam, (0.1, 2.0), 4),
            'pressure_vessel': (EngineeringProblems.pressure_vessel, (0.0625, 99), 4),
            'spring': (EngineeringProblems.tension_compression_spring, (0.05, 2.0), 3),
            'speed_reducer': (EngineeringProblems.speed_reducer, (2.6, 3.6), 7),
            'three_bar_truss': (EngineeringProblems.three_bar_truss, (0.001, 1.0), 2),
        }


# ======================================================================
# 3. 统计工具
# ======================================================================

class Stats:
    """统计分析工具"""

    @staticmethod
    def compute_metrics(results):
        """计算标准优化指标"""
        arr = np.array(results)
        return {
            'best': float(np.min(arr)),
            'median': float(np.median(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'success_rate': float(np.sum(arr < 1e-6) / len(arr)),
        }

    @staticmethod
    def wilcoxon_test(r1, r2):
        """Wilcoxon 秩和检验"""
        from scipy.stats import ranksums
        stat, p = ranksums(r1, r2)
        return {'statistic': float(stat), 'p_value': float(p)}

    @staticmethod
    def convergence_speed(curves, threshold=1e-6):
        """计算达到阈值所需的 FES"""
        speeds = []
        for curve in curves:
            for i, v in enumerate(curve):
                if v < threshold:
                    speeds.append(i)
                    break
            else:
                speeds.append(len(curve))
        return {
            'mean_fes': float(np.mean(speeds)),
            'median_fes': float(np.median(speeds)),
            'success_rate': float(sum(s < len(curves[0]) for s in speeds) / len(speeds)),
        }


# ======================================================================
# 4. 实验运行器
# ======================================================================

class ExperimentRunner:
    """统一实验运行器"""

    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or
                                ALGO_DIR.parent.parent / "outputs" / "pno_v4_experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_single(self, obj_func, bounds, n_dim, max_fes, config, seed, version='v3.5'):
        """单次优化运行"""
        np.random.seed(seed)
        if version == 'v4.0':
            pno = PhysarumNetworkOptimizerV4(
                n_dim=n_dim, bounds=bounds, max_fes=max_fes, **config
            )
        else:
            pno = PhysarumNetworkOptimizer(
                n_dim=n_dim, bounds=bounds, max_fes=max_fes, **config
            )
        start = time.time()
        best_x, best_fit, convergence = pno.optimize(obj_func, verbose=False)
        elapsed = time.time() - start
        return {
            'best_fit': best_fit,
            'convergence': convergence,
            'time': elapsed,
            'n_fes': pno.n_fes,
        }

    def run_experiment(self, name, obj_func, bounds, n_dim, max_fes,
                       configs, n_runs=10, seeds=None, version='v3.5'):
        """运行一组对比实验"""
        if seeds is None:
            seeds = list(range(n_runs))

        results = {}
        for config_name, config in configs.items():
            print(f"\n  > {config_name} ({n_runs} runs)...")
            run_results = []
            curves = []
            times = []

            for i, seed in enumerate(seeds):
                res = self.run_single(obj_func, bounds, n_dim, max_fes, config, seed, version)
                run_results.append(res['best_fit'])
                curves.append(res['convergence'])
                times.append(res['time'])
                if (i + 1) % max(1, n_runs // 5) == 0:
                    print(f"    [{i+1}/{n_runs}] median={np.median(run_results):.6e}")

            metrics = Stats.compute_metrics(run_results)
            metrics['mean_time'] = float(np.mean(times))
            metrics['convergence_speed'] = Stats.convergence_speed(curves)
            results[config_name] = {
                'metrics': metrics,
                'runs': run_results,
                'curves': curves,
            }
            print(f"    → median={metrics['median']:.6e}, "
                  f"best={metrics['best']:.6e}, "
                  f"time={metrics['mean_time']:.2f}s")

        self.results[name] = results
        return results

    def save_results(self, name=None):
        """保存结果到 JSON"""
        fname = f"{name or 'results'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fpath = self.output_dir / fname

        # 转换为可序列化格式
        serializable = {}
        for exp_name, exp_data in self.results.items():
            serializable[exp_name] = {}
            for config_name, config_data in exp_data.items():
                serializable[exp_name][config_name] = {
                    'metrics': config_data['metrics'],
                    'runs': config_data['runs'],
                    # 不保存完整收敛曲线（太大）
                }

        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {fpath}")
        return fpath

    def print_comparison_table(self, name):
        """打印对比表格"""
        if name not in self.results:
            print(f"实验 {name} 不存在")
            return

        results = self.results[name]
        print(f"\n{'='*70}")
        print(f"实验: {name}")
        print(f"{'='*70}")
        print(f"{'配置':<25} {'Median':<15} {'Mean±Std':<20} {'Best':<15} {'Time':<8}")
        print(f"{'-'*70}")

        for config_name, data in results.items():
            m = data['metrics']
            print(f"{config_name:<25} "
                  f"{m['median']:<15.6e} "
                  f"{m['mean']:.2e}±{m['std']:.2e}  "
                  f"{m['best']:<15.6e} "
                  f"{m['mean_time']:<8.2f}")


# ======================================================================
# 5. 消融实验配置
# ======================================================================

def get_ablation_configs():
    """E1: 机制消融配置"""
    return {
        'v3.5_full': {
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        },
        'no_shca': {
            'use_shca': False, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        },
        'no_cgpsr': {
            'use_shca': True, 'use_cgpsr': False, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        },
        'no_cas': {
            'use_shca': True, 'use_cgpsr': True, 'use_cas': False,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        },
        'no_cauchy': {
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': False, 'use_nm': True, 'use_twophase': True,
        },
        'no_nm': {
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': False, 'use_twophase': True,
        },
        'no_twophase': {
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': False,
        },
        'all_off': {
            'use_shca': False, 'use_cgpsr': False, 'use_cas': False,
            'use_cauchy': False, 'use_nm': False, 'use_twophase': False,
        },
    }


def get_fes_split_configs():
    """E2: FES 分割敏感性"""
    configs = {}
    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
        configs[f'phase1_{int(ratio*100)}'] = {
            'phase1_ratio': ratio,
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        }
    return configs


def get_restart_configs():
    """E3: 重启阈值敏感性"""
    configs = {}
    for interval in [5, 8, 10, 15, 20, 30]:
        configs[f'restart_{interval}'] = {
            'restart_interval': interval,
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        }
    return configs


def get_knn_configs():
    """E4: k_neighbors 敏感性"""
    configs = {}
    for k in [3, 5, 7, 10, 15]:
        configs[f'knn_{k}'] = {
            'k_neighbors': k,
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        }
    return configs


def get_nm_timing_configs():
    """E5: NM 触发时机"""
    configs = {}
    for ratio in [0.6, 0.7, 0.8, 0.9]:
        configs[f'nm_phase1_{int(ratio*100)}'] = {
            'phase1_ratio': ratio,
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
        }
    return configs


# ======================================================================
# 6. 主入口
# ======================================================================

def run_ablation(runner, n_dim=30, max_fes=30000, n_runs=10):
    """E1: 机制消融实验"""
    print("\n" + "="*70)
    print("E1: 机制消融实验")
    print("="*70)
    configs = get_ablation_configs()
    benchmarks = BenchmarkFunctions.get_all()

    for func_name, (func, bounds) in benchmarks.items():
        print(f"\n{'-'*50}")
        print(f"函数: {func_name} ({n_dim}D)")
        print(f"{'-'*50}")
        runner.run_experiment(
            f'ablation_{func_name}', func, bounds, n_dim, max_fes,
            configs, n_runs
        )
        runner.print_comparison_table(f'ablation_{func_name}')


def run_fes_sensitivity(runner, n_dim=30, max_fes=30000, n_runs=10):
    """E2: FES 分割敏感性"""
    print("\n" + "="*70)
    print("E2: FES 分割敏感性实验")
    print("="*70)
    configs = get_fes_split_configs()
    benchmarks = BenchmarkFunctions.get_all()

    for func_name, (func, bounds) in benchmarks.items():
        print(f"\n{'-'*50}")
        print(f"函数: {func_name} ({n_dim}D)")
        print(f"{'-'*50}")
        runner.run_experiment(
            f'fes_split_{func_name}', func, bounds, n_dim, max_fes,
            configs, n_runs
        )
        runner.print_comparison_table(f'fes_split_{func_name}')


def run_restart_sensitivity(runner, n_dim=30, max_fes=30000, n_runs=10):
    """E3: 重启阈值敏感性"""
    print("\n" + "="*70)
    print("E3: 重启阈值敏感性实验")
    print("="*70)
    configs = get_restart_configs()
    test_funcs = {
        'rastrigin': (BenchmarkFunctions.rastrigin, (-5.12, 5.12)),
        'ackley': (BenchmarkFunctions.ackley, (-32.768, 32.768)),
        'griewank': (BenchmarkFunctions.griewank, (-600, 600)),
    }

    for func_name, (func, bounds) in test_funcs.items():
        print(f"\n{'-'*50}")
        print(f"函数: {func_name} ({n_dim}D)")
        print(f"{'-'*50}")
        runner.run_experiment(
            f'restart_{func_name}', func, bounds, n_dim, max_fes,
            configs, n_runs
        )
        runner.print_comparison_table(f'restart_{func_name}')


def run_knn_sensitivity(runner, n_dim=30, max_fes=30000, n_runs=10):
    """E4: k_neighbors 敏感性"""
    print("\n" + "="*70)
    print("E4: k_neighbors 敏感性实验")
    print("="*70)
    configs = get_knn_configs()
    benchmarks = BenchmarkFunctions.get_all()

    for func_name, (func, bounds) in benchmarks.items():
        print(f"\n{'-'*50}")
        print(f"函数: {func_name} ({n_dim}D)")
        print(f"{'-'*50}")
        runner.run_experiment(
            f'knn_{func_name}', func, bounds, n_dim, max_fes,
            configs, n_runs
        )
        runner.print_comparison_table(f'knn_{func_name}')


def run_nm_timing(runner, n_dim=30, max_fes=30000, n_runs=10):
    """E5: NM 触发时机"""
    print("\n" + "="*70)
    print("E5: NM 触发时机实验")
    print("="*70)
    configs = get_nm_timing_configs()
    test_funcs = {
        'sphere': (BenchmarkFunctions.sphere, (-100, 100)),
        'rosenbrock': (BenchmarkFunctions.rosenbrock, (-30, 30)),
        'ackley': (BenchmarkFunctions.ackley, (-32.768, 32.768)),
    }

    for func_name, (func, bounds) in test_funcs.items():
        print(f"\n{'-'*50}")
        print(f"函数: {func_name} ({n_dim}D)")
        print(f"{'-'*50}")
        runner.run_experiment(
            f'nm_timing_{func_name}', func, bounds, n_dim, max_fes,
            configs, n_runs
        )
        runner.print_comparison_table(f'nm_timing_{func_name}')


def run_v4_comparison(runner, n_dim=30, max_fes=30000, n_runs=30):
    """v4.0 vs v3.5 全函数对比"""
    print("\n" + "="*70)
    print("v4.0 vs v3.5 全函数对比")
    print("="*70)

    # v3.5 默认配置
    v35_configs = {
        'v3.5_default': {
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
            'phase1_ratio': 0.8, 'restart_interval': 10,
        },
    }

    # v4.0 改进配置
    v40_configs = {
        'v4.0_improved': {
            'use_shca': True, 'use_cgpsr': False, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
            'phase1_ratio': 0.7, 'restart_interval': 15,
        },
    }

    benchmarks = BenchmarkFunctions.get_all()

    for func_name, (func, bounds) in benchmarks.items():
        print(f"\n{'-'*50}")
        print(f"函数: {func_name} ({n_dim}D)")
        print(f"{'-'*50}")

        # 运行 v3.5
        print("\n  [v3.5]")
        runner.run_experiment(
            f'v4_compare_{func_name}_v35', func, bounds, n_dim, max_fes,
            v35_configs, n_runs, version='v3.5'
        )

        # 运行 v4.0
        print("\n  [v4.0]")
        runner.run_experiment(
            f'v4_compare_{func_name}_v40', func, bounds, n_dim, max_fes,
            v40_configs, n_runs, version='v4.0'
        )

        # 打印对比
        v35_metrics = runner.results[f'v4_compare_{func_name}_v35']['v3.5_default']['metrics']
        v40_metrics = runner.results[f'v4_compare_{func_name}_v40']['v4.0_improved']['metrics']

        print(f"\n  对比结果:")
        print(f"  {'配置':<15} {'Median':<15} {'Mean±Std':<20} {'Best':<15}")
        print(f"  {'-'*60}")
        print(f"  {'v3.5':<15} {v35_metrics['median']:<15.6e} "
              f"{v35_metrics['mean']:.2e}±{v35_metrics['std']:.2e}  "
              f"{v35_metrics['best']:<15.6e}")
        print(f"  {'v4.0':<15} {v40_metrics['median']:<15.6e} "
              f"{v40_metrics['mean']:.2e}±{v40_metrics['std']:.2e}  "
              f"{v40_metrics['best']:<15.6e}")

        # 计算提升倍数
        if v35_metrics['median'] > 0:
            improvement = v35_metrics['median'] / v40_metrics['median']
            print(f"  提升倍数: {improvement:.2f}x")


def run_engineering_benchmark(runner, n_runs=30):
    """工程问题对比"""
    print("\n" + "="*70)
    print("工程问题对比")
    print("="*70)

    # v3.5 默认配置
    v35_configs = {
        'v3.5_default': {
            'use_shca': True, 'use_cgpsr': True, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
            'phase1_ratio': 0.8, 'restart_interval': 10,
        },
    }

    # v4.0 改进配置
    v40_configs = {
        'v4.0_improved': {
            'use_shca': True, 'use_cgpsr': False, 'use_cas': True,
            'use_cauchy': True, 'use_nm': True, 'use_twophase': True,
            'phase1_ratio': 0.7, 'restart_interval': 15,
        },
    }

    problems = EngineeringProblems.get_classic()

    for prob_name, (func, bounds, n_dim) in problems.items():
        print(f"\n{'-'*50}")
        print(f"问题: {prob_name} ({n_dim}D)")
        print(f"{'-'*50}")

        max_fes = 10000 * n_dim  # 根据维度调整 FES

        # 运行 v3.5
        print("\n  [v3.5]")
        runner.run_experiment(
            f'eng_{prob_name}_v35', func, bounds, n_dim, max_fes,
            v35_configs, n_runs, version='v3.5'
        )

        # 运行 v4.0
        print("\n  [v4.0]")
        runner.run_experiment(
            f'eng_{prob_name}_v40', func, bounds, n_dim, max_fes,
            v40_configs, n_runs, version='v4.0'
        )

        # 打印对比
        v35_metrics = runner.results[f'eng_{prob_name}_v35']['v3.5_default']['metrics']
        v40_metrics = runner.results[f'eng_{prob_name}_v40']['v4.0_improved']['metrics']

        print(f"\n  对比结果:")
        print(f"  {'配置':<15} {'Median':<15} {'Best':<15}")
        print(f"  {'-'*40}")
        print(f"  {'v3.5':<15} {v35_metrics['median']:<15.6e} {v35_metrics['best']:<15.6e}")
        print(f"  {'v4.0':<15} {v40_metrics['median']:<15.6e} {v40_metrics['best']:<15.6e}")


def main():
    parser = argparse.ArgumentParser(description='PNO-GWO v4.0 实验框架')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['ablation', 'fes_split', 'restart', 'knn',
                                 'nm_timing', 'v4_compare', 'engineering', 'all'],
                        help='实验类型')
    parser.add_argument('--runs', type=int, default=10, help='独立运行次数')
    parser.add_argument('--dim', type=int, default=30, help='问题维度')
    parser.add_argument('--max_fes', type=int, default=30000, help='最大 FES')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    args = parser.parse_args()

    runner = ExperimentRunner(output_dir=args.output)
    print(f"PNO-GWO v4.0 实验框架")
    print(f"实验: {args.experiment} | 运行次数: {args.runs} | "
          f"维度: {args.dim}D | FES: {args.max_fes}")
    print(f"输出目录: {runner.output_dir}")

    exp = args.experiment
    if exp in ('ablation', 'all'):
        run_ablation(runner, args.dim, args.max_fes, args.runs)
    if exp in ('fes_split', 'all'):
        run_fes_sensitivity(runner, args.dim, args.max_fes, args.runs)
    if exp in ('restart', 'all'):
        run_restart_sensitivity(runner, args.dim, args.max_fes, args.runs)
    if exp in ('knn', 'all'):
        run_knn_sensitivity(runner, args.dim, args.max_fes, args.runs)
    if exp in ('nm_timing', 'all'):
        run_nm_timing(runner, args.dim, args.max_fes, args.runs)
    if exp in ('v4_compare', 'all'):
        run_v4_comparison(runner, args.dim, args.max_fes, args.runs)
    if exp in ('engineering', 'all'):
        run_engineering_benchmark(runner, args.runs)

    runner.save_results(exp)
    print("\n[OK] 实验完成")


if __name__ == '__main__':
    main()
