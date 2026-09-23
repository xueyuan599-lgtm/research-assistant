"""
Phase 3: 工程问题验证
使用改进的约束处理机制测试 v4.0
"""

import numpy as np
import sys
import os
import json
import time
from typing import Dict, List, Tuple

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_v4 import PhysarumNetworkOptimizer
from constraint_handler import ConstraintHandler, get_all_engineering_problems


class Phase3Engineer:
    """Phase 3 工程问题验证器"""

    def __init__(self, output_dir: str = 'outputs/pno_v4_experiments'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}

    def run_single_engineering(self, problem, n_runs: int = 30,
                              constraint_method: str = 'adaptive_penalty') -> dict:
        """运行单个工程问题"""
        print(f"\n  问题: {problem.name} ({problem.n_dim}D)")
        print(f"  约束处理: {constraint_method}")
        print(f"  运行次数: {n_runs}")

        # 约束处理器
        handler = ConstraintHandler(method=constraint_method)

        all_results = []
        all_obj_values = []
        all_violations = []
        all_feasible = []
        all_times = []

        for run_id in range(n_runs):
            seed = 42 + run_id
            np.random.seed(seed)

            # 调整 FES 根据维度
            max_fes = max(30000, 10000 * problem.n_dim)

            # 创建优化器（使用 v4.0 默认配置）
            # 转换 bounds 格式：从 [(min1, max1), ...] 到 (lb_array, ub_array)
            lb = np.array([b[0] for b in problem.bounds])
            ub = np.array([b[1] for b in problem.bounds])
            bounds_tuple = (lb, ub)

            pno = PhysarumNetworkOptimizer(
                n_dim=problem.n_dim,
                bounds=bounds_tuple,
                max_fes=max_fes,
                seed=seed
            )

            # 包装目标函数（包含约束处理）
            def constrained_obj(x):
                obj_val, violation = problem.evaluate(x)

                # 根据约束处理方法返回惩罚后的值
                if constraint_method == 'adaptive_penalty':
                    # 自适应惩罚
                    progress = pno.n_fes / max_fes
                    penalty_mult = 1.0 + 10.0 * progress
                    return obj_val + penalty_mult * violation
                elif constraint_method == 'feasibility_rule':
                    # 可行性规则：不可行解返回极大值
                    if violation > 0:
                        return 1e10 + violation
                    return obj_val
                elif constraint_method == 'epsilon_constraint':
                    # ε-约束
                    epsilon = 1e-6 * (0.95 ** (pno.n_fes / max_fes * 100))
                    if violation <= epsilon:
                        return obj_val
                    return obj_val + 1e6 * (violation - epsilon)
                else:
                    # 传统罚函数
                    return obj_val + 1e6 * violation

            # 运行优化
            start_time = time.time()
            try:
                best_x, best_f, convergence = pno.optimize(constrained_obj)
                elapsed = time.time() - start_time

                # 评估原始目标函数和约束违反
                obj_val, violation = problem.evaluate(best_x)
                is_feasible = violation == 0

                all_results.append({
                    'run_id': run_id,
                    'seed': seed,
                    'best_x': best_x.tolist(),
                    'obj_value': float(obj_val),
                    'violation': float(violation),
                    'is_feasible': is_feasible,
                    'time': elapsed,
                    'fes': pno.n_fes,
                })

                all_obj_values.append(obj_val)
                all_violations.append(violation)
                all_feasible.append(is_feasible)
                all_times.append(elapsed)

                if (run_id + 1) % 5 == 0:
                    feasible_count = sum(all_feasible)
                    print(f"    [{run_id+1}/{n_runs}] "
                          f"obj={obj_val:.4e}, viol={violation:.2e}, "
                          f"feasible={feasible_count}/{run_id+1}")

            except Exception as e:
                print(f"    [{run_id+1}/{n_runs}] ERROR: {e}")
                all_results.append({
                    'run_id': run_id,
                    'seed': seed,
                    'error': str(e),
                })

        # 计算统计
        feasible_obj = [v for v, f in zip(all_obj_values, all_feasible) if f]
        feasible_count = sum(all_feasible)

        if feasible_obj:
            stats = {
                'feasible_count': feasible_count,
                'feasible_rate': feasible_count / n_runs,
                'best_obj': float(min(feasible_obj)),
                'median_obj': float(np.median(feasible_obj)),
                'mean_obj': float(np.mean(feasible_obj)),
                'std_obj': float(np.std(feasible_obj)),
                'mean_time': float(np.mean(all_times)),
                'mean_violation': float(np.mean(all_violations)),
                'gap_to_optimum': problem.gap_to_optimum(min(feasible_obj)) if problem.known_optimum else None,
            }
        else:
            stats = {
                'feasible_count': 0,
                'feasible_rate': 0.0,
                'best_obj': None,
                'median_obj': None,
                'mean_obj': None,
                'std_obj': None,
                'mean_time': float(np.mean(all_times)),
                'mean_violation': float(np.mean(all_violations)),
                'gap_to_optimum': None,
            }

        return {
            'problem': problem.name,
            'n_dim': problem.n_dim,
            'constraint_method': constraint_method,
            'n_runs': n_runs,
            'stats': stats,
            'details': all_results,
        }

    def run_all_engineering(self, n_runs: int = 30,
                           constraint_methods: List[str] = None) -> dict:
        """运行所有工程问题"""
        if constraint_methods is None:
            constraint_methods = ['adaptive_penalty', 'feasibility_rule', 'epsilon_constraint']

        problems = get_all_engineering_problems()
        all_results = {}

        print("="*70)
        print("Phase 3: 工程问题验证")
        print("="*70)
        print(f"问题数: {len(problems)}")
        print(f"约束处理方法: {constraint_methods}")
        print(f"运行次数: {n_runs}")

        for prob_name, problem in problems.items():
            print(f"\n{'='*50}")
            print(f"问题: {prob_name}")
            print(f"{'='*50}")

            all_results[prob_name] = {}

            for method in constraint_methods:
                result = self.run_single_engineering(problem, n_runs, method)
                all_results[prob_name][method] = result

        return all_results

    def print_comparison_table(self, results: dict):
        """打印对比表"""
        print("\n" + "="*100)
        print("工程问题约束处理对比")
        print("="*100)

        # 表头
        print(f"\n{'问题':<15} {'方法':<20} {'可行率':<10} {'Best Obj':<15} {'Median Obj':<15} {'Gap%':<10}")
        print("-"*85)

        for prob_name, methods in results.items():
            for method_name, data in methods.items():
                stats = data['stats']
                feasible_rate = f"{stats['feasible_rate']*100:.1f}%"
                best_obj = f"{stats['best_obj']:.4e}" if stats['best_obj'] else "N/A"
                median_obj = f"{stats['median_obj']:.4e}" if stats['median_obj'] else "N/A"
                gap = f"{stats['gap_to_optimum']:.2f}%" if stats['gap_to_optimum'] else "N/A"

                print(f"{prob_name:<15} {method_name:<20} {feasible_rate:<10} "
                      f"{best_obj:<15} {median_obj:<15} {gap:<10}")
            print("-"*85)

    def save_results(self, results: dict, filename: str = None):
        """保存结果"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'phase3_engineering_{timestamp}.json'

        filepath = os.path.join(self.output_dir, filename)

        # 转换为可序列化格式
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable = {}
        for prob_name, methods in results.items():
            serializable[prob_name] = {}
            for method_name, data in methods.items():
                serializable[prob_name][method_name] = {
                    'problem': data['problem'],
                    'n_dim': int(data['n_dim']),
                    'constraint_method': data['constraint_method'],
                    'n_runs': int(data['n_runs']),
                    'stats': convert_to_serializable(data['stats']),
                    # 不保存 details 以减小文件大小
                }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {filepath}")
        return filepath


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3: 工程问题验证')
    parser.add_argument('--runs', '-n', type=int, default=30,
                        help='每个问题的运行次数 (默认: 30)')
    parser.add_argument('--methods', '-m', nargs='+',
                        default=['adaptive_penalty', 'feasibility_rule', 'epsilon_constraint'],
                        help='约束处理方法')
    parser.add_argument('--output', '-o', default='outputs/pno_v4_experiments',
                        help='输出目录')
    args = parser.parse_args()

    # 创建验证器
    tester = Phase3Engineer(output_dir=args.output)

    # 运行所有工程问题
    results = tester.run_all_engineering(n_runs=args.runs, constraint_methods=args.methods)

    # 打印对比表
    tester.print_comparison_table(results)

    # 保存结果
    tester.save_results(results)

    print("\n[OK] Phase 3 工程问题验证完成")


if __name__ == '__main__':
    main()
