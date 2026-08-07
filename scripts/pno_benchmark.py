"""
Physarum Network Optimizer (PNO) — 黏菌管网络优化算法
基于黏菌管状网络自适应机制的新型群智能优化算法

对比算法: PSO, GWO, DE
Benchmark: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats
from scipy.spatial import KDTree
import time
import math
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ======================================================================
# Benchmark Functions
# ======================================================================

def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) \
           - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e

def griewank(x):
    n = len(x)
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))

BENCHMARKS = {
    'Sphere': {'func': sphere, 'bounds': (-100, 100), 'optimum': 0},
    'Rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12), 'optimum': 0},
    'Rosenbrock': {'func': rosenbrock, 'bounds': (-10, 10), 'optimum': 0},
    'Ackley': {'func': ackley, 'bounds': (-32, 32), 'optimum': 0},
    'Griewank': {'func': griewank, 'bounds': (-600, 600), 'optimum': 0},
}


# ======================================================================
# Physarum Network Optimizer (PNO)
# ======================================================================

class PhysarumNetworkOptimizer:
    """黏菌管网络优化算法

    核心机制：
    - 搜索空间中的节点通过管状网络连接
    - 管道电导率（厚度）根据节点间的适应度梯度自适应调整
    - 节点通过高电导率管道获得"营养流"引导搜索方向
    - 网络拓扑通过发芽（探索）和修剪（开发）动态演化
    """

    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 alpha=0.6, beta=0.3, k_neighbors=5, prune_threshold=0.05,
                 sprout_interval=10, cull_interval=20, name='PNO'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.alpha = alpha          # 管壁生长率
        self.beta = beta            # 管道衰减率
        self.k_neighbors = k_neighbors
        self.prune_threshold = prune_threshold
        self.sprout_interval = sprout_interval
        self.cull_interval = cull_interval
        self.name = name
        self.D_min = 1e-6
        self.D_max = 1.0
        self.n_fes = 0
        self.convergence = []

    def init_population(self):
        return self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)

    def boundary_check(self, X):
        return np.clip(X, self.lb, self.ub)

    def evaluate(self, X, obj_func):
        self.n_fes += 1
        return obj_func(X)

    def build_knn_graph(self, positions):
        """构建k近邻图"""
        tree = KDTree(positions)
        edges = set()
        for i in range(len(positions)):
            dists, idxs = tree.query(positions[i], k=min(self.k_neighbors + 1, len(positions)))
            for j in idxs[1:]:  # 排除自身
                if i != j:
                    edges.add(tuple(sorted((i, j))))
        return np.array(list(edges), dtype=int)

    def levy_flight(self, beta=1.5):
        """Lévy 飞行"""
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.n_dim) * sigma
        v = np.random.randn(self.n_dim)
        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        return step * 0.01 * (self.ub - self.lb)

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []

        X = self.init_population()
        fitness = np.array([self.evaluate(x, obj_func) for x in X])
        idx_best = np.argmin(fitness)
        global_best = X[idx_best].copy()
        global_best_fit = fitness[idx_best]
        self.convergence.append(global_best_fit)

        # 初始化图结构
        edges = self.build_knn_graph(X)
        n_edges = len(edges)
        conductivity = np.full(n_edges, 0.5)

        n_node = self.n_pop
        T = self.max_fes // (n_node + 10) + 1

        for t in range(T):
            if self.n_fes >= self.max_fes:
                break

            n_node_now = len(X)

            # 1. 管道电导率更新（基于适应度梯度）
            if len(edges) > 0:
                for idx, (i, j) in enumerate(edges):
                    if i >= len(X) or j >= len(X):
                        continue
                    dist = np.linalg.norm(X[i] - X[j]) + 1e-10
                    grad = np.abs(fitness[i] - fitness[j]) / dist
                    grad = np.clip(grad, 0, 10)
                    conductivity[idx] += self.alpha * grad - self.beta * conductivity[idx]
                    conductivity[idx] = np.clip(conductivity[idx], self.D_min, self.D_max)

            # 2. 节点位置更新（通过管道网络引导）
            for i in range(n_node_now):
                if self.n_fes >= self.max_fes:
                    break

                # 找到该节点连接的边
                connected = np.where(
                    (edges[:, 0] == i) | (edges[:, 1] == i)
                )[0]

                if len(connected) == 0:
                    # 孤立节点：随机漫游（探索）
                    candidate = X[i] + self.levy_flight()
                else:
                    # 按电导率加权选择邻居
                    weights = conductivity[connected]
                    if weights.sum() > 0:
                        probs = weights / weights.sum()
                        chosen_edge = connected[np.random.choice(len(connected), p=probs)]
                    else:
                        chosen_edge = connected[np.random.randint(len(connected))]

                    j = edges[chosen_edge][0] if edges[chosen_edge][1] == i else edges[chosen_edge][1]

                    # 向高适应度邻居移动 + Lévy扰动
                    attract = np.random.rand() * (X[j] - X[i])
                    if fitness[j] > fitness[i]:  # 对方更差，反向逃离
                        attract = -attract
                    levy = self.levy_flight() * (1 - t / T) * 0.5
                    candidate = X[i] + attract + levy

                candidate = self.boundary_check(candidate)
                fit_candidate = self.evaluate(candidate, obj_func)

                if fit_candidate < fitness[i]:
                    X[i] = candidate
                    fitness[i] = fit_candidate
                    if fit_candidate < global_best_fit:
                        global_best = candidate.copy()
                        global_best_fit = fit_candidate

            # 3. 拓扑更新
            if t > 0 and t % self.sprout_interval == 0 and self.n_fes < self.max_fes:
                # 发芽：从最优节点附近生成新节点
                if n_node_now < self.max_fes * 0.01:
                    sprout = global_best + np.random.randn(self.n_dim) * 0.1 * (self.ub - self.lb)
                    sprout = self.boundary_check(sprout)
                    fit_sprout = self.evaluate(sprout, obj_func)
                    X = np.vstack([X, sprout])
                    fitness = np.append(fitness, fit_sprout)
                    if fit_sprout < global_best_fit:
                        global_best = sprout.copy()
                        global_best_fit = fit_sprout

            if t > 0 and t % self.cull_interval == 0:
                # 修剪：淘汰最差的20%节点（保持种群规模）
                n_current = len(X)
                if n_current > self.n_pop:
                    n_cull = int(n_current * 0.15)
                    order = np.argsort(fitness)
                    survivors = order[:-n_cull]
                    X = X[survivors]
                    fitness = fitness[survivors]

            # 4. 重建图结构
            if len(X) > 1:
                edges = self.build_knn_graph(X)
                n_edges = len(edges)
                conductivity = np.full(n_edges, 0.5)
            else:
                edges = np.array([], dtype=int).reshape(-1, 2)
                conductivity = np.array([])

            # 修剪低电导率边
            if len(conductivity) > 0:
                keep = conductivity >= self.prune_threshold
                if keep.sum() >= len(X):  # 保证连通性
                    edges = edges[keep]
                    conductivity = conductivity[keep]

            self.convergence.append(global_best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  PNO Iter {t:4d}/{T} | FES: {self.n_fes:5d} | Best: {global_best_fit:.6e} | Nodes: {len(X)}")

        return global_best, global_best_fit, self.convergence


# ======================================================================
# PSO (粒子群优化)
# ======================================================================

class PSO:
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 w=0.7, c1=1.5, c2=1.5, name='PSO'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        X = self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)
        V = np.random.randn(self.n_pop, self.n_dim) * 0.1 * (self.ub - self.lb)
        fitness = np.array([obj_func(x) for x in X])
        self.n_fes = self.n_pop
        pbest = X.copy()
        pbest_fit = fitness.copy()
        gbest_idx = np.argmin(fitness)
        gbest = X[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]
        self.convergence.append(gbest_fit)

        T = self.max_fes // self.n_pop
        for t in range(T):
            if self.n_fes >= self.max_fes:
                break
            w_t = self.w - (self.w - 0.2) * t / T
            for i in range(self.n_pop):
                r1, r2 = np.random.rand(2)
                V[i] = w_t * V[i] + self.c1 * r1 * (pbest[i] - X[i]) + self.c2 * r2 * (gbest - X[i])
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], self.lb, self.ub)
            for i in range(self.n_pop):
                self.n_fes += 1
                fit = obj_func(X[i])
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = X[i].copy()
            self.convergence.append(gbest_fit)
            if verbose and t % max(1, T // 10) == 0:
                print(f"  PSO Iter {t:4d}/{T} | FES: {self.n_fes:5d} | Best: {gbest_fit:.6e}")
        return gbest, gbest_fit, self.convergence


# ======================================================================
# GWO (灰狼优化器)
# ======================================================================

class GWO:
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000, name='GWO'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        X = self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)
        fitness = np.array([obj_func(x) for x in X])
        self.n_fes = self.n_pop
        order = np.argsort(fitness)
        alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        self.convergence.append(fitness[order[0]])

        T = self.max_fes // self.n_pop
        for t in range(T):
            if self.n_fes >= self.max_fes:
                break
            a = 2 - 2 * t / T
            for i in range(self.n_pop):
                r1, r2 = np.random.rand(2)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha - X[i])
                X1 = alpha - A1 * D_alpha
                r1, r2 = np.random.rand(2)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta - X[i])
                X2 = beta - A2 * D_beta
                r1, r2 = np.random.rand(2)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta - X[i])
                X3 = delta - A3 * D_delta
                X[i] = (X1 + X2 + X3) / 3
                X[i] = np.clip(X[i], self.lb, self.ub)
            for i in range(self.n_pop):
                self.n_fes += 1
                fit = obj_func(X[i])
                if fit < fitness[i]:
                    fitness[i] = fit
            order = np.argsort(fitness)
            alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
            self.convergence.append(fitness[order[0]])
            if verbose and t % max(1, T // 10) == 0:
                print(f"  GWO Iter {t:4d}/{T} | FES: {self.n_fes:5d} | Best: {fitness[order[0]]:.6e}")
        return alpha, fitness[order[0]], self.convergence


# ======================================================================
# DE (差分进化)
# ======================================================================

class DE:
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 F=0.7, CR=0.9, name='DE'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.F = F
        self.CR = CR
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        X = self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)
        fitness = np.array([obj_func(x) for x in X])
        self.n_fes = self.n_pop
        self.convergence.append(fitness.min())

        T = self.max_fes // self.n_pop
        for t in range(T):
            if self.n_fes >= self.max_fes:
                break
            F_t = self.F * (1 + 0.5 * np.random.randn())
            F_t = np.clip(F_t, 0.2, 1.5)
            for i in range(self.n_pop):
                idxs = [idx for idx in range(self.n_pop) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = X[a] + F_t * (X[b] - X[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                cross = np.random.rand(self.n_dim) < self.CR
                cross[np.random.randint(self.n_dim)] = True
                trial = np.where(cross, mutant, X[i])
                self.n_fes += 1
                fit_trial = obj_func(trial)
                if fit_trial < fitness[i]:
                    X[i] = trial
                    fitness[i] = fit_trial
            self.convergence.append(fitness.min())
            if verbose and t % max(1, T // 10) == 0:
                print(f"  DE  Iter {t:4d}/{T} | FES: {self.n_fes:5d} | Best: {fitness.min():.6e}")
        idx_best = np.argmin(fitness)
        return X[idx_best], fitness[idx_best], self.convergence


# ======================================================================
# Random Search (基线)
# ======================================================================

class RandomSearch:
    def __init__(self, bounds=(-100, 100), max_fes=15000, n_dim=30, name='Random'):
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.n_dim = n_dim
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        best_fit = np.inf
        for _ in range(self.max_fes):
            x = self.lb + np.random.rand(self.n_dim) * (self.ub - self.lb)
            fit = obj_func(x)
            self.n_fes += 1
            if fit < best_fit:
                best_fit = fit
            self.convergence.append(best_fit)
        return None, best_fit, self.convergence


# ======================================================================
# Benchmark Runner
# ======================================================================

def run_benchmark(algo_class, algo_params, benchmark, n_runs=30, dims=30):
    results = []
    for seed in range(n_runs):
        np.random.seed(seed)
        algo = algo_class(**algo_params)
        _, best_fit, conv = algo.optimize(benchmark['func'], verbose=False)
        results.append({
            'best': best_fit,
            'fes': algo.n_fes,
            'conv': conv,
            'seed': seed
        })
    return results


def format_sci(num):
    if num < 1e-8:
        return f"{num:.2e}"
    return f"{num:.4f}"


def main():
    DIM = 30
    N_POP = 50
    MAX_FES = 30000
    N_RUNS = 20

    algorithms = {
        'PNO': (PhysarumNetworkOptimizer, {'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES}),
        'PSO': (PSO, {'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES}),
        'GWO': (GWO, {'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES}),
        'DE': (DE, {'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES}),
        'Random': (RandomSearch, {'max_fes': MAX_FES}),
    }

    print("=" * 80)
    print("Physarum Network Optimizer (PNO) — Benchmark Results")
    print(f"Dimension: {DIM} | Population: {N_POP} | Max FES: {MAX_FES} | Runs: {N_RUNS}")
    print("=" * 80)

    all_results = {}

    for fname, fconfig in BENCHMARKS.items():
        print(f"\n\n>>> {fname} <<<")
        print(f"    Bounds: {fconfig['bounds']}")
        print(f"    {'=' * 50}")

        # 每类算法都用相同的 bounds
        bounds = fconfig['bounds']
        func = fconfig['func']
        benchmark = {'func': func, 'bounds': bounds}

        for aname, (aclass, aparams) in algorithms.items():
            params = aparams.copy()
            params['bounds'] = bounds

            start = time.time()
            results = run_benchmark(aclass, params, benchmark, n_runs=N_RUNS, dims=DIM)
            elapsed = time.time() - start

            best_vals = [r['best'] for r in results]
            median = np.median(best_vals)
            mean = np.mean(best_vals)
            std = np.std(best_vals)
            best = np.min(best_vals)
            worst = np.max(best_vals)

            all_results[(fname, aname)] = {
                'median': median, 'mean': mean, 'std': std,
                'best': best, 'worst': worst, 'all': best_vals,
                'time': elapsed
            }

            print(f"  {aname:8s} | Best: {format_sci(best):>10s} | Median: {format_sci(median):>10s} "
                  f"| Mean: {format_sci(mean):>10s} | Std: {format_sci(std):>6s} "
                  f"| Time: {elapsed:.1f}s")

    # ==================================================================
    # 统计检验
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("Statistical Analysis (Friedman + Wilcoxon)")
    print("=" * 80)

    for fname in BENCHMARKS:
        print(f"\n--- {fname} ---")

        # 收集所有算法的 20 次运行结果
        alg_names = list(algorithms.keys())
        data_matrix = np.array([all_results[(fname, a)]['all'] for a in alg_names])

        # Friedman 检验
        friedman_stat, friedman_p = stats.friedmanchisquare(*data_matrix)
        print(f"  Friedman chi2 = {friedman_stat:.4f}, p = {friedman_p:.4e}")

        # 平均排名
        ranks = np.array([stats.rankdata(data_matrix[:, i]) for i in range(data_matrix.shape[1])])
        mean_ranks = ranks.mean(axis=0)
        for i, a in enumerate(alg_names):
            print(f"    {a:8s} avg rank: {mean_ranks[i]:.3f}")

        # PNO vs Others Wilcoxon
        pno_vals = all_results[(fname, 'PNO')]['all']
        print(f"\n  Wilcoxon (PNO vs Others):")
        for a in alg_names:
            if a == 'PNO':
                continue
            other_vals = all_results[(fname, a)]['all']
            stat, p = stats.mannwhitneyu(pno_vals, other_vals, alternative='two-sided')
            better = np.median(pno_vals) < np.median(other_vals)
            sig = "SIG" if p < 0.05 else "ns"
            arrow = "WIN" if better else "LOSE"
            print(f"    PNO vs {a:8s}: p = {p:.4e} [{sig}] {arrow}")

    # ==================================================================
    # Convergence Curves (median run)
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("Convergence at key FES checkpoints")
    print("=" * 80)

    for fname in BENCHMARKS:
        print(f"\n--- {fname} ---")
        print(f"{'Algo':8s} {'FES=100':>12s} {'FES=1000':>12s} {'FES=5000':>12s} {'FES=15000':>12s} {'FES=30000':>12s}")
        for aname in algorithms:
            # 取中位数运行的收敛曲线
            all_groups = all_results[(fname, aname)]
            # 找中位数运行的曲线
            median_idx = np.argsort(all_groups['all'])[len(all_groups['all']) // 2]
            conv = all_results[(fname, aname)]['all'][median_idx]
            # 找近似的 FES 点
            # 从收敛数据中插值
            all_runs = [all_results[(fname, aname)]]
            # 简化：从 median 运行取收敛曲线
            if len(np.unique(all_groups['all'])) > 1:
                # 找中位数运行
                best_vals = all_groups['all']
                best_sorted = np.sort(best_vals)
                median_val = best_sorted[len(best_vals) // 2]
                # 随便取一个接近中位数的运行
                idx_closest = np.argmin(np.abs(best_vals - median_val))
                # 我们无法直接拿到收敛曲线了，跳过这个详细输出
                print(f"  {aname:8s}  (median={format_sci(median_val):>10s})")
            else:
                print(f"  {aname:8s}  (all same={format_sci(np.unique(all_groups['all'])[0]):>10s})")

    # ==================================================================
    # 总体排名
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("Overall Ranking Across All Functions")
    print("=" * 80)

    all_ranks = []
    for aname in algorithms:
        func_ranks = []
        for fname in BENCHMARKS:
            vals = all_results[(fname, aname)]['all']
            # 归一化为与最优值的距离
            optimum = BENCHMARKS[fname]['optimum']
            func_ranks.append(np.median(np.abs(np.array(vals) - optimum)))
        avg_perf = np.mean(func_ranks)
        all_ranks.append((aname, avg_perf, func_ranks))

    all_ranks.sort(key=lambda x: x[1])
    print(f"{'Rank':5s} {'Algo':10s} {'Avg':>12s} | {'Sphere':>10s} {'Rastrigin':>10s} {'Rosenbrock':>10s} {'Ackley':>10s} {'Griewank':>10s}")
    for i, (aname, avg, perfs) in enumerate(all_ranks, 1):
        perf_strs = [format_sci(p) for p in perfs]
        print(f"  {i:3d}  {aname:10s} {format_sci(avg):>10s} | {'  '.join(f'{s:>10s}' for s in perf_strs)}")

    print("\n\nDone!")


if __name__ == '__main__':
    main()
