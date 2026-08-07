"""
PNO-GWO v3.0 vs v2.0 vs GWO vs DE vs PSO 基准测试
5 个经典函数: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank
设置: 30D, FES=30000, 10 runs
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
from collections import OrderedDict
from algorithm import PhysarumNetworkOptimizer as PNOv3
from algorithm_v2 import PhysarumNetworkOptimizer as PNOv2

# ================================================================
# 测试函数
# ================================================================
def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def ackley(x):
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

def griewank(x):
    n = len(x)
    return 1 + np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))

# ================================================================
# 对比算法
# ================================================================
class GWO:
    def __init__(self, n_pop=50):
        self.name = 'GWO'
        self.n_pop = n_pop
    def optimize(self, obj_func, n_dim, bounds, max_fes, verbose=False):
        lb, ub = bounds
        X = lb + np.random.rand(self.n_pop, n_dim) * (ub - lb)
        fit = np.array([obj_func(x) for x in X])
        n_fes = self.n_pop
        alpha_idx = np.argmin(fit)
        alpha_pos, alpha_fit = X[alpha_idx].copy(), fit[alpha_idx]
        beta_pos = X[np.argsort(fit)[1]].copy()
        delta_pos = X[np.argsort(fit)[2]].copy()
        conv = [alpha_fit]
        T = max_fes // self.n_pop
        for t in range(T):
            if n_fes >= max_fes:
                break
            a = 2 - 2 * t / T
            for i in range(self.n_pop):
                if n_fes >= max_fes:
                    break
                # GWO 包围
                r1, r2 = np.random.rand(2)
                A1, C1 = 2*a*r1-a, 2*r2
                D_a = np.abs(C1 * alpha_pos - X[i])
                X1 = alpha_pos - A1 * D_a
                r1, r2 = np.random.rand(2)
                A2, C2 = 2*a*r1-a, 2*r2
                D_b = np.abs(C2 * beta_pos - X[i])
                X2 = beta_pos - A2 * D_b
                r1, r2 = np.random.rand(2)
                A3, C3 = 2*a*r1-a, 2*r2
                D_d = np.abs(C3 * delta_pos - X[i])
                X3 = delta_pos - A3 * D_d
                X[i] = np.clip((X1+X2+X3)/3, lb, ub)
                fit[i] = obj_func(X[i])
                n_fes += 1
                if fit[i] < alpha_fit:
                    alpha_fit = fit[i]
                    alpha_pos = X[i].copy()
            # 更新 beta, delta
            order = np.argsort(fit)
            alpha_pos, alpha_fit = X[order[0]].copy(), fit[order[0]]
            beta_pos = X[order[1]].copy()
            delta_pos = X[order[2]].copy()
            conv.append(alpha_fit)
        return alpha_pos, alpha_fit, conv

class DE:
    def __init__(self, n_pop=50, F=0.7, CR=0.9):
        self.name = 'DE'
        self.n_pop = n_pop
        self.F, self.CR = F, CR
    def optimize(self, obj_func, n_dim, bounds, max_fes, verbose=False):
        lb, ub = bounds
        X = lb + np.random.rand(self.n_pop, n_dim) * (ub - lb)
        fit = np.array([obj_func(x) for x in X])
        n_fes = self.n_pop
        best_idx = np.argmin(fit)
        conv = [fit[best_idx]]
        T = max_fes // self.n_pop
        for t in range(T):
            if n_fes >= max_fes:
                break
            for i in range(self.n_pop):
                if n_fes >= max_fes:
                    break
                ids = [j for j in range(self.n_pop) if j != i]
                a, b, c = np.random.choice(ids, 3, replace=False)
                j_rand = np.random.randint(n_dim)
                mutant = X[a] + self.F * (X[b] - X[c])
                trial = np.array([mutant[d] if np.random.rand() < self.CR or d == j_rand else X[i, d] for d in range(n_dim)])
                trial = np.clip(trial, lb, ub)
                ftrial = obj_func(trial)
                n_fes += 1
                if ftrial < fit[i]:
                    X[i] = trial
                    fit[i] = ftrial
            best_idx = np.argmin(fit)
            conv.append(fit[best_idx])
        return X[best_idx], fit[best_idx], conv

class PSO:
    def __init__(self, n_pop=50, w=0.7, c1=1.5, c2=1.5):
        self.name = 'PSO'
        self.n_pop = n_pop
        self.w, self.c1, self.c2 = w, c1, c2
    def optimize(self, obj_func, n_dim, bounds, max_fes, verbose=False):
        lb, ub = bounds
        span = ub - lb
        X = lb + np.random.rand(self.n_pop, n_dim) * span
        V = np.random.uniform(-1, 1, (self.n_pop, n_dim)) * 0.1 * span
        fit = np.array([obj_func(x) for x in X])
        n_fes = self.n_pop
        pbest = X.copy()
        pbest_fit = fit.copy()
        gbest_idx = np.argmin(fit)
        gbest = X[gbest_idx].copy()
        gbest_fit = fit[gbest_idx]
        conv = [gbest_fit]
        T = max_fes // self.n_pop
        for t in range(T):
            if n_fes >= max_fes:
                break
            w_t = self.w - (self.w - 0.2) * t / T
            for i in range(self.n_pop):
                if n_fes >= max_fes:
                    break
                r1, r2 = np.random.rand(n_dim), np.random.rand(n_dim)
                V[i] = w_t * V[i] + self.c1 * r1 * (pbest[i] - X[i]) + self.c2 * r2 * (gbest - X[i])
                X[i] = np.clip(X[i] + V[i], lb, ub)
                fit[i] = obj_func(X[i])
                n_fes += 1
                if fit[i] < pbest_fit[i]:
                    pbest_fit[i] = fit[i]
                    pbest[i] = X[i].copy()
                if fit[i] < gbest_fit:
                    gbest_fit = fit[i]
                    gbest = X[i].copy()
            conv.append(gbest_fit)
        return gbest, gbest_fit, conv


# ================================================================
# 新增对比算法
# ================================================================

class WOA:
    """Whale Optimization Algorithm (Mirjalili, 2016)"""
    def __init__(self, n_pop=50):
        self.name = 'WOA'
        self.n_pop = n_pop
    def optimize(self, obj_func, n_dim, bounds, max_fes, verbose=False):
        lb, ub = bounds
        X = lb + np.random.rand(self.n_pop, n_dim) * (ub - lb)
        fit = np.array([obj_func(x) for x in X])
        n_fes = self.n_pop
        best_idx = np.argmin(fit)
        best_pos = X[best_idx].copy()
        best_fit = fit[best_idx]
        conv = [best_fit]
        T = max_fes // self.n_pop
        for t in range(T):
            if n_fes >= max_fes: break
            a = 2 - 2 * t / T
            a2 = -1 + t * (-1) / T
            for i in range(self.n_pop):
                if n_fes >= max_fes: break
                r = np.random.rand()
                p = np.random.rand()
                A = 2 * a * r - a
                C = 2 * np.random.rand()
                l = (a2 - 1) * np.random.rand() + 1
                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * best_pos - X[i])
                        X[i] = best_pos - A * D
                    else:
                        rand_idx = np.random.randint(len(X))
                        X_rand = X[rand_idx]
                        D = np.abs(C * X_rand - X[i])
                        X[i] = X_rand - A * D
                else:
                    D_prime = np.abs(best_pos - X[i])
                    X[i] = D_prime * np.exp(0.5 * l) * np.cos(2 * np.pi * l) + best_pos
                X[i] = np.clip(X[i], lb, ub)
                fit[i] = obj_func(X[i]); n_fes += 1
                if fit[i] < best_fit:
                    best_fit = fit[i]; best_pos = X[i].copy()
            conv.append(best_fit)
        return best_pos, best_fit, conv


class HHO:
    """Harris Hawks Optimization (Heidari et al., 2019)"""
    def __init__(self, n_pop=50):
        self.name = 'HHO'
        self.n_pop = n_pop
    def optimize(self, obj_func, n_dim, bounds, max_fes, verbose=False):
        lb, ub = bounds
        X = lb + np.random.rand(self.n_pop, n_dim) * (ub - lb)
        fit = np.array([obj_func(x) for x in X])
        n_fes = self.n_pop
        best_idx = np.argmin(fit)
        rabbit = X[best_idx].copy()
        rabbit_fit = fit[best_idx]
        conv = [rabbit_fit]
        T = max_fes // self.n_pop
        for t in range(T):
            if n_fes >= max_fes: break
            E1 = 2 * (1 - t / T)
            for i in range(self.n_pop):
                if n_fes >= max_fes: break
                E0 = 2 * np.random.rand() - 1
                E = E1 * E0
                r = np.random.rand()
                J = 2 * (1 - np.random.rand())
                if abs(E) >= 1:
                    # exploration
                    rand_idx = np.random.randint(len(X))
                    if r < 0.5:
                        X[i] = X[rand_idx] - np.random.rand() * abs(X[rand_idx] - 2 * np.random.rand() * X[i])
                    else:
                        X[i] = (rabbit - X[rand_idx]) - np.random.rand() * (lb + np.random.rand() * (ub - lb))
                else:
                    if r >= 0.5 and abs(E) >= 0.5:
                        X[i] = rabbit - E * abs(J * rabbit - X[i])
                    elif r >= 0.5 and abs(E) < 0.5:
                        X[i] = rabbit - E * abs(rabbit - X[i])
                    elif r < 0.5 and abs(E) >= 0.5:
                        Y = rabbit - E * abs(J * rabbit - X[i])
                        if obj_func(Y) < fit[i]:
                            X[i] = Y; n_fes += 1
                        else:
                            Z = Y + np.random.rand(n_dim) * (ub - lb)
                            if obj_func(Z) < fit[i]:
                                X[i] = Z; n_fes += 1
                    elif r < 0.5 and abs(E) < 0.5:
                        Y = rabbit - E * abs(rabbit - X[i])
                        if obj_func(Y) < fit[i]:
                            X[i] = Y; n_fes += 1
                        else:
                            Z = Y + np.random.rand(n_dim) * (ub - lb)
                            if obj_func(Z) < fit[i]:
                                X[i] = Z; n_fes += 1
                X[i] = np.clip(X[i], lb, ub)
                fit[i] = obj_func(X[i]); n_fes += 1
                if fit[i] < rabbit_fit:
                    rabbit_fit = fit[i]; rabbit = X[i].copy()
            conv.append(rabbit_fit)
        return rabbit, rabbit_fit, conv


class SHADE:
    """Success-History Adaptive Differential Evolution (Tanabe & Fukunaga, 2013)
    简化实现: current-to-pbest/1 + 成功历史参数自适应
    """
    def __init__(self, n_pop=50, H=100):
        self.name = 'SHADE'
        self.n_pop = n_pop
        self.H = H
    def optimize(self, obj_func, n_dim, bounds, max_fes, verbose=False):
        lb, ub = bounds
        N = self.n_pop
        X = lb + np.random.rand(N, n_dim) * (ub - lb)
        fit = np.array([obj_func(x) for x in X])
        n_fes = N
        # 初始化记忆体
        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        k = 0
        archive = []
        archive_max = N
        best_idx = np.argmin(fit)
        conv = [fit[best_idx]]
        T = max_fes // N
        for t in range(T):
            if n_fes >= max_fes: break
            S_F, S_CR, delta = [], [], []
            for i in range(N):
                if n_fes >= max_fes: break
                # 采样 F, CR
                r = np.random.randint(self.H)
                CR = np.clip(M_CR[r] + 0.1 * np.random.randn(), 0, 1)
                F = np.clip(M_F[r] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0: F = np.clip(M_F[r] + 0.1 * np.random.standard_cauchy(), 0, 1)
                # pbest 选择
                p = np.random.rand() * 0.2
                pN = max(2, int(N * p))
                sorted_idx = np.argsort(fit)
                pbest_idx = sorted_idx[np.random.randint(pN)]
                x_pbest = X[pbest_idx]
                # 从种群+存档选 r1, r2
                pool = list(range(N)) + list(range(N, N + len(archive)))
                r1 = np.random.choice(N)
                while r1 == i: r1 = np.random.randint(N)
                r2 = np.random.choice(len(pool))
                while pool[r2] == i or (len(archive) > 0 and pool[r2] >= N and pool[r2] - N >= len(archive)):
                    r2 = np.random.choice(len(pool))
                if pool[r2] >= N:
                    x_r2 = archive[pool[r2] - N]
                else:
                    x_r2 = X[pool[r2]]
                # 变异: current-to-pbest/1
                mutant = X[i] + F * (x_pbest - X[i]) + F * (X[r1] - x_r2)
                # 交叉
                j_rand = np.random.randint(n_dim)
                trial = np.array([mutant[d] if np.random.rand() < CR or d == j_rand else X[i, d] for d in range(n_dim)])
                trial = np.clip(trial, lb, ub)
                ftrial = obj_func(trial); n_fes += 1
                if ftrial < fit[i]:
                    archive.append(X[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    S_F.append(F); S_CR.append(CR)
                    delta.append(fit[i] - ftrial)
                    X[i] = trial; fit[i] = ftrial
            # 更新记忆体
            if len(S_F) > 0:
                w = np.array(delta) / sum(delta)
                M_F[k] = sum(w[i]**2 * S_F[i] for i in range(len(S_F))) / (sum(w[i] * S_F[i] for i in range(len(S_F))) + 1e-10)
                M_CR[k] = sum(w[i]**2 * S_CR[i] for i in range(len(S_CR))) / (sum(w[i] * S_CR[i] for i in range(len(S_CR))) + 1e-10)
                k = (k + 1) % self.H
            best_idx = np.argmin(fit)
            conv.append(fit[best_idx])
        return X[best_idx], fit[best_idx], conv


class SSA:
    """Salp Swarm Algorithm (Mirjalili et al., 2017)"""
    def __init__(self, n_pop=50):
        self.name = 'SSA'
        self.n_pop = n_pop
    def optimize(self, obj_func, n_dim, bounds, max_fes, verbose=False):
        lb, ub = bounds
        X = lb + np.random.rand(self.n_pop, n_dim) * (ub - lb)
        fit = np.array([obj_func(x) for x in X])
        n_fes = self.n_pop
        best_idx = np.argmin(fit)
        best_pos = X[best_idx].copy()
        conv = [fit[best_idx]]
        T = max_fes // self.n_pop
        for t in range(T):
            if n_fes >= max_fes: break
            c1 = 2 * np.exp(-(4 * t / T) ** 2)
            for i in range(self.n_pop):
                if n_fes >= max_fes: break
                if i == 0:
                    # 领导者
                    r1, r2, r3 = np.random.rand(3)
                    if r3 < 0.5:
                        X[i] = best_pos + c1 * ((ub - lb) * r2 + lb)
                    else:
                        X[i] = best_pos - c1 * ((ub - lb) * r2 + lb)
                else:
                    # 追随者
                    X[i] = (X[i] + X[i-1]) / 2
                X[i] = np.clip(X[i], lb, ub)
                fit[i] = obj_func(X[i]); n_fes += 1
                if fit[i] < conv[-1]:
                    best_pos = X[i].copy()
                    conv[-1] = fit[i]
            best_idx = np.argmin(fit)
            if fit[best_idx] < fit[np.argmin([obj_func(x) if False else 0 for x in X])]:
                pass
            conv.append(fit[best_idx])
        best_idx = np.argmin(fit)
        return X[best_idx], fit[best_idx], conv

# ================================================================
# 基准测试配置
# ================================================================
BENCHMARKS = {
    'Sphere': {'func': sphere, 'bounds': (-100, 100), 'optimum': 0},
    'Rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12), 'optimum': 0},
    'Rosenbrock': {'func': rosenbrock, 'bounds': (-10, 10), 'optimum': 0},
    'Ackley': {'func': ackley, 'bounds': (-32, 32), 'optimum': 0},
    'Griewank': {'func': griewank, 'bounds': (-600, 600), 'optimum': 0},
}

N_DIM = 30
MAX_FES = 30000
N_RUNS = 10
N_POP = 50

# ================================================================
# 运行
# ================================================================
def run_trial(algo_class, algo_kwargs, func_cfg, seed, n_dim=N_DIM, max_fes=MAX_FES):
    np.random.seed(seed)
    lb, ub = func_cfg['bounds']

    # PNO 系列: n_dim 在 __init__ 传入
    if algo_class in [PNOv3, PNOv2]:
        algo = algo_class(n_dim=n_dim, bounds=(lb, ub), max_fes=max_fes, **algo_kwargs)
        best_x, best_fit, conv = algo.optimize(func_cfg['func'], verbose=False)
    else:
        algo = algo_class(**algo_kwargs)
        best_x, best_fit, conv = algo.optimize(
            func_cfg['func'], n_dim=n_dim,
            bounds=(lb, ub), max_fes=max_fes,
            verbose=False
        )
    return best_fit, conv

def run_all():
    results = {}  # {func_name: {algo_name: [fits]}}

    # 算法配置（使用 OrderedDict 保持输出顺序）
    algorithms = OrderedDict([
        ('PNO-GWO v3.0', {'class': PNOv3, 'kwargs': {'n_pop': N_POP}}),
        ('PNO-GWO v2.0', {'class': PNOv2, 'kwargs': {'n_pop': N_POP}}),
        ('SHADE',        {'class': SHADE, 'kwargs': {'n_pop': N_POP}}),
        ('GWO',          {'class': GWO, 'kwargs': {'n_pop': N_POP}}),
        ('WOA',          {'class': WOA, 'kwargs': {'n_pop': N_POP}}),
        ('HHO',          {'class': HHO, 'kwargs': {'n_pop': N_POP}}),
        ('SSA',          {'class': SSA, 'kwargs': {'n_pop': N_POP}}),
        ('DE',           {'class': DE, 'kwargs': {'n_pop': N_POP}}),
        ('PSO',          {'class': PSO, 'kwargs': {'n_pop': N_POP}}),
    ])

    for fname, fcfg in BENCHMARKS.items():
        print(f"\n{'='*60}")
        print(f"  {fname} (30D, FES={MAX_FES}, runs={N_RUNS})")
        print(f"{'='*60}")

        results[fname] = {}
        for aname, ainfo in algorithms.items():
            fits = []
            times = []
            for seed in range(N_RUNS):
                t0 = time.time()
                fit, conv = run_trial(ainfo['class'], ainfo['kwargs'], fcfg, seed)
                t1 = time.time()
                fits.append(fit)
                times.append(t1 - t0)

            median_fit = np.median(fits)
            mean_fit = np.mean(fits)
            std_fit = np.std(fits)
            best_fit = np.min(fits)
            worst_fit = np.max(fits)
            mean_time = np.mean(times)

            results[fname][aname] = {
                'fits': fits, 'median': median_fit, 'mean': mean_fit,
                'std': std_fit, 'best': best_fit, 'worst': worst_fit,
                'time': mean_time
            }

            print(f"  {aname:20s} | Best: {best_fit:.6e} | "
                  f"Median: {median_fit:.6e} | Mean: {mean_fit:.6e} | "
                  f"Std: {std_fit:.4e} | Time: {mean_time:.2f}s")

    return results, list(algorithms.keys())


def print_summary(results, algos):
    """打印汇总对比表"""
    print("\n\n")
    print("=" * 100)
    print("  PNO-GWO v3.0 vs 9 种启发式算法 — 汇总对比 (30D, FES=30000, median)")
    print("=" * 100)
    print(f"{'函数':15s}", end='')
    for a in algos:
        print(f"{a:20s}", end='')
    print()

    for fname in BENCHMARKS:
        print(f"{fname:15s}", end='')
        for a in algos:
            med = results[fname][a]['median']
            if med < 1e-10:
                print(f"{'<1e-10':>20s}", end='')
            elif med < 1:
                print(f"{med:.6e}".rjust(20), end='')
            else:
                print(f"{med:.4f}".rjust(20), end='')
        print()

    # 提升倍数
    print(f"\n{'提升倍数 (v3/v2)':-^80}")
    print(f"{'函数':15s}{'v2 Median':>15s}{'v3 Median':>15s}{'提升倍数':>15s}")
    for fname in BENCHMARKS:
        v2 = results[fname]['PNO-GWO v2.0']['median']
        v3 = results[fname]['PNO-GWO v3.0']['median']
        if v2 > 0 and v3 > 0:
            ratio = v2 / v3 if v3 > 0 else float('inf')
        else:
            ratio = float('inf') if v2 > 0 else 1.0
        print(f"{fname:15s}{v2:15.6e}{v3:15.6e}", end='')
        if ratio == float('inf'):
            print(f"{'∞ (v3→0)':>15s}")
        elif ratio > 1000:
            print(f"{ratio:.1e}".rjust(15))
        else:
            print(f"{ratio:.2f}x".rjust(15))


if __name__ == '__main__':
    print("PNO-GWO v3.0 基准测试")
    print(f"维度: {N_DIM}, FES: {MAX_FES}, Runs: {N_RUNS}")
    results, algos = run_all()
    print_summary(results, algos)
