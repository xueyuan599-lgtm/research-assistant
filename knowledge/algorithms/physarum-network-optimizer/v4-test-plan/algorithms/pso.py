"""
Particle Swarm Optimization (PSO)
==================================
Kennedy & Eberhart (1995)

经典粒子群优化，惯性权重线性递减。
"""

import numpy as np


class PSO:
    """Particle Swarm Optimization"""

    def __init__(self, n_pop=None, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, **kwargs):
        self.n_pop = n_pop
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.name = 'PSO'

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False):
        lb, ub = bounds
        n_pop = self.n_pop if self.n_pop is not None else min(18 * n_dim, 200)
        v_max = 0.2 * (ub - lb)

        # 初始化位置和速度
        X = lb + np.random.rand(n_pop, n_dim) * (ub - lb)
        V = np.random.uniform(-v_max, v_max, (n_pop, n_dim))
        fitness = np.array([obj_func(x) for x in X])
        n_fes = n_pop

        # 个体最优
        p_best = X.copy()
        p_best_fit = fitness.copy()

        # 全局最优
        g_idx = np.argmin(fitness)
        g_best = X[g_idx].copy()
        g_best_fit = fitness[g_idx]

        convergence = [g_best_fit]
        T = max_fes // n_pop

        for t in range(T):
            if n_fes >= max_fes:
                break

            # 惯性权重线性递减
            w = self.w_start - (self.w_start - self.w_end) * t / T

            for i in range(n_pop):
                if n_fes >= max_fes:
                    break

                r1, r2 = np.random.rand(n_dim), np.random.rand(n_dim)

                # 速度更新
                V[i] = (w * V[i]
                        + self.c1 * r1 * (p_best[i] - X[i])
                        + self.c2 * r2 * (g_best - X[i]))

                # 速度裁剪
                V[i] = np.clip(V[i], -v_max, v_max)

                # 位置更新
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lb, ub)

                fitness[i] = obj_func(X[i])
                n_fes += 1

                # 更新个体最优
                if fitness[i] < p_best_fit[i]:
                    p_best[i] = X[i].copy()
                    p_best_fit[i] = fitness[i]

                    # 更新全局最优
                    if fitness[i] < g_best_fit:
                        g_best = X[i].copy()
                        g_best_fit = fitness[i]

            convergence.append(g_best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  PSO Iter {t:4d}/{T} | FES: {n_fes:5d} | Best: {g_best_fit:.6e}")

        return g_best, g_best_fit, convergence