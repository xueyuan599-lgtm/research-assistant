"""
Grey Wolf Optimizer (GWO)
=========================
Mirjalili et al. (2014), Advances in Engineering Software

经典群智能算法，α-β-δ 等级结构包围猎物。
"""

import numpy as np


class GWO:
    """Grey Wolf Optimizer"""

    def __init__(self, n_pop=None, **kwargs):
        self.n_pop = n_pop  # None → auto (18*dim)
        self.name = 'GWO'

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False):
        lb, ub = bounds
        n_pop = self.n_pop if self.n_pop is not None else min(18 * n_dim, 200)

        # 初始化
        X = lb + np.random.rand(n_pop, n_dim) * (ub - lb)
        fitness = np.array([obj_func(x) for x in X])
        n_fes = n_pop
        convergence = []

        # 排序找 α-β-δ
        order = np.argsort(fitness)
        alpha_pos, alpha_fit = X[order[0]].copy(), fitness[order[0]]
        beta_pos, beta_fit = X[order[1]].copy(), fitness[order[1]]
        delta_pos, delta_fit = X[order[2]].copy(), fitness[order[2]]
        convergence.append(alpha_fit)

        T = max_fes // n_pop
        for t in range(T):
            if n_fes >= max_fes:
                break

            a = 2 - 2 * t / T  # 线性递减

            for i in range(n_pop):
                if n_fes >= max_fes:
                    break

                # α 包围
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - X[i])
                X1 = alpha_pos - A1 * D_alpha

                # β 包围
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_pos - X[i])
                X2 = beta_pos - A2 * D_beta

                # δ 包围
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_pos - X[i])
                X3 = delta_pos - A3 * D_delta

                X[i] = (X1 + X2 + X3) / 3
                X[i] = np.clip(X[i], lb, ub)
                fitness[i] = obj_func(X[i])
                n_fes += 1

            # 更新 α-β-δ
            order = np.argsort(fitness)
            if fitness[order[0]] < alpha_fit:
                alpha_pos, alpha_fit = X[order[0]].copy(), fitness[order[0]]
            if fitness[order[1]] < beta_fit:
                beta_pos, beta_fit = X[order[1]].copy(), fitness[order[1]]
            if fitness[order[2]] < delta_fit:
                delta_pos, delta_fit = X[order[2]].copy(), fitness[order[2]]

            convergence.append(alpha_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  GWO Iter {t:4d}/{T} | FES: {n_fes:5d} | Best: {alpha_fit:.6e}")

        return alpha_pos, alpha_fit, convergence