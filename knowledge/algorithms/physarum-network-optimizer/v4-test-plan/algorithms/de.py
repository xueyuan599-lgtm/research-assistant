"""
Differential Evolution (DE/rand/1/bin)
=======================================
Storn & Price (1997)

经典差分进化算法，rand/1/bin 变异策略。
"""

import numpy as np


class DE:
    """Differential Evolution (DE/rand/1/bin)"""

    def __init__(self, n_pop=None, F=0.5, CR=0.9, **kwargs):
        self.n_pop = n_pop
        self.F = F       # 缩放因子
        self.CR = CR     # 交叉概率
        self.name = 'DE'

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False):
        lb, ub = bounds
        n_pop = self.n_pop if self.n_pop is not None else max(10 * n_dim, 50)

        # 初始化
        X = lb + np.random.rand(n_pop, n_dim) * (ub - lb)
        fitness = np.array([obj_func(x) for x in X])
        n_fes = n_pop

        best_idx = np.argmin(fitness)
        best_x = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        T = max_fes // n_pop

        for t in range(T):
            if n_fes >= max_fes:
                break

            for i in range(n_pop):
                if n_fes >= max_fes:
                    break

                # === DE/rand/1/bin 变异 ===
                # 选择 3 个不同的随机索引
                idxs = list(range(n_pop))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)

                # 变异
                mutant = X[a] + self.F * (X[b] - X[c])
                mutant = np.clip(mutant, lb, ub)

                # 二项交叉
                j_rand = np.random.randint(n_dim)
                trial = X[i].copy()
                for j in range(n_dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # 贪心选择
                fit_trial = obj_func(trial)
                n_fes += 1

                if fit_trial <= fitness[i]:
                    X[i] = trial
                    fitness[i] = fit_trial
                    if fit_trial < best_fit:
                        best_x = trial.copy()
                        best_fit = fit_trial

            convergence.append(best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  DE  Iter {t:4d}/{T} | FES: {n_fes:5d} | Best: {best_fit:.6e}")

        return best_x, best_fit, convergence