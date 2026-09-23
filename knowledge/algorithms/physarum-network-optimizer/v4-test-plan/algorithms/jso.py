"""
jSO: improved jSO
==================
Brest et al. (2017) — CEC 2016 约束优化竞赛冠军

jSO 是 SHADE 的改进版，核心改动：
1. 自适应 F 缩放：成功时减小 F（exploit），失败时增大（explore）
2. 更激进的种群缩减策略
3. 改进的 p-best 选择（动态 p）
"""

import numpy as np


class JSO:
    """jSO"""

    def __init__(self, n_pop=None, H=100, **kwargs):
        self.n_pop = n_pop
        self.H = H
        self.name = 'jSO'

    @staticmethod
    def _lehmer_mean(values, weights, eps=1e-10):
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        return np.sum(w * v ** 2) / (np.sum(w * v) + eps)

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False):
        lb, ub = bounds
        N_init = self.n_pop if self.n_pop is not None else max(18 * n_dim, 100)
        N_min = 4
        H = self.H

        M_F = np.full(H, 0.3)    # jSO: 初始 F 较小
        M_CR = np.full(H, 0.8)   # jSO: 初始 CR 较高
        k = 0

        X = lb + np.random.rand(N_init, n_dim) * (ub - lb)
        fitness = np.array([obj_func(x) for x in X])
        n_fes = N_init

        best_idx = np.argmin(fitness)
        best_x = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        archive = []
        archive_max = N_init

        T = max_fes // N_init
        for t in range(T):
            if n_fes >= max_fes:
                break

            N = len(X)
            N_curr = round(N_min + (N_init - N_min) * (1 - n_fes / max_fes))
            N_curr = max(N_curr, N_min)

            succ_F, succ_CR, succ_imp = [], [], []

            for i in range(N):
                if n_fes >= max_fes:
                    break

                r = np.random.randint(0, H)
                F_i = np.clip(np.random.standard_cauchy() * 0.1 + M_F[r], 0, 1)
                CR_i = np.clip(np.random.normal(M_CR[r], 0.1), 0, 1)

                # jSO: 动态 p-best（从 0.2 线性减到 0.05）
                p = 0.2 - 0.15 * (n_fes / max_fes)
                p = max(p, 0.05)
                n_p = max(2, int(p * N))
                sorted_idx = np.argsort(fitness)[:n_p]
                p_best_idx = np.random.choice(sorted_idx)
                p_best = X[p_best_idx]

                r1 = np.random.randint(N)
                while r1 == i:
                    r1 = np.random.randint(N)

                pool_size = N + len(archive)
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)

                x_r2 = X[r2] if r2 < N else archive[r2 - N]

                # jSO: F 缩放因子（F_i 用于 p-best, F_i*0.8 用于差分）
                F_pbest = F_i
                F_diff = F_i * 0.8
                mutant = X[i] + F_pbest * (p_best - X[i]) + F_diff * (X[r1] - x_r2)

                # 二项交叉
                j_rand = np.random.randint(n_dim)
                trial = X[i].copy()
                for j in range(n_dim):
                    if np.random.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                trial = np.clip(trial, lb, ub)
                fit_trial = obj_func(trial)
                n_fes += 1

                if fit_trial <= fitness[i]:
                    if len(archive) < archive_max:
                        archive.append(X[i].copy())
                    else:
                        archive[np.random.randint(archive_max)] = X[i].copy()

                    succ_F.append(F_i)
                    succ_CR.append(CR_i)
                    succ_imp.append(abs(fitness[i] - fit_trial))

                    X[i] = trial
                    fitness[i] = fit_trial

                    if fit_trial < best_fit:
                        best_x = trial.copy()
                        best_fit = fit_trial

            if len(succ_F) > 0:
                weights = np.array(succ_imp)
                weights = weights / (weights.sum() + 1e-10)
                M_F[k] = self._lehmer_mean(np.array(succ_F), weights)
                M_CR[k] = self._lehmer_mean(np.array(succ_CR), weights)
                k = (k + 1) % H

            if N > N_curr:
                order = np.argsort(fitness)
                keep = order[:N_curr]
                X = X[keep]
                fitness = fitness[keep]

            convergence.append(best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  jSO Iter {t:4d}/{T} | FES: {n_fes:5d} | "
                      f"Best: {best_fit:.6e} | Pop: {len(X)}")

        return best_x, best_fit, convergence
