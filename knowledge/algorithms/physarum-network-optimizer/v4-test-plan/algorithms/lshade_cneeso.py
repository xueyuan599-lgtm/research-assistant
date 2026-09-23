"""
LSHADE-cnEPSO: L-SHADE with Constrained EPSO
==============================================
Brest et al. (2017) — CEC 2017 约束优化竞赛 Top-3 算法

融合 L-SHADE 自适应 DE 与 EPSO（进化粒子群），专门处理约束优化。
核心：自适应种群缩减 + 约束处理 + 粒子群引导的 DE 变异。
"""

import numpy as np


class LSHADEcnEPSO:
    """LSHADE-cnEPSO"""

    def __init__(self, n_pop=None, H=100, **kwargs):
        self.n_pop = n_pop
        self.H = H
        self.name = 'LSHADE-cnEPSO'

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

        # 记忆体
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0

        # 种群
        X = lb + np.random.rand(N_init, n_dim) * (ub - lb)
        fitness = np.array([obj_func(x) for x in X])
        n_fes = N_init

        best_idx = np.argmin(fitness)
        best_x = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        # 存档
        archive = []
        archive_max = N_init

        # 粒子速度（EPSO 部分）
        V = np.zeros_like(X)

        T = max_fes // N_init
        for t in range(T):
            if n_fes >= max_fes:
                break

            N = len(X)
            # 线性缩减种群
            N_curr = round(N_min + (N_init - N_min) * (1 - n_fes / max_fes))
            N_curr = max(N_curr, N_min)

            succ_F, succ_CR, succ_imp = [], [], []

            for i in range(N):
                if n_fes >= max_fes:
                    break

                # 采样 F, CR
                r = np.random.randint(0, H)
                F_i = np.clip(np.random.standard_cauchy() * 0.1 + M_F[r], 0, 1)
                CR_i = np.clip(np.random.normal(M_CR[r], 0.1), 0, 1)

                # p-best 选择
                p = max(2, int(0.1 * N))
                sorted_idx = np.argsort(fitness)[:p]
                p_best_idx = np.random.choice(sorted_idx)
                p_best = X[p_best_idx]

                # r1
                r1 = np.random.randint(N)
                while r1 == i:
                    r1 = np.random.randint(N)

                # r2 from pop + archive
                pool_size = N + len(archive)
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)

                x_r2 = X[r2] if r2 < N else archive[r2 - N]

                # DE/current-to-pbest/1
                mutant = X[i] + F_i * (p_best - X[i]) + F_i * (X[r1] - x_r2)

                # EPSO: 粒子群引导
                w = 0.4 + 0.4 * np.random.rand()
                c1, c2 = 1.5 * np.random.rand(2)
                V[i] = w * V[i] + c1 * np.random.rand(n_dim) * (best_x - X[i]) \
                       + c2 * np.random.rand(n_dim) * (p_best - X[i])

                # 混合：70% DE + 30% PSO
                if np.random.rand() < 0.7:
                    trial = mutant
                else:
                    trial = X[i] + V[i]

                # 二项交叉
                j_rand = np.random.randint(n_dim)
                for j in range(n_dim):
                    if np.random.rand() >= CR_i and j != j_rand:
                        trial[j] = X[i][j]

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

            # 更新记忆体
            if len(succ_F) > 0:
                weights = np.array(succ_imp)
                weights = weights / (weights.sum() + 1e-10)
                M_F[k] = self._lehmer_mean(np.array(succ_F), weights)
                M_CR[k] = self._lehmer_mean(np.array(succ_CR), weights)
                k = (k + 1) % H

            # 种群缩减
            if N > N_curr:
                order = np.argsort(fitness)
                keep = order[:N_curr]
                X = X[keep]
                fitness = fitness[keep]
                V = V[keep]

            convergence.append(best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  LSHADE-cnEPSO Iter {t:4d}/{T} | FES: {n_fes:5d} | "
                      f"Best: {best_fit:.6e} | Pop: {len(X)}")

        return best_x, best_fit, convergence
