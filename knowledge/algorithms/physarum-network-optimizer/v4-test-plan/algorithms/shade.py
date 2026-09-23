"""
SHADE: Success-History Adaptive Differential Evolution
=======================================================
Tanabe & Fukunaga (2013), CEC 2013

自适应 DE，用成功历史记忆体动态调整 F 和 CR。
"""

import numpy as np


class SHADE:
    """SHADE (Success-History Adaptive DE)"""

    def __init__(self, n_pop=None, H=100, **kwargs):
        self.n_pop = n_pop
        self.H = H          # 记忆体大小
        self.name = 'SHADE'

    @staticmethod
    def _weighted_lehmer_mean(values, weights, eps=1e-10):
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        return np.sum(w * v ** 2) / (np.sum(w * v) + eps)

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False):
        lb, ub = bounds
        N_init = self.n_pop if self.n_pop is not None else max(10 * n_dim, 50)
        H = self.H

        # 初始化记忆体
        M_F = np.full(H, 0.5)      # F 记忆
        M_CR = np.full(H, 0.5)     # CR 记忆
        k = 0                       # 记忆体写入位置

        # 初始化种群
        X = lb + np.random.rand(N_init, n_dim) * (ub - lb)
        fitness = np.array([obj_func(x) for x in X])
        n_fes = N_init

        best_idx = np.argmin(fitness)
        best_x = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        # 外部存档
        archive = []
        archive_max = N_init

        T = max_fes // N_init
        for t in range(T):
            if n_fes >= max_fes:
                break

            N = len(X)
            succ_F = []
            succ_CR = []
            succ_imp = []

            for i in range(N):
                if n_fes >= max_fes:
                    break

                # 从记忆体采样 F 和 CR
                r = np.random.randint(0, H)
                mu_F = M_F[r]
                mu_CR = M_CR[r]

                # 生成 F: Cauchy 分布截断
                F_i = np.clip(np.random.standard_cauchy() * 0.1 + mu_F, 0, 1)
                # 生成 CR: 正态分布截断
                CR_i = np.clip(np.random.normal(mu_CR, 0.1), 0, 1)

                # === DE/current-to-pbest/1 ===
                # 选择 p-best
                p = max(2, int(0.05 * N))  # top 5%
                sorted_idx = np.argsort(fitness)[:p]
                p_best_idx = np.random.choice(sorted_idx)
                p_best = X[p_best_idx]

                # 选择 r1 ≠ i
                r1 = np.random.randint(N)
                while r1 == i:
                    r1 = np.random.randint(N)

                # 选择 r2 ≠ i, 从种群 + 存档联合
                combined = list(range(N))
                if len(archive) > 0:
                    arch_idx = len(combined)
                    combined.extend(range(len(combined), len(combined) + len(archive)))
                    pool_size = N + len(archive)
                else:
                    pool_size = N

                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)

                # 变异
                if r2 < N:
                    x_r2 = X[r2]
                else:
                    x_r2 = archive[r2 - N]

                mutant = X[i] + F_i * (p_best - X[i]) + F_i * (X[r1] - x_r2)
                mutant = np.clip(mutant, lb, ub)

                # 二项交叉
                j_rand = np.random.randint(n_dim)
                trial = X[i].copy()
                for j in range(n_dim):
                    if np.random.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                # 贪心选择
                fit_trial = obj_func(trial)
                n_fes += 1

                if fit_trial <= fitness[i]:
                    # 存档旧解
                    if len(archive) < archive_max:
                        archive.append(X[i].copy())
                    else:
                        archive[np.random.randint(archive_max)] = X[i].copy()

                    # 记录成功参数
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
                M_F[k] = self._weighted_lehmer_mean(np.array(succ_F), weights)
                M_CR[k] = self._weighted_lehmer_mean(np.array(succ_CR), weights)
                k = (k + 1) % H

            convergence.append(best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  SHADE Iter {t:4d}/{T} | FES: {n_fes:5d} | Best: {best_fit:.6e} | Pop: {N}")

        return best_x, best_fit, convergence