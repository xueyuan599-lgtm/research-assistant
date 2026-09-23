"""
EA4eig: EA with Eigenvector-based Crossover
=============================================
Uses eigenvector-based crossover for better exploration in high-dimensional
constrained optimization spaces. Strong performance on CEC 2017 benchmarks.

核心思想：用协方差矩阵的主方向（特征向量）引导交叉操作，
在高维空间中更高效地探索。
"""

import numpy as np


class EA4eig:
    """EA with Eigenvector-based crossover"""

    def __init__(self, n_pop=None, **kwargs):
        self.n_pop = n_pop
        self.name = 'EA4eig'

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False):
        lb, ub = bounds
        N = self.n_pop if self.n_pop is not None else max(10 * n_dim, 50)

        X = lb + np.random.rand(N, n_dim) * (ub - lb)
        fitness = np.array([obj_func(x) for x in X])
        n_fes = N

        best_idx = np.argmin(fitness)
        best_x = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        F_base = 0.5
        CR_base = 0.9
        eig_update_freq = max(10, n_dim)  # 每 N*D 次评估更新特征向量

        # 初始特征向量
        eig_vecs = np.eye(n_dim)
        eig_vals = np.ones(n_dim)

        T = max_fes // N
        for t in range(T):
            if n_fes >= max_fes:
                break

            # 定期更新特征向量
            if t % eig_update_freq == 0 and t > 0:
                try:
                    # 用 top-p 个体估计协方差
                    p = max(10, N // 4)
                    top_idx = np.argsort(fitness)[:p]
                    C = np.cov(X[top_idx].T) + 1e-6 * np.eye(n_dim)
                    eig_vals, eig_vecs = np.linalg.eigh(C)
                    # 排序（最大特征值在前）
                    order = np.argsort(eig_vals)[::-1]
                    eig_vals = eig_vals[order]
                    eig_vecs = eig_vecs[:, order]
                except np.linalg.LinAlgError:
                    pass

            succ_F, succ_CR, succ_imp = [], [], []

            for i in range(N):
                if n_fes >= max_fes:
                    break

                # 自适应 F 和 CR
                F_i = np.clip(F_base + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR_i = np.clip(CR_base + 0.1 * np.random.randn(), 0, 1)

                # 选择 r1, r2, r3
                idxs = list(range(N))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                # 特征向量空间中的变异
                # 将差分向量投影到特征空间
                diff = X[r1] - X[r2]
                # 在特征空间中缩放（大特征值方向更多探索）
                scale = 1.0 / (np.sqrt(np.abs(eig_vals) + 1e-10))
                scale = scale / (scale.max() + 1e-10)  # 归一化
                diff_eig = eig_vecs.T @ diff
                diff_eig *= scale
                diff_transformed = eig_vecs @ diff_eig

                mutant = X[r3] + F_i * diff_transformed

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
                    succ_F.append(F_i)
                    succ_CR.append(CR_i)
                    succ_imp.append(abs(fitness[i] - fit_trial))
                    X[i] = trial
                    fitness[i] = fit_trial
                    if fit_trial < best_fit:
                        best_x = trial.copy()
                        best_fit = fit_trial

            # 更新基准 F 和 CR
            if len(succ_F) > 0:
                w = np.array(succ_imp) + 1e-10
                w /= w.sum()
                F_base = np.clip(np.sum(w * np.array(succ_F)), 0.1, 0.9)
                CR_base = np.clip(np.sum(w * np.array(succ_CR)), 0.1, 0.95)

            convergence.append(best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  EA4eig Iter {t:4d}/{T} | FES: {n_fes:5d} | "
                      f"Best: {best_fit:.6e}")

        return best_x, best_fit, convergence
