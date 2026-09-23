"""
CMODE: Constrained Multi-Objective Differential Evolution
==========================================================
Zhang et al. (2008) — 专门处理约束的 DE 变体

核心思想：将约束优化转化为双目标优化（最小化 f(x) 和约束违反度），
用 Pareto 支配关系选择个体，平衡目标优化和约束满足。
"""

import numpy as np


class CMODE:
    """Constrained Multi-Objective DE"""

    def __init__(self, n_pop=None, **kwargs):
        self.n_pop = n_pop
        self.name = 'CMODE'

    @staticmethod
    def _constraint_violation(x, constraints):
        """计算约束违反度（如果有约束函数）"""
        if constraints is None:
            return 0.0
        total_violation = 0.0
        for g in constraints:
            violation = max(0, g(x))
            total_violation += violation
        return total_violation

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False, constraints=None):
        lb, ub = bounds
        N = self.n_pop if self.n_pop is not None else max(10 * n_dim, 50)

        X = lb + np.random.rand(N, n_dim) * (ub - lb)
        fitness = np.array([obj_func(x) for x in X])
        violation = np.array([self._constraint_violation(x, constraints) for x in X])
        n_fes = N

        # 找可行最优
        feasible = violation <= 1e-10
        if feasible.any():
            best_idx = np.where(feasible)[0][np.argmin(fitness[feasible])]
        else:
            best_idx = np.argmin(violation)

        best_x = X[best_idx].copy()
        best_fit = fitness[best_idx]
        best_viol = violation[best_idx]
        convergence = [best_fit]

        F = 0.5
        CR = 0.9

        T = max_fes // N
        for t in range(T):
            if n_fes >= max_fes:
                break

            # 自适应 F, CR
            F = np.clip(0.5 + 0.3 * np.random.randn(), 0.1, 0.9)
            CR = np.clip(0.9 + 0.1 * np.random.randn(), 0, 1)

            succ_count = 0

            for i in range(N):
                if n_fes >= max_fes:
                    break

                # DE/rand/1
                idxs = list(range(N))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                mutant = X[r1] + F * (X[r2] - X[r3])
                mutant = np.clip(mutant, lb, ub)

                j_rand = np.random.randint(n_dim)
                trial = X[i].copy()
                for j in range(n_dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial = np.clip(trial, lb, ub)
                fit_trial = obj_func(trial)
                viol_trial = self._constraint_violation(trial, constraints)
                n_fes += 1

                # CMODE 选择规则：Pareto 支配
                dominated = False
                if viol_trial <= 1e-10 and violation[i] <= 1e-10:
                    # 都可行：比目标值
                    dominated = fit_trial <= fitness[i]
                elif viol_trial <= 1e-10:
                    # trial 可行，当前不可行 → trial 优
                    dominated = True
                elif violation[i] <= 1e-10:
                    # 当前可行，trial 不可行 → 当前优
                    dominated = False
                else:
                    # 都不可行：比违反度
                    dominated = viol_trial <= violation[i]

                if dominated:
                    X[i] = trial
                    fitness[i] = fit_trial
                    violation[i] = viol_trial
                    succ_count += 1

                    # 更新全局最优
                    if viol_trial <= 1e-10:
                        if best_viol > 1e-10 or fit_trial < best_fit:
                            best_x = trial.copy()
                            best_fit = fit_trial
                            best_viol = viol_trial
                    elif viol_trial < best_viol:
                        best_x = trial.copy()
                        best_fit = fit_trial
                        best_viol = viol_trial

            convergence.append(best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  CMODE Iter {t:4d}/{T} | FES: {n_fes:5d} | "
                      f"Best: {best_fit:.6e} | Viol: {best_viol:.2e}")

        return best_x, best_fit, convergence
