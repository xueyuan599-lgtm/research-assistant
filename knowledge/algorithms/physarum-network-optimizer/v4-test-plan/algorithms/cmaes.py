"""
CMA-ES: Covariance Matrix Adaptation Evolution Strategy
========================================================
Hansen & Ostermeier (2001)

进化策略，自适应协方差矩阵。使用 cma 库实现。
如果 cma 不可用，使用简化的自实现版本。
"""

import numpy as np

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False


def _to_scalar_bounds(lb, ub, n_dim):
    """统一边界为标量或等长数组"""
    lb_arr = np.full(n_dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.full(n_dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)
    return lb_arr, ub_arr


class CMAES:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)"""

    def __init__(self, sigma0=0.3, **kwargs):
        self.sigma0 = sigma0
        self.name = 'CMA-ES'

    def optimize(self, obj_func, n_dim=30, bounds=(-100, 100),
                 max_fes=30000, verbose=False):
        if HAS_CMA:
            return self._optimize_cma(obj_func, n_dim, bounds, max_fes, verbose)
        else:
            return self._optimize_simple(obj_func, n_dim, bounds, max_fes, verbose)

    def _optimize_cma(self, obj_func, n_dim, bounds, max_fes, verbose):
        """使用 cma 库的实现"""
        lb_arr, ub_arr = _to_scalar_bounds(bounds[0], bounds[1], n_dim)
        x0 = np.random.uniform(lb_arr, ub_arr)
        sigma = self.sigma0 * float(np.mean(ub_arr - lb_arr))

        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fes
        opts['verb_disp'] = 0
        opts['verb_log'] = 0
        opts['bounds'] = [lb_arr.tolist(), ub_arr.tolist()]
        opts['popsize'] = min(4 + int(3 * np.log(n_dim)), 100)

        es = cma.CMAEvolutionStrategy(x0, sigma, opts)

        best_x = x0.copy()
        best_fit = obj_func(x0)
        n_fes = 1
        convergence = [best_fit]

        gen = 0
        while not es.stop() and n_fes < max_fes:
            solutions = es.ask()
            fits = [obj_func(s) for s in solutions]
            n_fes += len(fits)
            es.tell(solutions, fits)

            for s in solutions:
                s[:] = np.clip(s, lb_arr, ub_arr)

            gen_best_idx = np.argmin(fits)
            if fits[gen_best_idx] < best_fit:
                best_fit = fits[gen_best_idx]
                best_x = solutions[gen_best_idx].copy()

            convergence.append(best_fit)
            gen += 1

            if verbose and gen % 10 == 0:
                print(f"  CMA-ES Gen {gen:4d} | FES: {n_fes:5d} | Best: {best_fit:.6e}")

        return best_x, best_fit, convergence

    def _optimize_simple(self, obj_func, n_dim, bounds, max_fes, verbose):
        """简化 CMA-ES（不依赖 cma 库）"""
        lb_arr, ub_arr = _to_scalar_bounds(bounds[0], bounds[1], n_dim)
        span = ub_arr - lb_arr
        n_pop = min(4 + int(3 * np.log(n_dim)), 100)
        sigma = self.sigma0 * float(np.mean(span))

        mean = np.random.uniform(lb_arr, ub_arr)
        C = np.eye(n_dim)

        fit_mean = obj_func(mean)
        n_fes = 1
        best_x = mean.copy()
        best_fit = fit_mean
        convergence = [best_fit]

        mu = n_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        cs = (mu_eff + 2) / (n_dim + mu_eff + 5)
        c1 = 2 / ((n_dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n_dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + cs
        chi_n = np.sqrt(n_dim) * (1 - 1 / (4 * n_dim) + 1 / (21 * n_dim ** 2))

        pc = np.zeros(n_dim)
        ps = np.zeros(n_dim)

        T = max_fes // n_pop
        for t in range(T):
            if n_fes >= max_fes:
                break

            try:
                B = np.linalg.cholesky(C).T
            except np.linalg.LinAlgError:
                C = np.eye(n_dim)
                B = np.eye(n_dim)

            Z = np.random.randn(n_pop, n_dim)
            X = mean + sigma * (Z @ B)
            X = np.clip(X, lb_arr, ub_arr)

            fits = np.array([obj_func(x) for x in X])
            n_fes += n_pop

            sorted_idx = np.argsort(fits)
            old_mean = mean.copy()
            mean = sum(weights[i] * X[sorted_idx[i]] for i in range(mu))

            ps = ((1 - cs) * ps
                  + np.sqrt(cs * (2 - cs) * mu_eff)
                  * np.linalg.solve(B.T, (mean - old_mean) / sigma))
            hsig = (np.linalg.norm(ps)
                    / np.sqrt(1 - (1 - cs) ** (2 * n_fes / n_pop))
                    / chi_n < 1.4 + 2 / (n_dim + 1))
            pc = ((1 - cc) * pc
                  + hsig * np.sqrt(cc * (2 - cc) * mu_eff)
                  * (mean - old_mean) / sigma)

            artmp = (X[sorted_idx[:mu]] - old_mean) / sigma
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu * sum(weights[i] * np.outer(artmp[i], artmp[i])
                            for i in range(mu)))

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, float(np.mean(span)))

            if fits[sorted_idx[0]] < best_fit:
                best_fit = fits[sorted_idx[0]]
                best_x = X[sorted_idx[0]].copy()

            convergence.append(best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  CMA-ES(simple) Gen {t:4d} | FES: {n_fes:5d} | Best: {best_fit:.6e}")

        return best_x, best_fit, convergence