"""
PNO-GWO: 黏菌管网络-灰狼混合优化算法 (v2.0)
=============================================
Physarum Network Optimizer enhanced with Grey Wolf Optimizer hierarchical
encircling mechanism.

核心创新：
1. 双模节点更新：自适应切换 PNO 图模式 ↔ GWO 等级模式
2. 电导率调制 GWO 步长：管道网络信息指导包围精度
3. 节点分群策略（前 20% 精英强制 GWO 开发）
4. 多基点发芽（从 α-β-δ 等级群体中选择基点）
5. Omega 停滞节点替换机制

依赖: numpy, scipy
"""
import numpy as np
import math
from scipy.spatial import KDTree


class PhysarumNetworkOptimizer:
    """PNO-GWO 混合优化算法 (v2.0)

    融合 PNO 的管状网络探索能力与 GWO 的等级包围开发能力。

    Parameters
    ----------
    n_pop : int
        初始种群规模 (default: 50)
    n_dim : int
        搜索空间维度
    bounds : tuple (lb, ub)
        搜索空间边界
    max_fes : int
        最大函数评估次数
    alpha : float
        管壁生长率 (default: 0.5)
    beta : float
        管道衰减率 (default: 0.2)
    k_neighbors : int
        k近邻图连接数 (default: 5)
    prune_threshold : float
        管道修剪阈值 (default: 0.05)
    sprout_interval : int
        发芽间隔代数 (default: 8)
    cull_interval : int
        淘汰间隔代数 (default: 15)
    p_gwo_init : float
        初始 GWO 模式概率 (default: 0.1)
    p_gwo_final : float
        最终 GWO 模式概率 (default: 0.85)

    Examples
    --------
    >>> import numpy as np
    >>> from algorithm import PhysarumNetworkOptimizer
    >>> def sphere(x):
    ...     return np.sum(x ** 2)
    >>> pno = PhysarumNetworkOptimizer(n_pop=30, n_dim=10,
    ...                                 bounds=(-100, 100), max_fes=5000)
    >>> best_x, best_fit, conv = pno.optimize(sphere)
    >>> print(f"Best: {best_fit:.6e}")
    """

    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 alpha=0.5, beta=0.2, k_neighbors=5, prune_threshold=0.05,
                 sprout_interval=8, cull_interval=15,
                 p_gwo_init=0.1, p_gwo_final=0.85, name='PNO-GWO'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.alpha = alpha
        self.beta = beta
        self.k_neighbors = k_neighbors
        self.prune_threshold = prune_threshold
        self.sprout_interval = sprout_interval
        self.cull_interval = cull_interval
        self.p_gwo_init = p_gwo_init
        self.p_gwo_final = p_gwo_final
        self.name = name
        self.D_min = 1e-6
        self.D_max = 1.0
        self.n_fes = 0
        self.convergence = []

    def init_population(self):
        """在边界内均匀随机初始化种群"""
        return self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)

    def boundary_check(self, X):
        """将解约束到搜索空间内"""
        return np.clip(X, self.lb, self.ub)

    def evaluate(self, X, obj_func):
        """评估一次目标函数"""
        self.n_fes += 1
        return obj_func(X)

    def _build_knn_graph(self, positions):
        """构建k近邻图"""
        tree = KDTree(positions)
        edges = set()
        for i in range(len(positions)):
            dists, idxs = tree.query(
                positions[i], k=min(self.k_neighbors + 1, len(positions))
            )
            for j in idxs[1:]:
                if i != j:
                    edges.add(tuple(sorted((i, j))))
        return np.array(list(edges), dtype=int)

    def _levy_flight(self, beta=1.5):
        """Lévy 飞行（Mantegna 算法）"""
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.n_dim) * sigma
        v = np.random.randn(self.n_dim)
        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        return step * 0.01 * (self.ub - self.lb)

    def _gwo_encircling(self, x, alpha, beta, delta, a):
        """GWO 包围机制（单个节点）"""
        r1, r2 = np.random.rand(2)
        A1, C1 = 2 * a * r1 - a, 2 * r2
        D_alpha = np.abs(C1 * alpha - x)
        X1 = alpha - A1 * D_alpha

        r1, r2 = np.random.rand(2)
        A2, C2 = 2 * a * r1 - a, 2 * r2
        D_beta = np.abs(C2 * beta - x)
        X2 = beta - A2 * D_beta

        r1, r2 = np.random.rand(2)
        A3, C3 = 2 * a * r1 - a, 2 * r2
        D_delta = np.abs(C3 * delta - x)
        X3 = delta - A3 * D_delta

        return (X1 + X2 + X3) / 3

    def _gwo_probability(self, t, T):
        """自适应 GWO 概率（sigmoid 曲线，中心在 t/T=0.4）"""
        return self.p_gwo_init + (self.p_gwo_final - self.p_gwo_init) / (
            1 + np.exp(-6 * (t / T - 0.4))
        )

    def optimize(self, obj_func, verbose=True):
        """运行 PNO-GWO 混合优化

        Parameters
        ----------
        obj_func : callable
            目标函数，接受一维数组 x, 返回标量适应度
        verbose : bool
            是否打印迭代信息

        Returns
        -------
        best_x : ndarray
            找到的最优解
        best_fit : float
            最优适应度值
        convergence : list
            收敛曲线
        """
        self.n_fes = 0
        self.convergence = []

        # ---------- 初始化 ----------
        X = self.init_population()
        fitness = np.array([self.evaluate(x, obj_func) for x in X])

        idx_best = np.argmin(fitness)
        global_best = X[idx_best].copy()
        global_best_fit = fitness[idx_best]
        self.convergence.append(global_best_fit)

        # 初始化图结构
        edges = self._build_knn_graph(X)
        conductivity = np.full(len(edges), 0.5)

        n_node = self.n_pop
        T = self.max_fes // (n_node + 10) + 1

        stagnation = np.zeros(len(X), dtype=int)

        for t in range(T):
            if self.n_fes >= self.max_fes:
                break

            n_node_now = len(X)

            # ---- Step 0: 适应度排序，确定 α-β-δ 等级 ----
            order = np.argsort(fitness)
            alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
            alpha_pos = X[alpha_idx].copy()
            beta_pos = X[beta_idx].copy()
            delta_pos = X[delta_idx].copy()

            # ---- Step 1: 管道电导率更新 ----
            if len(edges) > 0:
                for idx_ed, (i, j) in enumerate(edges):
                    if i >= n_node_now or j >= n_node_now:
                        continue
                    dist = np.linalg.norm(X[i] - X[j]) + 1e-10
                    grad = np.abs(fitness[i] - fitness[j]) / dist
                    grad = np.clip(grad, 0, 10)
                    conductivity[idx_ed] += self.alpha * grad - self.beta * conductivity[idx_ed]
                    conductivity[idx_ed] = np.clip(conductivity[idx_ed], self.D_min, self.D_max)

            # ---- Step 2: 混合节点更新 ----
            a = 2 - 2 * t / T
            p_gwo = self._gwo_probability(t, T)

            for i in range(n_node_now):
                if self.n_fes >= self.max_fes:
                    break

                rank_ratio = np.where(order == i)[0][0] / n_node_now

                # 精英节点（前 20%）强制 GWO 模式
                use_gwo = np.random.rand() < p_gwo
                if rank_ratio < 0.2:
                    use_gwo = True

                if use_gwo:
                    # ----- GWO 模式：等级包围 -----
                    candidate = self._gwo_encircling(X[i], alpha_pos, beta_pos, delta_pos, a)

                    # 电导率调制步长
                    connected = np.where(
                        (edges[:, 0] == i) | (edges[:, 1] == i)
                    )[0]
                    valid_conn = connected[connected < len(conductivity)]
                    if len(valid_conn) > 0:
                        mean_cond = np.mean(conductivity[valid_conn])
                        scale = 1.0 + 0.5 * (0.5 - mean_cond)
                        candidate = alpha_pos + scale * (candidate - alpha_pos)

                    # 非精英加微扰动
                    if rank_ratio > 0.1:
                        candidate += 0.05 * (self.ub - self.lb) * np.random.randn(self.n_dim) * (1 - t / T)

                else:
                    # ----- PNO 模式：管道网络探索 -----
                    connected = np.where(
                        (edges[:, 0] == i) | (edges[:, 1] == i)
                    )[0]

                    if len(connected) == 0:
                        candidate = X[i] + self._levy_flight() * (1 + 0.5 * rank_ratio)
                    else:
                        valid_conn = connected[connected < len(conductivity)]
                        if len(valid_conn) == 0:
                            valid_conn = connected
                        weights = conductivity[valid_conn]
                        if len(valid_conn) > 0 and weights.sum() > 0:
                            probs = weights / weights.sum()
                            chosen_edge = valid_conn[np.random.choice(len(valid_conn), p=probs)]
                        else:
                            chosen_edge = valid_conn[np.random.randint(len(valid_conn))]

                        j = (edges[chosen_edge][0]
                             if edges[chosen_edge][1] == i
                             else edges[chosen_edge][1])

                        attract = np.random.rand() * (X[j] - X[i])
                        if fitness[j] > fitness[i]:
                            attract = -attract
                        levy = self._levy_flight() * (1 - t / T) * 0.5
                        candidate = X[i] + attract + levy

                candidate = self.boundary_check(candidate)
                fit_candidate = self.evaluate(candidate, obj_func)

                if fit_candidate < fitness[i]:
                    X[i] = candidate
                    fitness[i] = fit_candidate
                    stagnation[i] = 0
                    if fit_candidate < global_best_fit:
                        global_best = candidate.copy()
                        global_best_fit = fit_candidate
                else:
                    stagnation[i] += 1

            # ---- Step 3: 多基点发芽 ----
            if (t > 0 and t % self.sprout_interval == 0 and self.n_fes < self.max_fes):
                n_node_now = len(X)
                if n_node_now < self.max_fes * 0.01:
                    base_idx = np.random.choice([alpha_idx, beta_idx, delta_idx])
                    base_pos = X[base_idx] if base_idx < len(X) else global_best
                    sprout = base_pos + np.random.randn(self.n_dim) * 0.08 * (self.ub - self.lb)
                    sprout = self.boundary_check(sprout)
                    fit_sprout = self.evaluate(sprout, obj_func)
                    X = np.vstack([X, sprout])
                    fitness = np.append(fitness, fit_sprout)
                    stagnation = np.append(stagnation, 0)
                    if fit_sprout < global_best_fit:
                        global_best = sprout.copy()
                        global_best_fit = fit_sprout

            # ---- Step 4: 淘汰 + Omega 替换 ----
            if t > 0 and t % self.cull_interval == 0:
                n_current = len(X)
                if n_current > self.n_pop:
                    n_cull = int(n_current * 0.15)

                    # 长时间未改善的节点重置到 α-β-δ 附近
                    stale_idx = np.where(stagnation >= self.cull_interval * 2)[0]
                    if len(stale_idx) > 0:
                        n_replace = min(len(stale_idx), max(1, n_cull // 2))
                        replace_idx = np.random.choice(stale_idx, n_replace, replace=False)
                        for ri in replace_idx:
                            leader = np.random.choice([0, 1, 2])
                            base = [alpha_pos, beta_pos, delta_pos][leader]
                            X[ri] = base + np.random.randn(self.n_dim) * 0.15 * (self.ub - self.lb)
                            X[ri] = self.boundary_check(X[ri])

                    order_cull = np.argsort(fitness)
                    survivors = order_cull[:-n_cull]
                    X = X[survivors]
                    fitness = fitness[survivors]
                    stagnation = stagnation[survivors]

            # ---- Step 5: 重建图结构 + 修剪 ----
            if len(X) > 1:
                edges = self._build_knn_graph(X)
                conductivity = np.full(len(edges), 0.5)
                keep = conductivity >= self.prune_threshold
                if keep.sum() >= len(X):
                    edges = edges[keep]
                    conductivity = conductivity[keep]
            else:
                edges = np.array([], dtype=int).reshape(-1, 2)
                conductivity = np.array([])

            self.convergence.append(global_best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  {self.name} Iter {t:4d}/{T} | FES: {self.n_fes:5d} | "
                      f"Best: {global_best_fit:.6e} | Nodes: {len(X)} | "
                      f"p_gwo: {p_gwo:.2f}")

        return global_best, global_best_fit, self.convergence
