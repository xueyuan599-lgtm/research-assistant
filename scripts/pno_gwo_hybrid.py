"""
PNO-GWO Hybrid: 黏菌管网络-灰狼混合优化算法
=============================================
将 GWO 的等级包围机制嵌入 PNO 的管状网络节点更新中，
实现探索（PNO 图结构）与开发（GWO 等级引导）的自适应平衡。

核心创新：
1. 双模节点更新：自适应切换 PNO 图模式 ↔ GWO 等级模式
2. 电导率调制 GWO 步长：管道网络信息指导包围精度
3. 节点分群策略：精英集中开发 + 底层维持探索
4. 多基点发芽：从 α-β-δ 等级群体中选择基点

v2.0 — 2026-07-27
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats
from scipy.spatial import KDTree
import time
import math
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# Benchmark Functions
# ======================================================================

def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) \
           - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e

def griewank(x):
    n = len(x)
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))

BENCHMARKS = {
    'Sphere': {'func': sphere, 'bounds': (-100, 100), 'optimum': 0},
    'Rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12), 'optimum': 0},
    'Rosenbrock': {'func': rosenbrock, 'bounds': (-10, 10), 'optimum': 0},
    'Ackley': {'func': ackley, 'bounds': (-32, 32), 'optimum': 0},
    'Griewank': {'func': griewank, 'bounds': (-600, 600), 'optimum': 0},
}


# ======================================================================
# PNO-GWO Hybrid Algorithm
# ======================================================================

class PNO_GWO_Hybrid:
    """PNO-GWO 混合优化算法

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
    """
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 alpha=0.5, beta=0.2, k_neighbors=5, prune_threshold=0.05,
                 sprout_interval=8, cull_interval=15,
                 p_gwo_init=0.1, p_gwo_final=0.85, name='PNO-GWO'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.alpha = alpha          # 管壁生长率
        self.beta = beta            # 管道衰减率
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
        """均匀随机初始化"""
        return self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)

    def boundary_check(self, X):
        return np.clip(X, self.lb, self.ub)

    def evaluate(self, X, obj_func):
        self.n_fes += 1
        return obj_func(X)

    def _build_knn_graph(self, positions):
        """构建 k 近邻图"""
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
        """自适应 GWO 概率（sigmoid 曲线）"""
        # 中心在 t/T = 0.4，斜率 6
        return self.p_gwo_init + (self.p_gwo_final - self.p_gwo_init) / (
            1 + np.exp(-6 * (t / T - 0.4))
        )

    def optimize(self, obj_func, verbose=True):
        """运行 PNO-GWO 混合优化"""
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

        # 记录未改善次数（用于 omega 替换触发）
        stagnation = np.zeros(len(X), dtype=int)

        for t in range(T):
            if self.n_fes >= self.max_fes:
                break

            n_node_now = len(X)
            n_node_current = n_node_now  # 快照

            # ---- Step 0: 适应度排序，确定 α-β-δ 等级 ----
            order = np.argsort(fitness)
            alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
            alpha_pos = X[alpha_idx].copy()
            beta_pos = X[beta_idx].copy()
            delta_pos = X[delta_idx].copy()

            # ---- Step 1: 管道电导率更新（基于适应度梯度） ----
            if len(edges) > 0:
                for idx_ed, (i, j) in enumerate(edges):
                    if i >= n_node_current or j >= n_node_current:
                        continue
                    dist = np.linalg.norm(X[i] - X[j]) + 1e-10
                    grad = np.abs(fitness[i] - fitness[j]) / dist
                    grad = np.clip(grad, 0, 10)
                    conductivity[idx_ed] += self.alpha * grad - self.beta * conductivity[idx_ed]
                    conductivity[idx_ed] = np.clip(conductivity[idx_ed], self.D_min, self.D_max)

            # ---- Step 2: 混合节点更新 ----
            a = 2 - 2 * t / T  # GWO 衰减因子
            p_gwo = self._gwo_probability(t, T)  # 当前 GWO 概率

            for i in range(n_node_current):
                if self.n_fes >= self.max_fes:
                    break

                # 分群策略：节点排名比例
                rank_ratio = np.where(order == i)[0][0] / n_node_current if n_node_current > 0 else 0

                # 决定使用 GWO 模式还是 PNO 模式
                use_gwo = np.random.rand() < p_gwo

                # 精英节点（前 20%）强制 GWO 模式加强局部开发
                if rank_ratio < 0.2:
                    use_gwo = True

                if use_gwo:
                    # ---------- GWO 模式 ----------
                    # 基础 GWO 包围
                    candidate = self._gwo_encircling(X[i], alpha_pos, beta_pos, delta_pos, a)

                    # 电导率调制步长
                    # 找到节点 i 的连接边，计算平均入射电导率
                    connected = np.where(
                        (edges[:, 0] == i) | (edges[:, 1] == i)
                    )[0]
                    if len(connected) > 0 and len(conductivity) > 0:
                        valid_conn = connected[connected < len(conductivity)]
                        if len(valid_conn) > 0:
                            mean_cond = np.mean(conductivity[valid_conn])
                            # 高电导率 → 缩小包围步长；低电导率 → 放大
                            scale = 1.0 + 0.5 * (0.5 - mean_cond)
                            candidate = alpha_pos + scale * (candidate - alpha_pos)

                    # 添加微扰动（避免过早停滞）
                    if rank_ratio > 0.1:  # 非最优节点加扰动
                        candidate += 0.05 * (self.ub - self.lb) * np.random.randn(self.n_dim) * (1 - t / T)

                else:
                    # ---------- PNO 图模式 ----------
                    connected = np.where(
                        (edges[:, 0] == i) | (edges[:, 1] == i)
                    )[0]

                    if len(connected) == 0:
                        # 孤立节点：Lévy 飞行探索
                        candidate = X[i] + self._levy_flight() * (1 + 0.5 * rank_ratio)
                    else:
                        # 按电导率加权选择邻居
                        valid_conn = connected[connected < len(conductivity)]
                        if len(valid_conn) == 0:
                            valid_conn = connected
                        weights = conductivity[valid_conn] if len(valid_conn) <= len(conductivity) else np.ones(len(valid_conn))
                        if len(np.unique(valid_conn)) > 0:
                            if weights.sum() > 0:
                                probs = weights / weights.sum()
                                chosen_edge = valid_conn[np.random.choice(len(valid_conn), p=probs)]
                            else:
                                chosen_edge = valid_conn[np.random.randint(len(valid_conn))]

                            j = (edges[chosen_edge][0]
                                 if edges[chosen_edge][1] == i
                                 else edges[chosen_edge][1])

                            # 向高适应度邻居移动 / 从低适应度逃离
                            attract = np.random.rand() * (X[j] - X[i])
                            if fitness[j] > fitness[i]:
                                attract = -attract
                            levy = self._levy_flight() * (1 - t / T) * 0.5
                            candidate = X[i] + attract + levy
                        else:
                            candidate = X[i] + self._levy_flight()

                candidate = self.boundary_check(candidate)
                fit_candidate = self.evaluate(candidate, obj_func)

                # 贪心接受
                if fit_candidate < fitness[i]:
                    X[i] = candidate
                    fitness[i] = fit_candidate
                    stagnation[i] = 0
                    if fit_candidate < global_best_fit:
                        global_best = candidate.copy()
                        global_best_fit = fit_candidate
                else:
                    stagnation[i] += 1

            # ---- Step 3: 发芽（多基点） ----
            if (t > 0 and t % self.sprout_interval == 0 and self.n_fes < self.max_fes):
                n_node_now = len(X)
                if n_node_now < self.max_fes * 0.01:
                    # 随机从 alpha、beta、delta 中选择一个作为基点
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

            # ---- Step 4: 淘汰（最差节点） + Omega 替换 ----
            if t > 0 and t % self.cull_interval == 0:
                n_current = len(X)
                if n_current > self.n_pop:
                    n_cull = int(n_current * 0.15)

                    # GWO 启发式：长时间未改善的节点被替换到 α-β-δ 附近
                    stale_idx = np.where(stagnation >= self.cull_interval * 2)[0]
                    if len(stale_idx) > 0:
                        n_replace = min(len(stale_idx), max(1, n_cull // 2))
                        replace_idx = np.random.choice(stale_idx, n_replace, replace=False)
                        for ri in replace_idx:
                            leader = np.random.choice([0, 1, 2])
                            base = [alpha_pos, beta_pos, delta_pos][leader]
                            X[ri] = base + np.random.randn(self.n_dim) * 0.15 * (self.ub - self.lb)
                            X[ri] = self.boundary_check(X[ri])
                            # 不额外评估——在下一轮迭代中会评估

                    # 移除最差的
                    order_cull = np.argsort(fitness)
                    survivors = order_cull[:-n_cull]
                    X = X[survivors]
                    fitness = fitness[survivors]
                    stagnation = stagnation[survivors]

            # ---- Step 5: 重建图结构 + 修剪低电导率边 ----
            if len(X) > 1:
                edges = self._build_knn_graph(X)
                conductivity = np.full(len(edges), 0.5)
                # 修剪
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
                      f"p_gwo: {p_gwo:.2f} | a: {a:.3f}")

        return global_best, global_best_fit, self.convergence


# ======================================================================
# PNO (原始版本 — 复用于对比)
# ======================================================================

class PhysarumNetworkOptimizer:
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 alpha=0.6, beta=0.3, k_neighbors=5, prune_threshold=0.05,
                 sprout_interval=10, cull_interval=20, name='PNO'):
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
        self.name = name
        self.D_min = 1e-6
        self.D_max = 1.0
        self.n_fes = 0
        self.convergence = []

    def init_population(self):
        return self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)

    def boundary_check(self, X):
        return np.clip(X, self.lb, self.ub)

    def evaluate(self, X, obj_func):
        self.n_fes += 1
        return obj_func(X)

    def _build_knn_graph(self, positions):
        tree = KDTree(positions)
        edges = set()
        for i in range(len(positions)):
            dists, idxs = tree.query(positions[i], k=min(self.k_neighbors + 1, len(positions)))
            for j in idxs[1:]:
                if i != j:
                    edges.add(tuple(sorted((i, j))))
        return np.array(list(edges), dtype=int)

    def _levy_flight(self, beta=1.5):
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.n_dim) * sigma
        v = np.random.randn(self.n_dim)
        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        return step * 0.01 * (self.ub - self.lb)

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []

        X = self.init_population()
        fitness = np.array([self.evaluate(x, obj_func) for x in X])
        idx_best = np.argmin(fitness)
        global_best = X[idx_best].copy()
        global_best_fit = fitness[idx_best]
        self.convergence.append(global_best_fit)

        edges = self._build_knn_graph(X)
        n_edges = len(edges)
        conductivity = np.full(n_edges, 0.5)

        n_node = self.n_pop
        T = self.max_fes // (n_node + 10) + 1

        for t in range(T):
            if self.n_fes >= self.max_fes:
                break

            n_node_now = len(X)

            if len(edges) > 0:
                for idx, (i, j) in enumerate(edges):
                    if i >= len(X) or j >= len(X):
                        continue
                    dist = np.linalg.norm(X[i] - X[j]) + 1e-10
                    grad = np.abs(fitness[i] - fitness[j]) / dist
                    grad = np.clip(grad, 0, 10)
                    conductivity[idx] += self.alpha * grad - self.beta * conductivity[idx]
                    conductivity[idx] = np.clip(conductivity[idx], self.D_min, self.D_max)

            for i in range(n_node_now):
                if self.n_fes >= self.max_fes:
                    break

                connected = np.where(
                    (edges[:, 0] == i) | (edges[:, 1] == i)
                )[0]

                if len(connected) == 0:
                    candidate = X[i] + self._levy_flight()
                else:
                    weights = conductivity[connected]
                    if weights.sum() > 0:
                        probs = weights / weights.sum()
                        chosen_edge = connected[np.random.choice(len(connected), p=probs)]
                    else:
                        chosen_edge = connected[np.random.randint(len(connected))]

                    j = edges[chosen_edge][0] if edges[chosen_edge][1] == i else edges[chosen_edge][1]

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
                    if fit_candidate < global_best_fit:
                        global_best = candidate.copy()
                        global_best_fit = fit_candidate

            if t > 0 and t % self.sprout_interval == 0 and self.n_fes < self.max_fes:
                if n_node_now < self.max_fes * 0.01:
                    sprout = global_best + np.random.randn(self.n_dim) * 0.1 * (self.ub - self.lb)
                    sprout = self.boundary_check(sprout)
                    fit_sprout = self.evaluate(sprout, obj_func)
                    X = np.vstack([X, sprout])
                    fitness = np.append(fitness, fit_sprout)
                    if fit_sprout < global_best_fit:
                        global_best = sprout.copy()
                        global_best_fit = fit_sprout

            if t > 0 and t % self.cull_interval == 0:
                n_current = len(X)
                if n_current > self.n_pop:
                    n_cull = int(n_current * 0.15)
                    order = np.argsort(fitness)
                    survivors = order[:-n_cull]
                    X = X[survivors]
                    fitness = fitness[survivors]

            if len(X) > 1:
                edges = self._build_knn_graph(X)
                n_edges = len(edges)
                conductivity = np.full(n_edges, 0.5)
            else:
                edges = np.array([], dtype=int).reshape(-1, 2)
                conductivity = np.array([])

            if len(conductivity) > 0:
                keep = conductivity >= self.prune_threshold
                if keep.sum() >= len(X):
                    edges = edges[keep]
                    conductivity = conductivity[keep]

            self.convergence.append(global_best_fit)

            if verbose and t % max(1, T // 10) == 0:
                print(f"  {self.name} Iter {t:4d}/{T} | FES: {self.n_fes:5d} | "
                      f"Best: {global_best_fit:.6e} | Nodes: {len(X)}")

        return global_best, global_best_fit, self.convergence


# ======================================================================
# PSO
# ======================================================================

class PSO:
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 w=0.7, c1=1.5, c2=1.5, name='PSO'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        X = self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)
        V = np.random.randn(self.n_pop, self.n_dim) * 0.1 * (self.ub - self.lb)
        fitness = np.array([obj_func(x) for x in X])
        self.n_fes = self.n_pop
        pbest = X.copy()
        pbest_fit = fitness.copy()
        gbest_idx = np.argmin(fitness)
        gbest = X[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]
        self.convergence.append(gbest_fit)

        T = self.max_fes // self.n_pop
        for t in range(T):
            if self.n_fes >= self.max_fes:
                break
            w_t = self.w - (self.w - 0.2) * t / T
            for i in range(self.n_pop):
                r1, r2 = np.random.rand(2)
                V[i] = w_t * V[i] + self.c1 * r1 * (pbest[i] - X[i]) + self.c2 * r2 * (gbest - X[i])
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], self.lb, self.ub)
            for i in range(self.n_pop):
                self.n_fes += 1
                fit = obj_func(X[i])
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = X[i].copy()
            self.convergence.append(gbest_fit)
            if verbose and t % max(1, T // 10) == 0:
                print(f"  PSO Iter {t:4d}/{T} | FES: {self.n_fes:5d} | Best: {gbest_fit:.6e}")
        return gbest, gbest_fit, self.convergence


# ======================================================================
# GWO
# ======================================================================

class GWO:
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000, name='GWO'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        X = self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)
        fitness = np.array([obj_func(x) for x in X])
        self.n_fes = self.n_pop
        order = np.argsort(fitness)
        alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        self.convergence.append(fitness[order[0]])

        T = self.max_fes // self.n_pop
        for t in range(T):
            if self.n_fes >= self.max_fes:
                break
            a = 2 - 2 * t / T
            for i in range(self.n_pop):
                r1, r2 = np.random.rand(2)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha - X[i])
                X1 = alpha - A1 * D_alpha
                r1, r2 = np.random.rand(2)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta - X[i])
                X2 = beta - A2 * D_beta
                r1, r2 = np.random.rand(2)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta - X[i])
                X3 = delta - A3 * D_delta
                X[i] = (X1 + X2 + X3) / 3
                X[i] = np.clip(X[i], self.lb, self.ub)
            for i in range(self.n_pop):
                self.n_fes += 1
                fit = obj_func(X[i])
                if fit < fitness[i]:
                    fitness[i] = fit
            order = np.argsort(fitness)
            alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
            self.convergence.append(fitness[order[0]])
            if verbose and t % max(1, T // 10) == 0:
                print(f"  GWO Iter {t:4d}/{T} | FES: {self.n_fes:5d} | Best: {fitness[order[0]]:.6e}")
        return alpha, fitness[order[0]], self.convergence


# ======================================================================
# DE
# ======================================================================

class DE:
    def __init__(self, n_pop=50, n_dim=30, bounds=(-100, 100), max_fes=15000,
                 F=0.7, CR=0.9, name='DE'):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.F = F
        self.CR = CR
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        X = self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)
        fitness = np.array([obj_func(x) for x in X])
        self.n_fes = self.n_pop
        self.convergence.append(fitness.min())

        T = self.max_fes // self.n_pop
        for t in range(T):
            if self.n_fes >= self.max_fes:
                break
            F_t = self.F * (1 + 0.5 * np.random.randn())
            F_t = np.clip(F_t, 0.2, 1.5)
            for i in range(self.n_pop):
                idxs = [idx for idx in range(self.n_pop) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = X[a] + F_t * (X[b] - X[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                cross = np.random.rand(self.n_dim) < self.CR
                cross[np.random.randint(self.n_dim)] = True
                trial = np.where(cross, mutant, X[i])
                self.n_fes += 1
                fit_trial = obj_func(trial)
                if fit_trial < fitness[i]:
                    X[i] = trial
                    fitness[i] = fit_trial
            self.convergence.append(fitness.min())
            if verbose and t % max(1, T // 10) == 0:
                print(f"  DE  Iter {t:4d}/{T} | FES: {self.n_fes:5d} | Best: {fitness.min():.6e}")
        idx_best = np.argmin(fitness)
        return X[idx_best], fitness[idx_best], self.convergence


# ======================================================================
# Random Search
# ======================================================================

class RandomSearch:
    def __init__(self, bounds=(-100, 100), max_fes=15000, n_dim=30, name='Random'):
        self.lb, self.ub = bounds
        self.max_fes = max_fes
        self.n_dim = n_dim
        self.name = name
        self.n_fes = 0
        self.convergence = []

    def optimize(self, obj_func, verbose=True):
        self.n_fes = 0
        self.convergence = []
        best_fit = np.inf
        for _ in range(self.max_fes):
            x = self.lb + np.random.rand(self.n_dim) * (self.ub - self.lb)
            fit = obj_func(x)
            self.n_fes += 1
            if fit < best_fit:
                best_fit = fit
            self.convergence.append(best_fit)
        return None, best_fit, self.convergence


# ======================================================================
# Benchmark Framework
# ======================================================================

def run_single_benchmark(algo_class, algo_params, benchmark, n_runs=30):
    """对单个算法在某个函数上运行 n_runs 次"""
    results = []
    all_convergences = []
    for seed in range(n_runs):
        np.random.seed(seed)
        algo = algo_class(**algo_params)
        _, best_fit, conv = algo.optimize(benchmark['func'], verbose=False)
        results.append(best_fit)
        all_convergences.append(conv)
    return results, all_convergences


def format_sci(num):
    if num < 1e-8:
        return f"{num:.2e}"
    return f"{num:.4f}"


def main():
    DIM = 30
    N_POP = 50
    MAX_FES = 30000
    N_RUNS = 20

    algorithms = {
        'PNO-GWO': (PNO_GWO_Hybrid, {
            'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES,
        }),
        'PNO': (PhysarumNetworkOptimizer, {
            'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES,
        }),
        'GWO': (GWO, {
            'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES,
        }),
        'DE': (DE, {
            'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES,
        }),
        'PSO': (PSO, {
            'n_pop': N_POP, 'n_dim': DIM, 'max_fes': MAX_FES,
        }),
        'Random': (RandomSearch, {
            'max_fes': MAX_FES, 'n_dim': DIM,
        }),
    }

    print("=" * 80)
    print("PNO-GWO Hybrid — 混合算法 Benchmark 报告")
    print(f"维度: {DIM}D | 种群: {N_POP} | Max FES: {MAX_FES} | 独立运行: {N_RUNS} 次")
    print("=" * 80)

    all_results = {}  # (func_name, algo_name) -> stats dict
    all_convs = {}    # (func_name, algo_name) -> list of convergence curves

    # ==================================================================
    # 运行所有 Benchmark
    # ==================================================================
    for fname, fconfig in BENCHMARKS.items():
        print(f"\n{'=' * 60}")
        print(f"  {fname}")
        print(f"{'=' * 60}")
        bounds = fconfig['bounds']
        benchmark = {'func': fconfig['func'], 'bounds': bounds}

        for aname, (aclass, aparams) in algorithms.items():
            params = aparams.copy()
            if 'bounds' in params:
                params['bounds'] = bounds
            else:
                params['bounds'] = bounds

            start = time.time()
            results, convs = run_single_benchmark(aclass, params, benchmark, n_runs=N_RUNS)
            elapsed = time.time() - start

            median = np.median(results)
            mean = np.mean(results)
            std = np.std(results)
            best = np.min(results)
            worst = np.max(results)

            all_results[(fname, aname)] = {
                'median': median, 'mean': mean, 'std': std,
                'best': best, 'worst': worst, 'all': results,
                'time': elapsed,
            }
            all_convs[(fname, aname)] = convs

            print(f"  {aname:8s} | Best: {format_sci(best):>10s} | "
                  f"Median: {format_sci(median):>10s} | "
                  f"Mean: {format_sci(mean):>10s} | "
                  f"Std: {format_sci(std):>8s} | "
                  f"Time: {elapsed:.1f}s")

    # ==================================================================
    # 结果汇总表
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("结果汇总（中位数 Median）")
    print("=" * 80)

    # 表头
    header = f"{'Function':<12s}"
    for aname in algorithms:
        header += f" {aname:>10s}"
    print(header)
    print("-" * (12 + 11 * len(algorithms)))

    for fname in BENCHMARKS:
        row = f"{fname:<12s}"
        for aname in algorithms:
            med = all_results[(fname, aname)]['median']
            row += f" {format_sci(med):>10s}"
        print(row)

    # ==================================================================
    # 统计检验：Friedman + Mann-Whitney
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("统计检验 (Friedman + Mann-Whitney U, α=0.05)")
    print("=" * 80)

    for fname in BENCHMARKS:
        print(f"\n--- {fname} ---")
        alg_names = list(algorithms.keys())
        data_matrix = np.array([all_results[(fname, a)]['all'] for a in alg_names])

        # Friedman 检验
        friedman_stat, friedman_p = stats.friedmanchisquare(*data_matrix)
        print(f"  Friedman χ² = {friedman_stat:.4f}, p = {friedman_p:.4e}")

        # 平均排名
        ranks = np.array([
            stats.rankdata(data_matrix[:, i])
            for i in range(data_matrix.shape[1])
        ])
        mean_ranks = ranks.mean(axis=0)
        print(f"  {'Algo':10s} {'Avg Rank':>10s}")
        print(f"  {'-'*22}")
        for i, a in enumerate(alg_names):
            rank_str = f"{mean_ranks[i]:.3f}"
            stars = ""
            if mean_ranks[i] == mean_ranks.min():
                stars = " ← best"
            print(f"  {a:10s} {rank_str:>10s}{stars}")

        # PNO-GWO vs 其他 (Mann-Whitney)
        pno_gwo_vals = all_results[(fname, 'PNO-GWO')]['all']
        print(f"\n  PNO-GWO vs Others (Mann-Whitney):")
        for a in alg_names:
            if a == 'PNO-GWO':
                continue
            other_vals = all_results[(fname, a)]['all']
            stat, p = stats.mannwhitneyu(pno_gwo_vals, other_vals, alternative='two-sided')
            better = np.median(pno_gwo_vals) < np.median(other_vals)
            sig = "p<0.05" if p < 0.05 else "n.s."
            arrow = "WIN  " if better else "LOSE"
            md = np.median(pno_gwo_vals) / np.median(other_vals) if np.median(other_vals) > 0 else float('inf')
            ratio_str = f" (ratio={md:.2e})" if np.isfinite(md) and md > 0 else ""
            print(f"    vs {a:8s}: p = {p:.4e} [{sig}] {arrow}{ratio_str}")

    # ==================================================================
    # 与 GWO 和 PNO 的详细对比
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("PNO-GWO vs PNO vs GWO — 详细对比")
    print("=" * 80)

    print(f"\n{'Function':<12s} {'Metric':>8s} {'PNO':>12s} {'GWO':>12s} {'PNO-GWO':>12s} {'Imprv/PNO':>10s} {'Imprv/GWO':>10s}")
    print("-" * 80)
    for fname in BENCHMARKS:
        pno_med = all_results[(fname, 'PNO')]['median']
        gwo_med = all_results[(fname, 'GWO')]['median']
        hybrid_med = all_results[(fname, 'PNO-GWO')]['median']

        imp_vs_pno = pno_med / hybrid_med if hybrid_med != 0 else float('inf')
        imp_vs_gwo = gwo_med / hybrid_med if hybrid_med != 0 else float('inf')

        print(f"{fname:<12s} {'Median':>8s} {format_sci(pno_med):>12s} {format_sci(gwo_med):>12s} "
              f"{format_sci(hybrid_med):>12s} "
              f"{'x' + format_sci(imp_vs_pno) if np.isfinite(imp_vs_pno) else 'INF':>10s} "
              f"{'x' + format_sci(imp_vs_gwo) if np.isfinite(imp_vs_gwo) else 'INF':>10s}")

        for metric in ['Best', 'Mean', 'Std']:
            if metric == 'Best':
                pno_v = all_results[(fname, 'PNO')]['best']
                gwo_v = all_results[(fname, 'GWO')]['best']
                hybrid_v = all_results[(fname, 'PNO-GWO')]['best']
            elif metric == 'Mean':
                pno_v = all_results[(fname, 'PNO')]['mean']
                gwo_v = all_results[(fname, 'GWO')]['mean']
                hybrid_v = all_results[(fname, 'PNO-GWO')]['mean']
            else:
                pno_v = all_results[(fname, 'PNO')]['std']
                gwo_v = all_results[(fname, 'GWO')]['std']
                hybrid_v = all_results[(fname, 'PNO-GWO')]['std']

            print(f"{'':12s} {metric:>8s} {format_sci(pno_v):>12s} {format_sci(gwo_v):>12s} "
                  f"{format_sci(hybrid_v):>12s}")

    # ==================================================================
    # 总体排名
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("总体排名（跨所有函数）")
    print("=" * 80)

    all_ranks = []
    for aname in algorithms:
        func_medians = []
        for fname in BENCHMARKS:
            vals = all_results[(fname, aname)]['all']
            optimum = BENCHMARKS[fname]['optimum']
            func_medians.append(np.median(np.abs(np.array(vals) - optimum)))
        avg_perf = np.mean(func_medians)
        all_ranks.append((aname, avg_perf, func_medians))

    all_ranks.sort(key=lambda x: x[1])

    print(f"{'Rank':5s} {'Algo':10s} {'Avg Dist':>10s} | ", end="")
    for fname in BENCHMARKS:
        print(f"{fname:>10s}", end=" ")
    print()
    print("-" * (25 + 11 * len(BENCHMARKS)))

    for i, (aname, avg, perfs) in enumerate(all_ranks, 1):
        perf_strs = [format_sci(p) for p in perfs]
        print(f"  {i:3d}  {aname:10s} {format_sci(avg):>10s} | ", end="")
        for s in perf_strs:
            print(f"{s:>10s}", end=" ")
        print()

    # ==================================================================
    # PNO-GWO vs PNO 获胜统计
    # ==================================================================
    print("\n\n" + "=" * 80)
    print("PNO-GWO vs PNO — 胜率统计")
    print("=" * 80)

    for fname in BENCHMARKS:
        pno_vals = np.array(all_results[(fname, 'PNO')]['all'])
        hybrid_vals = np.array(all_results[(fname, 'PNO-GWO')]['all'])
        wins = np.sum(hybrid_vals < pno_vals)
        ties = np.sum(hybrid_vals == pno_vals)
        losses = np.sum(hybrid_vals > pno_vals)
        print(f"  {fname:12s}: PNO-GWO wins {wins}/{N_RUNS}, "
              f"ties {ties}, losses {losses} "
              f"(win rate: {100*wins/N_RUNS:.0f}%)")

    print("\n\n✅ Benchmark 完成!")


if __name__ == '__main__':
    main()
