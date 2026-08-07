"""
PNO-GWO v3.5: 黏菌管网络-灰狼混合优化算法（简化+多峰增强版）
============================================================
Physarum Network Optimizer enhanced with Grey Wolf Optimizer,
simplified for speed + Nelder-Mead local search + Cauchy restart.

v3.5 相对于 v3.0 的改动:
  [简化] KNN 图重建从每代改为每 5 代（加速 3-4×）
  [简化] CMEA 外部存档改为实验性开关，默认关闭
  [移除] GC-MMS 策略池完整移除（不稳定）
  [新增] M8 Nelder-Mead 局部精化（最后 20% FES 对 alpha 节点运行）
  [新增] M9 Cauchy 重启（停滞节点从最优附近 Cauchy 采样重生）
  [新增] M10 两阶段 FES 分配（前 80% PNO-GWO 搜索 → 后 20% 纯 GWO + NM）

依赖: numpy, scipy
"""
import numpy as np
import math
from copy import deepcopy
from scipy.spatial import KDTree


# ======================================================================
# 辅助函数
# ======================================================================

def _weighted_lehmer_mean(values, weights, eps=1e-10):
    """加权 Lehmer 均值（SHADE 使用的更新方式）"""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    return np.sum(w * v ** 2) / (np.sum(w * v) + eps)


def _ensure_2d(arr, ndim):
    """确保返回 2D 数组"""
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[1] != ndim:
        arr = arr.T
    return arr


# ======================================================================
# M1: Success-History 记忆体 (SHCA)
# ======================================================================

class SHCAMemory:
    """Success-History Conductivity Adaptation 记忆体

    存储成功 (alpha, beta) 对的档案，按 SHADE 方式更新。
    """
    def __init__(self, H=20, dim=2):
        self.H = H
        self.dim = dim          # 2: (alpha, beta) 或 1: (p_gwo)
        self.memory = np.full((H, dim), 0.5)   # 初始化为 0.5
        self.k = 0                             # 当前写入位置

    def sample(self):
        """从记忆体采样一组参数（Cauchy 分布）"""
        r = np.random.randint(0, self.H)
        base = self.memory[r]
        # Cauchy 分布采样
        params = base + 0.1 * np.random.standard_cauchy(self.dim)
        return params

    def update(self, successful_params, improvements):
        """用成功参数更新记忆体（加权 Lehmer 均值）"""
        if len(successful_params) == 0:
            return
        succ = np.asarray(successful_params, dtype=float)
        imp = np.abs(np.asarray(improvements, dtype=float)) + 1e-10

        if self.dim == 1:
            # succ is 1-D list of scalars
            succ = succ.ravel() if succ.ndim > 1 else succ
            imp = imp.ravel() if imp.ndim > 1 else imp
            self.memory[self.k, 0] = _weighted_lehmer_mean(succ, imp)
        else:
            for d in range(self.dim):
                self.memory[self.k, d] = _weighted_lehmer_mean(succ[:, d], imp)
        self.k = (self.k + 1) % self.H


# ======================================================================
# M3: 外部存档 (CMEA)
# ======================================================================

class ExternalArchive:
    """Conductivity-Modulated External Archive

    存储被替换的旧解，电导率越高 → 更积极从存档探索。
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.solutions = []     # list of ndarray
        self.fitness = []       # list of float

    def add(self, solution, fit):
        """添加一个被丢弃的解到存档"""
        if len(self.solutions) >= self.max_size:
            # 随机替换一个存档中的解（简单 FIFO）
            idx = np.random.randint(0, self.max_size)
            self.solutions[idx] = solution
            self.fitness[idx] = fit
        else:
            self.solutions.append(solution)
            self.fitness.append(fit)

    def sample(self, current_pos, cond_scale, n_dim, bounds):
        """基于电导率距离采样一个存档解

        cond_scale 高 → 更积极探索存档
        """
        if len(self.solutions) == 0:
            return None

        n_arch = len(self.solutions)
        if n_arch > 20:
            idx = np.random.choice(n_arch, 20, replace=False)
        else:
            idx = np.arange(n_arch)

        # 计算电导率调制的距离权重
        lb, ub = bounds
        span = ub - lb
        dists = np.array([np.linalg.norm(current_pos - self.solutions[i]) / span
                          for i in idx])
        weights = np.exp(-dists / (max(cond_scale, 0.01) + 1e-10))
        if weights.sum() < 1e-10:
            return None
        weights /= weights.sum()
        chosen = self.solutions[idx[np.random.choice(len(idx), p=weights)]]
        return chosen


# ======================================================================
# M4: 策略池 (GC-MMS)
# ======================================================================

class StrategyPool:
    """Graph-Coupled Multiple Mutation Strategies

    4 种策略，每种跟踪成功率和尝试次数。
    """
    STRATEGY_NAMES = ['PNO_graph', 'GWO_encircle', 'graph_DE', 'archive_guided']

    def __init__(self, n_strategies=4):
        self.n = n_strategies
        self.success = np.zeros(n_strategies, dtype=float)
        self.trials = np.ones(n_strategies, dtype=float)  # 避免除零
        self.scores = np.ones(n_strategies, dtype=float) / n_strategies

    def select(self, avg_conductivity, temperature_base=0.1):
        """电导率调制的 softmax 选择

        高电导率 → 低温度 → 更倾向 exploit 已知好策略
        低电导率 → 高温度 → 更倾向 explore 各种策略
        """
        T = temperature_base + 0.9 * (1 - np.clip(avg_conductivity, 0, 1))
        rates = self.success / (self.trials + 1e-10)
        exp_vals = np.exp(rates / (T + 1e-10))
        self.scores = exp_vals / (exp_vals.sum() + 1e-10)
        return np.random.choice(self.n, p=self.scores)

    def update(self, strategy_idx, improved):
        """更新策略的成功/尝试计数"""
        self.trials[strategy_idx] += 1
        if improved:
            self.success[strategy_idx] += 1


# ======================================================================
# 主算法类
# ======================================================================

class PhysarumNetworkOptimizer:
    """PNO-GWO 混合优化算法 (v3.5)

    融合 PNO 管状网络探索能力、GWO 等级包围开发能力，
    以及自适应参数控制 + Nelder-Mead 局部精化 + Cauchy 重启。

    Parameters
    ----------
    n_pop : int
        初始种群规模 (default: 18*dim, 遵循 L-SHADE 惯例)
    n_dim : int
        搜索空间维度
    bounds : tuple (lb, ub)
        搜索空间边界
    max_fes : int
        最大函数评估次数
    use_shca : bool
        是否启用成功历史参数自适应 (M1)
    use_cgpsr : bool
        是否启用种群缩减 (M2)
    use_cas : bool
        是否启用自适应 sigmoid (M6)
    use_cauchy : bool
        是否启用柯西变异 (M7)
    use_cmea : bool
        是否启用外部存档 (M3, 实验性, 默认关闭)
    use_nm : bool
        是否启用 Nelder-Mead 局部精化 (M8, 默认开启)
    use_twophase : bool
        是否启用两阶段 FES 分配 (M10, 默认开启)
    alpha : float
        管壁生长率默认初始值 (M1 关闭时固定)
    beta : float
        管道衰减率默认初始值 (M1 关闭时固定)
    k_neighbors : int
        k 近邻图连接数
    graph_rebuild_interval : int
        图重建间隔代数 (default: 5, 减少开销)
    **kwargs
        额外参数

    Other Parameters
    ----------------
    p_gwo_init : float
        初始 GWO 概率 (CAS 关闭时的 sigmoid 起点, default: 0.1)
    p_gwo_final : float
        最终 GWO 概率 (CAS 关闭时的 sigmoid 终点, default: 0.85)
    shca_h : int
        SHCA 记忆体大小 (default: 20)
    nm_max_iter : int
        Nelder-Mead 最大迭代 (default: 100)
    restart_interval : int
        Cauchy 重启间隔 (default: 10)
    """
    def __init__(self, n_pop=None, n_dim=30, bounds=(-100, 100), max_fes=30000,
                 use_shca=True, use_cgpsr=True, use_cas=True, use_cauchy=True,
                 use_cmea=False, use_nm=True, use_twophase=True,
                 alpha=0.5, beta=0.2, k_neighbors=5, graph_rebuild_interval=5,
                 **kwargs):
        self.n_dim = n_dim
        self.lb, self.ub = bounds
        self.n_pop = n_pop if n_pop is not None else min(18 * n_dim, 200)
        self.max_fes = max_fes
        self.name = kwargs.get('name', 'PNO-GWO-v3.5')

        # 机制开关
        self.use_shca = use_shca
        self.use_cgpsr = use_cgpsr
        self.use_cmea = use_cmea
        self.use_cas = use_cas
        self.use_cauchy = use_cauchy
        self.use_nm = use_nm
        self.use_twophase = use_twophase

        # PNO 基本参数
        self.alpha = alpha
        self.beta = beta
        self.k_neighbors = k_neighbors
        self.graph_rebuild_interval = graph_rebuild_interval
        self.prune_threshold = kwargs.get('prune_threshold', 0.05)
        self.sprout_interval = kwargs.get('sprout_interval', 8)
        self.cull_interval = kwargs.get('cull_interval', 15)

        # GWO 模式概率（CAS 关闭时的回退值）
        self.p_gwo_init = kwargs.get('p_gwo_init', 0.1)
        self.p_gwo_final = kwargs.get('p_gwo_final', 0.85)

        # SHCA 参数
        self.shca_h = kwargs.get('shca_h', 20)

        # 存档参数
        self.archive_max_ratio = kwargs.get('archive_max_ratio', 1.0)

        # Nelder-Mead 参数
        self.nm_max_iter = kwargs.get('nm_max_iter', 100)

        # Cauchy 重启参数
        self.restart_interval = kwargs.get('restart_interval', 10)

        # 两阶段 FES 比例
        self.phase1_ratio = kwargs.get('phase1_ratio', 0.8)

        # 内部状态（优化时重置）
        self.D_min = 1e-6
        self.D_max = 1.0
        self.n_fes = 0
        self.convergence = []
        self._reset_state()

    def _reset_state(self):
        """重置机制内部状态"""
        # M1: SHCA 记忆体
        self.mem_alpha = SHCAMemory(self.shca_h, dim=1) if self.use_shca else None
        self.mem_beta = SHCAMemory(self.shca_h, dim=1) if self.use_shca else None
        self.mem_pgwo = SHCAMemory(self.shca_h, dim=1) if self.use_shca else None

        # M3: 外部存档
        self.archive = ExternalArchive(
            int(self.n_pop * self.archive_max_ratio)
        ) if self.use_cmea else None

        # 收集成功参数（SHCA）
        self._succ_alpha = []
        self._succ_beta = []
        self._succ_pgwo = []
        self._succ_imp = []

        # 停滞跟踪
        self.stagnation = None
        self._generation = 0

    # ================================================================
    # 基础函数
    # ================================================================

    def init_population(self):
        return self.lb + np.random.rand(self.n_pop, self.n_dim) * (self.ub - self.lb)

    def boundary_check(self, X):
        return np.clip(X, self.lb, self.ub)

    def evaluate(self, X, obj_func):
        self.n_fes += 1
        return obj_func(X)

    def _build_knn_graph(self, positions):
        tree = KDTree(positions)
        n = len(positions)
        k = min(self.k_neighbors + 1, n)
        if k < 2:
            return np.array([], dtype=int).reshape(-1, 2)
        edges = set()
        for i in range(n):
            dists, idxs = tree.query(positions[i], k=k)
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

    # ================================================================
    # M6: CAS — 电导率自适应 sigmoid
    # ================================================================

    def _conductivity_entropy(self, conductivity, edges):
        """计算电导率熵 H_cond ∈ [0, 1]"""
        if len(edges) == 0 or len(conductivity) == 0:
            return 0.5
        D = np.clip(conductivity, self.D_min, self.D_max)
        D_sum = D.sum()
        if D_sum < 1e-10:
            return 0.5
        P = D / D_sum
        H = -np.sum(P * np.log(P + 1e-10))
        H_max = np.log(len(conductivity) + 1)
        return H / H_max if H_max > 0 else 0.5

    def _compute_p_gwo(self, t, T, cond_entropy):
        """CAS: 用电导率熵替代固定时间调度"""
        if not self.use_cas:
            # 回退到 v2.0 的固定 sigmoid
            return self.p_gwo_init + (self.p_gwo_final - self.p_gwo_init) / (
                1 + np.exp(-6 * (t / T - 0.4))
            )
        # CAS: H_target 从 0.8 线性下降到 0.3
        H_target = 0.8 - 0.5 * min(t / T, 1.0)
        k_steep = 4.0
        p = 1.0 / (1.0 + np.exp(-k_steep * (cond_entropy - H_target)))
        # 限定范围
        return np.clip(p, 0.05, 0.95)

    # ================================================================
    # M7: Cauchy-Conductivity Mutation
    # ================================================================

    def _cauchy_conductivity_mutation(self, node_idx, positions, conductivity,
                                       edges, stagnation, gamma_0=0.02):
        """柯西变异（重尾分布），尺度由电导率和停滞次数调制"""
        if not self.use_cauchy:
            return np.zeros(self.n_dim)

        scale = gamma_0 * (self.ub - self.lb)

        # 计算该节点的平均入射电导率
        n_edges = len(edges)
        incident = np.where((edges[:, 0] == node_idx) | (edges[:, 1] == node_idx))[0]
        valid = incident[incident < len(conductivity)]
        if len(valid) > 0:
            mean_cond = np.mean(conductivity[valid])
        else:
            mean_cond = 0.0

        # 电导率调制: 高电导率 → 小步长, 低电导率 → 大步长
        cond_factor = 1.0 - np.clip(mean_cond, 0, 1)

        # 停滞调制
        stg = stagnation[node_idx] if stagnation is not None else 0
        stg_factor = 1.0 + 0.5 * min(stg / 30.0, 2.0)

        mut_scale = scale * cond_factor * stg_factor
        # Cauchy 分布（重尾）
        return mut_scale * np.random.standard_cauchy(self.n_dim)

    # ================================================================
    # M8: Nelder-Mead 局部精化（轻量版）
    # ================================================================

    def _nelder_mead_local_search(self, obj_func, x0, max_iter=100, alpha=1.0,
                                   gamma=2.0, rho=0.5, sigma=0.5):
        """对初始点 x0 运行 Nelder-Mead 单纯形局部搜索

        Parameters
        ----------
        obj_func : callable
            目标函数
        x0 : ndarray
            起始点
        max_iter : int
            最大迭代次数（严格限制评估次数）
        alpha, gamma, rho, sigma : float
            NM 标准参数: 反射/扩展/收缩/缩紧

        Returns
        -------
        best_x : ndarray
            最优解
        best_fit : float
            最优适应度
        n_evals : int
            本次局部搜索消耗的评估次数
        """
        n = len(x0)
        span = self.ub - self.lb

        # 构建初始单纯形: x0 + 步长向量
        step = 0.05 * span
        if np.isscalar(step):
            step = np.full(n, step)
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            base = x0.copy()
            base[i] += step[i]
            base = np.clip(base, self.lb, self.ub)
            simplex[i + 1] = base

        # 评估
        fits = np.array([obj_func(simplex[i]) for i in range(n + 1)])
        n_evals = n + 1
        best_idx = np.argmin(fits)
        best_fit = fits[best_idx]

        for _ in range(max_iter):
            if n_evals >= self.max_fes * 0.2:
                break  # 不超过总预算的 20%

            # 排序
            order = np.argsort(fits)
            simplex = simplex[order]
            fits = fits[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # 反射
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, self.lb, self.ub)
            fr = obj_func(xr)
            n_evals += 1

            if fits[0] <= fr < fits[-2]:
                simplex[-1] = xr
                fits[-1] = fr
            elif fr < fits[0]:
                # 扩展
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, self.lb, self.ub)
                fe = obj_func(xe)
                n_evals += 1
                simplex[-1] = xe if fe < fr else xr
                fits[-1] = min(fe, fr)
            else:
                # 收缩
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, self.lb, self.ub)
                fc = obj_func(xc)
                n_evals += 1
                if fc < fits[-1]:
                    simplex[-1] = xc
                    fits[-1] = fc
                else:
                    # 缩紧
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], self.lb, self.ub)
                    fits[1:] = [obj_func(simplex[i]) for i in range(1, n + 1)]
                    n_evals += n

            # 更新最优
            best_idx = np.argmin(fits)
            if fits[best_idx] < best_fit:
                best_fit = fits[best_idx]

            # 收敛检查: 单纯形大小
            if np.std(simplex, axis=0).max() < 1e-8 * np.max(span):
                break

        best_idx = np.argmin(fits)
        return simplex[best_idx], fits[best_idx], n_evals

    # ================================================================
    # M9: Cauchy 重启
    # ================================================================

    def _cauchy_restart(self, idx, X, fitness, global_best, stagnation):
        """Cauchy 重启: 停滞节点从最优附近 Cauchy 采样重生

        替代 v3.0 的简单重置到等级狼附近。
        """
        stg = stagnation[idx]
        if stg < self.restart_interval:
            return X[idx], fitness[idx], False

        # Cauchy 重尾扰动
        scale = (self.ub - self.lb) * 0.1 * min(stg / self.restart_interval, 3.0)
        new_pos = global_best + np.random.standard_cauchy(self.n_dim) * scale
        new_pos = self.boundary_check(new_pos)
        return new_pos, fitness[idx], True  # 标记已重启

    # ================================================================
    # M1: 电导率更新（含 SHCA）
    # ================================================================

    def _update_conductivity(self, X, fitness, edges, conductivity, t, T):
        """带 SHCA 的电导率更新"""
        n_node = len(X)

        if self.use_shca and self.mem_alpha is not None:
            # 从 SHCA 记忆体采样每个节点的 alpha/beta
            alphas = np.array([np.clip(self.mem_alpha.sample()[0], 0.1, 1.0)
                               for _ in range(n_node)])
            betas = np.array([np.clip(self.mem_beta.sample()[0], 0.05, 0.5)
                              for _ in range(n_node)])
        else:
            alphas = np.full(n_node, self.alpha)
            betas = np.full(n_node, self.beta)

        # 存储当前参数用于训练（如果 SHCA 启用）
        self._current_alphas = alphas.copy()
        self._current_betas = betas.copy()
        self._current_edges = edges.copy()

        for idx_ed, (i, j) in enumerate(edges):
            if i >= n_node or j >= n_node:
                continue
            dist = np.linalg.norm(X[i] - X[j]) + 1e-10
            grad = np.abs(fitness[i] - fitness[j]) / dist
            grad = np.clip(grad, 0, 10)
            alpha_ij = (alphas[i] + alphas[j]) / 2
            beta_ij = (betas[i] + betas[j]) / 2
            conductivity[idx_ed] += alpha_ij * grad - beta_ij * conductivity[idx_ed]
            conductivity[idx_ed] = np.clip(conductivity[idx_ed], self.D_min, self.D_max)

        return conductivity

    # ================================================================
    # M4: 策略实现
    # ================================================================

    def _strategy_pno_graph(self, i, X, fitness, edges, conductivity, t, T, archive):
        """策略 1: PNO 图探索（原版改进）"""
        connected = np.where(
            (edges[:, 0] == i) | (edges[:, 1] == i)
        )[0]

        # 从存档获取引导（如启用 CMEA）
        archive_guide = None
        if self.use_cmea and archive is not None and np.random.rand() < 0.2:
            incident = connected[connected < len(conductivity)] if len(connected) > 0 else []
            cond_scale = np.mean(conductivity[incident]) if len(incident) > 0 else 0.5
            archive_guide = archive.sample(X[i], cond_scale, self.n_dim,
                                           (self.lb, self.ub))

        if len(connected) == 0:
            candidate = X[i] + self._levy_flight() * (1 + 0.5 * np.random.rand())
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

            # 存档引导
            if archive_guide is not None:
                attract += 0.3 * np.random.rand() * (archive_guide - X[i])

            levy = self._levy_flight() * (1 - t / T) * 0.5
            candidate = X[i] + attract + levy

        return candidate

    def _strategy_gwo_encircling(self, i, X, alpha_pos, beta_pos, delta_pos, a,
                                  conductivity, edges):
        """策略 2: GWO 等级包围（电导率调制步长）"""
        r1, r2 = np.random.rand(2)
        A1, C1 = 2 * a * r1 - a, 2 * r2
        D_alpha = np.abs(C1 * alpha_pos - X[i])
        X1 = alpha_pos - A1 * D_alpha

        r1, r2 = np.random.rand(2)
        A2, C2 = 2 * a * r1 - a, 2 * r2
        D_beta = np.abs(C2 * beta_pos - X[i])
        X2 = beta_pos - A2 * D_beta

        r1, r2 = np.random.rand(2)
        A3, C3 = 2 * a * r1 - a, 2 * r2
        D_delta = np.abs(C3 * delta_pos - X[i])
        X3 = delta_pos - A3 * D_delta

        candidate = (X1 + X2 + X3) / 3

        # 电导率调制步长
        connected = np.where(
            (edges[:, 0] == i) | (edges[:, 1] == i)
        )[0]
        valid_conn = connected[connected < len(conductivity)]
        if len(valid_conn) > 0:
            mean_cond = np.mean(conductivity[valid_conn])
            scale = 1.0 + 0.5 * (0.5 - mean_cond)
            candidate = alpha_pos + scale * (candidate - alpha_pos)

        return candidate

    def _strategy_graph_differential(self, i, X, fitness, edges, F):
        """策略 3: 图差分变异 (PNO 原生)

        从图邻居中选择两个不同节点做差分,
        区别于 DE 的全局随机选择.
        """
        connected = np.where(
            (edges[:, 0] == i) | (edges[:, 1] == i)
        )[0]
        if len(connected) < 2:
            # 邻居不够, 回退到 PNO 图
            return None

        # 从邻居中随机选两个不同的节点
        neighbor_nodes = []
        for e in connected:
            if e < len(edges):
                j = edges[e][0] if edges[e][1] == i else edges[e][1]
                neighbor_nodes.append(j)
        neighbor_nodes = list(set(neighbor_nodes))  # 去重
        if len(neighbor_nodes) < 2:
            return None

        chosen = np.random.choice(neighbor_nodes, 2, replace=False)
        j, k = chosen[0], chosen[1]
        return X[i] + F * (X[j] - X[k])

    def _strategy_archive_guided(self, i, X, alpha_pos, archive, F, r1, r2):
        """策略 4: 存档引导 (类似 JADE current-to-pbest/1)

        使用种群内随机解和存档解的差分.
        """
        if archive is None or len(archive.solutions) < 1:
            return None

        # 从种群选一个随机解
        pop_idx = np.random.choice(len(X))
        while pop_idx == i and len(X) > 1:
            pop_idx = np.random.choice(len(X))

        # 从存档选一个随机解
        arch_idx = np.random.randint(0, len(archive.solutions))
        arch_sol = archive.solutions[arch_idx]

        mutant = X[i] + F * (alpha_pos - X[i]) + F * (X[pop_idx] - arch_sol)
        return mutant

    # ================================================================
    # M5: CLLS — 电导率景观局部搜索
    # ================================================================

    def _local_search(self, idx, X, fitness, edges, obj_func):
        """轻量局部搜索: 从图邻居构建二次模型

        收集邻居点, 拟合二次曲面, 取最小值.
        如果邻居不够, 回退到 Nelder-Mead 一次迭代.
        """
        connected = np.where(
            (edges[:, 0] == idx) | (edges[:, 1] == idx)
        )[0]
        neighbor_nodes = []
        for e in connected:
            if e < len(edges):
                j = edges[e][0] if edges[e][1] == idx else edges[e][1]
                neighbor_nodes.append(j)
        neighbor_nodes = list(set(neighbor_nodes))

        if len(neighbor_nodes) < 2:
            return X[idx], fitness[idx]

        # 简单加权平均 + 随机扰动
        nbs = neighbor_nodes[:min(5, len(neighbor_nodes))]
        weights = np.array([1.0 / (abs(fitness[n]) + 1e-10) for n in nbs])
        weights /= weights.sum()
        weighted_center = np.zeros(self.n_dim)
        for w, n in zip(weights, nbs):
            weighted_center += w * X[n]

        # 向加权中心收缩
        alpha_node = idx == np.argmin(fitness)
        step = 0.05 if alpha_node else 0.15
        candidate = X[idx] + step * (weighted_center - X[idx])
        candidate = self.boundary_check(candidate)
        fit_candidate = self.evaluate(candidate, obj_func)

        if fit_candidate < fitness[idx]:
            return candidate, fit_candidate

        # 如果加权平均没改善, 对 alpha 节点做精细扰动
        if alpha_node:
            pert = 0.01 * (self.ub - self.lb) * np.random.randn(self.n_dim)
            candidate2 = X[idx] + pert
            candidate2 = self.boundary_check(candidate2)
            fit_candidate2 = self.evaluate(candidate2, obj_func)
            if fit_candidate2 < fitness[idx]:
                return candidate2, fit_candidate2

        return X[idx], fitness[idx]

    # ================================================================
    # M2: CG-PSR — 种群缩减
    # ================================================================

    def _compute_target_population(self, fes_ratio):
        """计算目标种群大小"""
        N_init = self.n_pop
        N_min = 4
        if not self.use_cgpsr:
            return N_init
        # 非线性缩减 (gamma=0.8)
        return round(N_min + (N_init - N_min) * (1 - fes_ratio) ** 0.8)

    def _select_nodes_for_removal(self, X, fitness, stagnation, edges,
                                   conductivity, n_remove):
        """多标准选择要移除的节点"""
        if n_remove <= 0:
            return []

        n = len(X)
        if n_remove >= n:
            return list(range(n))

        # 标准 1: 适应度排名 (0=best, 1=worst)
        order = np.argsort(fitness)
        rank_fit = np.zeros(n)
        for r, idx in enumerate(order):
            rank_fit[idx] = r / max(n - 1, 1)

        # 标准 2: 度中心性 (越小越可能是外围节点)
        degree = np.zeros(n)
        for e in edges:
            if e[0] < n and e[1] < n:
                degree[e[0]] += 1
                degree[e[1]] += 1
        degree_norm = degree / (degree.max() + 1e-10)

        # 标准 3: 停滞次数归一化
        stg_max = max(stagnation.max(), 1)
        stg_norm = stagnation / stg_max

        # 综合评分: fitness 权重最高
        scores = 0.5 * rank_fit + 0.2 * (1 - degree_norm) + 0.3 * stg_norm
        remove_idx = np.argsort(scores)[-n_remove:]  # 得分最高的移除
        return list(remove_idx)

    # ================================================================
    # 主优化循环
    # ================================================================

    def optimize(self, obj_func, n_dim=None, bounds=None, max_fes=None, verbose=True):
        """运行 PNO-GWO v3.0 混合优化

        Parameters
        ----------
        obj_func : callable
            目标函数
        verbose : bool
            是否打印迭代信息

        Returns
        -------
        best_x : ndarray
            最优解
        best_fit : float
            最优适应度
        convergence : list
            收敛曲线
        """
        self.n_fes = 0
        self.convergence = []
        self._reset_state()

        # --------------- 初始化 ---------------
        X = self.init_population()
        fitness = np.array([self.evaluate(x, obj_func) for x in X])

        idx_best = np.argmin(fitness)
        global_best = X[idx_best].copy()
        global_best_fit = fitness[idx_best]
        self.convergence.append(global_best_fit)

        # 初始化图结构
        edges = self._build_knn_graph(X)
        conductivity = np.full(len(edges), 0.5)

        n_node = len(X)
        T = self.max_fes // max(n_node, 1) + 1
        self.stagnation = np.zeros(len(X), dtype=int)

        # SHCA 成功参数缓存
        if self.use_shca:
            prev_alpha = np.full(len(X), self.alpha)
            prev_beta = np.full(len(X), self.beta)
            prev_fitness = fitness.copy()

        for t in range(T):
            if self.n_fes >= self.max_fes:
                break

            n_node_now = len(X)
            fes_ratio = self.n_fes / self.max_fes

            # ---- Step 0: 适应度排序，确定 α-β-δ 等级 ----
            order = np.argsort(fitness)
            alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
            alpha_pos = X[alpha_idx].copy()
            beta_pos = X[beta_idx].copy()
            delta_pos = X[delta_idx].copy()

            # ---- CAS: 电导率熵计算 ----
            cond_entropy = self._conductivity_entropy(conductivity, edges)

            # ---- Step 1: 电导率更新（含 SHCA）----
            if len(edges) > 0:
                # M1: SHCA 采样的 alpha/beta
                if self.use_shca and self.mem_alpha is not None:
                    sampled_alphas = np.array([
                        np.clip(self.mem_alpha.sample()[0], 0.1, 1.0)
                        for _ in range(n_node_now)
                    ])
                    sampled_betas = np.array([
                        np.clip(self.mem_beta.sample()[0], 0.05, 0.5)
                        for _ in range(n_node_now)
                    ])
                else:
                    sampled_alphas = np.full(n_node_now, self.alpha)
                    sampled_betas = np.full(n_node_now, self.beta)

                for idx_ed, (i, j) in enumerate(edges):
                    if i >= n_node_now or j >= n_node_now:
                        continue
                    dist = np.linalg.norm(X[i] - X[j]) + 1e-10
                    grad = np.abs(fitness[i] - fitness[j]) / dist
                    grad = np.clip(grad, 0, 10)
                    alpha_ij = (sampled_alphas[i] + sampled_alphas[j]) / 2
                    beta_ij = (sampled_betas[i] + sampled_betas[j]) / 2
                    conductivity[idx_ed] += alpha_ij * grad - beta_ij * conductivity[idx_ed]
                    conductivity[idx_ed] = np.clip(conductivity[idx_ed], self.D_min, self.D_max)

            # ---- Step 2: 混合节点更新 ----
            a = 2 - 2 * t / T
            p_gwo = self._compute_p_gwo(t, T, cond_entropy)

            # M10: 两阶段 FES 分配 — 后 20% 切换到纯 GWO 模式
            fes_phase2 = self.use_twophase and (self.n_fes / self.max_fes) > self.phase1_ratio

            for i in range(n_node_now):
                if self.n_fes >= self.max_fes:
                    break

                rank_ratio = np.where(order == i)[0][0] / n_node_now if n_node_now > 0 else 0

                # M10: Phase 2 → 纯 GWO + Nelder-Mead
                if fes_phase2:
                    use_gwo = True
                else:
                    use_gwo = np.random.rand() < p_gwo
                    if rank_ratio < 0.2:
                        use_gwo = True

                if use_gwo:
                    # GWO 包围
                    candidate = self._strategy_gwo_encircling(
                        i, X, alpha_pos, beta_pos, delta_pos, a,
                        conductivity, edges
                    )
                    if rank_ratio > 0.1:
                        candidate += 0.05 * (self.ub - self.lb) * np.random.randn(self.n_dim) * (1 - t / T)
                else:
                    # PNO 图探索
                    candidate = self._strategy_pno_graph(
                        i, X, fitness, edges, conductivity, t, T,
                        self.archive if self.use_cmea else None
                    )

                # M7: Cauchy 变异（停滞节点）
                if (self.stagnation[i] > 5 and np.random.rand() < 0.3
                        and self.use_cauchy):
                    cauchy_delta = self._cauchy_conductivity_mutation(
                        i, X, conductivity, edges, self.stagnation
                    )
                    candidate = X[i] + cauchy_delta

                candidate = self.boundary_check(candidate)
                fit_candidate = self.evaluate(candidate, obj_func)

                improved = fit_candidate < fitness[i]
                if improved:
                    # M3: 将旧解加入存档（仅当启用）
                    if self.use_cmea and self.archive is not None:
                        self.archive.add(X[i].copy(), fitness[i])

                    X[i] = candidate
                    fitness[i] = fit_candidate
                    self.stagnation[i] = 0

                    if fit_candidate < global_best_fit:
                        global_best = candidate.copy()
                        global_best_fit = fit_candidate
                else:
                    self.stagnation[i] += 1

                # 收集 SHCA 训练数据
                if self.use_shca and improved and self.mem_alpha is not None:
                    self._succ_alpha.append(sampled_alphas[i])
                    self._succ_beta.append(sampled_betas[i])
                    self._succ_imp.append(fitness[i] - fit_candidate)

            # 更新 SHCA 记忆体
            if self.use_shca and self.mem_alpha is not None and len(self._succ_alpha) > 0:
                self.mem_alpha.update(self._succ_alpha, self._succ_imp)
                self.mem_beta.update(self._succ_beta, self._succ_imp)
                self._succ_alpha.clear()
                self._succ_beta.clear()
                self._succ_imp.clear()

            # ---- Step 3: 多基点发芽 ----
            n_current = len(X)
            if (t > 0 and t % self.sprout_interval == 0
                    and self.n_fes < self.max_fes and n_current < self.max_fes * 0.01):
                base_idx = np.random.choice([alpha_idx, beta_idx, delta_idx])
                if base_idx < len(X):
                    base_pos = X[base_idx]
                else:
                    base_pos = global_best
                sprout = base_pos + np.random.standard_cauchy(self.n_dim) * 0.08 * (self.ub - self.lb)
                sprout = self.boundary_check(sprout)
                fit_sprout = self.evaluate(sprout, obj_func)
                X = np.vstack([X, sprout])
                fitness = np.append(fitness, fit_sprout)
                self.stagnation = np.append(self.stagnation, 0)
                if fit_sprout < global_best_fit:
                    global_best = sprout.copy()
                    global_best_fit = fit_sprout

            # ---- Step 4: Culling + CG-PSR + M9 Cauchy 重启 ----
            if t > 0 and t % self.cull_interval == 0:
                n_current = len(X)
                target_pop = self._compute_target_population(fes_ratio)
                n_target = max(target_pop, self.n_pop // 2)

                # M9: Cauchy 重启 — 停滞节点从最优附近 Cauchy 采样重生
                for i in range(n_current):
                    new_pos, fit_val, restarted = self._cauchy_restart(
                        i, X, fitness, global_best, self.stagnation
                    )
                    if restarted:
                        new_fit = self.evaluate(new_pos, obj_func)
                        if new_fit < fitness[i]:
                            X[i] = new_pos
                            fitness[i] = new_fit
                            self.stagnation[i] = 0
                            if new_fit < global_best_fit:
                                global_best = new_pos.copy()
                                global_best_fit = new_fit
                        else:
                            self.stagnation[i] = min(self.stagnation[i], self.restart_interval)

                # CG-PSR 种群缩减
                n_current = len(X)
                if n_current > n_target:
                    n_remove = n_current - n_target
                    remove_idx = self._select_nodes_for_removal(
                        X, fitness, self.stagnation, edges, conductivity, n_remove
                    )
                    if len(remove_idx) > 0 and len(remove_idx) < n_current:
                        keep_mask = np.ones(n_current, dtype=bool)
                        keep_mask[remove_idx] = False
                        X = X[keep_mask]
                        fitness = fitness[keep_mask]
                        self.stagnation = self.stagnation[keep_mask]

            # ---- Step 5: 重建图结构（每 graph_rebuild_interval 代）----
            if len(X) > 1 and (t % self.graph_rebuild_interval == 0 or t == 0):
                # 重建后需要补齐边数变化的 conductivity
                new_edges = self._build_knn_graph(X)
                if len(new_edges) != len(edges):
                    conductivity = np.full(len(new_edges), 0.5)
                edges = new_edges
                # 修剪
                keep = conductivity >= self.prune_threshold
                if keep.sum() >= len(X):
                    edges = edges[keep]
                    conductivity = conductivity[keep]
            elif len(X) <= 1:
                edges = np.array([], dtype=int).reshape(-1, 2)
                conductivity = np.array([])

            # ---- M8: Nelder-Mead 局部精化（最后阶段 / Phase 2）----
            if (self.use_nm and fes_phase2
                    and t % max(1, self.graph_rebuild_interval) == 0
                    and self.n_fes < self.max_fes):
                # 对当前全局最优运行 Nelder-Mead
                nm_iter = min(self.nm_max_iter,
                              int((self.max_fes - self.n_fes) // (self.n_dim + 1)))
                if nm_iter > self.n_dim:
                    new_best, new_fit, n_evals = self._nelder_mead_local_search(
                        obj_func, global_best, max_iter=nm_iter
                    )
                    # 手动计入 FES（NM 内部已经调用 evaluate 计数）
                    if new_fit < global_best_fit:
                        global_best = new_best.copy()
                        global_best_fit = new_fit
                        # 更新 alpha 节点
                        if len(X) > 1:
                            X[0] = new_best.copy()
                            fitness[0] = new_fit

            self.convergence.append(global_best_fit)
            self._generation += 1

            if verbose and t % max(1, T // 10) == 0:
                n_node = len(X)
                phase_str = "P2-NM" if fes_phase2 else "P1"
                print(f"  {self.name} [{phase_str}] Iter {t:4d}/{T} | "
                      f"FES: {self.n_fes:5d} | Best: {global_best_fit:.6e} | "
                      f"Nodes: {n_node:3d} | p_gwo: {p_gwo:.2f} | "
                      f"H_cond: {cond_entropy:.3f}")

        return global_best, global_best_fit, self.convergence

    # ================================================================
    # 向后兼容（旧测试引用的方法名）
    # ================================================================
    def _gwo_encircling(self, x, alpha, beta, delta, a):
        """旧版 _gwo_encircling — 对单个节点做 GWO 包围（测试兼容）"""
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
        """旧版 _gwo_probability — 固定 sigmoid（测试兼容）"""
        return self.p_gwo_init + (self.p_gwo_final - self.p_gwo_init) / (
            1 + np.exp(-6 * (t / T - 0.4))
        )
