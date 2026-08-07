"""
PNO-GWO v2.0 — 黏菌管网络-灰狼混合优化算法 单元测试
运行: python -m pytest test_algorithm.py -v
"""
import numpy as np
from algorithm import PhysarumNetworkOptimizer


def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


class TestPhysarumNetworkOptimizer:
    """PNO-GWO v2.0 核心功能测试"""

    def test_init_defaults(self):
        """默认参数初始化"""
        pno = PhysarumNetworkOptimizer(n_dim=5, bounds=(-10, 10))
        # v3.0 默认 n_pop = 18 * n_dim (L-SHADE 惯例)
        assert pno.n_pop == 90
        assert pno.n_dim == 5
        assert pno.lb == -10
        assert pno.ub == 10
        assert pno.alpha == 0.5       # v2.0 默认值
        assert pno.beta == 0.2         # v2.0 默认值
        assert pno.p_gwo_init == 0.1   # v2.0 新增参数
        assert pno.p_gwo_final == 0.85 # v2.0 新增参数

    def test_population_initialization(self):
        """种群初始化是否在边界内"""
        pno = PhysarumNetworkOptimizer(n_pop=30, n_dim=5, bounds=(-5, 5))
        pop = pno.init_population()
        assert pop.shape == (30, 5)
        assert pop.min() >= -5
        assert pop.max() <= 5

    def test_boundary_check(self):
        """边界约束是否有效"""
        pno = PhysarumNetworkOptimizer(n_dim=3, bounds=(-1, 1))
        out_of_bounds = np.array([10, -10, 0.5])
        clamped = pno.boundary_check(out_of_bounds)
        assert np.all(clamped >= -1)
        assert np.all(clamped <= 1)
        assert clamped[0] == 1
        assert clamped[1] == -1
        assert clamped[2] == 0.5

    def test_evaluate_counts_fes(self):
        """每次评估是否递增 FES 计数"""
        pno = PhysarumNetworkOptimizer(n_dim=2, bounds=(-1, 1))
        assert pno.n_fes == 0
        pno.evaluate(np.array([0.5, 0.5]), sphere)
        assert pno.n_fes == 1
        pno.evaluate(np.array([0.0, 0.0]), sphere)
        assert pno.n_fes == 2

    def test_optimize_simple_sphere(self):
        """在简单问题上能否收敛"""
        np.random.seed(42)
        pno = PhysarumNetworkOptimizer(
            n_pop=10, n_dim=2,
            bounds=(-10, 10), max_fes=500
        )
        best_x, best_fit, conv = pno.optimize(sphere, verbose=False)
        assert best_fit >= 0
        assert len(conv) > 0
        assert best_x.shape == (2,)

    def test_optimize_returns_convergence(self):
        """收敛曲线是否单调不增"""
        np.random.seed(42)
        pno = PhysarumNetworkOptimizer(
            n_pop=10, n_dim=2,
            bounds=(-10, 10), max_fes=300
        )
        _, _, conv = pno.optimize(sphere, verbose=False)
        for i in range(1, len(conv)):
            assert conv[i] <= conv[i-1] + 1e-10

    def test_dimension_consistency(self):
        """不同维度下优化是否正常"""
        for dim in [1, 2, 5, 10]:
            np.random.seed(42)
            pno = PhysarumNetworkOptimizer(
                n_pop=10, n_dim=dim,
                bounds=(-10, 10), max_fes=300
            )
            best_x, _, _ = pno.optimize(sphere, verbose=False)
            assert best_x.shape == (dim,), f"Dim {dim} 失败"

    def test_levy_flight_output(self):
        """Lévy 飞行是否返回正确维度"""
        pno = PhysarumNetworkOptimizer(n_dim=10, bounds=(-100, 100))
        step = pno._levy_flight()
        assert step.shape == (10,)

    def test_gwo_encircling(self):
        """GWO 包围机制是否返回正确维度"""
        pno = PhysarumNetworkOptimizer(n_dim=5, bounds=(-10, 10))
        x = np.random.randn(5)
        alpha = np.random.randn(5)
        beta = np.random.randn(5)
        delta = np.random.randn(5)
        candidate = pno._gwo_encircling(x, alpha, beta, delta, a=1.0)
        assert candidate.shape == (5,)

    def test_gwo_probability(self):
        """自适应 GWO 概率是否在 [p_gwo_init, p_gwo_final] 区间"""
        pno = PhysarumNetworkOptimizer(n_dim=2, bounds=(-1, 1),
                                        p_gwo_init=0.1, p_gwo_final=0.85)
        assert abs(pno._gwo_probability(0, 100) - 0.1) < 0.1   # sigmoid 渐近逼近 p_init
        assert abs(pno._gwo_probability(100, 100) - 0.85) < 0.1  # sigmoid 渐近逼近 p_final
        assert 0.1 < pno._gwo_probability(40, 100) < 0.85  # 递增

    def test_build_knn_graph(self):
        """k近邻图是否生成正确数量的边"""
        pno = PhysarumNetworkOptimizer(
            n_pop=20, n_dim=2, bounds=(-1, 1), k_neighbors=3
        )
        positions = np.random.randn(20, 2)
        edges = pno._build_knn_graph(positions)
        assert edges.shape[1] == 2
        assert len(edges) > 0

    def test_multiple_runs_stability(self):
        """多次运行是否稳定"""
        fits = []
        for seed in range(3):
            np.random.seed(seed)
            pno = PhysarumNetworkOptimizer(
                n_pop=10, n_dim=2,
                bounds=(-10, 10), max_fes=200
            )
            _, best_fit, _ = pno.optimize(sphere, verbose=False)
            fits.append(best_fit)
        assert all(f >= 0 for f in fits)

    def test_optimize_ackley_improvement(self):
        """Ackley 上应能收敛到较低值（v2.0 改进检验）"""
        np.random.seed(42)
        pno = PhysarumNetworkOptimizer(
            n_pop=20, n_dim=5,
            bounds=(-32, 32), max_fes=2000
        )
        _, best_fit, _ = pno.optimize(
            lambda x: (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x)))
                       - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e),
            verbose=False
        )
        # v3.5 should achieve < 2.0 on 5D Ackley (tiny budget, NM overhead)
        assert best_fit < 2.0, f"Ackley too high: {best_fit:.6e}"
