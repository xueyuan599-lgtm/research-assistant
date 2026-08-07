"""
PNO-GWO v2.0 — 黏菌管网络-灰狼混合优化算法 使用示例
运行: python demo.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from algorithm import PhysarumNetworkOptimizer


def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def ackley(x):
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


BENCHMARKS = {
    'Sphere': {'func': sphere, 'bounds': (-100, 100), 'optimum': 0},
    'Rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12), 'optimum': 0},
    'Rosenbrock': {'func': rosenbrock, 'bounds': (-10, 10), 'optimum': 0},
    'Ackley': {'func': ackley, 'bounds': (-32, 32), 'optimum': 0},
}


# ======================================================================
# 演示 1: 单函数优化（含与原始 PNO 对比）
# ======================================================================
print("=" * 60)
print("演示 1: PNO-GWO v2.0 优化 Sphere (10维, FES=3000)")
print("=" * 60)

np.random.seed(42)
pno = PhysarumNetworkOptimizer(
    n_pop=20, n_dim=10,
    bounds=(-100, 100), max_fes=3000
)
best_x, best_fit, conv = pno.optimize(sphere, verbose=True)
print(f"\n最优适应度: {best_fit:.6e}")
print(f"理论最优: 0")

# ======================================================================
# 演示 2: 多函数对比
# ======================================================================
print("\n\n" + "=" * 60)
print("演示 2: PNO-GWO 在 4 个基准函数上的表现 (10D, 3000 FES)")
print("=" * 60)

results = {}
for name, cfg in BENCHMARKS.items():
    np.random.seed(42)
    pno = PhysarumNetworkOptimizer(
        n_pop=20, n_dim=10,
        bounds=cfg['bounds'], max_fes=3000
    )
    _, best_fit, conv = pno.optimize(cfg['func'], verbose=False)
    results[name] = {'best': best_fit, 'conv': conv}
    print(f"  {name:12s} → Best: {best_fit:.6e}  (optimum: {cfg['optimum']})")

# ======================================================================
# 演示 3: 参数影响
# ======================================================================
print("\n\n" + "=" * 60)
print("演示 3: p_gwo_final (GWO占比) 参数敏感性")
print("=" * 60)

pgwo_values = [0.3, 0.5, 0.7, 0.85, 0.95]
for pgwo in pgwo_values:
    np.random.seed(42)
    pno = PhysarumNetworkOptimizer(
        n_pop=20, n_dim=10,
        bounds=(-100, 100), max_fes=3000,
        p_gwo_final=pgwo
    )
    _, best_fit, _ = pno.optimize(sphere, verbose=False)
    print(f"  p_gwo_final={pgwo:.2f} → Sphere: {best_fit:.6e}")

# ======================================================================
# 演示 4: 收敛曲线图
# ======================================================================
print("\n\n" + "=" * 60)
print("演示 4: 绘制收敛曲线 → figures/convergence.png")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('PNO-GWO v2.0 Convergence Curves (10D, 3000 FES)', fontsize=14)

for ax, (name, cfg) in zip(axes.flat, BENCHMARKS.items()):
    conv = results[name]['conv']
    ax.plot(conv, linewidth=1.5, color='#2c7bb6')
    ax.set_title(name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Fitness')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=cfg['optimum'], color='r', linestyle='--',
               alpha=0.5, label=f"opt={cfg['optimum']}")
    ax.legend()

plt.tight_layout()
fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / 'convergence.png', dpi=150, bbox_inches='tight')
print(f"  → 已保存 figures/convergence.png")
plt.close()

print("\nDemo completed!")
