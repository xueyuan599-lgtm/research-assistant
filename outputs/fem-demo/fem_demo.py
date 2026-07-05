"""FEM Demo: Poisson equation -∇²u = f 在单位正方形上, u=0 边界条件"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skfem import *
from skfem.helpers import dot, grad

# ── 1. 创建网格 (三角形网格, 细化程度) ──
mesh = MeshTri.init_tensor(
    np.linspace(0, 1, 15),
    np.linspace(0, 1, 15)
)
for _ in range(2): mesh = mesh.refined()  # 加密 2 次

# ── 2. 定义有限元空间 (P1: 线性单元) ──
basis = Basis(mesh, ElementTriP1())

# ── 3. 组装刚度矩阵和右端项 ──
@BilinearForm
def laplacian(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm
def rhs(v, w):
    x, y = w["x"].value[0], w["x"].value[1]
    f = 8 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)
    return f * v

A = asm(laplacian, basis)
b = asm(rhs, basis)

# ── 4. 施加边界条件 (u=0 在边界上) ──
dofs = basis.get_dofs(
    lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1)
            | np.isclose(x[1], 0) | np.isclose(x[1], 1)
)
u = solve(*condense(A, b, D=dofs))

# ── 5. 计算精确解作对比 ──
def exact(x):
    return np.sin(2*np.pi * x[0]) * np.sin(2*np.pi * x[1])

u_exact = exact(mesh.p)

# ── 6. 误差 ──
error = np.sqrt(np.sum((u - u_exact)**2) / np.sum(u_exact**2))
print(f"Relative L2 Error: {error:.6f}")

# ── 7. 出版级可视化 ──
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 数值解
ax = axes[0]
ax.tripcolor(mesh.p[0], mesh.p[1], mesh.t.T, u,
             shading="gouraud", cmap="viridis", rasterized=True)
ax.set_title("Numerical Solution (P1 FEM)", fontsize=10)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(ax.collections[0], ax=ax, shrink=0.6, label="u(x,y)")

# 精确解
ax = axes[1]
X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
Z = exact(np.array([X, Y]))
cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
ax.set_title("Exact Solution", fontsize=10)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(cf, ax=ax, shrink=0.6, label="u(x,y)")

# 误差分布
ax = axes[2]
err = np.abs(u - u_exact)
ax.tripcolor(mesh.p[0], mesh.p[1], mesh.t.T, err,
             shading="gouraud", cmap="Reds", rasterized=True)
ax.set_title(f"Absolute Error (Rel={error:.4f})", fontsize=10)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(ax.collections[0], ax=ax, shrink=0.6, label="Error")

fig.suptitle(r"Poisson Equation $-\nabla^2 u = 8\pi^2\sin(2\pi x)\sin(2\pi y)$", y=1.02, fontsize=11)

plt.tight_layout()
plt.savefig("E:/wuyi/数学建模半自动/research-assistant/outputs/fem_poisson_demo.png", dpi=200, bbox_inches="tight")
print("OK")
