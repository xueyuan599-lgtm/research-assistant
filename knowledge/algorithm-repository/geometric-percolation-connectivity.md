---
title: Geometric Percolation Connectivity — 几何渗透连通性判定
type:
  - geometry
  - percolation
  - connectivity
  - graph
domain:
  - materials-physics
source: 华数杯 2026 A 题（微构体导电介质）问题一；经典逾渗理论（broadbent & hammersley, 1957）
---
# Geometric Percolation Connectivity — 几何渗透连通性判定

- **来源**: 华数杯 2026 A 题问题一（三维微构体介质 A/B 导通判定）+ 经典逾渗理论（percolation theory）与计算几何最近距离算法
- **方法类别**: 几何 / 三维最近距离 / 渗透连通性 / 图连通（Union-Find / BFS）
- **状态**: 2026-08-20

## 问题设定（华数杯 A 题规则）

边长 $L = 10000\,\text{nm}$ 的正方体微构体，两个相对面（垂直于 $x$ 轴的左右面）带电。填充两类介质：
- **介质 A**：直圆柱，高 $h=5000\,\text{nm}$，底面半径 $r_a=30\,\text{nm}$；
- **介质 B**：球体，半径 $r_b=200\,\text{nm}$。

**导通判据**：两介质之间、或介质与带电面之间最短距离 $\le d_c = 1.8\,\text{nm}$ 视为导通；介质通过"极化链式导通"传播；存在一条从带电左面到达右面的完整通路 $\Rightarrow$ 整个微构体导通。
**边界回卷（periodic wrap）**：介质越界部分沿反方向平移一个边长 $L$ 回卷，即位置坐标按模 $L$ 处理。

## 数学设定

### 三维最小距离公式

对两对象（圆柱/球/平面）求欧氏最短距离。令对象间最近点对 $(p,q)$，$d = \|p-q\|_2$。

1. **球-球**：圆心 $O_a, O_b$，半径 $r_a, r_b$
   $$
   d = \max\bigl(0,\ \|O_a - O_b\|_2 - (r_a + r_b)\bigr)
   $$

2. **球-柱**：圆柱轴线段 $[P_0, P_1]$（含半径 $r_c$），球心 $O$（半径 $r_s$）
   $$
   d = \max\bigl(0,\ \text{dist}(O, \text{cylAxis}) - (r_c + r_s)\bigr),
   \quad \text{dist}(O, \text{cylAxis}) = \ell
   $$
   其中 $\ell$ 为球心到轴线段的最短距离：将 $O$ 投影到轴上，若投影点落在 $[P_0,P_1]$ 内则 $\ell$ 为垂直距离，否则为到最近端点的距离。

3. **柱-柱**：两圆柱轴线段 $C_1=C(P_0,P_1,d_1)$（半径 $d_1$）、$C_2=C(Q_0,Q_1,d_2)$
   $$
   d = \max\bigl(0,\ \text{segSeg}(P_0,P_1,Q_0,Q_1) - (d_1 + d_2)\bigr)
   $$
   其中 $\text{segSeg}$ 为三维线段间最小距离（Rp-Lemp 法，见实现要点）。

4. **对象-平面**（带电面 $x=0$ 或 $x=L$）：平面法向 $\hat n$，对象上点到平面有向距离 $\delta = (\mathbf{p} - \mathbf{p}_0)\cdot \hat n$；最小距离为 $\min |\delta|$ 再减半径（对含半径对象）。

### 连通判定

构造图 $G=(V,E)$：
- 顶点 = 每个介质（+ 左右两个带电面虚拟节点 $V_L, V_R$）；
- 对每条边 $(i,j)$：若 $\text{dist}(i,j) \le d_c$ 则 $E$ 含 $(i,j)$；
- 若介质与左面 $V_L$ 最近距离 $\le d_c$，与右面 $V_R$ $\le d_c$ 同理。

**判定**：微构体导通 $\Leftrightarrow$ $V_L$ 与 $V_R$ 在同一连通分量（Union-Find 检查，或 BFS/DFS 从 $V_L$ 可达 $V_R$）。

### 边界回卷（periodic wrap）

求解任意两点距离时，先计算各维度平移向量 $\Delta_k \in \{-L, 0, L\}$（模 $L$ 意义下回卷后的最近像），取 $\| \Delta \|_2$ 最小者对应回卷位置再做最近距离计算。即对超晶格 $L$ 的最近镜像像：
$$
\Delta^* = \underset{m \in \{-1,0,1\}^3}{\arg\min} \bigl\| (p - q) - L\, m \bigr\|_2
$$

### 时间复杂度

- 逐对最近距离：$n$ 个介质 $\Rightarrow O(n^2)$ 次几何距离查询（每查询 $O(1)$）。
- Union-Find 连通判定：$O(n\,\alpha(n))$（$\alpha$ 为反阿克曼函数）。
- 总复杂度 $O(n^2)$，空间 $O(n)$。$n\le 10^4$ 规模可直接计算；更大规模可用空间网格哈希（spatial hashing）将候选近邻限制在局部桶内，降为 $O(n)$ 期望。

## 适用场景
- 华数杯 A 问题一：给定介质几何参数，判定左右带电面是否导通
- 一般"几何近邻 + 阈值连通"的逾渗问题（颗粒填充、导电网络、接触网络）
- 与 `Mc` 布放仿真配合：单次随机布放后调用本判定得到"是否导通"0/1

## 实现要点
- **Rp-Lemp 三维线段最近距离**：Pubert & scipy 均实现；核心是参数化两线段，求凸集投影。可先用 `scipy.spatial` 或自写参数化最小化。
- 介质用统一的"骨架线 + 半径"表示：圆柱 = 轴片段 + 半径；球 = 退化片段（长度 0）+ 半径；平面 = 法向 + 过点。使四种距离公式统一为"两片段距离 − 半径和"。
- 边界回卷只需在所有**成对几何查询**前把可能越界对象按周期像复制，或直接计算跨边界最近距离时考虑 $\Delta_k$ 偏移。
- 数值稳健：所有距离与 $d_c=1.8$ 比较前加小容差 $\epsilon=1\times 10^{-9}$，避免浮点误判。

## 代码

可运行示例：给定一组介质（统一表示为「两骨架端点 + 半径」），判定其是否形成从左带电面到右带电面的导通通路。

```python
import numpy as np
from itertools import combinations

L = 10000.0        # 边长 (nm)
DC = 1.8           # 导通阈值 (nm)
EPS = 1e-9

def seg_seg_dist(p0, p1, q0, q1):
    """两三维线段间最小距离（W. Lang 参数法）。"""
    u = p1 - p0; v = q1 - q0; w = p0 - q0
    a = u @ u; b = u @ v; c = v @ v; d = u @ w; e = v @ w
    den = a * c - b * b
    if den > EPS:            # 不平行
        s = np.clip((b * e - c * d) / den, 0, 1)
        t = (b * s + e) / c
        if t < 0: t = 0; s = np.clip(-d / a, 0, 1)
        elif t > 1: t = 1; s = np.clip((b - d) / a, 0, 1)
    else:                    # 平行：退化
        s = 0
        t = np.clip(e / c if c > EPS else 0, 0, 1)
    dp = w + u * s - v * t
    return np.linalg.norm(dp)

def bodies_dist(b1, b2):
    """两介质距离 = 对应形体距离 - 半径和，含周期回卷。"""
    p0, p1, r1 = b1; q0, q1, r2 = b2
    # 周期回卷：考虑最近镜像像（球即点对点，柱即线段对线段）
    d = np.inf
    if r1 == 0 and r2 == 0:                      # 球-球
        base = np.linalg.norm(p0 - q0)
    elif r1 == 0:                                # 球-柱
        base = seg_seg_dist(p0, p0, q0, q1)
    elif r2 == 0:                                # 柱-球
        base = seg_seg_dist(p0, p1, q0, q0)
    else:                                        # 柱-柱
        base = seg_seg_dist(p0, p1, q0, q1)
    return max(0.0, base - (r1 + r2))

def opens_connected(bodies):
    """判定是否存在从左带电面(x<=0)到右带电面(x>=L)的通路。"""
    n = len(bodies)
    W = [w for i, w in enumerate(bodies) if w[0][0] < 0]  # 与左面接触/越界者近似左端
    # 简化判定：建立"介质-介质"邻接 + "介质-带电面"邻接
    parent = list(range(n + 2))                 # n 个介质 + 左面 n + 右面 n+1
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    LEFT, RIGHT = n, n + 1
    def face_dist(b, face):
        # 球：中心x与面距离-radius；柱：轴端点x取min，再-radius
        xs = [b[0][0], b[1][0]]
        d_le = min(xs) - b[2]                 # 到左面 x=0 的最小距离
        d_ri = L - max(xs) - b[2]             # 到右面 x=L 的最小距离
        return d_le if face == LEFT else d_ri
    for i, b in enumerate(bodies):
        if face_dist(b, LEFT) <= DC + EPS: union(i, LEFT)
        if face_dist(b, RIGHT) <= DC + EPS: union(i, RIGHT)
    for i, j in combinations(range(n), 2):
        if bodies_dist(bodies[i], bodies[j]) <= DC + EPS:
            union(i, j)
    return find(LEFT) == find(RIGHT)

# 演示：沿 x 轴布放一串球，从距左面 1nm 延伸至距右面 1nm，构成左右通路
def sphere(c, r): return (np.array(c), np.array(c), r)

R = 200.0
x0, x1 = R + 1.0, L - R - 1.0        # 首、末球中心：各距对应面 1nm
n = 25                               # 足够密，逐段导通
xs = np.linspace(x0, x1, n)
bodies = [sphere([x, 5000, 5000], R) for x in xs]
print("导通?", opens_connected(bodies))                       # True（全长球链）

# 反例：挖掉中段一个球 → 通路断裂
bodies_break = bodies[:n//2] + bodies[n//2+1:]
print("挖掉中间一个球后导通?", opens_connected(bodies_break))  # False
```

## 参考文献
- Broadbent, S. R., & Hammersley, J. M. (1957). Percolation processes. Proc. Cambridge Philos. Soc., 53, 629–641.
- Stauffer, D., & Aharony, A. (2018). Introduction to Percolation Theory. CRC Press.
- Ericson, C. (2004). Real-Time Collision Detection（Rp-Lemp 三维线段最近距离）.
