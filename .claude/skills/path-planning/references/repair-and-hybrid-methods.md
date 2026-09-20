# 修复与混合方法

> 当松弛后求解器仍然找不到可行解，需要直接从问题结构出发构造可行解。

## 方法 1：两阶段法（Phase 1 / Phase 2）

最经典且最通用的可行解寻找方法。

### Phase 1: 最小化不可行
将原目标函数替换为"总的约束违反量"，求解这个辅助问题：

```python
# Phase 1：最小化总松弛
phase1_obj = pulp.lpSum([s_i for s_i in slack_vars])
phase1_prob += phase1_obj
phase1_prob.solve()

# 记录 Phase 1 的解作为初始可行解
feasible_solution = {v.name: v.varValue for v in phase1_prob.variables()
                     if "slack" not in v.name}
```

### Phase 2: 固定可行解，优化原目标
用 Phase 1 得到的解做 warm-start，重新优化原问题：

```python
# Phase 2：用 Phase 1 的解 warm-start
phase2_prob = original_model
for v in phase2_prob.variables():
    if v.name in feasible_solution:
        v.setInitialValue(feasible_solution[v.name])

# 如果可行解在 Phase 1 中所有松弛变量为 0 → 原问题可行
# 如果松弛变量 > 0 → 该约束被违反
```

### 适用场景
- 约束较多，难以手工定位冲突
- 需要系统化的修复而非启发式猜测
- 与松弛变量法配合，一次得到"最接近可行"的解

## 方法 2：构造启发式

直接从问题结构出发，用贪心算法构造可行解。

### 最近邻法（TSP）
```python
def nearest_neighbor(dist_matrix, start=0):
    """构造 TSP 可行解（总是可行的）"""
    n = len(dist_matrix)
    visited = [False] * n
    route = [start]
    visited[start] = True
    
    for _ in range(n - 1):
        last = route[-1]
        # 找最近的未访问城市
        next_city = min(
            [j for j in range(n) if not visited[j]],
            key=lambda j: dist_matrix[last][j]
        )
        route.append(next_city)
        visited[next_city] = True
    
    route.append(start)  # 回到起点
    return route  # 始终是可行解
```

### Savings 法（VRP）
```python
def clarke_wright_savings(dist_matrix, demands, capacity):
    """构造 CVRP 可行解"""
    n = len(dist_matrix)
    # 计算 savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = dist_matrix[0][i] + dist_matrix[0][j] - dist_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)  # 按 savings 从大到小
    
    # 初始化：每个客户自己一条路径
    routes = [[i] for i in range(1, n)]
    route_loads = [demands[i] for i in range(1, n)]
    
    # 合并路径
    for s, i, j in savings:
        ri = find_route(routes, i)
        rj = find_route(routes, j)
        if ri != rj and route_loads[ri] + route_loads[rj] <= capacity:
            # 合并 ri 和 rj
            routes[ri] = merge_routes(routes[ri], routes[rj], i, j)
            route_loads[ri] += route_loads[rj]
            routes[rj] = []  # 标记为已合并
    
    return [r for r in routes if r]  # 可行路径集
```

### 插入法（VRPTW 调度）
```python
def regret_insertion(customers, routes, cost_matrix, time_windows):
    """后悔插入：选择"不现在插入损失最大"的客户"""
    unassigned = set(customers)
    while unassigned:
        best_customer = None
        best_regret = -inf
        best_position = None
        best_route = None
        
        for c in unassigned:
            # 计算 c 在所有可能插入位置的增量成本
            costs = []
            for r_idx, route in enumerate(routes):
                for pos in range(len(route)):
                    delta = insertion_cost(c, route, pos, cost_matrix, time_windows)
                    if delta is not None:  # 插入后时间窗可行
                        costs.append((delta, r_idx, pos))
            
            costs.sort()
            if len(costs) >= 2:
                regret = costs[1][0] - costs[0][0]  # 次优 - 最优
            else:
                regret = 0  # 只有一种选择或没有选择
            
            if regret > best_regret:
                best_regret = regret
                best_customer = c
                best_position = costs[0] if costs else None
        
        if best_position:
            _, r_idx, pos = best_position
            routes[r_idx].insert(pos, best_customer)
            unassigned.remove(best_customer)
    
    return routes
```

### 启发式对比

| 方法 | 适用 | 质量 | 速度 | 是否保证可行 |
|------|------|------|------|------------|
| 最近邻 | TSP | 中等 | 极快 O(n²) | 是 |
| Savings | CVRP | 好 | 快 O(n²log n) | 是（容量允许时） |
| 插入法 | VRPTW | 好 | 中等 O(n³) | 是（时间窗允许时） |
| 随机 + 修复 | 通用 | 差 | 快 | 否 |

## 方法 3：大邻域搜索（LNS）

构造启发式给了你一个可行解但不一定好。LNS 通过 destroy → repair 循环改进。

```python
def large_neighborhood_search(initial_solution, destroy, repair, max_iter=100):
    """LNS：销毁部分解 → 重建 → 接受更好解"""
    current = copy.deepcopy(initial_solution)
    best = copy.deepcopy(current)
    
    for iteration in range(max_iter):
        # Destroy: 移除部分决策
        destroyed = destroy(current, removal_ratio=0.3)
        
        # Repair: 重新填充
        candidate = repair(destroyed)
        
        # Accept: 如果更好则接受
        if candidate.cost < current.cost:
            current = candidate
            if candidate.cost < best.cost:
                best = candidate
    return best
```

常用 destroy 操作：
- **Random removal**：随机移除 k 个客户
- **Shaw removal**：移除相似（空间/时间）的客户
- **Worst removal**：移除代价最高的客户
- **Route removal**：移除整条路径

常用 repair 操作：
- **Greedy insertion**：找最低增量成本位置插入
- **Regret insertion**：后悔插入
- **Exact subproblem**：对被移除的客户，用 MILP 精确求解最优插入方案

## 方法 4：元启发修复

当问题高度约束时，用元启发式（GA/SA）搜索可行空间。

```python
def simulated_annealing_feasibility(initial, max_iter=10000):
    """SA 搜索可行解（以最小化不可行为目标）"""
    current = initial
    violation = compute_violations(current)
    best_solution = current
    best_violation = violation
    
    T = 1.0  # 初始温度
    T_min = 0.001
    
    for k in range(max_iter):
        if violation == 0:
            return current  # 已得到可行解
        
        neighbor = random_move(current)
        new_violation = compute_violations(neighbor)
        delta = new_violation - violation
        
        if delta < 0 or random() < exp(-delta / T):
            current = neighbor
            violation = new_violation
            if violation < best_violation:
                best_solution = current
                best_violation = violation
        
        T = max(T * 0.995, T_min)
    
    return best_solution  # 返回最接近可行的解
```

## 方法 5：Warm-start（混合方法）

用构造启发式生成初始解，喂给精确求解器作为 MIP start。

### Gurobi
```python
# 构造启发式生成初始解
heuristic_solution = nearest_neighbor(dist_matrix)

# 设为 MIP start
start_vals = {}
for i in range(n):
    for j in range(n):
        # x[i,j] = 1 如果路径从 i 到 j
        # ... 将 heuristic_solution 映射到变量
        start_vals[x[i, j]] = 1 if condition else 0

# 设置 start
for var, val in start_vals.items():
    var.Start = val

model.optimize()
```

### PuLP
```python
heuristic_solution = savings_algorithm(demands, dist_matrix, capacity)
for v in prob.variables():
    if v.name in mapping:
        v.setInitialValue(mapping[v.name])
prob.solve(pulp.PULP_CBC_CMD(msg=True))
```

## 选择指南

| 情况 | 推荐方法 |
|------|---------|
| 首次面对新问题 | 两阶段法（系统化，通用） |
| 已知问题结构（TSP/VRP） | 构造启发式（快且问题专用） |
| 已有可行解但质量差 | LNS（提升质量） |
| 约束极其复杂 | SA/GA 元启发（搜索空间大） |
| 有精确求解器且小规模 | Warm-start（精确最优） |
