# 约束松弛策略

> 确定了根因后，如何"最小干预"地松弛约束，使问题可行但不丢失太多解质量？

## 松弛决策树

```
诊断找到了冲突约束
     │
     ├─ 约束是否有物理/业务含义必须满足？
     │     │
     │     ├─ 是 → 检查数据是否正确 → 修复数据而非松弛
     │     │
     │     └─ 否 → 往下走
     │
     ├─ 冲突在"硬约束"还是"软约束"？
     │     │
     │     ├─ 软 → 直接调整惩罚权重
     │     │
     │     └─ 硬 → 考虑转换为软约束
     │
     ├─ 约束数量多还是少？
     │     │
     │     ├─ 少（1-3条）→ 松弛变量法
     │     │
     │     └─ 多（4+条）→ 分层松弛或拉格朗日法
     │
     └─ 解质量要求？
           │
           ├─ 严格最优 → 拉格朗日松弛 + 精确求解
           │
           └─ 近似可行 → 松弛变量 + 启发式修复
```

## 方法 1：松弛变量法（最常用）

将硬约束变为软约束：添加松弛变量 s >= 0，并在目标函数中加惩罚项。

### 单边松弛（<= 约束）
```
原约束: sum_j a_ij * x_j <= b_i
松弛后: sum_j a_ij * x_j - s_i <= b_i
目标项: + M * s_i   (s_i 在最小化目标中受罚)
```

### 双边松弛（== 约束）
```
原约束: sum_j a_ij * x_j == b_i
松弛后: sum_j a_ij * x_j + s_i_pos - s_i_neg == b_i
目标项: + M * (s_i_pos + s_i_neg)
```

### 代码示例（PuLP）
```python
import pulp

# 原约束
prob += pulp.lpSum([x[j] for j in J]) <= capacity, "orig_capacity"

# 松弛后
s = pulp.LpVariable("slack_capacity", lowBound=0)
prob += pulp.lpSum([x[j] for j in J]) - s <= capacity, "soft_capacity"
prob += 1000 * s  # 惩罚项
```

### 代码示例（Gurobi）
```python
# 直接使用 feasRelax API
model.feasRelax(
    relaxobjtype=0,        # 0=目标惩罚, 1=最小化松弛数
    minsumweight=None,
    vars=None,
    constrs=[capacity_constr],
    lbpen=None,
    ubpen=None,
    constrpen=[1.0]        # 约束违反的惩罚权重
)
```

### 罚函数系数怎么定？

| 约束类型 | 惩罚系数建议 | 说明 |
|---------|-------------|------|
| 容量约束 | obj_scale * 0.1 | 略高于正常目标项 |
| 时间窗 | obj_scale * 0.5 | 时间违规模拟成本较高 |
| 资源约束 | obj_scale * 1.0 | 资源约束通常较重要 |
| 逻辑约束 | obj_scale * 10 | 逻辑约束应尽量不违反 |

`obj_scale` = 目标函数中正常项的典型量级。
**经验法则**：惩罚系数应足够大，使松弛在无必要时不会被使用；但不要过大导致数值问题。

## 方法 2：分层松弛（Lexicographic Relaxation）

当有多组约束需要放松时，按优先级逐层处理。

```
优先级 1（最高）: 安全约束 —— 绝对不能违反
优先级 2: 法律/合同约束 —— 尽量不违反
优先级 3: 效率约束 —— 可以适度放松
优先级 4（最低）: 偏好约束 —— 放松影响小
```

### 实现步骤
1. 将约束按优先级分组
2. 先固定最高优先级，求解
3. 将最高优先级约束固定为已求解值，释放下一优先级
4. 依次求解

```python
# 伪代码：分层松弛
priority_groups = [
    (1, [constr_safety_1, constr_safety_2]),    # 最高优先级
    (2, [constr_capacity_1, constr_capacity_2]),
    (3, [constr_preference_1]),                  # 最低优先级
]

for priority, constraints in sorted(priority_groups):
    # 当前优先级及其以上的约束已被固定
    for c in constraints:
        make_soft(c, penalty=priority_penalty[priority])
    solve()
    # 如果可行 → 固定当前优先级的解 → 进入下一层
    # 如果不可行 → 报告无法满足当前优先级
```

## 方法 3：拉格朗日松弛（Lagrangian Relaxation）

将"难处理"约束放入目标函数，用拉格朗日乘子惩罚违反。

适用于：约束数量少、但破坏问题结构的约束（如 TSP 子环约束）。

### 基本形式
```
原问题:
min  f(x)
s.t. g_i(x) <= 0, i ∈ I     # 难处理的约束
     h_j(x) <= 0, j ∈ J     # 好处理的约束

拉格朗日松弛:
min  f(x) + sum_i λ_i * g_i(x)
s.t. h_j(x) <= 0, j ∈ J

其中 λ_i >= 0 是拉格朗日乘子
```

### 乘子更新（次梯度法）
```python
λ = [0] * len(I)           # 初始乘子
for k in range(max_iter):
    x = solve_relaxed(λ)    # 求解松弛后的问题
    violation = g_i(x)       # 计算违反度
    λ[i] = max(0, λ[i] + α_k * violation[i])  # 更新乘子
    α_k = α_0 / (1 + k)     # 步长衰减
```

## 方法 4：松弛量自动调节

不知道多少松弛够？用自动递增法：

```python
def auto_relax(model, violated_constraints, max_attempts=10):
    """自动递增松弛量，直到模型可行"""
    relax_level = 0.0
    for attempt in range(max_attempts):
        new_model = relax_constraints(model, violated_constraints, relax_level)
        status = solve(new_model)
        if status == OPTIMAL:
            return new_model, relax_level
        relax_level += 0.1  # 每次增加 10% 松弛量
    return None  # 可能需换修复策略
```

## 选择指南

| 场景 | 推荐方法 |
|------|---------|
| 1-3 条已知冲突约束 | 松弛变量法 |
| 多条约束但可排序 | 分层松弛 |
| 约束破坏了问题特殊结构 | 拉格朗日松弛 |
| 不确定哪条约束冲突 | 自动递增松弛 |
| 需要保证解质量 | 拉格朗日+精确求解 |
| 只需要任意一个可行解 | 松弛变量+大惩罚 |

## 松弛记录模板

每次做松弛后，建议记录以下信息（用于 Phase 4 验证）：

```
松弛记录:
  松弛类型: [软约束 / 分层 / 拉格朗日]
  目标约束: [约束名称列表]
  松弛量: [具体数值或百分比]
  惩罚系数: [具体值]
  惩罚系数确定依据: [与 obj_scale 的关系]
  是否影响最优性: [是/否]
  替代方案: [未采用的其他方案]
```
