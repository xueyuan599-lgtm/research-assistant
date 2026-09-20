# 不可行性诊断方法论

> 求解器说 INFEASIBLE ≠ 问题不可解。它告诉你的是约束系统有冲突，而你需要找到冲突点。

## 诊断工作流

```
求解器返回 INFEASIBLE
     │
     1. 确认终止状态（排除 INF_OR_UNBD / 超时误报）
     │
     2. 提取 IIS / Conflict 集 → 定位冲突约束
     │
     3. 对照根因分类表 → 识别根因
     │
     4. 修复 → 重求解 → 验证
     │
     5. 如仍 INFEASIBLE → 回到步骤 2
```

### Step 1: 确认终止状态

| 求解器状态 | 含义 | 后续操作 |
|-----------|------|---------|
| OPTIMAL | 找到最优解 | 验证合理性 |
| INFEASIBLE | 无可行解 | 进入诊断流程 |
| INF_OR_UNBD | 不可行或无界 | 先排除无界（加一个松的约束限） |
| TIME_LIMIT | 超时，可能有可行解 | 增加时间或改善初始解 |
| NUMERIC | 数值问题 | 检查 Big-M、缩放数据 |

### Step 2: 提取 IIS（Irreducible Inconsistent Subsystem）

IIS 是不可行模型中**最小的冲突约束子集**——去掉其中任何一个约束，模型就变可行。

**Gurobi**
```python
model.computeIIS()          # 计算 IIS
model.write("model.ilp")    # 写入 IIS 文件（只包含冲突约束）
# 阅读 model.ilp 查看具体冲突
for c in model.getConstrs():
    if c.IISConstr:
        print(f"冲突约束: {c.constrName}, 类型: {c.sense}")
```

**SCIP**
```python
model.freeTransform()
model.writeStartSolution()  # 保存当前状态
# SCIP 自动进行 conflict analysis
# 使用 SCIPgetConflictAnalyzer() 获取冲突约束
```

**OR-Tools (CP-SAT)**
```python
solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.INFEASIBLE:
    # CP-SAT 自动推断冲突（需开启 logging）
    solver.parameters.log_search_progress = True
```

### Step 3: 根因分类

| # | 根因类别 | 症状 | 典型示例 | 修复方法 |
|---|---------|------|---------|---------|
| 1 | **约束过定义** | IIS 含多个冗余约束 | `sum x = 1` + `sum x >= 1` + `sum x <= 1` | 删除冗余约束 |
| 2 | **变量界冲突** | IIS 含单个变量的上下界 | `x >= 5` + `x <= 3` | 调整变量界 |
| 3 | **Big-M 过小** | IIS 含 Big-M 约束 | `y = 1 → x <= M` 但 M < 最大可能 x | 增大 M |
| 4 | **数据不一致** | 需求 > 供应/容量 | `sum demand_i > sum supply_j` | 检查数据源 |
| 5 | **连通性问题** | 路径规划中图不连通 | 客户点在服务半径外 | 增加虚拟节点 |
| 6 | **索引错误** | 维度不匹配 | `x[i,j]` 的 i,j 集合范围不一致 | 检查索引定义 |
| 7 | **数值问题** | 求解器警告数值困难 | 系数跨 1e6 量级 | 缩放数据 |
| 8 | **逻辑链断裂** | If-Then-Else 约束未全覆盖 | 某些条件下无可行的变量赋值 | 补充逻辑约束 |

### Step 4: 常见不可行模式详解

#### 模式 1：约束过定义（最频繁）
症状：IIS 包含 2-3 个同时互斥的约束。
```
# 错误示例
约束1: x[1] + x[2] + x[3] == 1  # 恰好一个
约束2: x[1] + x[2] >= 1         # 至少一个 → 与约束1冲突吗？不冲突
约束3: x[1] + x[3] <= 0         # 只有 x[2] 可选 → 如果 x[2] 被其他约束禁止...
```
修复：删掉冗余约束，只保留必要的最小约束集。

#### 模式 2：Big-M 过小（路径规划中极常见）
```
# 错误示例：MTZ 子环消除
u[i] - u[j] + N * x[i,j] <= N - 1
# 如果 Big-M = N（城市数），当 u[i]=N-1, u[j]=0 时没问题
# 但如果 u 的界是 [0, N]，且 N 太小，某些路径组合无法表示
```
修复：确保 M >= max(u) - min(u) 的理论最大值。

#### 模式 3：VRP 中累积需求 > 总容量
```
# 错误示例
sum(customer_demand[i]) > sum(vehicle_capacity[k])
# 或某条路径上累积需求超过单车容量
```
修复：检查每辆车的容量约束，必要时增加车辆或使用软容量。

### Step 5: 诊断检查表（15 项）

每次遇到 INFEASIBLE 时，按以下清单逐一检查：

#### 模型层面
- [ ] 1. 变量是否存在？是否所有下标都在正确范围内？
- [ ] 2. 约束是否成对出现导致互斥？（== + >= + <=）
- [ ] 3. Big-M 系数是否足够大？（取理论最大值的 1.1-1.5 倍）
- [ ] 4. 二进制/整数变量是否在应有位置？（连续当整数用？）
- [ ] 5. 目标函数是否包含所有必要项？（无目标时某些求解器报错）

#### 数据层面
- [ ] 6. 需求/距离/成本矩阵是否有 NaN 或 Inf？
- [ ] 7. 总需求是否超过总供给/容量？
- [ ] 8. 图的连通性是否满足？（距离矩阵对称吗？）
- [ ] 9. 时间窗是否合理？（开始 < 结束？）
- [ ] 10. 数据规模是否符合预期？（缩放问题？）

#### 求解器层面
- [ ] 11. 是否确认了终止状态？（不是 INF_OR_UNBD？）
- [ ] 12. 是否提取了 IIS？IIS 是否提供了明确线索？
- [ ] 13. 求解器参数是否合理？（MIPFocus、MIPGap 等）

#### 逻辑层面
- [ ] 14. 是否忘记了某些约束？（如 TSP 的子环消除）
- [ ] 15. 逻辑链是否闭合？（If-Then-Else 的所有分支覆盖？）

### 求解器命令速查

| 操作 | Gurobi | SCIP | OR-Tools | PuLP |
|------|--------|------|----------|------|
| 计算 IIS | `computeIIS()` | SCIPgetConflictAnalyzer | (built-in) | 导出 LP + SCIP |
| 写 IIS 文件 | `write("model.ilp")` | `writeConflictStat()` | — | — |
| Feasibility Relax | `feasRelax()` | — | `soft_constraint` | — |
| 关闭 presolve | `Presolve=0` | `presolving/enable=FALSE` | — | — |
| 输出 LP | `write("model.lp")` | `writeProblem("model.lp")` | `model.Write("model.lp")` | `prob.writeLP("model.lp")` |
| 设置 MIP start | `model.Start` | `readStartSolution()` | `solution_hint` | `setInitialSolution()` |

## 总结

诊断的核心是：**用 IIS 缩小排查范围，用根因分类指导修复方向，用检查表确保不遗漏。** 大多数 INFEASIBLE 问题在 15 分钟内可定根因。
