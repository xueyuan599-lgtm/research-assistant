---
name: path-planning
description: 路径规划问题的建模与求解方法论，重点解决求解器报"无可行解"(INFEASIBLE)的诊断、松弛、修复与验证。覆盖 TSP、CVRP、VRPTW、调度等经典问题类型。
license: MIT
metadata:
  skill-author: research-assistant
---

# 路径规划优化 Skill — 当求解器说"No"

## 概述

路径规划问题（TSP、VRP、物流调度等）中，建模者最常遇到的卡点是求解器返回 **INFEASIBLE / 无可行解**。

本 skill 的核心命题是：**INFEASIBLE 不是终点，而是一个诊断信号。** 它意味着约束系统存在冲突，但冲突点往往可以通过系统化方法定位并修复。

本 skill 提供一套**四阶段流水线**，将"无可行解"从死胡同变为可操作的诊断修复流程：

```
求解器报告 INFEASIBLE
     │
     ├─ Phase 1: 诊断 → 哪里冲突？模型错/数据错/约束冲突？
     │                 [references/infeasibility-diagnosis.md]
     │
     ├─ Phase 2: 松弛 → 哪些约束可放松？怎么放松？
     │                 [references/constraint-relaxation.md]
     │
     ├─ Phase 3: 修复 → 如何找到第一个可行解？
     │                 [references/repair-and-hybrid-methods.md]
     │
     └─ Phase 4: 验证 → 松弛代价多大？解是否合理？
                     [Cross-ref: optimization-validation-framework]
```

## Trigger Keywords

| 中文 | English |
|------|---------|
| 路径规划 | path planning |
| TSP, 旅行商 | traveling salesman |
| VRP, 车辆路径 | vehicle routing |
| 物流调度 | logistics scheduling |
| 无可行解 | INFEASIBLE |
| 约束太紧 | constraint too tight |
| 求解器报错 | solver error |
| 配送路径优化 | delivery route optimization |
| 无人机路径 | UAV path planning |
| 整数规划(MILP) | mixed-integer programming |

## 核心流水线

### Phase 1: 诊断（Diagnose）

**目标**：确定 INFEASIBLE 的根本原因 — 是模型建错了，数据有问题，还是约束本身冲突？

诊断步骤：
1. 查看求解器终止状态（INFEASIBLE vs INF_OR_UNBD vs 超时）
2. 提取 IIS（Irreducible Inconsistent Subsystem）定位冲突约束集
3. 对照根因分类表逐一排查
4. 验证修复方案

详细方法论 → `references/infeasibility-diagnosis.md`

### Phase 2: 松弛（Relax）

**目标**：找到"最小干预"的松弛方案，在可行性与解质量之间取得平衡。

松弛策略决策：
- 哪个约束最可能是过度约束？→ 优先松弛
- 如何松弛？→ 松弛变量 / 分层优先级 / 拉格朗日松弛
- 惩罚系数设多少？→ 与目标函数量级匹配

详细方法论 → `references/constraint-relaxation.md`

### Phase 3: 修复（Repair）

**目标**：当松弛后求解器仍然找不到可行解，使用构造启发式或元启发方法直接生成可行解。

修复方法：
- 两阶段法（Phase 1 = 最小化不可行, Phase 2 = 优化原目标）
- 构造启发式（最近邻、Savings、插入法）
- 大邻域搜索（Destroy → Repair）
- Warm-start + 精确求解器混合

详细方法论 → `references/repair-and-hybrid-methods.md`

### Phase 4: 验证（Validate）

**目标**：确认可行解的合理性，量化松弛代价。

- 松弛了什么？松弛程度多少？
- 松弛的边际代价（影子价格分析）
- 敏感性分析（关键参数波动下解是否仍可行）
- 交叉验证：优化验证框架 (`knowledge/optimization-validation-framework.md`)

## 问题类型速查

| 问题类型 | 常见不可行原因 | 首选修复策略 | 参考 |
|---------|--------------|-------------|------|
| TSP | MTZ 子环约束缺失、Big-M 过小 | DFJ + lazy constraints | `references/problem-templates.md#tsp` |
| CVRP | 车辆容量不足、需求数据溢出 | 松弛容量 + 插入启发式 | `references/problem-templates.md#cvrp` |
| VRPTW | 时间窗不可调和 | 软时间窗松弛 | `references/problem-templates.md#vrptw` |
| Job Shop | 顺序/资源约束冲突 | 松弛机器容量 + 优先级调度 | `references/problem-templates.md#调度` |
| 分配问题 | 成本矩阵不一致、配对冲突 | 数据检查 + 松弛整数约束 | `references/problem-templates.md#分配` |

## 工具选型

| 求解器 | 诊断能力 | 松弛支持 | 适用场景 | 许可 |
|--------|---------|---------|---------|------|
| **Gurobi** | IIS 分析、feasibility relaxation 自动工具 | 内置 `Model.feasRelax()` | 大规模 MILP | 商业（有学术许可） |
| **SCIP** | Conflict analysis、约束处理 | 内置约束处理器 | 通用 MILP | 开源 |
| **OR-Tools** | CP-SAT 验证、Solution hint | Routing library 内置软约束 | VRP / 调度 | 开源 |
| **PuLP/CBC** | 基本诊断（建议搭配 SCIP） | 手动松弛 | 小规模 / 教学 | 开源 |
| **自定义启发式** | N/A | 完全灵活 | 超大规模问题 | — |

## Agent 集成

当秘书 Agent 识别为路径规划类型时：

```
Secretary Agent → "无可行解, TSP, VRP" 等关键词
     │
Orchestrator → PATH_PLANNING 领域标识
     │
     ├─ Algorithm Agent (formalizer + coder) → 形式化建模 + 求解
     ├─ path-planning skill → 不可行诊断 + 松弛 + 修复方法论
     ├─ bio-inspired-algo skill → 元启发修复组件（如需）
     └─ optimization-validation-framework → 验证
```

## 最佳实践

1. **从小实例开始**：用 N=5 的 TSP 验证模型，再扩展到 N=100
2. **保存模型文件**：求解前保存 LP 格式（`model.write("model.lp")`），便于事后分析
3. **数据模型分离**：数据变更不应修改模型代码
4. **Warm-start**：先用构造启发式生成初始解，再喂给精确求解器
5. **逐一测试约束**：先不加约束跑，看目标值是否合理，再加约束逐个检查
6. **记录松弛**：每当做了松弛，记录 WHAT、WHY、BY_HOW_MUCH
7. **检查 Big-M**：Big-M 应刚好大到能激活/失活约束，不要取 1e6

## 参考文档

| 文件 | 内容 |
|------|------|
| `references/infeasibility-diagnosis.md` | 不可行性诊断方法论、IIS 分析、根因分类、诊断检查表 |
| `references/constraint-relaxation.md` | 松弛策略目录、松弛变量、分层松弛、罚函数调参 |
| `references/repair-and-hybrid-methods.md` | 修复方法、两阶段法、构造启发式、LNS、元启发修复 |
| `references/problem-templates.md` | TSP/CVRP/VRPTW/调度问题模板 + 不可行原因表 |

## 关联资源

- [优化验证框架](../../research-assistant/knowledge/optimization-validation-framework.md)
- [Bio-Inspired Algorithm Design](../bio-inspired-algo/SKILL.md)
- [00-tool-selection 规则](../../rules/04-tool-selection.md)
