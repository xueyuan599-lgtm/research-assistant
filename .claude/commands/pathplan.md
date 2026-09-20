# /pathplan

一键启动路径规划问题的诊断-松弛-修复流水线。求解器报 INFEASIBLE / 无可行解时，快速定位并修复。

## 用法

```
/pathplan <问题描述> [--solver Gurobi/SCIP/OR-Tools] [--type TSP/CVRP/VRPTW/调度]
```

## 示例

```
/pathplan 20个客户点配送，每辆车容量100，求解器报无可行解，帮我诊断修复
/pathplan TSP 问题，15 个城市，报 INFEASIBLE
/pathplan VRPTW，50个客户，时间窗太紧求解器找不到可行解
```

## 执行流程

1. 加载 `.claude/skills/path-planning/SKILL.md`
2. 按四阶段流水线执行：
   ```
   Phase 1: 诊断 → 提取 IIS → 定位冲突约束 → 根因分类
   Phase 2: 松弛 → 松弛变量/分层松弛 → 设惩罚系数
   Phase 3: 修复 → 两阶段法/构造启发式 → 生成可行解
   Phase 4: 验证 → 松弛代价分析 → 敏感性分析 → PASS/FAIL
   ```
3. 引用 `skills/path-planning/references/` 下的参考文档
4. 引用 `optimization-validation-framework` 做验证
5. 输出：可行解 + 松弛报告 + 问题类型模板

## 关联 Skill

该命令加载：
- `path-planning` — 核心不可行性诊断与修复方法论
- `bio-inspired-algo` — 如需元启发修复组件
