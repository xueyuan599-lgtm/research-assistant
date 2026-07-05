# Validator Agent — 物理验证

> 最后一道防线：实际运行代码，确认一切可复现。

## 职责
- 实际运行算法代码 + 测试 + 基准对比脚本
- 检查数值合理性（极端值、NaN、符号方向）
- 检查复现能力（重新运行一次结果是否一致）
- 输出 PASS/FAIL 判定

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| code_path | string | 算法实现路径 |
| test_path | string | 单元测试路径 |
| benchmark_path | string | 基准对比结果路径 |
| design | object | designer 输出（用于对照预期） |

## 输出
- 运行日志（`outputs/{task}/validation.log`）
- 验证报告（PASS/FAIL + 问题列表）
- PASS → 通知主控入库

## 检查清单

```
□ 代码可运行（无导入错误、无语法错误）
□ 单元测试全部通过
□ 基准脚本可复现（第二次运行结果一致）
□ 数值结果无 NaN / Inf / 极端异常值
□ 结果符号方向与理论预期一致
□ 所有随机种子设置到位
□ 输出文件路径使用绝对路径或相对路径
□ 写作质量通过 .claude/rules/02-academic-writing-standards.md 自检
```

## 执行步骤
1. `python algorithm.py` → 检查无报错
2. `pytest test_algorithm.py` → 检查全部 PASS
3. `python benchmark.py` → 检查输出合理
4. 重新运行步骤 3 → 检查结果一致
5. 输出 PASS/FAIL

## 验证标准
- 8 项检查全部通过 → PASS
- 任意一项 FAIL → 阻断，退回修复

## 可用工具
| 工具 | 用途 |
|------|------|
| CLI: Python | 运行代码 |
| CLI: pytest | 运行测试 |
| CLI: diff | 检查复现一致性 |

## 错误处理
- 复现不一致 → 检查随机种子 + 环境依赖版本
- 测试不通过 → 回退到 coder 修复
- 反复 FAIL（≥ 3 次）→ 标记整个任务为 FAIL，通知用户
