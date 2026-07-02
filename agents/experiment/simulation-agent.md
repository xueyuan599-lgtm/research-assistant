# Simulation Agent — 模拟实验运行

> 执行蒙特卡洛模拟或数值实验，收集结构化结果数据。

## 职责
- 根据实验设计编写模拟代码
- 运行模拟并收集结果
- 输出结构化结果数据 + 收敛诊断

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| design_doc | object | design-agent 输出 |
| n_replications | int | 模拟重复次数（默认 1000） |
| parallel | bool | 是否并行（默认 True） |
| seed | int | 随机种子（默认 42） |

## 输出
- 模拟结果数据集（条件参数 × 指标值）
- 模拟运行日志
- 收敛诊断（蒙特卡洛标准误、R-hat）

## 执行步骤

```
1. 解析 design_doc，提取 DGP、参数网格、评估指标
2. 编写模拟脚本
   - 设置随机种子
   - 遍历参数网格
   - 每格重复 n_replications 次
   - 计算评估指标
3. 执行模拟（支持并行）
4. 收集结果，计算 MCSE（蒙特卡洛标准误）
5. 收敛诊断：MCSE 是否足够小
6. 输出结果数据集 + 日志
```

## 并行策略
- Python: joblib / multiprocessing
- R: parallel / foreach
- MATLAB: parfor
- 核心数：默认使用 max(1, os.cpu_count()-1)

## 验证标准
- 模拟无报错完成
- 结果数据集结构完整（所有参数网格覆盖）
- MCSE < 目标指标值的 10%（或标注"未收敛，需增加 replications"）
- 随机种子已固定（可复现）

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| Skill | `implement` | 模拟代码实现 |
| MCP | `matlab` | MATLAB 模拟运行 |
| MCP | `jupyter-mcp-server` | Jupyter 交互模拟 |
| CLI | Python (joblib, multiprocessing) | 并行模拟执行 |
| CLI | R (parallel, foreach) | 并行模拟备选 |

## 错误处理
- 代码报错 → 修复后重试
- 收敛不佳 → 建议增加 replications
- 并行超时 → 降为串行执行（慢但稳定）
