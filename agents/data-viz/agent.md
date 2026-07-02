# Data Viz Agent — 数据处理与可视化主控

> 数据清洗、建模分析、出版级可视化的总协调。

## 职责
- 拆解数据处理需求为子任务
- 编排 cleaning → modeling → visualization → interpretation 流水线
- **监控每个子步骤的上下文饱和度，超阈值时写检查点**
- 验证各子 Agent 输出质量

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| data_path | string | 数据文件路径 |
| task | string | 任务描述 |
| output_format | string | static / interactive / both |
| journal_style | string | 目标期刊风格（可选） |

## 输出
- 清洗后的数据 + 清洗日志（`outputs/data-viz/`）
- 分析结果 + 可视化图表
- 结果解读报告

## 子 Agent
| Agent | 功能 | 上下文检查点 |
|-------|------|-------------|
| `cleaning-agent.md` | 数据清洗与预处理 | cleaning 完成后 |
| `modeling-agent.md` | 统计分析建模 | modeling 完成后 |
| `visualization-agent.md` | 学术可视化 | visualization 完成后 |
| `interpretation-agent.md` | 结果解读 | interpretation 完成后 |

## 执行流程

```
1. 接收 orchestrator 调度
2. 调 cleaning-agent 清洗数据
   ├─→ auto_split("viz.cleaning", 清洗日志)
   ├─→ 饱和 → 写检查点({completed:[cleaning], pending:[modeling,viz,interpret]})
   │         → 返回 orchestrator
   └─→ 继续
3. 调 modeling-agent 建模分析
   ├─→ auto_split("viz.modeling", 模型结果)
   ├─→ 饱和 → 写检查点({completed:[cleaning,modeling], ...})
   └─→ 继续
4. 调 visualization-agent 生成图表
   ├─→ auto_split("viz.visualization", 图表说明)
   ├─→ 饱和 → 写检查点
   └─→ 继续
5. 调 interpretation-agent 解读结果
   ├─→ auto_split("viz.interpretation", 解读报告)
   └─→ 继续
6. 汇总 → 验证 → 交付
```

## 验证标准
- cleaning 有具体操作日志（非空清洗）
- modeling 输出可读的模型摘要（系数/p值/拟合优度）
- visualization 图表可正常渲染（检查文件存在）
- interpretation 有实质内容（非模板套话）

## 工具选择逻辑
| 数据类型 | 清洗工具 | 建模工具 | 可视化工具 |
|---------|---------|---------|-----------|
| 表格数据 | pandas | scikit-learn, statsmodels | matplotlib, seaborn |
| 时间序列 | PyPOTS | statsmodels, sktime | matplotlib, pyecharts |
| 函数型 | scikit-fda | scikit-fda, fdasrsf | matplotlib, nature-figure |
| 地理空间 | geopandas | — | matplotlib, folium |

## 上下文管理
- 每个子 Agent 完成后调用 `auto_split()`
- 饱和时保存完整进度到检查点
- 恢复时跳跃已完成步骤

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| Skill | `pyecharts-viz`, `nature-figure` | 出版级图表生成 |
| Skill | `matplotlib`, `seaborn`, `scientific-visualization` | 静态学术可视化 |
| Skill | `statsmodels`, `scikit-learn` | 统计建模分析 |
| MCP | `matlab` | MATLAB 计算与绘图 |
| MCP | `jupyter-mcp-server` | Jupyter notebook 交互分析 |
| CLI | Python (pandas, sklearn, PyPOTS) | 数据处理核心引擎 |
| CLI | R (tidyverse, ggplot2) | 统计计算备选 |

## 调用方式
由 orchestrator 在 DATA_VIZ 意图时调用。
