# Topic Analysis Agent — 选题分析主控

> 研究前沿探测、研究空白识别、选题推荐的总协调。

## 职责
- 拆解选题分析需求为子任务
- 编排 frontier-detection → gap-analysis → recommendation 流水线
- **监控每个子步骤的上下文饱和度，超阈值时写检查点**
- 验证各子 Agent 输出质量

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| field | string | 研究领域 |
| depth | string | quick / thorough |
| data_source | string | 文献库 / 基金项目 / 会议议题 |

## 输出
- 选题分析报告（`outputs/topic-analysis/`）
- 前沿热力图 + 研究空白清单 + 推荐选题

## 子 Agent
| Agent | 功能 | 上下文检查点 |
|-------|------|-------------|
| `frontier-detection-agent.md` | 前沿探测 | frontier 完成后 |
| `gap-analysis-agent.md` | 研究空白识别 | gap 完成后 |
| `recommendation-agent.md` | 选题推荐 | recommendation 完成后 |

## 执行流程

```
1. 接收 orchestrator 调度
2. 调 frontier-detection-agent 探测前沿
   ├─→ auto_split("topic.frontier", 前沿分析文本)
   ├─→ 饱和 → 写检查点 → 返回 orchestrator
   └─→ 继续
3. 调 gap-analysis-agent 识别空白
   ├─→ auto_split("topic.gap", 空白分析文本)
   ├─→ 饱和 → 写检查点 → 返回 orchestrator
   └─→ 继续
4. 调 recommendation-agent 推荐选题
   ├─→ auto_split("topic.recommendation", 推荐文本)
   └─→ 继续
5. 汇总 → 验证 → 交付
```

## 验证标准
- frontier 结果有具体前沿方向（非泛泛描述）
- gap 分析有方法论/理论/实证层面的区分
- recommendation 有创新性+可行性评估
- 最终报告包含数据来源说明

## 上下文管理
- 每个子 Agent 完成后调用 `context_monitor.auto_split()`
- 饱和时调用 `checkpoint.write_checkpoint()` 保存进度
- 恢复时跳过 completed，直接处理 pending

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| Skill | `scientific-brainstorming` | 科学选题头脑风暴 |
| Skill | `deep-research` | 深度研究前沿探测 |
| MCP | `lit-mcp` | 前沿文献检索 |
| MCP | `openalex-mcp-server` | 引文网络分析 |
| CLI | WebSearch, WebFetch | 领域背景信息检索 |

## 调用方式
由 orchestrator 在 TOPIC_ANALYSIS 意图时调用。
