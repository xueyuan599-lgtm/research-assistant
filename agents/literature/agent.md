# Literature Agent — 文献主控

> 文献检索、筛选、综述合成的总协调。

## 职责
- 拆解用户文献需求为子任务
- 编排 search → screening → synthesis 流水线
- **监控每个子步骤的上下文饱和度，超阈值时写检查点**
- 验证各子 Agent 输出质量

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| research_question | string | 研究问题或关键词 |
| depth | string | **分发前由用户选定**（quick / standard / thorough），档位数字见 `.claude/rules/05-exploration-budget.md` §1 |
| output_format | string | summary / review / annotated_bibliography |

## 输出
- 文献综述报告（`outputs/literature/`）
- 文献列表（含 DOI/标题/摘要/质量评分）

## 子 Agent
| Agent | 功能 | 上下文检查点 |
|-------|------|-------------|
| `search-agent.md` | 多数据库文献检索 | search 完成后 |
| `screening-agent.md` | 文献筛选与质量评估 | screening 完成后 |
| `synthesis-agent.md` | 综述合成与总结 | synthesis 完成后 |

## 执行流程

```
0. **【前置】向用户报出三档及其预算信封，由用户选定档位**（推荐 quick）
   └─ 依据 `.claude/rules/05-exploration-budget.md` §0；选定后锁定，本次任务不自行升档
1. 接收 orchestrator 调度
2. 调 search-agent 执行检索
   ├─→ auto_split("literature.search", 检索结果文本)
   ├─→ 饱和 → 写检查点(state: completed=[search], pending=[screening, synthesis])
   │         → 返回饱和信号给 orchestrator → 结束
   └─→ 继续
3. 调 screening-agent 执行筛选
   ├─→ auto_split("literature.screening", 筛选结果文本)
   ├─→ 饱和 → 写检查点(state: completed=[search, screening], pending=[synthesis])
   │         → 返回饱和信号给 orchestrator → 结束
   └─→ 继续
4. 调 synthesis-agent 执行综述合成
   ├─→ auto_split("literature.synthesis", 综述文本)
   └─→ 继续
5. 汇总结果 → 验证 → 交付
```

## 验证标准
- search 结果非空，且文献数与探头数 **≤ 当前档位上限**（见 `.claude/rules/05-exploration-budget.md` §1）
- screening 剔除/保留理由明确
- synthesis 有主题结构（非简单罗列）
- 综述包含共识与争议点

## 上下文管理
- 每个子 Agent 完成后调用 `context_monitor.auto_split()`
- 饱和时调用 `checkpoint.write_checkpoint()` 保存 {completed, pending}
- 恢复时读取检查点，跳过 completed，直接处理 pending

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| Skill | `research`, `deep-research` | 文献检索与深度调研 |
| CLI | WebSearch, WebFetch | **检索主力**：arXiv / DBLP / Semantic Scholar / OpenAlex 公开页面 |
| ⚠️ | ~~`lit-mcp` / `mcp-research` / `openalex-mcp-server`~~ | **未安装的 MCP，勿调用**（2026-09-19 核实：任何 `.mcp.json` 均未声明）。恢复属独立任务 |

## 调用方式
由 orchestrator 在 LITERATURE 意图时调用。
