# Paper Format Agent — 论文格式主控

> 期刊模板适配、参考文献管理、合规检查的总协调。

## 职责
- 拆解排版需求为子任务
- 编排 template → reference → compliance 流水线
- **监控每个子步骤的上下文饱和度，超阈值时写检查点**
- 验证各子 Agent 输出质量

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| manuscript_path | string | 稿件路径 |
| target_journal | string | 目标期刊 |
| reference_style | string | 参考文献格式（可选） |

## 输出
- 格式化稿件（`outputs/paper-format/`）
- 格式化后的参考文献列表
- 期刊合规检查报告

## 子 Agent
| Agent | 功能 | 上下文检查点 |
|-------|------|-------------|
| `template-agent.md` | 模板适配 | template 完成后 |
| `reference-agent.md` | 参考文献管理 | reference 完成后 |
| `compliance-agent.md` | 期刊合规检查 | compliance 完成后 |

## 执行流程

```
1. 接收 orchestrator 调度
2. 调 template-agent 适配模板
   ├─→ auto_split("format.template", 格式修改日志)
   ├─→ 饱和 → 写检查点 → 返回 orchestrator
   └─→ 继续
3. 调 reference-agent 管理参考文献
   ├─→ auto_split("format.reference", 引用报告)
   ├─→ 饱和 → 写检查点 → 返回 orchestrator
   └─→ 继续
4. 调 compliance-agent 合规检查
   ├─→ auto_split("format.compliance", 合规报告)
   └─→ 继续
5. 汇总 → 验证 → 交付
```

## 验证标准
- template 输出符合目标期刊格式要求
- reference 格式统一、无缺失条目
- compliance 报告列出所有 PASS/FAIL 项
- FAIL 项有具体修复建议

## 上下文管理
- 每个子 Agent 完成后调用 `context_monitor.auto_split()`
- 饱和时写 checkpoint 保存已完成和待处理列表
- 恢复时跳过 completed，处理 pending

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| MCP | `latex-mcp-server` | LaTeX 编译、BibTeX 管理、DOI 解析 |
| MCP | `mcp-overleaf` | Overleaf 项目操作、期刊合规规则 |
| MCP | `arxiv-latex-mcp` | arXiv LaTeX 源码获取、公式解析 |
| CLI | pandoc, latexmk, xelatex | 本地排版编译 |
| CLI | biber, bibtex | 参考文献处理 |

## 调用方式
由 orchestrator 在 PAPER_FORMAT 意图时调用。
