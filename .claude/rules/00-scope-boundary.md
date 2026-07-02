# 作用域边界（沙箱协议）

## 核心规则
所有操作**只允许在 `research-assistant/` 目录内**，禁止触碰外层文件。

## 允许的操作（白名单）
| 操作 | 范围 |
|------|------|
| 读写文件 | `research-assistant/**`（含子目录） |
| 创建目录 | `research-assistant/**` |
| 安装依赖 | 仅 Python/R 包，不影响项目文件 |
| 读取工具 | 允许读取外层 skills 参考文档 |
| 运行代码 | 仅运行 `research-assistant/` 下的脚本 |

## 禁止的操作
- 修改 `research-assistant/` 以外的任何文件
- 在外层目录创建文件或目录
- 修改父项目的 `CLAUDE.md`、`.claude/rules/`、`.claude/agents/`
- 运行外层目录下的脚本（除非只读调用）

## 例外（需用户明确批准）
- 需要用到外层 `quality_reports/`、`test/` 等目录时
- 需要跨项目引用数据文件时

## 输出约定
- 所有实验结果、图表 → `research-assistant/outputs/`
- 所有新增 Agent → `research-assistant/agents/`
- 工作流修改 → `research-assistant/workflows/`
