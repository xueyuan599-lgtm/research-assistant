# Codex ↔ Claude Code 协作协议

> **唯一权威源。** 本文件定义 Codex（Controller）与 Claude Code（Worker）之间的固定协作协议。被 `agents/orchestrator.md` 与 `workflows/dynamic-workflow.md` 引用。
> 适用范围：**仅**通过 `scripts/claude_worker.py` 启动的 Claude Worker 任务。用户在终端直接运行的 Claude Code 会话不纳入本协议，但秘书守门（`.claude/rules/06-cost-discipline.md` §零）与 orchestrator 编排依然强制适用。
>
> **本文件历史上的断链问题**：orchestrator 与 dynamic-workflow 曾长期引用本文件作为"唯一权威源"而文件缺失，导致调度无法落地。本文件现为既有协议的唯一落点；后续如需修改协作协议，只改本文件并同步 orchestrator 的摘要小节。
>
> **⚠️ 未落地的前置（2026-09-19 核实）**：本协议的执行器 `scripts/claude_worker.py` 在本仓库
> **从未提交过**（`git log --all` 无任何记录，`scripts/` 下亦无此文件）。**在脚本补齐前，本协议不可执行**——
> 下列 `init` / `run` / `pause` 均为设计意图而非可用命令。文件缺失不等于协议作废：这是一份待实现的设计记录，
> 保留在此以便日后补齐执行器时直接对接。

## 角色

| 角色 | 承担者 | 职责 |
|------|--------|------|
| Controller（唯一） | Codex | 任务调度、进度控制、产物验收 |
| Worker | Claude Code（DeepSeek） | 执行 Codex 委派的边界明确的子任务 |

Codex 主导的非 trivial 任务，Claude Code 参与是固定流程，不再询问"是否启用 Claude"。秘书只确认其具体子任务、预算、超时、工具与交付格式。

## 五项专家职责（保留给 Codex）

1. 数学推导
2. 可视化
3. 逻辑与代码审查
4. 创新发现
5. 语言润色

## 协作门（固定顺序）

```text
Secretary + 用户确认（T2/T3 任务，见 .claude/rules/06-cost-discipline.md §零 分级）
  → collaboration init
  → DeepSeek / MCP / LaTeX preflight
  → Codex 与 Claude 独立工作区执行
  → schema + SHA-256 handoff
  → Codex 五项专家职责
  → 独立 Critic
  → deliverables 提升
```

## 调度规则

1. 用户确认秘书方案后，`python scripts/claude_worker.py init ...` 初始化任务 manifest。
2. 将边界明确的 Claude 请求写入任务 `.coord/`，再通过执行器 `run`；禁止直接拼接或裸调用 `claude`。
3. 仅接受 `_claude/staging/handoff.json` 中通过 schema、路径和哈希校验的产物。
4. Codex 完成数学推导、可视化、逻辑/代码审查、创新发现、语言润色；独立 Critic 通过后才提升到 `deliverables/`。
5. Claude 子任务必须写明：输入、只允许的输出目录、验收标准、所需 MCP/CLI、预算、停止点。

## 停止与恢复

- 用户说"停止项目"时，秘书不得重新分解或自动恢复；执行器调用 `pause`，保存检查点且不删除文件。
- 恢复必须产生新的 `run_id`。
- 用户说"停止"时，Orchestrator 按本协议将当前任务转为 `PAUSED`。