# Agent Shared Memory Protocol

> 轻量工作记忆系统。每个 Agent 完成任务后写入关键发现，下游 Agent 启动前读取上游记忆。
> 理论基础：Li et al. (2024) "A survey on LLM-based multi-agent systems" 第 3.3 节 Self-Action 模块。

---

## 1. 三种记忆定义

| 记忆类型 | 生命周期 | 存储位置 | 用途 |
|---------|---------|---------|------|
| **短期记忆** (Short-term) | 当前 Agent 执行期间，完成后清空 | Agent 上下文内 | 当前任务的中间推理步骤 |
| **长期记忆** (Long-term) | 跨任务持久化 | `knowledge/` 目录 | 可复用的方法论、经验、算法 |
| **共享记忆** (Shared Memory) | 当前 task 生命周期 | `outputs/{task_id}/shared_memory/` | Agent 间传递的关键发现、数据路径、警告 |

本文件定义**共享记忆**的写入/读取协议。

---

## 2. 存储位置

```
outputs/{task_id}/shared_memory/
├── memory_index.json          # 记忆映射表（Orchestrator 据此确定读取链）
├── {agent_name}.json          # 每个 Agent 的共享记忆
└── memory_chain.md            # 人类可读的记忆链摘要（调试用）
```

---

## 3. 共享记忆 JSON 格式

```json
{
  "task_id": "cumcm2024c",
  "agent_name": "data-cleaner",
  "timestamp": "2026-07-05T10:30:00",
  "status": "completed",
  "key_findings": [
    "发现 1：具体描述，含数值/文件名/行号",
    "发现 2：..."
  ],
  "data_artifacts": [
    {
      "path": "outputs/{task_id}/clean/cleaned_data.csv",
      "description": "清洗后主数据表",
      "schema": "outputs/{task_id}/clean/schema.json"
    }
  ],
  "warnings": [
    "警告 1：需要注意的陷阱或限制",
    "警告 2：..."
  ],
  "handoff_notes": "给下游 Agent 的自由文本备注。"
}
```

### 字段约束

| 字段 | 必填 | 说明 |
|------|------|------|
| `task_id` | 是 | 与秘书分解方案中的 task_id 一致 |
| `agent_name` | 是 | 与秘书分解方案中的 Agent 标识一致 |
| `timestamp` | 是 | ISO 8601 格式 |
| `status` | 是 | `completed` / `failed` / `partial` |
| `key_findings` | 是 | 至少 1 条；每条需含具体信息（数值/文件名/方法名），禁止空泛 |
| `data_artifacts` | 否 | 有产出数据时必填 |
| `warnings` | 否 | 有陷阱/限制时必填 |
| `handoff_notes` | 是 | 至少 1-2 句 |

---

## 4. memory_index.json 格式

```json
{
  "task_id": "cumcm2024c",
  "created": "2026-07-05T10:00:00",
  "updated": "2026-07-05T11:30:00",
  "agents": {
    "data-inspector": {
      "file": "data-inspector.json",
      "upstream": [],
      "downstream": ["data-cleaner"],
      "status": "completed"
    },
    "data-cleaner": {
      "file": "data-cleaner.json",
      "upstream": ["data-inspector"],
      "downstream": ["modeler"],
      "status": "completed"
    }
  }
}
```

---

## 5. memory_chain.md 格式（人类可读）

```markdown
# 记忆传递链 — cumcm2024c

| 时间 | Agent | 关键发现数 | 警告数 | 数据产物 | 状态 |
|------|-------|-----------|--------|---------|------|
| 10:00 | data-inspector | 3 | 1 | raw/data.csv | completed |
| 10:15 | data-cleaner | 3 | 2 | clean/cleaned_data.csv | completed |
```

---

## 6. 与长期记忆的边界

| 场景 | 共享记忆 | 长期记忆 |
|------|---------|---------|
| "Excel 中 season 列标错了" | 是（当前 task 特有） | 否 |
| "水稻产量数据通常缺失率 5-10%" | 否（可复用知识） | 是 |
| "本次建模用 OLS，R²=0.73" | 是（当前 task 结果） | 否 |
| "DID 平行趋势检验标准流程" | 否（可复用方法） | 是 |

**判断规则：只在当前 task 内有用 → 共享记忆；可跨 task 复用 → 长期记忆。**
