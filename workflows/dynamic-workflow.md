# Dynamic Workflow Protocol

不预设固定流水线，根据用户输入动态识别需求、构建管线。

**核心新增：每完成一个步骤检查上下文饱和度，超过 50% 自动切割调度新 Agent。**

---

## 流程总图

```
用户输入 → orchestrator 意图识别 → 匹配已有 Agent → 动态组装管线 → 执行
                                                              │
                                                   每步检查 auto_split()
                                                      │
                                          ┌───────────┴───────────┐
                                          ≤50% 继续         >50% 写检查点
                                            │                  spawn 新 Agent
                                            ↓                       │
                                        下一步 → ...          从断点继续
                                                                  ↓
                                                              执行完成后汇总
                                                                  ↓
                                                              验证 → 交付
```

---

## 步骤

### 1. 意图识别
orchestrator 分析用户输入，判断所属领域和任务类型。

**领域**：literature / topic-analysis / data-viz / experiment / paper-format / research-qa / 综合

**任务类型**：检索 / 分析 / 生成 / 优化 / 问答 / 可视化

**复杂度**：单任务 / 多步 / 复杂对抗

### 2. 管线构建

| 场景 | 处理 |
|------|------|
| 单领域单任务 | 直调子 Agent，主控验证 |
| 单领域多步 | 主控拆解 → 依次调子 Agent → 每步 auto_split |
| 跨领域综合 | 并行调多个主控 → 各自带上下文管理 → 汇总 |
| 无匹配 | orchestrator 动态生成临时 Agent → 执行完销毁 |

### 3. 上下文管理（新增）

每个 Agent 执行后必须调用上下文监控工具：

```python
from scripts.context_monitor import auto_split
from scripts.checkpoint import write_checkpoint

if auto_split(stage, output_text):
    write_checkpoint(
        task_id="...",
        state={"completed": [...], "pending": [...]},
        params={...},
        intermediate={...}
    )
    # 返回饱和信号 → orchestrator 调度新 Agent
```

详细协议见 `agents/orchestrator.md` 第 3 节。

### 4. 执行与验证
- 子 Agent 执行任务，输出结果到 `outputs/`
- 主控验证子 Agent 输出（完整性 + 合理性）
- 验证 FAIL → 重试或上报 orchestrator
- 复杂任务启用 critic 对抗（最多 3 轮）

### 5. 交付
- 结果文件 → `outputs/` 目录
- 向用户呈现：做了什么 + 结果摘要 + 文件位置
- 附复现说明（工具、参数、数据来源）

---

## 管线模板

| 用户需求 | 管线 | 可能的切割点 |
|---------|------|-------------|
| "综述XX方法" | search → screening → synthesis | screening 后（检索结果量大时） |
| "分析XX研究热点" | search → frontier-detection → gap-analysis → recommendation | frontier 后（文献多时） |
| "可视化这份数据" | cleaning → modeling → visualization → interpretation | modeling 后（大模型输出时） |
| "设计XX实验" | design → simulation → optimization | simulation 后（参数网格大时） |
| "投XX期刊，排版" | template → reference → compliance | —（通常较小，很少需要切割） |
| "解释XX方法" | method-explanation → formula-derivation → code-demo | code-demo 前（代码+输出较大） |

---

## 工具链

| 工具 | 路径 | 用途 |
|------|------|------|
| 上下文监控 | `scripts/context_monitor.py` | auto_split() 估算 Token 饱和 |
| 检查点协议 | `scripts/checkpoint.py` | write_checkpoint / latest_checkpoint 状态序列化 |
| 主控 Agent | `agents/<domain>/agent.md` | 各领域编排 |
| 子 Agent | `agents/<domain>/*-agent.md` | 具体任务执行 |
