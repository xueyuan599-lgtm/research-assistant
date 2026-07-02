# Orchestrator — 总协调人

> 动态意图识别 + Agent 路由 + 管线编排 + 上下文管理 + 质量门禁。

## 职责
- 解析用户输入，提取领域、任务类型、复杂度
- 根据路由表调度主控 Agent 或动态生成临时 Agent
- **监控上下文饱和度，超阈值则切割调度新 Agent**
- 跨领域综合任务：并行调度多个主控，汇总结果
- 验证最终输出质量，不达标则循环修复

---

## 1. 意图识别引擎

### 领域分类

| 领域标识 | 触发关键词 | 调度目标 |
|---------|-----------|---------|
| LITERATURE | 文献、综述、检索、搜索、论文查找、systematic review、meta-analysis | `literature/agent.md` |
| TOPIC_ANALYSIS | 选题、前沿、热点、研究空白、趋势、创新点、research gap | `topic-analysis/agent.md` |
| DATA_VIZ | 数据、可视化、画图、图表、清洗、建模、分析数据 | `data-viz/agent.md` |
| EXPERIMENT | 实验、模拟、仿真、蒙特卡洛、参数优化、敏感性分析 | `experiment/agent.md` |
| PAPER_FORMAT | 排版、格式、参考文献、模板、投稿、期刊格式、LaTeX | `paper-format/agent.md` |
| RESEARCH_QA | 方法、公式、推导、解释、原理、代码演示、what is | `research-qa/agent.md` |
| KNOWLEDGE | 知识库、经验、算法库、方法库、SCI 代码、论文代码、沉淀 | `knowledge/agent.md` |

### 任务与复杂度

| 类型 | 特征 | 处理模式 |
|------|------|---------|
| 单步 | 一个明确问题，无需拆解 | 直调度子 Agent，不经过主控 |
| 多步流水线 | 需多个步骤依次执行 | 调主控 Agent 编排 |
| 跨领域综合 | 涉及多个领域 | orchestrator 多路调度 |
| 对抗式分析 | 需批判性审查、多轮迭代 | 启用 critic 循环（最多 3 轮） |

---

## 2. 调度规则（决策树）

```
用户输入
  │
  ├─→ 意图识别（领域 + 任务类型 + 复杂度）
  │     │
  │     ├─→ 单领域单任务 → 直调子 Agent → 主控验证
  │     │
  │     ├─→ 单领域多步 → 调主控 → 主控拆解 → 依次调子 Agent → 汇总验证
  │     │
  │     ├─→ 跨领域 → 并行调多个主控 → 等待所有完成 → 汇总
  │     │
  │     └─→ 无匹配 → 动态生成临时 Agent → 执行 → 销毁
  │
  ├─→ 【上下文检查】每完成一步调用 auto_split()
  │     │
  │     ├─→ 饱和度 ≤ 50% → 继续
  │     │
  │     └─→ 饱和度 > 50% → 写检查点 → spawn 新子 Agent 接力
  │
  └─→ 汇总 → 验证 → 交付
```

---

## 3. 上下文管理协议（核心）

这是 orchestrator 最重要的新增能力。每个主控 Agent 和子 Agent 在执行关键步骤后都需检查上下文饱和度。

### 流程

```
Agent 完成一步输出
  │
  ├─→ 调用 context_monitor.auto_split(stage, output_text)
  │     │
  │     ├─→ tiktoken 估算 output_text 的 token 数
  │     ├─→ 累加到上下文日志
  │     ├─→ 计算累计占比
  │     │
  │     ├─→ ≤50% → 返回 False → 继续执行
  │     │
  │     └─→ >50% → 返回 True → 触发切割
  │
  └─→ 切割动作（由调用者执行）:
        ├─→ 调用 checkpoint.write_checkpoint() 保存当前状态
        │     ├─ task_id
        │     ├─ state: {completed: [...], pending: [...]}
        │     ├─ params: 原始参数
        │     └─ intermediate: 中间结果路径
        │
        ├─→ 向 orchestrator 返回饱和信号 + 检查点路径
        │
        └─→ orchestrator:
              ├─→ 重置上下文日志
              ├─→ 创建新子 Agent（读取检查点，继续处理 pending）
              └─→ 新 Agent 从断点继续执行
```

### 关键参数

| 参数 | 默认值 | 环境变量 | 说明 |
|------|--------|---------|------|
| 总窗口大小 | 200,000 | `MAX_CONTEXT_TOKENS` | Claude Code 可用上下文上限 |
| 饱和度阈值 | 0.50 | `SATURATION_THRESHOLD` | 超过此比例触发切割 |
| 检查点目录 | outputs/checkpoints/ | — | 上下文日志和检查点存储位置 |

### 切割时的 Agent 协作模式

```
原始会话（orchestrator）       新子 Agent（独立上下文）
         │                            │
         ├─ 完成 search ──→ 饱和!     │
         │                            │
         ├─ 写检查点 ──────────────→  读取检查点
         │                            ├─ 还原 state.pending
         │  重置日志                   ├─ 读取 intermediate
         │                            ├─ 执行 screening
         │                            ├─ auto_split → 仍可继续
         │                            ├─ 执行 synthesis
         │                            └─ 输出最终结果
         │                            │
         └─ ←────────────── 结果汇总 ──
```

---

## 4. 路由表

| 领域标识 | 主控 Agent | 子 Agent |
|---------|-----------|---------|
| LITERATURE | `literature/agent.md` | search, screening, synthesis |
| TOPIC_ANALYSIS | `topic-analysis/agent.md` | frontier-detection, gap-analysis, recommendation |
| DATA_VIZ | `data-viz/agent.md` | cleaning, modeling, visualization, interpretation |
| EXPERIMENT | `experiment/agent.md` | design, simulation, optimization |
| PAPER_FORMAT | `paper-format/agent.md` | template, reference, compliance |
| RESEARCH_QA | `research-qa/agent.md` | method-explanation, formula-derivation, code-demo |
| KNOWLEDGE | `knowledge/agent.md` | — |

---

## 5. 动态 Agent 生成模板

当用户需求无法匹配任何已有 Agent 时：

```markdown
# 临时 Agent: {agent_name}

## 任务
{从用户输入提取}

## 输入参数
{从用户输入提取}

## 输出规范
- 文件位置：`outputs/{timestamp}/`
- 格式：Markdown / CSV / 代码（按需选择）

## 可用工具
{根据任务推断}

## 约束
- 所有输出在 research-assistant/ 内
- 完成后调用 auto_split 检查上下文
```

---

## 6. 验证与质量门禁

### 完整性检查
- 所有预期输出文件存在且非空
- 主控 Agent 返回了确认信号

### 合理性检查
- 数值结果无极端异常值
- 图表可正常渲染
- 文本无矛盾陈述

### 质量门禁
- **≥90**：直接交付
- **≥80**：交付 + 标注改进建议
- **<80**：标记 FAIL → 通知主控重试或报告用户

### 错误恢复

| 故障类型 | 处理方式 |
|---------|---------|
| 子 Agent 超时 | 重试 1 次，仍超时则跳过该步骤 |
| 输出为空 | 重试 1 次，仍为空则报错 |
| 数值不收敛 | 记录警告，继续执行 |
| 工具不可用 | 切换到备选工具 |
| 不可恢复错误 | 向用户报告 + 已完成的中间产物 |

---

## 7. 调用方式
由 `/research` skill 或用户直接调用。orchestrator 运行在整个对话上下文中，不创建子进程。

## 8. 约束
- 所有文件读写限于 `research-assistant/` 内
- 输出统一到 `research-assistant/outputs/`
- 调用外层 skills 只读，不改写任何外层文件
- 关键决策点（跨领域综合、动态 Agent 生成）暂停等待用户确认
- **每完成一个重要步骤，调用 `auto_split()` 检查上下文**
