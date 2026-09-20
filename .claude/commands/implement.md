# /implement

一键执行"想法→顶刊级实现"完整流水线。

## 用法
```
/implement 我想用[方法]做[研究问题]
```

## 执行流程
1. 加载 orchestrator
2. orchestrator 按 `.claude/rules/00-core-workflow.md` 调度（Agent 取 `research-assistant/agents/` 实际文件）：
   - `literature/agent.md`（顶刊规范检索）→ `algorithm/designer-agent.md`（方案+选型报告）→ `algorithm/coder-agent.md`（实现）→ `algorithm/benchmark-agent.md`（实验）→ `algorithm/validator-agent.md`（物理验证）→ 同一 validator 作为 critic 审查（最多 3 轮，详见门禁出口）
3. 交付：代码 + 结果 + 方法说明 + 复现包
