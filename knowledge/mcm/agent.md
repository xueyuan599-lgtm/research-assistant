# MCM 知识库 — 查询入口说明（stub）

> **本文件不是 Agent 定义**。orchestrator 只路由 `agents/` 下的文件，本文件在 `knowledge/` 下，
> 结构上不可能被路由。Agent 定义见 `agents/knowledge/agent.md`。
>
> 留这个 stub 纯粹是**路标**：`knowledge/mcm/agent.md` 这个路径看起来像一个 Agent 定义，
> 实际定义在 `agents/` 下。它没有任何入站引用（全仓 0 命中），删掉也不会断链。

- **查询**：由 `agents/knowledge/agent.md` 执行，按 `_index.md` §查询接口 的 `problem_profile` 参数匹配 `patterns/`
- **写入**：本库自身**只读**；赛后复盘由 MCM 主控协调 `agents/knowledge/agent.md` 落库（规则见 `_index.md` §生长机制）
