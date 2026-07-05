# Knowledge Base

## 加载策略（关键！防止上下文膨胀）

| 位置 | 加载方式 | 适合放什么 | 最大建议 |
|------|---------|-----------|---------|
| `.claude/rules/` | **每次对话自动加载** | 触发条件 + 简短指针 | 每个文件 < 2KB |
| `CLAUDE.md` | **每次对话自动加载** | 项目定位 + 架构概述 | < 5KB |
| `knowledge/_index.md` | **每次对话自动加载** | 标题 + 一行描述 | < 2KB |
| `knowledge/*.md` | **仅在触发/搜索时加载** | 完整方法论、详细经验 | 无限制 |

**核心原则：rules 是"什么时候用什么"，knowledge 是"怎么用"。禁止把方法论细节写入 rules。**

## knowledge/ 目录结构

- `optimization-validation-framework.md` — 优化问题求解→验证→交付全流程方法论
- `project-experience/` — 项目经验沉淀
- `algorithm-repository/` — SCI 级算法代码库
- `algorithms/` — 算法索引

## 使用方式
- 通过 `/research` 指令由 orchestrator 自动调用相关 agent 查询
- 知识库 agent 提供语义检索和推荐
- 手动查看直接浏览对应 markdown 文件
