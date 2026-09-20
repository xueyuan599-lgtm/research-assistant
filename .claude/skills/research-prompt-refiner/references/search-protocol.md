# 强制检索协议（MANDATORY SEARCH PROTOCOL）

> 本协议是 skill 的强制前置步骤。优化任何科研提示词之前必须执行；**未完成本协议不得输出优化稿**。

## 目标

在**高质量期刊/论文库**与 **GitHub** 上检索与本任务主题相关的 prompt 方法与开源项目，把最新、经过验证的方法注入优化稿。检索不是形式，检索结果必须可见地改变优化稿的内容（技术选择、示例构造、约束、验收指标）。

## 步骤

### 1. 领域判定（30 秒内）

读取任务，判定领域标签：`数学建模` / `机器学习` / `深度学习` / `时序预测` / `因果推断` / `文献综述` / `优化求解` / `通用科研`。记录标签（本会话不重复检索同一标签）。

### 2. 检索词构造（按领域定制）

| 领域 | 检索词模板 | 示例 |
|---|---|---|
| 数学建模 | `prompt math modeling competition LLM`、`LLM mathematical reasoning prompt [题型+模型]` | `TOPSIS 熵权法 python math modeling LLM prompt` |
| 机器学习/深度学习 | `prompt [任务/模型] data science`、`few-shot [任务] prompt evaluation` | `prompt time series forecasting LLM evaluation` |
| 因果推断/统计 | `prompt causal inference econometrics LLM` | `prompt difference-in-differences LLM` |
| 文献综述 | `LLM literature review prompt survey`、`prompt academic writing generation` | `LLM systematic review prompt evaluation` |
| 通用科研 | `prompt [具体方向] scientific research agents` | `prompt scientific discovery LLM agent` |
| 代码生成 | `prompt code generation reproducible evaluation` | `prompt code interpretation data analysis reproducible` |

### 3. 来源清单（至少覆盖「论文」与「GitHub」两类，每类 2–5 个结果即停）

| 类别 | 来源 | 检索工具（示例） |
|---|---|---|
| 论文/期刊 | arXiv API：`https://export.arxiv.org/api/query?search_query=ti:"..."` | curl / WebFetch |
| 论文/期刊 | Semantic Scholar API：`https://api.semanticscholar.org/graph/v1/paper/search?query=...` | curl / WebFetch |
| 论文/期刊 | ACL Anthology 检索 | WebFetch / WebSearch |
| 论文/期刊 | Nature 系列 / Science / JMLR / JAIR / NeurIPS / ICML 会议 | WebSearch / WebFetch |
| GitHub 项目 | GitHub API：`https://api.github.com/search/repositories?q=...&sort=stars` | curl / WebFetch |
| GitHub 项目 | GitHub 网页搜索 keywords/prompt | WebSearch / WebFetch |
| 基线库（兜底） | `references/prompt-resources.md`（条目标注"已验证 2026-08-19"） | 本地读取 |

### 4. 命中筛选与去重

- 只保留与任务主题**直接相关**的成果；每类 2–5 个即停，避免无限检索。
- 同一会话内已检索过的领域标签记录在案，不重复检索。
- `references/known-pitfalls.md` 已覆盖的坑不再花检索预算。

### 5. 注入与记录（交付必须附上，格式如下）

```markdown
## 资源检索记录
- 检索时间：YYYY-MM-DD
- 领域标签：<标签>（本会话已检，不重复）
- 检索词清单：(逐条列出)
- 论文/期刊命中：
  - <标题>（<作者/年份>，arXiv:<ID> 或 <期刊/会议+链接>）→ 注入点：<优化稿第 N 步>
- GitHub 命中：
  - <owner/repo>（★<stars>，<链接>）→ 注入点：<优化稿第 N 步>
- 检索不可用说明：（若某来源失败，明说 + 用基线库兜底）
```

## 兜底规则

所有检索工具不可用（网络断连、API 限流等）：明说"实时检索不可用（原因）"，退回 `references/prompt-resources.md` 基线库，引用时标注"基线库条目，未实时核验"，再基于通用知识优化。

## 检索预算

- 每类来源 2–5 个结果即停
- 全文检索总时长目标 < 2 分钟
- 检索质量优先：宁缺毋滥，不凑数