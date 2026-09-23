# /paper

期刊论文结构写作：固定尺子 → 逐步骤推进 → 结构审稿，**审稿人导向**。

## 用法
```
/paper 目标期刊是 <刊名>，材料在 <材料路径>
/paper 继续 <工作区路径>
/paper 只做 <步骤号>，工作区 <工作区路径>
```

### 示例
```
/paper 目标期刊是《管理世界》，草稿和材料在 my-paper/
/paper 继续 outputs/journal_glsj_20260921/
/paper 只做 05，工作区 outputs/journal_glsj_20260921/
```

## 执行流程
1. 加载 `.claude/rules/09-journal-track.md`
2. 识别为 JOURNAL 领域后调用 `agents/journal/agent.md`（与 PAPER_FORMAT 的消歧见 `agents/secretary.md` 领域识别表）
3. **固定尺子**：提炼 `00-journal-rules.md`。**未固定不得执行 `01`**
4. **步骤循环**：`01` 标题 → `02` 摘要 → `03` 引言 → `04` 方法 → `05` 证据映射 → `06` 讨论
   —— **一步一文件、每步产出后停止等指令**；用户显式说"连跑 N 步"时除外
5. **收尾**：委派 `agents/journal/structure-reviewer-agent.md` 出 `07-review.md`，
   再委派 `agents/paper-format/agent.md` 做格式与合规
6. 交付：工作区 `outputs/journal_{期刊简称}_{时间戳}/`

## 约束
- 各步骤的**检查项与文件名**唯一源见 `.claude/rules/09-journal-track.md` 的归属表——本命令不复述判据
- 主控**不自行起草全文**；`my-paper/draft.md` 由用户提供或委派用户级 `academic-paper`（逻辑名）
- **换期刊只替换 `source/` 与 `00-journal-rules.md`**，文件链不变
- 所有输出限于本仓库内（`.claude/rules/00-scope-boundary.md`）
