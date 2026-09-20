# /research-prompt-refiner

优化/完善科研类提示词（数学建模 / 机器学习 / 文献综述 / 通用科研），强制前置检索增强。

## 用法
```
/research-prompt-refiner <科研提示词草稿 或 仅科研主题>
```

### 示例
```
/research-prompt-refiner 我的建模提示词：请用TOPSIS-熵权法给10个方案排序，输出权重和排名
```
```
/research-prompt-refiner 帮我写一份完整的文献综述提示词，主题是DID在劳动经济学中的应用
```

## 执行流程
1. 加载 skill：`.claude/skills/research-prompt-refiner/SKILL.md`
2. 严格按 `references/search-protocol.md` 执行强制资源检索（arXiv / ACL Anthology / Nature/JMLR/NeurIPS + GitHub），未检索不输出
3. 按领域读取专项文档：数学建模 → `references/math-modeling.md`；机器学习 → `references/machine-learning.md`；已知坑 → `references/known-pitfalls.md`；基线库 → `references/prompt-resources.md`
4. 结构化优化（角色/背景/目标/步骤/输入输出/约束/验收），验收须含"代码一次可运行、指标齐全、固定种子重跑一致"，建模类另含灵敏度/稳健性
5. 交付：优化稿 + 变更说明 + 资源检索记录；先出简洁档，经用户确认后进入执行/保存

## 约束
- 强制检索为前置，不可跳过；检索工具不可用时退回基线库并明说
- 输出语言默认中文
- 通用娱乐性提示词优化不触发本 command
