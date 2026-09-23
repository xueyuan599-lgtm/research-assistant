# Academic Writing Standards — 学术写作质量标准（常驻核心）

> 解决 AI 写作痕迹的核心：不在"改词"，而在改变**句子结构、信息密度、论证节奏**。
> 本文件承载**可执行的操作部分**（原则 + 禁用词表 + 自检协议）；**正面范式与领域指导**已下沉到
> `knowledge/writing/academic-writing-patterns.md`（写综述/方法论/结果报告/论文段落前读一遍）。

---

## 0. 核心原则

| 原则 | 说明 |
|------|------|
| **信息密度优先** | 每句话至少传递一个具体信息（系数、方法名、文献、机制），删掉所有可删的词 |
| **用词精准** | 能用"TWFE"不用"传统双向固定效应模型"，能用"ATT"不用"平均处理效应" |
| **句式自然变化** | 长短句交替，不要每句都是"主-谓-宾"结构 |
| **少用过渡词** | 段落之间靠内容逻辑连接，不靠"Furthermore/Moreover/In addition" |

---

## 1. 禁用与慎用词汇表（Kill List）

以下词汇是 AI 痕迹的最强信号，**严禁出现在最终输出中**（综述/报告/论文段落）。

### 1.1 一级禁用（出现即 AI 痕迹）

| 词汇 | 问题 | 替代方案 |
|------|------|---------|
| `Moreover` | 教授公认 99% AI 指标 | 直接开始新句子，或删除 |
| `Furthermore` | 同上，AI 过渡词榜首 | 删除，或换 `We next consider` |
| `Additionally` | 同上 | 删除，用 `Also`（句首）或 `;` 连接 |
| `In addition` | 低信息量填充 | 删除，或换具体连接 |
| `Notably` | 空洞强调 | 直接说事实，让事实自己 notable |
| `Importantly` | 同上 | 删除 |
| `It is worth noting that` | 6 词废话 | 直接说事实 |
| `It is important to` | AI 套话 | 删掉 |
| `plays a crucial role in` | 空洞 | 直接说机制或效应量 |
| `serves as a` | 冗述 | 直接说是什么 |
| `a wide range of` | 模糊概括 | 说具体数量或范围 |
| `a variety of` | 同上 | 同上 |
| `pivotal` | 过度夸张 | 根本别用 |
| `delves` | Science Advances 确认最高 AI 信号词 | `examines` / `investigates` / `studies` |
| `underscores` | 高 AI 信号词 | `shows` / `indicates` / `suggests` |
| `showcasing` | 高 AI 信号词 | `demonstrating` / 直接描述 |
| `intricate` | 高 AI 信号词 | `complex`（如果必须用） |
| `multifaceted` | AI filler | 说具体有几个方面 |
| `landscape` (抽象用法) | "research landscape" 等 | `literature` / `field` |
| `tapestry` | 完全禁止 | — |
| `interplay` | AI 过度使用 | `interaction` / `relationship` |

### 1.2 二级慎用（每篇最多出现 1 次）

| 词汇 | 说明 |
|------|------|
| `leverage` | 用 `use` / `employ` / `utilize` |
| `enhance` | 太泛，说具体怎么提高 |
| `foster` | 同上 |
| `garner` | 非经济学术语 |
| `robust` | 滥用，每篇不超过 2 次 |
| `comprehensive` | 滥用 |
| `significant` | 仅在统计意义上用，不要做普通形容词 |

### 1.3 句式级禁用

| 句式 | 问题 | 替代 |
|------|------|------|
| `It can be observed that...` | 5 词废话 | 直接陈述 |
| `It is noteworthy that...` | 同上 | 直接陈述 |
| `This is because...` | 稚拙 | `because...` |
| `The reason is that...` | 同上 | `because...` |
| `due to the fact that` | 冗述 | `because` / `due to` |
| `In the context of...` | 套话开头 | 直接进入主题 |
| `It is widely known that` | 模糊 | 删掉或加引用 |
| `There is no doubt that` | 过度自信 | `Our results suggest that` |
| `As can be seen from...` | 赘词 | `Figure 1 shows` |

> **本表是英文表。** 中文写作的 AI 痕迹词与降痕改写引用户级技能 `aigc-reduce`（**只写逻辑名**，
> 依据 `.claude/rules/01-agent-standards.md` §1 沙箱外不写路径）。**本文件不复制其中文词表**，避免双源漂移。

---

## 2. 自检协议（Agent 输出前必须执行）

每次生成文本（综述/报告/论文段落）后，检查以下 5 项：

1. **词汇检查**：全文搜索 `Moreover / Furthermore / Additionally / Notably / Importantly / It is worth / plays a crucial / pivotal / underscores / delves / intricate` —— 有任何出现 → 删除或替换
2. **句长检查**：连续 3 句长度都在 15-25 词 → 改写（拉长一句或缩短一句）
3. **开头检查**：连续 3 个段落以相同方式开头 → 变化
4. **信息密度检查**：每段是否有至少一个具体数字、方法名或引用？→ 没有则补
5. **自然度检查**：读一遍，是否有任何句子听起来像"模板"→ 改掉

**评分**：写作质量 10 分，每违反一项扣 2 分。**< 6 分 → 不能交付，需要重写。**

> **本文件只管句子与词，不管结构与论证。** 标题 / 摘要 / 引言问题链 / 方法复现 / 证据映射 / 讨论的判据，
> 唯一源见 `knowledge/writing/paper-argument-structure.md`——**非赛道 T1/T2 的写作任务在此接入结构层**。

> **正面范式**（顶刊写法示例、句法标准、计量/ML 领域对照表、各输出类型要求）见
> `knowledge/writing/academic-writing-patterns.md`。写文献综述、方法论描述、结果报告、论文段落前**读该文件**——
> 本文件的禁用词表告诉你"不能怎么写"，该文件告诉你"该怎么写"。
