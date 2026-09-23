# Research Assistant — 科研全流程智能辅助助手

> 面向科学研究全过程的智能辅助系统 | Multi-Agent 架构 | 10大功能模块 | 80个算法条目

---

## 🎯 项目简介

**Research Assistant** 是一款基于大语言模型驱动的多智能体（Multi-Agent）架构的科研辅助系统，覆盖文献检索、选题分析、数据处理、实验优化、论文排版等科研全流程。

**一句话**：输入自然语言需求 → 输出可用的科研成果。

## ✨ 核心功能

| 模块 | 功能 |
|------|------|
| 📚 文献检索 | OpenAlex/ArXiv 搜索 → 智能筛选 → 自动综述 |
| 🎯 选题分析 | 前沿探测 → 研究空白识别 → 选题推荐 |
| 📊 数据可视化 | 数据清洗 → 统计建模 → 出版级图表 |
| 🔬 实验优化 | 方案设计 → 参数优化 → 敏感性分析 |
| 📝 论文排版 | 模板适配 → 参考文献管理 → 合规检查 |
| 💡 知识问答 | 方法解释 → 公式推导 → 代码示例 |
| 🧠 算法创造 | 形式化 → 设计 → 编码 → 基准 → 验证入库 |
| 🏆 Kaggle 竞赛 | 数据探查 → 基线 → 特征工程 → 模型 → 集成 → 提交 |
| 🏆 数学建模竞赛 | 选题评估 → 审题 → 模型 → 代码 → 检验 → 论文（国赛/美赛） |
| 📄 期刊论文 | 固定尺子 → 逐步骤结构写作 → 结构审稿 → 格式合规 |

## 🏗️ 系统架构

```
用户输入 → 秘书Agent(任务分解) → Orchestrator(协调调度)
           → 11大领域Agent并行/串行执行
           → 共享记忆传递
           → Critic对抗式审查
           → 交付(代码+结果+报告)
```

## 🚀 快速开始

### 环境要求
- Node.js 18+（用于 Claude Code）
- Python 3.10+
- Git

### 安装步骤

```bash
# 1. 安装 Claude Code
npm install -g @anthropic-ai/claude-code

# 2. 克隆项目
git clone https://github.com/xueyuan599-lgtm/research-assistant.git
cd research-assistant

# 3. 安装 Python 依赖
pip install numpy scipy pandas scikit-learn statsmodels
pip install matplotlib seaborn plotnine
pip install xgboost lightgbm catboost
pip install python-docx openpyxl sympy requests tqdm

# 4. 配置 Claude Code（首次使用）
cp .claude/settings.template.json .claude/settings.local.json

# 5. 启动
claude
```

### 使用示例

直接输入需求即可：
```
/research 帮我搜索2024年双重机器学习的前沿文献
```
或
```
用这份数据做DID分析并生成事件研究图
```

## 📁 项目结构

```
research-assistant/
├── agents/                    # 58个 .md（55个领域Agent + 2个元Agent + 1个共享记忆模板）
│   ├── secretary.md           # 任务分解守门人
│   ├── orchestrator.md        # 协调器
│   ├── shared-memory-template.md # 跨 Agent 共享记忆模板
│   ├── literature/            # 文献检索（4个Agent）
│   ├── topic-analysis/        # 选题分析（4个Agent）
│   ├── data-viz/              # 数据可视化（9个Agent）
│   ├── experiment/            # 实验优化（4个Agent）
│   ├── research-qa/           # 科研问答（4个Agent）
│   ├── paper-format/          # 排版格式（4个Agent）
│   ├── algorithm/             # 算法创造（6个Agent）
│   ├── kaggle/                # Kaggle竞赛（8个Agent）
│   ├── mcm/                   # 数学建模竞赛（9个Agent）
│   ├── journal/               # 期刊论文结构写作（2个Agent）
│   └── knowledge/             # 知识检索（1个Agent）
├── .claude/
│   ├── rules/                 # 11条工作流规则（沙箱/Agent规范/写作标准/赛道/成本纪律/工具选型…）
│   ├── skills/                # 17个专业技能（matlab/matplotlib/scikit-learn/数据预处理/数模写作…）
│   ├── agents/                # 3个子Agent工具分档（ra-scan / ra-write / ra-build）
│   ├── commands/              # 6个斜杠命令（research / implement / preprocess / pathplan / paper / research-prompt-refiner）
│   └── settings.template.json # 配置模板（`settings.local.json` 已 gitignore）
├── knowledge/                 # 知识库（133个 .md）
│   ├── _index.md              # 知识库索引
│   ├── algorithm-repository/  # 顶刊算法实现库（80个条目 + _index/_SCHEMA）
│   ├── algorithms/            # 自创算法库（2个算法 + 索引与实验报告）
│   ├── kaggle/                # 12个竞赛模式库
│   ├── mcm/                   # 10个数学建模知识（含 templates/format2026）
│   ├── writing/               # 4个写作判据（正面范式/论证结构/期刊尺子/结构审稿）
│   ├── project-experience/    # 项目经验沉淀
│   ├── outputs/               # PNO 实验 JSON（历史产物）
│   └── optimization-validation-framework.md
├── workflows/                 # 工作流协议（3个：动态管线 + Codex 协作 + 交接 Schema）
├── scripts/                   # 辅助脚本（round_gate / context_monitor / checkpoint 等）
├── outputs/                   # 实战案例输出（15个已入库案例目录）
│   ├── cumcm2024c/            # 数学建模竞赛（CUMCM 2024 C 题）
│   ├── ftz-did/               # DID 因果推断
│   ├── kaggle_titanic/        # Kaggle 竞赛
│   ├── worldcup-prediction/   # 世界杯预测
│   ├── project-webpage/       # 项目主页
│   └── ...                    # 另有 13 个大体量产出目录仅存本地（已 gitignore）
├── CLAUDE.md                  # 项目配置
└── AGENTS.md                  # 架构总览（跨工具，含完整目录树与入库状态）
```

## 📊 项目数据

| 指标 | 数值 |
|------|------|
| 领域 Agent | 55个（11个领域）+ 2个元 Agent（secretary / orchestrator） |
| 专业技能 | 17个 |
| 斜杠命令 | 6个 |
| 工作流规则 | 11条 |
| 代码文件 | 171个（Python 170 · MATLAB 1；全仓 `.py`/`.m`/`.R` 口径） |
| 算法知识库 | 顶刊算法实现库 80个条目 · 自创算法库 2个（含源码、测试与实验报告） |
| 实战案例 | 15个已入库案例目录（含 CUMCM 2024 C 题） |
| 图形文件 | 75个（png/pdf/svg；其中 outputs/ 内 52个） |

> 以上数字以**已入库文件**为准（2026-09-23 复核，口径 = `git ls-files` 可见文件 + 本次提交
> 纳入的新文件）——本地未入库的实验产出（约 434MB，见 `.gitignore`）不计入，故与磁盘实况
> 不同。**代码文件**与**图形文件**为全仓扩展名计数，可用一条 `git ls-files` 复现。改动目录
> 结构时请同步**本节**与上方**项目结构**树；架构的权威描述见 `AGENTS.md`。

