# Research Assistant — 科研全流程智能辅助助手

> 面向科学研究全过程的智能辅助系统 | Multi-Agent 架构 | 8大功能模块 | 70+算法知识库

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

## 🏗️ 系统架构

```
用户输入 → 秘书Agent(任务分解) → Orchestrator(协调调度)
           → 7大领域Agent并行/串行执行
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
├── agents/                    # 56个Agent定义（核心）
│   ├── secretary.md          # 任务分解守门人
│   ├── orchestrator.md       # 协调器
│   ├── literature/           # 文献检索（4个Agent）
│   ├── topic-analysis/       # 选题分析（4个Agent）
│   ├── data-viz/             # 数据可视化（9个Agent）
│   ├── experiment/           # 实验优化（4个Agent）
│   ├── research-qa/          # 科研问答（4个Agent）
│   ├── paper-format/         # 排版格式（4个Agent）
│   ├── algorithm/            # 算法创造（6个Agent）
│   ├── kaggle/               # Kaggle竞赛（8个Agent）
│   └── mcm/                  # 数学建模竞赛（9个Agent）
├── knowledge/                # 知识库
│   ├── algorithm-repository/ # 70+算法文档
│   ├── kaggle/               # 竞赛模式库
│   └── mcm/                  # 数学建模竞赛知识库
├── outputs/                  # 实战案例输出
│   ├── cumcm2024c/           # 数学建模竞赛
│   ├── 2026_7_25/            # CUMCM 2025 A题（烟幕干扰弹）
│   ├── ftz-did/              # DID因果推断论文
│   ├── kaggle_titanic/       # Kaggle竞赛
│   ├── worldcup-prediction/  # 世界杯预测
│   └── ...
├── .claude/rules/            # 8条工作流规则
└── CLAUDE.md                 # 项目配置
```

## 📊 项目数据

| 指标 | 数值 |
|------|------|
| Agent 定义 | 56个 |
| 代码文件 | 170+（Python / MATLAB / R） |
| 算法知识库 | 70+个算法文档 |
| 实战案例 | 10+个完整案例（含 2024 C 题、2025 A 题数学建模） |
| 生成图表 | 50+张出版级图表 |

## 📄 许可证

本项目为浙江研究院、文华创新学院2026年智能体创新应用培育项目成果。
