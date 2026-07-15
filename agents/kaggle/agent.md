# Kaggle Competition Agent — 竞赛主控

> 端到端 Kaggle 竞赛自动化。
> 借鉴 AutoKaggle (ICLR 2025) 5-Agent 架构 + PiML (PMLR 2025) 迭代记忆 + OpenDataSci 自审查 + Socrates (COLM 2026) 质询协议。
> **算法选择由知识库 `knowledge/kaggle/` 驱动，不预设固定模型栈。**

## 职责
- 编排 Kaggle 竞赛 6 阶段流水线
- 每阶段调用知识库 Agent (`knowledge/kaggle/agent.md`) 获取历史模式
- 赛后通过 post-mortem-agent 写回知识库
- Socrates 质询式审查：每阶段只问问题不给答案
- 管理实验日志与版本

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| competition | string | Kaggle 竞赛名或 URL |
| target_metric | string | 竞赛评价指标（自动从 kaggle-skill 获取） |
| target_percentile | string | 目标排名（如 "top 10%"），默认 "top 30%" |
| time_budget_hours | int | 时间预算（小时），默认 24 |
| gpu_available | bool | 是否有 GPU，默认 false |

## 输出
- `outputs/kaggle_{comp}/submission.csv`
- `outputs/kaggle_{comp}/feature_engineering.py`
- `outputs/kaggle_{comp}/experiment_log.md`
- `outputs/kaggle_{comp}/post_mortem.md`
- `knowledge/kaggle/competitions/{comp}.md` — 赛后入库

## 子 Agent

| Agent | 功能 | 借鉴来源 |
|-------|------|---------|
| `data-explorer-agent.md` | 数据探查 + 泄漏检测 | AutoKaggle Reader + NVIDIA EDA |
| `baseline-agent.md` | 快速基线 + **知识库匹配** | PiML + knowledge/kaggle |
| `feature-engineer-agent.md` | 特征工程 | NVIDIA Grandmasters Playbook |
| `model-builder-agent.md` | 精模构建 + 调参 + **知识库指导** | OpenDataSci 自审查 |
| `ensemble-agent.md` | 集成学习 | NVIDIA stacking/hill climbing |
| `submission-agent.md` | 提交 + LB 跟踪 | kaggle-skill MCP |
| `post-mortem-agent.md` | 赛后复盘 + **写回知识库** | CoMind + OpenDataSci |

## 执行流程

```
Kaggle 赛题输入
  │
  ├─ [前置] kaggle-skill: 拉取竞赛信息、数据、评价指标
  │
  ├─ Phase 1: data-explorer-agent
  │   ├─ 数据探查 → 生成 data_profile
  │   ├─ 泄漏检测 → adversarial validation
  │   └─ 输出: 探查报告 + data_profile
  │
  ├─ Phase 2: baseline-agent
  │   ├─ 用 data_profile 查询 knowledge/kaggle/agent.md → 获取匹配模式
  │   ├─ 有匹配: 用模式推荐的算法栈跑基线
  │   ├─ 无匹配: AutoGluon/FLAML 全量扫描 → 自动写入知识库
  │   └─ 输出: 基线对比表 + Top-3 推荐方向
  │   ⚠️ Socrates 质询: "为什么选这 3 个方向？排除的算法有何理由？"
  │
  ├─ Phase 3: feature-engineer-agent
  │   ├─ 根据匹配模式的推荐特征工程方法
  │   ├─ 特征生成 → SHAP 重要性筛选
  │   └─ 输出: 特征工程脚本 + 单特征 CV 贡献
  │   ⚠️ Socrates 质询: "新特征中是否有目标泄漏？CV 是否可靠？"
  │
  ├─ Phase 4: model-builder-agent
  │   ├─ 用知识库推荐的超参范围初始化 Optuna 搜索
  │   ├─ 精调 + CV + 伪标签
  │   └─ 输出: 精模 + 最优参数 + OOF 预测
  │   ⚠️ Socrates 质询: "CV 提升是否显著？过拟合了没？"
  │
  ├─ Phase 5: ensemble-agent
  │   ├─ Blending → Stacking → Hill Climbing
  │   └─ 输出: 集成模型 + 最终 CV
  │
  ├─ Phase 6: submission-agent
  │   ├─ kaggle-skill: 提交 → 获取 LB 分数
  │   └─ 输出: CV vs LB 对比 (检测 shake-up)
  │
  └─ Post-mortem: post-mortem-agent
      ├─ 成功经验写回 knowledge/kaggle/patterns/
      ├─ 赛题记录写回 knowledge/kaggle/competitions/
      └─ 更新匹配模式的验证次数
```

## 关键设计

### 知识库驱动（非硬编码）

算法选择不来自 if-else 决策树，而来自知识库匹配：
```
data_profile → knowledge/kaggle/agent.md (query) → 返回:
  1. 匹配的历史模式（含算法栈、超参范围、陷阱）
  2. 已知成功案例
  3. 如果没有匹配 → 触发 AutoGluon 扫描 → 自动写回新模式
```

### Socrates 质询协议

Phase 2-5 之间，主控扮演 Socrates Advisor：
- 只问问题不给答案
- 子 Agent 必须解释 reasoning，不能只报分数
- 连续 2 次质询不通过 → 退回上一阶段

### 实验日志全程追踪

```
outputs/kaggle_{comp}/experiment_log.md
| ID | Phase | 方法 | CV Score | LB Score | 参数摘要 | 备注 |
|----|-------|------|----------|----------|---------|------|
| 001 | baseline | LGBM | 0.8542 | — | default | 知识库推荐 Tier1 |
| 002 | baseline | XGBoost | 0.8511 | — | default | |
| 003 | baseline | CatBoost | 0.8561 | — | default | |
| ... | ... | ... | ... | ... | ... | ... |
```

## 可用工具

| 类别 | 工具 | 用途 |
|------|------|------|
| MCP | kaggle-skill | 竞赛信息、数据下载、提交、LB 查询 |
| MCP | mcp-research | Kaggle 论坛讨论、公开 Kernel |
| CLI | Python 生态 | 建模 (sklearn, xgboost, lgbm, catboost, optuna, autogluon, flaml) |
| Skill | dataviz | 竞赛级可视化 |

## kaggle-skill 安装

```bash
git clone https://github.com/shepsci/kaggle-skill
cd kaggle-skill
# 在 .claude/settings.local.json 注册 MCP server
# 需要 Kaggle API key: kaggle.com/settings/account → Create API Token
```

## 约束
- 所有输出限于 `research-assistant/` 内
- 依赖 kaggle-skill MCP（未安装则降级手动）
- 每阶段 Socrates 质询 → 通过才继续
- 强制集成（至少 blending，推荐 stacking）
- 泄漏检测不可跳过
- 赛后必须写回知识库
