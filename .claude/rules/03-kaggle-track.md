# Kaggle Competition Track — 竞赛流水线

> 当用户提出 Kaggle 竞赛任务时触发，提供端到端自动化方案。
> 借鉴 AutoKaggle / PiML / OpenDataSci / CoMind / Socrates 等社区最佳实践。

## 触发条件

以下任一条件满足时，秘书 Agent 应识别为 Kaggle 赛道：

| 触发信号 | 示例 |
|---------|------|
| 关键词 "Kaggle" | "帮我打这个 Kaggle 比赛" |
| 竞赛链接 | "https://kaggle.com/competitions/xxx" |
| 竞赛数据 + 分数目标 | "用这份数据建模型，争取 top 10%" |
| leaderboard 提交要求 | "生成 submission.csv 提交" |
| 关键词 "竞赛" + 数据建模 | "这个数据竞赛怎么搞" |

## 参考项目

本赛道设计融合以下社区最佳实践：

| 项目 | 借鉴点 |
|------|--------|
| **AutoKaggle** (ICLR 2025) | 5-Agent 阶段式流水线，85% 有效提交率 |
| **PiML** (PMLR 2025) | 迭代推理 + 自适应记忆 + 系统性调试 |
| **OpenDataSci** (Apache 2.0) | 自审查 + 并行 Worker + 持久项目记忆 |
| **CoMind / MLE-Live** (CMU) | 社区知识共享 — 阅读论坛/Kernel 后迭代 |
| **Socrates** (COLM 2026) | 质询式 Advisor 协议（只问问题不给答案） |
| **NVIDIA Grandmasters Playbook** | 7 大实战杀招（EDA→基线→特征→集成→伪标签→全量重训） |

## 流水线架构（借鉴 AutoKaggle 的 6 阶段 + PiML 的迭代记忆）

```
Kaggle 赛题
  │
  ├─ Phase 1: data-explorer-agent
  │   借鉴: AutoKaggle Reader + NVIDIA EDA 技巧
  │   输出: 数据探查报告（分布、泄漏检测、train/test 偏移、数据类型识别）
  │
  ├─ Phase 2: baseline-agent
  │   借鉴: PiML 快速迭代 + FLAML/AutoGluon 自动基线
  │   输出: 多模型基线对比表 + CV 分数 + 问题类型自适应工具推荐
  │
  ├─ Phase 3: feature-engineer-agent
  │   借鉴: NVIDIA 海量特征工程 (cuDF GPU加速) + tsfresh 时序特征
  │   输出: 特征生成脚本 + 特征重要性排序 + 选择后的特征集
  │
  ├─ Phase 4: model-builder-agent
  │   借鉴: OpenDataSci 自审查迭代 + PiML 自适应记忆
  │   输出: 精模代码 + 调参记录 + CV 分数 + 伪标签（如适用）
  │
  ├─ Phase 5: ensemble-agent
  │   借鉴: NVIDIA 爬山集成 + stacking + blending
  │   输出: 集成模型 + 最终 CV 分数
  │
  ├─ Phase 6: submission-agent
  │   借鉴: kaggle-skill MCP 工具链
  │   输出: submission.csv + LB 分数 + 版本记录
  │
  └─ Post-mortem: post-mortem-agent
      借鉴: OpenDataSci 项目记忆 + CoMind 社区知识
      输出: 赛后复盘 → 沉淀到 knowledge/kaggle/
```

## 工具选型：按问题类型自适应

**不预设固定算法栈，根据数据特征动态匹配：**

| 数据特征 | 推荐工具 | 理由 |
|---------|---------|------|
| 表格数据 <10万行 | XGBoost, LightGBM, CatBoost, RandomForest | GBDT 在中小规模表格数据上最强 |
| 表格数据 >10万行 | LightGBM (GPU), cuML, FLAML, AutoGluon | GPU 加速 + 自动调参 |
| 图像数据 | PyTorch (timm), TensorFlow (EfficientNet) | 预训练模型 + 微调 |
| 文本/NLP | HuggingFace Transformers, sentence-transformers | LLM embedding + 分类器 |
| 时间序列 | sktime, tsfresh, TimesFM, PatchTST | 时序特征 + 现代 forecasting |
| 多模态 (图+文+表) | AutoGluon multimodal, PyTorch 多分支 | 多模态自动融合 |
| 推荐系统 | LightGBM ranker, CatBoost ranking, implicit | Learning-to-rank |
| 小样本 <1000行 | CatBoost, 贝叶斯优化, 强正则化 XGBoost | 抗过拟合优先 |
| 高维稀疏 >1000特征 | Lasso, ElasticNet, XGBoost (max_bin=256) | 特征选择 + 稀疏感知 |

**基线阶段自动扫描，选择 CV 最优的前 3 个方向进入精模阶段。**

## 关键差异化规则

### vs 学术流水线

| 维度 | 学术流水线 | Kaggle 赛道 |
|------|-----------|------------|
| 目标 | 方法论正确 | **分数最大化** |
| 模型可解释性 | 必须 | 可选（黑盒可接受） |
| 特征工程 | 审慎、领域驱动 | **激进、自动化优先** |
| 集成 | 可选 | **强制（至少 stacking）** |
| CV 策略 | k-fold (k=5/10) | **stratified group k-fold + 时间序列 split** |
| 迭代速度 | 慢、深 | **快、多实验并行** |
| 伪标签 | 一般不用 | **强烈推荐（如有 test 数据）** |
| 数据泄漏 | 仔细检查 | **极其仔细检查（LB  probe 风险）** |
| 交付物 | 论文 + 复现代码 | **submission.csv + notebook + 特征工程脚本** |

### 强制规则

1. **必须检测数据泄漏**：train/test 分布偏移、目标泄漏、时间泄漏
2. **必须跑基线**：至少 3 种模型类型（线性 + 树模型 + 一个神经网络或 AutoML）
3. **必须集成**：至少 blending（加权平均），推荐 stacking
4. **CV 必须可靠**：时间序列用 time series split，有 group 用 group k-fold
5. **必须跟踪 LB**：每次提交记录 CV 分数 + LB 分数 + 差异
6. **必须保持实验日志**：每次实验的参数、分数、特征变更

## 集成 kaggle-skill MCP

本赛道依赖 `kaggle-skill` MCP server 进行：
- 竞赛信息获取（描述、数据、评价指标、时间线）
- 数据集下载 / 模型上传
- Notebook 执行
- 提交管理 + Leaderboard 查询
- Badge 收集

安装命令见 `agents/kaggle/agent.md`。
