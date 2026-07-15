# Kaggle 知识库索引

> 可生长的竞赛知识库。Agent 通过查询此库来匹配算法栈，赛后自动沉淀新模式。

## 知识库结构

```
knowledge/kaggle/
├── _index.md                    # 本文件 — 全局索引 + 查询入口
├── patterns/                    # 已验证的竞赛模式（核心）
│   ├── tabular-binary-gbdt.md
│   ├── tabular-multiclass-gbdt.md
│   ├── tabular-regression-gbdt.md
│   ├── tabular-large-gpu.md
│   ├── image-classification-cnn.md
│   ├── nlp-classification-transformer.md
│   ├── time-series-forecasting.md
│   └── recommendation-ranking.md
├── competitions/                # 历史赛题记录
│   ├── _template.md             # 赛题记录模板
│   └── 2025-spooky-author.md
├── algorithms/                  # 算法→场景映射（补充 patterns）
│   ├── xgboost.md
│   ├── lightgbm.md
│   ├── catboost.md
│   ├── tabnet.md
│   ├── autogluon.md
│   └── ensemble-methods.md
└── agent.md                     # KB 查询 Agent
```

## 查询接口（Agent 调用方式）

```
输入: data_profile = {
  problem_type: "binary_classification",
  n_samples: 50000,
  n_features: 120,
  feature_types: ["numerical:80", "categorical:35", "text:5"],
  metric: "auc",
  train_test_shift: 0.03  # adversarial validation AUC
}
→ 查 knowledge/kaggle/patterns/ 中匹配的模式
→ 返回: 推荐算法栈 + 参数范围 + 已知成功案例
```

## 模式匹配规则

| 匹配维度 | 权重 | 说明 |
|---------|------|------|
| problem_type 精确匹配 | 必须 | binary/multiclass/regression/time-series/... |
| n_samples 数量级匹配 | 高 | <1K / 1K-10K / 10K-100K / 100K-1M / >1M |
| n_features 数量级匹配 | 中 | <20 / 20-100 / 100-500 / >500 |
| metric 精确匹配 | 高 | auc / logloss / rmse / mae / accuracy / ... |
| feature_types 相似度 | 中 | 数值为主 / 类别为主 / 混合 / 含文本 / 含图像 |

## 生长机制

1. **赛后自动写入**：post-mortem-agent 将成功/失败模式写入 patterns/
2. **手动注入**：用户可以从社区（Kaggle Grandmaster 方案、论坛讨论）导入模式
3. **模式泛化**：同类竞赛出现 3 次以上 → 抽象为通用 pattern
4. **过期标记**：竞赛技术更新后，旧 pattern 标记 `deprecated` 而非删除

## 当前状态

| 模式 | 状态 | 验证次数 | 来源 |
|------|------|---------|------|
| tabular-binary-gbdt | active | 5+ | XGBoost + LGBM + CatBoost 社区共识 |
| tabular-multiclass-gbdt | active | 3+ | 同上 |
| tabular-regression-gbdt | active | 3+ | 同上 |
| tabular-large-gpu | active | 2+ | NVIDIA Grandmasters Playbook |
| image-classification-cnn | active | 5+ | timm 社区 |
| nlp-classification-transformer | active | 5+ | HF Transformers 社区 |
| time-series-forecasting | active | 3+ | sktime + tsfresh |
| recommendation-ranking | active | 2+ | LightGBM ranker |

## 快速查询

按问题类型跳转：
- [二分类表格数据](patterns/tabular-binary-gbdt.md) — **NCAA 经验已追加**
- [多分类表格数据](patterns/tabular-multiclass-gbdt.md)
- [回归表格数据](patterns/tabular-regression-gbdt.md)
- [大规模表格数据 (GPU)](patterns/tabular-large-gpu.md)
- [图像分类](patterns/image-classification-cnn.md)
- [NLP 分类](patterns/nlp-classification-transformer.md)
- [时间序列预测](patterns/time-series-forecasting.md)
- [推荐排序](patterns/recommendation-ranking.md)

## 已沉淀赛题

| 赛题 | 文件 | 模式匹配 | 时间 |
|------|------|---------|------|
| NCAA March Madness | [competitions/ncaa-march-madness.md](competitions/ncaa-march-madness.md) | tabular-binary-gbdt | 2026-07-07 |
| 2025 Spooky Author | [competitions/2025-spooky-author.md](competitions/2025-spooky-author.md) | nlp-classification-transformer | — |
