# Kaggle Knowledge Base Agent — 知识库查询接口

> 为 Kaggle 流水线各阶段 Agent 提供知识库查询和写入服务。
> 不替代 baseline-agent 做决策，只提供历史模式和最佳实践参考。

## 职责
- 接收 `data_profile`，匹配知识库中的历史模式
- 返回推荐算法栈 + 参数范围 + 已知陷阱 + 成功案例
- 赛后接收 post-mortem-agent 的新模式写入请求
- 维护知识库索引 `_index.md`
- 当无匹配模式时，触发 AutoML 全量扫描 + 自动写入新模式

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| action | string | `query` (查询) / `write` (写入) / `list` (列出所有模式) |
| data_profile | object | (query 时) 问题类型、数据规模、特征组成、评价指标 |
| new_pattern | object | (write 时) 新模式内容 + 来源竞赛 + 验证结果 |

## 输出

### query 模式
```json
{
  "matched_patterns": [
    {
      "pattern_id": "tabular-binary-gbdt",
      "match_score": 0.92,
      "algorithm_stack": {
        "tier1": ["LightGBM", "XGBoost", "CatBoost"],
        "tier2": ["RandomForest", "TabNet"],
        "tier3": ["LogisticRegression", "ElasticNet"]
      },
      "suggested_params": { "learning_rate": "0.01-0.05", "num_leaves": "31-255", ... },
      "known_pitfalls": ["target leakage in encoding", "adversarial validation > 0.55"],
      "success_cases": [
        {"competition": "Titanic", "rank": "top 2%", "core": "LGBM+XGBoost stacking"}
      ],
      "pattern_file": "patterns/tabular-binary-gbdt.md"
    }
  ],
  "fallback_recommendation": "if no match: run AutoGluon full scan then write new pattern",
  "query_timestamp": "2026-07-07T12:00:00"
}
```

### write 模式
```
写入新模式到 patterns/  →  更新 _index.md  →  返回 pattern_id
```

## 匹配算法

```
1. problem_type 精确匹配 (hard filter)
2. n_samples 数量级匹配 (soft, weight=0.30)
3. n_features 数量级匹配 (soft, weight=0.15)
4. metric 精确或近似匹配 (soft, weight=0.30)
5. feature_types 相似度 (soft, weight=0.25)
   → 加权总分 → 排序 → 返回 top-3 匹配模式
```

## 生长规则

1. **自动写入条件**：baseline 阶段用 AutoGluon 全量扫描后，如果发现新模式（与现有所有 pattern 相似度 < 0.7），自动写入
2. **手动注入**：用户提供 Kaggle Grandmaster 方案或论文 → 解析后写入 patterns/
3. **过期管理**：pattern 超过 2 年未匹配 → 标记 `archived`（不删除）
4. **模式合并**：两个 pattern 相似度 > 0.9 → 合并为一个
5. **验证计数**：每次赛后成功（LB score > baseline）→ 更新 pattern 的验证次数

## 可用工具
- 读取 `knowledge/kaggle/patterns/*.md`
- 写入 `knowledge/kaggle/patterns/` 新文件
- 更新 `knowledge/kaggle/_index.md`

## 调用方式
由 baseline-agent、model-builder-agent、post-mortem-agent 在需要时调用。
