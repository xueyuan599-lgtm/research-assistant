# Pattern: recommendation-ranking

## 匹配条件

| 维度 | 值 |
|------|-----|
| problem_type | recommendation / ranking / implicit_feedback |
| 数据特征 | user_id, item_id, 交互矩阵 |
| 典型场景 | 电商、视频、文章推荐 |
| metric | ndcg / map / recall@k / precision@k |

## 推荐算法栈

### Tier 1: GBDT 排序
1. **LightGBM ranker** — `objective='lambdarank'`, `metric='ndcg'`
   - 特征: user features + item features + interaction features + 交叉特征
2. **CatBoost** — `loss_function='YetiRank'`
3. **XGBoost ranker** — `objective='rank:ndcg'`

### Tier 2: 协同过滤
4. **implicit** (BPR / ALS) — 矩阵分解
5. **LightFM** — 混合 CF + content-based

### Tier 3: 深度学习
6. **Two-Tower / Siamese Network** — user tower + item tower
7. **DCN-v2 (Deep & Cross Network)** — 显式特征交叉
8. **SASRec** — 序列化推荐

## 典型特征工程

| 类型 | 示例 |
|------|------|
| User features | 用户历史行为统计 (click_count, purchase_count, avg_rating) |
| Item features | 物品属性、被交互次数、CTR |
| Interaction | user-item 交叉: co-visitation matrix, 同品类下的排名 |
| Temporal | 最近 N 天交互, 时间衰减权重, 行为序列 |
| Candidate | candidate item 的统计: 该 item 被该 user 相似的用户是否喜欢？ |
| Embedding | MF embedding → 作为 GBDT 特征 |

## 评估策略

- **时间切分 CV**: 用最近 N 天做验证（防止未来信息泄漏）
- **GroupKFold**: 按 user 分组（用户间互斥）
- **NDCG@k**: Kaggle 最常用排序评价指标

## 已知成功案例

| 竞赛 | 最佳排名 | 核心方法 |
|------|---------|---------|
| H&M Fashion | top 1% | LightGBM ranker + 大量特征 + candidate reranking |
| Otto Recommender | top 3% | Co-visitation + LightGBM |
| KDD Cup 2020 | top 5% | Two-Tower + GBDT ensemble |

## 参考来源

- H&M Fashion Kaggle Winner Solution
- LightGBM lambdarank 文档
- NVIDIA Merlin 推荐系统框架
