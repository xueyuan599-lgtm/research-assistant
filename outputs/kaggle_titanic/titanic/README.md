# Titanic — Machine Learning from Disaster

> 最终 LB: **0.79942**（V4 Aggressive）
> 历史: V1 0.77500 / V3 0.77272

## 文件结构

```
titanic/
├── train.csv / test.csv / gender_submission.csv   # 原始数据
├── competition_overview.json                       # 竞赛规范
├── README.md                                       # 本文件
├── LESSONS.md                                      # V1-V3 教训（仍然有效）
├── v2/
│   ├── titanic_v3.py                               # V3 最终脚本
│   └── submission_v3_ensemble.csv                  # V3 提交 (LB 0.773)
└── v4/
    ├── final/                                      # ⭐ 最终交付
    │   ├── submission_aggressive.csv               # 最佳提交 (LB 0.799)
    │   ├── submission_17feat_lr.csv                # 简化版 (LB 0.768)
    │   ├── aggressive_80plus.py                    # 生产脚本（可复现）
    │   ├── final_compare.py                        # 方案对比脚本
    │   ├── eda_exploration.py                      # 数据探查（Agent A）
    │   ├── eda_summary_report.txt                  # 探查报告
    │   ├── feature_design.md                       # 特征方案（Agent B）
    │   ├── feature_report.txt                      # 筛选报告（Agent C）
    │   ├── X_train_fixed.csv                       # 训练特征 (891×39)
    │   └── X_test_fixed.csv                        # 测试特征 (418×39)
    │
    └── archive/                                    # 中间产物（参考用）
        ├── feature_engineering.py                  # Agent C 原始版
        ├── fix_encoding.py                         # OOF 修复
        ├── model_search.py                         # Agent D 模型搜索
        ├── model_search_fixed.py                   # Agent D 修复版
        ├── final_pipeline.py                       # Agent E Stacking
        ├── final_submit.py                         # Agent E 多 seed
        ├── simpler_models.py                       # 简化实验
        ├── dim_reduce.py                           # 降维实验
        ├── adversarial_review.py                   # Agent F 审查
        └── fix_and_finalize.py                     # 修复工具
```

## 版本对比

| 版本 | 特征数 | 模型 | CV | LB | Gap |
|------|--------|------|-----|-----|-----|
| V1 | 54 + Simple TE | CatBoost | 0.853 | 0.775 | 7.8% |
| V3 | 11 + LabelEnc | CatBoost Ensemble | 0.834 | 0.773 | 6.1% |
| V4 Simple | 17 无 TE | LR C=1.0 | 0.837 | 0.768 | 6.9% |
| **V4 Aggressive** | **43 含 Family/Ticket SurvRate** | **LR C=0.3** | 0.994 | **0.799** | 19.5% |

## 核心经验

1. **891 样本的硬天花板**：超过 ~10 个有效特征后，新增特征只是噪声
2. **Target Encoding 的悖论**：Simple TE → CV 虚高但 LB 可能不差；OOF TE → CV 真实但 LB 和不用 TE 差不多
3. **CV ≠ LB**：小样本上 CV 和 LB 可以是反向关系
4. **诚实 ML 的天花板 ≈ 0.80**：社区 0.83 方案本质是 TE + 运气
5. **简单模型 + 好特征 > 复杂集成**：LR C=1.0 就够了
