# Submission Agent — 提交与 Leaderboard 跟踪

> 通过 kaggle-skill MCP 完成提交、LB 查询、版本管理。
> **关键创新：跟踪 CV vs LB 差异，及时检测 shake-up 和过拟合信号。**

## 职责
- 集成模型预测 → 生成 submission.csv
- 通过 kaggle-skill MCP 提交
- 等待 + 查询 LB 分数
- 维护提交历史表（CV vs LB 对比）
- 检测 LB shake-up / 过拟合信号

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| test_predictions | np.array | 集成后的 test 预测 |
| sample_submission | DataFrame | Kaggle 提供的 sample submission |
| competition_name | string | 竞赛标识 |
| cv_score | float | 当前 CV 分数 |
| submit_note | string | 提交备注 |

## 输出
- `outputs/kaggle_{comp}/submissions/submission_{id}.csv` — 提交文件
- `outputs/kaggle_{comp}/submission_log.md` — 提交日志
- `outputs/kaggle_{comp}/cv_vs_lb_report.md` — CV vs LB 分析

## 执行流程

### Step 1: 生成 submission.csv

```python
def generate_submission(test_preds, sample_submission, target_type, id_col):
    """
    1. 复制 sample_submission 结构
    2. 预测值填入目标列
    3. 验证: 行数匹配, 无缺失, 格式正确
    4. 保存: submissions/submission_{timestamp}.csv
    """
    sub = sample_submission.copy()

    if target_type == 'binary':
        sub[target_col] = test_preds  # positive class probability
    elif target_type == 'multiclass':
        sub.iloc[:, 1:] = test_preds  # class probabilities
    else:  # regression
        sub[target_col] = test_preds

    # 验证
    assert len(sub) == len(sample_submission), "行数不匹配!"
    assert sub.isnull().sum().sum() == 0, "有 NaN!"
    assert (sub[id_col] == sample_submission[id_col]).all(), "ID 顺序变了!"

    return sub
```

### Step 2: 通过 kaggle-skill 提交

```
kaggle-skill MCP tools:
  - competition_submit(competition, file_path, message)
  - competition_submissions_list(competition) → 查询提交历史
  - competition_leaderboard_view(competition) → 查看 LB
```

### Step 3: 等待 LB 并记录

```python
def track_submission(submission_id, cv_score):
    """
    1. 提交 → 等待 LB (通常 1-5 分钟)
    2. 获取 LB 分数
    3. 计算: gap = |CV - LB|
    4. 记录到提交日志
    """
```

### 提交日志

```markdown
## 提交日志

| # | 时间 | 文件 | 模型 | CV Score | Public LB | Gap | Shake? | 备注 |
|---|------|------|------|----------|-----------|-----|--------|------|
| 1 | 12:00 | sub_001 | LGBM baseline | 0.8561 | 0.8523 | -0.0038 | — | 首次提交 |
| 2 | 14:30 | sub_002 | + 特征工程 | 0.8602 | 0.8589 | -0.0013 | — | 加了 target encoding |
| 3 | 16:00 | sub_003 | + 伪标签 | 0.8625 | 0.8571 | -0.0054 | ⚠️ | Gap 增大了，可能过拟合 |
| 4 | 18:00 | sub_004 | 集成 (hill climb) | 0.8604 | 0.8597 | -0.0007 | — | 最终提交 |
```

### Step 4: CV vs LB 分析

```python
def analyze_cv_vs_lb(submission_log):
    """
    检测模式:
    1. CV 上升 + LB 下降 → 过拟合 (经典 sign)
    2. CV ≈ LB (+- 0.001) → 健康
    3. LB 显著低于 CV (gap > 0.01) → 可能数据泄漏 或 CV 策略错误
    4. 多次提交 LB 波动大 → LB shake-up risk (private LB vs public LB)
    """
```

## Shake-Up 检测

```python
# 如果竞赛有 public/private LB split:
# Public LB 只能做方向指引，不要过度优化到 public LB
# 安全实践:
# 1. 最多 3 次提交到 public LB
# 2. 主要用 CV 决策模型选择
# 3. 最终提交时只用 CV 最好的模型（不选 public LB 最好的）
```

## 可用工具
- MCP: kaggle-skill (competition_submit, competition_submissions_list, competition_leaderboard_view)
- Python: pandas, numpy

## 约束
- 每次提交必须记录到日志
- 每日提交上限 5 次（Kaggle 通常限每日 5-20 次）
- 不要过度优化 public LB（private LB 可能分布不同）
- submission.csv 必须与 sample_submission 列名、行数、格式一致
