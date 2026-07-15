# Competition Record Template

> 赛后记录模板。post-mortem-agent 用此模板写入 `competitions/{name}.md`

---

# {competition_name}

## 基本信息
| 字段 | 值 |
|------|-----|
| URL | {url} |
| 类型 | {binary/multiclass/regression/time-series/NLP/CV/recommendation} |
| 评价指标 | {metric} |
| 参赛队伍数 | {n_teams} |
| 时间 | {start_date} – {end_date} |

## 数据概况
| 字段 | 值 |
|------|-----|
| 训练样本 | {n_train} |
| 测试样本 | {n_test} |
| 特征数 | {n_features} |
| 特征类型 | {feature_types} |
| 缺失值比例 | {missing_pct}% |
| train/test 偏移 | {adversarial_auc} |

## 最终成绩
| 指标 | 值 |
|------|-----|
| 最终排名 | {rank} / {total} (top {percentile}%) |
| Public LB | {public_score} |
| Private LB | {private_score} |
| CV (5-fold) | {cv_mean} ± {cv_std} |

## 技术方案
### 特征工程
- {feature_1}
- {feature_2}

### 模型
| 模型 | CV Score | LB Score | 权重 |
|------|----------|----------|------|
| {model_1} | {cv_1} | - | {w1} |
| {model_2} | {cv_2} | - | {w2} |

### 集成方法
- {ensemble_method}

## 经验教训
### 有效的
- {what_worked_1}
- {what_worked_2}

### 无效的
- {what_failed_1}

### 下次改进
- {next_time}

## 匹配的 Pattern
- Pattern: `{pattern_id}` (相似度: {similarity})
- 是否需要更新 pattern: {yes/no}
