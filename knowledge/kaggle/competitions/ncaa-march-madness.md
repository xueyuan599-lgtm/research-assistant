# NCAA March Madness (Kaggle)

> NCAA 一级男子/女子篮球锦标赛结果预测。给定常规赛数据和球队信息，预测每场 tournament 比赛的结果。

---

## 基本信息

| 字段 | 值 |
|------|-----|
| URL | https://www.kaggle.com/competitions/ncaa-march-madness-2026 |
| 类型 | binary_classification |
| 评价指标 | Brier Score |
| 参赛队伍数 | ~1,500+ |
| 时间 | 每年 3月-4月（长期竞赛） |

## 数据概况

| 字段 | 值 |
|------|-----|
| 训练样本 | 男子 2,898（1449 场 x 2 方向）/ 女子 1,922（961 场 x 2 方向） |
| 测试样本 | 男子 66,430 / 女子 65,703 对比赛 |
| 特征数 | 男子 40 / 女子 34 |
| 特征类型 | 全部数值（Elo 差、效率差、种子差、滚动均值差 + Massey Ordinals） |
| 缺失值比例 | <1%（填充中位数） |
| 数据文件 | 34 个 CSV（男子 20 个 + 女子 14 个）|

## 最终成绩（方案A）

| 指标 | 男子 | 女子 |
|------|------|------|
| Ensemble Brier (CV) | 0.1796 | 0.1412 |
| CatBoost CV Brier | 0.17999 | 0.14208 |
| LightGBM CV Brier | 0.17963 | 0.14551 |
| 特征数 | 40 | 34 |
| 训练样本 | 2,898 | 1,922 |

## 技术方案

### 特征工程

| 特征组 | 具体特征 | 重要性 |
|--------|---------|--------|
| Elo 评分 | `elo_diff` | **最高** — 两个性别的 top 1 特征 |
| 赛季聚合 | `avg_PF/PA/win_rate/net_avg/seed/num_games` 差值 | 高 |
| 滚动窗口 | 最近 3/5/10/20 场的得分/失分/胜率 | 中-高（10/20 场优于 3/5 场） |
| 效率指标 | `off_eff/def_eff/net_eff` 赛季值 + 滚动 3/5/10 | 中-高 |
| Massey Ordinals（仅男子） | 7 个系统的均值/标准差/极值 | 中 |
| 场景特征 | `rest_diff/is_neutral/same_conf` | 低（几乎无贡献） |

**特殊处理**:
- 双向样本增广：每场比赛生成 (赢家-输家, label=1) + (输家-赢家, label=0)，样本数翻倍
- 仅使用 7 个全覆盖 Massey 系统（AP, COL, DOL, MOR, POM, USA, WLK），排除不完整系统
- 滚动窗口特征按每支球队-赛季独立计算，避免跨赛季泄漏

### 模型

| 模型 | 男子 CV Brier | 女子 CV Brier | 男子权重 | 女子权重 |
|------|-------------|-------------|---------|---------|
| CatBoost | 0.17999 | 0.14208 | 0.500 | 1.000 |
| LightGBM | 0.17963 | 0.14551 | 0.500 | 0.000 |

CatBoost 参数（男子）: `iterations=500, lr=0.081, depth=5, l2_leaf_reg=6.96, border_count=101`
CatBoost 参数（女子）: `iterations=1000, lr=0.029, depth=4, l2_leaf_reg=4.13, border_count=141`
LightGBM 参数（男子）: `n_estimators=2000, lr=0.016, num_leaves=225, min_child_samples=76`
LightGBM 参数（女子）: `n_estimators=500, lr=0.022, num_leaves=182, min_child_samples=91`

### 调参方法

Optuna TPESampler (seed=42), 50 trials per model.
搜索空间:
- CatBoost: iterations [500-3000], lr [0.01-0.1], depth [4-10], l2_leaf_reg [1-10], border_count [32-255]
- LightGBM: n_estimators [500-3000], lr [0.01-0.1], num_leaves [31-255], min_child_samples [20-100], subsample [0.7-1.0], colsample_bytree [0.7-1.0], reg_alpha/lambda [0-1]

### CV 策略

Expanding Window:

男子:
- Fold 1: train 2003-2008 → val 2009-2012
- Fold 2: train 2003-2012 → val 2013-2016
- Fold 3: train 2003-2016 → val 2017-2019
- Fold 4: train 2003-2019 → val 2021-2023
- Fold 5: train 2003-2023 → val 2024-2025

女子:
- Fold 1: train 2010-2013 → val 2014-2016
- Fold 2: train 2010-2016 → val 2017-2019
- Fold 3: train 2010-2019 → val 2021-2022
- Fold 4: train 2010-2022 → val 2023-2025

### 集成方法

1. 1/Brier Score 加权平均（模型级）
2. SLSQP 爬山优化（在 OOF 预测上搜索最优权重，bounds=[0,1], sum=1）
3. **男子**: 优化后 50/50 均分；**女子**: 优化后仅使用 CatBoost（LGBM 权重降至 0）

## 经验教训

### 有效的

- **Elo 评分是体育预测最可靠的单特征**，优于种子、Massey 排名等静态指标
- **双向样本增广**使训练样本翻倍，显著提升模型稳定性
- **Expanding Window CV** 暴露了疫情后赛季的分布偏移（Fold 4 Brier 显著升高）
- **CatBoost 在小数据集上的表现优于 LightGBM**（女子数据验证）
- **Optuna 调参**中等规模搜索（50 trials）即能找到接近最优的超参

### 无效的

- `is_neutral` / `same_conf` / `rest_diff` 场景特征几乎无预测力
- LightGBM 在女子数据上未超越 CatBoost（爬山优化权重归零）
- 伪标签因阈值设定不合理（0.9）导致可用样本过少，未能提升性能
- 男子 Maskey 特征虽有用但增益有限（仅 +0.001-0.002），7 个系统可能不够

### 下次改进

1. 加入 XGBoost 和/或 Random Forest 提升集成多样性
2. 引入 TabNet 或 FT-Transformer 进行特征交互捕获
3. 更精细的 COVID 赛季处理（降权/标识/独立 fold）
4. 加入教练经验、赛区强度、对阵前 25 战绩等高级特征
5. 使用 Purged Group Time Series Split 减少数据泄漏
6. 伪标签使用 soft labels + confidence weighting
7. 按种子级别或赛区强弱分组建模
8. 赛后集成 Top 10% 方案的公开 kernel

## 匹配的 Pattern

- Pattern: `tabular-binary-gbdt` (相似度: 高)
- 是否需要更新 pattern: **是** — 追加 NCAA 特定经验（见 pattern 文件更新）
