# Post-Mortem: Kaggle NCAA March Madness — 方案A GBDT集成路线

> 生成时间: 2026-07-07
> 模型: CatBoost + LightGBM + Optuna + Blending Ensemble

---

## 方法总览

采用 GBDT 集成路线，对男子和女子分别训练独立的 CatBoost 和 LightGBM 模型，使用 Optuna 贝叶斯调参和 expanding window CV，最后以 Brier Score 倒数加权 blending 集成。

**核心流水线**:
1. 数据加载：34 个 CSV（男子 + 女子，包含常规赛详细数据、锦标赛结果、种子、Massey Ordinals、球队会议归属）
2. Elo 评分系统：按时间顺序计算每支球队每个赛季的赛前 Elo 评分
3. 球队赛季特征：赛季聚合 + 滚动窗口（最后 3/5/10/20 场）的得分/失分/胜率 + 进攻/防守/净效率
4. 双向样本增广：每场锦标赛生成 (赢家-输家, label=1) 和 (输家-赢家, label=0)
5. Expanding Window CV：按年份划分训练/验证集
6. Optuna 调参：每性别 50 trials 超参数搜索
7. Blending 集成：1/Brier 加权平均 + SLSQP 爬山优化
8. 生成 submission.csv

---

## 特征工程分析

### 特征体系（共 40 个男子 / 34 个女子）

| 特征组 | 特征数 | 说明 |
|--------|--------|------|
| **Elo 评分** | 1 | `elo_diff` — 比赛双方 Elo 评分之差 |
| **赛季聚合** | 6 | `avg_PF/PA/win_rate/net_avg/seed/num_games` 的差值 |
| **滚动窗口得分** | 12 | 最后 3/5/10/20 场的平均得分/失分/胜率差值 |
| **效率指标** | 12 | 赛季 + 滚动 3/5/10 场的进攻效率/防守效率/净效率差值 |
| **Massey Ordinals (仅男子)** | 7 | 7 个系统的均值/标准差/最好/最差/中位数/极差 |
| **场景特征** | 3 | `rest_diff`/`is_neutral`/`same_conf` |

### 哪些特征有用

1. **elo_diff** — 两个性别的 Top 1 特征，说明 Elo 体系在 NCAA 预测中非常有效
2. **off_eff_diff / net_eff_diff** — 效率差值类特征在各 fold 中排名靠前
3. **win_rate_l10_diff / win_rate_l20_diff** — 近期胜率比赛季胜率更有预测力
4. **massey_mean_diff**（男子）— 7 个 Massey 系统的平均排名差值排名第 2-3

### 哪些特征贡献较小

1. **is_neutral / same_conf** — 锦标赛绝大多数在中立场地进行，`is_neutral` 几乎无区分度
2. **rest_diff** — NCAA 锦标赛赛程规则严格，各队休息天数差异有限
3. **seed_diff** — 种子差单独预测力弱，信息已被 Elo 和 Massey 吸收
4. **avg_PF_l3_diff / avg_PA_l3_diff** — 最近 3 场的窗口过小，噪声大

### 为什么 elo_diff 是王者特征

Elo 评分天然整合了：
- 赛季中的胜负历史（通过分数更新）
- 对手强度（战胜强队 vs 战胜弱队的 Elo 奖励不同）
- 时间衰减（赛季间 75% 回归）
- 主客场调整（虽然锦标赛全是中立场地）

种子和 Massey 排名是静态快照，而 Elo 是动态时序化的球队强度度量。

### 男子 vs 女子特征差异

| 维度 | 男子 | 女子 |
|------|------|------|
| 起始年份 | 2003（详细数据） | 2010（详细数据） |
| 特征数 | 40（含 Massey） | 34（无 Massey） |
| 样本量 | 2,898 | 1,922 |
| 关键差异 | 有 Massey Ordinals 7 系统 | 无 Massey 数据 |

---

## 模型选择对比

### CatBoost vs LightGBM

| 指标 | CatBoost (M) | LightGBM (M) | CatBoost (W) | LightGBM (W) |
|------|-------------|-------------|-------------|-------------|
| CV Brier | 0.17999 | 0.17963 | 0.14208 | 0.14551 |
| 集成权重 | 0.500 | 0.500 | 1.000 | 0.000 |
| 最佳迭代数 | 500 | 2000 | 1000 | 500 |
| 学习率 | 0.081 | 0.016 | 0.029 | 0.022 |
| depth / num_leaves | depth=5 | leaves=225 | depth=4 | leaves=182 |

**关键发现**:
1. **男子**: LightGBM 略优于 CatBoost（0.17963 vs 0.17999），但差异很小，集成后各占 50%
2. **女子**: CatBoost 明显优于 LightGBM（0.14208 vs 0.14551），爬山优化后 LGBM 权重降至 0%
3. **GPU 可用性**: CatBoost GPU 不可用（使用 CPU），LightGBM GPU 可用，但 LightGBM 在女子数据上表现反而不如 CPU 的 CatBoost
4. **过拟合观察**: 男子 CatBoost 第 4 fold (2021-2023) 的 Brier 高达 0.205，明显高于其他 fold，提示疫情后赛季模式变化导致分布偏移

### 为什么不跑 XGBoost

本方案仅使用 CatBoost + LightGBM，未包含 XGBoost。原因是 NCAA 数据以中等规模（2-3K 样本）和大量类别特征为主，CatBoost 的类别特征原生支持和 LightGBM 的 GOSS 加速更适合此场景。后续迭代建议加入 XGBoost 增加集成多样性。

---

## CV 策略评估

### Expanding Window 设计

| Fold | 训练区间 | 验证区间 | 男子 CatBoost Brier | 男子 LightGBM Brier | 女子 CatBoost Brier |
|------|---------|---------|-------------------|-------------------|-------------------|
| 1 | 2003-2008 | 2009-2012 | 0.1834 | 0.1829 | 0.1390* |
| 2 | 2003-2012 | 2013-2016 | 0.1714 | 0.1709 | 0.1441* |
| 3 | 2003-2016 | 2017-2019 | 0.1804 | 0.1782 | 0.1511* |
| 4 | 2003-2019 | 2021-2023 | 0.2052 | 0.2027 | 0.1341* |
| 5 | 2003-2023 | 2024-2025 | 0.1595 | 0.1634 | — |

*女子仅 4 fold（起始 2010 年，数据量更少）

**有效性评估**:
- 合理的时序模拟：每个 fold 都只用过去数据预测未来
- **Fold 4 异常偏高**（0.205 男子）：2021-2023 赛季受 COVID-19 影响（2020 赛事取消、2021 异常赛程），模型难以适应分布偏移。这是 expanding window 的价值所在——暴露了真实世界的时间不稳定性
- **Fold 5 偏低**（0.1595 男子）：2024-2025 样本量小 + 回归正常，CV 分数可能不具代表性
- **女子 Fold 4 异常偏低**（0.1341）：2023-2025 数据较少，CV 方差大

**改进空间**:
- 使用 Purged Group Time Series Split，确保验证集数据不会泄露到训练集
- 对 2020-2021 异常赛季做单独标记/降权
- 增加更细粒度的分组（按赛季 + 赛区）

---

## 男子 vs 女子差异深度分析

| 维度 | 男子 | 女子 | 差异分析 |
|------|------|------|---------|
| Ensemble Brier | 0.1796 | 0.1412 | 女子显著更好 |
| 训练样本 | 2,898 | 1,922 | 男子多 50% |
| 特征数 | 40 | 34 | 男子有 Massey |
| 历史跨度 | 2003-2025 | 2010-2025 | 男子多 7 年数据 |
| 赛季球队数 | ~350 | ~340 | 相当 |
| 最佳模型 | 集成各半 | CatBoost 独占 | 女子 GBDT 多样性不足 |

**为什么女子预测更容易（Brier 更低）？**
1. **女子 NCAA 竞争格局更集中**：少数强校统治力更强（如 UConn、South Carolina），强弱分明，预测确定性更高
2. **单淘汰赛爆冷更少**：男子锦标赛以高频爆冷闻名（如 16 号种子淘汰 1 号种子），女子类似事件显著更少
3. **数据跨度影响**：女子数据从 2010 年开始，篮球风格变化较小；男子从 2003 年开始，经历了多个篮球时代演变

**为什么 LightGBM 在女子上效果差？**
1. 女子样本更少（1,922 vs 2,898），GPU 版本的 LightGBM 可能在小样本上过拟合
2. 女子数据特征差异较小，CatBoost 的对称决策树天然更适合小数据集
3. 爬山优化将 LGBM 权重降至 0%，说明 LGBM OOF 预测与 CatBoost 相关性高（共享特征），但精度更低

---

## 改进方向（5 条以上可操作建议）

### 1. 加入 XGBoost 提升集成多样性
当前仅使用 CatBoost + LightGBM，两者同为 GBDT 变体，预测相关性较高。加入 XGBoost（使用 `hist` tree method）可引入不同的梯度更新逻辑，预期 ensemble 可再提升 0.002-0.005 Brier。

### 2. 引入深度神经网络 / TabNet
表格数据深度模型（TabNet、FT-Transformer）在中等规模数据上可学到不同的特征交互模式。NCAA 数据 2-3K 样本 + 30-40 特征，TabNet 有潜力但需注意小样本过拟合。推荐先用 AutoGluon 快速验证。

### 3. 处理 2020-2021 COVID 异常赛季
建议方案：
- 对 2021 赛季样本降权（设 weight=0.5）
- 在特征中加入 `is_covid_season` 标识
- 在 CV 中将 2021 单独作为验证 fold

### 4. 更激进的特征工程
- **教练效应**：已有 `MTeamCoaches.csv`，加入教练历史胜率、锦标赛经验（执教场次数）等特征
- **赛区强度**：计算每个 Conference 的历史 tournament 胜率作为赛区强度因子
- **SOS（Schedule Strength）**：从 Massey Ordinals 中提取对手强度信息
- **球员更新数据**：NCAA 每年有大量球员更替，加入 roster 稳定性指标
- **比赛型特征**：赛季最后 10 场比赛的趋势（斜率）、对阵前 25 球队的战绩

### 5. 伪标签 + 半监督学习
当前代码已包含伪标签框架但未有效利用。2026 年有 66K+ 待预测比赛，可用高置信度预测（>0.9 或 <0.1）扩充训练集。关键改进：
- 使用 soft pseudo-label（预测概率本身作为标签, weight by confidence）
- 多轮迭代：伪标签后重新训练，再预测，迭代 2-3 轮

### 6. 爬山优化与 stacking
当前 blending 用 SLSQP 优化权重，但未做 stacking。改进：
- 以 CatBoost/LightGBM/XGBoost 的 OOF 预测 + Elo/seed 等原始特征作为 meta-features
- meta-model 使用 Logistic Regression（简洁、可解释、校准好）
- 加入 k-fold stacking 避免过拟合

### 7. 按赛区/种子级别分组建模
当前全量数据统一建模。可尝试：
- 高种子 vs 低种子分组建模（1-4 号 vs 5-16 号）
- 权力联盟 vs 非权力联盟分组建模
- 常规赛胜率 > 0.5 vs < 0.5 分组
不同子集的预测模式可能完全不同，统一模型可能无法捕捉

### 8. 校准优化
Brier Score 由 sharpness（锐度）和 calibration（校准）组成。检查模型预测是否过度自信：
- 使用 Platt Scaling / Isotonic Regression 进行概率校准
- 对 CatBoost 设 `use_best_model=True` + 早停提升校准质量
- 绘制可靠性曲线（reliability diagram），检查 low/high 区间的系统偏差

### 9. 更 robust 的 CV 策略
- 使用 Purged Group Time Series Split + gap
- 对男子使用 6 fold（更细粒度）
- 对每个 fold 统计 Brier + AUC + LogLoss 多维度评估
- 记录每个 fold 的特征重要性漂移，识别不稳定特征

### 10. Kaggle 社区知识融合
赛后应阅读 Top 10% 方案，重点关注：
- 他们如何处理 Massey Ordinals（是否使用更多系统）
- 是否使用 SVD 矩阵分解或因子模型替代 Elo
- 是否有特殊的样本加权策略（按年份加权、按种子加权等）
