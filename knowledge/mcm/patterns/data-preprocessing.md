# 数据预处理方法参考（表格+时序·国赛通用）

> 表格数据（Excel 附件）与时序数据在进入建模前的标准化处理范式：缺失值填补、异常值处理、尺度变换、类别编码、防泄漏。国赛各题型（预测/评价/机理）的通用前置环节，评审可读性收益高。

## 元数据
```
type: prediction
complexity: basic
verified_count: 0
last_verified: 2026-09-05
status: active
```

## 适用条件
| 条件 | 说明 |
|------|------|
| 题型 | 预测/评价/机理均可适配（以 prediction 为主）；凡涉及表格或时序数据的任一赛题 |
| 数据要求 | 表格数据（Excel 附件）与时序数据为主，数据量中小（<10 万行） |
| 前置知识 | 描述统计、假设检验、Python pandas/sklearn 基础 |
| 典型赛题 | 国赛/美赛含附件数据的建模题（面板、横截面、日度/月度序列） |

## 推荐方法栈

### 1. 缺失值处理
| 场景 | 方法/工具 | 说明 |
|------|----------|------|
| 缺失机制判定 | 卢宾缺失机制框架 | 判定 MCAR（完全随机）/ MAR（随机）/ MNAR（非随机），Rubin (1976) |
| 表格多重插补 | MICE / sklearn IterativeImputer | 链式方程多重插补，5 次迭代生成多个完整数据集 → Rubin 规则合并（van Buuren 2011） |
| 时序智能插补 | SAITS / BRITS（PyPOTS） | 自注意力/双向循环时序缺失填补，国赛时序首选 |
| 退路 | 删除、均值/中位数填充 | 仅 MCAR 且缺失率低时勉强可用，需在论文注明局限 |

### 2. 异常值处理
| 场景 | 方法/工具 | 说明 |
|------|----------|------|
| 统计法 | IQR、3σ、Grubbs 检验 | 单变量异常剔除，Grubbs (1969) |
| 机器学习法 | IsolationForest、LOF | 多变量/非线性结构离群检测，Liu (2008)、Chandola (2009) 综述 |

### 3. 缩放与正态化
| 场景 | 方法/工具 | 说明 |
|------|----------|------|
| 连续变量 | Z-score / MinMax / Robust | Robust（中位数+IQR）抗离群 |
| 偏态校正 | Box-Cox / Yeo-Johnson 变换 | Box & Cox (1964)、Yeo & Johnson (2000)，Yeo-Johnson 支持非正数据 |
| 树模型 | 免缩放 | GBDT/随机森林对尺度不敏感，无需标准化 |

### 4. 类别编码
| 场景 | 方法/工具 | 说明 |
|------|----------|------|
| 低基数（≤ 若干类） | one-hot | 直观、可解释 |
| 高基数 | 目标编码 + 平滑（+ 折内防泄漏） | 用目标均值编码，加平滑正则；必须在交叉验证折内估计，Micci-Barreca (2001)、Pargent (2022) |

### 5. 防泄漏（贯穿始终）
| 场景 | 方法/工具 | 说明 |
|------|----------|------|
| 通用 | 先 split 再 fit | 一切缩放/插补/目标编码仅用训练折参数，Kaufman (2012) |
| 时序 | 按时间切分 | 勿随机打乱，未来窗口禁用 |
| 交叉验证 | 折内估计 | 每折独立拟合预处理参数 |

## 实施要点
1. 缺失机制判定先行：用表格/图形判断 MCAR/MAR/MNAR，据此选填补法并在论文中说明假设。
2. 对变量 X 的 n% 缺失采用 MICE 多重插补（5 次迭代），插补方差经 Rubin 规则合并——论文可直写为"对 X 的 12.3% 缺失采用链式方程多重插补（M=5），Little 检验拒绝 MCAR（p<0.05），按 MAR 假设进行多重插补并做敏感性分析 (Little 1988; Rubin 1976)，插补不确定性经 Rubin 规则合并"。
3. 时序缺失优先 SAITS：可写"对 2019–2023 日度序列中 8 处缺失采用 SAITS 自注意力插补，插补值较线性插值 RMSE 降低 xx%"。
4. 异常值用多方法交叉：先 3σ/IQR 初筛，再用 IsolationForest 复核，写明剔除占比及合理性。
5. 一切变换坚持"先切分再拟合"：将训练/测试/时间切分置于预处理之前，避免任何泄漏。
6. 对类别高基数采用目标编码并在折内估计，防止目标泄漏造成的虚高精度。
7. **低基数阈值**：类别数 ≤8 用 one-hot / 有序编码；>8 走目标编码 + 平滑系数。
8. **缩放按模型定**：树模型与量纲无关可不缩放；线性/正则化/kNN/PCA/神经网络必须标准化。
9. **异常值处置留痕**：区分"保留-标记 / 截尾 / 删除"三种处置，**删除前后做模型对比**作为稳健性检验，写明剔除占比。
10. **缺失率 >50%** 的变量建议剔除并在论文说明。
11. **CV 切分按数据结构选**：分类用分层 K 折；时序用时间序列切分（不随机）；有分组用 GroupKFold。
    泛化误差与预测区间一律用交叉验证报告，**不得用训练集 R²**（Stone 1974）。

## 常见陷阱
1. 均值/中位数填补低估方差：单值填充不反映不确定性，显著性检验易失真；应用多重插补。
2. 归一化在全量数据上拟合导致泄漏：标准化参数用了测试集信息，测试分数虚高。
3. "VIF>10 一刀切"武断：VIF 阈值无普适依据，需结合研究目标与共线性后果，勿机械剔除。
4. 目标编码不做折内防泄漏：直接全样本编码把目标信息漏给特征，造成过拟合与虚高 CV。
5. 时序数据随机切分：破坏时间顺序，未来信息泄露进训练，预测指标失真。

## 成功案例
- 待沉淀：本条目为通用范式，成功案例在首次赛后按 `competitions/_template.md` 补充具体赛题链接。

## 参考论文
- van Buuren S., Groothuis-Oudshoorn K. (2011). MICE: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software.
- Rubin D.B. (1976). Inference and Missing Data. Biometrika.
- Little R.J.A. (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. Journal of the American Statistical Association.
- Grubbs F.E. (1969). Procedures for Detecting Outlying Observations in Samples. Technometrics.
- Liu F.T. et al. (2008). Isolation Forest. ICDM.
- Chandola V. et al. (2009). Anomaly Detection: A Survey. ACM Computing Surveys.
- Box G.E.P., Cox D.R. (1964). An Analysis of Transformations. JRSS-B.
- Yeo I.-K., Johnson R.A. (2000). A New Family of Power Transformations. Biometrika.
- Micci-Barreca D. (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes. Data Mining and Knowledge Discovery.
- Pargent F. et al. (2022). Regularized Target Encoding Outperforms Traditional Methods. PLOS ONE.
- Kaufman S. et al. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. ACM TKDD.
